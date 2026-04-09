"""
train.py — PredictIQ Pro Ensemble v5.0
Replaces single CatBoost with XGBoost + LightGBM + CatBoost → meta-learner
Tiered calibration per league | Optuna tuning
Run: python train.py --data data/training_matches.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import optuna

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

from features import engineer_features

warnings.filterwarnings("ignore")
Path("models").mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# BASE MODELS
# ─────────────────────────────────────────────

def make_catboost(params: dict = None):
    p = params or {}
    return CatBoostClassifier(
        iterations     = p.get("cb_iterations", 600),
        depth          = p.get("cb_depth", 6),
        learning_rate  = p.get("cb_lr", 0.05),
        l2_leaf_reg    = p.get("cb_l2", 3.0),
        border_count   = 128,
        loss_function  = "MultiClass",
        eval_metric    = "MultiClass",
        random_seed    = 42,
        verbose        = 0,
    )

def make_xgboost(params: dict = None):
    p = params or {}
    return xgb.XGBClassifier(
        n_estimators      = p.get("xgb_n", 400),
        max_depth         = p.get("xgb_depth", 5),
        learning_rate     = p.get("xgb_lr", 0.05),
        subsample         = p.get("xgb_sub", 0.8),
        colsample_bytree  = p.get("xgb_col", 0.8),
        min_child_weight  = p.get("xgb_mcw", 3),
        objective         = "multi:softprob",
        num_class         = 3,
        use_label_encoder = False,
        eval_metric       = "mlogloss",
        random_state      = 42,
        verbosity         = 0,
    )

def make_lightgbm(params: dict = None):
    p = params or {}
    return lgb.LGBMClassifier(
        n_estimators     = p.get("lgb_n", 400),
        max_depth        = p.get("lgb_depth", 6),
        learning_rate    = p.get("lgb_lr", 0.05),
        num_leaves       = p.get("lgb_leaves", 50),
        subsample        = p.get("lgb_sub", 0.8),
        colsample_bytree = p.get("lgb_col", 0.8),
        min_child_samples= p.get("lgb_mcs", 20),
        objective        = "multiclass",
        num_class        = 3,
        metric           = "multi_logloss",
        random_state     = 42,
        verbosity        = -1,
    )


# ─────────────────────────────────────────────
# ENSEMBLE STACKER
# ─────────────────────────────────────────────

class EnsembleStacker:
    def __init__(self, n_folds: int = 5, params: dict = None):
        self.n_folds = n_folds
        self.params  = params or {}
        self.base_models = {
            "catboost": make_catboost(params),
            "xgboost":  make_xgboost(params),
            "lightgbm": make_lightgbm(params),
        }
        self.meta = LogisticRegression(C=1.0, max_iter=1000,
                                       multi_class="multinomial", solver="lbfgs")
        self.calibrators: dict = {}
        self.le = LabelEncoder()
        self.feature_names: list = []
        self.is_fitted = False

    def _oof(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        oof = np.zeros((len(X), 3 * len(self.base_models)))
        for fold, (tr, val) in enumerate(skf.split(X, y)):
            col = 0
            for m in self.base_models.values():
                m.fit(X[tr], y[tr])
                oof[val, col:col+3] = m.predict_proba(X[val])
                col += 3
            print(f"  fold {fold+1}/{self.n_folds} ✓")
        return oof

    def fit(self, X: pd.DataFrame, y: pd.Series, league_ids: pd.Series = None):
        self.feature_names = list(X.columns)
        Xa = X.values.astype(float)
        ye = self.le.fit_transform(y)

        print("── OOF base predictions ──")
        oof = self._oof(Xa, ye)

        print("── Meta-learner ──")
        self.meta.fit(oof, ye)

        print("── Full refit ──")
        for m in self.base_models.values():
            m.fit(Xa, ye)

        if league_ids is not None:
            print("── Per-league calibration ──")
            for lid in league_ids.unique():
                mask = (league_ids == lid).values
                if mask.sum() < 50:
                    continue
                cal = CalibratedClassifierCV(self.meta, method="sigmoid", cv="prefit")
                mp = self._raw_meta(Xa[mask])
                cal.fit(mp, ye[mask])
                self.calibrators[int(lid)] = cal
                print(f"   league {lid} ({mask.sum()} samples)")

        self.is_fitted = True
        print(f"✅ Ensemble trained on {len(self.feature_names)} features")
        return self

    def _raw_meta(self, Xa: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(
            np.hstack([m.predict_proba(Xa) for m in self.base_models.values()])
        )

    def predict_proba(self, X: pd.DataFrame, league_id: int = None) -> np.ndarray:
        Xa = X[self.feature_names].values.astype(float)
        p  = self._raw_meta(Xa)
        if league_id and int(league_id) in self.calibrators:
            p = self.calibrators[int(league_id)].predict_proba(p)
        return p

    def predict(self, X: pd.DataFrame, league_id: int = None) -> np.ndarray:
        return self.le.inverse_transform(
            np.argmax(self.predict_proba(X, league_id), axis=1)
        )

    def save(self, path: str = "models/ensemble_v5.pkl"):
        joblib.dump(self, path)
        print(f"💾 Saved → {path}")

    @staticmethod
    def load(path: str = "models/ensemble_v5.pkl") -> "EnsembleStacker":
        return joblib.load(path)


# ─────────────────────────────────────────────
# OPTUNA
# ─────────────────────────────────────────────

def run_optuna(X: pd.DataFrame, y: pd.Series, n_trials: int = 30):
    Xa = X.values.astype(float)
    le = LabelEncoder()
    ye = le.fit_transform(y)
    cv = StratifiedKFold(3, shuffle=True, random_state=42)

    def objective(trial):
        p = {
            "cb_iterations": trial.suggest_int("cb_iterations", 300, 800),
            "cb_depth":      trial.suggest_int("cb_depth", 4, 8),
            "cb_lr":         trial.suggest_float("cb_lr", 0.01, 0.15, log=True),
            "xgb_n":         trial.suggest_int("xgb_n", 200, 600),
            "xgb_depth":     trial.suggest_int("xgb_depth", 3, 7),
            "xgb_lr":        trial.suggest_float("xgb_lr", 0.01, 0.15, log=True),
            "lgb_n":         trial.suggest_int("lgb_n", 200, 600),
            "lgb_depth":     trial.suggest_int("lgb_depth", 3, 8),
            "lgb_lr":        trial.suggest_float("lgb_lr", 0.01, 0.15, log=True),
        }
        scores = []
        for m in [make_catboost(p), make_xgboost(p), make_lightgbm(p)]:
            s = cross_val_score(m, Xa, ye, cv=cv, scoring="neg_log_loss")
            scores.append(s.mean())
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best = study.best_params
    joblib.dump(best, "models/best_params.pkl")
    print(f"Best params: {best}")
    return best


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/training_matches.csv")
    parser.add_argument("--optuna", action="store_true")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    print(f"📂 Loading {args.data}")
    df = pd.read_csv(args.data)
    X  = engineer_features(df)
    y  = df["result"]                    # H / D / A
    leagues = df.get("league_id", pd.Series([0] * len(df)))

    params = None
    if args.optuna:
        print("🔍 Optuna search ...")
        params = run_optuna(X, y, n_trials=args.trials)

    model = EnsembleStacker(n_folds=5, params=params)
    model.fit(X, y, league_ids=leagues)
    model.save("models/ensemble_v5.pkl")
    print("🏆 Done")
    
