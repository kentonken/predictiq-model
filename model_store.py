"""
model_store.py — Supabase Storage for model persistence
Upload after training, download on every Railway container start.
Bucket: ml-models (create this in Supabase Storage dashboard)
"""

import os
import io
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET               = "ml-models"
MODEL_OBJECT         = "ensemble_v5.pkl"
LOCAL_PATH           = Path(os.getenv("MODEL_PATH", "models/ensemble_v5.pkl"))


def _client():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def upload_model(model_obj) -> bool:
    """Serialize model and upload to Supabase Storage."""
    try:
        buf = io.BytesIO()
        joblib.dump(model_obj, buf)
        buf.seek(0)
        data = buf.read()

        sb = _client()
        # Upsert — overwrites existing file
        sb.storage.from_(BUCKET).upload(
            path=MODEL_OBJECT,
            file=data,
            file_options={"content-type": "application/octet-stream",
                          "upsert": "true"},
        )
        logger.info(f"✅ Model uploaded to Supabase Storage → {BUCKET}/{MODEL_OBJECT}")
        return True
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def download_model() -> bool:
    """Download model from Supabase Storage to LOCAL_PATH."""
    try:
        LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        sb   = _client()
        data = sb.storage.from_(BUCKET).download(MODEL_OBJECT)
        LOCAL_PATH.write_bytes(data)
        logger.info(f"✅ Model downloaded from Supabase → {LOCAL_PATH}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Download failed (model may not exist yet): {e}")
        return False


def model_exists_in_storage() -> bool:
    """Check if model file exists in Supabase Storage."""
    try:
        sb    = _client()
        files = sb.storage.from_(BUCKET).list()
        return any(f["name"] == MODEL_OBJECT for f in files)
    except Exception as e:
        logger.warning(f"Storage check failed: {e}")
        return False
