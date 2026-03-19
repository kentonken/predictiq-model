create table if not exists ai_predictions (
  id uuid default gen_random_uuid() primary key,
  fixture_id text unique not null,
  home_team text,
  away_team text,
  prediction text,
  confidence float,
  home_win_prob float,
  draw_prob float,
  away_win_prob float,
  tip text,
  created_at timestamp default now(),
  updated_at timestamp default now()
);

create index if not exists idx_ai_predictions_fixture_id
  on ai_predictions(fixture_id);

alter table ai_predictions enable row level security;

create policy "Public read predictions"
  on ai_predictions for select
  using (true);

create policy "Service role write predictions"
  on ai_predictions for all
  using (auth.role() = 'service_role');
