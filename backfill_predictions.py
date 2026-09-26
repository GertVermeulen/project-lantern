"""
One-off backfill of historical predictions, for dates predict_daily.py
never ran against (i.e. before the daily cron existed, or before a given
model was added).

Unlike predict_daily.py - which reconstructs a trailing-24h feature
window because "today"'s finished gold row can't exist yet - a past date
already has a finished row in gold.daily_features (see
Transform/gold_pipeline.py), built from the exact same daily marine/
weather aggregates training used. So this reads that table directly
instead of re-deriving anything, which is actually a closer match to
training than the live path is.

zsd_lag1 for a given row is the previous row's zsd *in this table's
sorted order* for that location, matching `lag(zsd, n = 1)` in the R
training notebook - not necessarily the calendar day before, since
gold.daily_features can have gaps. The first available date for a
location has no predecessor, so it's skipped (same convention as a
missing zsd_lag1 elsewhere).

Usage:
    python backfill_predictions.py --since 2026-08-01
    python backfill_predictions.py --since 2026-08-01 --until 2026-09-01

Setup:
    Same DB_DSN / .env as the other pipeline scripts.
"""

import argparse
import os
from datetime import date

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

import models

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

# Set by GitHub Actions; falls back to "local" for manual/dev runs, so
# predictions stay attributable to the code version that produced them.
MODEL_VERSION = os.environ.get("GITHUB_SHA", "local")[:12] + "-backfill"

GOLD_SQL = """
SELECT
    g.location_id, l.name, g.date,
    g.wave_height_mean, g.wave_height_trend, g.wave_height_min, g.wave_height_max,
    g.wave_period_mean, g.wave_period_trend, g.wave_period_min, g.wave_period_max,
    g.wave_direction_sin_mean, g.wave_direction_cos_mean, g.wave_direction_concentration,
    g.ocean_current_velocity_mean, g.ocean_current_velocity_trend, g.ocean_current_velocity_min, g.ocean_current_velocity_max,
    g.ocean_current_direction_sin_mean, g.ocean_current_direction_cos_mean, g.ocean_current_direction_concentration,
    g.sea_surface_temperature_mean, g.sea_surface_temperature_trend, g.sea_surface_temperature_min, g.sea_surface_temperature_max,
    g.precipitation_sum, g.precipitation_trend, g.precipitation_min, g.precipitation_max,
    g.zsd
FROM gold.daily_features g
JOIN locations l ON l.location_id = g.location_id
WHERE g.date <= %(until)s
ORDER BY g.location_id, g.date
"""

FEATURE_COLUMNS = [
    "wave_height_mean", "wave_height_trend", "wave_height_min", "wave_height_max",
    "wave_period_mean", "wave_period_trend", "wave_period_min", "wave_period_max",
    "wave_direction_sin_mean", "wave_direction_cos_mean", "wave_direction_concentration",
    "ocean_current_velocity_mean", "ocean_current_velocity_trend", "ocean_current_velocity_min", "ocean_current_velocity_max",
    "ocean_current_direction_sin_mean", "ocean_current_direction_cos_mean", "ocean_current_direction_concentration",
    "sea_surface_temperature_mean", "sea_surface_temperature_trend", "sea_surface_temperature_min", "sea_surface_temperature_max",
    "precipitation_sum", "precipitation_trend", "precipitation_min", "precipitation_max",
]


def load_gold_rows(conn, until):
    """Every gold.daily_features row up to `until`, across all history - not
    just from `since` - so the first requested date's zsd_lag1 can still be
    derived from its true predecessor row instead of being skipped."""
    with conn.cursor() as cur:
        cur.execute(GOLD_SQL, {"until": until})
        columns = [desc[0] for desc in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    return rows


def upsert_predictions(conn, rows):
    """rows: list of (location_id, date, model_name, predicted_zsd, zsd_lag1, model_version)."""
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO predictions (location_id, date, model_name, predicted_zsd, zsd_lag1, model_version)
            VALUES %s
            ON CONFLICT (location_id, date, model_name) DO UPDATE SET
                predicted_zsd = EXCLUDED.predicted_zsd,
                zsd_lag1 = EXCLUDED.zsd_lag1,
                model_version = EXCLUDED.model_version,
                predicted_at = now()
            """,
            rows,
        )
    conn.commit()


def run(since, until):
    conn = psycopg2.connect(DB_DSN)
    gold_rows = load_gold_rows(conn, until)

    site_models = {}  # location name -> {model_name: (kind, model)}, loaded once
    rows = []
    skipped_first = set()
    prev_zsd_by_location = {}
    in_range_count = 0

    for gold_row in gold_rows:
        location_id = gold_row["location_id"]
        name = gold_row["name"]
        row_date = gold_row["date"]
        zsd_lag1 = prev_zsd_by_location.get(location_id)
        prev_zsd_by_location[location_id] = gold_row["zsd"]

        if row_date < since:
            continue
        in_range_count += 1

        if zsd_lag1 is None:
            if location_id not in skipped_first:
                print(f"[backfill] {name} {row_date}: first available row, no zsd_lag1 - skipping.")
                skipped_first.add(location_id)
            continue

        live_features = {feature: gold_row[feature] for feature in FEATURE_COLUMNS}

        if name not in site_models:
            site_models[name] = models.load_all(name)

        for model_name, (kind, model) in site_models[name].items():
            predicted = models.predict_visibility(kind, model, live_features, zsd_lag1, row_date)
            if predicted is None:
                continue
            rows.append((location_id, row_date, model_name, predicted, zsd_lag1, MODEL_VERSION))

    upsert_predictions(conn, rows)
    conn.close()
    print(f"Backfilled {len(rows)} prediction(s) across {in_range_count} gold row(s) from {since} to {until}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-08-01", help="Earliest gold.daily_features date to backfill (YYYY-MM-DD).")
    parser.add_argument("--until", default=str(date.today()), help="Latest gold.daily_features date to backfill (YYYY-MM-DD).")
    args = parser.parse_args()
    run(date.fromisoformat(args.since), date.fromisoformat(args.until))
