"""
Daily prediction logging.

Runs the same live-inference path app.py uses (trailing-24h marine/weather
aggregates + the latest available satellite zsd as zsd_lag1), but on a
schedule instead of on page load - once a day, one row per site per model,
so predicted values land at the same daily grain as the actual satellite
readings they'll eventually be compared against in the Visibility History
chart. Safe to rerun same-day (upsert on location_id + date + model_name).

Runs every regression model file that exists for a site (see models.py) -
a "short" model whose required marine variables aren't populated yet for
that site simply gets skipped for the day, same as a missing zsd_lag1.

Also runs the in-scope time series models (see ts_models.py). Unlike the
regression models, ts_models.rolling_forecast() naturally recomputes a
full historical series each call (it has to, to rebuild the ARMA state -
see that module's docstring), so this logs that whole series every day,
not just today - an upsert on every date it returns, self-healing any
gap since a prior run and backfilling automatically instead of needing a
separate backfill pass the way the regression models did.

Setup:
    Same DB_DSN / .env as the other pipeline scripts. Needs
    R Models/New Models/ to be present (see models.py/ts_models.py for
    the expected file naming).
"""

import os
from datetime import date

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

import models
import predict
import ts_models

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

# Set by GitHub Actions; falls back to "local" for manual/dev runs, so
# predictions stay attributable to the code version that produced them.
MODEL_VERSION = os.environ.get("GITHUB_SHA", "local")[:12]


def load_locations(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT location_id, name FROM locations")
        return cur.fetchall()


def latest_zsd(conn, location_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT zsd
            FROM ocean_color_daily
            WHERE location_id = %s AND zsd IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """,
            (location_id,),
        )
        row = cur.fetchone()
    return float(row[0]) if row else None


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


def run():
    conn = psycopg2.connect(DB_DSN)
    today = date.today()

    rows = []
    for location_id, name in load_locations(conn):
        zsd_lag1 = latest_zsd(conn, location_id)
        live_features = predict.get_live_features(conn, location_id)
        site_models = models.load_all(name)

        for model_name, (kind, model) in site_models.items():
            predicted = models.predict_visibility(kind, model, live_features, zsd_lag1, today)
            if predicted is None:
                print(f"[predict-daily] Skipping {name} ({model_name}): missing feature or no zsd_lag1.")
            else:
                rows.append((location_id, today, model_name, predicted, zsd_lag1, MODEL_VERSION))
                print(f"[predict-daily] {name} ({model_name}): predicted_zsd={predicted:.2f} (zsd_lag1={zsd_lag1 if zsd_lag1 is None else round(zsd_lag1, 2)})")

        for model_name, ts_model in ts_models.load_all(name).items():
            try:
                series = ts_models.rolling_forecast(conn, location_id, model_name, ts_model, live_features, today)
            except ValueError as e:
                print(f"[predict-daily] Skipping {name} ({model_name}): {e}")
                continue
            for series_date, predicted in series:
                rows.append((location_id, series_date, model_name, predicted, None, MODEL_VERSION))
            print(f"[predict-daily] {name} ({model_name}): logged {len(series)} date(s) through {today}.")

    upsert_predictions(conn, rows)
    conn.close()
    print(f"Logged {len(rows)} prediction(s) for {today}.")


if __name__ == "__main__":
    run()
