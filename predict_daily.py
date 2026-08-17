"""
Daily prediction logging.

Runs the same live-inference path app.py uses (trailing-24h marine/weather
aggregates + the latest available satellite zsd as zsd_lag1), but on a
schedule instead of on page load - once a day, one row per site per model,
so predicted values land at the same daily grain as the actual satellite
readings they'll eventually be compared against in the Visibility History
chart. Safe to rerun same-day (upsert on location_id + date + model_name).

Runs every model that has a file to load: the XGBoost model always (it's
required), and the reduced linear-log model (predict_linear.py) only if
R Models/linear_log_model.json is present - lets the linear model be added
later without breaking existing runs.

Setup:
    Same DB_DSN / .env as the other pipeline scripts. Needs
    R Models/XGmodel.json to be present (same model app.py loads).
"""

import os
from datetime import date

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

import predict
import predict_linear

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
    booster = predict.load_model()

    linear_model = None
    if os.path.exists(predict_linear.MODEL_PATH):
        linear_model = predict_linear.load_model()
    else:
        print(f"[predict-daily] {predict_linear.MODEL_PATH} not found - skipping linear model.")

    today = date.today()

    rows = []
    for location_id, name in load_locations(conn):
        zsd_lag1 = latest_zsd(conn, location_id)
        live_features = predict.get_live_features(conn, location_id)

        xgb_predicted = predict.predict_visibility(booster, live_features, name, zsd_lag1)
        if xgb_predicted is None:
            print(f"[predict-daily] Skipping {name} (xgboost): no zsd_lag1 available yet.")
        else:
            rows.append((location_id, today, "xgboost", xgb_predicted, zsd_lag1, MODEL_VERSION))
            print(f"[predict-daily] {name} (xgboost): predicted_zsd={xgb_predicted:.2f} (zsd_lag1={zsd_lag1:.2f})")

        if linear_model is not None:
            linear_predicted = predict_linear.predict_visibility(linear_model, live_features, zsd_lag1)
            if linear_predicted is None:
                print(f"[predict-daily] Skipping {name} (linear_log): missing feature or no zsd_lag1.")
            else:
                rows.append((location_id, today, "linear_log", linear_predicted, zsd_lag1, MODEL_VERSION))
                print(f"[predict-daily] {name} (linear_log): predicted_zsd={linear_predicted:.2f} (zsd_lag1={zsd_lag1:.2f})")

    upsert_predictions(conn, rows)
    conn.close()
    print(f"Logged {len(rows)} prediction(s) for {today}.")


if __name__ == "__main__":
    run()
