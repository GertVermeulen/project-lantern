"""
Open-Meteo Weather (rainfall) pipeline.

Pulls hourly rainfall for the past N hours (default 72) for your dive
locations, using Open-Meteo's standard Forecast API with the `past_days`
parameter. This is deliberately a "recent window" script, not a historical
backfill - it's meant to be run whenever you want the current rainfall
picture leading into a dive, e.g. shortly before a prediction is made.

Uses the same `locations` table as marine_pipeline.py, so location_ids
line up automatically as long as your LOCATIONS lists match.

Setup:
    Same DB_DSN / .env as marine_pipeline.py.
"""

import os
import time
from datetime import date, timedelta

import requests
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv

load_dotenv()

# --- Config -----------------------------------------------------------

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

# The same locations you're using for marine data - keep these in sync
LOCATIONS = [
    {"name": "The Cathedral (Flic-en-Flac)", "lat": -20.283, "lon": 57.360},
    {"name": "Coin de Mire (Djabeda Wreck)", "lat": -19.955, "lon": 57.625},
    {"name": "Blue Bay Marine Park", "lat": -20.443, "lon": 57.713},
]

# --- Database helpers ---------------------------------------------------

def ensure_locations(conn, locations):
    """Insert locations if they don't exist yet. Returns name -> location_id."""
    ids = {}
    with conn.cursor() as cur:
        for loc in locations:
            cur.execute(
                """
                INSERT INTO locations (name, latitude, longitude)
                VALUES (%s, %s, %s)
                ON CONFLICT (latitude, longitude) DO UPDATE SET name = EXCLUDED.name
                RETURNING location_id
                """,
                (loc["name"], loc["lat"], loc["lon"]),
            )
            ids[loc["name"]] = cur.fetchone()[0]
    conn.commit()
    return ids


def log_bronze_response(conn, source, request_params, response):
    """Persist a raw API response as-is, before any parsing."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bronze.api_responses (source, request_params, response)
            VALUES (%s, %s, %s)
            """,
            (source, Json(request_params), Json(response)),
        )
    conn.commit()


def upsert_rainfall(conn, location_id, hourly, model="best_match"):
    """Insert or update rainfall rows for one location."""
    n = len(hourly["time"])
    rows = list(
        zip(
            [location_id] * n,
            hourly["time"],
            hourly.get("precipitation", [None] * n),
            [model] * n,
        )
    )
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO weather_hourly (location_id, ts, precipitation, model)
            VALUES %s
            ON CONFLICT (location_id, ts, model) DO UPDATE SET
                precipitation = EXCLUDED.precipitation
            """,
            rows,
        )
    conn.commit()

# --- API call -------------------------------------------------------------

def fetch_rainfall(conn, locations, past_hours=72):
    """
    Fetch hourly precipitation for the last `past_hours` hours, for one or
    more locations, in a single API call.
    """
    lats = ",".join(str(l["lat"]) for l in locations)
    lons = ",".join(str(l["lon"]) for l in locations)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": "precipitation",
        "past_hours": past_hours,
        "forecast_days": 0,  # we only want the past window, not the forecast
        "timezone": "UTC",
    }
    resp = requests.get(WEATHER_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    log_bronze_response(conn, "open-meteo-forecast", params, data)
    return data if isinstance(data, list) else [data]

# --- Orchestration ----------------------------------------------------------

def collect_recent_rainfall(locations, past_hours=72):
    """
    Pull the last `past_hours` of rainfall for all locations and store it.
    Safe to rerun - overlapping hours are just updated, not duplicated.
    """
    conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    print(f"Fetching last {past_hours}h of rainfall for {len(locations)} location(s)...")
    results = fetch_rainfall(conn, locations, past_hours=past_hours)

    for loc, result in zip(locations, results):
        upsert_rainfall(conn, name_to_id[loc["name"]], result["hourly"])

    conn.close()
    print("Rainfall collection complete.")


if __name__ == "__main__":
    # 72h covers your 48-72h window in one go; trim to 48 if you'd rather
    # match visibility recovery windows more tightly.
    collect_recent_rainfall(LOCATIONS, past_hours=72)
