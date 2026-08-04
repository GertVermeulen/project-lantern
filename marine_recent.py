"""
Open-Meteo Marine (recent window) pipeline.

Pulls the last N hours of wave/current/SST for your dive locations, using
Open-Meteo Marine's standard forecast endpoint with `past_hours` - this is
the recent-window counterpart to Collect Historical Data/marine_pipeline.py
(which backfills the full historical archive and is not run on a schedule).

Nothing else keeps marine_hourly current going forward, and the XGBoost
model's live features (wave/current/SST mean/trend/min/max over a trailing
24h window) depend on it - this is what keeps that window populated.

No `models` param is passed, so Open-Meteo picks best-match - same
reasoning as live_data.py and the second pass in marine_pipeline.py:
era5_ocean's grid has no data at these coastal coordinates at all.

Setup:
    Same DB_DSN / .env as weather_pipeline.py.
"""

import os
import time

import requests
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv

load_dotenv()

# --- Config -----------------------------------------------------------

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"

HOURLY_VARS = (
    "wave_height,wave_direction,wave_period,"
    "ocean_current_velocity,ocean_current_direction,sea_surface_temperature"
)

# The same locations used across the other pipelines - keep these in sync.
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


def upsert_hourly(conn, location_id, hourly):
    """Insert or update one location's hourly records."""
    n = len(hourly["time"])
    rows = list(
        zip(
            [location_id] * n,
            hourly["time"],
            hourly.get("wave_height", [None] * n),
            hourly.get("wave_direction", [None] * n),
            hourly.get("wave_period", [None] * n),
            hourly.get("ocean_current_velocity", [None] * n),
            hourly.get("ocean_current_direction", [None] * n),
            hourly.get("sea_surface_temperature", [None] * n),
        )
    )
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO marine_hourly
                (location_id, ts, wave_height, wave_direction, wave_period,
                 ocean_current_velocity, ocean_current_direction,
                 sea_surface_temperature, era5_ocean_used, best_match_used)
            VALUES %s
            ON CONFLICT (location_id, ts) DO UPDATE SET
                wave_height = EXCLUDED.wave_height,
                wave_direction = EXCLUDED.wave_direction,
                wave_period = EXCLUDED.wave_period,
                ocean_current_velocity = EXCLUDED.ocean_current_velocity,
                ocean_current_direction = EXCLUDED.ocean_current_direction,
                sea_surface_temperature = EXCLUDED.sea_surface_temperature,
                best_match_used = true
            """,
            [row + (False, True) for row in rows],
        )
    conn.commit()

# --- API call -------------------------------------------------------------

def fetch_recent(conn, locations, past_hours=72):
    """
    Fetch the last `past_hours` of marine data for one or more locations in
    a single API call.
    """
    lats = ",".join(str(l["lat"]) for l in locations)
    lons = ",".join(str(l["lon"]) for l in locations)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": HOURLY_VARS,
        "past_hours": past_hours,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    resp = requests.get(MARINE_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    log_bronze_response(conn, "open-meteo-marine-recent", params, data)
    return data if isinstance(data, list) else [data]

# --- Orchestration ----------------------------------------------------------

def collect_recent_marine(locations, past_hours=72):
    """
    Pull the last `past_hours` of wave/current/SST for all locations and
    store it. Safe to rerun - overlapping hours are just updated, not
    duplicated.
    """
    conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    print(f"Fetching last {past_hours}h of marine data for {len(locations)} location(s)...")
    results = fetch_recent(conn, locations, past_hours=past_hours)

    for loc, result in zip(locations, results):
        upsert_hourly(conn, name_to_id[loc["name"]], result["hourly"])

    conn.close()
    print("Marine collection complete.")


if __name__ == "__main__":
    # 72h keeps a comfortable buffer past the 24h trailing window the live
    # feature aggregation actually needs.
    collect_recent_marine(LOCATIONS, past_hours=72)
