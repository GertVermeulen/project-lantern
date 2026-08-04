"""
Open-Meteo Historical Weather (rainfall) backfill.

Backfills historical rainfall for your dive locations into Postgres,
matching the same year range as your marine_pipeline.py backfill so the
two datasets can be joined by location_id + timestamp for model training.

Uses the Historical Weather API (ERA5 reanalysis, 1940-present) - this is
the historical counterpart to the live Forecast API used in
weather_pipeline.py, the same way marine_pipeline.py's ERA5-Ocean backfill
is the historical counterpart to a live marine forecast pull.

Setup:
    Same DB_DSN / .env and weather_hourly table as weather_pipeline.py.
"""

import os
import time
from datetime import date

import requests
import psycopg2
from psycopg2.extras import execute_values, Json
from dotenv import load_dotenv

load_dotenv()

# --- Config -----------------------------------------------------------

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

HISTORICAL_WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Keep this in sync with marine_pipeline.py's LOCATIONS
LOCATIONS = [
    {"name": "The Cathedral (Flic-en-Flac)", "lat": -20.283, "lon": 57.360},
    {"name": "Coin de Mire (Djabeda Wreck)", "lat": -19.955, "lon": 57.625},
    {"name": "Blue Bay Marine Park", "lat": -20.443, "lon": 57.713},
]

MODEL_TAG = "era5"  # tag for rows from this historical backfill

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


def upsert_rainfall(conn, location_id, hourly, model=MODEL_TAG):
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

def fetch_chunk(conn, locations, start, end):
    """Fetch one date range for one or more locations in a single API call."""
    lats = ",".join(str(l["lat"]) for l in locations)
    lons = ",".join(str(l["lon"]) for l in locations)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": "precipitation",
        "start_date": start,
        "end_date": end,
        "timezone": "UTC",
    }
    resp = requests.get(HISTORICAL_WEATHER_API_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    log_bronze_response(conn, "open-meteo-archive", params, data)
    return data if isinstance(data, list) else [data]

# --- Orchestration ----------------------------------------------------------

def backfill_rainfall(locations, start_year, end_year):
    conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    for year in range(start_year, end_year + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31" if year < end_year else date.today().isoformat()
        print(f"[rainfall] Fetching {start} to {end} for {len(locations)} location(s)...")

        results = fetch_chunk(conn, locations, start, end)

        for loc, result in zip(locations, results):
            upsert_rainfall(conn, name_to_id[loc["name"]], result["hourly"])

        time.sleep(1)  # be polite to the free-tier API

    conn.close()
    print("Rainfall backfill complete.")


if __name__ == "__main__":
    # Match this range to whatever you used in marine_pipeline.py
    backfill_rainfall(LOCATIONS, start_year=2020, end_year=2026)