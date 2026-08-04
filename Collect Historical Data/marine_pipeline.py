"""
Open-Meteo Marine data pipeline.

Backfills historical marine data (wave height, currents, sea surface
temperature, etc.) for a set of locations into Postgres.

Runs in two passes:
    1. Wave height/direction/period via the ERA5-Ocean model (available
       from 1940 onward).
    2. Ocean current velocity/direction + sea surface temperature, via
       Open-Meteo's best-match model (only available from 2022 onward)
       - this pass UPDATES the rows written in pass 1, filling in those
       columns rather than inserting duplicate rows.

       ERA5-Ocean's grid does not resolve ocean cells close to small
       islands/coastlines, so at these locations it returns null for both
       currents and SST at every date - not just before 2022. Best-match
       resolves a finer grid that actually has water there, but only has
       data from 2022 onward, so 2020-2021 currents/SST are unrecoverable
       from Open-Meteo at these coordinates.

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

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"

HOURLY_VARS = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "ocean_current_velocity",
    "ocean_current_direction",
    "sea_surface_temperature",
]

# ERA5-Ocean's grid doesn't resolve ocean cells at these (coastal/small
# island) locations, so it never returns currents or SST here, at any date.
# Both come from Open-Meteo's best-match model instead, which only goes
# back to 2022. These get fetched in a second pass and used to fill in
# these columns on the existing era5_ocean rows.
SECOND_PASS_VARS = ["ocean_current_velocity", "ocean_current_direction", "sea_surface_temperature"]
SECOND_PASS_AVAILABLE_FROM_YEAR = 2022

# The locations you want data for - keep in sync with weather_pipeline.py /
# weather_backfill.py so location_ids line up.
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
    """Insert or update one location's hourly records for a date range."""
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
            [True] * n,  # era5_ocean_used
        )
    )
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO marine_hourly
                (location_id, ts, wave_height, wave_direction, wave_period,
                 ocean_current_velocity, ocean_current_direction,
                 sea_surface_temperature, era5_ocean_used)
            VALUES %s
            ON CONFLICT (location_id, ts) DO UPDATE SET
                wave_height = EXCLUDED.wave_height,
                wave_direction = EXCLUDED.wave_direction,
                wave_period = EXCLUDED.wave_period,
                ocean_current_velocity = COALESCE(EXCLUDED.ocean_current_velocity, marine_hourly.ocean_current_velocity),
                ocean_current_direction = COALESCE(EXCLUDED.ocean_current_direction, marine_hourly.ocean_current_direction),
                sea_surface_temperature = COALESCE(EXCLUDED.sea_surface_temperature, marine_hourly.sea_surface_temperature),
                era5_ocean_used = EXCLUDED.era5_ocean_used
            """,
            rows,
        )
    conn.commit()

def fill_second_pass_vars(conn, location_id, hourly):
    """
    Fill ocean_current_velocity/direction/sea_surface_temperature on
    EXISTING rows, matched by timestamp, and mark best_match_used. Unlike
    upsert_hourly, this never inserts new rows and never touches
    wave_height/wave_direction/wave_period - it only patches these columns
    for rows that are already there.
    """
    n = len(hourly["time"])
    rows = list(
        zip(
            hourly.get("ocean_current_velocity", [None] * n),
            hourly.get("ocean_current_direction", [None] * n),
            hourly.get("sea_surface_temperature", [None] * n),
            hourly["time"],
            [location_id] * n,
        )
    )
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            UPDATE marine_hourly AS m
            SET ocean_current_velocity = v.velocity,
                ocean_current_direction = v.direction,
                sea_surface_temperature = v.sst::real,
                best_match_used = true
            FROM (VALUES %s) AS v(velocity, direction, sst, ts, location_id)
            WHERE m.location_id = v.location_id::int
              AND m.ts = v.ts::timestamp
            """,
            rows,
        )
    conn.commit()

# --- API call -------------------------------------------------------------

def fetch_chunk(conn, locations, start, end, model="era5_ocean", variables=None, retries=3):
    """Fetch one date range for one or more locations in a single API call."""
    lats = ",".join(str(l["lat"]) for l in locations)
    lons = ",".join(str(l["lon"]) for l in locations)
    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": ",".join(variables or HOURLY_VARS),
        "start_date": start,
        "end_date": end,
        "models": model,
        "timezone": "UTC",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(MARINE_API_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            log_bronze_response(conn, "open-meteo-marine", params, data)
            # Single location -> dict, multiple locations -> list of dicts (same order as input)
            return data if isinstance(data, list) else [data]
        except requests.exceptions.RequestException:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  Request failed (attempt {attempt}/{retries}), retrying in {wait}s...")
            time.sleep(wait)

# --- Orchestration ----------------------------------------------------------

def run_second_pass(locations, start_year, end_year, conn=None):
    """Fill currents + SST (2022 onward) on existing era5_ocean rows."""
    owns_conn = conn is None
    if owns_conn:
        conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    second_pass_start = max(start_year, SECOND_PASS_AVAILABLE_FROM_YEAR)
    if second_pass_start <= end_year:
        for year in range(second_pass_start, end_year + 1):
            start = f"{year}-01-01"
            end = f"{year}-12-31" if year < end_year else date.today().isoformat()
            print(f"[currents/sst] Fetching {start} to {end} for {len(locations)} location(s)...")

            # No `models` param -> Open-Meteo picks its best-match model,
            # which is where current/SST data actually comes from at these
            # coordinates.
            results = fetch_chunk(conn, locations, start, end, model="best_match", variables=SECOND_PASS_VARS)

            for loc, result in zip(locations, results):
                fill_second_pass_vars(conn, name_to_id[loc["name"]], result["hourly"])

            time.sleep(1)
    else:
        print(f"[currents/sst] Skipped - no data before {SECOND_PASS_AVAILABLE_FROM_YEAR}.")

    if owns_conn:
        conn.close()


def backfill(locations, start_year, end_year, model="era5_ocean"):
    conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    # --- Pass 1: wave height/period/direction via ERA5-Ocean ---
    for year in range(start_year, end_year + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31" if year < end_year else date.today().isoformat()
        print(f"[waves] Fetching {start} to {end} for {len(locations)} location(s)...")

        results = fetch_chunk(conn, locations, start, end, model=model)

        for loc, result in zip(locations, results):
            upsert_hourly(conn, name_to_id[loc["name"]], result["hourly"])

        time.sleep(1)  # be polite to the free-tier API

    # --- Pass 2: fill currents + SST where available (2022 onward) ---
    run_second_pass(locations, start_year, end_year, conn=conn)

    conn.close()
    print("Backfill complete.")


if __name__ == "__main__":
    backfill(LOCATIONS, start_year=2020, end_year=2026)