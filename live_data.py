"""
Live conditions snapshot - used by the Streamlit showcase (app.py), not
part of the historical pipelines.

Open-Meteo's `current` parameter (distinct from `hourly`) returns a single
instantaneous reading, refreshed roughly every 15 minutes, rather than a
time series - this is the actual "live data stream" for the app, as
opposed to the historical/forecast arrays the pipeline scripts use.

No `models` param is passed for the marine call, so Open-Meteo picks
best-match - same reasoning as the second pass in marine_pipeline.py:
era5_ocean's grid has no data at these coastal coordinates at all.

KD490/ZSD/CHL are NOT fetched here. That source only updates once a day
and a fetch takes 1-2 minutes (copernicusmarine subset download, not a
cheap HTTP GET) - a bad fit for "fetch on page load." See
update_ocean_color.py, which runs on a schedule and writes into
ocean_color_daily; the app just reads the latest row from there instead.
"""

import os

import psycopg2
import requests
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

MARINE_CURRENT_VARS = (
    "wave_height,wave_direction,wave_period,"
    "ocean_current_velocity,ocean_current_direction,sea_surface_temperature"
)


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


def fetch_current_conditions(conn, lat, lon):
    """Fetch a live current-conditions snapshot for one location."""
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "current": MARINE_CURRENT_VARS,
        "timezone": "UTC",
    }
    marine_resp = requests.get(MARINE_API_URL, params=marine_params, timeout=30)
    marine_resp.raise_for_status()
    marine_data = marine_resp.json()
    log_bronze_response(conn, "open-meteo-marine-current", marine_params, marine_data)

    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "precipitation",
        "timezone": "UTC",
    }
    weather_resp = requests.get(WEATHER_API_URL, params=weather_params, timeout=30)
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()
    log_bronze_response(conn, "open-meteo-forecast-current", weather_params, weather_data)

    m = marine_data["current"]
    w = weather_data["current"]
    return {
        "time": m["time"],
        "wave_height": m["wave_height"],
        "wave_direction": m["wave_direction"],
        "wave_period": m["wave_period"],
        "ocean_current_velocity": m["ocean_current_velocity"],
        "ocean_current_direction": m["ocean_current_direction"],
        "sea_surface_temperature": m["sea_surface_temperature"],
        "precipitation": w["precipitation"],
    }
