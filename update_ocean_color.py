"""
Updates ocean_color_daily with the latest available KD490/ZSD/CHL from
Copernicus Marine's NRT product.

Meant to run on a schedule (e.g. daily), NOT called live by the Streamlit
app - the source itself only updates once a day with roughly a 1 day
processing lag, and each subset() download takes 1-2 minutes, which is a
bad fit for "fetch on page load." The app just reads the latest row
already sitting in ocean_color_daily instead.

Pulls a trailing window (default 6 days, not just "today") so any days
that were missing/revised on a previous run get picked up too - same
upsert-on-conflict approach as the rest of the silver layer.

Setup:
    Same DB_DSN / .env as the other pipeline scripts. Requires
    `copernicusmarine login` to have been run already.
"""

import math
import os
import tempfile
from datetime import datetime, timedelta

import copernicusmarine
import psycopg2
import xarray as xr
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

TRANSPARENCY_NRT_DATASET = "cmems_obs-oc_glo_bgc-transp_nrt_l4-gapfree-multi-4km_P1D"
PLANKTON_NRT_DATASET = "cmems_obs-oc_glo_bgc-plankton_nrt_l4-gapfree-multi-4km_P1D"

# Keep in sync with the other pipeline scripts
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


def log_file_ingestion(conn, source, file_path, product_id, dataset_id, start, end):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bronze.file_ingestions
                (source, file_path, product_id, dataset_id, date_range_start, date_range_end)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (source, file_path, product_id, dataset_id, start, end),
        )
    conn.commit()


def _to_pyfloats(values):
    """numpy float32 array -> list of python floats/None (psycopg2 can't adapt numpy scalars)."""
    return [None if (v is None or math.isnan(v)) else float(v) for v in values]


def upsert_daily(conn, location_id, dates, kd490, zsd, chl):
    """Insert or update one location's daily ocean colour records."""
    dates = [d.item() if hasattr(d, "item") else d for d in dates]
    rows = list(zip(
        [location_id] * len(dates), dates,
        _to_pyfloats(kd490), _to_pyfloats(zsd), _to_pyfloats(chl),
    ))
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO ocean_color_daily (location_id, date, kd490, zsd, chl)
            VALUES %s
            ON CONFLICT (location_id, date) DO UPDATE SET
                kd490 = EXCLUDED.kd490,
                zsd = EXCLUDED.zsd,
                chl = EXCLUDED.chl
            """,
            rows,
        )
    conn.commit()

# --- Orchestration ----------------------------------------------------------

def update(locations=LOCATIONS, days_back=6):
    conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    lats = [loc["lat"] for loc in locations]
    lons = [loc["lon"] for loc in locations]
    pad = 0.1
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    with tempfile.TemporaryDirectory() as tmp_dir:
        copernicusmarine.subset(
            dataset_id=TRANSPARENCY_NRT_DATASET,
            variables=["KD490", "ZSD"],
            minimum_longitude=min(lons) - pad, maximum_longitude=max(lons) + pad,
            minimum_latitude=min(lats) - pad, maximum_latitude=max(lats) + pad,
            start_datetime=start.isoformat(), end_datetime=end.isoformat(),
            output_directory=tmp_dir, output_filename="transp.nc",
        )
        copernicusmarine.subset(
            dataset_id=PLANKTON_NRT_DATASET,
            variables=["CHL"],
            minimum_longitude=min(lons) - pad, maximum_longitude=max(lons) + pad,
            minimum_latitude=min(lats) - pad, maximum_latitude=max(lats) + pad,
            start_datetime=start.isoformat(), end_datetime=end.isoformat(),
            output_directory=tmp_dir, output_filename="plank.nc",
        )

        transp = xr.open_dataset(os.path.join(tmp_dir, "transp.nc"))
        plank = xr.open_dataset(os.path.join(tmp_dir, "plank.nc"))

        log_file_ingestion(
            conn, "copernicus-marine-nrt", "ephemeral",
            "OCEANCOLOUR_GLO_BGC_L4_NRT_009_102", TRANSPARENCY_NRT_DATASET,
            start.date(), end.date(),
        )
        log_file_ingestion(
            conn, "copernicus-marine-nrt", "ephemeral",
            "OCEANCOLOUR_GLO_BGC_L4_NRT_009_102", PLANKTON_NRT_DATASET,
            start.date(), end.date(),
        )

        for loc in locations:
            t = transp.sel(latitude=loc["lat"], longitude=loc["lon"], method="nearest")
            p = plank.sel(latitude=loc["lat"], longitude=loc["lon"], method="nearest")
            dates = t.time.dt.date.values
            kd490 = t.KD490.values
            zsd = t.ZSD.values
            chl = p.CHL.sel(time=t.time).values

            upsert_daily(conn, name_to_id[loc["name"]], dates, kd490, zsd, chl)
            print(f"[ocean-color] Updated {len(dates)} day(s) for {loc['name']}.")

        transp.close()
        plank.close()

    conn.close()
    print("Update complete.")


if __name__ == "__main__":
    update()
