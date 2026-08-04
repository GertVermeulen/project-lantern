"""
Copernicus Marine ocean colour loader.

Loads already-downloaded NetCDF subsets (KD490/ZSD transparency + CHL
plankton, from the OCEANCOLOUR_GLO_BGC_L4_MY_009_104 product) into
Postgres. Unlike marine_pipeline.py/weather_pipeline.py, this doesn't call
an API directly - the files are pulled manually via the Copernicus Marine
Data Store UI or the `copernicusmarine` CLI and dropped in "Bronze Data/",
then this script extracts the nearest pixel to each dive site and loads it.

Data is daily (satellite ocean colour), not hourly like marine_hourly/
weather_hourly - one row per location per day.

Setup:
    Same DB_DSN / .env as marine_pipeline.py.
"""

import os

import numpy as np
import psycopg2
import xarray as xr
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# --- Config -----------------------------------------------------------

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

TRANSPARENCY_FILE = "Collect Historical Data/Bronze Data/copernicus/cmems_obs-oc_glo_bgc-transp_my_l4-gapfree-multi-4km_P1D_1785519505080.nc"
PLANKTON_FILE = "Collect Historical Data/Bronze Data/copernicus/cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D_1785520001924.nc"

# Keep in sync with marine_pipeline.py / weather_pipeline.py
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


def log_file_ingestion(conn, source, file_path, ds):
    """Record a raw file ingestion, pulling product/dataset ID and date range from the NetCDF attrs."""
    start = ds.time.min().dt.date.item()
    end = ds.time.max().dt.date.item()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bronze.file_ingestions
                (source, file_path, product_id, dataset_id, date_range_start, date_range_end)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                source,
                file_path,
                ds.attrs.get("subset:productId"),
                ds.attrs.get("subset:datasetId"),
                start,
                end,
            ),
        )
    conn.commit()


def _to_pyfloats(values):
    """numpy float32 array -> list of python floats/None (psycopg2 can't adapt numpy scalars)."""
    return [None if np.isnan(v) else float(v) for v in values]


def upsert_daily(conn, location_id, dates, kd490, zsd, chl):
    """Insert or update one location's daily ocean colour records."""
    dates = [d.item() if hasattr(d, "item") else d for d in dates]
    kd490 = _to_pyfloats(kd490)
    zsd = _to_pyfloats(zsd)
    chl = _to_pyfloats(chl)
    rows = list(zip([location_id] * len(dates), dates, kd490, zsd, chl))
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

def load(locations, transparency_file=TRANSPARENCY_FILE, plankton_file=PLANKTON_FILE):
    conn = psycopg2.connect(DB_DSN)
    name_to_id = ensure_locations(conn, locations)

    transp = xr.open_dataset(transparency_file)
    plank = xr.open_dataset(plankton_file)

    log_file_ingestion(conn, "copernicus-marine", transparency_file, transp)
    log_file_ingestion(conn, "copernicus-marine", plankton_file, plank)

    for loc in locations:
        t = transp.sel(latitude=loc["lat"], longitude=loc["lon"], method="nearest")
        p = plank.sel(latitude=loc["lat"], longitude=loc["lon"], method="nearest")

        dates = t.time.dt.date.values
        kd490 = t.KD490.values
        zsd = t.ZSD.values
        chl = p.CHL.sel(time=t.time).values

        upsert_daily(conn, name_to_id[loc["name"]], dates, kd490, zsd, chl)
        print(f"[ocean-color] Loaded {len(dates)} days for {loc['name']}.")

    transp.close()
    plank.close()
    conn.close()
    print("Load complete.")


if __name__ == "__main__":
    load(LOCATIONS)
