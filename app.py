"""
Underwater Visibility Predictor - interactive showcase.

Shows live current conditions for all monitored dive sites at once.
Prediction UI comes later once the R-trained model is exported.
"""

import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

from live_data import fetch_current_conditions

load_dotenv()

DB_DSN = os.environ.get("DB_DSN") or st.secrets.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

st.set_page_config(page_title="Visibility Predictor", page_icon="🤿")
st.title("🤿 Underwater Visibility Predictor")
st.caption("Dive sites in Mauritius")


@st.cache_data(ttl=3600)
def load_locations():
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute("SELECT location_id, name, latitude, longitude FROM locations")
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(ttl=900)  # matches the ~15 min refresh cadence of Open-Meteo's `current` endpoint
def get_live_conditions(lat, lon):
    conn = psycopg2.connect(DB_DSN)
    data = fetch_current_conditions(conn, lat, lon)
    conn.close()
    return data


@st.cache_data(ttl=3600)
def load_latest_ocean_color():
    """Latest non-null row per location - updated on a schedule by update_ocean_color.py."""
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT ON (location_id) location_id, date, kd490, zsd, chl
            FROM ocean_color_daily
            WHERE zsd IS NOT NULL
            ORDER BY location_id, date DESC
        """)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=columns).set_index("location_id")


locations = load_locations()
ocean_color = load_latest_ocean_color()

st.map(locations.rename(columns={"latitude": "lat", "longitude": "lon"}))

st.caption("Wave/current/rain refresh every ~15 min; ocean colour (KD490/ZSD/CHL) updated daily by a scheduled job")

for _, row in locations.iterrows():
    live = get_live_conditions(row["latitude"], row["longitude"])
    oc = ocean_color.loc[row["location_id"]] if row["location_id"] in ocean_color.index else None

    st.subheader(row["name"])
    oc_caption = f"ocean colour as of {oc['date']}" if oc is not None else "ocean colour: no data yet - run update_ocean_color.py"
    st.caption(f"Wave/current/rain as of {live['time']} UTC - {oc_caption}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wave height", f"{live['wave_height']} m")
    c2.metric("Current speed", f"{live['ocean_current_velocity']} km/h")
    c3.metric("Sea temperature", f"{live['sea_surface_temperature']} °C")
    c4.metric("Rain right now", f"{live['precipitation']} mm")

    c5, c6, c7 = st.columns(3)
    c5.metric("Secchi depth (visibility)", f"{oc['zsd']:.1f} m" if oc is not None else "n/a")
    c6.metric("Attenuation (KD490)", f"{oc['kd490']:.3f} /m" if oc is not None else "n/a")
    c7.metric("Chlorophyll", f"{oc['chl']:.2f} mg/m³" if oc is not None else "n/a")

    st.divider()

st.info("Prediction model not wired up yet - visibility forecast comes once the R-trained model is exported.")
