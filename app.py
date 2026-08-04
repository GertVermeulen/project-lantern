"""
Underwater Visibility Predictor - interactive showcase.

Shows live current conditions for one monitored dive site at a time, with a
button to cycle through sites. Prediction UI comes later once the R-trained
model is exported.
"""

import base64
import os

import pandas as pd
import psycopg2
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

from live_data import fetch_current_conditions

load_dotenv()

# Classic map-pin marker, inlined as a data URI so the icon renders without an
# external asset request (deck.gl's TextLayer can't render color emoji glyphs,
# hence a real IconLayer here instead of a "📍" text glyph).
_PIN_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="36" viewBox="0 0 24 36">'
    '<path d="M12 0C5.373 0 0 5.373 0 12c0 9 12 24 12 24s12-15 12-24C24 5.373 18.627 0 12 0z" fill="#d62728"/>'
    '<circle cx="12" cy="12" r="5" fill="white"/>'
    "</svg>"
)
PIN_ICON = {
    "url": "data:image/svg+xml;base64," + base64.b64encode(_PIN_SVG.encode()).decode(),
    "width": 24,
    "height": 36,
    "anchorY": 36,
}

DB_DSN = os.environ.get("DB_DSN") or st.secrets.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

st.set_page_config(page_title="Visibility Predictor", page_icon="🤿", layout="wide")

# Compact everything down so a single site's data fits on screen with no scroll.
st.markdown(
    """
    <style>
    div.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    #MainMenu, footer {visibility: hidden;}
    h4 {font-size: 1.05rem !important; margin: 0 0 0.2rem 0 !important;}
    [data-testid="stCaptionContainer"] p {font-size: 0.7rem !important;}
    [data-testid="stMetricValue"] {font-size: 1.15rem !important;}
    [data-testid="stMetricLabel"] p {font-size: 0.65rem !important;}
    [data-testid="stMetric"] {padding: 0.2rem 0 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("##### 🤿 Underwater Visibility Predictor · Mauritius dive sites")


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

if "site_idx" not in st.session_state:
    st.session_state.site_idx = 0

# Cycle site BEFORE reading which row to display, so the click takes effect this run.
nav_prev, nav_label, nav_next = st.columns([1, 3, 1])
if nav_prev.button("◀ Prev", use_container_width=True):
    st.session_state.site_idx = (st.session_state.site_idx - 1) % len(locations)
if nav_next.button("Next ▶", use_container_width=True):
    st.session_state.site_idx = (st.session_state.site_idx + 1) % len(locations)

row = locations.iloc[st.session_state.site_idx]
nav_label.markdown(
    f"<div style='text-align:center; padding-top:0.4rem; font-size:0.85rem;'>"
    f"{row['name']} &nbsp;({st.session_state.site_idx + 1}/{len(locations)})</div>",
    unsafe_allow_html=True,
)

site_point = pd.DataFrame([{"lat": row["latitude"], "lon": row["longitude"], "name": row["name"]}])
site_point["icon_data"] = [PIN_ICON]

icon_layer = pdk.Layer(
    "IconLayer",
    data=site_point,
    get_icon="icon_data",
    get_position=["lon", "lat"],
    get_size=36,
    size_scale=1,
    get_color=[255, 255, 255],  # multiplies the icon's own colors - white = no tint
)
label_layer = pdk.Layer(
    "TextLayer",
    data=site_point,
    get_position=["lon", "lat"],
    get_text="name",
    get_size=15,
    get_color=[20, 20, 20],
    get_text_anchor="'middle'",
    get_alignment_baseline="'top'",
    get_pixel_offset=[0, 6],
)

st.pydeck_chart(
    pdk.Deck(
        layers=[icon_layer, label_layer],
        initial_view_state=pdk.ViewState(latitude=row["latitude"], longitude=row["longitude"], zoom=12),
        map_provider="carto",
        map_style="light",
        tooltip=False,
    ),
    height=440,
)

live = get_live_conditions(row["latitude"], row["longitude"])
oc = ocean_color.loc[row["location_id"]] if row["location_id"] in ocean_color.index else None

oc_caption = f"ocean colour as of {oc['date']}" if oc is not None else "ocean colour: no data yet"
st.caption(f"Wave/current/rain as of {live['time']} UTC · {oc_caption}")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Wave height", f"{live['wave_height']} m")
c2.metric("Current speed", f"{live['ocean_current_velocity']} km/h")
c3.metric("Sea temp", f"{live['sea_surface_temperature']} °C")
c4.metric("Rain now", f"{live['precipitation']} mm")
c5.metric("Secchi depth", f"{oc['zsd']:.1f} m" if oc is not None else "n/a")
c6.metric("KD490", f"{oc['kd490']:.3f} /m" if oc is not None else "n/a")
c7.metric("Chlorophyll", f"{oc['chl']:.2f} mg/m³" if oc is not None else "n/a")

st.caption("Prediction model not wired up yet - forecast comes once the R-trained model is exported.")
