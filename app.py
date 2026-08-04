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

# Fixed status palette - never themed, so severity reads the same in light/dark.
STATUS_COLORS = {
    "excellent": "#0ca30c",
    "good": "#fab219",
    "fair": "#ec835a",
    "poor": "#d03b3b",
}
TIER_RANK = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
TIER_LABEL = {"poor": "Poor", "fair": "Fair", "good": "Good", "excellent": "Excellent"}


def _hex_to_rgba(hex_color, alpha):
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# Simple, transparent heuristics (NOT the trained model) - each factor is
# tiered independently, then the overall badge takes the worst of the four,
# since diving conditions are only as good as the limiting factor.
def visibility_tier(zsd):
    if zsd >= 20:
        return "excellent"
    if zsd >= 10:
        return "good"
    if zsd >= 5:
        return "fair"
    return "poor"


def wave_tier(h):
    if h <= 1.0:
        return "excellent"
    if h <= 1.5:
        return "good"
    if h <= 2.5:
        return "fair"
    return "poor"


def current_tier(v):
    if v <= 2:
        return "excellent"
    if v <= 5:
        return "good"
    if v <= 8:
        return "fair"
    return "poor"


def rain_tier(r):
    if r <= 0:
        return "excellent"
    if r <= 1:
        return "good"
    if r <= 5:
        return "fair"
    return "poor"


def _card(icon, label, value, color=None):
    """A compact reading tile. color=None renders a neutral (informational, non-tiered) card."""
    accent = color or "var(--secondary-background-color)"
    bg = _hex_to_rgba(color, 0.12) if color else "var(--secondary-background-color)"
    return f"""
    <div style="flex:1; min-width:110px; background:{bg}; border-left:3px solid {accent};
                border-radius:6px; padding:6px 10px;">
      <div style="font-size:0.68rem; color:var(--text-color); opacity:0.7; white-space:nowrap;">{icon} {label}</div>
      <div style="font-size:1.05rem; font-weight:600; color:var(--text-color);">{value}</div>
    </div>
    """


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

tiers = {
    "wave": wave_tier(live["wave_height"]),
    "current": current_tier(live["ocean_current_velocity"]),
    "rain": rain_tier(live["precipitation"]),
}
if oc is not None:
    tiers["visibility"] = visibility_tier(oc["zsd"])

overall_tier = min(tiers.values(), key=lambda t: TIER_RANK[t])
overall_color = STATUS_COLORS[overall_tier]
overall_label = TIER_LABEL[overall_tier]

if oc is not None:
    vis_color = STATUS_COLORS[tiers["visibility"]]
    vis_value = f"{oc['zsd']:.1f} m"
    vis_pct = max(0, min(100, oc["zsd"] / 40 * 100))  # 40m ~ exceptional reef visibility ceiling
else:
    vis_color, vis_value, vis_pct = "#898781", "n/a", 0

cards = [
    _card("🌊", "Wave height", f"{live['wave_height']} m", STATUS_COLORS[tiers["wave"]]),
    _card("🌀", "Current", f"{live['ocean_current_velocity']} km/h", STATUS_COLORS[tiers["current"]]),
    _card("🌧️", "Rain now", f"{live['precipitation']} mm", STATUS_COLORS[tiers["rain"]]),
    _card("🌡️", "Sea temp", f"{live['sea_surface_temperature']} °C"),
]
if oc is not None:
    cards.append(_card("🔬", "Attenuation (KD490)", f"{oc['kd490']:.3f} /m"))
    cards.append(_card("🌿", "Chlorophyll", f"{oc['chl']:.2f} mg/m³"))

st.markdown(
    f"""
    <div style="background:{_hex_to_rgba(overall_color, 0.14)}; border-left:4px solid {overall_color};
                border-radius:8px; padding:8px 14px; margin-bottom:10px;
                display:flex; align-items:center; gap:8px;">
      <span style="font-size:1.3rem;">🤿</span>
      <span style="font-weight:700; font-size:0.95rem; color:var(--text-color);">{overall_label} diving conditions</span>
    </div>
    <div style="display:flex; align-items:baseline; gap:10px; margin-bottom:4px;">
      <span style="font-size:2.6rem; font-weight:700; color:var(--text-color); line-height:1;">{vis_value}</span>
      <span style="font-size:0.75rem; color:var(--text-color); opacity:0.65;">Visibility (Secchi depth)</span>
    </div>
    <div style="width:100%; height:10px; border-radius:6px; background:{_hex_to_rgba(vis_color, 0.18)}; margin-bottom:12px;">
      <div style="width:{vis_pct:.0f}%; height:100%; border-radius:6px; background:{vis_color};"></div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      {"".join(cards)}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Diving-conditions badge is a simple rule-of-thumb (worst of wave/current/rain/visibility), "
    "not the trained model - forecast comes once the R model is exported."
)
