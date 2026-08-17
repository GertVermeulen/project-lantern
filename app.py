"""
Underwater Visibility Predictor - interactive showcase.

Shows live current conditions and a model-predicted visibility for one
monitored dive site at a time, with a button to cycle through sites.
"""

import base64
import math
import os

import pandas as pd
import psycopg2
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv

import predict
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
    accent = color or "var(--card-bg)"
    bg = _hex_to_rgba(color, 0.12) if color else "var(--card-bg)"
    return f"""
    <div style="flex:1; min-width:110px; background:{bg}; border-left:3px solid {accent};
                border-radius:6px; padding:6px 10px;">
      <div style="font-size:0.68rem; color:var(--ink); opacity:0.7; white-space:nowrap;">{icon} {label}</div>
      <div style="font-size:1.05rem; font-weight:600; color:var(--ink);">{value}</div>
    </div>
    """


# Gauge geometry: a semicircle from 180deg (left) to 0deg (right), in a
# 0..GAUGE_MAX-scaled value space. Zone boundaries match visibility_tier()'s
# good/excellent cutoffs (10m/20m) - "fair" folds into "poor" here since the
# gauge only has three bands, same as the reference design.
GAUGE_MAX = 40
GAUGE_ZONE_BOUNDS = [0, 10, 20, GAUGE_MAX]
GAUGE_ZONE_COLORS = [STATUS_COLORS["poor"], STATUS_COLORS["good"], STATUS_COLORS["excellent"]]


def _polar_point(cx, cy, r, angle_deg):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy - r * math.sin(rad)


def _gauge_svg(value, ink_color):
    """A semicircular gauge (Poor/Good/Excellent zones + needle) for a 0..GAUGE_MAX reading."""
    cx, cy, r = 100, 100, 75
    arcs = []
    for i in range(3):
        a1 = 180 - (GAUGE_ZONE_BOUNDS[i] / GAUGE_MAX) * 180
        a2 = 180 - (GAUGE_ZONE_BOUNDS[i + 1] / GAUGE_MAX) * 180
        x1, y1 = _polar_point(cx, cy, r, a1)
        x2, y2 = _polar_point(cx, cy, r, a2)
        arcs.append(
            f'<path d="M {x1:.1f},{y1:.1f} A {r},{r} 0 0 1 {x2:.1f},{y2:.1f}" '
            f'stroke="{GAUGE_ZONE_COLORS[i]}" stroke-width="20" fill="none" stroke-linecap="round" />'
        )

    needle = ""
    if value is not None:
        angle = 180 - (max(0, min(GAUGE_MAX, value)) / GAUGE_MAX) * 180
        tip_x, tip_y = _polar_point(cx, cy, 58, angle)
        needle = (
            f'<line x1="{cx}" y1="{cy}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" '
            f'stroke="{ink_color}" stroke-width="4" stroke-linecap="round" />'
            f'<circle cx="{cx}" cy="{cy}" r="7" fill="{ink_color}" />'
        )

    return f"""
    <svg viewBox="0 0 200 112" style="width:100%; max-width:380px; display:block; margin:0 auto;">
      {"".join(arcs)}
      {needle}
    </svg>
    """


def _gauge_legend():
    items = "".join(
        f'<span style="display:inline-flex; align-items:center; gap:4px; margin:0 10px;">'
        f'<span style="width:9px; height:9px; border-radius:50%; background:{GAUGE_ZONE_COLORS[i]}; display:inline-block;"></span>'
        f'<span style="font-size:0.72rem; color:var(--ink); opacity:0.75;">{label}</span></span>'
        for i, label in enumerate(["Poor", "Good", "Excellent"])
    )
    return f'<div style="text-align:center; margin-top:2px;">{items}</div>'


DB_DSN = os.environ.get("DB_DSN") or st.secrets.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

st.set_page_config(page_title="Visibility Predictor", page_icon="🤿", layout="wide")

# Compact everything down so a single site's data fits on screen with no scroll.
# Streamlit doesn't actually expose --text-color/--secondary-background-color
# as CSS custom properties (verified: absent from its shipped bundle) - --ink
# and --card-bg below are our own, so custom HTML has something real to read.
st.markdown(
    """
    <style>
    :root {
      --ink: #0b0b0b;
      --card-bg: #f0efec;
    }
    @media (prefers-color-scheme: dark) {
      :root { --ink: #ffffff; --card-bg: #2c2c2a; }
    }
    div.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
    #MainMenu, footer, [data-testid="stHeader"] {display: none;}
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


@st.cache_resource
def load_model():
    return predict.load_model()


@st.cache_data(ttl=900)  # same cadence as the marine_recent.py / weather_pipeline.py schedule
def get_live_features(location_id):
    conn = psycopg2.connect(DB_DSN)
    data = predict.get_live_features(conn, location_id)
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


@st.cache_data(ttl=3600)
def load_zsd_history(location_id):
    """Daily satellite zsd (Secchi depth) readings for one location, last 3 months."""
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date, zsd
            FROM ocean_color_daily
            WHERE location_id = %s AND zsd IS NOT NULL
              AND date >= CURRENT_DATE - INTERVAL '3 months'
            ORDER BY date
            """,
            (int(location_id),),
        )
        rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["date", "zsd"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=3600)
def load_prediction_history(location_id):
    """Daily logged predictions for one location, last 3 months - see predict_daily.py."""
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date, predicted_zsd
            FROM predictions
            WHERE location_id = %s
              AND date >= CURRENT_DATE - INTERVAL '3 months'
            ORDER BY date
            """,
            (int(location_id),),
        )
        rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["date", "predicted_zsd"])
    df["date"] = pd.to_datetime(df["date"])
    return df


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


tab_live, tab_history = st.tabs(["Live Conditions", "Visibility History"])

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

tab_live.pydeck_chart(
    pdk.Deck(
        layers=[icon_layer, label_layer],
        initial_view_state=pdk.ViewState(latitude=row["latitude"], longitude=row["longitude"], zoom=12),
        map_provider="carto",
        map_style="light",
        tooltip=False,
    ),
    height=440,
)

try:
    live = get_live_conditions(row["latitude"], row["longitude"])
except Exception:
    # Open-Meteo's free tier rate-limits under heavy traffic - the map,
    # prediction, and last-satellite reading don't depend on this snapshot,
    # so a failure here shouldn't take down the whole page.
    live = None

oc = ocean_color.loc[row["location_id"]] if row["location_id"] in ocean_color.index else None

booster = load_model()
live_features = get_live_features(row["location_id"])
zsd_lag1 = float(oc["zsd"]) if oc is not None else None
predicted_zsd = predict.predict_visibility(booster, live_features, row["name"], zsd_lag1)

oc_caption = f"ocean colour as of {oc['date']}" if oc is not None else "ocean colour: no data yet"
if live is not None:
    tab_live.caption(f"Wave/current/rain as of {live['time']} UTC · {oc_caption}")
else:
    tab_live.caption(f"Wave/current/rain: temporarily unavailable (rate-limited) · {oc_caption}")

tiers = {}
if live is not None:
    tiers["wave"] = wave_tier(live["wave_height"])
    tiers["current"] = current_tier(live["ocean_current_velocity"])
    tiers["rain"] = rain_tier(live["precipitation"])

# The hero figure is the model's prediction for today - not yesterday's
# satellite reading, which becomes a supporting card instead.
if predicted_zsd is not None:
    hero_value = f"{predicted_zsd:.1f} m"
    hero_note = "Model prediction for today"
    hero_zsd = predicted_zsd
elif oc is not None:
    hero_value = f"{oc['zsd']:.1f} m"
    hero_note = f"Last satellite reading ({oc['date']}) - model needs a live feed first"
    hero_zsd = oc["zsd"]
else:
    hero_value, hero_note, hero_zsd = "n/a", "No data yet", None

if hero_zsd is not None:
    tiers["visibility"] = visibility_tier(hero_zsd)

if tiers:
    overall_tier = min(tiers.values(), key=lambda t: TIER_RANK[t])
    overall_color = STATUS_COLORS[overall_tier]
    overall_label = TIER_LABEL[overall_tier]
else:
    overall_color, overall_label = "#898781", "Unknown"

cards = []
if live is not None:
    cards.extend([
        _card("🌊", "Wave height", f"{live['wave_height']} m", STATUS_COLORS[tiers["wave"]]),
        _card("🌀", "Current", f"{live['ocean_current_velocity']} km/h", STATUS_COLORS[tiers["current"]]),
        _card("🌧️", "Rain now", f"{live['precipitation']} mm", STATUS_COLORS[tiers["rain"]]),
        _card("🌡️", "Sea temp", f"{live['sea_surface_temperature']} °C"),
    ])
else:
    cards.append(_card("⚠️", "Live conditions", "Rate-limited - retrying automatically"))
if oc is not None:
    cards.append(_card("🛰️", f"Last satellite ({oc['date']})", f"{oc['zsd']:.1f} m"))
    cards.append(_card("🔬", "Attenuation (KD490)", f"{oc['kd490']:.3f} /m"))
    cards.append(_card("🌿", "Chlorophyll", f"{oc['chl']:.2f} mg/m³"))

tab_live.markdown(
    f"""
    <div style="background:{_hex_to_rgba(overall_color, 0.14)}; border-left:4px solid {overall_color};
                border-radius:8px; padding:8px 14px; margin-bottom:10px;
                display:flex; align-items:center; gap:8px;">
      <span style="font-size:1.3rem;">🤿</span>
      <span style="font-weight:700; font-size:0.95rem; color:var(--ink);">{overall_label} diving conditions</span>
    </div>
    """,
    unsafe_allow_html=True,
)

gauge_col, cards_col = tab_live.columns([2, 3])

gauge_col.markdown(
    f"""
    <div style="text-align:center; font-size:0.68rem; color:var(--ink); opacity:0.6; margin-bottom:2px;">
      {hero_note}
    </div>
    {_gauge_svg(hero_zsd, "var(--ink)")}
    <div style="text-align:center; font-size:2.8rem; font-weight:700; color:var(--ink); line-height:1; margin-top:2px;">
      {hero_value}
    </div>
    <div style="text-align:center; font-size:0.85rem; color:var(--ink); opacity:0.7; margin-bottom:2px;">
      Underwater visibility
    </div>
    {_gauge_legend()}
    """,
    unsafe_allow_html=True,
)

cards_col.markdown(
    f"""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
      {"".join(cards)}
    </div>
    """,
    unsafe_allow_html=True,
)

tab_live.caption(
    "Predicted visibility comes from an XGBoost model trained on Coin de Mire (Djabeda Wreck) "
    "historical data - applied to the other two sites' own live readings, but not fit to them "
    "specifically. Diving-conditions badge is a simple rule-of-thumb (worst of "
    "wave/current/rain/visibility), not the model's own judgment."
)
with tab_history:
    actual = load_zsd_history(row["location_id"])
    predicted = load_prediction_history(row["location_id"])
    st.caption(f"Satellite-derived visibility (Secchi depth) at {row['name']} · last 3 months")
    if actual.empty and predicted.empty:
        st.info("No ocean-colour readings yet for this site.")
    else:
        merged = pd.merge(
            actual.rename(columns={"zsd": "Actual (satellite)"}),
            predicted.rename(columns={"predicted_zsd": "Predicted"}),
            on="date",
            how="outer",
        ).sort_values("date")
        st.line_chart(
            merged.set_index("date")[["Actual (satellite)", "Predicted"]],
            color=["#2a78d6", "#eb6834"],
            height=340,
            use_container_width=True,
        )
        if predicted.empty:
            st.caption("Predictions are logged daily going forward (see predict_daily.py) - none recorded yet for this site.")
        with st.expander("View as table"):
            st.dataframe(
                merged.rename(columns={"date": "Date"}),
                use_container_width=True,
                hide_index=True,
            )
