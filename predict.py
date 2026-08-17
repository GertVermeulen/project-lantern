"""
Live inference for the XGBoost visibility model trained in
`R Models/Lantern Models.Rmd` and exported to `R Models/XGmodel.json`.

The model predicts zsd (Secchi depth, i.e. visibility in meters) from:
  - a one-hot location column per site
  - daily wave/current/sea-temp/rain aggregates (mean, trend, min, max -
    the two direction columns instead get sin_mean/cos_mean/concentration,
    a unit-vector representation of the circular mean), computed in R from
    marine_hourly/weather_hourly via the same formulas as
    Transform/gold_pipeline.py
  - zsd_lag1, the previous day's actual (satellite) Secchi depth

At inference time there's no "yesterday" grouped by calendar day to read
off a finished gold row for - we want a prediction for right now, before
today's gold row can even exist (it needs a satellite zsd that isn't in
yet). So LIVE_FEATURES_SQL recomputes the same aggregates over a trailing
24h window ending now, instead of a calendar day. The trend regressor
switches from gold_pipeline's "hour of day, 0-23" (which wraps at
midnight) to "epoch hours" - both are affine in real time with the same
1-unit-per-hour scale, and a least-squares slope is invariant to an
affine shift of x, so the resulting trend values land on the same scale
the model was trained on.

Caveat worth knowing: the training notebook split/trained/tested only on
Coin de Mire (Djabeda Wreck) rows (`coin_mireShort`), even though the
feature matrix carries a one-hot for all three sites - so the two other
location columns are constant (always 0) across every training example,
and the model never learned to split on them. Applying it to Blue Bay or
The Cathedral works mechanically, but is really "the Coin de Mire
relationship applied to that site's own live readings," not a
site-specific fit.
"""

import numpy as np
import xgboost as xgb

MODEL_PATH = "R Models/XGmodel.json"

FEATURE_NAMES = [
    "location_nameBlue Bay Marine Park",
    "location_nameCoin de Mire (Djabeda Wreck)",
    "location_nameThe Cathedral (Flic-en-Flac)",
    "wave_height_mean",
    "wave_height_trend",
    "wave_height_min",
    "wave_height_max",
    "wave_period_mean",
    "wave_period_trend",
    "wave_period_min",
    "wave_period_max",
    "ocean_current_velocity_mean",
    "ocean_current_velocity_trend",
    "ocean_current_velocity_min",
    "ocean_current_velocity_max",
    "sea_surface_temperature_mean",
    "sea_surface_temperature_trend",
    "sea_surface_temperature_min",
    "sea_surface_temperature_max",
    "precipitation_sum",
    "precipitation_trend",
    "precipitation_min",
    "precipitation_max",
    "wave_direction_sin_mean",
    "wave_direction_cos_mean",
    "wave_direction_concentration",
    "ocean_current_direction_sin_mean",
    "ocean_current_direction_cos_mean",
    "ocean_current_direction_concentration",
    "zsd_lag1",
]

LIVE_FEATURES_SQL = """
SELECT
    m.wave_height_mean, m.wave_height_trend, m.wave_height_min, m.wave_height_max,
    m.wave_period_mean, m.wave_period_trend, m.wave_period_min, m.wave_period_max,
    m.wave_direction_sin_mean, m.wave_direction_cos_mean, m.wave_direction_concentration,
    m.ocean_current_velocity_mean, m.ocean_current_velocity_trend, m.ocean_current_velocity_min, m.ocean_current_velocity_max,
    m.ocean_current_direction_sin_mean, m.ocean_current_direction_cos_mean, m.ocean_current_direction_concentration,
    m.sea_surface_temperature_mean, m.sea_surface_temperature_trend, m.sea_surface_temperature_min, m.sea_surface_temperature_max,
    w.precipitation_sum, w.precipitation_trend, w.precipitation_min, w.precipitation_max
FROM (
    SELECT
        AVG(wave_height) AS wave_height_mean,
        regr_slope(wave_height, EXTRACT(EPOCH FROM ts) / 3600.0) AS wave_height_trend,
        MIN(wave_height) AS wave_height_min,
        MAX(wave_height) AS wave_height_max,

        AVG(wave_period) AS wave_period_mean,
        regr_slope(wave_period, EXTRACT(EPOCH FROM ts) / 3600.0) AS wave_period_trend,
        MIN(wave_period) AS wave_period_min,
        MAX(wave_period) AS wave_period_max,

        AVG(SIN(RADIANS(wave_direction))) AS wave_direction_sin_mean,
        AVG(COS(RADIANS(wave_direction))) AS wave_direction_cos_mean,
        SQRT(POWER(AVG(SIN(RADIANS(wave_direction))), 2) + POWER(AVG(COS(RADIANS(wave_direction))), 2))
            AS wave_direction_concentration,

        AVG(ocean_current_velocity) AS ocean_current_velocity_mean,
        regr_slope(ocean_current_velocity, EXTRACT(EPOCH FROM ts) / 3600.0) AS ocean_current_velocity_trend,
        MIN(ocean_current_velocity) AS ocean_current_velocity_min,
        MAX(ocean_current_velocity) AS ocean_current_velocity_max,

        AVG(SIN(RADIANS(ocean_current_direction))) AS ocean_current_direction_sin_mean,
        AVG(COS(RADIANS(ocean_current_direction))) AS ocean_current_direction_cos_mean,
        SQRT(POWER(AVG(SIN(RADIANS(ocean_current_direction))), 2) + POWER(AVG(COS(RADIANS(ocean_current_direction))), 2))
            AS ocean_current_direction_concentration,

        AVG(sea_surface_temperature) AS sea_surface_temperature_mean,
        regr_slope(sea_surface_temperature, EXTRACT(EPOCH FROM ts) / 3600.0) AS sea_surface_temperature_trend,
        MIN(sea_surface_temperature) AS sea_surface_temperature_min,
        MAX(sea_surface_temperature) AS sea_surface_temperature_max
    FROM marine_hourly
    WHERE location_id = %(location_id)s AND ts >= NOW() - INTERVAL '24 hours'
) m
CROSS JOIN (
    SELECT
        SUM(precipitation) AS precipitation_sum,
        regr_slope(precipitation, EXTRACT(EPOCH FROM ts) / 3600.0) AS precipitation_trend,
        MIN(precipitation) AS precipitation_min,
        MAX(precipitation) AS precipitation_max
    FROM weather_hourly
    WHERE location_id = %(location_id)s AND ts >= NOW() - INTERVAL '24 hours'
) w
"""


def load_model(model_path=MODEL_PATH):
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def get_live_features(conn, location_id):
    """Trailing-24h wave/current/rain aggregates for one location, as a dict."""
    with conn.cursor() as cur:
        # int(...): psycopg2 has no adapter for numpy.int64, which is what
        # callers pull out of a pandas DataFrame column.
        cur.execute(LIVE_FEATURES_SQL, {"location_id": int(location_id)})
        columns = [desc[0] for desc in cur.description]
        row = cur.fetchone()
    return dict(zip(columns, row))


def predict_visibility(booster, live_features, location_name, zsd_lag1):
    """
    Predict today's Secchi depth (m). Returns None if there's no zsd_lag1 to
    anchor on yet (i.e. no satellite ocean-colour reading has ever landed
    for this location) - the model leans on it heavily (see notebook), so a
    prediction without it isn't worth showing.
    """
    if zsd_lag1 is None:
        return None

    values = []
    for name in FEATURE_NAMES:
        if name.startswith("location_name"):
            values.append(1.0 if name == f"location_name{location_name}" else 0.0)
        elif name == "zsd_lag1":
            values.append(float(zsd_lag1))
        else:
            raw = live_features.get(name)
            values.append(float(raw) if raw is not None else float("nan"))

    dmatrix = xgb.DMatrix(np.array([values], dtype=float), feature_names=FEATURE_NAMES)
    prediction = float(booster.predict(dmatrix)[0])
    return max(0.0, prediction)  # zsd is physically non-negative (see notebook)
