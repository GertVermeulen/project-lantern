"""
Live inference for the ARMA(p,q)-with-seasonal-Fourier-and-marine-regressors
time series models exported to R Models/New Models/TSmodel_*.json (fit in R
with forecast::Arima, on either raw zsd or its Box-Cox transform).

Only the *d=0* (non-differenced) model files are wired up here:
    blue_bay:  arimax, arimax_bc, basic
    cathedral: arimax
    coin_mire: arimax
cathedral_arimax_bc and coin_mire_arimax_bc (both d=1) aren't - a
differenced series needs a different recursion this module doesn't
implement.

Why this can't just replay the JSON's trailing few rows once and stop
(unlike models.py's regression models, which only ever need one day's
features): the ARMA part carries state - its own recent history of
regression residuals (eta) and innovations (e) - and the JSON only exports
a handful of trailing rows of it. Recomputing eta/e from just those
trailing rows (the AR(3)/MA(2) recursion below, applied to nothing but
the 7-row tail) reproduces the model's own last known residual to only
~3%, not exactly - not accurate enough to forecast from directly. So
instead, rolling_forecast() replays
the model forward day by day from just after training ended through
yesterday, using real historical gold.daily_features/ocean_color_daily
data to recompute the *actual* eta and e for every real day along the
way. That walk is long enough (weeks/months, by now) that the small
error from the JSON tail's imperfect bootstrap decays to noise (these
processes are stationary/invertible, so the initial-condition error
shrinks geometrically each step) - what's left when the walk reaches
yesterday is close to what R's own forecast() would give if it kept
being refit/updated on the same live data.

Calendar alignment: the JSON stores only an abstract row index (n_train,
and xreg_tail's keys) - not the calendar date it corresponds to, and
that start date isn't consistent across files (n_train is 1673 for the
in-scope models, but 1461/3862 for the two d=1 ones this module skips).
_calibrate_anchor recovers it by finding the date in ocean_color_daily
whose zsd (or its Box-Cox transform) matches the model's last y_tail
value almost exactly, cross-checked against the second-to-last value
too so a coincidental near-match elsewhere doesn't get picked.
"""

import json
import math
import os
import re

from models import LOCATION_KEYS

MODELS_DIR = "R Models/New Models"

IN_SCOPE_KINDS = {
    "blue_bay": ["arimax", "arimax_bc", "basic"],
    "cathedral": ["arimax"],
    "coin_mire": ["arimax"],
}

MODEL_LABELS = {
    "ts_arimax": "ARIMAX (time series)",
    "ts_arimax_bc": "ARIMAX, Box-Cox (time series)",
    "ts_basic": "ARIMA, no marine predictors (time series)",
}

# Every marine predictor any in-scope model's predictor_names references -
# fetched in bulk from gold.daily_features/live features; unused columns
# for a given model are simply ignored.
FEATURE_COLUMNS = [
    "wave_height_mean", "wave_height_trend", "wave_height_min", "wave_height_max",
    "wave_period_mean", "wave_period_trend", "wave_period_min", "wave_period_max",
    "wave_direction_sin_mean", "wave_direction_cos_mean", "wave_direction_concentration",
    "ocean_current_velocity_mean", "ocean_current_velocity_trend", "ocean_current_velocity_min", "ocean_current_velocity_max",
    "ocean_current_direction_sin_mean", "ocean_current_direction_cos_mean", "ocean_current_direction_concentration",
    "sea_surface_temperature_mean", "sea_surface_temperature_trend", "sea_surface_temperature_min", "sea_surface_temperature_max",
    "precipitation_sum", "precipitation_trend", "precipitation_min", "precipitation_max",
]

GOLD_SQL = """
SELECT date, {columns}, zsd
FROM gold.daily_features
WHERE location_id = %(location_id)s AND date > %(anchor)s AND date <= %(through)s
ORDER BY date
""".format(columns=", ".join(FEATURE_COLUMNS))

_FOURIER_NAME_RE = re.compile(r"^([SC])(\d+)-365$")


def _model_path(location_key, kind):
    return os.path.join(MODELS_DIR, f"TSmodel_{location_key}_{kind}.json")


def load_all(location_name):
    """{ts_model_name: model_dict} for every in-scope TS variant that has a file for this site."""
    location_key = LOCATION_KEYS[location_name]
    models_out = {}
    for kind in IN_SCOPE_KINDS.get(location_key, []):
        path = _model_path(location_key, kind)
        if os.path.exists(path):
            with open(path) as f:
                models_out[f"ts_{kind}"] = json.load(f)
    return models_out


def _lambda_of(model):
    lam = model.get("lambda")
    return None if lam is None or lam == {} else lam


def _boxcox(x, lam):
    if lam is None:
        return x
    if abs(lam) < 1e-8:
        return math.log(x)
    return (x ** lam - 1) / lam


def _inv_boxcox(y, lam):
    if lam is None:
        return y
    if abs(lam) < 1e-8:
        return math.exp(y)
    value = max(1e-9, lam * y + 1)
    return value ** (1 / lam)


def _fourier_features(fourier_names, index):
    feats = {}
    for name in fourier_names:
        match = _FOURIER_NAME_RE.match(name)
        letter, harmonic = match.group(1), int(match.group(2))
        angle = 2 * math.pi * harmonic * index / 365.25
        feats[name] = math.sin(angle) if letter == "S" else math.cos(angle)
    return feats


def _regression_mean(model, index, feature_values):
    total = model["intercept"] + model["drift"] * index
    for name, coef in (model.get("xreg_coefs") or {}).items():
        total += coef * feature_values[name]
    return total


def _resid_tail_list(model):
    resid_tail = model["resid_tail"]
    if isinstance(resid_tail, list):
        return resid_tail
    # R/jsonlite serializes a length-1 numeric vector as a bare scalar.
    return [resid_tail] if model["order"]["q"] > 0 else []


def _bootstrap_state(model):
    """Seed eta/e history from the model's exported trailing rows."""
    y_tail = model["y_tail"]
    xreg_tail = model.get("xreg_tail") or {}
    if not isinstance(xreg_tail, dict):
        xreg_tail = {}  # empty list, e.g. "basic" (no predictors/fourier at all)
    n_train = model["n_train"]
    start_index = n_train - len(y_tail) + 1

    eta_history = {}
    for offset, idx in enumerate(range(start_index, n_train + 1)):
        feature_values = dict(xreg_tail.get(str(idx), {}))
        eta_history[idx] = y_tail[offset] - _regression_mean(model, idx, feature_values)

    resid_tail = _resid_tail_list(model)
    resid_start = n_train - len(resid_tail) + 1
    e_history = dict(zip(range(resid_start, n_train + 1), resid_tail))

    return n_train, eta_history, e_history


def _step(model, index, eta_history, e_history, feature_values, actual_y_transformed):
    """
    Advance the recursion by one index. feature_values: real predictor
    values for this index (fourier terms are added here). actual_y_transformed:
    that day's actual value on the model's own scale (Box-Cox-applied,
    if applicable), or None to forecast only, without updating state (used
    for "today", whose actual zsd doesn't exist yet).

    Returns (forecast_transformed, new_eta_history, new_e_history).
    """
    p, q = model["order"]["p"], model["order"]["q"]
    ar, ma = model["ar"], model["ma"]

    feats = dict(feature_values)
    feats.update(_fourier_features(model["fourier_names"], index))

    required = model.get("xreg_coefs") or {}
    if any(feats.get(name) is None for name in required):
        return None, eta_history, e_history

    mean_t = _regression_mean(model, index, feats)

    ar_part = sum(ar[f"ar{i}"] * eta_history.get(index - i, 0.0) for i in range(1, p + 1))
    ma_part = sum(ma[f"ma{j}"] * e_history.get(index - j, 0.0) for j in range(1, q + 1))
    eta_hat = ar_part + ma_part
    forecast_t = mean_t + eta_hat

    if actual_y_transformed is None:
        return forecast_t, eta_history, e_history

    eta_actual = actual_y_transformed - mean_t
    e_actual = eta_actual - eta_hat
    new_eta = {**eta_history, index: eta_actual}
    new_e = {**e_history, index: e_actual}
    return forecast_t, new_eta, new_e


def _calibrate_anchor(conn, location_id, model):
    """
    The calendar date corresponding to the model's n_train index - see
    module docstring. Raises ValueError if no confident match is found.
    """
    lam = _lambda_of(model)
    y_tail = model["y_tail"]
    target_last = _inv_boxcox(y_tail[-1], lam)
    target_prev = _inv_boxcox(y_tail[-2], lam)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT date FROM ocean_color_daily
            WHERE location_id = %s AND zsd IS NOT NULL AND ABS(zsd - %s) < 0.0005
            ORDER BY date
            """,
            (location_id, target_last),
        )
        candidates = [row[0] for row in cur.fetchall()]

    with conn.cursor() as cur:
        for candidate_date in candidates:
            cur.execute(
                """
                SELECT zsd FROM ocean_color_daily
                WHERE location_id = %s AND date < %s AND zsd IS NOT NULL
                ORDER BY date DESC LIMIT 1
                """,
                (location_id, candidate_date),
            )
            row = cur.fetchone()
            if row is not None and abs(float(row[0]) - target_prev) < 0.0005:
                return candidate_date

    raise ValueError(
        f"Could not calibrate {model.get('model_type')} model for location {location_id}: "
        f"no unambiguous zsd match for its last two training values."
    )


def rolling_forecast(conn, location_id, model_name, model, live_features, through_date):
    """
    [(date, predicted_zsd), ...] for every date from the model's calibrated
    anchor date through `through_date`, using real gold.daily_features data
    for every date that has a finished row, and `live_features` (the same
    trailing-24h dict models.py's live path uses) for `through_date` alone
    if it doesn't. Raises ValueError if calibration fails (see
    _calibrate_anchor) - callers should treat that as "skip this model",
    same as a None prediction elsewhere.
    """
    lam = _lambda_of(model)
    anchor_date = _calibrate_anchor(conn, location_id, model)
    index, eta_history, e_history = _bootstrap_state(model)

    with conn.cursor() as cur:
        cur.execute(GOLD_SQL, {"location_id": location_id, "anchor": anchor_date, "through": through_date})
        columns = [desc[0] for desc in cur.description]
        gold_rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    results = []

    for gold_row in gold_rows:
        index += 1
        feature_values = {name: gold_row[name] for name in FEATURE_COLUMNS}
        actual = None if gold_row["zsd"] is None else _boxcox(float(gold_row["zsd"]), lam)
        forecast_t, eta_history, e_history = _step(model, index, eta_history, e_history, feature_values, actual)
        if forecast_t is not None:
            results.append((gold_row["date"], max(0.0, _inv_boxcox(forecast_t, lam))))

    # gold.daily_features may have had a row for through_date but still be
    # missing a predictor it needs (still-syncing marine data) - fall back
    # to live features whenever the walk above didn't land on through_date,
    # not just when there was no gold row for it at all.
    if not results or results[-1][0] < through_date:
        # `index` already reflects n_train + len(gold_rows) from the loop
        # above (it advances once per gold row regardless of whether that
        # row's forecast succeeded), which is the right position for
        # through_date as long as gold.daily_features has no missing
        # calendar dates in this range - true in practice (see module docstring).
        feature_values = {name: live_features.get(name) for name in FEATURE_COLUMNS}
        forecast_t, _, _ = _step(model, index, eta_history, e_history, feature_values, None)
        if forecast_t is not None:
            results.append((through_date, max(0.0, _inv_boxcox(forecast_t, lam))))

    return results
