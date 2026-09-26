"""
Generic loader/predictor for the per-site model family trained in
`R Models/Lantern Models.Rmd` and exported to `R Models/New Models/`.

Two feature "horizons" per site, crossed with three regression types:
  - full:  trained on the full history, but excludes ocean_current_velocity
           and sea_surface_temperature - era5_ocean (their source) always
           trails "now" by about a week, so including them would force
           training to drop most of the available history.
  - short: trained on the shorter recent window where those two variables
           are actually populated, so it can use them.
  - linear:      lm(zsd ~ ...) - direct linear regression on zsd.
  - linear_log:  lm(log(zsd) ~ ...) - log-target, back-transformed with
                 Duan's smearing estimator (see the "smearing_factor" key).
  - xgboost:     XGBoost regressor.

Unlike the old single shared model (predict.py's retired FEATURE_NAMES
list), each of these is trained per-site - no location one-hot columns,
and each file carries its own feature list (explicit "coefficients" keys
for the linear models, "feature_names" in the XGBoost JSON), so this
module never hardcodes a feature list itself.

File naming: R Models/New Models/{horizon}_{location_key}_{suffix}.json
"""

import json
import math
import os

import numpy as np
import xgboost as xgb

MODELS_DIR = "R Models/New Models"

LOCATION_KEYS = {
    "Blue Bay Marine Park": "blue_bay",
    "The Cathedral (Flic-en-Flac)": "cathedral",
    "Coin de Mire (Djabeda Wreck)": "coin_mire",
}

FILENAME_SUFFIX = {
    "linear": "linear_model",
    "linear_log": "linear_log_model",
    "xgboost": "XGmodel",
}

# Order here fixes the History tab's model-picker order and the Live tab's
# hero/secondary preference (see app.py).
MODEL_VARIANTS = [
    ("full", "xgboost"),
    ("full", "linear_log"),
    ("full", "linear"),
    ("short", "xgboost"),
    ("short", "linear_log"),
    ("short", "linear"),
]

MODEL_LABELS = {
    "full_xgboost": "XGBoost (full history)",
    "full_linear_log": "Linear, log-target (full history)",
    "full_linear": "Linear (full history)",
    "short_xgboost": "XGBoost (recent history)",
    "short_linear_log": "Linear, log-target (recent history)",
    "short_linear": "Linear (recent history)",
}


def model_name(horizon, kind):
    return f"{horizon}_{kind}"


def _model_path(horizon, location_key, kind):
    return os.path.join(MODELS_DIR, f"{horizon}_{location_key}_{FILENAME_SUFFIX[kind]}.json")


def _load_one(path, kind):
    if kind == "xgboost":
        booster = xgb.Booster()
        booster.load_model(path)
        return booster
    with open(path) as f:
        return json.load(f)


def load_all(location_name):
    """
    {model_name: (kind, loaded_model)} for every variant that has a file
    for this location, in MODEL_VARIANTS order.
    """
    location_key = LOCATION_KEYS[location_name]
    models = {}
    for horizon, kind in MODEL_VARIANTS:
        path = _model_path(horizon, location_key, kind)
        if os.path.exists(path):
            models[model_name(horizon, kind)] = (kind, _load_one(path, kind))
    return models


def _seasonal_features(today):
    doy = today.timetuple().tm_yday
    return {
        "sin_doy": math.sin(2 * math.pi * doy / 365),
        "cos_doy": math.cos(2 * math.pi * doy / 365),
    }


def _build_feature_row(live_features, zsd_lag1, today):
    row = dict(live_features)
    row["zsd_lag1"] = zsd_lag1
    row.update(_seasonal_features(today))
    return row


def _predict_linear(model, row):
    coefs = model["coefficients"]
    total = coefs.get("(Intercept)", 0.0)
    for name, coef in coefs.items():
        if name == "(Intercept)":
            continue
        value = row.get(name)
        if value is None:
            return None
        total += coef * float(value)
    if "smearing_factor" in model:
        return max(0.0, model["smearing_factor"] * math.exp(total))
    return max(0.0, total)


def _predict_xgboost(booster, row):
    feature_names = booster.feature_names
    values = []
    for name in feature_names:
        value = row.get(name)
        if value is None:
            return None
        values.append(float(value))
    dmatrix = xgb.DMatrix(np.array([values], dtype=float), feature_names=feature_names)
    prediction = float(booster.predict(dmatrix)[0])
    return max(0.0, prediction)


def predict_visibility(kind, model, live_features, zsd_lag1, today):
    """
    Predict today's Secchi depth (m) with one loaded model. Returns None if
    there's no zsd_lag1 to anchor on yet, or a required feature is missing
    from live_features (e.g. a "short" model asked to run before its extra
    marine variables are populated).
    """
    if zsd_lag1 is None:
        return None
    row = _build_feature_row(live_features, zsd_lag1, today)
    if kind == "xgboost":
        return _predict_xgboost(model, row)
    return _predict_linear(model, row)
