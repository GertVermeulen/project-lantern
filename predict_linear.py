"""
Live inference for the reduced linear model (Back_model_reduced_NL), fit in R
via lm(log(zsd) ~ ..., data = d) and exported to
R Models/linear_log_model.json as {"coefficients": {...}, "smearing_factor": ...}.

Predicts zsd by evaluating the linear combination in log-space, then back-
transforming with exp() - corrected by Duan's smearing estimator
(mean(exp(residuals)) from the R fit, baked into smearing_factor at export
time). Naively exponentiating the raw log-scale prediction would
systematically underestimate zsd on the original scale (Jensen's
inequality: E[exp(X)] > exp(E[X])).

Generic over whichever coefficients the JSON contains - unlike predict.py's
FEATURE_NAMES list, there's no fixed feature order to match (a linear
combination doesn't care what order the terms are summed in), so this
works unmodified if the R model's formula changes, as long as every
coefficient name has a matching key in live_features (or is "zsd_lag1").
"""

import json
import math

MODEL_PATH = "R Models/linear_log_model.json"


def load_model(model_path=MODEL_PATH):
    with open(model_path) as f:
        return json.load(f)


def predict_visibility(model, live_features, zsd_lag1):
    """
    Predict today's Secchi depth (m). Returns None if there's no zsd_lag1 to
    anchor on yet (same convention as predict.predict_visibility), or if a
    coefficient's feature is missing from live_features.
    """
    if zsd_lag1 is None:
        return None

    features = dict(live_features)
    features["zsd_lag1"] = zsd_lag1

    coefs = model["coefficients"]
    log_zsd = coefs.get("(Intercept)", 0.0)
    for name, coef in coefs.items():
        if name == "(Intercept)":
            continue
        value = features.get(name)
        if value is None:
            return None
        log_zsd += coef * float(value)

    predicted = model["smearing_factor"] * math.exp(log_zsd)
    return max(0.0, predicted)
