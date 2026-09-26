"""
Live feature fetch shared by every model in models.py.

At inference time there's no "yesterday" grouped by calendar day to read
off a finished gold row for - we want a prediction for right now, before
today's gold row can even exist (it needs a satellite zsd that isn't in
yet). So LIVE_FEATURES_SQL recomputes the same aggregates over a trailing
24h window ending now, instead of a calendar day, computed the same way
gold_pipeline.py does from marine_hourly/weather_hourly. The trend
regressor switches from gold_pipeline's "hour of day, 0-23" (which wraps
at midnight) to "epoch hours" - both are affine in real time with the
same 1-unit-per-hour scale, and a least-squares slope is invariant to an
affine shift of x, so the resulting trend values land on the same scale
the models were trained on.
"""

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


def get_live_features(conn, location_id):
    """Trailing-24h wave/current/rain aggregates for one location, as a dict."""
    with conn.cursor() as cur:
        # int(...): psycopg2 has no adapter for numpy.int64, which is what
        # callers pull out of a pandas DataFrame column.
        cur.execute(LIVE_FEATURES_SQL, {"location_id": int(location_id)})
        columns = [desc[0] for desc in cur.description]
        row = cur.fetchone()
    return dict(zip(columns, row))
