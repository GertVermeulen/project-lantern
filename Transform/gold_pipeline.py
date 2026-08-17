"""
Gold layer: daily feature table joining marine_hourly, weather_hourly, and
ocean_color_daily (silver) into one row per location/day, ready for model
training.

Aggregation from hourly to daily:
    - Scalar variables (wave_height, wave_period, ocean_current_velocity,
      sea_surface_temperature, precipitation) get mean, trend, min, max.
      Trend is the least-squares slope of the hourly values against hour-
      of-day (Postgres regr_slope), not an endpoint difference - it uses
      all 24 hours and isn't thrown off by one noisy reading.
    - precipitation gets SUM instead of mean (it's a flux - total daily
      accumulation is what matters for runoff, not the average rate).
    - wave_direction/ocean_current_direction are circular (0/360 are the
      same value) - averaging or trending raw degrees breaks at the
      wraparound, so instead of a scalar mean these are represented as
      unit vectors: the mean sin and mean cos of the hourly readings, plus
      the mean resultant length R = sqrt(mean_sin^2 + mean_cos^2) as a
      "concentration" score (1 = all hourly readings pointed the same way,
      0 = scattered/opposing). No trend/min/max, since "max compass
      bearing" isn't a meaningful quantity.

ocean_color_daily is already daily grain, so it's joined in as-is with no
aggregation. It's the spine of the join (LEFT JOIN target) since it's the
narrowest date range and holds the training label (zsd/kd490).

Setup:
    Same DB_DSN / .env as marine_pipeline.py.
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

BUILD_SQL = """
WITH marine_agg AS (
    SELECT
        location_id,
        ts::date AS date,

        AVG(wave_height) AS wave_height_mean,
        regr_slope(wave_height, EXTRACT(HOUR FROM ts)) AS wave_height_trend,
        MIN(wave_height) AS wave_height_min,
        MAX(wave_height) AS wave_height_max,

        AVG(wave_period) AS wave_period_mean,
        regr_slope(wave_period, EXTRACT(HOUR FROM ts)) AS wave_period_trend,
        MIN(wave_period) AS wave_period_min,
        MAX(wave_period) AS wave_period_max,

        AVG(SIN(RADIANS(wave_direction))) AS wave_direction_sin_mean,
        AVG(COS(RADIANS(wave_direction))) AS wave_direction_cos_mean,
        SQRT(POWER(AVG(SIN(RADIANS(wave_direction))), 2) + POWER(AVG(COS(RADIANS(wave_direction))), 2))
            AS wave_direction_concentration,

        AVG(ocean_current_velocity) AS ocean_current_velocity_mean,
        regr_slope(ocean_current_velocity, EXTRACT(HOUR FROM ts)) AS ocean_current_velocity_trend,
        MIN(ocean_current_velocity) AS ocean_current_velocity_min,
        MAX(ocean_current_velocity) AS ocean_current_velocity_max,

        AVG(SIN(RADIANS(ocean_current_direction))) AS ocean_current_direction_sin_mean,
        AVG(COS(RADIANS(ocean_current_direction))) AS ocean_current_direction_cos_mean,
        SQRT(POWER(AVG(SIN(RADIANS(ocean_current_direction))), 2) + POWER(AVG(COS(RADIANS(ocean_current_direction))), 2))
            AS ocean_current_direction_concentration,

        AVG(sea_surface_temperature) AS sea_surface_temperature_mean,
        regr_slope(sea_surface_temperature, EXTRACT(HOUR FROM ts)) AS sea_surface_temperature_trend,
        MIN(sea_surface_temperature) AS sea_surface_temperature_min,
        MAX(sea_surface_temperature) AS sea_surface_temperature_max
    FROM marine_hourly
    GROUP BY location_id, ts::date
),
weather_agg AS (
    SELECT
        location_id,
        ts::date AS date,
        SUM(precipitation) AS precipitation_sum,
        regr_slope(precipitation, EXTRACT(HOUR FROM ts)) AS precipitation_trend,
        MIN(precipitation) AS precipitation_min,
        MAX(precipitation) AS precipitation_max
    FROM weather_hourly
    GROUP BY location_id, ts::date
)
INSERT INTO gold.daily_features (
    location_id, date,
    wave_height_mean, wave_height_trend, wave_height_min, wave_height_max,
    wave_period_mean, wave_period_trend, wave_period_min, wave_period_max,
    wave_direction_sin_mean, wave_direction_cos_mean, wave_direction_concentration,
    ocean_current_velocity_mean, ocean_current_velocity_trend, ocean_current_velocity_min, ocean_current_velocity_max,
    ocean_current_direction_sin_mean, ocean_current_direction_cos_mean, ocean_current_direction_concentration,
    sea_surface_temperature_mean, sea_surface_temperature_trend, sea_surface_temperature_min, sea_surface_temperature_max,
    precipitation_sum, precipitation_trend, precipitation_min, precipitation_max,
    kd490, zsd, chl
)
SELECT
    o.location_id, o.date,
    m.wave_height_mean, m.wave_height_trend, m.wave_height_min, m.wave_height_max,
    m.wave_period_mean, m.wave_period_trend, m.wave_period_min, m.wave_period_max,
    m.wave_direction_sin_mean, m.wave_direction_cos_mean, m.wave_direction_concentration,
    m.ocean_current_velocity_mean, m.ocean_current_velocity_trend, m.ocean_current_velocity_min, m.ocean_current_velocity_max,
    m.ocean_current_direction_sin_mean, m.ocean_current_direction_cos_mean, m.ocean_current_direction_concentration,
    m.sea_surface_temperature_mean, m.sea_surface_temperature_trend, m.sea_surface_temperature_min, m.sea_surface_temperature_max,
    w.precipitation_sum, w.precipitation_trend, w.precipitation_min, w.precipitation_max,
    o.kd490, o.zsd, o.chl
FROM ocean_color_daily o
LEFT JOIN marine_agg m ON m.location_id = o.location_id AND m.date = o.date
LEFT JOIN weather_agg w ON w.location_id = o.location_id AND w.date = o.date
ON CONFLICT (location_id, date) DO UPDATE SET
    wave_height_mean = EXCLUDED.wave_height_mean,
    wave_height_trend = EXCLUDED.wave_height_trend,
    wave_height_min = EXCLUDED.wave_height_min,
    wave_height_max = EXCLUDED.wave_height_max,
    wave_period_mean = EXCLUDED.wave_period_mean,
    wave_period_trend = EXCLUDED.wave_period_trend,
    wave_period_min = EXCLUDED.wave_period_min,
    wave_period_max = EXCLUDED.wave_period_max,
    wave_direction_sin_mean = EXCLUDED.wave_direction_sin_mean,
    wave_direction_cos_mean = EXCLUDED.wave_direction_cos_mean,
    wave_direction_concentration = EXCLUDED.wave_direction_concentration,
    ocean_current_velocity_mean = EXCLUDED.ocean_current_velocity_mean,
    ocean_current_velocity_trend = EXCLUDED.ocean_current_velocity_trend,
    ocean_current_velocity_min = EXCLUDED.ocean_current_velocity_min,
    ocean_current_velocity_max = EXCLUDED.ocean_current_velocity_max,
    ocean_current_direction_sin_mean = EXCLUDED.ocean_current_direction_sin_mean,
    ocean_current_direction_cos_mean = EXCLUDED.ocean_current_direction_cos_mean,
    ocean_current_direction_concentration = EXCLUDED.ocean_current_direction_concentration,
    sea_surface_temperature_mean = EXCLUDED.sea_surface_temperature_mean,
    sea_surface_temperature_trend = EXCLUDED.sea_surface_temperature_trend,
    sea_surface_temperature_min = EXCLUDED.sea_surface_temperature_min,
    sea_surface_temperature_max = EXCLUDED.sea_surface_temperature_max,
    precipitation_sum = EXCLUDED.precipitation_sum,
    precipitation_trend = EXCLUDED.precipitation_trend,
    precipitation_min = EXCLUDED.precipitation_min,
    precipitation_max = EXCLUDED.precipitation_max,
    kd490 = EXCLUDED.kd490,
    zsd = EXCLUDED.zsd,
    chl = EXCLUDED.chl
"""


def build():
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute(BUILD_SQL)
        print(f"Upserted {cur.rowcount} gold.daily_features rows.")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    build()
