"""
Exports gold.daily_features to a CSV file, for use outside this project
(e.g. loading into R for model training).

Setup:
    Same DB_DSN / .env as the other pipeline scripts.
"""

import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_DSN = os.environ.get(
    "DB_DSN", "dbname=mydb user=myuser password=mypass host=localhost"
)

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "gold_daily_features.csv")

EXPORT_SQL = """
SELECT l.name AS location_name, g.*
FROM gold.daily_features g
JOIN locations l ON l.location_id = g.location_id
ORDER BY g.location_id, g.date
"""


def export(output_file=OUTPUT_FILE):
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute(EXPORT_SQL)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_file, index=False)
    print(f"Wrote {len(df)} rows to {output_file}")


if __name__ == "__main__":
    export()
