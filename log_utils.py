# log_utils.py
import os, csv, pandas as pd

LOG_FILE = "prediction_log.csv"
LOG_COLS = ["date","temperature","wind","pressure","humidity","cloud","predicted_rainfall"]

def append_log(row: dict):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLS)
        if not file_exists:
            writer.writeheader()
        # ensure all keys exist
        writer.writerow({k: row.get(k, "") for k in LOG_COLS})

def read_recent(n=50):
    if not os.path.isfile(LOG_FILE):
        return []
    try:
        df = pd.read_csv(LOG_FILE)
        return df.tail(n).to_dict(orient="records")
    except Exception:
        return []
