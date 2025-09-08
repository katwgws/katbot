import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

DATA_PATH = Path("data/training_corpus.jsonl")


records = []
with DATA_PATH.open(encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))


df = pd.DataFrame(records)
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df["created_at"] = df["created_at"].dt.tz_convert(ZoneInfo("America/Los_Angeles"))
df["hour"] = df["created_at"].dt.hour
df["engagement"] = df["favs"] + df["rts"]

result = df.groupby("hour")["engagement"].mean().reset_index().sort_values("hour")

print(result)
