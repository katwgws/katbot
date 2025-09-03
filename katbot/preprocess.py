import json
import os
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

alpha_re = re.compile(r"[a-zA-Z]")
mention_re = re.compile(r"(^|[^@\w])@[\w_]{1,15}\b")
ts_re = re.compile(r"^window\.YTD\.(tweets|direct_messages|direct_messages_group)\.part\d+\s*=\s*")
url_re = re.compile(r"https?://", re.IGNORECASE)


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


DATA_DIR = Path("./data")
RAW_DIR = DATA_DIR / "raw"
TWEET_PATHS = [p for p in RAW_DIR.glob("tweet*.js")]
DM_PATHS = [p for p in RAW_DIR.glob("direct*.js")]
OUTPUT_PATH = DATA_DIR / "training_corpus.jsonl"

# WARNING: Using DMs is a REALLY BAD IDEA!!! DO NOT DO THIS!!!!!
USE_DMS = os.getenv("USE_DMS", "false").lower() in {"true", "yes", "1"}

SENDER_ID = "1350174234455117826"
MIN_CHARS = 10
MAX_CHARS = 220
MIN_WORDS = 3
DATE_FMT = r"%a %b %d %H:%M:%S %z %Y"


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def _parse_date(ts: str) -> str:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        dt = datetime.strptime(ts, DATE_FMT)
        return dt.isoformat()
    except Exception:
        return ts


def _scrub_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = "".join(
        ch for ch in cleaned if not unicodedata.category(ch).startswith("C") or ch in ("\n", "\t")
    ).strip()
    cleaned = mention_re.sub("", cleaned)
    cleaned = url_re.sub("", cleaned)
    if (
        (not alpha_re.search(cleaned))
        or len(cleaned) < MIN_CHARS
        or len(cleaned) >= MAX_CHARS
        or len(cleaned.split()) < MIN_WORDS
    ):
        return ""
    return cleaned.strip()


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def extract_tweets(in_path: Path, out_path: Path) -> None:
    raw = in_path.read_text(encoding="utf-8").strip()
    raw = json.loads(ts_re.sub("", raw))

    with out_path.open("a", encoding="utf-8") as f:
        for obj in tqdm(raw, desc=f"Extracting from {in_path.name}"):
            if not (t := obj.get("tweet")):
                continue

            if not (date := t.get("created_at")):
                continue
            date = _parse_date(date)

            if not (text := t.get("full_text", t.get("text", "")).strip()):
                continue
            if text.startswith("RT @"):
                continue
            if not (text := _scrub_text(text)):
                continue

            tweet = {"date": date, "type": "tweet", "text": text}
            f.write(json.dumps(tweet, ensure_ascii=False) + "\n")


def extract_dms(in_path: Path, out_path: Path) -> None:
    raw = in_path.read_text(encoding="utf-8").strip()
    raw = json.loads(ts_re.sub("", raw))

    with out_path.open("a", encoding="utf-8") as f:
        for conv in tqdm(raw, desc=f"Extracting from {in_path.name}"):
            if not (msgs := conv.get("dmConversation", {}).get("messages", [])):
                continue
            for m in reversed(msgs):
                if not (msg := m.get("messageCreate")):
                    continue
                if msg.get("senderId") != SENDER_ID:
                    continue

                if not (date := msg.get("createdAt")):
                    continue
                date = _parse_date(date)

                if not (text := msg.get("text", "").strip()):
                    continue
                if not (text := _scrub_text(text)):
                    continue

                dm = {"type": "dm", "date": date, "text": text}
                f.write(json.dumps(dm, ensure_ascii=False) + "\n")


def finalize_corpus(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    items = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]

    items.sort(key=lambda x: x["date"])

    final_count = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in tqdm(items, desc="Finalizing"):
            obj_hash = sha256(obj["text"].encode("utf-8")).hexdigest()
            f.write(json.dumps({"hash": obj_hash} | obj, ensure_ascii=False) + "\n")
            final_count += 1

    print(f"Pre-processed {final_count:,} items for training")


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def main(
    *,
    raw_tweet_paths: Sequence[Path] = TWEET_PATHS,
    raw_dm_paths: Sequence[Path] | None = DM_PATHS,
    output_path: Path = OUTPUT_PATH,
    use_dms: bool = USE_DMS,
) -> None:
    output_path.unlink(missing_ok=True)

    for path in raw_tweet_paths:
        extract_tweets(path, output_path)

    if use_dms:
        for path in raw_dm_paths or []:
            extract_dms(path, output_path)

    finalize_corpus(output_path)


if __name__ == "__main__":
    main()
