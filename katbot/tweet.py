import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Self

from dotenv import load_dotenv
from requests_oauthlib import OAuth1

from .helpers import request

load_dotenv()
logger = logging.getLogger(__name__)

MENTION_RE = re.compile(r"(^|[^@\w])@[\w_]{1,15}\b")
URL_RE = re.compile(r"https?://", re.IGNORECASE)

SEED = int(os.environ["SEED"]) if os.getenv("SEED") else None
if SEED is not None:
    random.seed(SEED)


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


DRY_RUN = os.getenv("DRY_RUN", "").casefold().strip() in {"true", "yes", "1"}

MODEL = os.getenv("KATBOT_MODEL", "")
if not MODEL:
    raise RuntimeError("'KATBOT_MODEL' must be set in environment")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:1234/v1")

DATA_DIR = Path("./data")
QUEUE_FILE = DATA_DIR / "tweet_queue.txt"
LOG_FILE = DATA_DIR / "tweet_log.jsonl"

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "32"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "200"))
MIN_LENGTH = int(os.getenv("MIN_LENGTH", "10"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TWEET_TOKEN = "<|tweet|>"

TWEET_BASE_URL = "https://twitter.com/i/web/status/"
TWITTER_URL = "https://api.x.com/2"
TWITTER_APP_NAME = os.getenv("TWITTER_APP_NAME", "katbot")
TWITTER_SECRETS: dict[str, Any] = {
    "client_key": os.getenv("TWITTER_API_KEY"),
    "client_secret": os.getenv("TWITTER_API_SECRET"),
    "resource_owner_key": os.getenv("TWITTER_ACCESS_TOKEN"),
    "resource_owner_secret": os.getenv("TWITTER_ACCESS_SECRET"),
}
for k, v in TWITTER_SECRETS.items():
    if not v:
        raise RuntimeError(f"'{k}' must be set in environment")


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


@dataclass(slots=True, frozen=True)
class Tweet:
    text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    url: str | None = None

    @property
    def hash_id(self):
        return sha256(self.text.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        try:
            return cls(
                text=data["text"],
                created_at=(
                    datetime.fromisoformat(data["created_at"])
                    if data.get("created_at")
                    else datetime.now(UTC)
                ),
                url=data["url"] if data.get("url") else None,
            )
        except KeyError as e:
            raise RuntimeError(f"Error parsing tweet: {data!r}") from e

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash_id,
            "created_at": self.created_at.isoformat(),
            "text": self.text,
            "length": len(self.text),
        } | ({"url": self.url} if self.url else {})

    def __repr__(self) -> str:
        return f"{self.text} <{self.url}>" if self.url else self.text


def _generate_text(
    *,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    max_tokens: int = MAX_TOKENS,
    **llm_kwargs,
) -> str:
    payload = {
        "model": MODEL,
        "prompt": TWEET_TOKEN,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "stream": False,
    } | llm_kwargs

    url = MODEL_URL.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    r = request("POST", url, headers=headers, json=payload)
    data = r.json()

    try:
        return str(data["choices"][0]["text"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Bad model response: {data!r}") from e


def _validate_tweet(tweet: Tweet) -> bool:
    return (
        len(tweet.text) >= MIN_LENGTH
        and len(tweet.text) <= MAX_LENGTH
        and not MENTION_RE.search(tweet.text)
        and not URL_RE.search(tweet.text)
    )


def generate_tweet() -> Tweet:
    print("Generating tweet ...")

    retries = MAX_RETRIES
    while retries >= 0:
        text = _generate_text().replace(TWEET_TOKEN, "").strip()
        tweet = Tweet(text)

        if _validate_tweet(tweet):
            print(f"[🐣]: {tweet}")
            return tweet

        print(f"[🚫]: {tweet}")
        retries -= 1
    raise RuntimeError(f"No valid tweet after {MAX_RETRIES} attempts")


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent)
    ) as tf:
        tf.write(content)
        tmp_name = tf.name
    os.replace(tmp_name, path)


def _append_jsonl(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_queue() -> list[str]:
    if not QUEUE_FILE.is_file():
        print(f"'{QUEUE_FILE}' not found, starting with empty queue.")
        return []
    return [ln.strip() for ln in QUEUE_FILE.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_queue(lines: Sequence[str]) -> None:
    content = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write_text(QUEUE_FILE, content)


def append_log(tweet: Tweet) -> None:
    _append_jsonl(LOG_FILE, tweet.to_dict())


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def post_tweet(tweet: Tweet) -> Tweet:
    url = TWITTER_URL.rstrip("/") + "/tweets"
    auth = OAuth1(**TWITTER_SECRETS)
    payload = {"text": tweet.text}
    headers = {"User-Agent": f"{TWITTER_APP_NAME}/1.0 (+https://x.com)"}

    try:
        r = request("POST", url, headers, auth=auth, json=payload)

        tweet_id = r.json()["data"]["id"]
        url = TWEET_BASE_URL.rstrip("/") + "/" + tweet_id

        print(f"[📡]: {url}")
        return replace(tweet, url=url)

    except Exception as e:
        print(f"[🚫]: {e}")
        return tweet


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def main() -> None:
    if DRY_RUN:
        print("Dry run enabled, no posts will be made or files modified.")

    queue = read_queue()

    picked_from_queue = False
    if queue:
        print(f"Found {len(queue):,} queued tweets.")
        text = random.choice(queue)
        picked_from_queue = True
        tweet = Tweet(text=text)
        print(f"[🐤]: {tweet}")
    else:
        tweet = generate_tweet()

    if DRY_RUN:
        return

    posted = post_tweet(tweet)
    if posted.url:
        append_log(posted)
        if picked_from_queue:
            try:
                idx = queue.index(tweet.text)
                del queue[idx]
            except ValueError:
                pass
            write_queue(queue)
    else:
        if picked_from_queue:
            queue.append(tweet.text)
            write_queue(queue)

    print("All done! ✨")


if __name__ == "__main__":
    main()
