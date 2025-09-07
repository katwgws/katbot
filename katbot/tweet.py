# katbot/tweet.py
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

import json
import logging
import os
import random
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1

load_dotenv()

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MENTION_RE = re.compile(r"(^|[^@\w])@[\w_]{1,15}\b")
URL_RE = re.compile(r"https?://", re.IGNORECASE)


# Config
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


SEED = int(os.environ["SEED"]) if os.getenv("SEED") else None
if SEED is not None:
    random.seed(SEED)

DRY_RUN = os.getenv("DRY_RUN", "").casefold().strip() in {"true", "yes", "1"}
NO_POST = os.getenv("NO_POST", "").casefold().strip() in {"true", "yes", "1"}
HTTP_MAX_TIMEOUT = 240

MODEL = os.getenv("KATBOT_MODEL", "")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:1234/v1")

DATA_DIR = Path("./data")
QUEUE_FILE = DATA_DIR / "tweet_queue.jsonl"
LOG_FILE = DATA_DIR / "tweet_log.jsonl"

TEMPERATURE = float(os.getenv("TEMPERATURE", "1.1"))
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


# HTTP
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def request(
    url: str,
    method: Literal["POST", "GET"] = "GET",
    **request_kwargs: Any,
) -> requests.Response:
    request_kwargs.setdefault("timeout", HTTP_MAX_TIMEOUT)
    try:
        r = requests.request(method, url, **request_kwargs)
        r.raise_for_status()
        return r
    except Exception:
        log.exception("HTTP error")
        raise


# Generator
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def _generate_tweet(
    model: str = MODEL,
    model_url: str = MODEL_URL,
    prompt: str = TWEET_TOKEN,
    *,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    repetition_penalty: float = REPETITION_PENALTY,
    max_tokens: int = MAX_TOKENS,
    **llm_kwargs: Any,
) -> str:
    if not model:
        raise RuntimeError("'KATBOT_MODEL' must be set in environment")

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
        "stream": False,
    } | llm_kwargs

    url = model_url.rstrip("/") + "/completions"
    headers = {"Content-Type": "application/json"}
    r = request(url, "POST", headers=headers, json=payload)
    data = r.json()
    try:
        return str(data["choices"][0]["text"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Bad model response: {data!r}") from e


def _validate_tweet(
    text: str,
    *,
    min_len: int = MIN_LENGTH,
    max_len: int = MAX_LENGTH,
) -> bool:
    return (
        len(text) >= min_len
        and len(text) <= max_len
        and not MENTION_RE.search(text)
        and not URL_RE.search(text)
    )


# Queue
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        log.warning("File not found: %s", path)
        return []
    raw = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


def _save_jsonl(path: Path, data: Sequence[Mapping[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = [json.dumps(ln, ensure_ascii=False) for ln in data]
    path.write_text("\n".join(raw), encoding="utf-8")


# Twitter
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def post_tweet(
    text: str,
    secrets: dict[str, Any] | None = None,
    *,
    base_url: str = TWEET_BASE_URL,
    api_url: str = TWITTER_URL.rstrip("/") + "/tweets",
    app_name: str = TWITTER_APP_NAME,
) -> str | None:
    log.info("Posting to: %s", api_url)
    _secrets = secrets or {
        "client_key": os.getenv("TWITTER_API_KEY"),
        "client_secret": os.getenv("TWITTER_API_SECRET"),
        "resource_owner_key": os.getenv("TWITTER_ACCESS_TOKEN"),
        "resource_owner_secret": os.getenv("TWITTER_ACCESS_SECRET"),
    }
    for k, v in _secrets.items():
        if not v:
            raise RuntimeError(f"'{k}' must be set in environment")
    auth = OAuth1(**_secrets)

    try:
        r = request(
            api_url,
            "POST",
            headers={"User-Agent": f"{app_name}/1.0 (+https://x.com)"},
            auth=auth,
            json={"text": text},
        )
        tweet_id = r.json()["data"]["id"]
        tweet_url = f"{base_url.rstrip('/')}/{tweet_id}"
        log.info("🔗 Posted: %s", tweet_url)
        return tweet_url

    except Exception:
        return None


# Main
# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def run_once(
    *,
    queue_file: Path = QUEUE_FILE,
    log_file: Path = LOG_FILE,
    gen_kwargs: Mapping[str, Any] | None = None,
    max_retries: int = MAX_RETRIES,
    validation_kwargs: Mapping[str, Any] | None = None,
    post_kwargs: Mapping[str, Any] | None = None,
) -> None:
    log.info("Katbot v1.2 🤖")
    if DRY_RUN:
        log.info("Dry run enabled, no posts will be made or files modified.")
    elif NO_POST:
        log.info("Posting disabled, generating & queueing tweet only.")

    queue: list[str] = [t["text"] for t in _load_jsonl(queue_file)]
    text = queue[0] if queue else ""
    is_gen = False

    if NO_POST or not text:
        retries = max_retries
        while retries > 0:
            text = _generate_tweet(**(gen_kwargs or {}))
            if _validate_tweet(text, **(validation_kwargs or {})):
                queue.append(text)
                is_gen = True
                break
            log.warning("Rejected: %s", text)
            retries -= 1
    if not text:
        raise RuntimeError(f"No valid tweet after {max_retries} attempts")
    log.info("🐣 %s: %s", "Generated" if is_gen else "Queued", text)

    if not DRY_RUN and not NO_POST:
        url = post_tweet(text, **(post_kwargs or {}))
        if not url:
            raise RuntimeError("Could not post tweet")
        if text in queue:
            queue.remove(text)
        tweet = {
            "text": text,
            "posted_at": datetime.now(UTC).isoformat(),
            "url": url,
        }
        posted = _load_jsonl(log_file)
        posted.append(tweet)
        _save_jsonl(log_file, posted)

    if not DRY_RUN:
        _save_jsonl(queue_file, [{"text": t.strip()} for t in queue if t.strip()])


if __name__ == "__main__":
    run_once()
