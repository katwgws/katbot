"""https://github.com/katwgws/katbot"""

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal
from xml.etree import ElementTree as ET

import requests
from dotenv import load_dotenv
from requests_oauthlib import OAuth1
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()


# -----------------------------------------------------------------------------
#  Settings
# -----------------------------------------------------------------------------


DEBUG: Final = os.getenv("KATBOT_DEBUG", "").lower() in {"true", "yes"}

HTTP_MAX_RETRIES: Final = 10
HTTP_MAX_TIMEOUT: Final = 30.0

USE_TOPIC: Final = os.getenv("KATBOT_USE_TOPIC", "").lower() in {"true", "yes"}
TOPIC_API: Final = "https://trends.google.com/trending/rss?geo=US"

MODEL: Final = os.getenv("KATBOT_MODEL", "")
MODEL_URL: Final = os.getenv("KATBOT_MODEL_URL", "")

MAX_TWEET_LEN: Final = 200
MIN_TWEET_LEN: Final = 20
TWEETS_PATH: Final = os.getenv("KATBOT_TWEETS_PATH", "./tweets.jsonl")

TWT_API: Final = "https://api.x.com/2"
TWT_BASE_URL: Final = "https://twitter.com/i/web/status/"


# --- HTTP helper -------------------------------------------------------------


def _can_retry(e: BaseException) -> bool:
    if isinstance(e, requests.Timeout | requests.ConnectionError):
        return True
    if isinstance(e, requests.HTTPError):
        try:
            code = e.response.status_code
        except Exception:
            return False
        return code in {429, 500, 520, 503, 504}
    return False


@retry(
    wait=wait_random_exponential(max=HTTP_MAX_TIMEOUT),
    stop=stop_after_attempt(HTTP_MAX_RETRIES),
    retry=retry_if_exception(_can_retry),
    reraise=True,
)
def request(
    method: Literal["POST", "GET"],
    url: str,
    **request_kwargs: Any,
) -> requests.Response:
    r = requests.request(
        method,
        url,
        headers={"User-Agent": "katbot/1.0 (+https://x.com)"}
        | request_kwargs.pop("headers", {}),
        timeout=HTTP_MAX_TIMEOUT,
        **request_kwargs,
    )
    r.raise_for_status()
    return r


# --- Tweet dataclass ---------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Tweet:
    text: str
    topic: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def abridged(self) -> str:
        if len(self.text) <= MAX_TWEET_LEN:
            return self.text
        else:
            return self.text[: max(0, MAX_TWEET_LEN - 3)] + "..."

    @property
    def hash_id(self) -> str:
        return hashlib.md5(self.text.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        data = {
            "created_at": self.created_at.isoformat(sep=" "),
            "hash": self.hash_id,
            "text": self.text,
        }
        if self.topic:
            return data | {"topic": self.topic}
        else:
            return data

    def as_str(self) -> str:
        return f"{self.text} <{self.hash_id}>"

    def __repr__(self) -> str:  # pragma: no cover
        return self.as_str()


# --- Generators --------------------------------------------------------------


def generate_topic() -> str:
    r = request("GET", TOPIC_API)
    root_xml = ET.fromstring(r.content)

    titles = [
        (el.text or "").strip()
        for el in root_xml.iter()
        if el.tag.endswith("news_item_title") and el.text and el.text.strip()
    ] or [
        (el.text or "").strip()
        for el in root_xml.findall("//item/title")
        if el.text and el.text.strip()
    ]
    return random.choice(titles) if titles else ""


def generate_text(
    prompt: str = "",
    system: str = "",
    **model_params,
) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{prompt} <TWEET>".strip()},
    ]

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    } | model_params

    url = MODEL_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    r = request("POST", url, headers=headers, json=payload)

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected response schema: {data!r}") from e


# --- Helpers -----------------------------------------------------------------


def make_tweet(**kwargs) -> Tweet:
    if USE_TOPIC:
        topic = generate_topic()
        text = generate_text(system=topic, **kwargs)
        tweet = Tweet(text=text, topic=topic)
    else:
        text = generate_text(**kwargs)
        tweet = Tweet(text)
    print(f"[Tweet]: {tweet}")
    return tweet


def post_tweet(tweet: Tweet, **kwargs) -> str | None:
    try:
        url = TWT_API.rstrip("/") + "/tweets"
        payload = {"text": tweet.abridged} | kwargs
        auth = OAuth1(
            os.getenv("KATBOT_TWT_API_KEY"),
            os.getenv("KATBOT_TWT_API_SECRET"),
            os.getenv("KATBOT_TWT_ACCESS_TOKEN"),
            os.getenv("KATBOT_TWT_ACCESS_SECRET"),
        )
        r = request("POST", url, auth=auth, json=payload)
        tweet_id = r.json()["data"]["id"]
        url = TWT_BASE_URL.rstrip("/") + f"/{tweet_id}"
        print(f"[Ok!]: {url}")
        return url

    except Exception as e:
        print(f"[Failed]: {e}")
        return None


def save_tweet(
    tweet: Tweet,
    url: str | None = None,
) -> None:
    raw = json.dumps(
        tweet.as_dict() | {"url": url} if url else {},
        ensure_ascii=False,
    )
    with Path(TWEETS_PATH).open("a", encoding="utf-8") as f:
        f.write(raw + "\n")
        f.flush()
        os.fsync(f.fileno())


# --- Main --------------------------------------------------------------------


if __name__ == "__main__":
    tweet = make_tweet()
    url = post_tweet(tweet) if not DEBUG else None
    save_tweet(tweet, url)
