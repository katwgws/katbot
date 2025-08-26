# ⋅⋆•°☙ katbot/tweet.py ❧°•⋆⋅

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from logging import getLogger
from operator import attrgetter
from typing import Any, Final, Self
from xml.etree import ElementTree as ET

from requests_oauthlib import OAuth1

from .config import cfg
from .http import request
from .tweet import Tweet

log = getLogger(__name__)

__all__: Final = [
    "Tweet",
    "load_tweets",
    "post_tweet",
    "save_tweets",
]

_MENTION_RE = re.compile(r"(^|[^@\w])@[\w_]{1,15}\b")
_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def _hash(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _make_text(prompt: str = "", system: str = "", **llm_kwargs) -> str:
    payload = {
        "model": cfg.model,
        "messages": [
            *([{"role": "system", "content": system}] if system else []),
            {"role": "user", "content": f"{prompt} <TWEET>".strip()},
        ],
        "stream": False,
    } | llm_kwargs

    url = cfg.model_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    r = request("POST", url, headers=headers, json=payload)
    data = r.json()

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Bad model response: {data!r}") from e


def _make_topic() -> str:
    r = request("GET", cfg.topic_api)
    root_xml = ET.fromstring(r.content)

    titles = [
        (el.text or "").strip()
        for el in root_xml.iter()
        if el.tag.endswith("news_item_title") and el.text and el.text.strip()
    ] or [
        (el.text or "").strip()
        for el in root_xml.findall(".//item/title")
        if el.text and el.text.strip()
    ]
    return random.choice(titles) if titles else ""


@dataclass(slots=True, frozen=True)
class Tweet:
    text: str
    hash_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    topic: str | None = None
    url: str | None = None

    @classmethod
    def generate(cls) -> Self:
        retries = cfg.max_generate_retries
        while retries >= 0:
            if random.random() <= cfg.use_topic:
                topic = _make_topic()
                text = _make_text(system=topic)
                tweet = cls(text, _hash(text), topic=topic)
            else:
                text = _make_text()
                tweet = cls(text, _hash(text))
            if tweet.validate():
                log.info("New tweet: %s", tweet)
                return tweet
            log.warning("Invalid tweet: %s", tweet)
            retries -= 1
        raise RuntimeError(f"No valid tweet after {cfg.max_generate_retries} attempts")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        try:
            params = (
                {
                    "created_at": datetime.fromisoformat(data["created_at"]),
                    "hash_id": data.get("hash", _hash(data["text"])),
                    "text": data["text"],
                }
                | ({"topic": data["topic"]} if data.get("topic") else {})
                | ({"url": data["url"]} if data.get("url") else {})
            )
            return cls(**params)
        except KeyError as e:
            raise RuntimeError(f"Error parsing tweet: {data!r}") from e

    def validate(self) -> bool:
        return (
            len(self.text) >= cfg.min_tweet_len
            and len(self.text) <= cfg.max_tweet_len
            and not _MENTION_RE.search(self.text)
            and not _URL_RE.search(self.text)
            and ("sorry" not in self.text or random.random() <= cfg.allow_apology)
            and ("ball" not in self.text or random.random() <= cfg.allow_ball)
        )

    def as_dict(self) -> dict[str, Any]:
        return (
            {
                "created_at": self.created_at.isoformat(),
                "hash": self.hash_id,
                "text": self.text,
                "length": len(self.text),
            }
            | ({"topic": self.topic} if self.topic else {})
            | ({"url": self.url} if self.url else {})
        )

    def as_str(self) -> str:
        return f"<{self.hash_id}> {self.text}"

    def __repr__(self) -> str:  # pragma: no cover
        return self.as_str()


def load_tweets() -> list[Tweet]:
    text = cfg.tweets_file.read_text(encoding="utf-8")
    raw_tweets = [json.loads(ln.strip()) for ln in text.splitlines() if ln.strip()]
    tweets = [Tweet.from_dict(raw) for raw in raw_tweets]
    return sorted(tweets, key=attrgetter("created_at"))


def save_tweets(tweets: Sequence[Tweet]) -> None:
    sorted_tweets = sorted(tweets, key=attrgetter("created_at"))
    lines = [json.dumps(t.as_dict(), ensure_ascii=False) for t in sorted_tweets]
    cfg.tweets_file.write_text("\n".join(lines), encoding="utf-8")


def post_tweet(tweet: Tweet) -> Tweet:
    url = cfg.twitter_api.rstrip("/") + "/tweets"
    auth = OAuth1(
        cfg.twitter_api_key,
        cfg.twitter_api_secret,
        cfg.twitter_access_token,
        cfg.twitter_access_secret,
    )
    payload = {"text": tweet.text}

    try:
        r = request("POST", url, auth=auth, json=payload)

        tweet_id = r.json()["data"]["id"]
        url = cfg.twitter_base_url.rstrip("/") + f"/{tweet_id}"

        log.info("Post successful: %s", url)
        return replace(tweet, url=url)

    except Exception as e:
        log.error("Post failed: %s", e)
        return tweet
