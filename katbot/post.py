# ⋅⋆•°☙ katbot/post.py ❧°•⋆⋅

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from xml.etree import ElementTree as ET

from requests_oauthlib import OAuth1

from .config import cfg
from .http import request

__all__: Final = ["Tweet", "run"]


@dataclass(slots=True, frozen=True)
class Tweet:
    text: str
    topic: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def abridged(self) -> str:
        if len(self.text) <= cfg.min_tweet_len:
            return self.text
        else:
            return self.text[: max(0, cfg.max_tweet_len - 3)] + "..."

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


def generate_topic() -> str:
    r = request("GET", cfg.topic_api)
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
        "model": cfg.model,
        "messages": messages,
        "stream": False,
    } | model_params

    url = cfg.model_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    r = request("POST", url, headers=headers, json=payload)

    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"unexpected response schema: {data!r}") from e


# ⋅⋆•°☙ HELPERS ❧°•⋆⋅


def make_tweet(**kwargs) -> Tweet:
    if cfg.use_topic:
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
        url = cfg.twitter_api.rstrip("/") + "/tweets"
        payload = {"text": tweet.abridged} | kwargs
        auth = OAuth1(
            cfg.twitter_api_key,
            cfg.twitter_api_secret,
            cfg.twitter_access_token,
            cfg.twitter_access_secret,
        )
        r = request("POST", url, auth=auth, json=payload)
        tweet_id = r.json()["data"]["id"]
        url = cfg.twitter_base_url.rstrip("/") + f"/{tweet_id}"
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
        tweet.as_dict() | ({"url": url} if url else {}),
        ensure_ascii=False,
    )
    with Path(cfg.tweets_path).open("a", encoding="utf-8") as f:
        f.write(raw + "\n")
        f.flush()
        os.fsync(f.fileno())


def run():
    tweet = make_tweet()
    url = post_tweet(tweet) if not cfg.debug else None
    save_tweet(tweet, url)
