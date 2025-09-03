import json
import os
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from operator import attrgetter
from pathlib import Path
from typing import Any, Literal, Self

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

mention_re = re.compile(r"(^|[^@\w])@[\w_]{1,15}\b")
url_re = re.compile(r"https?://", re.IGNORECASE)


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


_seed = os.getenv("SEED")
if _seed:
    SEED = int(_seed)
    random.seed(SEED)
else:
    SEED = None

DRY_RUN = os.getenv("DRY_RUN", "").casefold().strip() in {"true", "yes", "1"}

HTTP_MAX_RETRIES = 3
HTTP_MAX_TIMEOUT = 240

MODEL = os.getenv("KATBOT_MODEL", "")
if not MODEL:
    raise RuntimeError("'KATBOT_MODEL' must be set in environment")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:1234/v1")

DATA_DIR = Path("./data")
TWEETS_FILE = DATA_DIR / "tweets.jsonl"

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


def _http_can_retry(e: BaseException) -> bool:
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
    retry=retry_if_exception(_http_can_retry),
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
        headers={"User-Agent": f"{TWITTER_APP_NAME}/1.0 (+https://x.com)"}
        | request_kwargs.pop("headers", {}),
        timeout=HTTP_MAX_TIMEOUT,
        **request_kwargs,
    )
    r.raise_for_status()
    return r


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
            timestamp = datetime.now(UTC)
            if data.get("created_at"):
                timestamp = datetime.fromisoformat(data["created_at"])
            tweet_kwargs = {
                "created_at": timestamp,
                "text": data["text"],
            } | ({"url": data["url"]} if data.get("url") else {})
            return cls(**tweet_kwargs)
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
        return self.text + f" <{self.url}>" if self.url else ""


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
        and not mention_re.search(tweet.text)
        and not url_re.search(tweet.text)
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


def load_tweets() -> list[Tweet]:
    print(f"Loading from '{TWEETS_FILE}' ...")
    if not TWEETS_FILE.is_file():
        print(f"'{TWEETS_FILE}' not found, skipping.")
        return []
    raw_text = TWEETS_FILE.read_text(encoding="utf-8")
    raw_tweets = [json.loads(ln.strip()) for ln in raw_text.splitlines() if ln.strip()]
    tweets = [Tweet.from_dict(raw) for raw in raw_tweets]
    return sorted(tweets, key=attrgetter("created_at"))


def save_tweets(tweets: Sequence[Tweet]) -> None:
    print(f"Saving to '{TWEETS_FILE}' ...")
    TWEETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    sorted_tweets = sorted(tweets, key=attrgetter("created_at"))
    lines = [json.dumps(t.to_dict(), ensure_ascii=False) for t in sorted_tweets]
    TWEETS_FILE.write_text("\n".join(lines), encoding="utf-8")


# - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -


def post_tweet(tweet: Tweet) -> Tweet:
    url = TWITTER_URL.rstrip("/") + "/tweets"
    auth = OAuth1(**TWITTER_SECRETS)
    payload = {"text": tweet.text}

    try:
        r = request("POST", url, auth=auth, json=payload)

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
        print("Dry run enabled, no posts will be made and no URLs will be updated.")
        print("Messages will continue as normal for testing.")

    tweets = load_tweets()
    queue = [t for t in tweets if not t.url]

    if queue:
        print(f"Found {len(queue):,} tweets without URLs.")
        print("Choosing and posting a tweet ...")
        tweet = random.choice(queue)
        print(f"[🐤]: {tweet}")
        tweets.remove(tweet)

    else:
        tweet = generate_tweet()

    tweet = post_tweet(tweet) if not DRY_RUN else tweet
    tweets.append(tweet)
    save_tweets(tweets)

    print("All done! ✨")


if __name__ == "__main__":
    main()
