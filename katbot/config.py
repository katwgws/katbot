# ⋅⋆•°☙ katbot/config.py ❧°•⋆⋅

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

__all__: Final = [
    "cfg",
]


@dataclass(frozen=True, slots=True)
class Config:
    app_name: str
    model: str
    model_url: str = field(repr=False)
    twitter_api_key: str = field(repr=False)
    twitter_api_secret: str = field(repr=False)
    twitter_access_token: str = field(repr=False)
    twitter_access_secret: str = field(repr=False)
    twitter_api: str = "https://api.x.com/2"
    twitter_base_url: str = "https://twitter.com/i/web/status/"
    tweets_file: Path = Path("./tweets.jsonl")
    max_generate_retries: int = 12
    max_tweet_len: int = 200
    min_tweet_len: int = 20
    allow_apology: float = 0.1
    allow_ball: float = 0.1
    use_topic: float = 0.5
    topic_api: str = "https://trends.google.com/trending/rss?geo=US"
    http_max_retries: int = 10
    http_max_timeout: float = 30.0
    debug: bool = False

    @classmethod
    def from_env(cls):
        params: dict[str, Any] = {}
        for f in fields(cls):
            if raw := os.getenv("KATBOT_" + f.name.upper()):
                if f.name == "debug":
                    raw = raw.lower().strip() in {"true", "yes", "1"}
                elif not isinstance(f.type, str):
                    raw = f.type(raw.strip())
                params[f.name] = raw
        return cls(**params)

    def __post_init__(self):
        for f in fields(self):
            if f.name == "debug":
                continue
            if not (value := getattr(self, f.name, None)):
                raise ValueError(f"bad value for {f.name}: {value}")


load_dotenv()
cfg = Config.from_env()
