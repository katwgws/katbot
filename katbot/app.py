# ⋅⋆•°☙ katbot/app.py ❧°•⋆⋅

import random
from logging import getLogger

from .config import cfg
from .tweet import Tweet, load_tweets, post_tweet, save_tweets

log = getLogger(__name__)


def post_one() -> None:
    tweets = load_tweets()
    queue = [t for t in tweets if not t.url]

    if queue:
        log.info("Found %d unposted tweets", len(queue))
        tweet = random.choice(queue)
        log.info("Selected tweet: %s", tweet)
        tweets.remove(tweet)

    else:
        log.info("No queued tweets, generating ...")
        tweet = Tweet.generate()

    tweet = post_tweet(tweet) if not cfg.debug else tweet
    tweets.append(tweet)
    save_tweets(tweets)


def generate(n: int) -> None:
    tweets = load_tweets()
    for _ in range(n):
        t = Tweet.generate()
        tweets.append(t)
    save_tweets(tweets)
