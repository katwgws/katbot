# ⋅⋆•°☙ katbot/http.py ❧°•⋆⋅

from typing import Any, Final, Literal

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import cfg

__all__: Final = ["request"]


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
    wait=wait_random_exponential(max=cfg.http_max_timeout),
    stop=stop_after_attempt(cfg.http_max_retries),
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
        headers={"User-Agent": f"{cfg.app_name}/1.0 (+https://x.com)"}
        | request_kwargs.pop("headers", {}),
        timeout=cfg.http_max_timeout,
        **request_kwargs,
    )
    r.raise_for_status()
    return r
