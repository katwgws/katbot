from typing import Any, Literal

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()


HTTP_MAX_RETRIES = 3
HTTP_MAX_TIMEOUT = 240


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
    headers: dict[str, Any] | None = None,
    **request_kwargs: Any,
) -> requests.Response:
    r = requests.request(
        method,
        url,
        headers=headers or {},
        timeout=HTTP_MAX_TIMEOUT,
        **request_kwargs,
    )
    r.raise_for_status()
    return r
