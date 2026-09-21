"""
Fetch the delayed SPX option chain from Cboe.

This module handles HTTP only: requesting the raw JSON, retrying
transient failures, and recording when the fetch happened. Parsing
option symbols and building a DataFrame happen elsewhere.
"""

import logging
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"

# Cboe's CDN may reject requests that carry no browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (iv-surface research project)"}

TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2  # waits 2, 4, 8 seconds between attempts

# Transient failures worth retrying. Anything else (403, 404, ...) means
# the request itself is wrong, and repeating it will not help.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Raised when the chain cannot be fetched or is malformed."""


def fetch_raw_chain(url=CBOE_URL):
    """Fetch the raw option chain JSON.

    Returns (payload, fetched_at) where payload is the parsed JSON as a
    dict and fetched_at is a timezone-aware UTC datetime recorded when
    the response arrived.

    Raises FetchError on a non-retryable HTTP status, after exhausting
    retries, or if the response is not the expected JSON structure.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
        except (requests.ConnectionError, requests.Timeout) as exc:
            reason = f"network error: {exc}"
        else:
            if response.status_code == 200:
                fetched_at = datetime.now(timezone.utc)
                return _validate(response), fetched_at
            if response.status_code not in RETRYABLE_STATUS:
                raise FetchError(
                    f"HTTP {response.status_code} from {url}; not retrying"
                )
            reason = f"HTTP {response.status_code}"

        if attempt == MAX_ATTEMPTS:
            raise FetchError(
                f"Gave up after {MAX_ATTEMPTS} attempts; last error: {reason}"
            )
        wait = BACKOFF_BASE_SECONDS ** attempt
        logger.warning(
            "Attempt %d/%d failed (%s); retrying in %ds",
            attempt, MAX_ATTEMPTS, reason, wait,
        )
        time.sleep(wait)


def _validate(response):
    """Parse the body as JSON and check it has the structure we rely on."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise FetchError(f"Response was not valid JSON: {exc}") from exc

    try:
        options = payload["data"]["options"]
    except (KeyError, TypeError) as exc:
        raise FetchError("JSON is missing data.options; format may have changed") from exc

    if not options:
        raise FetchError("data.options is empty")
    return payload


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    payload, fetched_at = fetch_raw_chain()
    data = payload["data"]
    print(f"Fetched at (UTC):      {fetched_at.isoformat()}")
    print(f"Cboe timestamp:        {payload['timestamp']}")
    print(f"Underlying last trade: {data['last_trade_time']}  price {data['current_price']}")
    print(f"Option records:        {len(data['options'])}")