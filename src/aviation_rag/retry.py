"""Retry utilities for model-provider calls."""

from collections.abc import Callable
from time import sleep as default_sleep
from typing import TypeVar


ResultType = TypeVar("ResultType")


TRANSIENT_ERROR_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "unavailable",
    "resource_exhausted",
)

DAILY_QUOTA_MARKERS = (
    "generaterequestsperday",
    "requests per day",
    "free_tier_requests",
)


def call_with_retry(
    operation: Callable[[], ResultType],
    max_attempts: int = 4,
    base_wait_seconds: float = 15,
    sleep_function: Callable[
        [float],
        None,
    ] = default_sleep,
) -> ResultType:
    """Retry temporary provider failures."""

    if max_attempts <= 0:
        raise ValueError(
            "max_attempts must be positive"
        )

    if base_wait_seconds < 0:
        raise ValueError(
            "base_wait_seconds cannot be negative"
        )

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            return operation()

        except Exception as error:
            error_text = str(error).lower()

            daily_quota_exhausted = any(
                marker in error_text
                for marker
                in DAILY_QUOTA_MARKERS
            )

            if daily_quota_exhausted:
                raise

            is_transient = any(
                marker in error_text
                for marker
                in TRANSIENT_ERROR_MARKERS
            )

            if (
                not is_transient
                or attempt == max_attempts
            ):
                raise

            wait_seconds = (
                base_wait_seconds
                * 2 ** (attempt - 1)
            )

            sleep_function(wait_seconds)

    raise RuntimeError(
        "Retry loop ended unexpectedly"
    )