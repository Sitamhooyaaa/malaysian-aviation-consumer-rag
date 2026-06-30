import pytest

from aviation_rag.retry import (
    call_with_retry,
)


def test_call_with_retry_recovers_from_transient_error():
    state = {"attempts": 0}
    waits = []

    def operation():
        state["attempts"] += 1

        if state["attempts"] < 3:
            raise RuntimeError(
                "503 UNAVAILABLE"
            )

        return "success"

    result = call_with_retry(
        operation=operation,
        max_attempts=4,
        base_wait_seconds=1,
        sleep_function=waits.append,
    )

    assert result == "success"
    assert state["attempts"] == 3
    assert waits == [1, 2]


def test_call_with_retry_stops_for_daily_quota():
    state = {"attempts": 0}
    waits = []

    def operation():
        state["attempts"] += 1
        raise RuntimeError(
            "429 GenerateRequestsPerDay "
            "free_tier_requests"
        )

    with pytest.raises(
        RuntimeError,
        match="GenerateRequestsPerDay",
    ):
        call_with_retry(
            operation=operation,
            sleep_function=waits.append,
        )

    assert state["attempts"] == 1
    assert waits == []


def test_call_with_retry_stops_for_permanent_error():
    state = {"attempts": 0}

    def operation():
        state["attempts"] += 1
        raise ValueError(
            "Invalid configuration"
        )

    with pytest.raises(
        ValueError,
        match="Invalid configuration",
    ):
        call_with_retry(
            operation=operation,
            sleep_function=lambda _: None,
        )

    assert state["attempts"] == 1


def test_call_with_retry_stops_after_max_attempts():
    waits = []

    def operation():
        raise RuntimeError(
            "503 UNAVAILABLE"
        )

    with pytest.raises(
        RuntimeError,
        match="503",
    ):
        call_with_retry(
            operation=operation,
            max_attempts=2,
            base_wait_seconds=1,
            sleep_function=waits.append,
        )

    assert waits == [1]


@pytest.mark.parametrize(
    ("max_attempts", "base_wait_seconds"),
    [
        (0, 1),
        (1, -1),
    ],
)
def test_call_with_retry_rejects_invalid_settings(
    max_attempts,
    base_wait_seconds,
):
    with pytest.raises(ValueError):
        call_with_retry(
            operation=lambda: "unused",
            max_attempts=max_attempts,
            base_wait_seconds=(
                base_wait_seconds
            ),
        )