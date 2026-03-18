"""Tests for retry logic."""
import pytest

from llmwire.retry import retry_with_backoff


class TestRetryWithBackoff:
    async def test_succeeds_first_try(self) -> None:
        call_count = 0

        async def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(succeed, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 1

    async def test_retries_on_failure_then_succeeds(self) -> None:
        call_count = 0

        async def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        result = await retry_with_backoff(fail_twice, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count == 3

    async def test_raises_after_max_retries(self) -> None:
        async def always_fail() -> str:
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    async def test_no_retry_on_non_retryable(self) -> None:
        call_count = 0

        async def value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad input")

        with pytest.raises(ValueError):
            await retry_with_backoff(
                value_error,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
            )
        assert call_count == 1
