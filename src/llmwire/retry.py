"""Retry logic with exponential backoff."""
from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


async def retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError, OSError),
) -> T:
    """Execute an async function with exponential backoff retry.

    Args:
        fn: The async callable to execute.
        max_retries: Maximum number of attempts before raising.
        base_delay: Base delay in seconds for the first backoff interval.
        retryable_exceptions: Exception types that trigger a retry.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        Exception: The last retryable exception after all attempts are exhausted,
            or any non-retryable exception immediately.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except retryable_exceptions as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, base_delay * 0.1)
                await asyncio.sleep(delay)
        except Exception:
            raise
    assert last_exception is not None
    raise last_exception
