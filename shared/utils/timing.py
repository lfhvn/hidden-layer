"""
Timing utilities for profiling and benchmarking.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """Context manager and utility for timing code execution.

    Usage:
        # As context manager
        with Timer() as t:
            # code to time
        print(f"Elapsed: {t.elapsed:.2f}s")

        # Manual usage
        timer = Timer()
        timer.start()
        # code to time
        timer.stop()
        print(f"Elapsed: {timer.elapsed:.2f}s")

        # Lap times
        timer = Timer()
        timer.start()
        # first section
        timer.lap("section1")
        # second section
        timer.lap("section2")
        timer.stop()
        print(timer.laps)  # {"section1": 0.5, "section2": 0.3}
    """

    def __init__(self) -> None:
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._lap_start: float | None = None
        self._laps: dict[str, float] = {}

    def start(self) -> Timer:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._lap_start = self._start_time
        return self

    def stop(self) -> Timer:
        """Stop the timer."""
        self._end_time = time.perf_counter()
        return self

    def lap(self, name: str) -> float:
        """Record a lap time.

        Args:
            name: Name for this lap

        Returns:
            Time elapsed since last lap (or start)
        """
        now = time.perf_counter()
        if self._lap_start is None:
            raise RuntimeError("Timer not started")
        lap_time = now - self._lap_start
        self._laps[name] = lap_time
        self._lap_start = now
        return lap_time

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._end_time or time.perf_counter()
        return end - self._start_time

    @property
    def laps(self) -> dict[str, float]:
        """Get all recorded lap times."""
        return self._laps.copy()

    def __enter__(self) -> Timer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.4f}s)"


def timed(func: F) -> F:
    """Decorator to time function execution.

    Adds `_timing` attribute to the result if it's a dict,
    otherwise prints timing to stdout.

    Usage:
        @timed
        def my_function():
            # code to time
            return result
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer() as t:
            result = func(*args, **kwargs)

        # Try to add timing to result if it's a dict
        if isinstance(result, dict):
            result["_timing"] = {"elapsed_s": t.elapsed}
        else:
            print(f"{func.__name__} took {t.elapsed:.4f}s")

        return result

    return wrapper  # type: ignore[return-value]


@contextmanager
def timing_context(name: str = "Operation") -> Generator[Timer, None, None]:
    """Context manager that prints timing on exit.

    Usage:
        with timing_context("Data loading"):
            # code to time
        # Prints: "Data loading took 1.23s"
    """
    timer = Timer()
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
        print(f"{name} took {timer.elapsed:.2f}s")
