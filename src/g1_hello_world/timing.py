from __future__ import annotations

import time
from functools import update_wrapper
from typing import Any, Callable
from weakref import WeakKeyDictionary


class _BoundTimed:
    __slots__ = ("_fn", "_inst", "_count_total", "_freq_count_prev", "_freq_t_prev")

    def __init__(self, fn: Callable[..., Any], inst: Any) -> None:
        self._fn = fn
        self._inst = inst
        self._count_total = 0
        self._freq_count_prev: int | None = None
        self._freq_t_prev: float | None = None

    @property
    def freq(self) -> float:
        now = time.monotonic()
        if self._freq_t_prev is None or self._freq_count_prev is None:
            self._freq_t_prev = now
            self._freq_count_prev = self._count_total
            return 0.0
        dt = now - self._freq_t_prev
        dc = self._count_total - self._freq_count_prev
        self._freq_t_prev = now
        self._freq_count_prev = self._count_total
        if dt <= 0.0:
            return 0.0
        return float(dc) / dt

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._count_total += 1
        return self._fn(self._inst, *args, **kwargs)


class _TimedMethod:
    """Descriptor: instance access returns a callable with a `.freq` property (Hz)."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self._bound: WeakKeyDictionary[Any, _BoundTimed] = WeakKeyDictionary()
        update_wrapper(self, fn)

    def __get__(self, instance: Any, owner: type | None) -> _BoundTimed | _TimedMethod:
        if instance is None:
            return self
        if instance not in self._bound:
            self._bound[instance] = _BoundTimed(self._fn, instance)
        return self._bound[instance]


def timer_decorator(
    fn: Callable[..., Any] | None = None
) -> _TimedMethod | Callable[[Callable[..., Any]], _TimedMethod]:
    """Track callback rate as call count divided by elapsed time between ``freq`` queries."""

    if fn is None:
        return lambda f: _TimedMethod(f)
    return _TimedMethod(fn)
