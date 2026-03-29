from __future__ import annotations

import time
from functools import update_wrapper
from typing import Any, Callable
from weakref import WeakKeyDictionary


class _BoundTimed:
    __slots__ = ("_fn", "_inst", "_last_t", "_ema_dt", "_alpha")

    def __init__(self, fn: Callable[..., Any], inst: Any, *, alpha: float) -> None:
        self._fn = fn
        self._inst = inst
        self._last_t: float | None = None
        self._ema_dt: float | None = None
        self._alpha = alpha

    @property
    def freq(self) -> float:
        if self._ema_dt is None or self._ema_dt <= 0.0:
            return 0.0
        return 1.0 / self._ema_dt

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        now = time.monotonic()
        if self._last_t is not None:
            dt = now - self._last_t
            if self._ema_dt is None:
                self._ema_dt = dt
            else:
                self._ema_dt = (1.0 - self._alpha) * self._ema_dt + self._alpha * dt
        self._last_t = now
        return self._fn(self._inst, *args, **kwargs)


class _TimedMethod:
    """Descriptor: instance access returns a callable with a `.freq` property (Hz)."""

    def __init__(self, fn: Callable[..., Any], *, alpha: float = 0.2) -> None:
        self._fn = fn
        self._alpha = alpha
        self._bound: WeakKeyDictionary[Any, _BoundTimed] = WeakKeyDictionary()
        update_wrapper(self, fn)

    def __get__(self, instance: Any, owner: type | None) -> _BoundTimed | _TimedMethod:
        if instance is None:
            return self
        if instance not in self._bound:
            self._bound[instance] = _BoundTimed(self._fn, instance, alpha=self._alpha)
        return self._bound[instance]


def timer_decorator(
    fn: Callable[..., Any] | None = None, *, alpha: float = 0.2
) -> _TimedMethod | Callable[[Callable[..., Any]], _TimedMethod]:
    """Track callback rate with an EMA of inter-arrival times. Use ``self.Handler.freq`` (Hz)."""

    if fn is None:
        return lambda f: _TimedMethod(f, alpha=alpha)
    return _TimedMethod(fn, alpha=alpha)

