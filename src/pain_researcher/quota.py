"""Per-model quota enforcement.

TPM (input tokens/minute) is the binding constraint for Gemma 4 on the
AI Studio free tier, not RPM — see settings.yaml's `target_rpm` comment.
This module enforces three things, in order, before every LLM call:

1. **Daily cap** — hard stop once `effective_rpd` requests have been made
   today (persisted to disk so a restart mid-run doesn't reset it).
2. **Sliding-window RPM/TPM** — a safety net against ever exceeding the
   provider's real per-minute limits.
3. **Pacing** — a minimum inter-call interval derived from `target_rpm`,
   which spreads calls evenly across the minute instead of bursting up to
   the window ceiling and then stalling. This is what makes "take time
   and divide the workload" actually happen rather than just being caught
   by the safety net after the fact.

One `QuotaLimiter` per model, shared process-wide via `QuotaRegistry` so
concurrent candidate-validation branches (fanned out via LangGraph's
Send) are paced against the same counters rather than each thinking it
has the full budget to itself.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, TypeVar

from pain_researcher.config import ModelProfile, RetryConfig

T = TypeVar("T")

DEFAULT_QUOTA_DIR = Path(os.environ.get("PAIN_RESEARCHER_QUOTA_DIR", ".pain_researcher_quota"))


def estimate_tokens(text: str) -> int:
    """Rough token estimate for pre-flight budgeting — not exact billing.

    Tries tiktoken's cl100k_base encoding as a reasonable stand-in (Gemma
    doesn't have a public tiktoken encoding); falls back to the chars/4
    heuristic already used in `ollama_deep_researcher/utils.py`.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class QuotaExhausted(RuntimeError):
    """Raised when a model's daily request budget is used up.

    The graph should treat this as "pause / resume later", not a crash —
    see the SQLite checkpointer in graph.py, which lets a run spanning
    multiple days resume rather than restart.
    """


@dataclass
class _DailyCounters:
    date: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class QuotaLimiter:
    """Enforces RPM/TPM/RPD for one model, with disk-persisted daily counters."""

    def __init__(
        self, model_key: str, profile: ModelProfile, quota_dir: Optional[Path] = None
    ):
        self.model_key = model_key
        self.profile = profile
        self._quota_dir = Path(quota_dir) if quota_dir else DEFAULT_QUOTA_DIR
        self._quota_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._window: deque[tuple[float, int]] = deque()  # (monotonic ts, tokens), last 60s
        self._last_call_at: float = 0.0
        self._daily = self._load_daily()

    # -- persistence ------------------------------------------------------

    def _daily_path(self) -> Path:
        return self._quota_dir / f"{self.model_key}.json"

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_daily(self) -> _DailyCounters:
        path = self._daily_path()
        today = self._today()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("date") == today:
                    return _DailyCounters(**data)
            except Exception:
                pass
        return _DailyCounters(date=today)

    def _save_daily(self) -> None:
        self._daily_path().write_text(json.dumps(asdict(self._daily)), encoding="utf-8")

    def _roll_day_if_needed(self) -> None:
        today = self._today()
        if self._daily.date != today:
            self._daily = _DailyCounters(date=today)
            self._save_daily()

    # -- sliding window -----------------------------------------------------

    def _prune_window(self, now: float) -> None:
        while self._window and now - self._window[0][0] > 60:
            self._window.popleft()

    def _window_totals(self, now: float) -> tuple[int, int]:
        self._prune_window(now)
        return len(self._window), sum(t for _, t in self._window)

    # -- public API -----------------------------------------------------

    def reserve(self, estimated_input_tokens: int) -> None:
        """Block until it's safe to call, or raise QuotaExhausted for today.

        Re-checks after every sleep rather than computing one wait and
        sleeping through it, since concurrent branches update shared state
        between iterations. The lock is released during the sleep itself
        so it doesn't block other threads' `record()` calls or daily-cap
        checks while this one waits.
        """
        while True:
            with self._lock:
                self._roll_day_if_needed()
                if self._daily.requests >= self.profile.effective_rpd:
                    raise QuotaExhausted(
                        f"{self.model_key}: daily request budget exhausted "
                        f"({self._daily.requests}/{int(self.profile.effective_rpd)}). "
                        "Resume tomorrow, or raise rpd_limit / safety_margin_pct in settings.yaml."
                    )

                now = time.monotonic()
                count, tokens_in_window = self._window_totals(now)
                min_interval = 60.0 / self.profile.effective_rpm

                wait_pacing = (
                    max(0.0, min_interval - (now - self._last_call_at))
                    if self._last_call_at
                    else 0.0
                )
                wait_rpm = 0.0
                wait_tpm = 0.0
                if self._window:
                    window_age = now - self._window[0][0]
                    if count + 1 > self.profile.effective_rpm:
                        wait_rpm = 60.0 - window_age
                    if tokens_in_window + estimated_input_tokens > self.profile.effective_tpm:
                        wait_tpm = 60.0 - window_age

                wait_s = max(wait_pacing, wait_rpm, wait_tpm, 0.0)
                if wait_s <= 0:
                    self._last_call_at = time.monotonic()
                    self._window.append((self._last_call_at, estimated_input_tokens))
                    return

            time.sleep(min(wait_s, 5.0))

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Record actual usage after a call completes, persisted to disk."""
        with self._lock:
            self._roll_day_if_needed()
            self._daily.requests += 1
            self._daily.input_tokens += input_tokens
            self._daily.output_tokens += output_tokens
            self._save_daily()

    def usage_snapshot(self) -> dict:
        with self._lock:
            self._roll_day_if_needed()
            count, tokens = self._window_totals(time.monotonic())
            return {
                "model": self.model_key,
                "requests_today": self._daily.requests,
                "rpd_limit": int(self.profile.effective_rpd),
                "input_tokens_today": self._daily.input_tokens,
                "output_tokens_today": self._daily.output_tokens,
                "requests_last_minute": count,
                "tpm_used_last_minute": tokens,
                "tpm_limit": int(self.profile.effective_tpm),
            }


def with_backoff(
    fn: Callable[[], T],
    retry: RetryConfig,
    is_retryable: Callable[[Exception], bool],
) -> T:
    """Exponential backoff with jitter, for provider 429s specifically.

    Distinct from `QuotaLimiter.reserve`: reserve() prevents us from
    *causing* a 429 in the normal case; this handles the provider
    tightening limits out from under us mid-run (which Google did to the
    free Gemma tier in July 2026) or a transient server-side hiccup.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt >= retry.max_retries:
                raise
            delay = min(retry.backoff_max_seconds, retry.backoff_base_seconds * (2**attempt))
            delay *= 0.5 + random.random()  # jitter, avoid thundering herd across branches
            time.sleep(delay)
            attempt += 1


class QuotaRegistry:
    """One QuotaLimiter per model, shared across the process."""

    def __init__(self, quota_dir: Optional[Path] = None):
        self._quota_dir = quota_dir
        self._limiters: dict[str, QuotaLimiter] = {}
        self._lock = threading.Lock()

    def get(self, model_key: str, profile: ModelProfile) -> QuotaLimiter:
        with self._lock:
            if model_key not in self._limiters:
                self._limiters[model_key] = QuotaLimiter(model_key, profile, self._quota_dir)
            return self._limiters[model_key]

    def usage_report(self) -> list[dict]:
        with self._lock:
            return [limiter.usage_snapshot() for limiter in self._limiters.values()]
