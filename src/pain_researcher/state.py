"""LangGraph state schemas.

Two separate state graphs are involved:

- `ResearchState` — the main pipeline: discovery through export.
- `CandidateValidationState` — a per-candidate subgraph (corroborate ->
  competitor_scan -> judge) invoked once per surviving candidate via the
  `Send` API. Making this a genuinely separate compiled subgraph (rather
  than nodes sharing the main graph's state) is what keeps concurrent
  candidate validations from clobbering each other's working fields —
  only reducer-annotated fields (`scored_pitches`, `usage_log`) flow back
  into the parent state when each branch finishes.
"""

import operator
from dataclasses import dataclass, field
from typing import Optional

from typing_extensions import Annotated

from pain_researcher.models import (
    Competitor,
    JudgeSignals,
    PainPointCandidate,
    ScoredPitch,
    Thread,
)


@dataclass(kw_only=True)
class UsageRecord:
    """One LLM call's accounting — the raw material for the per-run usage report."""

    node: str
    model_key: str
    input_tokens: int
    output_tokens: int
    timestamp: float


# --------------------------------------------------------------------------
# Main pipeline state
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class ResearchState:
    run_id: str = field(default="")
    discovery_mode: str = field(default="autonomous")
    seed_niche: Optional[str] = field(default=None)
    subreddit_watchlist: list[str] = field(default_factory=list)
    dry_run: bool = field(default=False)

    niches: list[str] = field(default_factory=list)
    target_subreddits: list[str] = field(default_factory=list)

    threads: Annotated[list[Thread], operator.add] = field(default_factory=list)
    raw_candidates: Annotated[list[PainPointCandidate], operator.add] = field(
        default_factory=list
    )
    candidates: list[PainPointCandidate] = field(default_factory=list)

    scored_pitches: Annotated[list[ScoredPitch], operator.add] = field(
        default_factory=list
    )
    ranked_pitches: list[ScoredPitch] = field(default_factory=list)

    usage_log: Annotated[list[UsageRecord], operator.add] = field(default_factory=list)


@dataclass(kw_only=True)
class ResearchStateInput:
    discovery_mode: Optional[str] = field(default=None)
    seed_niche: Optional[str] = field(default=None)
    subreddit_watchlist: list[str] = field(default_factory=list)
    dry_run: bool = field(default=False)


@dataclass(kw_only=True)
class ResearchStateOutput:
    ranked_pitches: list[ScoredPitch] = field(default_factory=list)
    usage_log: list[UsageRecord] = field(default_factory=list)


# --------------------------------------------------------------------------
# Per-candidate validation subgraph (fanned out via Send, one per candidate)
# --------------------------------------------------------------------------


@dataclass(kw_only=True)
class CandidateValidationState:
    candidate: PainPointCandidate
    competitors: list[Competitor] = field(default_factory=list)
    judge_signals: Optional[JudgeSignals] = field(default=None)
    dry_run: bool = field(default=False)
    usage_log: Annotated[list[UsageRecord], operator.add] = field(default_factory=list)


@dataclass(kw_only=True)
class CandidateValidationOutput:
    scored_pitches: list[ScoredPitch] = field(default_factory=list)
    usage_log: list[UsageRecord] = field(default_factory=list)
