"""Deterministic scoring.

The judge LLM (31B) emits *signals* — severity, a willingness-to-pay
flag, a solution-gap category, buildability, hard blockers. This module
turns those signals plus the evidence-derived engagement metrics into one
number, using weights from `config.ScoringConfig`.

Scoring is deliberately not "ask the LLM to rate this 1-10": an LLM's
holistic rating drifts between runs and can't be decomposed or retuned.
A weighted sum of discrete signals is reproducible, auditable (every
score ships with its `score_breakdown`), and — critically — retunable
from settings.yaml without touching a prompt or re-running the judge.
"""

from __future__ import annotations

import math

from pain_researcher.config import ScoringConfig
from pain_researcher.models import (
    Competitor,
    JudgeSignals,
    PainPointCandidate,
    ScoredPitch,
    SolutionGap,
)


def score_candidate(
    candidate: PainPointCandidate,
    judge_signals: JudgeSignals | None,
    scoring: ScoringConfig,
) -> tuple[float, dict[str, float]]:
    """Returns (total_score, breakdown) so every score is auditable.

    Evidence-volume terms (authors/threads/spread/engagement) use
    log1p — more corroboration should always help, but a candidate with
    200 mentions shouldn't automatically crush one with 15 strong,
    well-judged mentions just because it's from a bigger subreddit.
    """
    breakdown: dict[str, float] = {
        "distinct_authors": scoring.weight_distinct_authors
        * math.log1p(len(candidate.distinct_authors)),
        "distinct_threads": scoring.weight_distinct_threads
        * math.log1p(len(candidate.distinct_threads)),
        "subreddit_spread": scoring.weight_subreddit_spread
        * math.log1p(len(candidate.subreddits)),
        "engagement": scoring.weight_engagement
        * math.log1p(max(0, candidate.total_engagement)),
    }

    if judge_signals is not None:
        breakdown["severity"] = scoring.weight_severity * judge_signals.severity
        breakdown["wtp_signal"] = (
            scoring.weight_wtp_signal if judge_signals.willingness_to_pay else 0.0
        )
        breakdown["buildability"] = scoring.weight_buildability * judge_signals.buildability

        if judge_signals.solution_gap == SolutionGap.STRONG_COMPETITORS:
            breakdown["solution_gap"] = scoring.penalty_strong_competitor
        elif judge_signals.solution_gap == SolutionGap.WEAK_COMPETITORS:
            # Weak/criticized incumbents is a positive signal: it proves
            # a market exists AND that it's underserved.
            breakdown["solution_gap"] = scoring.weight_solution_gap
        else:
            # No competitors found is ambiguous on its own — could be an
            # untapped niche, could be no real demand. Neutral, not a bonus.
            breakdown["solution_gap"] = 0.0

        if judge_signals.hard_blockers:
            # Capped so one candidate with a long, repetitive blocker list
            # doesn't get penalized disproportionately more than one with
            # a single decisive blocker (e.g. "needs a banking license").
            breakdown["hard_blockers"] = scoring.penalty_hard_blocker * min(
                len(judge_signals.hard_blockers), 3
            )
        else:
            breakdown["hard_blockers"] = 0.0

    return sum(breakdown.values()), breakdown


def build_scored_pitch(
    candidate: PainPointCandidate,
    competitors: list[Competitor],
    judge_signals: JudgeSignals | None,
    scoring: ScoringConfig,
) -> ScoredPitch:
    score, breakdown = score_candidate(candidate, judge_signals, scoring)
    return ScoredPitch(
        candidate=candidate,
        competitors=competitors,
        judge_signals=judge_signals,
        score=score,
        score_breakdown=breakdown,
    )


def rank_pitches(pitches: list[ScoredPitch]) -> list[ScoredPitch]:
    """Sort descending by score and assign 1-indexed ranks (in place, returned for chaining)."""
    ranked = sorted(pitches, key=lambda p: p.score, reverse=True)
    for i, pitch in enumerate(ranked, start=1):
        pitch.rank = i
    return ranked
