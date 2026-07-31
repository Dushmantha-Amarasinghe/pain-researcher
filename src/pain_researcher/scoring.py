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
    Platform,
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
    well-judged mentions just because it's from a bigger community.
    """
    breakdown: dict[str, float] = {
        "distinct_authors": scoring.weight_distinct_authors
        * math.log1p(len(candidate.distinct_authors)),
        "distinct_threads": scoring.weight_distinct_threads
        * math.log1p(len(candidate.distinct_threads)),
        "community_spread": scoring.weight_community_spread
        * math.log1p(len(candidate.communities)),
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

    raw_score = sum(breakdown.values())

    # Websearch-sourced evidence (general search + crawl, no structured
    # API) has no real engagement metric behind it — no upvotes, no
    # comment count, usually no identifiable author — so it's scaled down
    # in proportion to how much of a candidate's evidence relies on it.
    # A candidate corroborated entirely by Reddit/HN/SE is unaffected.
    websearch_count = sum(1 for e in candidate.evidence if e.platform == Platform.WEBSEARCH)
    websearch_fraction = websearch_count / len(candidate.evidence) if candidate.evidence else 0.0
    if websearch_fraction > 0:
        discount = scoring.websearch_trust_discount * websearch_fraction
        final_score = raw_score * (1 - discount)
        breakdown["websearch_trust_discount"] = final_score - raw_score
        return final_score, breakdown

    return raw_score, breakdown


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
