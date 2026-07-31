"""Domain models for the pain-point researcher.

These are the structured records that replace the original repo's
single markdown-blob output — everything here is meant to be sorted,
filtered, and exported as JSON/CSV, not just read as prose.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class RedditComment(BaseModel):
    id: str
    author: Optional[str] = None
    body: str
    score: int = 0
    created_utc: float
    permalink: str


class RedditThread(BaseModel):
    id: str
    subreddit: str
    title: str
    selftext: str = ""
    author: Optional[str] = None
    score: int = 0
    num_comments: int = 0
    created_utc: float
    permalink: str
    flair: Optional[str] = None
    comments: list[RedditComment] = Field(default_factory=list)


class EvidenceSource(str, Enum):
    REDDIT_THREAD = "reddit_thread"
    REDDIT_COMMENT = "reddit_comment"


class Evidence(BaseModel):
    """One concrete, traceable data point backing a pain-point candidate."""

    source_type: EvidenceSource
    subreddit: str
    author: Optional[str] = None
    score: int = 0
    excerpt: str
    permalink: str
    created_utc: float

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(self.created_utc, tz=timezone.utc)


class PainPointCandidate(BaseModel):
    """A pain point extracted from Reddit, before or after validation.

    `evidence` accumulates across extraction and corroboration; distinct
    author/thread counts are derived from it rather than tracked
    separately, so they can never drift out of sync with the evidence list.
    """

    id: str = Field(default_factory=_new_id)
    niche: Optional[str] = None
    title: str
    description: str
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def distinct_authors(self) -> set[str]:
        return {e.author for e in self.evidence if e.author}

    @property
    def distinct_threads(self) -> set[str]:
        """Reddit thread IDs referenced by this candidate's evidence.

        Parses the `/r/{sub}/comments/{thread_id}/...` permalink shape
        rather than string-splitting on a fixed prefix, since a naive
        split on "/comment" also matches the plural "/comments/" segment
        and silently collapses distinct threads together.
        """
        threads: set[str] = set()
        for e in self.evidence:
            parts = [p for p in e.permalink.split("/") if p]
            if "comments" in parts:
                idx = parts.index("comments")
                if idx + 1 < len(parts):
                    threads.add(parts[idx + 1])
                    continue
            threads.add(e.permalink)
        return threads

    @property
    def subreddits(self) -> set[str]:
        return {e.subreddit for e in self.evidence}

    @property
    def total_engagement(self) -> int:
        return sum(e.score for e in self.evidence)


class CompetitorStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class Competitor(BaseModel):
    name: str
    url: str
    description: str = ""
    strength: CompetitorStrength = CompetitorStrength.MODERATE
    criticized_in_evidence: bool = False


class SolutionGap(str, Enum):
    NONE_FOUND = "none_found"       # no competitors found — could be no market, could be untapped
    WEAK_COMPETITORS = "weak"        # competitors exist but are criticized / clearly inadequate
    STRONG_COMPETITORS = "strong"    # well-regarded competitors already solve this


class JudgeSignals(BaseModel):
    """Structured output from the 31B judge step — signals, not a final score.

    Keeping this as discrete signals (rather than an LLM-assigned 1-10
    score) is what lets scoring.py apply consistent, tunable weights and
    stay comparable across runs and model swaps.
    """

    severity: float = Field(ge=0, le=5, description="0=mild annoyance, 5=blocks work/costs money")
    willingness_to_pay: bool = False
    wtp_evidence: list[str] = Field(default_factory=list)
    solution_gap: SolutionGap = SolutionGap.NONE_FOUND
    buildability: float = Field(ge=0, le=5, description="0=needs a team+years, 5=solo-buildable MVP")
    hard_blockers: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ScoredPitch(BaseModel):
    candidate: PainPointCandidate
    competitors: list[Competitor] = Field(default_factory=list)
    judge_signals: Optional[JudgeSignals] = None
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)
    rank: Optional[int] = None
