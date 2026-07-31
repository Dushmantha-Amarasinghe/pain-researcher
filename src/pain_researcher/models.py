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


class Platform(str, Enum):
    """Which data source a Thread/Comment/Evidence item came from.

    Adding a 4th source later means adding one value here and one
    provider — nothing downstream (prefilter, digest builder, scoring)
    needs to change, since they all operate on the generic Thread/
    Comment/Evidence shapes below, not per-platform types.
    """

    REDDIT = "reddit"
    HACKERNEWS = "hackernews"
    STACKEXCHANGE = "stackexchange"
    WEBSEARCH = "websearch"


class Comment(BaseModel):
    id: str
    author: Optional[str] = None
    body: str
    score: int = 0
    created_utc: float
    permalink: str


class Thread(BaseModel):
    """A submission/story/question and its top comments/answers, in a
    shape common across every source provider.

    `community` generalizes "subreddit" to whatever the source's grouping
    concept is: a subreddit name, the literal string "hackernews" (HN has
    no sub-communities), or a Stack Exchange site slug like "money".
    """

    id: str
    platform: Platform
    community: str
    title: str
    body: str = ""
    author: Optional[str] = None
    score: int = 0
    num_comments: int = 0
    created_utc: float
    permalink: str
    flair: Optional[str] = None  # Reddit-only; None elsewhere
    comments: list[Comment] = Field(default_factory=list)


class ContentKind(str, Enum):
    """Generic across sources: THREAD = submission/story/question,
    COMMENT = comment/answer/reply."""

    THREAD = "thread"
    COMMENT = "comment"


class Evidence(BaseModel):
    """One concrete, traceable data point backing a pain-point candidate."""

    content_kind: ContentKind
    platform: Platform
    community: str
    thread_id: str
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
        """Distinct thread/story/question IDs referenced by this
        candidate's evidence.

        Reads `Evidence.thread_id` directly rather than parsing it back
        out of a permalink URL — thread grouping is known at the moment
        each Evidence is constructed (the provider/digest-builder already
        has the parent thread's id in hand), and every source shapes its
        URLs differently (Reddit's `/comments/{id}/`, HN's flat
        `item?id=`, Stack Exchange's `/questions/{id}/`), so deriving it
        from the URL string would need one fragile parser per platform.
        """
        return {e.thread_id for e in self.evidence}

    @property
    def communities(self) -> set[str]:
        return {e.community for e in self.evidence}

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
