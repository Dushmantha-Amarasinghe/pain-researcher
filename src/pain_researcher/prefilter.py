"""Deterministic, pre-LLM filtering.

The single biggest token saver in the pipeline: every thread that
survives `filter_threads` is one the LLM must actually read, and every
candidate that survives `gate_candidates` is one that spends judge-model
(31B) quota. All thresholds and phrase lists come from
`config.PrefilterConfig` / `config.CandidateGatingConfig` — tune
aggressiveness in settings.yaml, not here.
"""

from __future__ import annotations

import re
import time
from typing import Optional

from pain_researcher.config import CandidateGatingConfig, PrefilterConfig
from pain_researcher.models import Comment, PainPointCandidate, Thread


def _compile_phrase_pattern(phrases: list[str]) -> Optional[re.Pattern]:
    if not phrases:
        return None
    return re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)


class Prefilter:
    def __init__(self, config: PrefilterConfig):
        self._config = config
        self._complaint_re = _compile_phrase_pattern(config.complaint_phrases)
        self._wtp_re = _compile_phrase_pattern(config.wtp_phrases)
        self._drop_flairs = {f.lower() for f in config.drop_flairs}

    def matches_complaint(self, text: str) -> bool:
        return bool(text and self._complaint_re and self._complaint_re.search(text))

    def matches_wtp(self, text: str) -> bool:
        return bool(text and self._wtp_re and self._wtp_re.search(text))

    def _passes_thresholds(self, thread: Thread) -> bool:
        cfg = self._config
        if thread.score < cfg.min_upvotes:
            return False
        if thread.num_comments < cfg.min_comments:
            return False
        age_days = (time.time() - thread.created_utc) / 86400
        if age_days > cfg.max_age_days:
            return False
        if thread.flair and thread.flair.lower() in self._drop_flairs:
            return False
        return True

    def _has_complaint_signal(self, thread: Thread) -> bool:
        """A thread is worth spending extraction tokens on only if
        something in it actually reads like a complaint — high engagement
        alone isn't enough; a wildly upvoted joke thread has no pain
        point to extract, and would just burn TPM for nothing.
        """
        if self.matches_complaint(thread.title) or self.matches_complaint(thread.body):
            return True
        return any(self.matches_complaint(c.body) for c in thread.comments)

    def _reprioritize_comments(self, thread: Thread) -> Thread:
        """Sort comments so complaint/WTP-bearing ones lead, ahead of pure
        score — a low-score "I'd pay for this" reply matters more to
        extraction than a high-score joke reply, and content_budget's
        per-thread comment cap means order determines what the LLM sees.
        """

        def relevance(c: Comment) -> tuple[int, int]:
            signal = int(self.matches_complaint(c.body)) + int(self.matches_wtp(c.body))
            return (signal, c.score)

        ranked = sorted(thread.comments, key=relevance, reverse=True)
        return thread.model_copy(update={"comments": ranked})

    def filter_threads(self, threads: list[Thread]) -> list[Thread]:
        """Threshold + signal + dedupe pass. Order of checks matters:
        cheapest checks (dedupe, thresholds) run before the regex scan.
        """
        seen_ids: set[str] = set()
        kept: list[Thread] = []
        for thread in threads:
            if thread.id in seen_ids:
                continue
            seen_ids.add(thread.id)
            if not self._passes_thresholds(thread):
                continue
            if not self._has_complaint_signal(thread):
                continue
            kept.append(self._reprioritize_comments(thread))
        return kept


def gate_candidates(
    candidates: list[PainPointCandidate], gating: CandidateGatingConfig
) -> list[PainPointCandidate]:
    """Drop candidates below the evidence floor, then cap how many proceed
    to expensive (31B) validation.

    `max_candidates_to_validate` is the single biggest lever on judge-model
    quota spend in the whole system — raising or lowering it in
    settings.yaml directly trades validation depth against daily budget.
    """
    survivors = [
        c
        for c in candidates
        if len(c.distinct_authors) >= gating.min_distinct_authors
        and len(c.distinct_threads) >= gating.min_distinct_threads
    ]
    survivors.sort(key=lambda c: (len(c.distinct_authors), c.total_engagement), reverse=True)
    return survivors[: gating.max_candidates_to_validate]
