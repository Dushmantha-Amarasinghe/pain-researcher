"""Static, config-only usage projection — no network, no graph execution.

Distinct from the per-node `dry_run` flag in graph.py, which projects
cost using *real* harvested data once you actually have target
subreddits: this estimates a worst-case ceiling purely from
settings.yaml. It's what actually answers "if I retune these
thresholds, does projected spend go down" without needing Reddit or
Google credentials at all — clustering and judging are downstream of
real extraction output, which doesn't exist until you run a real (or
credentialed dry-run) pass, so a pure config-based estimate is the only
way to project their cost in advance.

Run directly: `python -m pain_researcher.estimate`
"""

from __future__ import annotations

import json

from pain_researcher.config import PainResearcherSettings, get_settings
from pain_researcher.quota import estimate_tokens


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def estimate_worst_case_usage(settings: PainResearcherSettings) -> dict:
    cb = settings.content_budget
    disc = settings.discovery
    gating = settings.candidate_gating

    # -- discovery (role: cheap) — small, fixed-shape prompts --
    if disc.mode == "autonomous":
        discovery_calls = 1 + disc.max_niches_per_run  # 1 niche-proposal + N subreddit-proposals
    elif disc.mode == "seed":
        discovery_calls = 1
    else:  # watchlist: verification is free PRAW calls, no LLM
        discovery_calls = 0
    discovery_input_tokens = discovery_calls * 400

    # -- extraction (role: cheap) — driven by real content-budget caps --
    max_threads = disc.max_subreddits_total * cb.max_threads_per_subreddit
    threads_per_call = cb.max_threads_per_extraction_call
    extraction_calls = max(1, _ceil_div(max_threads, threads_per_call))
    per_thread_chars = (
        150 + cb.max_chars_per_source + cb.max_comments_per_thread * (60 + cb.max_chars_per_comment)
    )
    extraction_tokens_per_call = estimate_tokens("x" * (per_thread_chars * threads_per_call))
    extraction_input_tokens = extraction_calls * extraction_tokens_per_call

    # -- clustering (role: cheap) — worst-case proxy: one raw candidate per
    # surviving thread (real extraction usually yields fewer, since not
    # every thread has an extractable pain point) --
    assumed_raw_candidates = max_threads
    cluster_batch = threads_per_call * 2
    cluster_calls = max(1, _ceil_div(assumed_raw_candidates, cluster_batch))
    cluster_tokens_per_call = estimate_tokens("x" * (cluster_batch * 120))
    cluster_input_tokens = cluster_calls * cluster_tokens_per_call

    # -- judging (role: judge) — exactly max_candidates_to_validate calls,
    # the real hard cap on this step regardless of how much data comes in --
    judge_calls = gating.max_candidates_to_validate
    representative_evidence_chars = 15 * 150  # export.py's brief also caps at 15 evidence lines
    representative_competitor_chars = cb.max_competitor_pages * 250
    judge_tokens_per_call = estimate_tokens(
        "x" * (representative_evidence_chars + representative_competitor_chars + 500)
    )
    judge_input_tokens = judge_calls * judge_tokens_per_call

    cheap_profile = settings.model_for_role("cheap")
    judge_profile = settings.model_for_role("judge")

    return {
        "discovery": {"calls": discovery_calls, "input_tokens": discovery_input_tokens},
        "extraction": {"calls": extraction_calls, "input_tokens": extraction_input_tokens},
        "clustering": {"calls": cluster_calls, "input_tokens": cluster_input_tokens},
        "judging": {"calls": judge_calls, "input_tokens": judge_input_tokens},
        "cheap_model": {
            "role_maps_to": settings.roles.cheap,
            "total_calls": discovery_calls + extraction_calls + cluster_calls,
            "total_input_tokens": discovery_input_tokens + extraction_input_tokens + cluster_input_tokens,
            "rpd_limit": int(cheap_profile.effective_rpd),
        },
        "judge_model": {
            "role_maps_to": settings.roles.judge,
            "total_calls": judge_calls,
            "total_input_tokens": judge_input_tokens,
            "rpd_limit": int(judge_profile.effective_rpd),
        },
    }


if __name__ == "__main__":
    settings = get_settings()
    print(json.dumps(estimate_worst_case_usage(settings), indent=2))
