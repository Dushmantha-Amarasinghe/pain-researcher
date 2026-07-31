"""LangGraph pipeline: discovery through export.

Shape (see the plan for the full rationale):

    select_targets -> harvest_threads -> extract_pain_points
      -> cluster_pain_points -> prefilter_candidates
      -> [fan-out, one Send per surviving candidate]
           validate_candidate (subgraph: corroborate -> competitor_scan -> judge)
      -> [fan-in]
      -> score_and_rank -> export_results

Discovery is wide and cheap (role "cheap" / 26B); validation is narrow
and expensive (role "judge" / 31B), and only runs on candidates that
already cleared the evidence floor in `prefilter_candidates` — this
split is what makes the two models' separate quota pools add up to
usable daily throughput instead of one shared bottleneck.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ollama_deep_researcher.utils import duckduckgo_search
from pain_researcher.config import Credentials, RunOptions, RuntimeConfig, get_settings
from pain_researcher.export import export_run, export_usage_report
from pain_researcher.models import (
    Competitor,
    CompetitorStrength,
    ContentKind,
    Evidence,
    JudgeSignals,
    PainPointCandidate,
    Platform,
    Thread,
)
from pain_researcher.prefilter import Prefilter, gate_candidates
from pain_researcher.prompts import (
    CLUSTERING_PROMPT,
    EXTRACTION_PROMPT,
    JUDGE_PROMPT,
    NICHE_PROPOSAL_PROMPT,
    SUBREDDIT_PROPOSAL_PROMPT,
    WEB_RESEARCH_REFLECTION_PROMPT,
    get_current_date,
)
from pain_researcher.providers.crawl import CrawlProvider
from pain_researcher.providers.hackernews import HackerNewsProvider
from pain_researcher.providers.llm import ContentTooLargeError, LLMParseError, LLMRouter
from pain_researcher.providers.reddit import RedditProvider
from pain_researcher.providers.stackexchange import StackExchangeProvider
from pain_researcher.scoring import build_scored_pitch, rank_pitches
from pain_researcher.state import (
    CandidateValidationOutput,
    CandidateValidationState,
    ResearchState,
    ResearchStateInput,
    ResearchStateOutput,
    UsageRecord,
)

# --------------------------------------------------------------------------
# Provider singletons — constructed once per settings path, shared across
# every node call and every concurrent Send branch. This matters beyond
# efficiency: the quota limiter's in-memory pacing state (see quota.py)
# only works if concurrent candidate-validation branches share the same
# QuotaLimiter instance rather than each starting with a fresh one that's
# unaware of the others' recent calls.
# --------------------------------------------------------------------------


@dataclass
class Providers:
    llm: LLMRouter
    crawl: CrawlProvider
    reddit: Optional[RedditProvider]
    hackernews: Optional[HackerNewsProvider]
    stackexchange: Optional[StackExchangeProvider]


@lru_cache(maxsize=4)
def _build_providers(settings_path: str) -> Providers:
    settings = get_settings(settings_path or None)
    credentials = Credentials.from_env()
    runtime = RuntimeConfig(settings=settings, options=RunOptions())
    enabled = settings.sources.enabled

    reddit = None
    if "reddit" in enabled:
        if not credentials.has_reddit():
            raise RuntimeError(
                "sources.enabled includes 'reddit' but REDDIT_CLIENT_ID/SECRET/USER_AGENT "
                "aren't set. Either fill those in .env, or remove 'reddit' from "
                "sources.enabled in settings.yaml to run on hackernews/stackexchange only."
            )
        reddit = RedditProvider(credentials, settings.content_budget, settings.discovery)

    hackernews = HackerNewsProvider(settings.content_budget) if "hackernews" in enabled else None
    stackexchange = (
        StackExchangeProvider(settings.content_budget, credentials.stackexchange_api_key)
        if "stackexchange" in enabled
        else None
    )

    return Providers(
        llm=LLMRouter(runtime, credentials),
        crawl=CrawlProvider(settings.content_budget),
        reddit=reddit,
        hackernews=hackernews,
        stackexchange=stackexchange,
    )


def get_providers(options: RunOptions) -> Providers:
    return _build_providers(options.settings_path or "")


def _run_with_timeout(fn, timeout_s: float, default):
    """Bound an external call that has no reliable timeout of its own.

    Confirmed live: `competitor_scan`'s DuckDuckGo search + Crawl4AI
    browser launch hung the entire pipeline indefinitely with neither
    call bounded. `future.result(timeout=...)` stops *waiting* on a
    genuinely stuck call rather than killing the underlying thread — an
    acceptable tradeoff here, since the goal is keeping the node from
    freezing the whole run, not guaranteeing the stray thread stops.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return default
        except Exception:
            return default


# --------------------------------------------------------------------------
# Digest builders — turn structured data into labeled text for prompts,
# and keep a label -> source-object index so LLM responses only ever need
# to reference a label ("T1_C2", "C3") rather than reproduce a URL,
# author, or quote themselves.
# --------------------------------------------------------------------------


def _digest_threads(threads: list[Thread]) -> tuple[str, dict[str, Evidence]]:
    """Labels threads/comments generically across sources — `{platform}/
    {community}` instead of Reddit's `r/{subreddit}` — so a batch mixing
    Reddit, HN, and Stack Exchange content reads sensibly to the LLM.

    Comment-level Evidence uses `thread_id=t.id` (the *parent* thread's
    id), not the comment's own id — distinct-thread counting depends on
    grouping every comment under its actual parent, not counting each
    comment as its own thread.
    """
    lines: list[str] = []
    index: dict[str, Evidence] = {}
    for i, t in enumerate(threads, start=1):
        label = f"T{i}"
        lines.append(
            f"[{label}] {t.platform.value}/{t.community} "
            f"(score {t.score}, {t.num_comments} comments): {t.title}"
        )
        if t.body:
            lines.append(f"  body: {t.body[:500]}")
        index[label] = Evidence(
            content_kind=ContentKind.THREAD,
            platform=t.platform,
            community=t.community,
            thread_id=t.id,
            author=t.author,
            score=t.score,
            excerpt=t.title,
            permalink=t.permalink,
            created_utc=t.created_utc,
        )
        for j, c in enumerate(t.comments, start=1):
            clabel = f"{label}_C{j}"
            lines.append(f"  [{clabel}] (score {c.score}): {c.body}")
            index[clabel] = Evidence(
                content_kind=ContentKind.COMMENT,
                platform=t.platform,
                community=t.community,
                thread_id=t.id,
                author=c.author,
                score=c.score,
                excerpt=c.body,
                permalink=c.permalink,
                created_utc=c.created_utc,
            )
    return "\n".join(lines), index


def _digest_candidates(
    candidates: list[PainPointCandidate],
) -> tuple[str, dict[str, PainPointCandidate]]:
    lines: list[str] = []
    index: dict[str, PainPointCandidate] = {}
    for i, c in enumerate(candidates, start=1):
        label = f"C{i}"
        lines.append(f"[{label}] {c.title}: {c.description}")
        index[label] = c
    return "\n".join(lines), index


# --------------------------------------------------------------------------
# Main pipeline nodes
# --------------------------------------------------------------------------


def select_targets(state: ResearchState, config: RunnableConfig) -> dict:
    """Resolve discovery_mode into a verified subreddit list.

    Every LLM-proposed subreddit (autonomous/seed modes) is checked via
    `RedditProvider.verify_subreddit` before use — the model will invent
    plausible-sounding names that don't exist, and harvesting against a
    nonexistent subreddit would just waste the step silently.

    Dry-run note: with this flag, autonomous/seed mode's projected spend
    covers only this discovery step itself (no real niches/subreddits
    come back without a real LLM call, so nothing downstream has real
    data to project against). For a full-pipeline worst-case projection
    that doesn't depend on real data at all, use
    `estimate.estimate_worst_case_usage` instead (`python -m
    pain_researcher.estimate`) — it's the tool for "if I retune these
    thresholds, does projected spend drop," including the clustering and
    judge steps this per-node flag can't reach without real upstream
    output. This flag is still useful in watchlist mode, where subreddit
    verification (a free PRAW call) and harvesting run for real even
    during a dry run, giving an accurate projection through extraction.
    """
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)

    if "reddit" not in settings.sources.enabled:
        # Reddit-specific subreddit discovery has nothing to do if Reddit
        # isn't an active source this run (e.g. access still pending
        # approval) — hackernews/stackexchange need no per-run target
        # selection, so this node is a no-op for them.
        return {"niches": [], "target_subreddits": [], "usage_log": []}

    providers = get_providers(options)

    mode = options.discovery_mode or state.discovery_mode or settings.discovery.mode
    seed_niche = options.seed_niche or state.seed_niche or settings.discovery.seed_niche
    watchlist = (
        options.subreddit_watchlist
        or state.subreddit_watchlist
        or settings.discovery.subreddit_watchlist
    )
    dry_run = state.dry_run or options.dry_run

    usage: list[UsageRecord] = []

    def _call(node_name: str, prompt: str) -> dict:
        result = (
            providers.llm.project_call("cheap", node_name, "", prompt)
            if dry_run
            else providers.llm.generate_structured("cheap", node_name, "", prompt)
        )
        usage.append(result.usage)
        return result.data

    niches: list[str] = []
    proposed_subreddits: list[str] = []

    if mode == "watchlist":
        proposed_subreddits = list(watchlist)
    elif mode == "seed" and seed_niche:
        niches = [seed_niche]
        data = _call(
            "select_targets.subreddit_proposal",
            SUBREDDIT_PROPOSAL_PROMPT.format(
                niche=seed_niche, max_subreddits=settings.discovery.max_subreddits_per_niche
            ),
        )
        proposed_subreddits = data.get("subreddits", [])
    elif mode == "autonomous":
        data = _call(
            "select_targets.niche_proposal",
            NICHE_PROPOSAL_PROMPT.format(
                current_date=get_current_date(),
                excluded_niches="none",
                max_niches=settings.discovery.max_niches_per_run,
            ),
        )
        niches = [n.get("niche", "") for n in data.get("niches", []) if n.get("niche")]
        for niche in niches:
            sub_data = _call(
                "select_targets.subreddit_proposal",
                SUBREDDIT_PROPOSAL_PROMPT.format(
                    niche=niche, max_subreddits=settings.discovery.max_subreddits_per_niche
                ),
            )
            proposed_subreddits.extend(sub_data.get("subreddits", []))

    # Dry run with no real LLM output to verify against: fall back to the
    # configured watchlist (if any) purely so downstream nodes have
    # something concrete to project harvest/extraction/judge costs for.
    if dry_run and not proposed_subreddits and watchlist:
        proposed_subreddits = list(watchlist)

    verified: list[str] = []
    seen: set[str] = set()
    for name in proposed_subreddits:
        if len(verified) >= settings.discovery.max_subreddits_total:
            break
        key = name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        info = providers.reddit.verify_subreddit(name)
        if info:
            verified.append(info["name"])

    return {"niches": niches, "target_subreddits": verified, "usage_log": usage}


def _sanitize_search_query(query: str) -> str:
    """Defensive backstop for the reflection prompt's query-format
    instruction — confirmed live that `site:` filters and quoted
    boolean queries reliably return zero results from this search
    backend, so this strips that syntax even if the model doesn't fully
    follow the instruction (LLMs don't always comply with format
    constraints 100% of the time, so the instruction alone isn't enough
    to rely on for something this consequential to run cost).
    """
    query = re.sub(r"site:\S+", " ", query, flags=re.IGNORECASE)
    query = re.sub(r"\b(OR|AND|NOT)\b", " ", query)
    query = query.replace('"', " ").replace("(", " ").replace(")", " ")
    return re.sub(r"\s+", " ", query).strip()


def _run_web_research(niche: str, settings, providers: "Providers") -> tuple[list[Thread], list[UsageRecord]]:
    """Iterative general web search -> crawl -> reflect -> next-query loop.

    Unlike the other three sources, this one has no structured target
    list to fetch from — each cycle decides where to look next based on
    what's already been found, using `web_search.decision_role` (judge
    by default: assessing evidence quality and spotting gaps is real
    judgment, not bulk extraction, matching the cheap/judge split used
    everywhere else in this pipeline).

    Crawled pages become `Thread` objects like any other source, with
    `platform=WEBSEARCH` — they flow through the exact same extract/
    cluster/gate pipeline afterward. No separate extraction path here;
    scoring.py is what actually discounts this source's lack of real
    engagement metrics (see `websearch_trust_discount`).
    """
    import hashlib

    wcfg = settings.web_search
    usage: list[UsageRecord] = []
    seen_urls: set[str] = set()
    threads: list[Thread] = []
    past_queries: list[str] = []
    findings: list[str] = []

    query = niche
    for cycle in range(1, wcfg.max_cycles + 1):
        print(f"[websearch] cycle {cycle}/{wcfg.max_cycles}: searching {query!r}...")
        past_queries.append(query)

        search_results = _run_with_timeout(
            lambda q=query: duckduckgo_search(q, max_results=wcfg.pages_per_search, fetch_full_page=False),
            timeout_s=20,
            default={"results": []},
        )
        new_urls = [
            r["url"]
            for r in search_results.get("results", [])
            if r.get("url") and r["url"] not in seen_urls
        ]
        seen_urls.update(new_urls)

        # Scales with pages_per_search rather than a flat number — a fresh
        # browser launch plus N sequential page loads (each already
        # bounded at 20s via CrawlerRunConfig.page_timeout) can
        # legitimately take longer than a flat 60s once N is more than
        # 2-3. Confirmed live: a flat 60s cut off a real (not hung) crawl
        # of 5 pages before it produced anything, including before it
        # could even log which URLs it was attempting.
        crawl_timeout = 30 + wcfg.pages_per_search * 25
        pages = (
            _run_with_timeout(
                lambda u=new_urls: providers.crawl.fetch_pages(u, limit=wcfg.pages_per_search),
                timeout_s=crawl_timeout,
                default=[],
            )
            if new_urls
            else []
        )
        print(
            f"[websearch] cycle {cycle}/{wcfg.max_cycles}: {len(new_urls)} new link(s), "
            f"{len(pages)} page(s) crawled"
        )

        for page in pages:
            domain = page.url.split("//")[-1].split("/")[0].lower()
            body = page.markdown[: wcfg.max_chars_per_page]
            threads.append(
                Thread(
                    id=hashlib.sha256(page.url.encode()).hexdigest()[:12],
                    platform=Platform.WEBSEARCH,
                    community=domain,
                    title=page.title or page.url,
                    body=body,
                    author=None,
                    score=0,
                    num_comments=0,
                    created_utc=time.time(),
                    permalink=page.url,
                    comments=[],
                )
            )
            findings.append(f"- {page.title or domain}: {body[:200]}")

        if cycle == wcfg.max_cycles:
            break  # last cycle: no need to spend a reflection call deciding a next query

        findings_digest = "\n".join(findings[-10:]) or "Nothing found yet."
        prompt = WEB_RESEARCH_REFLECTION_PROMPT.format(
            niche=niche,
            past_queries="\n".join(f"- {q}" for q in past_queries),
            findings_digest=findings_digest,
        )
        try:
            result = providers.llm.generate_structured(
                wcfg.decision_role, "web_research_reflect", "", prompt
            )
            usage.append(result.usage)
            next_query = _sanitize_search_query(result.data.get("next_query") or "")
            print(f"[websearch] cycle {cycle}/{wcfg.max_cycles} gap: {result.data.get('gap_analysis', '')[:150]}")
        except Exception as e:
            print(f"[websearch] reflection failed, broadening query instead: {e}")
            next_query = ""

        query = next_query or f"{niche} complaints frustration workaround"

    print(f"[websearch] done: {len(threads)} page(s) collected across {wcfg.max_cycles} cycle(s)")
    return threads, usage


def harvest_threads(state: ResearchState, config: RunnableConfig) -> dict:
    """Pull from every enabled source, then deterministic prefilter —
    no LLM involved yet.

    Filtering happens in this same node (not a separate step) so
    `state.threads` never holds anything the extraction node would waste
    tokens reading; this is the single biggest token saver in the pipeline.
    """
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    prefilter = Prefilter(settings.prefilter)

    all_threads: list[Thread] = []
    usage: list[UsageRecord] = []
    # Each per-source call is individually timeout-bounded, not just via
    # the http client's own timeout setting — confirmed live that a
    # library-level timeout can fail to fire on some networks (proxy/AV
    # SSL interception holding a connection open), so a hard
    # thread-based bound is the only reliable guarantee here.

    if providers.reddit is not None:
        print(f"[harvest] reddit: fetching from {len(state.target_subreddits)} subreddit(s)...")
        for sub in state.target_subreddits:
            found = _run_with_timeout(
                lambda sub=sub: providers.reddit.fetch_threads(sub), timeout_s=30, default=[]
            )
            print(f"[harvest] reddit r/{sub}: {len(found)} threads")
            all_threads.extend(found)

    queries = settings.sources.harvest_queries
    query_limit = settings.sources.max_results_per_query

    if providers.hackernews is not None:
        if queries:
            for q in queries:
                print(f"[harvest] hackernews: searching {q!r}...")
                found = _run_with_timeout(
                    lambda q=q: providers.hackernews.search(
                        q, limit=query_limit, max_age_days=settings.prefilter.max_age_days
                    ),
                    timeout_s=60,
                    default=[],
                )
                print(f"[harvest] hackernews {q!r}: {len(found)} threads")
                all_threads.extend(found)
        else:
            print("[harvest] hackernews: fetching Ask HN...")
            found = _run_with_timeout(providers.hackernews.fetch_ask_hn, timeout_s=60, default=[])
            print(f"[harvest] hackernews ask_hn: {len(found)} threads")
            all_threads.extend(found)

    if providers.stackexchange is not None:
        for site in settings.sources.stackexchange_sites:
            if queries:
                for q in queries:
                    print(f"[harvest] stackexchange/{site}: searching {q!r}...")
                    found = _run_with_timeout(
                        lambda q=q, site=site: providers.stackexchange.search(
                            q, site=site, limit=query_limit
                        ),
                        timeout_s=60,
                        default=[],
                    )
                    print(f"[harvest] stackexchange/{site} {q!r}: {len(found)} threads")
                    all_threads.extend(found)
            else:
                print(f"[harvest] stackexchange/{site}: fetching questions...")
                found = _run_with_timeout(
                    lambda site=site: providers.stackexchange.fetch_questions(site),
                    timeout_s=60,
                    default=[],
                )
                print(f"[harvest] stackexchange/{site}: {len(found)} threads")
                all_threads.extend(found)

    if "websearch" in settings.sources.enabled:
        niche = (
            (state.niches[0] if state.niches else None)
            or state.seed_niche
            or settings.web_search.fallback_seed_query
        )
        if niche:
            web_threads, web_usage = _run_web_research(niche, settings, providers)
            all_threads.extend(web_threads)
            usage.extend(web_usage)
        else:
            print(
                "[websearch] skipped: no niche available (not in seed/autonomous mode, and "
                "web_search.fallback_seed_query isn't set in settings.yaml)"
            )

    kept = prefilter.filter_threads(all_threads)
    print(f"[harvest] total harvested: {len(all_threads)}, survived prefilter: {len(kept)}")
    return {"threads": kept, "usage_log": usage}


def extract_pain_points(state: ResearchState, config: RunnableConfig) -> dict:
    """Batched extraction over harvested threads (role: cheap / 26B)."""
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    dry_run = state.dry_run or options.dry_run
    batch_size = settings.content_budget.max_threads_per_extraction_call

    usage: list[UsageRecord] = []
    raw_candidates: list[PainPointCandidate] = []
    num_batches = -(-len(state.threads) // batch_size) if state.threads else 0
    print(f"[extract] {len(state.threads)} threads in {num_batches} batch(es)...")

    for batch_num, i in enumerate(range(0, len(state.threads), batch_size), start=1):
        batch = state.threads[i : i + batch_size]
        digest, evidence_index = _digest_threads(batch)
        prompt = EXTRACTION_PROMPT.format(threads_digest=digest)

        try:
            if dry_run:
                result = providers.llm.project_call("cheap", "extract_pain_points", "", prompt)
                usage.append(result.usage)
                continue
            result = providers.llm.generate_structured("cheap", "extract_pain_points", "", prompt)
        except (LLMParseError, ContentTooLargeError) as e:
            print(f"[extract] batch {batch_num}/{num_batches} skipped: {e}")
            continue
        usage.append(result.usage)

        found_this_batch = 0
        for pp in result.data.get("pain_points", []):
            refs = pp.get("evidence_refs", [])
            evidence = [evidence_index[r] for r in refs if r in evidence_index]
            if not evidence:
                continue
            raw_candidates.append(
                PainPointCandidate(
                    title=pp.get("title", "untitled"),
                    description=pp.get("description", ""),
                    evidence=evidence,
                )
            )
            found_this_batch += 1
        print(f"[extract] batch {batch_num}/{num_batches}: {found_this_batch} pain point(s) found")

    print(f"[extract] done: {len(raw_candidates)} raw candidate(s) total")
    return {"raw_candidates": raw_candidates, "usage_log": usage}


def cluster_pain_points(state: ResearchState, config: RunnableConfig) -> dict:
    """Merge duplicate/overlapping raw candidates (role: cheap / 26B)."""
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    dry_run = state.dry_run or options.dry_run

    if not state.raw_candidates:
        print("[cluster] no raw candidates to cluster, skipping")
        return {"candidates": []}

    batch_size = settings.content_budget.max_threads_per_extraction_call * 2
    usage: list[UsageRecord] = []
    merged: list[PainPointCandidate] = []
    num_batches = -(-len(state.raw_candidates) // batch_size)
    print(f"[cluster] {len(state.raw_candidates)} raw candidate(s) in {num_batches} batch(es)...")

    for batch_num, i in enumerate(range(0, len(state.raw_candidates), batch_size), start=1):
        batch = state.raw_candidates[i : i + batch_size]
        digest, index = _digest_candidates(batch)
        prompt = CLUSTERING_PROMPT.format(candidates_digest=digest)

        try:
            if dry_run:
                result = providers.llm.project_call("cheap", "cluster_pain_points", "", prompt)
                usage.append(result.usage)
                continue
            result = providers.llm.generate_structured("cheap", "cluster_pain_points", "", prompt)
        except (LLMParseError, ContentTooLargeError) as e:
            print(f"[cluster] batch {batch_num}/{num_batches} failed, keeping unclustered: {e}")
            merged.extend(batch)  # keep them unclustered rather than losing them
            continue
        usage.append(result.usage)

        grouped_refs: set[str] = set()
        for group in result.data.get("groups", []):
            member_refs = group.get("member_refs", [])
            members = [index[r] for r in member_refs if r in index]
            if not members:
                continue
            grouped_refs.update(r for r in member_refs if r in index)
            all_evidence = [e for m in members for e in m.evidence]
            merged.append(
                PainPointCandidate(
                    title=group.get("title", members[0].title),
                    description=group.get("description", members[0].description),
                    evidence=all_evidence,
                )
            )
        for ref, cand in index.items():
            if ref not in grouped_refs:
                merged.append(cand)  # model didn't group it: keep it standalone
        print(f"[cluster] batch {batch_num}/{num_batches}: {len(merged)} candidate(s) so far")

    print(f"[cluster] done: {len(merged)} candidate(s) after merging")
    return {"candidates": merged, "usage_log": usage}


def prefilter_candidates(state: ResearchState, config: RunnableConfig) -> dict:
    """Drop candidates below the evidence floor, cap how many proceed to
    expensive (31B) validation. No LLM call — pure deterministic gating.
    """
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    gated = gate_candidates(state.candidates, settings.candidate_gating)
    print(
        f"[gate] {len(state.candidates)} candidate(s) in, {len(gated)} passed the evidence floor "
        f"(min {settings.candidate_gating.min_distinct_authors} authors / "
        f"{settings.candidate_gating.min_distinct_threads} threads)"
    )
    return {"candidates": gated}


def fan_out_to_validation(state: ResearchState, config: RunnableConfig):
    """Send one candidate per branch into the validation subgraph.

    Bypasses Send entirely (routes straight to score_and_rank) when
    nothing survived gating, so a quiet run still completes cleanly
    instead of hanging on zero fan-out targets.
    """
    if not state.candidates:
        print("[validate] no candidates cleared gating, nothing to validate")
        return ["score_and_rank"]
    print(f"[validate] validating {len(state.candidates)} candidate(s)...")
    return [
        Send("validate_candidate", {"candidate": c, "dry_run": state.dry_run})
        for c in state.candidates
    ]


def score_and_rank(state: ResearchState, config: RunnableConfig) -> dict:
    ranked = rank_pitches(list(state.scored_pitches))
    print(f"[rank] {len(ranked)} candidate(s) scored and ranked")
    return {"ranked_pitches": ranked}


def export_results(state: ResearchState, config: RunnableConfig) -> dict:
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    run_id = state.run_id or time.strftime("%Y%m%d-%H%M%S")

    if state.dry_run:
        report = export_usage_report(state.usage_log, settings.output, run_id=f"dryrun-{run_id}")
        print(json.dumps(report, indent=2))
        return {}

    result = export_run(state.ranked_pitches, settings.output, run_id=run_id)
    usage_report = export_usage_report(state.usage_log, settings.output, run_id=result["run_id"])
    print(json.dumps({**result, "usage_report": usage_report}, indent=2))
    return {}


# --------------------------------------------------------------------------
# Per-candidate validation subgraph — corroborate -> competitor_scan -> judge.
# A genuinely separate compiled subgraph (not nodes on the main graph) so
# concurrent Send branches never write to shared non-reducer fields; only
# `scored_pitches` / `usage_log` (both reducer-annotated) flow back to the
# parent when each branch finishes. See state.py's module docstring.
# --------------------------------------------------------------------------


def corroborate(state: CandidateValidationState, config: RunnableConfig) -> dict:
    """Search every enabled source for more instances of this pain point
    beyond the initially harvested set, to strengthen the distinct-author
    count.

    Deliberately LLM-free: every source's search is keyword-based and the
    candidate's own title is already a reasonable query, so spending an
    LLM call to craft one isn't worth the tokens.
    """
    if state.dry_run:
        return {}

    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    prefilter = Prefilter(settings.prefilter)
    candidate = state.candidate
    print(f"[corroborate] {candidate.title!r}...")
    # Deliberately smaller than the harvest limit — each hit here costs a
    # further sequential comment-tree fetch per source (confirmed live:
    # `max_threads_per_subreddit`'s 25 here meant up to 25 sequential
    # HTTP round-trips just to corroborate one candidate). This step only
    # needs a handful of corroborating hits, not bulk harvesting depth.
    limit = settings.sources.max_results_per_query

    extra_threads: list[Thread] = []
    if providers.reddit is not None:
        try:
            extra_threads.extend(
                _run_with_timeout(
                    lambda: providers.reddit.search_threads(candidate.title, limit=limit),
                    timeout_s=30,
                    default=[],
                )
            )
        except Exception:
            pass
    if providers.hackernews is not None:
        try:
            extra_threads.extend(
                _run_with_timeout(
                    lambda: providers.hackernews.search(candidate.title, limit=limit),
                    timeout_s=30,
                    default=[],
                )
            )
        except Exception:
            pass
    if providers.stackexchange is not None:
        for site in settings.sources.stackexchange_sites:
            try:
                extra_threads.extend(
                    _run_with_timeout(
                        lambda site=site: providers.stackexchange.search(
                            candidate.title, site=site, limit=limit
                        ),
                        timeout_s=30,
                        default=[],
                    )
                )
            except Exception:
                continue

    existing = {e.permalink for e in candidate.evidence}
    new_evidence: list[Evidence] = []
    for t in extra_threads:
        if (t.permalink not in existing) and (
            prefilter.matches_complaint(t.title) or prefilter.matches_complaint(t.body)
        ):
            new_evidence.append(
                Evidence(
                    content_kind=ContentKind.THREAD,
                    platform=t.platform,
                    community=t.community,
                    thread_id=t.id,
                    author=t.author,
                    score=t.score,
                    excerpt=t.title,
                    permalink=t.permalink,
                    created_utc=t.created_utc,
                )
            )
        for c in t.comments:
            if c.permalink not in existing and prefilter.matches_complaint(c.body):
                new_evidence.append(
                    Evidence(
                        content_kind=ContentKind.COMMENT,
                        platform=t.platform,
                        community=t.community,
                        thread_id=t.id,
                        author=c.author,
                        score=c.score,
                        excerpt=c.body,
                        permalink=c.permalink,
                        created_utc=c.created_utc,
                    )
                )

    print(f"[corroborate] {candidate.title!r}: +{len(new_evidence)} evidence item(s)")
    return {"candidate": candidate.model_copy(update={"evidence": candidate.evidence + new_evidence})}


def competitor_scan(state: CandidateValidationState, config: RunnableConfig) -> dict:
    """Web search + Crawl4AI — the only place Crawl4AI is used, since
    product pages/review sites have no clean API the way Reddit does.
    """
    if state.dry_run:
        return {}

    print(f"[competitor_scan] {state.candidate.title!r}...")

    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    prefilter = Prefilter(settings.prefilter)

    query = f"{state.candidate.title} tool app software"
    search_results = _run_with_timeout(
        lambda: duckduckgo_search(
            query, max_results=settings.content_budget.max_competitor_pages, fetch_full_page=False
        ),
        timeout_s=20,
        default={"results": []},
    )
    urls = [r["url"] for r in search_results.get("results", []) if r.get("url")]

    # Crawl4AI's browser (playwright) not installed, a hung page load, or
    # an anti-bot wall are all real possibilities on any given candidate —
    # bounded so one of them can't freeze the whole validation branch (or,
    # confirmed live, the whole pipeline) when everything else here
    # degrades gracefully already.
    pages = (
        _run_with_timeout(lambda: providers.crawl.fetch_pages(urls), timeout_s=45, default=[])
        if urls
        else []
    )

    evidence_text = " ".join(e.excerpt for e in state.candidate.evidence).lower()
    competitors: list[Competitor] = []
    for page in pages:
        domain = page.url.split("//")[-1].split("/")[0].lower()
        name = page.title or domain
        mentioned = domain in evidence_text or (page.title and page.title.lower() in evidence_text)
        criticized = bool(mentioned and prefilter.matches_complaint(evidence_text))
        competitors.append(
            Competitor(
                name=name,
                url=page.url,
                description=page.markdown[:200],
                strength=CompetitorStrength.MODERATE,
                criticized_in_evidence=criticized,
            )
        )

    print(f"[competitor_scan] {state.candidate.title!r}: {len(competitors)} competitor(s) found")
    return {"competitors": competitors}


def judge(state: CandidateValidationState, config: RunnableConfig) -> dict:
    """The only step that runs on the judge model (31B) — severity, WTP,
    solution gap, buildability signals. Scoring itself happens
    deterministically afterward in scoring.py, not here.
    """
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    candidate = state.candidate
    print(f"[judge] {candidate.title!r}...")

    evidence_digest = "\n".join(
        f'- ({e.author or "unknown"} on {e.platform.value}/{e.community}, {e.score} pts): "{e.excerpt}"'
        for e in candidate.evidence[:25]
    ) or "No evidence excerpts available."
    competitor_digest = (
        "\n".join(f"- {c.name} ({c.url}): {c.description}" for c in state.competitors)
        if state.competitors
        else "No competitors found in web search."
    )
    prompt = JUDGE_PROMPT.format(
        title=candidate.title,
        description=candidate.description,
        evidence_digest=evidence_digest,
        competitor_digest=competitor_digest,
    )

    usage: list[UsageRecord] = []
    if state.dry_run:
        result = providers.llm.project_call("judge", "judge", "", prompt)
        usage.append(result.usage)
        return {"scored_pitches": [], "usage_log": usage}

    judge_signals: Optional[JudgeSignals] = None
    try:
        result = providers.llm.generate_structured("judge", "judge", "", prompt)
        usage.append(result.usage)
        judge_signals = JudgeSignals(**result.data)
    except Exception as e:
        # Keep the candidate in the output (unjudged, scored on evidence
        # alone) rather than dropping it for a single bad LLM response.
        print(f"[judge] {candidate.title!r}: judging failed, scoring on evidence alone: {e}")

    pitch = build_scored_pitch(candidate, state.competitors, judge_signals, settings.scoring)
    print(f"[judge] {candidate.title!r}: score={pitch.score:.2f}")
    return {"scored_pitches": [pitch], "usage_log": usage}


def build_validation_subgraph():
    builder = StateGraph(CandidateValidationState, output_schema=CandidateValidationOutput)
    builder.add_node("corroborate", corroborate)
    builder.add_node("competitor_scan", competitor_scan)
    builder.add_node("judge", judge)
    builder.add_edge(START, "corroborate")
    builder.add_edge("corroborate", "competitor_scan")
    builder.add_edge("competitor_scan", "judge")
    builder.add_edge("judge", END)
    return builder.compile()


# --------------------------------------------------------------------------
# Main graph assembly
# --------------------------------------------------------------------------


def build_graph(checkpointer=None):
    validation_subgraph = build_validation_subgraph()

    builder = StateGraph(
        ResearchState,
        input_schema=ResearchStateInput,
        output_schema=ResearchStateOutput,
    )
    builder.add_node("select_targets", select_targets)
    builder.add_node("harvest_threads", harvest_threads)
    builder.add_node("extract_pain_points", extract_pain_points)
    builder.add_node("cluster_pain_points", cluster_pain_points)
    builder.add_node("prefilter_candidates", prefilter_candidates)
    builder.add_node("validate_candidate", validation_subgraph)
    builder.add_node("score_and_rank", score_and_rank)
    builder.add_node("export_results", export_results)

    builder.add_edge(START, "select_targets")
    builder.add_edge("select_targets", "harvest_threads")
    builder.add_edge("harvest_threads", "extract_pain_points")
    builder.add_edge("extract_pain_points", "cluster_pain_points")
    builder.add_edge("cluster_pain_points", "prefilter_candidates")
    builder.add_conditional_edges(
        "prefilter_candidates", fan_out_to_validation, ["validate_candidate", "score_and_rank"]
    )
    builder.add_edge("validate_candidate", "score_and_rank")
    builder.add_edge("score_and_rank", "export_results")
    builder.add_edge("export_results", END)

    return builder.compile(checkpointer=checkpointer)


# No checkpointer here: this is the entry point `langgraph.json` points
# Studio at, and Studio provides its own run persistence. For standalone
# multi-day autonomous runs, use `run_with_checkpoint` below instead.
graph = build_graph(checkpointer=None)


def run_with_checkpoint(
    initial_state: ResearchStateInput, config: Optional[RunnableConfig] = None
):
    """Run (or resume) a research pass with SQLite-backed checkpointing.

    At 14.4K requests/day, a broad autonomous run can genuinely span
    multiple days. `thread_id` identifies which run to resume — set
    PAIN_RESEARCHER_THREAD_ID to continue a specific prior run, or leave
    it as the default to always resume "the" default run.
    """
    import sqlite3
    from contextlib import closing

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)

    run_config: RunnableConfig = dict(config or {})
    configurable = dict(run_config.get("configurable", {}))
    configurable.setdefault(
        "thread_id", os.environ.get("PAIN_RESEARCHER_THREAD_ID", "pain-researcher-default")
    )
    run_config["configurable"] = configurable

    # Registers our custom Pydantic/dataclass types with the checkpoint
    # serializer explicitly — confirmed live: without this, every one
    # prints a "will be blocked in a future version" deserialization
    # warning on every resumed run, and would eventually break outright.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("pain_researcher.models", "Platform"),
            ("pain_researcher.models", "Thread"),
            ("pain_researcher.models", "Comment"),
            ("pain_researcher.models", "ContentKind"),
            ("pain_researcher.models", "Evidence"),
            ("pain_researcher.models", "PainPointCandidate"),
            ("pain_researcher.models", "Competitor"),
            ("pain_researcher.models", "CompetitorStrength"),
            ("pain_researcher.models", "SolutionGap"),
            ("pain_researcher.models", "JudgeSignals"),
            ("pain_researcher.models", "ScoredPitch"),
            ("pain_researcher.state", "UsageRecord"),
        ]
    )
    with closing(sqlite3.connect(settings.checkpoint.db_path, check_same_thread=False)) as conn:
        checkpointer = SqliteSaver(conn, serde=serde)
        return build_graph(checkpointer=checkpointer).invoke(initial_state, config=run_config)


if __name__ == "__main__":
    # Only needed here: `langgraph dev` auto-loads .env itself via
    # langgraph.json's "env" pointer, but a plain `python -m
    # pain_researcher.graph` invocation has no such wrapper around it.
    from dotenv import load_dotenv

    load_dotenv()

    # Python fully buffers stdout when it isn't a live terminal (piped,
    # redirected to a file, or run under some IDE/CI runners) — confirmed
    # while testing this exact script, where progress prints didn't
    # appear until the process exited. Forcing line-buffering means the
    # `[harvest]`/`[extract]`/etc. progress lines show up as they happen
    # regardless of how this is run, not just in an interactive terminal.
    #
    # encoding="utf-8", errors="replace" is separately load-bearing now
    # that web_search crawls arbitrary, potentially non-English pages —
    # confirmed live: Windows' console defaults to cp1252, which crashed
    # outright trying to print a Vietnamese page title. Every print() in
    # this program touches content that ultimately comes from the open
    # web, so this needs to be crash-proof everywhere, not just here.
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

    result = run_with_checkpoint(ResearchStateInput())
    print(f"Run complete: {len(result.get('ranked_pitches', []))} ranked pitches.")
