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
import time
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
    Thread,
)
from pain_researcher.prefilter import Prefilter, gate_candidates
from pain_researcher.prompts import (
    CLUSTERING_PROMPT,
    EXTRACTION_PROMPT,
    JUDGE_PROMPT,
    NICHE_PROPOSAL_PROMPT,
    SUBREDDIT_PROPOSAL_PROMPT,
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

    if providers.reddit is not None:
        for sub in state.target_subreddits:
            try:
                all_threads.extend(providers.reddit.fetch_threads(sub))
            except Exception:
                continue  # one bad subreddit shouldn't abort harvesting the rest

    if providers.hackernews is not None:
        try:
            all_threads.extend(providers.hackernews.fetch_ask_hn())
        except Exception:
            pass  # HN being briefly unreachable shouldn't abort the whole harvest

    if providers.stackexchange is not None:
        for site in settings.sources.stackexchange_sites:
            try:
                all_threads.extend(providers.stackexchange.fetch_questions(site))
            except Exception:
                continue  # one bad site shouldn't abort harvesting the rest

    return {"threads": prefilter.filter_threads(all_threads)}


def extract_pain_points(state: ResearchState, config: RunnableConfig) -> dict:
    """Batched extraction over harvested threads (role: cheap / 26B)."""
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    dry_run = state.dry_run or options.dry_run
    batch_size = settings.content_budget.max_threads_per_extraction_call

    usage: list[UsageRecord] = []
    raw_candidates: list[PainPointCandidate] = []

    for i in range(0, len(state.threads), batch_size):
        batch = state.threads[i : i + batch_size]
        digest, evidence_index = _digest_threads(batch)
        prompt = EXTRACTION_PROMPT.format(threads_digest=digest)

        try:
            if dry_run:
                result = providers.llm.project_call("cheap", "extract_pain_points", "", prompt)
                usage.append(result.usage)
                continue
            result = providers.llm.generate_structured("cheap", "extract_pain_points", "", prompt)
        except (LLMParseError, ContentTooLargeError):
            continue
        usage.append(result.usage)

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

    return {"raw_candidates": raw_candidates, "usage_log": usage}


def cluster_pain_points(state: ResearchState, config: RunnableConfig) -> dict:
    """Merge duplicate/overlapping raw candidates (role: cheap / 26B)."""
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    dry_run = state.dry_run or options.dry_run

    if not state.raw_candidates:
        return {"candidates": []}

    batch_size = settings.content_budget.max_threads_per_extraction_call * 2
    usage: list[UsageRecord] = []
    merged: list[PainPointCandidate] = []

    for i in range(0, len(state.raw_candidates), batch_size):
        batch = state.raw_candidates[i : i + batch_size]
        digest, index = _digest_candidates(batch)
        prompt = CLUSTERING_PROMPT.format(candidates_digest=digest)

        try:
            if dry_run:
                result = providers.llm.project_call("cheap", "cluster_pain_points", "", prompt)
                usage.append(result.usage)
                continue
            result = providers.llm.generate_structured("cheap", "cluster_pain_points", "", prompt)
        except (LLMParseError, ContentTooLargeError):
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

    return {"candidates": merged, "usage_log": usage}


def prefilter_candidates(state: ResearchState, config: RunnableConfig) -> dict:
    """Drop candidates below the evidence floor, cap how many proceed to
    expensive (31B) validation. No LLM call — pure deterministic gating.
    """
    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    return {"candidates": gate_candidates(state.candidates, settings.candidate_gating)}


def fan_out_to_validation(state: ResearchState, config: RunnableConfig):
    """Send one candidate per branch into the validation subgraph.

    Bypasses Send entirely (routes straight to score_and_rank) when
    nothing survived gating, so a quiet run still completes cleanly
    instead of hanging on zero fan-out targets.
    """
    if not state.candidates:
        return ["score_and_rank"]
    return [
        Send("validate_candidate", {"candidate": c, "dry_run": state.dry_run})
        for c in state.candidates
    ]


def score_and_rank(state: ResearchState, config: RunnableConfig) -> dict:
    return {"ranked_pitches": rank_pitches(list(state.scored_pitches))}


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
    limit = settings.content_budget.max_threads_per_subreddit

    extra_threads: list[Thread] = []
    if providers.reddit is not None:
        try:
            extra_threads.extend(providers.reddit.search_threads(candidate.title, limit=limit))
        except Exception:
            pass
    if providers.hackernews is not None:
        try:
            extra_threads.extend(providers.hackernews.search(candidate.title, limit=limit))
        except Exception:
            pass
    if providers.stackexchange is not None:
        for site in settings.sources.stackexchange_sites:
            try:
                extra_threads.extend(
                    providers.stackexchange.search(candidate.title, site=site, limit=limit)
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

    return {"candidate": candidate.model_copy(update={"evidence": candidate.evidence + new_evidence})}


def competitor_scan(state: CandidateValidationState, config: RunnableConfig) -> dict:
    """Web search + Crawl4AI — the only place Crawl4AI is used, since
    product pages/review sites have no clean API the way Reddit does.
    """
    if state.dry_run:
        return {}

    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)
    providers = get_providers(options)
    prefilter = Prefilter(settings.prefilter)

    query = f"{state.candidate.title} tool app software"
    try:
        search_results = duckduckgo_search(
            query, max_results=settings.content_budget.max_competitor_pages, fetch_full_page=False
        )
        urls = [r["url"] for r in search_results.get("results", []) if r.get("url")]
    except Exception:
        urls = []

    try:
        pages = providers.crawl.fetch_pages(urls) if urls else []
    except Exception as e:
        # Crawl4AI's browser (playwright) not installed yet, a page timing
        # out, or an anti-bot wall are all real possibilities on any given
        # candidate — one of them shouldn't crash that candidate's whole
        # validation branch when everything else here degrades gracefully.
        print(f"Warning: competitor page crawl failed: {e}")
        pages = []

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
    except Exception:
        # Keep the candidate in the output (unjudged, scored on evidence
        # alone) rather than dropping it for a single bad LLM response.
        pass

    pitch = build_scored_pitch(candidate, state.competitors, judge_signals, settings.scoring)
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
    from langgraph.checkpoint.sqlite import SqliteSaver

    options = RunOptions.from_runnable_config(config)
    settings = get_settings(options.settings_path)

    run_config: RunnableConfig = dict(config or {})
    configurable = dict(run_config.get("configurable", {}))
    configurable.setdefault(
        "thread_id", os.environ.get("PAIN_RESEARCHER_THREAD_ID", "pain-researcher-default")
    )
    run_config["configurable"] = configurable

    with SqliteSaver.from_conn_string(settings.checkpoint.db_path) as checkpointer:
        return build_graph(checkpointer=checkpointer).invoke(initial_state, config=run_config)


if __name__ == "__main__":
    result = run_with_checkpoint(ResearchStateInput())
    print(f"Run complete: {len(result.get('ranked_pitches', []))} ranked pitches.")
