"""Prompt templates for every LLM step in the pipeline.

All prompts assume prompt-based JSON output (Gemma 4's default profile
has `supports_tool_calling: false`), so every template ends with an
explicit FORMAT block and an example, mirroring the JSON-mode style in
`ollama_deep_researcher/prompts.py` — small/mid models are measurably
more reliable at valid JSON when shown the exact shape first.

Extraction and clustering prompts ask the model for *indices* back into
the evidence it was shown, never for it to reproduce URLs or permalinks
itself — Reddit permalinks come from PRAW data the graph already has, so
letting the LLM regenerate them would just be an opportunity to
hallucinate a plausible-looking but wrong link.
"""

from __future__ import annotations

from datetime import datetime


def get_current_date() -> str:
    return datetime.now().strftime("%B %d, %Y")


# --------------------------------------------------------------------------
# Autonomous discovery: niche + subreddit proposal (role: cheap)
# --------------------------------------------------------------------------

NICHE_PROPOSAL_PROMPT = """You help find software business opportunities by identifying niches worth researching on Reddit for recurring, monetizable pain points.

<CONTEXT>
Current date: {current_date}
Already explored this run (do not repeat): {excluded_niches}
</CONTEXT>

<GOAL>
Propose up to {max_niches} distinct niches or communities (professions, hobbies, small-business types, workflows) where people are likely to have recurring, describable, software-solvable problems. Favor specific, addressable niches ("freelance bookkeepers doing manual invoice reconciliation") over broad ones ("small business").
</GOAL>

<FORMAT>
Respond with ONLY a JSON object:
{{
  "niches": [
    {{"niche": "short niche label", "rationale": "why this niche likely has an underserved, describable pain point"}}
  ]
}}
</FORMAT>

Example:
{{
  "niches": [
    {{"niche": "solo Etsy sellers doing shipping/customs", "rationale": "frequent complaints about manual customs forms across seller subreddits, low existing tooling"}}
  ]
}}"""


SUBREDDIT_PROPOSAL_PROMPT = """Given a niche, propose Reddit subreddit names likely to contain real discussion from people in that niche.

<NICHE>
{niche}
</NICHE>

<REQUIREMENTS>
- Up to {max_subreddits} subreddit names, no "r/" prefix
- Prefer active, on-topic communities over guesses at obscure ones
- Every name will be verified against Reddit's API before use, so it is fine to propose a name you are not fully certain exists
</REQUIREMENTS>

<FORMAT>
{{"subreddits": ["name1", "name2"]}}
</FORMAT>"""


# --------------------------------------------------------------------------
# Extraction (role: cheap) — batched over threads
# --------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract concrete, recurring pain points from these Reddit threads. A pain point is something people are actively frustrated by, working around manually, or wishing a tool existed for — not a general topic or a one-off rant with no repeatable problem.

<THREADS>
{threads_digest}
</THREADS>

<REQUIREMENTS>
1. Only extract pain points that a piece of software could plausibly address.
2. For each pain point, list the evidence_refs (the labels like "T1", "T1_C2") for every thread/comment that supports it. A pain point mentioned in only one place is still valid — corroboration happens later.
3. Do not invent URLs, usernames, or quotes not present in the input.
4. Skip threads that don't contain an extractable pain point.
</REQUIREMENTS>

<FORMAT>
{{
  "pain_points": [
    {{
      "title": "short title",
      "description": "1-3 sentence description of the problem, in your own words",
      "evidence_refs": ["T1", "T1_C2"]
    }}
  ]
}}
</FORMAT>"""


# --------------------------------------------------------------------------
# Clustering (role: cheap) — merge duplicate/overlapping candidates
# --------------------------------------------------------------------------

CLUSTERING_PROMPT = """These pain points were extracted independently from different Reddit threads and may describe the same underlying problem in different words. Group them.

<CANDIDATES>
{candidates_digest}
</CANDIDATES>

<REQUIREMENTS>
1. Merge candidates that describe the same underlying pain point, even if worded differently.
2. Keep genuinely distinct pain points separate — do not over-merge.
3. For each merged group, write one clear title and description covering all its members, and list every original candidate index (like "C1", "C4") that belongs to the group.
4. Every input candidate index must appear in exactly one group.
</REQUIREMENTS>

<FORMAT>
{{
  "groups": [
    {{"title": "merged title", "description": "merged description", "member_refs": ["C1", "C4"]}}
  ]
}}
</FORMAT>"""


# --------------------------------------------------------------------------
# Judge (role: judge) — the only LLM step that runs on 31B
# --------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating whether a Reddit-sourced pain point is a real, worthwhile software business opportunity. Emit signals for scoring — do not compute an overall score yourself, a separate deterministic step does that from your signals.

<PAIN POINT>
{title}
{description}
</PAIN POINT>

<EVIDENCE EXCERPTS>
{evidence_digest}
</EVIDENCE EXCERPTS>

<COMPETITOR RESEARCH>
{competitor_digest}
</COMPETITOR RESEARCH>

<GOAL>
1. severity: 0-5, how much this problem actually costs people (time, money, risk) vs. mild annoyance.
2. willingness_to_pay: true only if the evidence contains an explicit or strongly implied signal someone would pay (already paying for a workaround, said they'd pay, hired help, built their own tool). Quote the exact evidence line(s) that justify this in wtp_evidence — if none, return false and an empty list, do not infer without a quote.
3. solution_gap: "none_found" if no competitors were found in the research, "weak" if competitors exist but are criticized in the evidence or clearly incomplete, "strong" if well-regarded competitors already solve this well.
4. buildability: 0-5, how buildable this is as a solo/small-team MVP (0 = needs a large team, proprietary data, or regulatory approval; 5 = clearly scoped, buildable in weeks).
5. hard_blockers: list any structural blockers (e.g. "requires proprietary bank data access", "needs FDA approval", "needs a two-sided marketplace to have any value") — empty list if none.
</GOAL>

<FORMAT>
{{
  "severity": 3.5,
  "willingness_to_pay": true,
  "wtp_evidence": ["exact quoted line from evidence"],
  "solution_gap": "weak",
  "buildability": 4.0,
  "hard_blockers": [],
  "reasoning": "1-2 sentence justification"
}}
</FORMAT>"""
