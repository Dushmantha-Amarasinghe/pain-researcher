"""Export ranked pitches to JSON/CSV plus per-candidate markdown briefs.

This is the deliverable the whole pipeline exists to produce: a
sortable/filterable record (JSON/CSV) for triaging many ideas at once,
and a one-page evidence brief per top candidate for actually deciding
whether to build it — replacing the original repo's single markdown-blob
output, which can't be sorted or compared across candidates.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pain_researcher.config import OutputConfig
from pain_researcher.models import ScoredPitch
from pain_researcher.state import UsageRecord


def _run_dir(output: OutputConfig, run_id: str) -> Path:
    d = Path(output.output_dir) / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def export_json(pitches: list[ScoredPitch], output: OutputConfig, run_id: str) -> Path:
    path = _run_dir(output, run_id) / "ranked_pitches.json"
    payload = [p.model_dump(mode="json") for p in pitches]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_csv(pitches: list[ScoredPitch], output: OutputConfig, run_id: str) -> Path:
    path = _run_dir(output, run_id) / "ranked_pitches.csv"
    fieldnames = [
        "rank",
        "title",
        "score",
        "distinct_authors",
        "distinct_threads",
        "subreddits",
        "severity",
        "willingness_to_pay",
        "solution_gap",
        "buildability",
        "hard_blockers",
        "top_competitor",
        "description",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pitches:
            js = p.judge_signals
            writer.writerow(
                {
                    "rank": p.rank,
                    "title": p.candidate.title,
                    "score": round(p.score, 2),
                    "distinct_authors": len(p.candidate.distinct_authors),
                    "distinct_threads": len(p.candidate.distinct_threads),
                    "subreddits": ", ".join(sorted(p.candidate.subreddits)),
                    "severity": js.severity if js else "",
                    "willingness_to_pay": js.willingness_to_pay if js else "",
                    "solution_gap": js.solution_gap.value if js else "",
                    "buildability": js.buildability if js else "",
                    "hard_blockers": "; ".join(js.hard_blockers) if js else "",
                    "top_competitor": p.competitors[0].name if p.competitors else "",
                    "description": p.candidate.description,
                }
            )
    return path


def _brief_markdown(pitch: ScoredPitch) -> str:
    c = pitch.candidate
    js = pitch.judge_signals
    lines = [
        f"# #{pitch.rank}: {c.title}",
        f"**Score:** {pitch.score:.2f}",
        "",
        "## Problem",
        c.description,
        "",
        f"## Evidence ({len(c.distinct_authors)} distinct people, "
        f"{len(c.distinct_threads)} threads, subreddits: {', '.join(sorted(c.subreddits))})",
    ]
    for e in c.evidence[:15]:
        lines.append(
            f'- "{e.excerpt}" — u/{e.author or "unknown"}, {e.score} pts ([source]({e.permalink}))'
        )

    lines += ["", "## Judge Assessment"]
    if js:
        lines.append(f"- **Severity:** {js.severity}/5")
        lines.append(f"- **Willingness to pay:** {'Yes' if js.willingness_to_pay else 'No'}")
        for q in js.wtp_evidence:
            lines.append(f"  - > {q}")
        lines.append(f"- **Solution gap:** {js.solution_gap.value}")
        lines.append(f"- **Buildability:** {js.buildability}/5")
        if js.hard_blockers:
            lines.append(f"- **Hard blockers:** {', '.join(js.hard_blockers)}")
        if js.reasoning:
            lines.append(f"- **Reasoning:** {js.reasoning}")
    else:
        lines.append("_Not judged (below candidate-gating threshold, or run stopped early)_")

    lines += ["", "## Competitors Found"]
    if pitch.competitors:
        for comp in pitch.competitors:
            crit = " (criticized in evidence)" if comp.criticized_in_evidence else ""
            lines.append(f"- [{comp.name}]({comp.url}) — {comp.strength.value}{crit}: {comp.description}")
    else:
        lines.append("_None found_")

    lines += ["", "## Score Breakdown"]
    for k, v in pitch.score_breakdown.items():
        lines.append(f"- {k}: {v:+.2f}")

    return "\n".join(lines)


def export_briefs(pitches: list[ScoredPitch], output: OutputConfig, run_id: str) -> list[Path]:
    d = _run_dir(output, run_id) / "briefs"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for pitch in pitches[: output.top_n_briefs]:
        slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in pitch.candidate.title.lower())
        slug = slug.strip("-")[:60] or "untitled"
        path = d / f"{pitch.rank:02d}-{slug}.md"
        path.write_text(_brief_markdown(pitch), encoding="utf-8")
        paths.append(path)
    return paths


def export_usage_report(
    usage_log: list[UsageRecord], output: OutputConfig, run_id: str
) -> dict:
    """Per-node, per-model request/token breakdown for one run.

    This is the "which node is burning my budget" view the Google AI
    Studio dashboard can't give you — it only reports aggregate usage per
    model, not per pipeline step. Written for both live and dry runs so a
    dry run's projected spend and a live run's actual spend are
    directly comparable.
    """
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    by_node: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    for rec in usage_log:
        for bucket, key in ((by_model, rec.model_key), (by_node, rec.node)):
            bucket[key]["calls"] += 1
            bucket[key]["input_tokens"] += rec.input_tokens
            bucket[key]["output_tokens"] += rec.output_tokens

    report = {
        "run_id": run_id,
        "by_model": {k: dict(v) for k, v in by_model.items()},
        "by_node": {k: dict(v) for k, v in by_node.items()},
        "total_calls": len(usage_log),
        "total_input_tokens": sum(r.input_tokens for r in usage_log),
        "total_output_tokens": sum(r.output_tokens for r in usage_log),
    }
    path = _run_dir(output, run_id) / "usage_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["path"] = str(path)
    return report


def export_run(
    pitches: list[ScoredPitch], output: OutputConfig, run_id: Optional[str] = None
) -> dict:
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = export_json(pitches, output, run_id)
    csv_path = export_csv(pitches, output, run_id)
    brief_paths = export_briefs(pitches, output, run_id)
    return {
        "run_id": run_id,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "brief_paths": [str(p) for p in brief_paths],
    }
