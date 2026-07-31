"""Crawl4AI wrapper — competitor discovery only.

Scoped deliberately narrow: Reddit data always comes from PRAW
(`reddit.py`), never scraped. Crawl4AI is used only here, for pages that
have no clean API — product landing pages, review-site listings,
Product Hunt / Indie Hackers threads surfaced by a web search — to check
whether a candidate pain point already has a solution on the market.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pain_researcher.config import ContentBudgetConfig


@dataclass
class CompetitorPage:
    url: str
    markdown: str
    title: str = ""


def _extract_markdown(result) -> str:
    """Crawl4AI's `result.markdown` is a plain string in the basic case, or
    a `MarkdownGenerationResult` object exposing `.fit_markdown` /
    `.raw_markdown` when a content filter is configured (as we do below).
    Handle both shapes rather than assuming one — this field's type has
    changed across Crawl4AI releases before (e.g. `markdown_v2` removal).
    """
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    return getattr(md, "fit_markdown", None) or getattr(md, "raw_markdown", None) or ""


class CrawlProvider:
    def __init__(self, content_budget: ContentBudgetConfig):
        self._content_budget = content_budget

    def fetch_pages(self, urls: list[str]) -> list[CompetitorPage]:
        """Fetch and markdown-ify up to `max_competitor_pages` URLs.

        Runs Crawl4AI's async crawler synchronously via `asyncio.run` —
        the rest of this pipeline is sync (matching the LangGraph node
        style used throughout), and competitor scanning is low-frequency
        enough (only for shortlisted candidates, capped per candidate)
        that per-call event-loop overhead doesn't matter.
        """
        capped = urls[: self._content_budget.max_competitor_pages]
        if not capped:
            return []
        return asyncio.run(self._fetch_pages_async(capped))

    async def _fetch_pages_async(self, urls: list[str]) -> list[CompetitorPage]:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        char_cap = self._content_budget.max_chars_per_competitor_page
        browser_config = BrowserConfig(headless=True, verbose=False)
        # Pruning filter trims boilerplate/nav/footer before markdown-ifying,
        # which matters here: every page we crawl eventually feeds the 31B
        # judge call, and TPM is the binding constraint on that call.
        # page_timeout set explicitly (confirmed live: a hung crawl here
        # can freeze the whole pipeline indefinitely with no timeout at all).
        run_config = CrawlerRunConfig(
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter()
            ),
            page_timeout=20000,
        )

        pages: list[CompetitorPage] = []
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url, config=run_config)
                    if not result or not getattr(result, "success", False):
                        continue
                    markdown = _extract_markdown(result)[:char_cap]
                    metadata = getattr(result, "metadata", None) or {}
                    title = metadata.get("title", "") if isinstance(metadata, dict) else ""
                    if markdown.strip():
                        pages.append(CompetitorPage(url=url, markdown=markdown, title=title))
                except Exception:
                    # One bad competitor URL (timeout, anti-bot wall, dead
                    # link) shouldn't abort the whole competitor scan.
                    continue
        return pages
