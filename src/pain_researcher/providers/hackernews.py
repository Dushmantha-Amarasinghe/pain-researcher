"""Hacker News data access via Algolia's HN Search API.

Free, keyless, no documented rate limit (confirmed live as of mid-2026)
— strictly less friction than Reddit's official API. "Ask HN" threads in
particular are close to pre-filtered pain-point data: people are
literally asking "is there a tool for X" or describing a workaround they
built.

Two endpoints, response shapes confirmed against the live API:
- `/search` — text/tag search over stories, returns `points`/
  `num_comments` directly (the engagement signal) but comments only as
  an ID list under `children`.
- `/items/{id}` — the full nested tree for one story, with actual
  comment text. Comments don't carry a public score via this API
  (`points` is always null on comment items), so top-level comments are
  taken in API order rather than sorted by score — a real limitation of
  this source compared to Reddit/Stack Exchange.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any, Optional

import httpx

from pain_researcher.config import ContentBudgetConfig
from pain_researcher.models import Comment, Platform, Thread

BASE_URL = "https://hn.algolia.com/api/v1"


def _strip_html(text: Optional[str]) -> str:
    """HN item text arrives as lightly-tagged HTML (<p>, <a>, <i>, <code>, entities)."""
    if not text:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(without_tags).strip()


class HackerNewsProvider:
    def __init__(self, content_budget: ContentBudgetConfig):
        self._content_budget = content_budget
        self._client = httpx.Client(base_url=BASE_URL, timeout=10.0)

    def fetch_ask_hn(self, limit: Optional[int] = None) -> list[Thread]:
        """Pull Ask HN threads, Algolia's default relevance/recency mix."""
        limit = limit or self._content_budget.max_threads_per_subreddit
        hits = self._search(tags="ask_hn", limit=limit)
        return [self._to_thread(h) for h in hits]

    def search(
        self, query: str, limit: int = 25, max_age_days: Optional[float] = None
    ) -> list[Thread]:
        """Full-text search across stories.

        `max_age_days`, when given, bounds Algolia's relevance ranking to
        posts newer than that — plain relevance search over a generic
        phrase like "is there a tool" systematically surfaces HN's
        best-known posts from its full 15+ year history (confirmed live:
        every unbounded hit came back 3-17 years old), while the
        date-sorted endpoint swings the other way to posts too new to
        have any engagement yet. Bounding relevance search by recency
        gets both: established engagement, within a window still worth
        acting on. Left unbounded for the corroborate step, where an
        older mention is still valid corroborating evidence.
        """
        hits = self._search(query=query, tags="story", limit=limit, max_age_days=max_age_days)
        return [self._to_thread(h) for h in hits]

    def _search(
        self,
        query: str = "",
        tags: str = "story",
        limit: int = 25,
        max_age_days: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": query, "tags": tags, "hitsPerPage": limit}
        if max_age_days is not None:
            cutoff = int(time.time() - max_age_days * 86400)
            params["numericFilters"] = f"created_at_i>{cutoff}"
        try:
            resp = self._client.get("/search", params=params)
            resp.raise_for_status()
            return resp.json().get("hits", [])
        except Exception as e:
            print(f"Warning: HN search failed: {e}")
            return []

    def _to_thread(self, hit: dict[str, Any]) -> Thread:
        story_id = hit.get("objectID") or str(hit.get("story_id", ""))
        return Thread(
            id=story_id,
            platform=Platform.HACKERNEWS,
            community="hackernews",
            title=hit.get("title") or "",
            body=_strip_html(hit.get("story_text"))[: self._content_budget.max_chars_per_source],
            author=hit.get("author"),
            score=hit.get("points") or 0,
            num_comments=hit.get("num_comments") or 0,
            created_utc=float(hit.get("created_at_i") or 0),
            permalink=f"https://news.ycombinator.com/item?id={story_id}",
            comments=self._fetch_comments(story_id),
        )

    def _fetch_comments(self, story_id: str) -> list[Comment]:
        """Fetch the full item tree for one story and flatten its
        top-level comments only — same "top comments, not deep reply
        chains" pattern used by the Reddit provider.
        """
        cb = self._content_budget
        try:
            resp = self._client.get(f"/items/{story_id}")
            resp.raise_for_status()
            item = resp.json()
        except Exception as e:
            print(f"Warning: HN item fetch failed for {story_id}: {e}")
            return []

        comments: list[Comment] = []
        for child in (item.get("children") or [])[: cb.max_comments_per_thread]:
            if child.get("type") != "comment":
                continue
            body = _strip_html(child.get("text"))[: cb.max_chars_per_comment]
            if not body:
                continue
            comments.append(
                Comment(
                    id=str(child["id"]),
                    author=child.get("author"),
                    body=body,
                    score=child.get("points") or 0,
                    created_utc=float(child.get("created_at_i") or 0),
                    permalink=f"https://news.ycombinator.com/item?id={child['id']}",
                )
            )
        return comments
