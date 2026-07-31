"""Stack Exchange data access via the official API (api.stackexchange.com).

Works fully unauthenticated (300 requests/day/IP, confirmed live) — an
optional app key (free registration at stackapps.com, no OAuth/user
login involved) raises that to 10,000/day. Response shapes below are
confirmed against the live API, not guessed from docs.

Stack Exchange is multi-tenant: ~170 sites (Stack Overflow,
money.stackexchange, webmasters, sysadmin, ux, salesforce, ...), each a
`site` slug analogous to a subreddit — every fetch needs one as a target.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any, Optional

import httpx

from pain_researcher.config import ContentBudgetConfig
from pain_researcher.models import Comment, Platform, Thread

BASE_URL = "https://api.stackexchange.com/2.3"


def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(without_tags).strip()


class StackExchangeProvider:
    def __init__(self, content_budget: ContentBudgetConfig, api_key: Optional[str] = None):
        self._content_budget = content_budget
        self._api_key = api_key
        self._client = httpx.Client(base_url=BASE_URL, timeout=10.0)
        # Unauthenticated SE traffic hits a burst-rate 429 well before the
        # 300/day quota is anywhere near exhausted — confirmed live by
        # firing requests with zero delay. A small fixed pace between
        # calls, plus one retry on 429, is enough to avoid it in practice.
        self._min_interval = 0.15
        self._last_request_at = 0.0

    def _params(self, **kwargs: Any) -> dict[str, Any]:
        params = {"filter": "withbody", **kwargs}
        if self._api_key:
            params["key"] = self._api_key
        return params

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        wait = self._min_interval - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)
        resp = self._client.get(path, params=params)
        self._last_request_at = time.monotonic()
        if resp.status_code == 429:
            time.sleep(2.0)
            resp = self._client.get(path, params=params)
            self._last_request_at = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def fetch_questions(self, site: str, limit: Optional[int] = None) -> list[Thread]:
        """Pull recently-active questions from one Stack Exchange site."""
        limit = limit or self._content_budget.max_threads_per_subreddit
        try:
            data = self._get(
                "/questions",
                self._params(site=site, pagesize=limit, order="desc", sort="activity"),
            )
            items = data.get("items", [])
        except Exception as e:
            print(f"Warning: Stack Exchange fetch_questions failed for {site}: {e}")
            return []
        return [self._to_thread(item, site) for item in items]

    def search(self, query: str, site: str, limit: int = 25) -> list[Thread]:
        """Full-text search on one site — used by the corroborate step."""
        try:
            data = self._get(
                "/search/advanced", self._params(q=query, site=site, pagesize=limit, sort="relevance")
            )
            items = data.get("items", [])
        except Exception as e:
            print(f"Warning: Stack Exchange search failed for {site}: {e}")
            return []
        return [self._to_thread(item, site) for item in items]

    def _to_thread(self, item: dict[str, Any], site: str) -> Thread:
        cb = self._content_budget
        question_id = str(item["question_id"])
        owner = item.get("owner") or {}
        link = item.get("link") or f"https://{site}.stackexchange.com/questions/{question_id}"
        return Thread(
            id=question_id,
            platform=Platform.STACKEXCHANGE,
            community=site,
            title=_strip_html(item.get("title")),
            body=_strip_html(item.get("body"))[: cb.max_chars_per_source],
            author=owner.get("display_name"),
            score=item.get("score") or 0,
            num_comments=item.get("answer_count") or 0,
            created_utc=float(item.get("creation_date") or 0),
            permalink=link,
            comments=self._fetch_answers(question_id, site, link),
        )

    def _fetch_answers(self, question_id: str, site: str, question_link: str) -> list[Comment]:
        """Answers stand in for "comments" here — sorted by vote score,
        same "top N by engagement" pattern as Reddit's comment selection.

        Answer permalinks are built as a fragment on the question's own
        link (`#{answer_id}`) rather than guessing a `{site}.stackexchange.com`
        domain, since Stack Overflow itself lives at stackoverflow.com,
        not stackoverflow.stackexchange.com — reusing the real question
        URL sidesteps that per-site domain exception entirely.
        """
        cb = self._content_budget
        try:
            data = self._get(
                f"/questions/{question_id}/answers",
                self._params(site=site, pagesize=cb.max_comments_per_thread, order="desc", sort="votes"),
            )
            items = data.get("items", [])
        except Exception as e:
            print(f"Warning: Stack Exchange answers fetch failed for {question_id}: {e}")
            return []

        answers: list[Comment] = []
        for a in items[: cb.max_comments_per_thread]:
            body = _strip_html(a.get("body"))[: cb.max_chars_per_comment]
            if not body:
                continue
            owner = a.get("owner") or {}
            answer_id = a.get("answer_id")
            answers.append(
                Comment(
                    id=str(answer_id),
                    author=owner.get("display_name"),
                    body=body,
                    score=a.get("score") or 0,
                    created_utc=float(a.get("creation_date") or 0),
                    permalink=f"{question_link}#{answer_id}",
                )
            )
        return answers
