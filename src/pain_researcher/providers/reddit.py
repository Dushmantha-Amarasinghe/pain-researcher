"""Reddit data access via PRAW (the official, OAuth-backed Reddit API).

Scraping Reddit directly was considered and rejected: as of mid-2026,
unauthenticated `.json` endpoints frequently 403, Reddit's policy names
unauthorized scraping a Rule 8 violation, and — most importantly for this
system — PRAW returns upvotes, comment counts, and authors as structured
fields. That structure *is* the validation signal this whole pipeline is
built around; scraped HTML would just be prose the LLM has to guess-parse
numbers out of.
"""

from __future__ import annotations

from typing import Any, Optional

from pain_researcher.config import ContentBudgetConfig, Credentials, DiscoveryConfig
from pain_researcher.models import RedditComment, RedditThread


def clean_subreddit_name(name: str) -> str:
    """Normalize a subreddit reference like "r/foo", "/r/foo/", or "foo" to "foo"."""
    cleaned = name.strip()
    for prefix in ("/r/", "r/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned.strip("/").strip()


class SubredditVerificationError(RuntimeError):
    """Raised only for unexpected errors — a nonexistent/private/small
    subreddit is a normal outcome and returns None, not an exception."""


class RedditProvider:
    def __init__(
        self,
        credentials: Credentials,
        content_budget: ContentBudgetConfig,
        discovery: DiscoveryConfig,
    ):
        import praw

        self._reddit = praw.Reddit(
            client_id=credentials.reddit_client_id,
            client_secret=credentials.reddit_client_secret,
            user_agent=credentials.reddit_user_agent,
        )
        self._reddit.read_only = True
        self._content_budget = content_budget
        self._discovery = discovery

    def verify_subreddit(self, name: str) -> Optional[dict[str, Any]]:
        """Confirm a (possibly LLM-proposed) subreddit actually exists,
        is public, and meets the minimum subscriber floor.

        This is the guard against autonomous/seed discovery mode inventing
        a plausible-sounding subreddit that doesn't exist — every proposed
        name is checked here before any harvesting happens against it.
        """
        import prawcore

        clean = clean_subreddit_name(name)
        if not clean:
            return None
        try:
            sub = self._reddit.subreddit(clean)
            subscribers = sub.subscribers  # first access triggers the actual fetch
            if getattr(sub, "subreddit_type", None) not in ("public", "restricted"):
                return None
            if subscribers is None or subscribers < self._discovery.min_subreddit_subscribers:
                return None
            return {
                "name": sub.display_name,
                "subscribers": subscribers,
                "public_description": sub.public_description or "",
            }
        except (
            prawcore.exceptions.NotFound,
            prawcore.exceptions.Redirect,
            prawcore.exceptions.Forbidden,
        ):
            return None
        except Exception:
            # Any other transient PRAW/network hiccup: treat as "couldn't
            # verify" rather than crashing target selection for the whole run.
            return None

    def search_subreddits(self, query: str, limit: int) -> list[str]:
        """Find candidate subreddit names for a niche via Reddit's own search."""
        results: list[str] = []
        try:
            for sub in self._reddit.subreddits.search(query, limit=limit):
                results.append(sub.display_name)
        except Exception:
            pass
        return results

    def fetch_threads(
        self,
        subreddit_name: str,
        limit: Optional[int] = None,
        time_filter: str = "year",
    ) -> list[RedditThread]:
        """Pull top threads (by score) from a verified subreddit."""
        limit = limit or self._content_budget.max_threads_per_subreddit
        sub = self._reddit.subreddit(clean_subreddit_name(subreddit_name))
        return [self._to_thread(s) for s in sub.top(time_filter=time_filter, limit=limit)]

    def search_threads(
        self, query: str, subreddit_name: Optional[str] = None, limit: int = 25
    ) -> list[RedditThread]:
        """Search for more instances of a pain point — used by the
        corroborate step to count distinct authors beyond the initially
        harvested thread set, and optionally scoped site-wide via r/all.
        """
        target = self._reddit.subreddit(
            clean_subreddit_name(subreddit_name) if subreddit_name else "all"
        )
        return [
            self._to_thread(s) for s in target.search(query, limit=limit, sort="relevance")
        ]

    def _to_thread(self, submission: Any) -> RedditThread:
        cb = self._content_budget
        try:
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)
            top_comments = list(submission.comments)[: cb.max_comments_per_thread]
        except Exception:
            top_comments = []

        comments = [
            RedditComment(
                id=c.id,
                author=str(c.author) if c.author else None,
                body=(c.body or "")[: cb.max_chars_per_comment],
                score=c.score or 0,
                created_utc=c.created_utc,
                permalink=f"https://reddit.com{c.permalink}",
            )
            for c in top_comments
            if getattr(c, "body", None)
        ]

        return RedditThread(
            id=submission.id,
            subreddit=str(submission.subreddit),
            title=submission.title,
            selftext=(submission.selftext or "")[: cb.max_chars_per_source],
            author=str(submission.author) if submission.author else None,
            score=submission.score or 0,
            num_comments=submission.num_comments or 0,
            created_utc=submission.created_utc,
            permalink=f"https://reddit.com{submission.permalink}",
            flair=submission.link_flair_text,
            comments=comments,
        )
