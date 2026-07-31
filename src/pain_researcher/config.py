"""Configuration for the pain-point researcher.

Two kinds of configuration are deliberately kept separate:

- **Tunable settings** (models, thresholds, phrase lists, scoring weights) live in
  `settings.yaml` and are loaded into the Pydantic models below. Retuning
  aggressiveness or swapping which Gemma model backs a role is a YAML edit,
  never a code change.
- **Run options and secrets** (discovery mode, API keys) come from environment
  variables / the LangGraph `RunnableConfig`, following the same
  override pattern used in `ollama_deep_researcher.configuration.Configuration`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, model_validator

DEFAULT_SETTINGS_PATH = Path(__file__).parent / "settings.yaml"


# --------------------------------------------------------------------------
# Tunable settings (settings.yaml)
# --------------------------------------------------------------------------


class ModelProfile(BaseModel):
    """Rate-limit and capability profile for one LLM backend.

    Swapping models later means adding/editing a profile here and
    repointing a role at it in `RolesConfig` — no code changes.
    """

    model_id: str
    rpm_limit: int = Field(gt=0)
    tpm_limit: int = Field(gt=0)
    rpd_limit: int = Field(gt=0)
    context_window: int = Field(gt=0)
    max_input_tokens_per_call: int = Field(gt=0)
    safety_margin_pct: float = Field(gt=0, le=100)
    target_rpm: float = Field(gt=0)
    supports_tool_calling: bool = False
    supports_system_role: bool = False

    @model_validator(mode="after")
    def _check_internal_coherence(self) -> "ModelProfile":
        if self.max_input_tokens_per_call > self.tpm_limit:
            raise ValueError(
                f"model '{self.model_id}': max_input_tokens_per_call "
                f"({self.max_input_tokens_per_call}) exceeds tpm_limit ({self.tpm_limit})"
            )
        if self.target_rpm > self.rpm_limit:
            raise ValueError(
                f"model '{self.model_id}': target_rpm ({self.target_rpm}) "
                f"exceeds rpm_limit ({self.rpm_limit})"
            )
        return self

    @property
    def effective_tpm(self) -> float:
        """Token-per-minute budget after applying the safety margin."""
        return self.tpm_limit * (self.safety_margin_pct / 100)

    @property
    def effective_rpm(self) -> float:
        """Request-per-minute budget after applying the safety margin, capped at target_rpm."""
        return min(self.target_rpm, self.rpm_limit * (self.safety_margin_pct / 100))

    @property
    def effective_rpd(self) -> float:
        """Request-per-day budget after applying the safety margin."""
        return self.rpd_limit * (self.safety_margin_pct / 100)


class RolesConfig(BaseModel):
    """Maps pipeline roles to entries in the `models` block."""

    cheap: str
    judge: str


class DiscoveryConfig(BaseModel):
    mode: Literal["autonomous", "seed", "watchlist"] = "autonomous"
    seed_niche: Optional[str] = None
    subreddit_watchlist: list[str] = Field(default_factory=list)
    max_niches_per_run: int = Field(default=5, gt=0)
    max_subreddits_per_niche: int = Field(default=6, gt=0)
    max_subreddits_total: int = Field(default=20, gt=0)
    min_subreddit_subscribers: int = Field(default=1000, ge=0)


class ContentBudgetConfig(BaseModel):
    max_threads_per_subreddit: int = Field(gt=0)
    max_comments_per_thread: int = Field(gt=0)
    max_chars_per_comment: int = Field(gt=0)
    max_threads_per_extraction_call: int = Field(gt=0)
    max_sources_per_call: int = Field(gt=0)
    max_chars_per_source: int = Field(gt=0)
    max_competitor_pages: int = Field(gt=0)
    max_chars_per_competitor_page: int = Field(gt=0)


class PrefilterConfig(BaseModel):
    min_upvotes: int = Field(ge=0)
    min_comments: int = Field(ge=0)
    max_age_days: int = Field(gt=0)
    drop_flairs: list[str] = Field(default_factory=list)
    complaint_phrases: list[str] = Field(default_factory=list)
    wtp_phrases: list[str] = Field(default_factory=list)


class CandidateGatingConfig(BaseModel):
    min_distinct_authors: int = Field(gt=0)
    min_distinct_threads: int = Field(gt=0)
    max_candidates_to_validate: int = Field(gt=0)


class ScoringConfig(BaseModel):
    weight_distinct_authors: float
    weight_distinct_threads: float
    weight_subreddit_spread: float
    weight_engagement: float
    weight_severity: float
    weight_wtp_signal: float
    weight_solution_gap: float
    weight_buildability: float
    penalty_strong_competitor: float
    penalty_hard_blocker: float


class RetryConfig(BaseModel):
    backoff_base_seconds: float = Field(gt=0)
    backoff_max_seconds: float = Field(gt=0)
    max_retries: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> "RetryConfig":
        if self.backoff_base_seconds > self.backoff_max_seconds:
            raise ValueError("backoff_base_seconds must be <= backoff_max_seconds")
        return self


class OutputConfig(BaseModel):
    output_dir: str = "research_output"
    top_n_briefs: int = Field(default=10, gt=0)


class CheckpointConfig(BaseModel):
    db_path: str = "pain_researcher_checkpoints.sqlite"


class PainResearcherSettings(BaseModel):
    """Root of settings.yaml."""

    models: dict[str, ModelProfile]
    roles: RolesConfig
    discovery: DiscoveryConfig
    content_budget: ContentBudgetConfig
    prefilter: PrefilterConfig
    candidate_gating: CandidateGatingConfig
    scoring: ScoringConfig
    retry: RetryConfig
    output: OutputConfig
    checkpoint: CheckpointConfig

    @model_validator(mode="after")
    def _check_roles_resolve(self) -> "PainResearcherSettings":
        for role_name, model_key in self.roles.model_dump().items():
            if model_key not in self.models:
                raise ValueError(
                    f"roles.{role_name} references model '{model_key}', "
                    f"which is not defined in `models` "
                    f"(available: {sorted(self.models)})"
                )
        return self

    def model_for_role(self, role: Literal["cheap", "judge"]) -> ModelProfile:
        key = getattr(self.roles, role)
        return self.models[key]


def load_settings(path: Optional[str | Path] = None) -> PainResearcherSettings:
    """Load and validate settings.yaml.

    Raises pydantic.ValidationError immediately if the config is internally
    inconsistent (e.g. a per-call token cap above the TPM limit, or a role
    pointing at a model profile that doesn't exist) — fail at startup, not
    partway through a run that already spent quota.
    """
    resolved = Path(path) if path else Path(
        os.environ.get("PAIN_RESEARCHER_SETTINGS_PATH", DEFAULT_SETTINGS_PATH)
    )
    with open(resolved, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PainResearcherSettings(**raw)


# --------------------------------------------------------------------------
# Run options (env vars / RunnableConfig) — mirrors the override pattern in
# ollama_deep_researcher.configuration.Configuration.from_runnable_config
# --------------------------------------------------------------------------


class RunOptions(BaseModel):
    """Per-run knobs, overridable via env var or LangGraph Studio's config tab."""

    discovery_mode: Optional[Literal["autonomous", "seed", "watchlist"]] = None
    seed_niche: Optional[str] = None
    subreddit_watchlist: Optional[list[str]] = None
    dry_run: bool = False
    settings_path: Optional[str] = None

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "RunOptions":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        def _get(name: str) -> Any:
            env_val = os.environ.get(f"PAIN_RESEARCHER_{name.upper()}")
            if env_val is not None:
                return env_val
            return configurable.get(name)

        watchlist_raw = _get("subreddit_watchlist")
        watchlist = None
        if watchlist_raw:
            if isinstance(watchlist_raw, str):
                watchlist = [s.strip() for s in watchlist_raw.split(",") if s.strip()]
            else:
                watchlist = list(watchlist_raw)

        dry_run_raw = _get("dry_run")
        dry_run = str(dry_run_raw).lower() in ("1", "true", "yes") if dry_run_raw is not None else False

        raw_values: dict[str, Any] = {
            "discovery_mode": _get("discovery_mode"),
            "seed_niche": _get("seed_niche"),
            "subreddit_watchlist": watchlist,
            "dry_run": dry_run,
            "settings_path": _get("settings_path"),
        }
        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)


# --------------------------------------------------------------------------
# Secrets (env vars only — never in settings.yaml)
# --------------------------------------------------------------------------


class Credentials(BaseModel):
    google_api_key: str
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str

    @classmethod
    def from_env(cls) -> "Credentials":
        missing = [
            name
            for name, env in [
                ("GOOGLE_API_KEY", "GOOGLE_API_KEY"),
                ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_ID"),
                ("REDDIT_CLIENT_SECRET", "REDDIT_CLIENT_SECRET"),
                ("REDDIT_USER_AGENT", "REDDIT_USER_AGENT"),
            ]
            if not os.environ.get(env)
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill these in."
            )
        return cls(
            google_api_key=os.environ["GOOGLE_API_KEY"],
            reddit_client_id=os.environ["REDDIT_CLIENT_ID"],
            reddit_client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            reddit_user_agent=os.environ["REDDIT_USER_AGENT"],
        )


# --------------------------------------------------------------------------
# Combined runtime config
# --------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Everything a graph run needs: tunable settings + this run's options."""

    settings: PainResearcherSettings
    options: RunOptions

    model_config = {"arbitrary_types_allowed": True}

    @property
    def effective_discovery_mode(self) -> str:
        return self.options.discovery_mode or self.settings.discovery.mode

    @property
    def effective_seed_niche(self) -> Optional[str]:
        return self.options.seed_niche or self.settings.discovery.seed_niche

    @property
    def effective_subreddit_watchlist(self) -> list[str]:
        return self.options.subreddit_watchlist or self.settings.discovery.subreddit_watchlist

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "RuntimeConfig":
        options = RunOptions.from_runnable_config(config)
        settings = load_settings(options.settings_path)
        return cls(settings=settings, options=options)


@lru_cache(maxsize=8)
def _cached_settings(path_str: str) -> PainResearcherSettings:
    return load_settings(path_str)


def get_settings(path: Optional[str | Path] = None) -> PainResearcherSettings:
    """Cached settings load, keyed by resolved path — avoids re-parsing YAML per node."""
    resolved = str(Path(path) if path else Path(
        os.environ.get("PAIN_RESEARCHER_SETTINGS_PATH", DEFAULT_SETTINGS_PATH)
    ))
    return _cached_settings(resolved)
