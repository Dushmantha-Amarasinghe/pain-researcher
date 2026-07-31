"""Gemma 4 client and role-based router.

Every LLM call in the pipeline goes through `LLMRouter.generate_structured`,
which ties together three concerns that must never be handled ad hoc in a
graph node:

1. **Role -> model indirection** — nodes ask for `"cheap"` or `"judge"`,
   never a specific model name. Retargeting a role to a different model is
   a `settings.yaml` edit (see `config.RolesConfig`).
2. **Quota enforcement** — every call is paced and budgeted through
   `quota.QuotaLimiter` before it happens, not caught after a 429.
3. **Capability-flag-driven structured output** — `supports_tool_calling`
   and `supports_system_role` on the model profile decide *how* a call is
   shaped, so moving to a model with proper tool calling later is a config
   flip, not a rewrite. Gemma 4 via the Gemini API has historically lacked
   reliable function calling and a true system role, so both flags default
   to False and callers get defensive prompt-based JSON parsing — the
   salvage strategy is adapted from `ollama_deep_researcher/lmstudio.py`.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from pain_researcher.config import Credentials, ModelProfile, RetryConfig, RuntimeConfig
from pain_researcher.quota import QuotaRegistry, estimate_tokens, with_backoff
from pain_researcher.state import UsageRecord

Role = Literal["cheap", "judge"]


class LLMParseError(RuntimeError):
    """Raised when a model's response can't be salvaged into valid JSON."""


class ContentTooLargeError(ValueError):
    """Raised when a caller hands the router more content than the model's
    per-call token cap allows.

    Deliberately fails loudly instead of silently truncating — truncation
    is prefilter.py's job, done *before* content reaches the LLM layer, so
    it can be done intelligently (drop whole low-value comments) rather
    than blindly cutting off whatever happens to be last (which could
    sever the JSON-format instructions at the end of a prompt).
    """


@dataclass
class LLMCallResult:
    data: dict[str, Any]
    usage: UsageRecord
    raw_text: str


def _salvage_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response that may include
    preamble, markdown fences, or trailing commentary.

    Same first-`{`-to-last-`}` strategy as the tool-free JSON mode in
    `ollama_deep_researcher/lmstudio.py`, generalized into a standalone
    helper since it's needed for every prompt-based structured call here.
    """
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        raise LLMParseError(f"No JSON object found in response: {text[:300]!r}")
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"Malformed JSON in response: {exc}. Text: {candidate[:300]!r}") from exc


def _is_retryable_error(exc: Exception) -> bool:
    """Best-effort 429/rate-limit/transient-server-error detection.

    Avoids a hard dependency on google-api-core's exact exception classes
    (which have moved between packages before) by pattern-matching the
    exception's type name and message instead.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    signals = ("resourceexhausted", "429", "rate limit", "quota", "503", "unavailable", "deadline")
    return any(s in name or s in text for s in signals)


class LLMRouter:
    """Routes role-based LLM calls to the right Gemma model, paced and quota-checked."""

    def __init__(self, runtime_config: RuntimeConfig, credentials: Credentials):
        self._settings = runtime_config.settings
        self._credentials = credentials
        self._quota = QuotaRegistry()
        self._clients_lock = threading.Lock()
        self._clients: dict[str, Any] = {}

    def _profile_for(self, role: Role) -> tuple[str, ModelProfile]:
        model_key = getattr(self._settings.roles, role)
        return model_key, self._settings.models[model_key]

    def _client_for(self, model_key: str, profile: ModelProfile):
        with self._clients_lock:
            if model_key not in self._clients:
                # Imported lazily so the rest of the package works without
                # langchain-google-genai installed until an LLM call is made.
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._clients[model_key] = ChatGoogleGenerativeAI(
                    model=profile.model_id,
                    google_api_key=self._credentials.google_api_key,
                    temperature=0,
                )
            return self._clients[model_key]

    def _build_messages(
        self, profile: ModelProfile, system_prompt: str, user_prompt: str
    ) -> list[BaseMessage]:
        if profile.supports_system_role:
            return [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        # Fold system content into the human turn — some Gemma API paths
        # don't reliably honor a separate system role.
        return [HumanMessage(content=f"{system_prompt}\n\n{user_prompt}")]

    def generate_structured(
        self,
        role: Role,
        node_name: str,
        system_prompt: str,
        user_prompt: str,
        tool_class: Optional[Type[BaseModel]] = None,
    ) -> LLMCallResult:
        """Make one structured-output LLM call for the given role.

        If `tool_class` is provided and the model profile has
        `supports_tool_calling: true`, uses LangChain tool binding.
        Otherwise falls back to prompt-based JSON with salvage parsing —
        the path Gemma 4 needs today.
        """
        model_key, profile = self._profile_for(role)
        limiter = self._quota.get(model_key, profile)
        client = self._client_for(model_key, profile)

        messages = self._build_messages(profile, system_prompt, user_prompt)
        combined_text = "\n".join(m.content for m in messages if isinstance(m.content, str))
        estimated_input = estimate_tokens(combined_text)

        if estimated_input > profile.max_input_tokens_per_call:
            raise ContentTooLargeError(
                f"[{node_name}] prompt for role '{role}' ({model_key}) is ~{estimated_input} "
                f"input tokens, over the configured cap of {profile.max_input_tokens_per_call}. "
                "Reduce content_budget settings or trim upstream before calling the LLM."
            )

        use_tools = profile.supports_tool_calling and tool_class is not None

        def _call():
            if use_tools:
                bound = client.bind_tools([tool_class])
                return bound.invoke(messages)
            return client.invoke(messages)

        retry_cfg: RetryConfig = self._settings.retry
        limiter.reserve(estimated_input)
        result = with_backoff(_call, retry_cfg, _is_retryable_error)

        usage_meta = getattr(result, "usage_metadata", None) or {}
        actual_input = usage_meta.get("input_tokens", estimated_input)
        actual_output = usage_meta.get("output_tokens", estimate_tokens(str(result.content)))
        limiter.record(actual_input, actual_output)

        usage = UsageRecord(
            node=node_name,
            model_key=model_key,
            input_tokens=actual_input,
            output_tokens=actual_output,
            timestamp=time.time(),
        )

        if use_tools:
            if not getattr(result, "tool_calls", None):
                raise LLMParseError(f"[{node_name}] model returned no tool call for {tool_class}")
            data = result.tool_calls[0]["args"]
            return LLMCallResult(data=data, usage=usage, raw_text=str(result.content))

        raw_text = result.content if isinstance(result.content, str) else str(result.content)
        data = _salvage_json(raw_text)
        return LLMCallResult(data=data, usage=usage, raw_text=raw_text)

    def project_call(
        self,
        role: Role,
        node_name: str,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMCallResult:
        """Estimate what a call would cost without making it — the engine
        behind dry-run mode.

        Reuses the same message-building and token-budget guard as a real
        call, so a dry run surfaces `ContentTooLargeError` in exactly the
        same place a live run would. Returns an empty `data` dict since no
        real response exists; callers must treat dry-run results as
        usage-projection only, not as real pipeline output.
        """
        model_key, profile = self._profile_for(role)
        messages = self._build_messages(profile, system_prompt, user_prompt)
        combined_text = "\n".join(m.content for m in messages if isinstance(m.content, str))
        estimated_input = estimate_tokens(combined_text)

        if estimated_input > profile.max_input_tokens_per_call:
            raise ContentTooLargeError(
                f"[{node_name}] prompt for role '{role}' ({model_key}) is ~{estimated_input} "
                f"input tokens, over the configured cap of {profile.max_input_tokens_per_call}. "
                "Reduce content_budget settings or trim upstream before calling the LLM."
            )

        # Rough output estimate: structured-JSON responses here are short
        # relative to their input (extraction/judging, not long-form text).
        estimated_output = min(400, max(80, estimated_input // 4))
        usage = UsageRecord(
            node=node_name,
            model_key=model_key,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            timestamp=time.time(),
        )
        return LLMCallResult(data={}, usage=usage, raw_text="")

    def usage_report(self) -> list[dict]:
        return self._quota.usage_report()
