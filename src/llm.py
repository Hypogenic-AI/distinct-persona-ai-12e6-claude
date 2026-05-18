"""Thin OpenAI client wrapper with retry, caching, and cost tracking."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

CACHE_DIR = Path(__file__).resolve().parent.parent / "logs" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Per-1M-token pricing (USD, approx Oct 2025); used only for tracking, not billing.
PRICING = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class UsageTracker:
    total_in: int = 0
    total_out: int = 0
    by_model: dict[str, tuple[int, int]] = field(default_factory=dict)

    def add(self, model: str, in_tok: int, out_tok: int) -> None:
        self.total_in += in_tok
        self.total_out += out_tok
        prev = self.by_model.get(model, (0, 0))
        self.by_model[model] = (prev[0] + in_tok, prev[1] + out_tok)

    def cost_usd(self) -> float:
        c = 0.0
        for m, (i, o) in self.by_model.items():
            pin, pout = PRICING.get(m, (0.0, 0.0))
            c += i * pin / 1e6 + o * pout / 1e6
        return c

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_in,
            "total_output_tokens": self.total_out,
            "by_model": {m: {"in": i, "out": o} for m, (i, o) in self.by_model.items()},
            "estimated_cost_usd": round(self.cost_usd(), 4),
        }


USAGE = UsageTracker()


def _cache_key(model: str, messages: list, params: dict) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "params": params},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception_type((
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )),
)
def _raw_call(model: str, messages: list, **kwargs):
    return _client.chat.completions.create(model=model, messages=messages, **kwargs)


def chat(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 400,
    top_p: float = 1.0,
    seed: int | None = 42,
    use_cache: bool = True,
) -> str:
    """Call the chat API, with on-disk caching keyed on (model, messages, params)."""
    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "seed": seed,
    }
    key = _cache_key(model, messages, params)
    cache_file = CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        with cache_file.open() as f:
            cached = json.load(f)
        # cache still tracks usage so we don't undercount
        USAGE.add(model, cached.get("in", 0), cached.get("out", 0))
        return cached["content"]

    resp = _raw_call(model, messages, **params)
    content = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    USAGE.add(model, in_tok, out_tok)
    if use_cache:
        with cache_file.open("w") as f:
            json.dump({"content": content, "in": in_tok, "out": out_tok}, f)
    return content


def reset_usage() -> None:
    global USAGE
    USAGE = UsageTracker()


def build_messages(system: Optional[str], turns: list[dict]) -> list[dict]:
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    for t in turns:
        # SycophancyEval stores {"type": "human"|"ai", "content": ...}
        role = {"human": "user", "ai": "assistant"}.get(t.get("type", t.get("role")), t.get("role"))
        msgs.append({"role": role, "content": t["content"]})
    return msgs
