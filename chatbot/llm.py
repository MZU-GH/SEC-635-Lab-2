"""Chat-completions HTTP client for the two llama.cpp servers (LLM and guard).

The client talks to a local llama.cpp server; set LLM_URL/GUARD_URL to point at the lab GPU
server.

The lab brief asks for llama3.1:8b and llama-guard3:1b under Ollama. Ollama is not
installed and git.lab cannot host an 8B model (no GPU, ~8 GiB free), so both models run
under llama.cpp and are served over the same /v1/chat/completions contract.
Model identity is preserved; only the server differs (report/REPRODUCING.md, section 14).

Sampling is pinned explicitly. Chapter 4 never sends an `options` dict, so its runs
inherit whatever the server defaults to; for a red-team harness that reports success
rates over 10 runs, an unpinned temperature makes the number unreproducible.
"""

from __future__ import annotations

import os

import requests

# Endpoints: a local llama.cpp server; set LLM_URL/GUARD_URL (base URLs, no path) to point
# at the lab GPU server. The /v1/chat/completions suffix is appended here.
_PATH = "/v1/chat/completions"
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:11434").rstrip("/") + _PATH
GUARD_URL = os.environ.get("GUARD_URL", "http://127.0.0.1:11435").rstrip("/") + _PATH

LLM_MODEL = "/root/models-gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
GUARD_MODEL = "/root/models-gguf/Llama-Guard-3-1B.Q4_K_M.gguf"


def chat(messages: list[dict], *, temperature: float = 0.7, top_p: float = 0.9,
         seed: int | None = None, max_tokens: int = 256, url: str = LLM_URL,
         model: str = LLM_MODEL, timeout: int = 180) -> str:
    """One chat completion. Returns the assistant text, or an error marker.

    Failure is returned as a string rather than raised because the red-team harness
    runs hundreds of calls and one timeout must not discard the rest of a suite; the
    caller counts them via the `ERROR:` prefix.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 - see docstring
        return f"ERROR: {type(exc).__name__}: {exc}"


def ping() -> dict:
    """Confirm both endpoints answer, before a suite starts."""
    out = {}
    for name, url, model in (("llm", LLM_URL, LLM_MODEL), ("guard", GUARD_URL, GUARD_MODEL)):
        try:
            base = url.rsplit("/v1/", 1)[0]
            r = requests.get(f"{base}/v1/models", timeout=10)
            r.raise_for_status()
            out[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unreachable: {type(exc).__name__}"
    out["llm_model"] = LLM_MODEL
    out["guard_model"] = GUARD_MODEL
    return out
