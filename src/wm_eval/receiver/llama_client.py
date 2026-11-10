"""Thin client for llama.cpp's OpenAI-compatible server (`llama-server`).

Returns raw payloads so the caller can persist exactly what was sent and received.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from openai import OpenAI

from wm_eval.config import ReceiverSpec


class LlamaClient:
    def __init__(self, spec: ReceiverSpec, client: OpenAI | None = None):
        self.spec = spec
        self.root_url = spec.base_url.rstrip("/").removesuffix("/v1")
        self.client = client or OpenAI(
            base_url=spec.base_url,
            api_key=os.environ.get(spec.api_key_env) or "not-needed",
            timeout=spec.timeout_s,
            max_retries=0,  # retries are the runner's job (resume), so failures stay visible
        )

    def health(self, timeout: float = 3.0) -> dict[str, Any]:
        """`/health` and `/props` (model path, build). Never raises."""
        out: dict[str, Any] = {"base_url": self.spec.base_url}
        try:
            r = httpx.get(f"{self.root_url}/health", timeout=timeout)
            out["ok"] = r.status_code == 200
            out["health"] = r.json() if r.content else None
        except Exception as e:
            return {**out, "ok": False, "error": repr(e)}
        try:
            props = httpx.get(f"{self.root_url}/props", timeout=timeout).json()
            out["model_path"] = props.get("model_path")
            out["build_info"] = props.get("build_info")
            out["n_ctx"] = (props.get("default_generation_settings") or {}).get("n_ctx")
        except Exception as e:  # older servers may lack /props
            out["props_error"] = repr(e)
        return out

    def build_request(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if tools:
            req["tools"] = tools
        if self.spec.chat_template_kwargs:
            req["chat_template_kwargs"] = self.spec.chat_template_kwargs
        return req

    def send(self, request: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """POST a chat completion. Returns (raw response dict, seconds). Raises on failure."""
        body = dict(request)
        extra = {k: body.pop(k) for k in ("chat_template_kwargs",) if k in body}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**body, extra_body=extra or None)
        return resp.model_dump(), time.perf_counter() - t0
