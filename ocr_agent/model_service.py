from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request

from .data_models import BackendConfig


class VisionLanguageBackend(ABC):
    """Common interface for image-aware language model backends."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._last_request_time = 0.0

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, image_paths: List[str]) -> str:
        raise NotImplementedError

    def _wait_for_rate_limit(self) -> None:
        interval = max(0.0, float(self.config.request_interval_seconds))
        elapsed = time.monotonic() - self._last_request_time
        if interval > elapsed:
            time.sleep(interval - elapsed)
        self._last_request_time = time.monotonic()

    def _request_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(max(1, self.config.max_retries + 1)):
            self._wait_for_rate_limit()
            req = request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json", **headers})
            try:
                with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                    value = json.loads(raw)
                    if isinstance(value, dict):
                        return value
                    raise ValueError("backend response must be a JSON object")
            except (error.HTTPError, error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < max(1, self.config.max_retries + 1):
                    time.sleep(min(30.0, 2.0 ** attempt))
        raise RuntimeError(f"model request failed after retries: {last_error}") from last_error


class MockBackend(VisionLanguageBackend):
    """Deterministic backend for smoke tests and offline pipeline checks."""

    def generate(self, system_prompt: str, user_prompt: str, image_paths: List[str]) -> str:
        prompt = user_prompt.lower()
        if "verify a proposed" in prompt:
            return json.dumps({"decision": "REJECT", "reason": "mock backend"})
        if "check whether an ocr answer" in prompt:
            return json.dumps({"decision": "KEEP", "evidence": [], "diagnosis": "mock backend", "action": "recheck visible text"})
        if "independently solve" in prompt:
            return json.dumps({"answer": "mock answer", "evidence": "mock image evidence"})
        if "refine the previous answer" in prompt:
            return json.dumps({"answer": "mock answer", "evidence": "mock image evidence"})
        if '"reasoning"' in prompt:
            return json.dumps({"reasoning": "mock reasoning", "answer": "mock answer"})
        return "mock answer"


def _image_data_url(path: str) -> str:
    image_path = Path(path)
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    return f"data:{mime};base64,{data}"


class OpenAICompatibleBackend(VisionLanguageBackend):
    def generate(self, system_prompt: str, user_prompt: str, image_paths: List[str]) -> str:
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(image_path)}})
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
        }
        if self.config.random_seed is not None:
            payload["seed"] = self.config.random_seed

        if self.config.api_protocol == "responses":
            payload = {
                "model": self.config.model_name,
                "input": [
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_prompt}] + [
                        {"type": "input_image", "image_url": _image_data_url(path)} for path in image_paths
                    ]},
                ],
                "temperature": self.config.temperature,
            }
            if self.config.random_seed is not None:
                payload["seed"] = self.config.random_seed
            endpoint = f"{self.config.base_url.rstrip('/')}/responses"
        else:
            endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"

        headers = {"Authorization": f"Bearer {self.config.api_key or ''}"}
        response = self._request_json(endpoint, payload, headers)
        if self.config.api_protocol == "responses":
            output_text = response.get("output_text")
            if isinstance(output_text, str):
                return output_text
            for item in response.get("output", []) if isinstance(response.get("output"), list) else []:
                for part in item.get("content", []) if isinstance(item, dict) and isinstance(item.get("content"), list) else []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"]
            return ""
        choices = response.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message", {})
        value = message.get("content") if isinstance(message, dict) else ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(str(part.get("text", "")) for part in value if isinstance(part, dict))
        return str(value or "")


class GeminiBackend(VisionLanguageBackend):
    def generate(self, system_prompt: str, user_prompt: str, image_paths: List[str]) -> str:
        parts: List[Dict[str, Any]] = [{"text": f"{system_prompt}\n\n{user_prompt}"}]
        for image_path in image_paths:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            parts.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(path.read_bytes()).decode("ascii")}})
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"temperature": self.config.temperature}}
        endpoint = f"{self.config.base_url.rstrip('/')}/models/{self.config.model_name}:generateContent?key={self.config.api_key or ''}"
        response = self._request_json(endpoint, payload, {})
        candidates = response.get("candidates", [])
        if not candidates or not isinstance(candidates[0], dict):
            return ""
        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        return "".join(str(part.get("text", "")) for part in parts if isinstance(part, dict))


def build_backend(config: BackendConfig) -> VisionLanguageBackend:
    backend_type = config.backend_type.lower().strip()
    if backend_type == "mock":
        return MockBackend(config)
    if backend_type == "gemini":
        return GeminiBackend(config)
    if backend_type in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleBackend(config)
    raise ValueError(f"unsupported backend type: {config.backend_type}")
