#!/usr/bin/env python3
from __future__ import annotations

from contextlib import suppress

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


class StreamRequest:
    def __init__(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        log_path: Path,
        error_path: Path,
        timeout_s: int = 3600,
    ) -> None:
        self.url = url
        self.headers = headers
        self.payload = payload
        self.log_path = log_path
        self.error_path = error_path
        self.timeout_s = timeout_s
        self.first_token_event = threading.Event()
        self.finished_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._response = None
        self.error: Optional[str] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._response is not None:
            with suppress(Exception):
                self._response.close()
        self._thread.join(timeout=5)

    def wait_for_first_token(self, timeout_s: float) -> bool:
        return self.first_token_event.wait(timeout_s)

    def _run(self) -> None:
        data = json.dumps(self.payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                self._response = response
                with self.log_path.open("w", encoding="utf-8") as log_file:
                    while not self._stop_event.is_set():
                        line = response.readline()
                        if not line:
                            break
                        text = line.decode("utf-8", errors="replace")
                        log_file.write(text)
                        log_file.flush()
                        if text.startswith("data: ") and text.strip() != "data: [DONE]":
                            self.first_token_event.set()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.error_path.write_text(self.error + "\n", encoding="utf-8")
        finally:
            self.finished_event.set()


def http_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: Optional[dict[str, Any]] = None,
    timeout_s: int = 3600,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if payload is not None else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.getcode(), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def resolve_model(base_url: str, headers: dict[str, str]) -> str:
    status, body = http_json(f"{base_url}/v1/models", headers=headers, payload=None)
    if status != 200:
        raise RuntimeError(f"/v1/models failed with status {status}: {body.strip()}")
    payload = json.loads(body)
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no models")
    return models[0]["id"]


def build_request(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach to an already-running vLLM server, start one streaming request, "
            "wait for the first token, then use vLLM's native profiling endpoints."
        )
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server base URL, for example http://127.0.0.1:8000",
    )
    parser.add_argument("--mode", choices=("decode", "mtp"), default="decode")
    parser.add_argument(
        "--model",
        default=None,
        help="Model id. Auto-detected from /v1/models when omitted.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Inline prompt. Ignored when --prompt-file is set.",
    )
    parser.add_argument(
        "--prompt-file", default=None, help="Read the prompt from a file."
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--capture-seconds", type=float, default=10.0)
    parser.add_argument(
        "--out-dir", default=None, help="Local output directory for logs and metadata."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional bearer token. Falls back to OPENAI_API_KEY.",
    )
    parser.add_argument("--first-token-timeout", type=float, default=60.0)
    return parser.parse_args()


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    return (
        "Repeat the exact word token separated by spaces. "
        "Do not stop early. Keep emitting tokens until the token limit is reached."
    )


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    out_dir = Path(
        args.out_dir or f"/tmp/vllm-native-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = load_prompt(args)
    model = args.model or resolve_model(base_url, headers)
    request_body = build_request(model, prompt, args.max_tokens)

    request_path = out_dir / "request.json"
    stream_log = out_dir / "stream.log"
    error_log = out_dir / "request.error.txt"
    start_response = out_dir / "start_profile.response"
    stop_response = out_dir / "stop_profile.response"
    manifest_path = out_dir / "manifest.json"
    request_path.write_text(json.dumps(request_body, indent=2) + "\n", encoding="utf-8")

    print(f"output dir: {out_dir}")
    print(f"model: {model}")
    print(f"mode: {args.mode}")
    print("starting streaming request...")

    stream = StreamRequest(
        url=f"{base_url}/v1/completions",
        headers=headers,
        payload=request_body,
        log_path=stream_log,
        error_path=error_log,
    )
    stream.start()
    started_at = time.time()

    if not stream.wait_for_first_token(args.first_token_timeout):
        stream.stop()
        detail = ""
        if stream.error:
            detail = f" request error: {stream.error}"
        raise RuntimeError(f"timed out waiting for the first streamed token.{detail}")

    first_token_at = time.time()
    print("first streamed token observed; starting native profile...")

    status, body = http_json(f"{base_url}/start_profile", headers=headers, payload={})
    start_response.write_text(body, encoding="utf-8")
    if status == 404:
        stream.stop()
        raise RuntimeError(
            "/start_profile is not available on this server. "
            "For vLLM this usually means the server was not launched with --profiler-config."
        )
    if status != 200:
        stream.stop()
        raise RuntimeError(
            f"/start_profile failed with status {status}: {body.strip()}"
        )

    deadline = time.time() + args.capture_seconds
    while time.time() < deadline:
        if stream.finished_event.wait(timeout=0.2):
            break

    print("stopping native profile...")
    status, body = http_json(f"{base_url}/stop_profile", headers=headers, payload={})
    stop_response.write_text(body, encoding="utf-8")
    if status != 200:
        stream.stop()
        raise RuntimeError(f"/stop_profile failed with status {status}: {body.strip()}")

    finished_at = time.time()
    stream.stop()

    manifest = {
        "base_url": base_url,
        "mode": args.mode,
        "model": model,
        "out_dir": str(out_dir),
        "started_at_epoch_s": started_at,
        "first_token_at_epoch_s": first_token_at,
        "profile_stopped_at_epoch_s": finished_at,
        "capture_seconds": args.capture_seconds,
        "request_body": request_body,
        "notes": [
            "The background request was started before profiling and profiling began after the first streamed token.",
            "vLLM native HTTP profiling is window-based and does not provide a stage filter over the server API.",
            "For speculative servers, draft or verify regions should show up in the same trace window.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("native profiling completed.")
    print(f"stream log: {stream_log}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
