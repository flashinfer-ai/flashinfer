# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of serial header acquisition before parallel spec lint."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.cpu_only


@pytest.fixture
def prefetch(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_test_prefetch_headers", REPO_ROOT / "scripts/prefetch_jit_spec_headers.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Do not import FlashInfer or initialize a CUDA context in these CPU tests.
    loader = ModuleType("flashinfer.jit.cubin_loader")
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    return module, loader


def test_prefetch_uses_pinned_manifest_then_each_header_hash(prefetch):
    module, loader = prefetch
    objects = {
        "pinned/checksums.txt": ("manifest-hash", b"verified manifest"),
        "pinned/include/flashInferMetaInfo.h": ("meta-hash", b"metadata"),
        "pinned/include/export/trtllm/gen/Enums.h": ("export-hash", b"export"),
    }
    hashes = {
        "flashInferMetaInfo.h": "meta-hash",
        "export/trtllm/gen/Enums.h": "export-hash",
    }
    calls = []

    def get_artifact(path, digest):
        expected, contents = objects[path]
        assert digest == expected
        calls.append(path)
        return contents

    def get_meta_hash(manifest, header):
        assert manifest == b"verified manifest"
        return hashes[header]

    loader.get_artifact = get_artifact
    loader.get_meta_hash = get_meta_hash
    assert module.prefetch_group("pinned/", "manifest-hash", list(hashes)) == 2
    assert calls == list(objects)


@pytest.mark.parametrize("failure", ["manifest", "header", "checksum"])
def test_prefetch_stops_on_failed_or_unlisted_artifacts(prefetch, failure):
    module, loader = prefetch
    calls = []

    def get_artifact(path, digest):
        calls.append(path)
        if failure == "manifest" or path.endswith("bad.h"):
            return b""
        return b"manifest"

    def get_meta_hash(manifest, header):
        if failure == "checksum":
            raise ValueError("missing manifest entry")
        return "header-hash"

    loader.get_artifact = get_artifact
    loader.get_meta_hash = get_meta_hash
    with pytest.raises((RuntimeError, ValueError)):
        module.prefetch_group("pinned", "manifest-hash", ["bad.h", "later.h"])
    assert "pinned/include/later.h" not in calls
    assert len(calls) == (2 if failure == "header" else 1)


@pytest.mark.parametrize(
    ("concurrency", "providers", "prefetch_status", "expected_status"),
    [
        (3, ["10.0a", "10.3a", "12.0f"], 0, 0),
        (3, ["10.0a", "10.3a", "12.0f"], 42, 1),
        (1, ["10.0a", "10.3a"], 42, 0),
        (3, ["10.0a"], 42, 0),
        (3, [], 42, 0),
        (0, ["10.0a", "10.3a"], 0, 1),
    ],
)
def test_pool_prefetch_order_and_failure(
    tmp_path, concurrency, providers, prefetch_status, expected_status
):
    events = tmp_path / "events"
    env = {
        **os.environ,
        "PREFETCH_EVENTS": str(events),
        "PREFETCH_STATUS": str(prefetch_status),
    }
    # Stub external Python, but exercise the real shell entrypoint and pool.
    script = r"""
source "$1"
shift
python3() {
  if [[ "$1" == */prefetch_jit_spec_headers.py ]]; then
    echo prefetch >> "$PREFETCH_EVENTS"
    return "$PREFETCH_STATUS"
  fi
  echo "lint:$3" >> "$PREFETCH_EVENTS"
}
run_provider_spec_lint "$@"
"""
    result = subprocess.run(
        [
            "bash",
            "-eo",
            "pipefail",
            "-c",
            script,
            "test",
            str(REPO_ROOT / "scripts/jit_cache_build_common.sh"),
            str(concurrency),
            *providers,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    lines = events.read_text().splitlines() if events.exists() else []
    if concurrency == 0:
        assert lines == []
    elif concurrency > 1 and len(providers) > 1:
        assert lines[0] == "prefetch"
        if prefetch_status:
            assert lines == ["prefetch"]
            assert "Failed to prefetch" in result.stdout
        else:
            assert sorted(lines[1:]) == sorted(f"lint:{p}" for p in providers)
    else:
        assert sorted(lines) == sorted(f"lint:{p}" for p in providers)
