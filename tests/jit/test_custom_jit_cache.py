"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

CPU-only tests for custom jit-cache wheels: AOT module selection, the
FLASHINFER_JIT_USAGE_LOG recorder and scripts/custom_jit_cache.py recipes.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer import aot
from flashinfer.jit import core as jit_core

REPO_ROOT = Path(__file__).resolve().parents[2]


def _specs(*names):
    return [SimpleNamespace(name=name) for name in names]


@pytest.fixture
def custom_jit_cache():
    path = REPO_ROOT / "scripts" / "custom_jit_cache.py"
    spec = importlib.util.spec_from_file_location("_test_custom_jit_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_select_modules_combines_names_patterns_and_excludes():
    specs = _specs(
        "norm",
        "sampling",
        "batch_decode_dtype_q_bf16_head_dim_qk_128_use_swa_False",
        "batch_decode_dtype_q_bf16_head_dim_qk_128_use_swa_True",
        "batch_decode_dtype_q_f16_head_dim_qk_128_use_swa_False",
    )
    selection = aot.ModuleSelection(
        name="test",
        modules=("norm", "not_built_here"),
        include=("batch_decode_dtype_q_bf16_*", "gemm_*"),
        exclude=("*_use_swa_True",),
    )

    result = aot.select_modules(specs, selection)

    assert [spec.name for spec in result.selected] == [
        "norm",
        "batch_decode_dtype_q_bf16_head_dim_qk_128_use_swa_False",
    ]
    assert result.missing_modules == ["not_built_here"]
    assert result.unmatched_patterns == ["gemm_*"]


def test_module_selection_round_trips_and_rejects_unknown_keys():
    data = {
        "name": "r",
        "modules": ["norm"],
        "include": ["rope*"],
        "exclude": [],
        "aot_config": {"add_moe": False},
    }
    assert aot.ModuleSelection.from_dict(data).to_dict() == data
    with pytest.raises(ValueError, match="usage_logs"):
        aot.ModuleSelection.from_dict({"name": "r", "usage_logs": []})


def test_parse_aot_config_overrides():
    config = aot.parse_aot_config_overrides(
        {
            "f16_dtype": ["bfloat16"],
            "fa2_head_dim": [[128, 128]],
            "add_comm": False,
        }
    )
    assert config == {
        "f16_dtype": [torch.bfloat16],
        "fa2_head_dim": [(128, 128)],
        "add_comm": False,
    }
    assert aot.parse_aot_config_overrides(None) == {}
    with pytest.raises(ValueError, match="Unknown aot_config key"):
        aot.parse_aot_config_overrides({"add_everything": True})


def test_compile_and_package_modules_builds_only_the_selection(monkeypatch, tmp_path):
    built = {}
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "9.0a")
    monkeypatch.setattr(aot, "detect_sm_capabilities", lambda: {})
    monkeypatch.setattr(aot, "configure_aot_build_env", lambda *_: None)
    monkeypatch.setattr(
        aot,
        "gen_modules_from_config",
        lambda config, caps: _specs("norm", "rope", "sampling"),
    )
    monkeypatch.setattr(
        aot,
        "build_jit_specs",
        lambda specs, **_: built.setdefault("names", [s.name for s in specs]),
    )
    monkeypatch.setattr(aot, "copy_built_kernels", lambda specs, out_dir: None)

    names = aot.compile_and_package_modules(
        out_dir=tmp_path / "out",
        build_dir=tmp_path / "build",
        project_root=REPO_ROOT,
        selection=aot.ModuleSelection(name="t", include=("r*",), modules=("norm",)),
    )

    assert names == built["names"] == ["norm", "rope"]

    with pytest.raises(RuntimeError, match="matched no AOT modules"):
        aot.compile_and_package_modules(
            out_dir=None,
            build_dir=tmp_path / "build",
            project_root=REPO_ROOT,
            selection=aot.ModuleSelection(name="t", modules=("gemm",)),
        )


class _FakeSpec(jit_core.JitSpec):
    toolchain = "fake"

    def __init__(self, name):
        self.name = name

    lock_path = property(lambda self: Path("/nonexistent"))
    is_compiled = property(lambda self: True)

    def get_library_path(self):
        return Path("/nonexistent")

    def try_load(self):
        return object()

    def build(self):
        raise AssertionError("not reached")

    def load(self):
        raise AssertionError("not reached")


def test_usage_log_records_each_module_once(monkeypatch, tmp_path):
    log = tmp_path / "usage.jsonl"
    monkeypatch.setattr(jit_core, "_recorded_usage", set())
    monkeypatch.delenv("FLASHINFER_JIT_USAGE_LOG", raising=False)
    _FakeSpec("before").build_and_load()
    assert not log.exists()

    monkeypatch.setenv("FLASHINFER_JIT_USAGE_LOG", str(log))
    for name in ("norm", "rope", "norm"):
        _FakeSpec(name).build_and_load()

    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["name"] for r in records] == ["norm", "rope"]
    assert records[0]["toolchain"] == "fake"
    assert records[0]["aot"] is None
    assert "flashinfer_version" in records[0]


def _write_usage(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_recipe_from_usage_logs(custom_jit_cache, tmp_path):
    _write_usage(
        tmp_path / "usage.jsonl",
        [
            {"name": "norm", "toolchain": "nvcc", "cuda_archs": ["sm90a"]},
            {"name": "rope", "toolchain": "nvcc", "cuda_archs": ["sm90a"]},
            {"name": "gdn_cute", "toolchain": "cute_dsl", "cuda_archs": ["sm90a"]},
        ],
    )
    recipe_path = tmp_path / "recipes" / "r.json"
    recipe_path.parent.mkdir()

    assert (
        custom_jit_cache.main(
            [
                "init",
                "--name",
                "r",
                "--from-usage",
                str(tmp_path / "usage.jsonl"),
                "-o",
                str(recipe_path),
            ]
        )
        == 0
    )
    written = json.loads(recipe_path.read_text())
    assert written["cuda_archs"] == ["9.0a"]
    assert written["usage_logs"] == ["../usage.jsonl"]

    recipe = custom_jit_cache.load_recipe(recipe_path)
    recipe["modules"] = ["sampling", "norm"]
    selection, usage = custom_jit_cache.recipe_selection(recipe)
    assert selection["modules"] == ["norm", "rope", "sampling"]
    assert usage["runtime_only"] == {"gdn_cute": "cute_dsl"}
    assert custom_jit_cache.recipe_archs(recipe, ["sm100a", "10.0a"]) == ["10.0a"]


def test_recipe_rejects_unknown_keys_and_empty_selection(custom_jit_cache, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "r", "kernels": []}))
    with pytest.raises(SystemExit, match="unknown recipe keys"):
        custom_jit_cache.load_recipe(bad)

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"name": "r", "cuda_archs": ["9.0a"]}))
    with pytest.raises(SystemExit, match="selects nothing"):
        custom_jit_cache.recipe_selection(custom_jit_cache.load_recipe(empty))


def test_worker_applies_selection_per_arch(custom_jit_cache, monkeypatch):
    seen = {}

    def fake_gen(config, caps):
        seen["config"] = config
        return _specs("norm", "rope", "batch_decode_bf16")

    monkeypatch.setattr(aot, "gen_modules_from_config", fake_gen)
    monkeypatch.setattr(aot, "detect_sm_capabilities", lambda: {})
    monkeypatch.setattr(aot, "configure_aot_build_env", lambda *_: None)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "9.0a")

    result = custom_jit_cache._worker(
        "sm90a",
        {"name": "r", "modules": ["norm", "gone"], "include": ["batch_*"]},
        {"add_moe": False},
    )

    assert result == {
        "arch": "9.0a",
        "available": 3,
        "modules": ["norm", "batch_decode_bf16"],
        "missing_modules": ["gone"],
        "unmatched_patterns": [],
    }
    assert seen["config"]["add_moe"] is False
