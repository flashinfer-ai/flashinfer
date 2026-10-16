from __future__ import annotations

import ast
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts.pr_checks.targeted_lane_scope import coverage


ROOT = Path(__file__).resolve().parents[2]
PREFILL = "tests/attention/test_batch_prefill_kernels.py"
SCRIPTS = [ROOT / f"scripts/task_jit_run_tests_part{i}.sh" for i in range(1, 6)]


def _commands(script):
    return [
        shlex.split(line)
        for line in script.read_text().splitlines()
        if line.startswith("pytest ")
    ]


def test_a10g_coverage_and_t4_scope(tmp_path):
    counts = Counter(cmd[2] for script in SCRIPTS for cmd in _commands(script))
    expected = {
        "attention": [
            "batch_decode_kernels",
            "batch_prefill_kernels",
            "block_sparse",
            "deepseek_mla",
            "logits_cap",
            "mla_page",
            "rope",
            "sliding_window",
            "tensor_cores_decode",
        ],
        "utils": [
            "jit_example",
            "jit_warmup",
            "logits_processor",
            "norm",
            "quantization",
            "sampling",
            "topk",
        ],
        "cli": ["cli_cmds", "cli_cmds_gpu"],
        "moe": ["bgmv_moe", "bgmv_moe_lora_delta"],
        "gemm": ["group_gemm"],
        "experimental": ["experimental_api"],
    }
    files = {
        f"tests/{folder}/test_{name}.py"
        for folder, names in expected.items()
        for name in names
    }
    files.add("tests/test_artifacts.py")
    assert set(counts) == files
    assert counts.pop(PREFILL) == 3
    assert set(counts.values()) == {1}
    assert set(coverage(SCRIPTS)) == files
    assert coverage([ROOT / "scripts/task_jit_run_tests_t4.sh"]) == [
        "tests/utils/test_sampling.py",
        "tests/utils/test_topk.py",
    ]
    assert (
        sum(s.read_text().count("bash tests/moe_ep/run_tests.sh unit") for s in SCRIPTS)
        == 1
    )

    # Exercise pytest's actual -k matching on every current function name.
    functions = [
        n.name
        for n in ast.parse((ROOT / PREFILL).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    sample = tmp_path / "test_prefill.py"
    sample.write_text("\n".join(f"def {name}(): pass" for name in functions))
    selected = Counter()
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    for script in SCRIPTS:
        for command in _commands(script):
            if command[2] != PREFILL:
                continue
            expression = command[command.index("-k") + 1]
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--collect-only",
                    "-q",
                    str(sample),
                    "-k",
                    expression,
                ],
                cwd=tmp_path,
                env=env,
                text=True,
                capture_output=True,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            selected.update(
                line.split("::", 1)[1]
                for line in result.stdout.splitlines()
                if line.startswith("test_prefill.py::")
            )
    assert selected == Counter(functions)


def _load_prebuild(monkeypatch):
    torch = ModuleType("torch")
    for name in (
        "float16",
        "bfloat16",
        "int32",
        "uint8",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        setattr(torch, name, name)
    torch.half = torch.float16
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = []
    prefill = ModuleType("flashinfer.prefill")
    for kind in ("batch", "single"):

        def generate(*args, kind=kind, **kwargs):
            calls.append((kind, args))
            return SimpleNamespace(name=str((kind, args)))

        setattr(prefill, f"gen_{kind}_prefill_module", generate)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", prefill)
    utils = ModuleType("flashinfer.utils")
    utils.get_compute_capability = lambda device: (8, 6)
    utils.is_sm90a_supported = lambda device: False
    monkeypatch.setitem(sys.modules, "flashinfer.utils", utils)
    spec = importlib.util.spec_from_file_location(
        "a10g_prebuild", ROOT / "tests/attention/conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, torch, calls


def test_nvfp4_prebuild_follows_selected_cases(monkeypatch):
    module, torch, calls = _load_prebuild(monkeypatch)
    names = [
        "test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16",
        "test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16",
    ]
    module._batch_prefill_specs([SimpleNamespace(name=name) for name in names])
    batch = [args for kind, args in calls if kind == "batch"]
    assert len(batch) == 2
    assert {(args[1], args[5], args[7]) for args in batch} == {
        (torch.bfloat16, 512, 0),
        (torch.bfloat16, 512, 1),
    }
    calls.clear()
    item = SimpleNamespace(
        name="test_batch_prefill_with_paged_kv_cache_nvfp4_head_dim_256[bf16]",
        callspec=SimpleNamespace(params={"q_dtype": torch.bfloat16}),
    )
    module._batch_prefill_specs([item, item])
    batch = [args for kind, args in calls if kind == "batch"]
    assert len(batch) == 1
    assert (batch[0][1], batch[0][5]) == (torch.bfloat16, 256)


def test_nvfp4_only_warmup_skips_unrelated_grid(monkeypatch):
    _, torch, _ = _load_prebuild(monkeypatch)
    tree = ast.parse((ROOT / PREFILL).read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "warmup_jit"
    )
    fn.decorator_list = []
    calls = []
    namespace = {
        "torch": torch,
        "flashinfer": SimpleNamespace(
            jit=SimpleNamespace(build_jit_specs=lambda *a, **kw: calls.append(a))
        ),
        "gen_prefill_attention_modules": lambda *args: ["warmup"],
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), PREFILL, "exec"), namespace)
    owner = object()
    request = SimpleNamespace(
        module=owner,
        session=SimpleNamespace(
            items=[
                SimpleNamespace(module=owner, name="test_nvfp4"),
            ]
        ),
    )
    list(namespace["warmup_jit"](request))
    assert not calls
    request.session.items.append(SimpleNamespace(module=owner, name="test_base"))
    list(namespace["warmup_jit"](request))
    assert calls == [(["warmup"],)]


@pytest.mark.parametrize("fails", [False, True])
def test_logging_keeps_results_and_failure_exit(tmp_path, fails):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_sample.py").write_text(
        f"def test_case(): assert {not fails}\n"
    )
    script = tmp_path / "part.sh"
    script.write_text(
        f'set -e\nsource "{ROOT}/scripts/jit_test_logging.sh"\npytest -s tests/test_sample.py\n'
    )
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        ["bash", str(script)], cwd=tmp_path, env=env, text=True, capture_output=True
    )
    assert result.returncode == int(fails), result.stdout + result.stderr
    output = tmp_path / "junit/part/test_sample"
    assert output.with_suffix(".xml").is_file()
    telemetry = json.loads(output.with_suffix(".telemetry.json").read_text())
    assert telemetry
    assert output.with_suffix(".meta.json").is_file()
