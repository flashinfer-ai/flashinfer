"""Combined provider export integrity and architecture coexistence."""

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_ROOT = (
    Path(__file__).resolve().parents[2] / "flashinfer/moe_ep/kernels/cutedsl_megamoe"
)


def test_export_manifest_matches_sources():
    manifest = json.loads((_ROOT / "export_manifest.json").read_text())
    files = manifest["files"]
    assert set(files) == {str(path.relative_to(_ROOT)) for path in _ROOT.rglob("*.py")}
    for name, digest in files.items():
        assert hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() == digest, name


def test_reducer_benchmark_pins_match_vendored_sources():
    root = Path(__file__).resolve().parents[2]
    benchmark = root / "benchmarks/bench_cake_megamoe_topk_reduce.py"
    if not benchmark.is_file():
        pytest.skip("requires the source-tree reducer benchmark")
    names = {
        "_CUTEDSL_SOURCE",
        "_CUTEDSL_SOURCE_SHA256",
        "_RUBIN_CUTEDSL_SOURCE",
        "_RUBIN_CUTEDSL_SOURCE_SHA256",
    }
    bindings = {
        target.id: ast.literal_eval(node.value)
        for node in ast.parse(benchmark.read_text()).body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }
    assert set(bindings) == names
    for prefix in ("_CUTEDSL", "_RUBIN_CUTEDSL"):
        source = root / bindings[prefix + "_SOURCE"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == bindings[prefix + "_SOURCE_SHA256"]
        )


def test_provider_imports_are_lazy_and_namespaced():
    pytest.importorskip("cutlass")
    script = """
import importlib
import sys
import types
import torch

namespace = "flashinfer.moe_ep.kernels.cutedsl_megamoe"
foreign = {name: types.ModuleType(name) for name in ("sources", "common")}
sys.modules.update(foreign)
provider = importlib.import_module(namespace)
assert not any(name.startswith(namespace + ".sources.kernel_src.") for name in sys.modules)
getattr(provider, "BlackwellInferenceMegaMoE")
getattr(provider, "BlackwellInferenceGenphaseMegaMoE")
assert not any(name.startswith(namespace + ".sources.kernel_src.rubin.") for name in sys.modules)
getattr(provider, "RubinInferenceMegaMoE")
getattr(provider, "RubinInferenceGenphaseMegaMoE")
assert all(sys.modules[name] is value for name, value in foreign.items())
assert not torch.cuda.is_initialized()
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-c", script], env=environment, check=True)
