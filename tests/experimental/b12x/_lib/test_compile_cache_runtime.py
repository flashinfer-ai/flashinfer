"""Real CuTe object reuse across processes and allocation-stable graph replay."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = r"""
import json
import torch
from b12x._lib.compiler import compile_cache_info
from b12x.norm.hyperconnection._cute import grouped_rmsnorm

torch.manual_seed(19)
x = torch.randn((2, 4, 256), device="cuda", dtype=torch.bfloat16)
w = torch.randn((4, 256), device="cuda", dtype=torch.bfloat16)
y = torch.empty_like(x)
def run():
    grouped_rmsnorm(x, w, y, eps=1e-6, streams=4, hidden_size=256)
run()
torch.cuda.synchronize()
before = compile_cache_info()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    run()
allocated = torch.cuda.memory_allocated()
x.fill_(0.5)
graph.replay()
torch.cuda.synchronize()
reference = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * (1 + w.float())
torch.testing.assert_close(y, reference.to(y.dtype), rtol=1e-2, atol=1e-2)
assert torch.cuda.memory_allocated() >= allocated
del reference
assert torch.cuda.memory_allocated() == allocated
after = compile_cache_info()
assert after["compile_misses"] == before["compile_misses"]
assert after["disk_cache_hits"] == before["disk_cache_hits"]
print(json.dumps(before))
"""


def test_fresh_process_reuses_object_after_operational_environment_change(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM12x CUDA device required")
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(
            (str(root), str(root / "flashinfer/experimental/b12x/_compat"))
        ),
        B12X_COMPILE_CACHE_DIR=str(tmp_path),
        B12X_COMPILE_DISK_CACHE="1",
        B12X_COMPILE_MEMORY_CACHE="1",
    )

    def run():
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT],
            env=env,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    first = run()
    assert first["compile_misses"] == 1
    assert first["disk_cache_hits"] == 0
    env.update(
        B12X_UNUSED_SETTING="different",
        CUTLASS_UNUSED_SETTING="different",
        CUTE_UNUSED_SETTING="different",
        B12X_PRINT_COMPILE_PROGRESS="1",
    )
    second = run()
    assert second["compile_misses"] == 0
    assert second["disk_cache_hits"] == 1
