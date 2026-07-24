"""Reproduce the exact vLLM/SGLang autotune_v2 usage SEQUENCE against a real
op, short of a full model server:

    with autotune_v2(cache_root=..., measure=P):   # warmup
        for m in shape_grid: op(m)                 # tune buckets
    autotune_v2_reload()                           # rank-consistency finalize
    for m in serve_shapes: op(m)                   # bare serving

Asserts: warmup tunes+publishes; reload keeps winners; bare serving hits the
store with ZERO profiling.  Run for BOTH measure policies, including
execution_mode="cuda_graph" (per-candidate graph capture during profiling)
which had no on-GPU coverage before.
"""

import os
import sys
import tempfile

import torch

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
sm = next(
    i
    for i in range(torch.cuda.device_count())
    if torch.cuda.get_device_properties(i).major in (10, 12)
)
torch.cuda.set_device(sm)
print(f"device: {torch.cuda.get_device_properties(sm).name}")

import flashinfer
from flashinfer import MeasurementPolicy, autotune_v2, autotune_v2_reload
from flashinfer.autotuner import AutoTuner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
from utils_fp8 import to_float8  # noqa: E402

N = K = 4096
WARMUP_MS = [1, 4, 16, 64]  # a warmup tunes a range of buckets
SERVE_MS = [1, 4, 16, 64]  # serving replays them bare


def make_op(m):
    a = torch.randn([1, m, K], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([1, N, K], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    a8, a_s = to_float8(a)
    b8, b_s = to_float8(b)

    def call():
        return flashinfer.bmm_fp8(a8, b8, a_s, b_s, torch.bfloat16, backend="auto")

    return call


_PROFILE_CALLS = [0]
_orig_profile = AutoTuner._profile_single_kernel


def _counting_profile(self, *a, **k):
    _PROFILE_CALLS[0] += 1
    return _orig_profile(self, *a, **k)


AutoTuner._profile_single_kernel = _counting_profile


def run(policy_name, measure, root):
    print(f"\n=== policy={policy_name} ===")
    tuner = AutoTuner.get()
    tuner.clear_cache()
    ops = {m: make_op(m) for m in set(WARMUP_MS) | set(SERVE_MS)}

    # 1) vLLM/SGLang warmup: one context around the dummy forwards.
    _PROFILE_CALLS[0] = 0
    with torch.inference_mode(), autotune_v2(cache_root=root, measure=measure):
        for m in WARMUP_MS:
            ops[m]()
    tuned = _PROFILE_CALLS[0]
    print(f"warmup: {tuned} _profile_single_kernel calls (expected > 0)")
    assert tuned > 0, "warmup did not profile anything"
    entries = list(__import__("pathlib").Path(root).glob("v2/*/entries/*.json"))
    print(f"        entries on disk: {len(entries)}")

    # 2) rank-consistency finalize (vLLM patch calls this after a barrier).
    autotune_v2_reload()

    # 3) bare serving: no context; must hit the store with zero profiling.
    _PROFILE_CALLS[0] = 0
    with torch.inference_mode():
        for m in SERVE_MS:
            out = ops[m]()
    hits = len(tuner._managed_decoded)
    print(
        f"serve: {_PROFILE_CALLS[0]} profile calls (expected 0), "
        f"managed hits={hits}, out={tuple(out.shape)}"
    )
    assert _PROFILE_CALLS[0] == 0, "bare serving re-profiled — store not consulted!"
    assert hits > 0, "bare serving did not hit the managed store"
    print(f"OK: {policy_name} warmup->reload->bare-serve clean")


def main():
    root = tempfile.mkdtemp(prefix="fi_integ_")
    print(f"cache root: {root}")
    run("eager", MeasurementPolicy(execution_mode="eager"), root)
    # The previously-untested path: per-candidate CUDA-graph capture profiling.
    run("cuda_graph", MeasurementPolicy(execution_mode="cuda_graph"), root)
    print("\nALL INTEGRATION SEQUENCES PASSED")


if __name__ == "__main__":
    sys.exit(main())
