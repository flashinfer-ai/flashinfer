"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Benchmark: cuTile GDN prefill vs FLA Triton GDN on Blackwell (SM100, B200).

Compares kernel latency (profiler-based) and wall-clock latency (end-to-end
Python overhead included) between:
  - cuTile: NVIDIA cuTile-based kernel (Blackwell-optimized, SM100+)
  - FLA:    Flash Linear Attention Triton kernel (upstream baseline)

Both use batch-first [B, T, H, K] tensor format.

Usage:
    python benchmarks/bench_gdn_prefill_cutile_vs_fla_blackwell.py

    # Focus on large workloads only (>1ms total kernel time):
    python benchmarks/bench_gdn_prefill_cutile_vs_fla_blackwell.py --large-only
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

# cuTile main entry point — import directly to avoid tvm_ffi dependency
import importlib.util, sys, types, pathlib
_kernel_path = pathlib.Path(__file__).parent.parent / "flashinfer/gdn_kernels/cutile_gdn_prefill.py"
_SGLANG_ROOT = pathlib.Path("/home/scratch.xutingz_wwfo/gitsrc/sglang")
# Bootstrap sglang FLA modules needed by cutile_gdn_prefill at runtime
def _stub(name):
    parts = name.split('.')
    for i in range(1, len(parts) + 1):
        pkg = '.'.join(parts[:i])
        if pkg not in sys.modules:
            m = types.ModuleType(pkg); m.__path__ = []; m.__package__ = pkg
            sys.modules[pkg] = m
def _load(name, path):
    _stub(name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = name.rsplit('.', 1)[0]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
import torch as _torch
_stub("sglang.srt.utils.common")
sys.modules["sglang.srt.utils.common"].torch_release = tuple(
    int(x) for x in _torch.__version__.split(".")[:2] if x.isdigit()
)
_sgl_python = str(_SGLANG_ROOT / "python")
if _sgl_python not in sys.path:
    sys.path.insert(0, _sgl_python)
_fla = _SGLANG_ROOT / "python/sglang/srt/layers/attention/fla"
for _n in ["utils", "l2norm", "op", "index", "cumsum",
           "chunk_scaled_dot_kkt", "solve_tril", "wy_fast", "chunk_delta_h", "chunk_o"]:
    _load(f"sglang.srt.layers.attention.fla.{_n}", str(_fla / f"{_n}.py"))
try:
    _spec = importlib.util.spec_from_file_location("cutile_gdn_prefill", _kernel_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ct_fwd = _mod.chunk_gated_delta_rule_cutile
    CT_AVAILABLE = True
except Exception as e:
    CT_AVAILABLE = False
    print(f"[WARN] cuTile not available: {e}")

# FLA Triton baseline
try:
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd as fla_fwd
    FLA_AVAILABLE = True
except ImportError as e:
    FLA_AVAILABLE = False
    print(f"[WARN] fla not available: {e}")


# ---------------------------------------------------------------------------
# Benchmark configs: 8 representative workloads
# ---------------------------------------------------------------------------
CONFIGS_ALL = [
    # Qwen3.5 GDN parameters with TP8: K=128, V=128
    # 4B model: H=32/8=4 per GPU; 397B: H=64/8=8 per GPU
    # (B,  T,    H,  K,   V,   label)
    # --- Qwen3.5-4B TP8 (H=4) ---
    (1,  2048, 4, 128, 128, "B=1 ,T=2048,H=4"),
    (2,  2048, 4, 128, 128, "B=2 ,T=2048,H=4"),
    (4,  2048, 4, 128, 128, "B=4 ,T=2048,H=4"),
    (8,  1024, 4, 128, 128, "B=8 ,T=1024,H=4"),
    (1,  4096, 4, 128, 128, "B=1 ,T=4096,H=4"),
    (4,  4096, 4, 128, 128, "B=4 ,T=4096,H=4"),
    (8,  2048, 4, 128, 128, "B=8 ,T=2048,H=4"),
    (16, 1024, 4, 128, 128, "B=16,T=1024,H=4"),
    # --- Qwen3.5-397B TP8 (H=8) ---
    (1,  2048, 8, 128, 128, "B=1 ,T=2048,H=8"),
    (2,  2048, 8, 128, 128, "B=2 ,T=2048,H=8"),
    (4,  2048, 8, 128, 128, "B=4 ,T=2048,H=8"),
    (8,  1024, 8, 128, 128, "B=8 ,T=1024,H=8"),
    (1,  4096, 8, 128, 128, "B=1 ,T=4096,H=8"),
    (4,  4096, 8, 128, 128, "B=4 ,T=4096,H=8"),
    (8,  2048, 8, 128, 128, "B=8 ,T=2048,H=8"),
    (16, 1024, 8, 128, 128, "B=16,T=1024,H=8"),
]

CONFIGS_LARGE = [cfg for cfg in CONFIGS_ALL if cfg[0] * cfg[1] >= 4 * 2048]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _profiler_time_us(fn, warmup: int = 20, reps: int = 50) -> float:
    """Measure total CUDA kernel time per call (μs) via torch.profiler."""
    for _ in range(warmup):
        fn()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / reps


def _profiler_breakdown(fn, warmup: int = 10, reps: int = 30) -> dict:
    """Return per-kernel GPU time breakdown (μs) via torch.profiler."""
    for _ in range(warmup):
        fn()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    result = {}
    for e in prof.key_averages():
        if e.device_time_total > 0:
            result[e.key] = e.device_time_total / reps
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def _check_sm100():
    cc = torch.cuda.get_device_capability()
    if cc[0] != 10:
        print(
            f"[WARN] cuTile GDN requires SM100 (Blackwell B200), "
            f"but current device is SM{cc[0]}{cc[1]}. "
            "Results may be incorrect or unavailable."
        )
        return False
    return True


def bench_config(B, T, H, K, V, label):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(B, T, H, K, dtype=dtype, device=device)
    k = torch.randn(B, T, H, K, dtype=dtype, device=device)
    v = torch.randn(B, T, H, V, dtype=dtype, device=device)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    idx = torch.arange(B, dtype=torch.int32, device=device)
    scale = K ** -0.5

    # L2-normalize q and k (standard preprocessing)
    q_n = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k_n = F.normalize(k.float(), p=2, dim=-1).to(dtype)

    results = {}

    # ---- cuTile ----
    if CT_AVAILABLE:
        fn_ct = lambda: ct_fwd(
            q_n, k_n, v, g, beta, scale,
            h0.clone(), idx,
            use_qk_l2norm_in_kernel=False,
        )
        results["ct_prof_us"] = _profiler_time_us(fn_ct)

    # ---- FLA Triton ----
    if FLA_AVAILABLE:
        fn_fla = lambda: fla_fwd(
            q_n, k_n, v, g, beta, scale,
            initial_state=h0.clone(),
            output_final_state=False,
        )
        results["fla_prof_us"] = _profiler_time_us(fn_fla)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cuTile GDN prefill vs FLA Triton on Blackwell"
    )
    parser.add_argument(
        "--large-only",
        action="store_true",
        help="Only run configs with total tokens >= 4*2048 (>1ms kernel time)",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Show per-kernel GPU time breakdown for cuTile",
    )
    args = parser.parse_args()

    is_sm100 = _check_sm100()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: SM{torch.cuda.get_device_capability()}")
    print()

    configs = CONFIGS_LARGE if args.large_only else CONFIGS_ALL

    # Header
    print(f"{'Config':<18} {'FLA':>10} {'cuTile':>10} {'speedup':>10}")
    print("-" * 52)

    for B, T, H, K, V, label in configs:
        r = bench_config(B, T, H, K, V, label)

        ct_p = r.get("ct_prof_us", float("nan"))
        fla_p = r.get("fla_prof_us", float("nan"))
        spd = fla_p / ct_p if ct_p > 0 else float("nan")

        print(
            f"{label:<18}"
            f" {fla_p/1000:>9.3f}ms"
            f" {ct_p/1000:>9.3f}ms"
            f" {spd:>9.2f}x"
        )

        if args.breakdown and CT_AVAILABLE:
            import torch.nn.functional as F2
            device = torch.device("cuda")
            dtype = torch.bfloat16
            q = torch.randn(B, T, H, K, dtype=dtype, device=device)
            k = torch.randn(B, T, H, K, dtype=dtype, device=device)
            v = torch.randn(B, T, H, V, dtype=dtype, device=device)
            g = F2.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
            beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
            h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
            idx = torch.arange(B, dtype=torch.int32, device=device)
            q_n = F2.normalize(q.float(), p=2, dim=-1).to(dtype)
            k_n = F2.normalize(k.float(), p=2, dim=-1).to(dtype)
            fn_ct = lambda: ct_fwd(q_n, k_n, v, g, beta, K**-0.5, h0.clone(), idx, use_qk_l2norm_in_kernel=False)
            bd = _profiler_breakdown(fn_ct)
            total = sum(bd.values())
            print(f"  {'Kernel':<55} {'us':>8}  {'%':>5}")
            for kname, us in sorted(bd.items(), key=lambda x: -x[1]):
                print(f"  {kname:<55} {us:>8.1f}  {100*us/total:>5.1f}%")
            print()

    print()
    print("GPU kernel time only (torch.profiler CUDA events). NVIDIA B200 (SM100).")


if __name__ == "__main__":
    main()
