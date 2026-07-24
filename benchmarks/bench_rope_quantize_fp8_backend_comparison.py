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

Benchmark: Fused MLA RoPE + FP8-Quantize Backend Comparison

Compares FlashInfer's fused ``rope.rope_quantize_fp8`` op (RoPE + per-tensor
fp8_e4m3 quantization applied together) across backends over a ``num_tokens``
sweep:

  - ``cutile``     : FlashInfer ``rope_quantize_fp8(backend="cutile")`` (the
                     migrated cuTile backend, PR #4019)
  - ``flashinfer`` : FlashInfer native fused CUDA kernel (the SOTA baseline)
  - ``torch``      : PyTorch-native reference (``torch.compile`` RoPE + fp8 cast)

Providers whose backend is unavailable degrade to N/A (NaN) via try/except, so
``cutile`` (needs the cuda.tile toolchain image) can be measured separately from
the native kernel and the result CSVs merged. Emits a per-provider latency
table, a speedup summary vs the baseline, and a heatmap.

The op is modeled on the ``mla`` configuration used for regression/perf testing:
128 Q heads, 1 (shared 2D) K head, head_size 576 = rope_dim 64 + no_rope_dim 512,
bf16 in -> fp8_e4m3 out. ``gqa``/``mha`` configs are also selectable via
``--config``. Only ``backend="cutile"`` is limited to the MLA-style 2D key with
``is_neox=False``, so it raises and degrades to N/A on ``gqa``/``mha``; the
native ``flashinfer`` CUDA kernel and the ``torch`` provider support GQA/MHA
(3D key) and run on those configs too.

Usage:
    python bench_rope_quantize_fp8_backend_comparison.py
    python bench_rope_quantize_fp8_backend_comparison.py --providers cutile --csv out.csv
    python bench_rope_quantize_fp8_backend_comparison.py --config mla --baseline flashinfer

Requirements:
    - flashinfer (flashinfer / native kernel providers)
    - cuda.tile toolchain (cutile provider)
    - matplotlib for the heatmap
"""

import argparse
import csv as _csv
import os
import sys
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from flashinfer.testing.utils import bench_gpu_time

# Add the project root to Python path to import test helpers (same as the
# source bench_rope_quantize_fp8.py).
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from tests.test_helpers.rope_reference import RotaryEmbedding  # noqa: E402

INPUT_DTYPE = torch.bfloat16
QUANT_DTYPE = torch.float8_e4m3fn


# --------------------------------------------------------------------------- #
# Provider availability
# --------------------------------------------------------------------------- #
def _has_flashinfer() -> bool:
    try:
        import flashinfer  # noqa: F401
        from flashinfer.rope import rope_quantize_fp8  # noqa: F401

        return True
    except Exception:
        return False


def _has_cutile() -> bool:
    try:
        import cuda.tile  # noqa: F401
        from flashinfer.rope import rope_quantize_fp8  # noqa: F401

        return True
    except Exception:
        return False


ALL_PROVIDERS = ["cutile", "flashinfer", "torch"]
_AVAIL = {
    "cutile": _has_cutile,
    "flashinfer": _has_flashinfer,
    "torch": lambda: True,
}


# --------------------------------------------------------------------------- #
# Config + input construction (mirrors bench_rope_quantize_fp8.benchmark_config)
# --------------------------------------------------------------------------- #
def _config_params(config_name: str) -> Tuple[int, int, int, int]:
    """Return (num_qo_heads, num_kv_heads, rope_dim, no_rope_dim)."""
    if config_name == "mla":
        # MLA: shipped/perf configuration, 2D (shared) K tensor.
        return 128, 1, 64, 512
    elif config_name == "gqa":
        return 32, 8, 64, 64
    elif config_name == "mha":
        return 32, 32, 64, 64
    else:
        raise ValueError(f"Unknown config: {config_name}")


class _Inputs:
    """Container for one (config, num_tokens) input set."""

    def __init__(self, config_name: str, num_tokens: int, device: str, seed: int):
        torch.manual_seed(seed)
        num_qo_heads, num_kv_heads, rope_dim, no_rope_dim = _config_params(config_name)
        total_dim = rope_dim + no_rope_dim

        self.config_name = config_name
        self.num_tokens = num_tokens
        self.rope_dim = rope_dim
        self.no_rope_dim = no_rope_dim
        self.total_dim = total_dim

        self.q_in = torch.randn(
            num_tokens, num_qo_heads, total_dim, dtype=INPUT_DTYPE, device=device
        )
        if config_name == "mla":
            # MLA: 2D K tensor (shared across heads)
            self.k_in = torch.randn(
                num_tokens, total_dim, dtype=INPUT_DTYPE, device=device
            )
        else:
            self.k_in = torch.randn(
                num_tokens, num_kv_heads, total_dim, dtype=INPUT_DTYPE, device=device
            )

        self.pos_ids = torch.arange(num_tokens, device=device)

        self.rope_ref = RotaryEmbedding(
            head_size=total_dim,
            rotary_dim=rope_dim,
            max_position_embeddings=4096,
            base=10000,
            is_neox_style=False,
            dtype=INPUT_DTYPE,
            device=device,
        )


# --------------------------------------------------------------------------- #
# Reference + per-provider callables
# --------------------------------------------------------------------------- #
def _torch_reference(inp: _Inputs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager PyTorch-native RoPE + fp8 cast; used as the correctness reference."""
    q_out_f16, k_out_f16 = inp.rope_ref.forward_native(
        inp.pos_ids, inp.q_in, inp.k_in
    )
    return q_out_f16.to(QUANT_DTYPE), k_out_f16.to(QUANT_DTYPE)


def _make_execute(provider: str, inp: _Inputs) -> Callable[[], Tuple[torch.Tensor, ...]]:
    """Return a zero-arg callable that runs `provider`'s op and returns its outputs.

    For the fused kernels the returned tuple is
    ``(q_rope_out, q_nope_out, k_rope_out, k_nope_out)``; for torch it is
    ``(q_out_f8, k_out_f8)``.  ``_to_full_qk`` normalizes either into full q/k.
    """
    rope_dim = inp.rope_dim

    if provider in ("flashinfer", "cutile"):
        import flashinfer

        # Split tensors for the fused kernel (rope / no-rope halves).
        q_rope = inp.q_in[..., :rope_dim]
        q_nope = inp.q_in[..., rope_dim:]
        k_rope = inp.k_in[..., :rope_dim]
        k_nope = inp.k_in[..., rope_dim:]

        q_rope_out = torch.empty_like(q_rope, dtype=QUANT_DTYPE)
        q_nope_out = torch.empty_like(q_nope, dtype=QUANT_DTYPE)
        k_rope_out = torch.empty_like(k_rope, dtype=QUANT_DTYPE)
        k_nope_out = torch.empty_like(k_nope, dtype=QUANT_DTYPE)

        cos_sin_cache = inp.rope_ref.cos_sin_cache
        pos_ids = inp.pos_ids

        if provider == "flashinfer":

            def run():
                flashinfer.rope.rope_quantize_fp8(
                    q_rope=q_rope,
                    k_rope=k_rope,
                    q_nope=q_nope,
                    k_nope=k_nope,
                    cos_sin_cache=cos_sin_cache,
                    pos_ids=pos_ids,
                    is_neox=False,
                    q_rope_out=q_rope_out,
                    k_rope_out=k_rope_out,
                    q_nope_out=q_nope_out,
                    k_nope_out=k_nope_out,
                    quant_scale_q=1.0,
                    quant_scale_kv=1.0,
                    enable_pdl=False,
                )
                return q_rope_out, q_nope_out, k_rope_out, k_nope_out

        else:  # cutile

            def run():
                flashinfer.rope.rope_quantize_fp8(
                    q_rope=q_rope,
                    k_rope=k_rope,
                    q_nope=q_nope,
                    k_nope=k_nope,
                    cos_sin_cache=cos_sin_cache,
                    pos_ids=pos_ids,
                    is_neox=False,
                    q_rope_out=q_rope_out,
                    k_rope_out=k_rope_out,
                    q_nope_out=q_nope_out,
                    k_nope_out=k_nope_out,
                    quant_scale_q=1.0,
                    quant_scale_kv=1.0,
                    backend="cutile",
                )
                return q_rope_out, q_nope_out, k_rope_out, k_nope_out

    elif provider == "torch":
        q_in = inp.q_in
        k_in = inp.k_in
        pos_ids = inp.pos_ids
        rope_ref = inp.rope_ref

        @torch.compile
        def torch_rope_quantize(q_in, k_in, pos_ids):
            q_out_f16, k_out_f16 = rope_ref.forward_native(pos_ids, q_in, k_in)
            return q_out_f16.to(QUANT_DTYPE), k_out_f16.to(QUANT_DTYPE)

        def run():
            return torch_rope_quantize(q_in, k_in, pos_ids)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run


def _to_full_qk(
    provider: str, out: Tuple[torch.Tensor, ...], rope_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize a provider's raw output tuple into full (q, k) fp8 tensors."""
    if provider in ("flashinfer", "cutile"):
        q_rope_out, q_nope_out, k_rope_out, k_nope_out = out
        q_full = torch.cat([q_rope_out, q_nope_out], dim=-1)
        k_full = torch.cat([k_rope_out, k_nope_out], dim=-1)
        return q_full, k_full
    else:  # torch
        return out[0], out[1]


# --------------------------------------------------------------------------- #
# Correctness + timing
# --------------------------------------------------------------------------- #
def verify_correctness(
    config_name: str, num_tokens: int, provider: str
) -> Tuple[bool, float]:
    """Return (ok, cosine_similarity) of provider output vs torch reference."""
    try:
        inp = _Inputs(config_name, num_tokens, device="cuda", seed=42)
        out = _make_execute(provider, inp)()
        q, k = _to_full_qk(provider, out, inp.rope_dim)
        ref_q, ref_k = _torch_reference(inp)
        a = torch.cat(
            [q.to(torch.float32).reshape(-1), k.to(torch.float32).reshape(-1)]
        ).reshape(1, -1)
        b = torch.cat(
            [ref_q.to(torch.float32).reshape(-1), ref_k.to(torch.float32).reshape(-1)]
        ).reshape(1, -1)
        cos = torch.nn.functional.cosine_similarity(a, b).item()
        return cos > 0.99, cos
    except Exception:
        return False, 0.0


def bench_one(config_name: str, num_tokens: int, provider: str) -> float:
    """Median latency (ms) for one (config, num_tokens, provider); NaN if failed."""
    if not _AVAIL[provider]():
        return float("nan")
    try:
        inp = _Inputs(config_name, num_tokens, device="cuda", seed=0)
        run = _make_execute(provider, inp)
        run()  # warmup / compile
        times = bench_gpu_time(
            fn=run,
            enable_cupti=True,
            dry_run_iters=5,
            repeat_iters=30,
            cold_l2_cache=True,
            use_cuda_graph=False,
        )
        return float(np.median(times))
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_benchmark_sweep(
    config_name: str,
    num_tokens_values: List[int],
    providers: List[str],
) -> Dict[str, Dict[int, float]]:
    """Return {provider: {num_tokens: median_ms}}."""
    results: Dict[str, Dict[int, float]] = {p: {} for p in providers}
    total = len(num_tokens_values)
    current = 0

    print(f"\nBenchmarking rope_quantize_fp8  config={config_name}  bf16->fp8_e4m3")
    print("=" * (20 + 12 * len(providers)))
    header = f"{'ntok':>6} |"
    for p in providers:
        header += f" {p:>12}"
    print(header)
    print("-" * (20 + 12 * len(providers)))

    for ntok in num_tokens_values:
        current += 1
        row = f"{ntok:>6} |"
        for p in providers:
            ms = bench_one(config_name, ntok, p)
            results[p][ntok] = ms
            row += f" {ms:>12.5f}" if ms == ms else f" {'--':>12}"
        print(f"[{current:3d}/{total}] " + row)

    return results


def print_summary_table(
    num_tokens_values: List[int],
    results: Dict[str, Dict[int, float]],
    providers: List[str],
    baseline: str,
):
    """Speedup of each provider vs the baseline (>1 = provider faster)."""
    if baseline not in results:
        return
    for p in providers:
        if p == baseline:
            continue
        print(f"\n{'=' * 60}")
        print(f"Speedup: {p} vs {baseline} (baseline_ms / {p}_ms;  >1 = {p} faster)")
        print(f"{'=' * 60}")
        header = "num_tokens".ljust(12) + f"{'speedup':>12}"
        print(header)
        print("-" * 24)
        ratios = []
        for ntok in num_tokens_values:
            bt = results[baseline].get(ntok, float("nan"))
            pt = results[p].get(ntok, float("nan"))
            if pt == pt and bt == bt and pt > 0:
                r = bt / pt
                ratios.append(r)
                print(f"{ntok:<12}{r:>12.2f}")
            else:
                print(f"{ntok:<12}{'N/A':>12}")
        if ratios:
            print(
                f"\n  geomean {np.exp(np.mean(np.log(ratios))):.2f}x  "
                f"min {min(ratios):.2f}x  max {max(ratios):.2f}x  "
                f"({sum(1 for r in ratios if r > 1)}/{len(ratios)} shapes {p} faster)"
            )


def write_csv(path: str, config_name: str, results: Dict[str, Dict[int, float]]):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["provider", "config", "num_tokens", "median_ms"])
        for provider, d in results.items():
            for ntok, ms in sorted(d.items()):
                w.writerow([provider, config_name, ntok, f"{ms:.6f}"])
    print(f"\nWrote {path}")


def create_heatmap(
    config_name: str,
    num_tokens_values: List[int],
    results: Dict[str, Dict[int, float]],
    providers: List[str],
    baseline: str,
    output_file: str,
):
    """Heatmap of speedup vs `baseline` over (provider x num_tokens)."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap")
        return
    if baseline not in results:
        return
    others = [p for p in providers if p != baseline]
    if not others:
        return
    mat = np.full((len(others), len(num_tokens_values)), np.nan)
    for i, p in enumerate(others):
        for j, ntok in enumerate(num_tokens_values):
            bt = results[baseline].get(ntok, float("nan"))
            pt = results[p].get(ntok, float("nan"))
            if pt == pt and bt == bt and pt > 0:
                mat[i, j] = bt / pt
    if np.all(np.isnan(mat)):
        return
    fig, ax = plt.subplots(
        figsize=(max(6, 1.0 * len(num_tokens_values)), max(2.5, 1.0 * len(others)))
    )
    norm = mcolors.TwoSlopeNorm(
        vmin=min(0.5, np.nanmin(mat)), vcenter=1.0, vmax=max(1.5, np.nanmax(mat))
    )
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(f"{baseline}_ms / provider_ms", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(num_tokens_values)))
    ax.set_yticks(np.arange(len(others)))
    ax.set_xticklabels([str(n) for n in num_tokens_values])
    ax.set_yticklabels(others)
    for i in range(len(others)):
        for j in range(len(num_tokens_values)):
            if not np.isnan(mat[i, j]):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] < 0.7 or mat[i, j] > 1.5 else "black",
                )
    ax.set_xlabel("num_tokens")
    ax.set_ylabel("provider")
    ax.set_title(
        f"rope_quantize_fp8 ({config_name}): speedup vs {baseline}\n"
        f"(>1.0 = provider faster)"
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused rope_quantize_fp8 backends"
    )
    parser.add_argument(
        "--config",
        choices=["mla", "gqa", "mha"],
        default="mla",
        help="Attention configuration (default mla, the shipped/perf one)",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=",".join(ALL_PROVIDERS),
        help=f"Comma-separated subset of {ALL_PROVIDERS}",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="flashinfer",
        help="Speedup baseline provider (default flashinfer, the SOTA kernel)",
    )
    parser.add_argument(
        "--output-prefix", type=str, default="rope_quantize_fp8_backend_comparison"
    )
    parser.add_argument("--csv", type=str, default=None, help="Write raw medians to CSV")
    args = parser.parse_args()

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    providers = [p for p in requested if p in ALL_PROVIDERS]
    available = [p for p in providers if _AVAIL[p]()]
    print(f"Config:              {args.config}")
    print(f"Requested providers: {providers}")
    print(f"Available here:      {available}  (unavailable are skipped as N/A)")

    num_tokens_values = [1, 4, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 2048, 4096]

    # Inline correctness check (once per available provider at a mid shape).
    print("\nCorrectness (cosine-sim of q/k output vs torch ref, at 512 tokens):")
    for p in available:
        ok, cos = verify_correctness(args.config, 512, p)
        print(f"  {p:>10}: {'OK ' if ok else 'FAIL'} cos={cos:.4f}")

    results = run_benchmark_sweep(args.config, num_tokens_values, providers)
    baseline = args.baseline if args.baseline in providers else providers[0]
    print_summary_table(num_tokens_values, results, providers, baseline)

    if args.csv:
        write_csv(args.csv, args.config, results)

    create_heatmap(
        args.config,
        num_tokens_values,
        results,
        providers,
        baseline,
        f"{args.output_prefix}_{args.config}_vs_{baseline}.png",
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
