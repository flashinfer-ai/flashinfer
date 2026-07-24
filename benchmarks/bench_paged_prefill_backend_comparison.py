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

Benchmark: Paged Prefill Attention Backend Comparison (cuTile vs trtllm SOTA)

Compares FlashInfer's cuTile paged-prefill backend against the trtllm-gen SOTA
kernel over ocean-eval's PrefillPaged workload, swept across page sizes:

  - ``cutile`` : this checkout's ``prefill_attention_kv_paged_cutile`` cuTile kernel
  - ``trtllm`` : ``flashinfer.prefill.trtllm_batch_context_with_kv_cache`` (SOTA
                 baseline used by ocean-eval's dashboard for paged prefill)

NOTE: paged prefill is NOT (yet) wired into ``BatchPrefillWithPagedKVCacheWrapper``
in this checkout (only ragged is — commit a0600669f). The cuTile provider therefore
calls the raw kernel ``prefill_attention_kv_paged_cutile`` directly, mirroring how
ocean-eval's ``flashinfer_ops.prefill_attention_kv_paged`` dispatches.

TWO-FLASHINFER NOTE: the cuTile backend lives only in THIS checkout, while the
image's INSTALLED flashinfer carries the working trtllm path (the checkout's
trtllm path can crash with atomic_fmax). Each provider independently try/excepts
and degrades to NaN, so the bench is typically RUN TWICE and merged:

    python3 bench_paged_prefill_backend_comparison.py --providers cutile --csv cutile.csv
    python3 bench_paged_prefill_backend_comparison.py --providers trtllm --csv trtllm.csv

Workload: bf16, causal, head_dim=128, num_qo_heads=128, num_kv_heads=8 (GQA),
page_size sweep {16,64}. Metric = kernel self-time median (ms).
Speedup = trtllm_ms / cutile_ms  (>1 == cutile faster).

Requirements:
    - cuda.tile toolchain (cutile provider); flashinfer trtllm-gen (trtllm provider)
    - matplotlib for the heatmap
"""

import argparse
import csv as _csv
import math
import numpy as np
import torch
from typing import Callable, Dict, List, Tuple

from flashinfer.testing.utils import bench_gpu_time

# --------------------------------------------------------------------------- #
# Fixed workload (ocean PrefillPaged): head_dim=128, heads 128/8 (GQA), causal.
# --------------------------------------------------------------------------- #
DEV = torch.device("cuda")
DT = torch.bfloat16
NUM_QO_HEADS = 128
NUM_KV_HEADS = 8
HEAD_DIM = 128  # head_dim_qk == head_dim_vo


# --------------------------------------------------------------------------- #
# Provider availability
# --------------------------------------------------------------------------- #
def _has_cutile() -> bool:
    try:
        import cuda.tile  # noqa: F401
        from flashinfer.attention.kernels.cutile.fmha_prefill_bsr_cutile import (  # noqa: F401
            prefill_attention_kv_paged_cutile,
        )

        return True
    except Exception:
        return False


def _has_trtllm() -> bool:
    try:
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache  # noqa: F401

        return True
    except Exception:
        return False


ALL_PROVIDERS = ["cutile", "trtllm"]
_AVAIL = {
    "cutile": _has_cutile,
    "trtllm": _has_trtllm,
}


# --------------------------------------------------------------------------- #
# Input generation + per-provider callables
# --------------------------------------------------------------------------- #
def _make_inputs(batch: int, seq: int, page: int):
    """Build a shared paged KV in HND layout plus the derived NHD (cuTile) layout.

    HND (trtllm): k/v_cache [total_pages, num_kv_heads, page_size, head_dim]
    NHD (cuTile): k/v_cache [total_pages, page_size, num_kv_heads, head_dim]
    Both hold identical values (NHD is HND transposed) so the two providers can be
    correctness-compared.
    """
    torch.manual_seed(0)
    total_q = batch * seq
    ppb = math.ceil(seq / page)
    total_pages = batch * ppb

    q = torch.randn(total_q, NUM_QO_HEADS, HEAD_DIM, dtype=DT, device=DEV)
    k_hnd = torch.randn(total_pages, NUM_KV_HEADS, page, HEAD_DIM, dtype=DT, device=DEV)
    v_hnd = torch.randn(total_pages, NUM_KV_HEADS, page, HEAD_DIM, dtype=DT, device=DEV)
    # cuTile expects [total_pages, page_size, num_kv_heads, head_dim].
    k_nhd = k_hnd.transpose(1, 2).contiguous()
    v_nhd = v_hnd.transpose(1, 2).contiguous()

    block_tables = torch.arange(
        total_pages, dtype=torch.int32, device=DEV
    ).reshape(batch, ppb)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=DEV)
    # Exclusive-prefix-sum q offsets; cum[:-1] is the per-request q start row.
    cum = torch.arange(batch + 1, dtype=torch.int32, device=DEV) * seq
    seq_offset = cum[:-1].contiguous()
    return q, k_hnd, v_hnd, k_nhd, v_nhd, block_tables, seq_lens, cum, seq_offset


def _make_execute(provider: str, batch: int, seq: int, page: int) -> Callable[[], torch.Tensor]:
    (q, k_hnd, v_hnd, k_nhd, v_nhd, block_tables, seq_lens, cum, seq_offset) = (
        _make_inputs(batch, seq, page)
    )
    scale = 1.0 / math.sqrt(HEAD_DIM)

    if provider == "cutile":
        from flashinfer.attention.kernels.cutile.fmha_prefill_bsr_cutile import (
            prefill_attention_kv_paged_cutile,
        )

        def run():
            return prefill_attention_kv_paged_cutile(
                q=q,
                k_cache=k_nhd,
                v_cache=v_nhd,
                actual_seq_lens_q=seq_lens,
                actual_seq_lens_kv=seq_lens,
                actual_seq_offset=seq_offset,
                block_tables=block_tables,
                k_scale=scale,  # folds the softmax scale
                v_scale=1.0,
                num_batch=batch,
                max_seq_len=seq,
                is_causal=True,
            )

    elif provider == "trtllm":
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        ws = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device=DEV)

        def run():
            # NOTE: no `causal=` kwarg — the image's INSTALLED flashinfer signature
            # does not accept it (matches sota_bench/bench_prefill_paged_trtllm.py).
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=(k_hnd, v_hnd),
                workspace_buffer=ws,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=seq,
                max_kv_len=seq,
                bmm1_scale=scale,
                bmm2_scale=1.0,
                batch_size=batch,
                cum_seq_lens_q=cum,
                cum_seq_lens_kv=cum,
                kv_layout="HND",
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run


def _as_output(out) -> torch.Tensor:
    return out[0] if isinstance(out, (list, tuple)) else out


# --------------------------------------------------------------------------- #
# Correctness + timing
# --------------------------------------------------------------------------- #
def verify_correctness(batch: int, seq: int, page: int, providers: List[str]) -> None:
    """Cosine-sim of cutile vs trtllm output when both are available; else skip."""
    if not ("cutile" in providers and "trtllm" in providers):
        print("  correctness: need both cutile and trtllm available -> SKIP")
        return
    if not (_AVAIL["cutile"]() and _AVAIL["trtllm"]()):
        print("  correctness: both providers not importable here -> SKIP")
        return
    try:
        out_c = _as_output(_make_execute("cutile", batch, seq, page)()).float().reshape(1, -1)
        out_t = _as_output(_make_execute("trtllm", batch, seq, page)()).float().reshape(1, -1)
        cos = torch.nn.functional.cosine_similarity(out_c, out_t).item()
        print(f"  correctness @ batch={batch} s_kv={seq} page={page}: cos={cos:.4f} "
              f"{'OK' if cos > 0.99 else 'CHECK'}")
    except Exception as e:
        print(f"  correctness: FAILED ({repr(e)[:80]})")


def bench_one(batch: int, seq: int, page: int, provider: str) -> float:
    """Median kernel self-time (ms) for one (shape, provider); NaN if unavailable."""
    if not _AVAIL[provider]():
        return float("nan")
    try:
        run = _make_execute(provider, batch, seq, page)
        run()  # warmup / compile
        torch.cuda.synchronize()
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
    page_values: List[int],
    batch_values: List[int],
    s_kv_values: List[int],
    providers: List[str],
) -> Dict[str, Dict[Tuple[int, int, int], float]]:
    """Return {provider: {(page, batch, s_kv): median_ms}}."""
    results: Dict[str, Dict[Tuple[int, int, int], float]] = {p: {} for p in providers}
    configs = [
        (pg, b, s) for pg in page_values for b in batch_values for s in s_kv_values
    ]
    total = len(configs)

    print(f"\nPaged prefill (head_dim={HEAD_DIM}, heads {NUM_QO_HEADS}/{NUM_KV_HEADS}, "
          f"bf16, causal)")
    print("=" * (32 + 12 * len(providers)))
    header = f"{'page':>5} {'batch':>6} {'s_kv':>7} |"
    for p in providers:
        header += f" {p:>10}"
    print(header)
    print("-" * (32 + 12 * len(providers)))

    for i, (page, batch, seq) in enumerate(configs, 1):
        row = f"{page:>5} {batch:>6} {seq:>7} |"
        for p in providers:
            ms = bench_one(batch, seq, page, p)
            results[p][(page, batch, seq)] = ms
            row += f" {ms:>10.5f}" if ms == ms else f" {'--':>10}"
        print(f"[{i:3d}/{total}] " + row)
        torch.cuda.empty_cache()

    return results


def print_summary_table(
    page_values: List[int],
    batch_values: List[int],
    s_kv_values: List[int],
    results: Dict[str, Dict[Tuple[int, int, int], float]],
    providers: List[str],
    baseline: str,
):
    """Speedup of each provider vs the baseline (>1 = provider faster), per page."""
    if baseline not in results:
        return
    for p in providers:
        if p == baseline:
            continue
        for page in page_values:
            print(f"\n{'=' * 72}")
            print(f"Speedup [page={page}]: {p} vs {baseline} "
                  f"({baseline}_ms / {p}_ms;  >1 = {p} faster)")
            print(f"{'=' * 72}")
            header = "batch\\s_kv".ljust(12)
            for s in s_kv_values:
                header += f"{s:>10}"
            print(header)
            print("-" * (12 + 10 * len(s_kv_values)))
            ratios = []
            for batch in batch_values:
                row = f"{batch:<12}"
                for s in s_kv_values:
                    bt = results[baseline].get((page, batch, s), float("nan"))
                    pt = results[p].get((page, batch, s), float("nan"))
                    if pt == pt and bt == bt and pt > 0:
                        r = bt / pt
                        ratios.append(r)
                        row += f"{r:>10.2f}"
                    else:
                        row += f"{'N/A':>10}"
                print(row)
            if ratios:
                print(
                    f"\n  geomean {np.exp(np.mean(np.log(ratios))):.2f}x  "
                    f"min {min(ratios):.2f}x  max {max(ratios):.2f}x  "
                    f"({sum(1 for r in ratios if r > 1)}/{len(ratios)} shapes {p} faster)"
                )


def write_csv(path: str, results: Dict[str, Dict[Tuple[int, int, int], float]]):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            ["provider", "config", "batch", "s_kv", "page_size", "head_dim", "median_ms"]
        )
        for provider, d in results.items():
            for (page, batch, seq), ms in sorted(d.items()):
                w.writerow(
                    [provider, "paged", batch, seq, page, HEAD_DIM, f"{ms:.6f}"]
                )
    print(f"\nWrote {path}")


def create_heatmap(
    page: int,
    batch_values: List[int],
    s_kv_values: List[int],
    results: Dict[str, Dict[Tuple[int, int, int], float]],
    provider: str,
    baseline: str,
    output_file: str,
):
    """Heatmap of `provider` speedup vs `baseline` over (batch x s_kv) for one page."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap")
        return
    if provider not in results or baseline not in results:
        return
    mat = np.full((len(batch_values), len(s_kv_values)), np.nan)
    for i, batch in enumerate(batch_values):
        for j, s in enumerate(s_kv_values):
            bt = results[baseline].get((page, batch, s), float("nan"))
            pt = results[provider].get((page, batch, s), float("nan"))
            if pt == pt and bt == bt and pt > 0:
                mat[i, j] = bt / pt
    if np.all(np.isnan(mat)):
        return
    fig, ax = plt.subplots(
        figsize=(max(6, 1.5 * len(s_kv_values)), max(5, 0.5 * len(batch_values)))
    )
    norm = mcolors.TwoSlopeNorm(
        vmin=min(0.5, np.nanmin(mat)), vcenter=1.0, vmax=max(1.5, np.nanmax(mat))
    )
    im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(f"{baseline}_ms / {provider}_ms", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(s_kv_values)))
    ax.set_yticks(np.arange(len(batch_values)))
    ax.set_xticklabels([str(s) for s in s_kv_values])
    ax.set_yticklabels([str(b) for b in batch_values])
    for i in range(len(batch_values)):
        for j in range(len(s_kv_values)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 0.7 or mat[i, j] > 1.5 else "black")
    ax.set_xlabel("s_kv")
    ax.set_ylabel("batch")
    ax.set_title(
        f"paged prefill (page={page}): {provider} speedup vs {baseline}\n"
        f"(>1.0 = {provider} faster)"
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark paged prefill backends (cuTile vs trtllm SOTA)"
    )
    parser.add_argument(
        "--providers",
        type=str,
        default=",".join(ALL_PROVIDERS),
        help=f"Comma-separated subset of {ALL_PROVIDERS}",
    )
    parser.add_argument(
        "--baseline", type=str, default="trtllm", help="Speedup baseline provider"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="paged_prefill_backend_comparison"
    )
    parser.add_argument("--csv", type=str, default=None, help="Write raw medians to CSV")
    args = parser.parse_args()

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    providers = [p for p in requested if p in ALL_PROVIDERS]
    available = [p for p in providers if _AVAIL[p]()]
    print(f"Requested providers: {providers}")
    print(f"Available here:      {available}  (unavailable are skipped as N/A)")

    # Ocean PrefillPaged workload: page {16,64} x batch {1,16} x s_kv {1024..8192}.
    page_values = [16, 64]
    batch_values = [1, 16]
    s_kv_values = [1024, 2048, 4096, 8192]

    print("\nCorrectness check (cutile vs trtllm output):")
    verify_correctness(1, 1024, 16, providers)

    results = run_benchmark_sweep(page_values, batch_values, s_kv_values, providers)
    baseline = args.baseline if args.baseline in providers else providers[0]
    print_summary_table(
        page_values, batch_values, s_kv_values, results, providers, baseline
    )

    if args.csv:
        write_csv(args.csv, results)

    for p in providers:
        if p != baseline:
            for page in page_values:
                create_heatmap(
                    page, batch_values, s_kv_values, results, p, baseline,
                    f"{args.output_prefix}_{p}_vs_{baseline}_page{page}.png",
                )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
