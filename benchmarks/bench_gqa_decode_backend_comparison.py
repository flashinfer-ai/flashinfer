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

Benchmark: GQA Decode Backend Comparison (cuTile vs trtllm-gen SOTA)

Compares FlashInfer's cuTile GQA-decode backend against the trtllm-gen SOTA
kernel over ocean-eval's DecodePaged workload (LLaMA-style GQA: 128 query heads
/ 8 KV heads, head_dim=128, bf16), sweeping batch x s_kv x page_size:

  - ``cutile`` : this checkout's ``BatchDecodeWithPagedKVCacheWrapper`` planned/run
                 with ``backend="cutile"`` (pure cuda.tile kernel, kv_layout="NHD").
  - ``trtllm`` : ``flashinfer.decode.trtllm_batch_decode_with_kv_cache`` (trtllm-gen,
                 kv_layout="HND"), the ocean-eval dashboard SOTA baseline.

TWO-FLASHINFER NOTE: the two providers are measured in SEPARATE processes
(this checkout for ``cutile``; the image's installed flashinfer for ``trtllm``).
Each provider independently try/except-es and degrades to N/A (NaN), so the two
result CSVs can be produced separately and merged. Baseline default = trtllm.

Usage (typical two-run merge):
    python3 bench_gqa_decode_backend_comparison.py --providers cutile --csv cutile.csv
    python3 bench_gqa_decode_backend_comparison.py --providers trtllm --csv trtllm.csv
    # then concatenate cutile.csv + trtllm.csv (drop the duplicate header)

Requirements:
    - cuda.tile toolchain (cutile provider)
    - flashinfer with trtllm-gen cubins (trtllm provider)
    - matplotlib for the optional heatmap
"""

import argparse
import csv as _csv
import math
import numpy as np
import torch
from typing import Callable, Dict, List, Tuple

from flashinfer.testing.utils import bench_gpu_time

# LLaMA-style GQA problem geometry (fixed).
NUM_QO_HEADS = 128
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.bfloat16


# --------------------------------------------------------------------------- #
# Provider availability
# --------------------------------------------------------------------------- #
def _has_cutile() -> bool:
    try:
        import cuda.tile  # noqa: F401
        from flashinfer.decode import (  # noqa: F401
            BatchDecodeWithPagedKVCacheWrapper,
        )

        return True
    except Exception:
        return False


def _has_trtllm() -> bool:
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # noqa: F401

        return True
    except Exception:
        return False


ALL_PROVIDERS = ["cutile", "trtllm"]
_AVAIL = {
    "cutile": _has_cutile,
    "trtllm": _has_trtllm,
}


# --------------------------------------------------------------------------- #
# Shared paged-GQA input generation
# --------------------------------------------------------------------------- #
class _GQAInputs:
    """Raw tensors shared by both providers (identical values -> comparable outputs).

    KV cache is generated in NHD layout ``[num_pages, page, num_kv_heads, head_dim]``;
    the trtllm (HND) path takes a transposed view of the same values.
    """

    def __init__(self, batch: int, s_kv: int, page_size: int, device):
        torch.manual_seed(0)
        pages_per_batch = math.ceil(s_kv / page_size)
        total_pages = max(batch * pages_per_batch + 8, 64)

        self.batch = batch
        self.s_kv = s_kv
        self.page_size = page_size
        self.pages_per_batch = pages_per_batch
        self.total_pages = total_pages
        self.sm_scale = 1.0 / math.sqrt(HEAD_DIM)

        # q: [batch, num_qo_heads, head_dim] (one query token per request).
        self.q = torch.randn(
            batch, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        # NHD KV cache.
        self.k_cache = torch.randn(
            total_pages, page_size, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        self.v_cache = torch.randn(
            total_pages, page_size, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device
        )
        self.kv_lens = torch.full((batch,), s_kv, dtype=torch.int32, device=device)
        # Dense block table [batch, pages_per_batch], int32.
        self.page_table = (
            torch.arange(batch * pages_per_batch, dtype=torch.int32, device=device)
            .reshape(batch, pages_per_batch)
            % total_pages
        )
        # CSR plan state for the paged-KV wrapper (cutile path).
        rem = s_kv % page_size
        last = rem if rem != 0 else page_size
        self.indptr = (
            torch.arange(batch + 1, dtype=torch.int32, device=device) * pages_per_batch
        )
        self.indices = self.page_table.reshape(-1).to(torch.int32)
        self.last_page_len = torch.full(
            (batch,), last, dtype=torch.int32, device=device
        )


# --------------------------------------------------------------------------- #
# Per-provider callables
# --------------------------------------------------------------------------- #
def _make_execute(provider: str, inp: _GQAInputs) -> Callable[[], torch.Tensor]:
    device = inp.q.device

    if provider == "cutile":
        # This checkout's BatchDecodeWithPagedKVCacheWrapper(..., backend="cutile").
        # cuTile decode requires kv_layout="NHD" and materializes the block table at
        # plan() time from indptr/indices (the cutile branch of the paged-decode wrapper).
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", backend="cutile"
        )
        wrapper.plan(
            inp.indptr,
            inp.indices,
            inp.last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            inp.page_size,
            q_data_type=inp.q.dtype,
            kv_data_type=inp.k_cache.dtype,
            sm_scale=inp.sm_scale,
        )

        def run():
            return wrapper.run(inp.q, (inp.k_cache, inp.v_cache))

    elif provider == "trtllm":
        # SOTA: flashinfer.decode.trtllm_batch_decode_with_kv_cache (trtllm-gen).
        # Call body follows the trtllm-gen GQA decode reference.
        # HND cache is a transposed (contiguous) view of the same NHD values.
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device)
        k_hnd = inp.k_cache.transpose(1, 2).contiguous()  # [pages, kv_heads, page, dim]
        v_hnd = inp.v_cache.transpose(1, 2).contiguous()

        def run():
            return trtllm_batch_decode_with_kv_cache(
                query=inp.q,
                kv_cache=(k_hnd, v_hnd),
                workspace_buffer=workspace,
                block_tables=inp.page_table,
                seq_lens=inp.kv_lens.view(-1),
                max_seq_len=inp.s_kv,
                bmm1_scale=inp.sm_scale,
                bmm2_scale=1.0,
                kv_layout="HND",
            )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return run


def _canonical_out(out) -> torch.Tensor:
    """Reduce a provider output to [batch, num_qo_heads, head_dim] for comparison."""
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out


# --------------------------------------------------------------------------- #
# Correctness + timing
# --------------------------------------------------------------------------- #
def verify_correctness(
    batch: int, s_kv: int, page_size: int, providers: List[str]
) -> Tuple[bool, float]:
    """Cosine similarity of cutile vs trtllm GQA-decode output (both must be available)."""
    if not ("cutile" in providers and "trtllm" in providers):
        print("  correctness: need both cutile and trtllm selected -> SKIP")
        return False, float("nan")
    if not (_AVAIL["cutile"]() and _AVAIL["trtllm"]()):
        return False, float("nan")
    try:
        device = torch.device("cuda")
        inp = _GQAInputs(batch, s_kv, page_size, device)
        a = _canonical_out(_make_execute("cutile", inp)())
        b = _canonical_out(_make_execute("trtllm", inp)())
        torch.cuda.synchronize()
        cos = torch.nn.functional.cosine_similarity(
            a.float().reshape(1, -1), b.float().reshape(1, -1)
        ).item()
        return cos > 0.99, cos
    except Exception:
        return False, float("nan")


def bench_one(batch: int, s_kv: int, page_size: int, provider: str) -> float:
    """Median latency (ms) for one (config, provider); NaN if unavailable/failed."""
    if not _AVAIL[provider]():
        return float("nan")
    try:
        device = torch.device("cuda")
        inp = _GQAInputs(batch, s_kv, page_size, device)
        run = _make_execute(provider, inp)
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
    batch_values: List[int],
    s_kv_values: List[int],
    page_size_values: List[int],
    providers: List[str],
) -> Dict[str, Dict[Tuple[int, int, int], float]]:
    """Return {provider: {(batch, s_kv, page_size): median_ms}}."""
    results: Dict[str, Dict[Tuple[int, int, int], float]] = {p: {} for p in providers}
    configs = [
        (b, s, p) for b in batch_values for s in s_kv_values for p in page_size_values
    ]
    total = len(configs)

    print(
        f"\nBenchmarking GQA decode  qo_heads={NUM_QO_HEADS} kv_heads={NUM_KV_HEADS} "
        f"head_dim={HEAD_DIM}  bf16"
    )
    print("=" * (34 + 12 * len(providers)))
    header = f"{'batch':>6} {'s_kv':>7} {'page':>5} |"
    for p in providers:
        header += f" {p:>10}"
    print(header)
    print("-" * (34 + 12 * len(providers)))

    for current, (b, s, ps) in enumerate(configs, 1):
        row = f"{b:>6} {s:>7} {ps:>5} |"
        for p in providers:
            ms = bench_one(b, s, ps, p)
            results[p][(b, s, ps)] = ms
            row += f" {ms:>10.5f}" if ms == ms else f" {'--':>10}"
        print(f"[{current:3d}/{total}] " + row)

    return results


def print_summary_table(
    batch_values: List[int],
    s_kv_values: List[int],
    page_size_values: List[int],
    results: Dict[str, Dict[Tuple[int, int, int], float]],
    providers: List[str],
    baseline: str,
):
    """Speedup of each provider vs baseline (>1 = provider faster)."""
    if baseline not in results:
        return
    for p in providers:
        if p == baseline:
            continue
        print(f"\n{'=' * 72}")
        print(f"Speedup: {p} vs {baseline} (baseline_ms / {p}_ms;  >1 = {p} faster)")
        print(f"{'=' * 72}")
        print(f"{'batch':>6} {'s_kv':>7} {'page':>5} {'speedup':>10}")
        print("-" * 32)
        ratios = []
        for b in batch_values:
            for s in s_kv_values:
                for ps in page_size_values:
                    bt = results[baseline].get((b, s, ps), float("nan"))
                    pt = results[p].get((b, s, ps), float("nan"))
                    if pt == pt and bt == bt and pt > 0:
                        r = bt / pt
                        ratios.append(r)
                        print(f"{b:>6} {s:>7} {ps:>5} {r:>10.2f}")
                    else:
                        print(f"{b:>6} {s:>7} {ps:>5} {'N/A':>10}")
        if ratios:
            print(
                f"\n  geomean {np.exp(np.mean(np.log(ratios))):.2f}x  "
                f"min {min(ratios):.2f}x  max {max(ratios):.2f}x  "
                f"({sum(1 for r in ratios if r > 1)}/{len(ratios)} configs {p} faster)"
            )


def write_csv(path: str, results: Dict[str, Dict[Tuple[int, int, int], float]]):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(
            ["provider", "config", "batch", "s_kv", "page_size", "num_heads", "median_ms"]
        )
        for provider, d in results.items():
            for (b, s, ps), ms in sorted(d.items()):
                config = f"b{b}_s{s}_p{ps}"
                w.writerow([provider, config, b, s, ps, NUM_QO_HEADS, f"{ms:.6f}"])
    print(f"\nWrote {path}")


def create_heatmap(
    batch_values: List[int],
    s_kv_values: List[int],
    page_size: int,
    results: Dict[str, Dict[Tuple[int, int, int], float]],
    provider: str,
    baseline: str,
    output_file: str,
):
    """Heatmap of `provider` speedup vs `baseline` over (batch x s_kv) at one page_size."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not installed, skipping heatmap")
        return
    if provider not in results or baseline not in results:
        return
    mat = np.full((len(batch_values), len(s_kv_values)), np.nan)
    for i, b in enumerate(batch_values):
        for j, s in enumerate(s_kv_values):
            bt = results[baseline].get((b, s, page_size), float("nan"))
            pt = results[provider].get((b, s, page_size), float("nan"))
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
                ax.text(
                    j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] < 0.7 or mat[i, j] > 1.5 else "black",
                )
    ax.set_xlabel("s_kv")
    ax.set_ylabel("batch")
    ax.set_title(
        f"GQA decode (page_size={page_size}): {provider} speedup vs {baseline}\n"
        f"(>1.0 = {provider} faster)"
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GQA decode: cuTile vs trtllm-gen SOTA"
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
    parser.add_argument("--output-prefix", type=str, default="gqa_decode_backend")
    parser.add_argument("--csv", type=str, default=None, help="Write medians to CSV")
    args = parser.parse_args()

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    providers = [p for p in requested if p in ALL_PROVIDERS]
    available = [p for p in providers if _AVAIL[p]()]
    print(f"Requested providers: {providers}")
    print(f"Available here:      {available}  (unavailable are skipped as N/A)")

    # Ocean DecodePaged workload (GQA).
    batch_values = [1, 16, 64, 128, 256]
    s_kv_values = [1024, 2048, 4096, 8192]
    page_size_values = [32, 64]

    # Correctness only runs when BOTH providers are importable in this process.
    print("\nCorrectness (cutile vs trtllm cosine-sim, at batch=16 s_kv=2048 page=64):")
    ok, cos = verify_correctness(16, 2048, 64, providers)
    if cos == cos:
        print(f"  {'OK ' if ok else 'FAIL'} cos={cos:.4f}")
    else:
        print("  N/A (need both cutile and trtllm in one process)")

    results = run_benchmark_sweep(
        batch_values, s_kv_values, page_size_values, providers
    )
    baseline = args.baseline if args.baseline in providers else providers[0]
    print_summary_table(
        batch_values, s_kv_values, page_size_values, results, providers, baseline
    )

    if args.csv:
        write_csv(args.csv, results)

    for p in providers:
        if p != baseline:
            for ps in page_size_values:
                create_heatmap(
                    batch_values, s_kv_values, ps, results, p, baseline,
                    f"{args.output_prefix}_{p}_vs_{baseline}_p{ps}.png",
                )

    print("\n" + "=" * 72)
    print("BENCHMARK COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
