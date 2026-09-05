"""Microbench: GEMM2 kernel cost of tile-ready signaling.

Times the CuteDSL NVFP4 GEMM2 (finalize-fusion) kernel only — no consumer,
no dest fingerprint, no dispatch/combine. Same problem, same MMA, same SM
count; the only compile-time change is ``enable_tile_signal`` (destination-
complete bulk wait + ``st.release.gpu`` of one int32 flag per CTA tile).

Also reports ``store_permuted_c`` vs fused-finalize so the dense-C store is
not confused with the flag tax.

Geometry defaults to the fused-gemm2-combine reference cell on one rank:
32 local experts, 1024 packed rows/expert (8 GPU × 128 tokens), hidden 7168,
intermediate 2048, MMA 128×128 cluster (1,1).

    python benchmarks/bench_gemm2_tile_signal.py
    python benchmarks/bench_gemm2_tile_signal.py --world 4 --num-experts 256

Submit (1 GPU):

    sbatch --gpus-per-node=1 --nodes=1 \\
        dev/cutedsl_moe/run_bench_gemm2_tile_signal.sbatch
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Optional

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _here]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument(
        "--num-local-experts",
        type=int,
        default=None,
        help="Override local expert count (default: num_experts // world)",
    )
    p.add_argument(
        "--world", type=int, default=8, help="EP world size used to size the pack"
    )
    p.add_argument("--tokens-per-rank", type=int, default=128)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--intermediate", type=int, default=2048)
    p.add_argument("--mma-m", type=int, default=128)
    p.add_argument("--mma-n", type=int, default=128)
    p.add_argument("--cluster-m", type=int, default=1)
    p.add_argument("--cluster-n", type=int, default=1)
    p.add_argument(
        "--reserve-consumer-sms",
        type=int,
        default=0,
        help="Subtract this many SMs from every GEMM2 launch. Keep 0 to isolate "
        "the flag store from occupancy; production overlap reserves 8.",
    )
    p.add_argument(
        "--include-overlap-sms",
        action="store_true",
        default=True,
        help="Also time dense_c+signal with 8 SMs reserved (production occupancy).",
    )
    p.add_argument(
        "--no-include-overlap-sms",
        action="store_false",
        dest="include_overlap_sms",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument(
        "--cupti", action="store_true", help="Time kernels with CUPTI instead of graphs"
    )
    p.add_argument(
        "--epilogues",
        default="fused,dense_c",
        help="Comma list: fused (atomic finalize) and/or dense_c (store_permuted_c)",
    )
    return p.parse_args()


@dataclass(frozen=True)
class Problem:
    a: Any
    a_scale: Any
    b: Any
    b_scale: Any
    alpha: Any
    tile_idx_to_expert_idx: Any
    num_non_exiting_tiles: Any
    tile_idx_to_mn_limit: Any
    permuted_idx_to_expanded_idx: Any
    token_final_scales: Any
    permuted_m: int
    seq_len: int
    n: int
    k: int
    num_experts: int
    flags: int


def _build_problem(
    *,
    num_local_experts: int,
    cap: int,
    hidden: int,
    intermediate: int,
    mma_tiler_mn: tuple[int, int],
    tile_tokens_dim: int,
    device,
) -> Problem:
    import torch

    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        gemm2_tile_ready_numel,
    )
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

    seq_len = num_local_experts * cap
    selected = (
        torch.arange(num_local_experts, device=device, dtype=torch.int32)
        .repeat_interleave(cap)
        .reshape(seq_len, 1)
    )
    scales = torch.ones(seq_len, 1, dtype=torch.float32, device=device)
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _expanded_to_permuted,
        permuted_idx,
        _padded,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=selected,
        token_final_scales=scales,
        num_experts=num_local_experts,
        top_k=1,
        num_local_experts=num_local_experts,
        tile_tokens_dim=tile_tokens_dim,
    )
    permuted_m = int(permuted_idx.shape[0])

    gs = torch.ones(1, dtype=torch.float32, device=device)
    act = torch.randn(permuted_m, intermediate, dtype=torch.bfloat16, device=device)
    a, a_sf = fp4_quantize(
        act, global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
    )
    a_scale = convert_sf_to_mma_layout(
        a_sf, m=permuted_m, k=intermediate, num_groups=1, sf_vec_size=16
    )

    w2 = torch.randn(
        num_local_experts, hidden, intermediate, dtype=torch.bfloat16, device=device
    )
    w2_q, w2_sf = fp4_quantize(
        w2.reshape(num_local_experts * hidden, intermediate),
        global_scale=gs,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    b = w2_q.view(num_local_experts, hidden, intermediate // 2)
    b_scale = convert_sf_to_mma_layout(
        w2_sf,
        m=hidden,
        k=intermediate,
        num_groups=num_local_experts,
        sf_vec_size=16,
    )
    alpha = torch.ones(num_local_experts, dtype=torch.float32, device=device)
    flags = gemm2_tile_ready_numel(permuted_m, hidden, mma_tiler_mn)
    return Problem(
        a=a,
        a_scale=a_scale,
        b=b,
        b_scale=b_scale,
        alpha=alpha,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx,
        token_final_scales=scales,
        permuted_m=permuted_m,
        seq_len=seq_len,
        n=hidden,
        k=intermediate,
        num_experts=num_local_experts,
        flags=flags,
    )


def _out_buf(prob: Problem, store_permuted_c: bool, device):
    import torch

    rows = prob.permuted_m if store_permuted_c else prob.seq_len
    alloc = torch.empty if store_permuted_c else torch.zeros
    return alloc((rows, prob.n), dtype=torch.bfloat16, device=device)


def _aligned_sm_count(
    sm_full: int, reserve: int, cluster_shape_mn: tuple[int, int]
) -> int:
    cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
    usable = max(cluster_size, int(sm_full) - int(reserve))
    usable = (usable // cluster_size) * cluster_size
    return max(cluster_size, usable)


def _launch(
    prob: Problem,
    *,
    out,
    tile_ready,
    store_permuted_c: bool,
    mma_tiler_mn: tuple[int, int],
    cluster_shape_mn: tuple[int, int],
    sm_count: int,
):
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
    )

    return blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
        a=prob.a,
        b=prob.b,
        a_scale=prob.a_scale,
        b_scale=prob.b_scale,
        alpha=prob.alpha,
        tile_idx_to_expert_idx=prob.tile_idx_to_expert_idx,
        num_non_exiting_tiles=prob.num_non_exiting_tiles,
        tile_idx_to_mn_limit=prob.tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=prob.permuted_idx_to_expanded_idx,
        token_final_scales=prob.token_final_scales,
        out=out,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        enable_pdl=True,
        use_fused_finalize=True,
        tile_ready=tile_ready,
        store_permuted_c=store_permuted_c,
    )


def _time_ms(
    fn,
    *,
    warmup: int,
    repeat: int,
    use_cuda_graph: bool,
    enable_cupti: bool,
) -> float:
    import numpy as np
    import torch

    from flashinfer.testing.utils import bench_gpu_time

    # JIT / occupancy query live here, not in the timed region.
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()

    kwargs = dict(
        dry_run_iters=max(1, warmup),
        repeat_iters=repeat,
        enable_cupti=enable_cupti,
        use_cuda_graph=False if enable_cupti else use_cuda_graph,
        # Problem is far larger than L2; skip rotating-buffer clones.
        cold_l2_cache=False,
        num_iters_within_graph=10,
    )
    try:
        times = bench_gpu_time(fn, **kwargs)
    except Exception as exc:
        if enable_cupti or not use_cuda_graph:
            raise
        print(
            f"CUDA graph timing failed ({exc!r}); falling back to CUDA events",
            flush=True,
        )
        kwargs["use_cuda_graph"] = False
        times = bench_gpu_time(fn, **kwargs)
    return float(np.median(times))


def main() -> int:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise SystemExit("GEMM2 tile-signal microbench requires SM100+")

    from flashinfer.cute_dsl.utils import get_num_sm

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    if args.num_local_experts is not None:
        nle = int(args.num_local_experts)
    else:
        if args.num_experts % args.world != 0:
            raise SystemExit(
                f"num_experts={args.num_experts} not divisible by world={args.world}"
            )
        nle = args.num_experts // args.world
    if nle < 1:
        raise SystemExit("need at least one local expert")
    cap = args.tokens_per_rank * args.world
    mma_tiler_mn = (args.mma_m, args.mma_n)
    cluster_shape_mn = (args.cluster_m, args.cluster_n)
    sm_full = int(get_num_sm(device))
    sm_count = _aligned_sm_count(
        sm_full, int(args.reserve_consumer_sms), cluster_shape_mn
    )
    epilogues = [s.strip() for s in args.epilogues.split(",") if s.strip()]
    for name in epilogues:
        if name not in ("fused", "dense_c"):
            raise SystemExit(f"unknown epilogue {name!r}; use fused and/or dense_c")

    print(
        f"geometry local_experts={nle} cap={cap} tokens={nle * cap} "
        f"hidden={args.hidden} intermediate={args.intermediate} "
        f"mma={mma_tiler_mn} cluster={cluster_shape_mn} "
        f"sm={sm_count}/{sm_full} reserve={args.reserve_consumer_sms} "
        f"cc={'.'.join(map(str, torch.cuda.get_device_capability()))}",
        flush=True,
    )
    print("building problem (moe_sort + NVFP4 quantize) ...", flush=True)
    prob = _build_problem(
        num_local_experts=nle,
        cap=cap,
        hidden=args.hidden,
        intermediate=args.intermediate,
        mma_tiler_mn=mma_tiler_mn,
        tile_tokens_dim=args.mma_m,
        device=device,
    )
    print(
        f"permuted_m={prob.permuted_m} flags={prob.flags} "
        f"non_exiting_tiles={int(prob.num_non_exiting_tiles.item())}",
        flush=True,
    )

    use_graph = not args.no_cuda_graph
    rows: list[tuple[str, bool, int, float]] = []
    baseline_us: dict[tuple[str, int], float] = {}
    out_bufs = {name: _out_buf(prob, name == "dense_c", device) for name in epilogues}

    def _time_one(epi: str, signal: bool, sm: int) -> float:
        store_c = epi == "dense_c"
        out = out_bufs[epi]
        tile_ready: Optional[torch.Tensor] = None
        if signal:
            tile_ready = torch.zeros(prob.flags, dtype=torch.int32, device=device)

        def _fn(_out=out, _tr=tile_ready, _sc=store_c, _sm=sm):
            _launch(
                prob,
                out=_out,
                tile_ready=_tr,
                store_permuted_c=_sc,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                sm_count=_sm,
            )

        tag = f"{epi} signal={int(signal)} sm={sm}"
        print(f"timing {tag} (JIT on first call) ...", flush=True)
        ms = _time_ms(
            _fn,
            warmup=args.warmup,
            repeat=args.repeat,
            use_cuda_graph=use_graph,
            enable_cupti=bool(args.cupti),
        )
        us = ms * 1e3
        rows.append((epi, signal, sm, us))
        if not signal:
            baseline_us[(epi, sm)] = us
        base = baseline_us.get((epi, sm), us)
        delta = us - base
        pct = 100.0 * delta / base if base > 0 else float("nan")
        print(
            f"BENCH_CSV,{epi},{int(signal)},{sm},"
            f"{args.mma_m}x{args.mma_n},{args.cluster_m}x{args.cluster_n},"
            f"{us:.1f},{delta:.1f},{pct:.2f},{prob.flags},{prob.permuted_m}",
            flush=True,
        )
        return us

    print(
        "BENCH_CSV,epilogue,signal,sm,mma,cluster,gemm_us,delta_us,delta_pct,flags,"
        "permuted_m"
    )
    for epi in epilogues:
        for signal in (False, True):
            _time_one(epi, signal, sm_count)

    overlap_us = None
    overlap_sm = None
    if args.include_overlap_sms and "dense_c" in epilogues:
        overlap_sm = _aligned_sm_count(sm_full, 8, cluster_shape_mn)
        if overlap_sm != sm_count:
            overlap_us = _time_one("dense_c", True, overlap_sm)

    print("\nTile-signal tax (same epilogue, same SMs, same MMA):")
    for epi, signal, sm, us in rows:
        if not signal:
            continue
        base = baseline_us.get((epi, sm))
        if base is None:
            continue
        print(
            f"  {epi} sm={sm}: {base:.1f} us -> {us:.1f} us  "
            f"({us - base:+.1f} us, {100.0 * (us - base) / base:+.2f}%)"
        )
    if overlap_us is not None:
        signal_full = next(
            (
                us
                for epi, signal, sm, us in rows
                if epi == "dense_c" and signal and sm == sm_count
            ),
            None,
        )
        if signal_full is not None:
            print(
                f"\nOverlap SM reservation (dense_c+signal, {sm_count} -> {overlap_sm} SMs): "
                f"{signal_full:.1f} us -> {overlap_us:.1f} us  "
                f"({overlap_us - signal_full:+.1f} us, "
                f"{100.0 * (overlap_us - signal_full) / signal_full:+.2f}%)"
            )
    print(
        "Notes: no consumer, no dest fingerprint, no tile_ready.zero_ in the "
        "timed path. enable_tile_signal = dest-complete bulk wait + st.release.gpu."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
