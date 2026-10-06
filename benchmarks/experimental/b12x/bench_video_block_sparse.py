# SPDX-License-Identifier: Apache-2.0
"""Dense vs block-list sparse attention at MiniMax-H3 geometry.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_video_block_sparse.py

The sparse list is a per-tile +/- radius window of 64-token blocks, so the density is about
(2*radius+1)/num_k_blocks -- the FastH3 VSA operating point is ~90% sparsity (density 0.10).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess

import torch

from b12x.attention import varlen
from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from b12x.preparation._measurement import _prepare_race, measure_race_steps

TILE_M, BLOCK_K = 128, 64


def _window(S, radius, device):
    num_blocks = (S + BLOCK_K - 1) // BLOCK_K
    num_tiles = (S + TILE_M - 1) // TILE_M
    idx, off = [], [0]
    for m in range(num_tiles):
        c = (m * TILE_M) // BLOCK_K
        idx.extend(range(max(0, c - radius), min(num_blocks, c + radius + 1)))
        off.append(len(idx))
    return (torch.tensor(idx, device=device, dtype=torch.int32),
            torch.tensor(off, device=device, dtype=torch.int32))


def _check_output(output, q, k, v, indices, offsets):
    for start in range(0, q.shape[0], TILE_M):
        stop = min(start + TILE_M, q.shape[0])
        if indices is None:
            keys, values = k, v
        else:
            tile = start // TILE_M
            blocks = indices[int(offsets[tile]):int(offsets[tile + 1])].tolist()
            rows = [row for block in blocks
                    for row in range(block * BLOCK_K, min((block + 1) * BLOCK_K, k.shape[0]))]
            keys, values = k[rows], v[rows]
        qq = q[start:stop].transpose(0, 1).float()
        kk, vv = keys.transpose(0, 1).float(), values.transpose(0, 1).float()
        expected = ((qq @ kk.transpose(-1, -2)) * q.shape[-1] ** -0.5).softmax(-1) @ vv
        actual = output[start:stop].transpose(0, 1).float()
        assert torch.isfinite(actual).all()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert cosine >= 0.9999, (start, float(cosine))


def _time(S, H, D, radius, device, samples, warmup, rounds):
    torch.manual_seed(17)
    q = torch.randn(S, H, D, device=device, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    cu = torch.tensor([0, S], device=device, dtype=torch.int32)
    indices, offsets = _window(S, radius, device)
    calls, graphs, owners = [], [], []
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        for sparse in (False, True):
            bi, bo = (indices, offsets) if sparse else (None, None)
            declaration = varlen.plan(
                q, k, v, cu, cu, max_seqlen_q=S, max_seqlen_k=S, causal=False,
                block_sparse=sparse, num_q_tiles=((S + TILE_M - 1) // TILE_M if sparse else 0),
                total_blocks_cap=(indices.numel() if sparse else 0),
                override=varlen.VarlenAttentionConfig(tile_m=TILE_M, tile_n=BLOCK_K),
            )

            def prepare(state):
                spec, = state.scratch_plan.scratch_specs()
                scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
                binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                     cu_seqlens_k=cu, block_indices=bi, block_offsets=bo)
                return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

            session.prepare((declaration.request(name=f"block-sparse-{sparse}", prepare_call=prepare),))
            state = require_prepared(declaration, "attention.varlen", device)
            spec, = state.scratch_plan.scratch_specs()
            scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
            binding = state.bind(scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cu,
                                 cu_seqlens_k=cu, block_indices=bi, block_offsets=bo)
            output, _ = state.run(binding)
            _check_output(output, q, k, v, bi, bo)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state.run(binding)
            calls.append(PreparedCall(run=graph.replay, produce=lambda: None, owners=(graph, state, binding, scratch)))
            graphs.append(graph)
            owners.append((output, bi, bo))
        session.freeze()
        for _ in range(warmup):
            for graph in graphs:
                graph.replay()
        for output, _, _ in owners:
            output.fill_(float("nan"))
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
        for output, bi, bo in owners:
            _check_output(output, q, k, v, bi, bo)
        race = _prepare_race(tuple(calls), device_ordinal=device.index, samples=samples, primed=True)
        timings, seen = [], 0
        try:
            for _ in measure_race_steps(race, device_ordinal=device.index, rounds=rounds):
                if race.completed_rounds != seen:
                    seen = race.completed_rounds
                    timings.append(tuple(race.latest_round_us))
            if race.completed_rounds != seen:
                timings.append(tuple(race.latest_round_us))
        finally:
            race.close()
            for graph in graphs:
                graph.reset()
    density = indices.numel() / (((S + TILE_M - 1) // TILE_M) * ((S + BLOCK_K - 1) // BLOCK_K))
    return timings, density


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=18748, help="packed tokens in one local attention segment")
    ap.add_argument("--heads", type=int, default=56)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--radius", type=int, default=14, help="blocks; ~90% sparsity at 293 blocks")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=15)
    a = ap.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if min(a.seq, a.heads, a.head_dim, a.iters, a.rounds) <= 0 or a.radius < 0 or a.warmup < 0:
        ap.error("dimensions and sample counts must be positive; radius and warmup must be nonnegative")
    timings, density = _time(a.seq, a.heads, a.head_dim, a.radius, device, a.iters, a.warmup, a.rounds)
    dense_us, sparse_us = (statistics.median(row[i] for row in timings) for i in (0, 1))
    print(json.dumps({
        "gpu": props.name, "uuid": str(props.uuid), "sm_count": props.multi_processor_count,
        "commit": commit, "torch": torch.__version__, "cuda": torch.version.cuda,
        "arguments": vars(a), "tile_m": TILE_M, "tile_n": BLOCK_K,
        "method": "balanced cold-L2 graph replay using preparation timing; lower microseconds is better",
        "correctness": "full output against chunked FP32 Torch oracle; poisoned graph replay without allocation",
        "round_us_dense_sparse": timings, "dense_us": dense_us, "sparse_us": sparse_us,
        "density": density, "dense_over_sparse": dense_us / sparse_us,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
