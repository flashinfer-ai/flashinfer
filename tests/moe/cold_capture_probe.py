"""Fresh-process probe of the first-use CUDA-graph capture contract of the SM12x W4A4 MoE wrapper.

    python cold_capture_probe.py <case> [--prewarm]

Cases (E=8, H=256, top-k 2; capacity wrapper, use_cuda_graph=True):
  static          I=320, M=64      first call is the static kernel
  gated_dynamic   I=320, M=1024    first call is the branch-paired gated dynamic kernel (tile M128)
  generic_dynamic I=640, M=1024    first call is the generic dynamic kernel on legacy views (I > 512)
  single_slice    I=64,  M=64      first call is the static kernel with the 256-aligned static-family operands
Without --prewarm the very first forward is issued inside a CUDA-graph capture: the contract is a clear refusal
(RuntimeError naming the warm-up) and no preparation state left behind.  With --prewarm one eager call precedes the
capture, the capture must succeed and the replay must match the eager output.  One JSON line is printed with the
observed outcome and the cache state before / after so callers (tests, the compatibility matrix) can assert it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cache_state():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    return {
        "folded_w1_alpha": None
        if _WRAPPER is None
        else (
            None
            if _WRAPPER._folded_w1_alpha is None
            else int(_WRAPPER._folded_w1_alpha.data_ptr())
        ),
        "folded_w1_alpha_key": None
        if _WRAPPER is None
        else (
            None
            if _WRAPPER._folded_w1_alpha_key is None
            else [int(v) for v in _WRAPPER._folded_w1_alpha_key]
        ),
        "allocated_bytes": int(__import__("torch").cuda.memory_allocated()),
        "graph_pool_bytes": _graph_pool_bytes(),
        "padded_scale_cache": len(md._PADDED_SCALE_CACHE),
        "padded_fp4_cache": len(md._PADDED_FP4_CACHE),
        "weight_cache": len(md._WEIGHT_CACHE),
        "static_kernels": len(md._STATIC_KERNEL_CACHE),
        "dynamic_kernels": len(md._DYNAMIC_KERNEL_CACHE),
        "micro_kernels": len(md._MICRO_KERNEL_CACHE),
        "direct_micro_kernels": len(md._DIRECT_MICRO_KERNEL_CACHE),
        "workspaces": len(md._WORKSPACE_CACHE),
    }


def _graph_pool_bytes():
    """Active bytes held by CUDA-graph private memory pools: the per-call temporaries a
    captured forward legitimately allocates inside the graph (they replay with it)."""
    import torch

    total = 0
    for seg in torch.cuda.memory._snapshot()["segments"]:
        if tuple(seg.get("segment_pool_id", (0, 0))) == (0, 0):
            continue
        total += sum(
            b["size"] for b in seg["blocks"] if b["state"] == "active_allocated"
        )
    return int(total)


def _graph_overhead_bytes():
    """Default-pool bytes ``torch.cuda.graph`` itself keeps per live graph (allocator
    bookkeeping on the capture stream, 1 KiB on torch 2.11), measured on a trivial
    capture that is released again so the process state is unchanged."""
    import gc

    import torch

    x = torch.ones(16, device="cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x * 2
    torch.cuda.synchronize()
    overhead = torch.cuda.memory_allocated() - before - _graph_pool_bytes()
    del graph, y
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before, "control graph left allocations"
    return int(overhead)


_WRAPPER = None

CASES = {
    "static": dict(intermediate=320, tokens=64),
    "aligned_static": dict(
        intermediate=512, tokens=64
    ),  # no padding at all: the fold is the first preparation
    "gated_dynamic": dict(intermediate=320, tokens=1024),
    "generic_dynamic": dict(intermediate=640, tokens=1024),
    "single_slice": dict(intermediate=64, tokens=64),
}


def main() -> int:
    global _WRAPPER
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=sorted(CASES))
    ap.add_argument("--prewarm", action="store_true")
    ap.add_argument(
        "--input-global-scale",
        action="store_true",
        help="pass a non-null input_global_scale so the wrapper must fold w1_alpha (first-use allocation)",
    )
    ap.add_argument(
        "--rescale-after-prewarm",
        action="store_true",
        help="after the eager warm-up, hand a new input_global_scale tensor (pointer / version change) to the capture",
    )
    args = ap.parse_args()
    import torch

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from flashinfer import B12xMoEWrapper
    from tests.moe.test_b12x_static_extent_rules import (
        _kwargs,
        _reference_shape,
        _tensors_shape,
    )

    E, H, TOPK = 8, 256, 2
    I, M = CASES[args.case]["intermediate"], CASES[args.case]["tokens"]
    t = _tensors_shape(M, E, H, I, TOPK, seed=3)
    moe = B12xMoEWrapper(
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        use_cuda_graph=True,
        max_num_tokens=max(M, 1024),
    )
    _WRAPPER = moe
    kwargs = _kwargs(t)
    if args.input_global_scale:
        # Exact re-parameterisation: the FC1 input-quant scale moves into
        # input_global_scale and w1_alpha becomes the unit weight scale, so the
        # folded alpha (w1_alpha * input_global_scale) equals the default path's
        # alpha and the outputs stay comparable with the BF16 reference.
        kwargs["input_global_scale"] = kwargs["w1_alpha"].detach().clone()
        kwargs["w1_alpha"] = torch.ones_like(kwargs["w1_alpha"])
    result = {
        "case": args.case,
        "prewarm": args.prewarm,
        "intermediate": I,
        "tokens": M,
        "graph_overhead_bytes": _graph_overhead_bytes(),
        "before": cache_state(),
    }
    eager = None
    if args.prewarm:
        eager = moe.run(**kwargs).clone()
        torch.cuda.synchronize()
        if args.rescale_after_prewarm:
            # same values, new tensor: a pointer / version change of the fold key
            kwargs["input_global_scale"] = kwargs["input_global_scale"].clone()
        result["after_prewarm"] = cache_state()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = moe.run(**kwargs)
        result["capture"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["capture"] = "refused"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[:200]
        captured = None
        # the aborted capture's private memory pool holds the per-call temporaries allocated before the refusal
        # (expanded per-expert scale vectors); release the graph so the allocator state reflects only retained state
        graph = None
        import gc

        gc.collect()
        torch.cuda.synchronize()
    result["after_capture"] = cache_state()
    if captured is not None:
        graph.replay()
        torch.cuda.synchronize()
        ref = _reference_shape(t, M, E, H, I, TOPK)
        rel = ((captured.float() - ref).norm() / ref.norm()).item()
        result["replay_rel_l2_vs_bf16"] = rel
        if eager is not None:
            result["replay_matches_eager"] = bool(
                torch.allclose(captured, eager, atol=2e-2, rtol=2e-2)
            )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
