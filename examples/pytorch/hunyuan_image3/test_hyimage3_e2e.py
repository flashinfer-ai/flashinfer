#!/usr/bin/env python3
"""End-to-end correctness + performance test for the HunyuanImage-3 FlashInfer
example, runnable on a single GPU without the 165 GB checkpoint.

Mirrors the wan example's synthetic-forward testing style, in two stages:

1. **Fused-MoE unit test** (``--stage moe`` or ``all``): builds the shared
   ``FlashInferFusedMoE`` at HunyuanImage-3 shapes (or ``--small`` shapes),
   compares every requested backend (cutlass / cutlass_fp8 / trtllm) against
   the module's own eager ``torch`` reference **with identical weights and
   routing**, and benchmarks each backend.

2. **Backbone equivalence test** (``--stage backbone`` or ``all``): builds a
   small random-weight stack of *upstream* ``HunyuanImage3DecoderLayer``
   modules (loaded via the checkpoint's remote code — only the config and
   ``.py`` files are needed, no weights), deep-copies it, applies
   ``replace_backbone_with_flashinfer`` to the copy, and compares outputs:

   - masked prefill: both sides get the same 4D bool mask (SDPA fallback on
     the FlashInfer side — validates the norm/GEMM/MoE swaps in isolation);
   - mask-less prefill: the baseline gets an explicit *causal* mask while the
     FlashInfer side runs its mask-less kernel path with ``causal=True``
     (upstream SDPA without a mask would be full attention, so the causal
     mask keeps the two semantically identical).

   It then benchmarks baseline vs. FlashInfer forward latency.

Examples::

    # Everything, default (small) shapes, BF16 cutlass MoE:
    python test_hyimage3_e2e.py

    # Sweep MoE backends at real HunyuanImage-3 shapes on a B200:
    python test_hyimage3_e2e.py --stage moe --moe-shapes real \
        --moe-backends cutlass cutlass_fp8 trtllm

    # Backbone equivalence with the trtllm attention + MoE (SM100):
    python test_hyimage3_e2e.py --stage backbone \
        --attention-backend trtllm --moe-backends trtllm

The upstream remote code is loaded from ``--model-path`` (defaults to
``$FLASHINFER_HYIMAGE3_PATH`` or the HF repo id); only ``config.json`` and the
``*.py`` files are downloaded/read, never the weights.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXAMPLES_PYTORCH_DIR = _HERE.parent
for _p in (str(_HERE), str(_EXAMPLES_PYTORCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flashinfer_modules import (  # noqa: E402
    FlashInferFusedMoE,
    MoEBackend,
    _check_moe_backend_support,
)
from modeling_hunyuan_image3_flashinfer import (  # noqa: E402
    replace_backbone_with_flashinfer,
)

_DEFAULT_MODEL_PATH = os.getenv(
    "FLASHINFER_HYIMAGE3_PATH", "tencent/HunyuanImage-3.0-Instruct"
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _time_fn(fn: Callable[[], object], warmup: int = 5, iters: int = 20) -> float:
    """Median-free simple GPU timer (ms per call) using CUDA events."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _compare(name: str, ref: torch.Tensor, out: torch.Tensor) -> dict:
    ref_f = ref.float().flatten()
    out_f = out.float().flatten()
    cos = F.cosine_similarity(ref_f, out_f, dim=0).item()
    max_abs = (ref_f - out_f).abs().max().item()
    rel_l2 = ((ref_f - out_f).norm() / ref_f.norm().clamp(min=1e-12)).item()
    print(
        f"  [{name}] cos_sim={cos:.6f}  max_abs_err={max_abs:.4e}  "
        f"rel_l2={rel_l2:.4e}"
    )
    return {"name": name, "cos": cos, "max_abs": max_abs, "rel_l2": rel_l2}


def _easy_topk(logits: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replicates upstream HunyuanTopKGate.easy_topk (softmax -> topk -> renorm)."""
    gates = F.softmax(logits, dim=1)
    topk_weight, expert_index = torch.topk(gates, top_k)
    weight_sums = topk_weight.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return topk_weight / weight_sums, expert_index


# ----------------------------------------------------------------------------
# Stage 1: fused-MoE unit test + benchmark
# ----------------------------------------------------------------------------


def run_moe_stage(args) -> List[dict]:
    if args.moe_shapes == "real":
        # HunyuanImage-3.0: 64 routed experts, top-8, hidden 4096, inter 3072.
        num_experts, top_k, hidden, inter = 64, 8, 4096, 3072
    else:
        num_experts, top_k, hidden, inter = 16, 4, 1024, 512

    device = torch.device("cuda")
    dtype = torch.bfloat16
    results = []

    print(
        f"\n=== Stage 1: FlashInferFusedMoE unit test "
        f"(E={num_experts}, top_k={top_k}, hidden={hidden}, inter={inter}) ==="
    )

    for backend in args.moe_backends:
        if backend == "eager":
            continue
        if not _check_moe_backend_support(MoEBackend(backend), device):
            print(f"  [{backend}] not supported on this GPU, skipping.")
            continue

        torch.manual_seed(args.seed)
        moe = FlashInferFusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden,
            intermediate_size=inter,
            moe_backend=backend,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            moe.w13_weight.normal_(0, 0.02)
            moe.w2_weight.normal_(0, 0.02)
        moe.prepare_weights()

        for num_tokens in args.moe_num_tokens:
            x = torch.randn(num_tokens, hidden, device=device, dtype=dtype) * 0.5
            logits = torch.randn(
                num_tokens, num_experts, device=device, dtype=torch.float32
            )
            topk_weight, topk_ids = _easy_topk(logits, top_k)

            with torch.no_grad():
                ref = moe._forward_torch(x, topk_ids, topk_weight)
                out = moe(x, topk_ids, topk_weight)
            r = _compare(f"moe/{backend}/T={num_tokens}", ref, out)

            ms = _time_fn(
                lambda: moe(x, topk_ids, topk_weight),
                warmup=args.warmup, iters=args.iters,
            )
            # 3 * 2 * T * K_active GEMM flops: fc1 (2*inter x hidden) + fc2.
            flops = 2 * num_tokens * top_k * (2 * inter * hidden + hidden * inter)
            print(
                f"    perf: {ms:.3f} ms  ({flops / ms / 1e9:.1f} TFLOP/s dense-equiv)"
            )
            r.update({"ms": ms, "num_tokens": num_tokens, "backend": backend})
            results.append(r)
        del moe
        torch.cuda.empty_cache()
    return results


# ----------------------------------------------------------------------------
# Stage 2: decoder-stack equivalence vs upstream remote code
# ----------------------------------------------------------------------------


def _load_upstream_layer_cls(model_path: str):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    layer_cls = get_class_from_dynamic_module(
        "modeling_hunyuan_image_3.HunyuanImage3DecoderLayer", model_path
    )
    return config, layer_cls


def _make_small_config(config, args):
    """Shrink the checkpoint config to the test size (keeps all MoE knobs)."""
    config.hidden_size = args.hidden_size
    config.num_attention_heads = args.num_heads
    config.num_key_value_heads = args.num_kv_heads
    config.attention_head_dim = args.head_dim
    config.intermediate_size = args.moe_inter_size
    config.moe_intermediate_size = args.moe_inter_size
    config.num_experts = args.num_experts
    config.moe_topk = args.moe_topk
    config.num_shared_expert = 1
    config.moe_layer_num_skipped = 0
    config.num_hidden_layers = args.num_layers
    config.moe_impl = "eager"
    config._attn_implementation = "sdpa"
    return config


class _LayerStack(torch.nn.Module):
    """Minimal stand-in for HunyuanImage3Model: just the decoder layers."""

    def __init__(self, layers: torch.nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, hidden_states, attention_mask, custom_pos_emb):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                custom_pos_emb=custom_pos_emb,
            )[0]
        return hidden_states


class _ModelWrapper(torch.nn.Module):
    """Gives the stack the ``model.model.layers`` shape that
    ``replace_backbone_with_flashinfer`` expects."""

    def __init__(self, backbone: _LayerStack):
        super().__init__()
        self.model = backbone


def _make_rope(batch: int, seq: int, head_dim: int, device, dtype):
    """Standard rotate-half RoPE tables, shaped (B, S, head_dim) like the
    upstream custom_pos_emb."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
    sin = emb.sin().to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
    return cos, sin


def run_backbone_stage(args) -> List[dict]:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(
        f"\n=== Stage 2: backbone equivalence "
        f"(layers={args.num_layers}, hidden={args.hidden_size}, "
        f"heads={args.num_heads}/{args.num_kv_heads}x{args.head_dim}, "
        f"E={args.num_experts} top{args.moe_topk}, B={args.batch_size}, "
        f"S={args.seq_len}) ==="
    )
    print(f"Loading upstream remote code from {args.model_path} ...")
    config, layer_cls = _load_upstream_layer_cls(args.model_path)
    config = _make_small_config(config, args)

    torch.manual_seed(args.seed)
    layers = torch.nn.ModuleList(
        [layer_cls(config, layer_idx=i) for i in range(args.num_layers)]
    )
    baseline = _LayerStack(layers).to(device=device, dtype=dtype).eval()
    # The router weights are FP32 by design; nn.Module.to(dtype) above casts
    # them to BF16, so restore FP32 (matches from_pretrained behaviour where
    # the gate is declared with dtype=torch.float32).
    for layer in baseline.layers:
        if hasattr(layer.mlp, "gate"):
            layer.mlp.gate.wg.float()

    results = []

    for moe_backend in args.moe_backends:
        wrapper = _ModelWrapper(copy.deepcopy(baseline))
        opts = replace_backbone_with_flashinfer(
            wrapper,
            gemm_backend=args.gemm_backend,
            attention_backend=args.attention_backend,
            moe_backend=moe_backend,
            prepare_weights=True,
        )
        swapped = wrapper.model
        print(f"\n-- moe_backend={moe_backend} (resolved options: {opts})")

        torch.manual_seed(args.seed + 1)
        x = (
            torch.randn(
                args.batch_size, args.seq_len, args.hidden_size,
                device=device, dtype=dtype,
            )
            * 0.5
        )
        cos, sin = _make_rope(
            args.batch_size, args.seq_len, args.head_dim, device, dtype
        )

        causal_mask = torch.tril(
            torch.ones(args.seq_len, args.seq_len, dtype=torch.bool, device=device)
        ).view(1, 1, args.seq_len, args.seq_len).expand(args.batch_size, 1, -1, -1)
        # "Image block" style mask: causal text prefix + a bidirectional
        # block in the second half (like gen_image prefill).
        block_mask = causal_mask.clone()
        half = args.seq_len // 2
        block_mask[..., half:, half:] = True

        with torch.no_grad():
            # (a) masked prefill: identical mask on both sides (SDPA fallback
            # on the FlashInfer side; validates norm/GEMM/MoE swaps).
            ref = baseline(x, block_mask, (cos, sin))
            out = swapped(x, block_mask, (cos, sin))
            results.append(
                _compare(f"backbone/{moe_backend}/masked-prefill", ref, out)
            )

            # (b) mask-less prefill: FlashInfer runs its kernel path with
            # causal=True; baseline gets the equivalent explicit causal mask.
            ref = baseline(x, causal_mask, (cos, sin))
            out = swapped(x, None, (cos, sin))
            results.append(
                _compare(f"backbone/{moe_backend}/maskless-causal", ref, out)
            )

        ms_ref = _time_fn(
            lambda: baseline(x, causal_mask, (cos, sin)),
            warmup=args.warmup, iters=args.iters,
        )
        ms_out = _time_fn(
            lambda: swapped(x, None, (cos, sin)),
            warmup=args.warmup, iters=args.iters,
        )
        print(
            f"  perf: baseline {ms_ref:.2f} ms -> flashinfer {ms_out:.2f} ms "
            f"({ms_ref / ms_out:.2f}x)"
        )
        results[-1].update({"ms_baseline": ms_ref, "ms_flashinfer": ms_out})

        del wrapper, swapped
        torch.cuda.empty_cache()

    return results


# ----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", default="all", choices=["all", "moe", "backbone"])
    p.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    p.add_argument("--seed", type=int, default=0)

    # MoE stage.
    p.add_argument(
        "--moe-backends", nargs="+",
        default=["cutlass"],
        choices=["cutlass", "cutlass_fp8", "trtllm", "torch"],
    )
    p.add_argument("--moe-shapes", default="small", choices=["small", "real"])
    p.add_argument(
        "--moe-num-tokens", type=int, nargs="+", default=[128, 1024, 4096]
    )

    # Backbone stage.
    p.add_argument("--gemm-backend", default="torch")
    p.add_argument(
        "--attention-backend", default="auto",
        choices=["auto", "single", "cudnn", "trtllm", "torch", "sdpa"],
    )
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=1024)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-experts", type=int, default=16)
    p.add_argument("--moe-topk", type=int, default=4)
    p.add_argument("--moe-inter-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1024)

    # Timing.
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)

    # Pass/fail thresholds (BF16 kernels reorder reductions, so allow slack).
    p.add_argument("--min-cos", type=float, default=0.98)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    cc = torch.cuda.get_device_capability()
    print(
        f"GPU: {torch.cuda.get_device_name()} (SM{cc[0] * 10 + cc[1]}), "
        f"torch {torch.__version__}"
    )

    results = []
    if args.stage in ("all", "moe"):
        results += run_moe_stage(args)
    if args.stage in ("all", "backbone"):
        results += run_backbone_stage(args)

    failed = [r for r in results if r["cos"] < args.min_cos]
    print(f"\n=== Summary: {len(results)} checks, {len(failed)} failed "
          f"(min cos_sim threshold {args.min_cos}) ===")
    for r in failed:
        print(f"  FAILED: {r['name']} cos_sim={r['cos']:.6f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
