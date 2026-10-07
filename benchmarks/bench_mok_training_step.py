#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Full MoE training-step benchmark: Cake MoK (``flashinfer.mok``) against the
baselines B1 (native MoK #17) and B2 (DeepEP HybridEP + ``torch`` grouped GEMMs).

Launch one process per EP rank::

    torchrun --standalone --nproc-per-node=4 benchmarks/bench_mok_training_step.py \
        --shape glm52 --rows uniform --routing uniform --impls cake,mok17 \
        --macrobatch 262144 --mode full --json /path/row.json

Protocol (FlashInfer issue #6165): the same GPUs, in-process
interleaving of the implementations, three counterbalanced groups, per group
and rank at least 100 ms of warmup and 1000 ms of sampling, the slowest rank
per iteration, the full iteration (schedule build, source copies, forward,
optional context recompute, backward). Memory: retained context bytes, peak
allocated bytes during the step and symmetric workspace bytes. Every row uses
the same no-overflow schedule-capacity multiplier for all implementations.

Timing uses CUDA events around whole distributed iterations with a barrier
before each sampling block; per-kernel CUPTI tracing cannot time a
multi-kernel, multi-rank step and is reserved for the single-kernel
diagnostics registered in Cake's ``benchmark_data.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch
import torch.distributed as dist

SOURCE_LENGTHS_RAGGED = (8191, 16172, 16231, 32767)
WARMUP_MS = 100.0
SAMPLE_MS = 1000.0
MIN_SAMPLES = 5


@dataclasses.dataclass(frozen=True)
class Shape:
    name: str
    hidden: int
    intermediate: int
    experts: int
    topk: int
    swiglu_limit: float | None


SHAPES = {
    # GLM-5.2: 256 routed experts, one shared expert, plain SwiGLU.
    "glm52": Shape("glm52", 6144, 2048, 256, 8, None),
    # GLM-5.3-Flash: 288 routed experts, one shared expert, clamped SwiGLU L=10.
    "glm53": Shape("glm53", 4096, 2048, 288, 8, 10.0),
    # Small smoke shape for harness checks.
    "toy": Shape("toy", 512, 512, 16, 2, None),
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--shape", choices=sorted(SHAPES), default="glm52")
    p.add_argument(
        "--rows",
        default="uniform",
        help="uniform (16384 per rank) | ragged (8191,16172,16231,32767 rotating, every 4th rank empty) | <int>",
    )
    p.add_argument("--uniform-rows", type=int, default=16384)
    p.add_argument(
        "--routing",
        default="uniform",
        help="uniform | imbalanced (hot rank 2x) | skew:<s> (hot rank s x)",
    )
    p.add_argument(
        "--impls", default="cake,mok17", help="comma list of cake,mok17,deepep"
    )
    p.add_argument(
        "--deepep-sms",
        type=int,
        default=None,
        help="HybridEP dispatch/combine SMs (None = library default)",
    )
    p.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    p.add_argument(
        "--mode",
        choices=("full", "checkpoint"),
        default="full",
        help="full = forward+backward; checkpoint = forward+recompute_forward_context+backward",
    )
    p.add_argument("--macrobatch", type=int, default=262144)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument(
        "--capacity-mult",
        type=float,
        default=1.0,
        help="source_capacity = ceil(max_rows * mult) (the 2x source-capacity constraint row uses 2)",
    )
    p.add_argument(
        "--fwd-comm-sms", type=int, default=None, help="per impl default when omitted"
    )
    p.add_argument("--bwd-comm-sms", type=int, default=None)
    p.add_argument("--groups", type=int, default=3)
    p.add_argument("--warmup-ms", type=float, default=WARMUP_MS)
    p.add_argument("--sample-ms", type=float, default=SAMPLE_MS)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--check",
        action="store_true",
        help="cross-compare implementation outputs (sanity, not a gate)",
    )
    p.add_argument("--json", default=None, help="write the rank-0 record here")
    return p.parse_args()


def init_distributed():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", device_id=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return dist.get_rank(), dist.get_world_size(), device


def source_rows(args, rank, ep):
    if args.rows == "uniform":
        return [args.uniform_rows] * ep
    if args.rows == "ragged":
        rows = [
            SOURCE_LENGTHS_RAGGED[r % len(SOURCE_LENGTHS_RAGGED)] for r in range(ep)
        ]
        for r in range(ep):
            if r % 4 == 3:
                rows[r] = 0
        return rows
    return [int(args.rows)] * ep


def hot_factor(routing):
    if routing == "uniform":
        return 1.0
    if routing == "imbalanced":
        return 2.0
    if routing.startswith("skew:"):
        return float(routing.split(":", 1)[1])
    raise ValueError(f"unknown routing {routing!r}")


def generate_inputs(args, shape, rank, ep, rows, device):
    """Per-rank inputs with a fixed seed; the hot rank (rank 0's experts) gets
    ``hot_factor`` times the base routing probability."""
    local_experts = shape.experts // ep
    g = torch.Generator(device=device).manual_seed(args.seed + 7919 * rank)
    n = rows[rank]
    opts = dict(device=device, dtype=torch.bfloat16)
    n_alloc = max(n, 1)
    logits = torch.randn(n_alloc, shape.experts, generator=g, device=device)
    factor = hot_factor(args.routing)
    if factor != 1.0:
        logits[:, :local_experts] += math.log(factor)
    values, ids = torch.topk(logits, shape.topk, dim=1)
    scores = torch.softmax(values.float(), dim=-1).contiguous()
    x = torch.randn(n_alloc, shape.hidden, generator=g, **opts)
    dy = torch.randn(n_alloc, shape.hidden, generator=g, **opts) * shape.hidden**-0.5
    # Expert weights are private to the owning rank (routing ids are global,
    # the values need no cross-rank agreement): generate only the local slice.
    gw = torch.Generator(device=device).manual_seed(args.seed + 1000003 * rank)
    wg = (
        torch.randn(
            local_experts, shape.intermediate, shape.hidden, generator=gw, **opts
        )
        * shape.hidden**-0.5
    )
    wu = (
        torch.randn(
            local_experts, shape.intermediate, shape.hidden, generator=gw, **opts
        )
        * shape.hidden**-0.5
    )
    wd = (
        torch.randn(
            local_experts, shape.hidden, shape.intermediate, generator=gw, **opts
        )
        * shape.intermediate**-0.5
    )
    gs = torch.Generator(device=device).manual_seed(args.seed)
    sg = (
        torch.randn(shape.intermediate, shape.hidden, generator=gs, **opts)
        * shape.hidden**-0.5
    )
    su = (
        torch.randn(shape.intermediate, shape.hidden, generator=gs, **opts)
        * shape.hidden**-0.5
    )
    sd = (
        torch.randn(shape.hidden, shape.intermediate, generator=gs, **opts)
        * shape.intermediate**-0.5
    )
    weights = (sg, su, sd, wg, wu, wd)
    if n == 0:
        x, dy, ids, scores = x[:0], dy[:0], ids[:0], scores[:0]
    return x, ids.to(torch.int64), scores, dy, weights


def routed_capacity_factor(ids, shape, ep, topk):
    """Smallest integer multiple of ``rows * topk`` that holds every rank's
    received routes with per-expert 256-row padding (the no-overflow factor)."""
    local_experts = shape.experts // ep
    counts = torch.bincount(ids.flatten(), minlength=shape.experts).to(torch.int64)
    dist.all_reduce(counts)
    padded = ((counts + 255) // 256) * 256
    per_rank = padded.view(ep, local_experts).sum(1)
    needed = int(per_rank.max().item())
    return needed


class CakeImpl:
    name = "cake"

    def __init__(self, args, shape, rank, ep, rows, inputs, device, capacity, factor):
        from flashinfer.mok import create_mok_bf16_workspace, prepare_mok_bf16

        self.shape, self.rank, self.ep = shape, rank, ep
        self.x, self.ids, self.scores, self.dy, self.weights = inputs
        self.local_experts = shape.experts // ep
        self.functional = prepare_mok_bf16(
            ep_size=ep, local_experts=self.local_experts, topk=shape.topk
        )
        fwd = args.fwd_comm_sms or 40
        bwd = args.bwd_comm_sms or 40
        self.config, self.workspace = create_mok_bf16_workspace(
            group=dist.group.WORLD,
            device=device,
            num_local_tokens=rows[rank],
            hidden_size=shape.hidden,
            topk=shape.topk,
            source_capacity=capacity,
            fwd_num_comm_sms=fwd,
            bwd_num_comm_sms=bwd,
            minibatch_size=args.minibatch,
            macrobatch_size=args.macrobatch,
            schedule_capacity_multiplier=factor / ep,
        )
        self.settings = dict(
            fwd_num_comm_sms=fwd,
            bwd_num_comm_sms=bwd,
            minibatch=args.minibatch,
            macrobatch=args.macrobatch,
            schedule_capacity_multiplier=factor / ep,
            source_capacity=self.workspace.source_capacity,
            schedule_capacity=self.workspace.storage.schedule_capacity,
        )
        self.context = None

    def workspace_bytes(self):
        storage = self.workspace.storage
        total = 0
        for field in dataclasses.fields(storage):
            value = getattr(storage, field.name)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    @staticmethod
    def context_bytes(context):
        total = 0
        for field in dataclasses.fields(context):
            value = getattr(context, field.name)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
            elif isinstance(value, tuple):
                total += sum(
                    v.numel() * v.element_size()
                    for v in value
                    if isinstance(v, torch.Tensor)
                )
        return total

    def step(self, checkpoint):
        schedule = self.functional.build_schedule(
            self.workspace, self.config, self.ids, num_local_experts=self.local_experts
        )
        y, context = self.functional.forward(
            self.config,
            self.workspace,
            schedule,
            self.x,
            self.scores,
            *self.weights,
            swiglu_limit=self.shape.swiglu_limit,
        )
        if checkpoint:
            context = self.functional.recompute_forward_context(
                self.config,
                self.workspace,
                schedule,
                self.x,
                self.weights[0],
                self.weights[1],
                self.weights[3],
                self.weights[4],
                swiglu_limit=self.shape.swiglu_limit,
            )
        grads = self.functional.backward(
            self.config,
            self.workspace,
            schedule,
            context,
            self.dy,
            self.x,
            self.scores,
            *self.weights,
            swiglu_limit=self.shape.swiglu_limit,
        )
        self.context = context
        return (y, *grads)

    def teardown(self):
        self.context = None


class Mok17Impl:
    name = "mok17"

    def __init__(self, args, shape, rank, ep, rows, inputs, device, capacity, factor):
        from mok import functional as mokf

        self.mokf = mokf
        self.shape, self.rank, self.ep = shape, rank, ep
        self.x, self.ids, self.scores, self.dy, self.weights = inputs
        self.local_experts = shape.experts // ep
        fwd = args.fwd_comm_sms or 24
        bwd = args.bwd_comm_sms or 28
        # The native workspace capacity is common across ranks (multiple of 256, >= 512).
        self.capacity = capacity
        self.config = mokf.MoKConfig(
            fwd_num_comm_sms=fwd,
            bwd_num_comm_sms=bwd,
            minibatch_size=args.minibatch,
            macrobatch_size=args.macrobatch,
            schedule_capacity_multiplier=factor / ep,
        )
        self.workspace = mokf.create_workspace(
            self.config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=capacity,
            hidden_size=shape.hidden,
            topk=shape.topk,
        )
        self.precision = args.precision
        if self.precision == "mxfp8":
            from mok import ops

            self.wq = [ops.mxfp8_quantize(w, True, True) for w in self.weights[3:]]
        self.settings = dict(
            fwd_num_comm_sms=fwd,
            bwd_num_comm_sms=bwd,
            minibatch=args.minibatch,
            macrobatch=args.macrobatch,
            schedule_capacity_multiplier=factor / ep,
            source_capacity=capacity,
            schedule_capacity=self.workspace.schedule_capacity,
            precision=self.precision,
        )
        self.context = None

    def workspace_bytes(self):
        total = 0
        for field in dataclasses.fields(self.workspace):
            value = getattr(self.workspace, field.name)
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    context_bytes = staticmethod(CakeImpl.context_bytes)

    def _routed(self, forward):
        if self.precision == "bf16":
            return self.weights[3:]
        gate, up, down = self.wq
        if forward:
            return gate[:2], up[:2], down[:2]
        return gate, up, down[2:]

    def step(self, checkpoint):
        mokf, shape = self.mokf, self.shape
        schedule = mokf.build_schedule(
            self.workspace, self.config, self.ids, num_local_experts=self.local_experts
        )
        sg, su, sd = self.weights[:3]
        y, context = mokf.forward(
            self.config,
            self.workspace,
            schedule,
            self.x,
            self.scores,
            sg,
            su,
            sd,
            *self._routed(True),
            swiglu_limit=shape.swiglu_limit,
        )
        if checkpoint:
            rg, ru, _ = self._routed(True)
            context = mokf.recompute_forward_context(
                self.config,
                self.workspace,
                schedule,
                self.x,
                sg,
                su,
                rg,
                ru,
                swiglu_limit=shape.swiglu_limit,
            )
        grads = mokf.backward(
            self.config,
            self.workspace,
            schedule,
            context,
            self.dy,
            self.x,
            self.scores,
            sg,
            su,
            sd,
            *self._routed(False),
            swiglu_limit=shape.swiglu_limit,
        )
        self.context = context
        return (y, *grads)

    def teardown(self):
        self.context = None


def _swiglu_torch(gate, up, limit):
    """Plain-``torch`` SwiGLU in FP32 with the MoK clamp semantics (``torch.clamp``
    boundaries), rounded once to BF16; autograd yields the exact masked derivative."""
    g, u = gate.float(), up.float()
    if limit is not None:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(g) * u).to(torch.bfloat16)


class DeepEPImpl:
    """B2: DeepEP HybridEP dispatch/combine (permuted layout) + ``torch`` grouped
    GEMMs, structured like MoK #17's ``benchmarks/bench_deepep_torch.py`` (shared
    expert in plain ``torch`` autograd, routed experts as grouped GEMMs on the
    permuted rows, route weights applied in FP32 before the combine, the score
    gradient travels back through the combine's ``probs`` channel). MXFP8 uses
    MoK's ``impl_utils`` (torchao MXFP8 quantization + ``F.scaled_grouped_mm``)."""

    name = "deepep"

    def __init__(self, args, shape, rank, ep, rows, inputs, device, capacity, factor):
        import deep_ep

        self.shape, self.rank, self.ep, self.device = shape, rank, ep, device
        self.x, self.ids, self.scores, self.dy, self.weights = inputs
        self.local_experts = shape.experts // ep
        self.capacity = capacity
        self.precision = args.precision
        # torchao's MXFP8 kernels are needed only for the MXFP8 arm; the BF16 arm uses torch's grouped GEMM directly
        # (MoK impl_utils.grouped_mm's BF16 branch) so an older torchao does not block BF16 rows.
        if self.precision == "mxfp8":
            try:
                self.utils = _mok_impl_utils()
            except ImportError as error:
                raise RuntimeError(
                    f"deepep mxfp8 unavailable (torchao MXFP8 kernels): {error}"
                ) from error
        else:
            self.utils = None
        # MoK #17 pads MXFP8 expert groups to 128 rows (scaled grouped GEMM alignment).
        self.pad = 128 if self.precision == "mxfp8" else None
        free_before = torch.cuda.mem_get_info(device)[0]
        self.buffer = deep_ep.HybridEPBuffer(
            dist.group.WORLD,
            hidden_dim=shape.hidden,
            max_num_of_tokens_per_rank=capacity,
            num_local_experts=self.local_experts,
            use_fp8=False,
            num_sms_dispatch_api=args.deepep_sms,
            num_sms_combine_api=args.deepep_sms,
        )
        torch.cuda.synchronize(device)
        self._buffer_bytes = max(0, free_before - torch.cuda.mem_get_info(device)[0])
        # Padded permuted row count of the live handle; cached-handle dispatches
        # skip the metadata pass that derives it, so the caller supplies it.
        self.num_permuted = None
        wg, wu, wd = self.weights[3:]
        self.w_gate_t = wg.detach().transpose(1, 2).requires_grad_()
        self.w_up_t = wu.detach().transpose(1, 2).requires_grad_()
        self.w_down_t = wd.detach().transpose(1, 2).requires_grad_()
        if self.precision == "mxfp8":
            self.w_mx = [self.utils.prequantize_mxfp8_weight(w) for w in (wg, wu, wd)]
        else:
            self.w_mx = [None, None, None]
        self.settings = dict(
            deepep_sms=args.deepep_sms,
            pad_multiple=self.pad,
            source_capacity=capacity,
            precision=self.precision,
            buffer_bytes=self._buffer_bytes,
        )
        self.context = None
        self._ctx_bytes = 0

    def workspace_bytes(self):
        return self._buffer_bytes

    def context_bytes(self, context):
        # Autograd-retained activations (shapes known exactly): permuted x, gate,
        # up, hidden, expert output and the shared gate/up/hidden.
        return self._ctx_bytes

    def _source(self):
        """Real rows, or a one-row routed-nowhere placeholder for an empty rank."""
        n = self.x.shape[0]
        if n:
            return n, self.x, self.ids, self.scores, self.dy
        opts = dict(device=self.device)
        x = torch.zeros(1, self.shape.hidden, dtype=torch.bfloat16, **opts)
        ids = torch.full((1, self.shape.topk), -1, dtype=torch.int64, **opts)
        scores = torch.zeros(1, self.shape.topk, dtype=torch.float32, **opts)
        return 0, x, ids, scores, torch.zeros_like(x)

    def _routed_forward(self, x, ids, scores, handle=None):
        """Dispatch (permuted layout) + grouped gate/up GEMMs + SwiGLU + grouped
        down GEMM + FP32 route weighting; returns the graph leaves and handle."""
        kwargs = dict(
            hidden=x, pad_multiple=self.pad, num_of_tokens_per_rank=self.capacity
        )
        if handle is None:
            kwargs.update(
                topk_idx=ids,
                topk_weights=scores,
                num_of_experts=self.shape.experts,
                num_of_experts_per_rank=self.local_experts,
            )
        else:
            kwargs.update(handle=handle, num_permuted_tokens=self.num_permuted)
        permuted, probs, _, padded_counts, handle = self.buffer.dispatch_with_permute(
            **kwargs
        )
        self.num_permuted = permuted.shape[0]
        if self.pad is not None:
            # Padding rows hold no token: zero them so the weight gradients (K =
            # permuted rows) stay exact; the combine ignores them anyway.
            real = handle[7].to(padded_counts.device)
            starts = torch.cumsum(padded_counts, 0) - padded_counts
            group = torch.repeat_interleave(
                torch.arange(padded_counts.numel(), device=starts.device), padded_counts
            )
            local = (
                torch.arange(permuted.shape[0], device=starts.device) - starts[group]
            )
            keep = local < real[group]
            permuted = permuted * keep[:, None].to(permuted.dtype)
            probs = probs * keep.to(probs.dtype)
        offsets = torch.cumsum(padded_counts, 0, dtype=torch.int32).to(permuted.device)
        permuted = permuted.detach().requires_grad_()
        probs = probs.detach().float().requires_grad_()
        gmm = self.utils.grouped_mm if self.utils is not None else _grouped_mm_bf16
        gate = gmm(permuted, self.w_gate_t, self.w_mx[0], offsets)
        up = gmm(permuted, self.w_up_t, self.w_mx[1], offsets)
        hidden = _swiglu_torch(gate, up, self.shape.swiglu_limit)
        out = gmm(hidden, self.w_down_t, self.w_mx[2], offsets)
        weighted = (out.float() * probs[:, None]).to(torch.bfloat16)
        rows = permuted.shape[0]
        self._ctx_bytes = 2 * (
            rows * self.shape.hidden * 2 + 3 * rows * self.shape.intermediate
        )
        return permuted, probs, weighted, handle

    def step(self, checkpoint):
        shape = self.shape
        n, x, ids, scores, dy = self._source()
        sg, su, sd = (w.detach().requires_grad_() for w in self.weights[:3])
        x_s = x.detach().requires_grad_()
        hidden_s = _swiglu_torch(x_s @ sg.T, x_s @ su.T, shape.swiglu_limit)
        y_shared = hidden_s @ sd.T
        self._ctx_bytes = 0
        if checkpoint:
            with torch.no_grad():
                _, _, weighted, handle = self._routed_forward(x, ids, scores)
                y_routed, _ = self.buffer.combine_with_unpermute(
                    hidden=weighted, handle=handle, pad_multiple=self.pad
                )
            # Checkpoint step: rebuild the routed graph (dispatch + GEMMs + SwiGLU)
            # before the backward, as the MoK recompute does.
            permuted, probs, weighted, handle = self._routed_forward(
                x, ids, scores, handle
            )
        else:
            permuted, probs, weighted, handle = self._routed_forward(x, ids, scores)
            y_routed, _ = self.buffer.combine_with_unpermute(
                hidden=weighted.detach(), handle=handle, pad_multiple=self.pad
            )
        y = (y_routed[:n].float() + y_shared.detach()[:n].float()).to(torch.bfloat16)
        self._ctx_bytes += 3 * n * shape.intermediate * 2
        # Backward: dispatch dy into the permuted layout, autograd the expert and
        # shared graphs, combine dx and the per-route score gradient.
        d_weighted, _, _, _, _ = self.buffer.dispatch_with_permute(
            hidden=dy,
            handle=handle,
            pad_multiple=self.pad,
            num_of_tokens_per_rank=self.capacity,
            num_permuted_tokens=self.num_permuted,
        )
        d_x_s, d_sg, d_su, d_sd = torch.autograd.grad(y_shared, (x_s, sg, su, sd), dy)
        d_permuted, d_wg_t, d_wu_t, d_wd_t, d_probs = torch.autograd.grad(
            weighted,
            (permuted, self.w_gate_t, self.w_up_t, self.w_down_t, probs),
            d_weighted,
        )
        d_x_routed, d_probs_all = self.buffer.combine_with_unpermute(
            hidden=d_permuted, probs=d_probs, handle=handle, pad_multiple=self.pad
        )
        if d_probs_all.shape[-1] == shape.experts:
            d_scores = d_probs_all.gather(1, ids.clamp_min(0)) * (ids >= 0)
        else:
            d_scores = d_probs_all
        d_x = (d_x_routed[:n].float() + d_x_s[:n].float()).to(torch.bfloat16)
        self.context = handle
        return (
            y,
            d_x,
            d_scores[:n].float().contiguous(),
            d_wg_t.transpose(1, 2),
            d_wu_t.transpose(1, 2),
            d_wd_t.transpose(1, 2),
            d_sg,
            d_su,
            d_sd,
        )

    def teardown(self):
        self.context = None


def _grouped_mm_bf16(x, weight_t, weight_mxfp8, offsets):
    """BF16 branch of MoK impl_utils.grouped_mm: torch grouped GEMM over cumulative group offsets."""
    assert weight_mxfp8 is None
    return torch.nn.functional.grouped_mm(x, weight_t, offs=offsets)


def _mok_impl_utils():
    """MoK #17's ``benchmarks/impl_utils.py`` (torch grouped GEMM + torchao MXFP8
    helpers), loaded by path so FlashInfer's own ``benchmarks`` directory does not
    shadow it."""
    import importlib.util

    root = os.environ.get("MOK_B1_ROOT")
    if root is None:
        import mok

        root = str(Path(mok.__file__).resolve().parents[1])
    spec = importlib.util.spec_from_file_location(
        "mok_bench_impl_utils", Path(root) / "benchmarks" / "impl_utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMPLS = {"cake": CakeImpl, "mok17": Mok17Impl, "deepep": DeepEPImpl}


def gather_max(values, device):
    """Per-iteration slowest rank: all-gather the per-rank sample vectors."""
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered).max(dim=0).values.tolist()


def time_block(impl, checkpoint, warmup_ms, sample_ms, device):
    torch.cuda.synchronize()
    dist.barrier()
    # Warmup: at least warmup_ms of wall time on every rank (barrier-synchronous count).
    t0 = time.perf_counter()
    iters = 0
    while True:
        impl.step(checkpoint)
        iters += 1
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            [(time.perf_counter() - t0) * 1000.0], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MIN)
        if elapsed.item() >= warmup_ms and iters >= 2:
            break
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    t0 = time.perf_counter()
    while True:
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        impl.step(checkpoint)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        elapsed = torch.tensor(
            [(time.perf_counter() - t0) * 1000.0], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MIN)
        if elapsed.item() >= sample_ms and len(samples) >= MIN_SAMPLES:
            break
    count = torch.tensor([len(samples)], dtype=torch.int64, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.MIN)
    samples = samples[: int(count.item())]
    slowest = gather_max(samples, device)
    return dict(
        iterations=len(slowest),
        median_ms=statistics.median(slowest),
        min_ms=min(slowest),
        max_ms=max(slowest),
        mean_ms=statistics.fmean(slowest),
    )


def measure_memory(impl, checkpoint, device):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    outputs = impl.step(checkpoint)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) - base
    ctx = impl.context_bytes(impl.context) if impl.context is not None else 0
    out_bytes = sum(
        o.numel() * o.element_size() for o in outputs if isinstance(o, torch.Tensor)
    )
    del outputs
    return dict(
        peak_step_bytes=peak,
        context_bytes=ctx,
        output_bytes=out_bytes,
        workspace_bytes=impl.workspace_bytes(),
    )


def cross_check(results, device):
    """Max abs / relative L1 differences between implementations (sanity only)."""
    names = list(results)
    report = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]], results[names[j]]
            rows = []
            for k, (ta, tb) in enumerate(zip(a, b, strict=True)):
                if ta.numel() == 0:
                    rows.append(dict(index=k, empty=True))
                    continue
                diff = (ta.float() - tb.float()).abs()
                norm = tb.float().abs().sum().item()
                rows.append(
                    dict(
                        index=k,
                        max_abs=diff.max().item(),
                        rel_l1=(diff.sum().item() / norm) if norm else None,
                        finite=bool(torch.isfinite(ta).all().item()),
                    )
                )
            report[f"{names[i]}_vs_{names[j]}"] = rows
    return report


def main():
    args = parse_args()
    rank, ep, device = init_distributed()
    shape = SHAPES[args.shape]
    if shape.experts % ep:
        raise SystemExit(f"{shape.experts} experts do not divide EP={ep}")
    rows = source_rows(args, rank, ep)
    max_rows = max(rows)
    capacity = int(math.ceil(max_rows * args.capacity_mult))
    capacity = max(512, ((capacity + 255) // 256) * 256)
    inputs = generate_inputs(args, shape, rank, ep, rows, device)
    needed = routed_capacity_factor(inputs[1], shape, ep, shape.topk)
    base = capacity * shape.topk
    factor = max(2, math.ceil(needed / base + 1e-9))
    impl_names = [n for n in args.impls.split(",") if n]
    impls = {}
    for name in impl_names:
        if name not in IMPLS:
            raise SystemExit(
                f"unknown implementation {name!r} (available: {sorted(IMPLS)})"
            )
        impls[name] = IMPLS[name](
            args, shape, rank, ep, rows, inputs, device, capacity, factor
        )
    checkpoint = args.mode == "checkpoint"
    if rank == 0:
        print(
            f"== {shape.name} EP{ep} rows={rows} routing={args.routing} precision={args.precision} mode={args.mode} "
            f"macro={args.macrobatch} capacity={capacity} needed_routes={needed} factor={factor} impls={impl_names}",
            flush=True,
        )
    record = dict(
        shape=dataclasses.asdict(shape),
        ep=ep,
        rows=rows,
        routing=args.routing,
        hot_factor=hot_factor(args.routing),
        precision=args.precision,
        mode=args.mode,
        macrobatch=args.macrobatch,
        minibatch=args.minibatch,
        source_capacity=capacity,
        capacity_mult=args.capacity_mult,
        needed_routes_max_rank=needed,
        schedule_capacity_factor=factor,
        impls={n: impls[n].settings for n in impl_names},
        gpu=torch.cuda.get_device_name(device),
        sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        host=platform.node(),
        protocol=dict(
            groups=args.groups,
            warmup_ms=args.warmup_ms,
            sample_ms=args.sample_ms,
            min_samples=MIN_SAMPLES,
            timing="cuda events, full iteration, slowest rank per iteration, median per group",
        ),
        groups=[],
        memory={},
        env={k: os.environ[k] for k in sorted(os.environ) if k.startswith("C1105_")},
    )
    outputs = {}
    for name in impl_names:
        record["memory"][name] = measure_memory(impls[name], checkpoint, device)
        if args.check:
            outputs[name] = [o.detach().clone() for o in impls[name].step(checkpoint)]
    if args.check and len(outputs) > 1:
        record["cross_check"] = cross_check(outputs, device)
    outputs = None
    for group in range(args.groups):
        order = (
            impl_names[group % len(impl_names) :]
            + impl_names[: group % len(impl_names)]
        )
        if group % 2 == 1:
            order = list(reversed(order))
        timings = {}
        for name in order:
            timings[name] = time_block(
                impls[name], checkpoint, args.warmup_ms, args.sample_ms, device
            )
            if rank == 0:
                print(
                    f"  group {group} {name}: median {timings[name]['median_ms']:.3f} ms over "
                    f"{timings[name]['iterations']} iterations (min {timings[name]['min_ms']:.3f})",
                    flush=True,
                )
        record["groups"].append(dict(order=order, timings=timings))
    summary = {}
    for name in impl_names:
        medians = [g["timings"][name]["median_ms"] for g in record["groups"]]
        summary[name] = dict(
            median_of_group_medians_ms=statistics.median(medians),
            group_medians_ms=medians,
            spread_pct=100.0 * (max(medians) - min(medians)) / min(medians),
        )
    record["summary"] = summary
    for impl in impls.values():
        impl.teardown()
    if rank == 0:
        print(
            "RESULT "
            + json.dumps(
                dict(
                    shape=shape.name,
                    ep=ep,
                    rows=args.rows,
                    routing=args.routing,
                    precision=args.precision,
                    mode=args.mode,
                    macrobatch=args.macrobatch,
                    summary=summary,
                    memory=record["memory"],
                )
            ),
            flush=True,
        )
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w") as f:
                json.dump(record, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
