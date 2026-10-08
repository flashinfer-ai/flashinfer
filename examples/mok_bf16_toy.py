#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic MoK training example with independent BF16 autograd checks.

Run: torchrun --standalone --nproc-per-node=4 examples/mok_bf16_toy.py
Requires peer-accessible GPUs and PyTorch symmetric-memory multicast.

``MOK_TOY_PRECISION=mxfp8`` runs the native MXFP8 routed experts instead: the
routed weights are prequantized with ``flashinfer.mok.mxfp8_quantize`` and the
outputs are checked against the same recipe in FP32 PyTorch arithmetic (fake
quantization of the dispatched rows, the saved gate/up and the hidden and
gradient tiles, per row), the routed weight gradients against the plain FP32
reference with a relative-L1 bound (their token-axis block scales follow the
kernel's dispatch order, which the independent reference does not reproduce).
"""

import datetime
import json
import os

import torch
import torch.distributed as dist
from flashinfer.experimental.cake_mok_bf16.mxfp8_reference import (
    dequantize_mxfp8,
    mxfp8_quantize_reference,
)
from flashinfer.mok import (
    context_defined_rows,
    create_mok_bf16_workspace,
    mxfp8_quantize,
    prepare_mok_bf16,
)


RESULT_NAMES = (
    "y",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


class _ExpertLinear(torch.autograd.Function):
    """``x @ w.T`` whose weight gradient is the FP32 product of the operands.

    torch's BF16 ``grad_out.T @ x`` (transposed operand, K = routed row count)
    silently drops the last ``K mod 256`` rows for some K on current cuBLAS
    builds: observed for K in 9219..9731 with K mod 256 in 1..3 on sm_100a,
    sm_103a and sm_107a (CUDA 13.1 and 13.5 torch builds), e.g. the 9475
    routed rows of ``mok_bf16_unequal`` case 2.  The kernels accumulate in
    FP32 and round once, so the FP32 product is also the faithful reference.
    The input gradient keeps the regular matmul (K = feature count).
    """

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x @ w.T

    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        grad_x = grad_out @ w if ctx.needs_input_grad[0] else None
        grad_w = None
        if ctx.needs_input_grad[1]:
            tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                grad_w = (grad_out.float().T.contiguous() @ x.float()).to(w.dtype)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf32
        return grad_x, grad_w


def expert(x, gate, up, down, swiglu_limit=None):
    a, b = _ExpertLinear.apply(x, gate), _ExpertLinear.apply(x, up)
    a, b = a.float(), b.float()
    if swiglu_limit is not None:
        # GLM-5.3-Flash clamp; torch.clamp's inclusive-boundary derivative
        # matches the kernel's masks.
        a = torch.clamp(a, max=swiglu_limit)
        b = torch.clamp(b, min=-swiglu_limit, max=swiglu_limit)
    # Native SwiGLU promotes BF16 inputs to FP32, then rounds the product once.
    hidden = (torch.nn.functional.silu(a) * b).to(x.dtype)
    return _ExpertLinear.apply(hidden, down)


def reference(
    global_data, weights, *, fp32=False, source_counts=None, swiglu_limit=None
):
    rank, ep = dist.get_rank(), dist.get_world_size()
    device = weights[0].device
    dtype = torch.float32 if fp32 else torch.bfloat16
    x = global_data["x"].to(device=device, dtype=dtype)
    dy = global_data["d_output"].to(device=device, dtype=dtype)
    ids = global_data["expert_ids"].to(device)
    scores = global_data["scores"].to(device)
    t, h = x.shape
    local = t // ep
    if source_counts is None:
        source_counts = [local] * ep
    if len(source_counts) != ep or sum(source_counts) != t:
        raise ValueError("Source counts must cover the global input exactly")
    start = sum(source_counts[:rank])
    own_rows = slice(start, start + source_counts[rank])
    local_experts = weights[3].shape[0]
    y_sum = torch.zeros((t, h), dtype=torch.float32, device=device)
    dx_sum = torch.zeros_like(y_sum)
    dscores = torch.zeros_like(scores)
    dw = [torch.zeros_like(w, dtype=dtype) for w in weights[3:]]
    for e in range(local_experts):
        rows, slots = (ids == rank * local_experts + e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xe = x[rows].detach().requires_grad_()
        we = [w[e].detach().to(dtype).requires_grad_() for w in weights[3:]]
        out = expert(xe, *we, swiglu_limit=swiglu_limit)
        # Differentiating already-scaled score inputs; scaling is not repeated.
        se = scores[rows, slots].detach().requires_grad_()
        weighted = out.float() * se[:, None]
        grads = torch.autograd.grad(weighted, (xe, *we, se), dy[rows].float())
        y_sum.index_add_(0, rows, weighted.detach())
        dx_sum.index_add_(0, rows, grads[0].float())
        dscores[rows, slots] = grads[-1]
        for dest, grad in zip(dw, grads[1:4], strict=True):
            dest[e].copy_(grad)
    for tensor in (y_sum, dx_sum, dscores):
        dist.all_reduce(tensor)
    xs = x[own_rows].detach().requires_grad_()
    ws = [w.detach().to(dtype).requires_grad_() for w in weights[:3]]
    ys = expert(xs, *ws, swiglu_limit=swiglu_limit)
    shared_grads = torch.autograd.grad(ys, (xs, *ws), dy[own_rows])
    y = (y_sum[own_rows] + ys.detach().float()).to(dtype)
    dx = (dx_sum[own_rows] + shared_grads[0].float()).to(dtype)
    return (y, dx, dscores[own_rows], *dw, *shared_grads[1:])


def _fake_quant_rows(rows_bf16):
    """Dequantized MXFP8 values of BF16 rows (per-row 32-element blocks, as the kernels)."""
    padded = (rows_bf16.shape[0] + 127) // 128 * 128
    buffer = torch.zeros(
        padded, rows_bf16.shape[1], dtype=torch.bfloat16, device=rows_bf16.device
    )
    buffer[: rows_bf16.shape[0]] = rows_bf16
    fp8, sc, _, _ = mxfp8_quantize_reference(buffer, True, False)
    return dequantize_mxfp8(fp8, sc)[: rows_bf16.shape[0]]


def _dequant_experts(quantized, index):
    """Dequantized FP32 weights of one ``mxfp8_quantize`` tuple member (0: normal, 2: transposed)."""
    data, scales = quantized[index], quantized[index + 1]
    tiles = scales.view(data.shape[0], -1, *scales.shape[1:])
    return [dequantize_mxfp8(data[e], tiles[e]) for e in range(data.shape[0])]


def reference_mxfp8(
    global_data, weights, quantized, *, source_counts=None, swiglu_limit=None
):
    """Fake-quant FP32 reference of the MXFP8 routed experts; the shared experts and the
    routed weight gradients come from :func:`reference` (BF16 and FP32 respectively)."""
    rank, ep = dist.get_rank(), dist.get_world_size()
    device = weights[0].device
    x = global_data["x"].to(device)
    dy = global_data["d_output"].to(device)
    ids = global_data["expert_ids"].to(device)
    scores = global_data["scores"].to(device)
    t, h = x.shape
    local = t // ep
    if source_counts is None:
        source_counts = [local] * ep
    start = sum(source_counts[:rank])
    own_rows = slice(start, start + source_counts[rank])
    local_experts = weights[3].shape[0]
    wg, wu, wd = (_dequant_experts(q, 0) for q in quantized)
    wg_t, wu_t, wd_t = (_dequant_experts(q, 2) for q in quantized)
    y_sum = torch.zeros((t, h), dtype=torch.float32, device=device)
    dx_sum = torch.zeros_like(y_sum)
    dscores = torch.zeros_like(scores)
    for e in range(local_experts):
        rows, slots = (ids == rank * local_experts + e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        xq = _fake_quant_rows(x[rows])
        gate = (xq @ wg[e].T).bfloat16()
        up = (xq @ wu[e].T).bfloat16()
        # The saved context holds gate/up as E4M3 (per-row blocks of the BF16 values).
        gate_f, up_f = _fake_quant_rows(gate), _fake_quant_rows(up)
        if swiglu_limit is not None:
            gate_mask = gate_f <= swiglu_limit
            up_mask = (up_f >= -swiglu_limit) & (up_f <= swiglu_limit)
            gate_f = torch.clamp(gate_f, max=swiglu_limit)
            up_f = torch.clamp(up_f, min=-swiglu_limit, max=swiglu_limit)
        gate_b, up_b = gate.float(), up.float()
        if swiglu_limit is not None:
            gate_b = torch.clamp(gate_b, max=swiglu_limit)
            up_b = torch.clamp(up_b, min=-swiglu_limit, max=swiglu_limit)
        hidden = (gate_b * torch.sigmoid(gate_b) * up_b).bfloat16()
        se = scores[rows, slots]
        # Per-expert outputs are BF16 ring rows; the combine scales and sums them in FP32.
        y_e = (_fake_quant_rows(hidden) @ wd[e].T).bfloat16().float()
        y_sum.index_add_(0, rows, y_e * se[:, None])
        dh = (_fake_quant_rows(dy[rows]) @ wd_t[e].T).bfloat16().float()
        sigmoid = torch.sigmoid(gate_f)
        silu = gate_f * sigmoid
        dscores[rows, slots] = (dh * (silu * up_f)).sum(-1)
        dhs = dh * se[:, None]
        dg = ((1.0 - silu) * sigmoid + silu) * up_f * dhs
        du = silu * dhs
        if swiglu_limit is not None:
            dg = torch.where(gate_mask, dg, torch.zeros_like(dg))
            du = torch.where(up_mask, du, torch.zeros_like(du))
        dgq, duq = _fake_quant_rows(dg.bfloat16()), _fake_quant_rows(du.bfloat16())
        dx_sum.index_add_(
            0, rows, (dgq @ wg_t[e].T + duq @ wu_t[e].T).bfloat16().float()
        )
    for tensor in (y_sum, dx_sum, dscores):
        dist.all_reduce(tensor)
    xs = x[own_rows].detach().requires_grad_()
    ws = [w.detach().requires_grad_() for w in weights[:3]]
    ys = expert(xs, *ws, swiglu_limit=swiglu_limit)
    shared_grads = torch.autograd.grad(ys, (xs, *ws), dy[own_rows])
    y = (y_sum[own_rows] + ys.detach().float()).bfloat16()
    dx = (dx_sum[own_rows] + shared_grads[0].float()).bfloat16()
    # Routed weight gradients: plain FP32 reference (relative-L1 gate).
    oracle = reference(
        global_data,
        weights,
        fp32=True,
        source_counts=source_counts,
        swiglu_limit=swiglu_limit,
    )
    return (y, dx, dscores[own_rows], *oracle[3:6], *shared_grads[1:])


# Gates per output. BF16: exact elementwise atol/rtol. MXFP8: the same rule with the
# ``max(4, 2e-7 * numel)`` exception allowance of the fake-quant reference, and a
# relative-L1 bound for the routed weight gradients against the FP32 reference.
BF16_GATE = dict(atol=1e-2, rtol=1e-2)
MXFP8_GATES = {
    name: (
        dict(relative_l1=0.12)
        if name.startswith("d_w_routed")
        else dict(atol=1e-2, rtol=1e-2, exceptions=True)
        if name in ("y", "d_x", "d_router_weights")
        else BF16_GATE  # shared experts stay BF16: exact rule as in BF16 mode
    )
    for name in RESULT_NAMES
}


def error_report(actual, expected, gate):
    """Apply the per-output gates on every rank; retain aggregate diagnostics."""
    gates = gate if set(gate) == set(RESULT_NAMES) else {n: gate for n in RESULT_NAMES}
    reports = {}
    for name, a, b in zip(RESULT_NAMES, actual, expected, strict=True):
        rule = gates[name]
        atol, rtol = float(rule.get("atol", 1e-2)), float(rule.get("rtol", 1e-2))
        if not (0 < atol < float("inf") and 0 <= rtol < float("inf")):
            raise ValueError("Require finite atol > 0 and rtol >= 0")
        assert a.shape == b.shape, (name, a.shape, b.shape)
        # First three fields use MAX; the remaining five use SUM across ranks.
        # Chunking bounds temporary storage for full expert weight gradients.
        stats = torch.zeros(8, dtype=torch.float64, device=a.device)
        worst = torch.zeros(4, dtype=torch.float64, device=a.device)
        offset = 0
        for aa, bb in zip(
            a.flatten().split(1048576), b.flatten().split(1048576), strict=True
        ):
            if aa.numel() == 0:
                continue
            af, bf = aa.float(), bb.float()
            diff = (af - bf).abs()
            allowed = atol + rtol * bf.abs()
            finite = torch.isfinite(af) & torch.isfinite(bf)
            mismatch = (~finite) | (diff > allowed)
            ratio = torch.nan_to_num(
                diff / allowed, nan=float("inf"), posinf=float("inf")
            )
            peak, index = ratio.max(dim=0)
            example = torch.stack(
                (
                    index.double() + offset,
                    af[index].double(),
                    bf[index].double(),
                    allowed[index].double(),
                )
            )
            worst = torch.where(peak > stats[1], example, worst)
            stats[0] = torch.maximum(
                stats[0], torch.nan_to_num(diff, nan=float("inf")).max().double()
            )
            stats[1] = torch.maximum(stats[1], peak.double())
            stats[2] = torch.maximum(
                stats[2],
                torch.nan_to_num(diff - allowed, nan=float("inf")).max().double(),
            )
            stats[3] += diff.double().sum()
            stats[4] += bf.abs().double().sum()
            stats[5] += (~torch.isfinite(af)).sum() + (~torch.isfinite(bf)).sum()
            stats[6] += mismatch.sum()
            stats[7] += aa.numel()
            offset += aa.numel()
        local = stats.tolist()
        local_worst = worst.tolist()
        dist.all_reduce(stats[:3], op=dist.ReduceOp.MAX)
        dist.all_reduce(stats[3:], op=dist.ReduceOp.SUM)
        maximum, ratio, excess, error, norm, nonfinite, mismatched, elements = (
            stats.tolist()
        )
        relative = error / norm if norm else (0.0 if error == 0 else float("inf"))
        local_relative = (
            local[3] / local[4]
            if local[4]
            else (0.0 if local[3] == 0 else float("inf"))
        )
        reports[name] = {
            "shape": list(a.shape),
            "actual_dtype": str(a.dtype),
            "atol": atol,
            "rtol": rtol,
            "rank_max_absolute": local[0],
            "rank_max_error_ratio": local[1],
            "rank_max_tolerance_excess": local[2],
            "rank_abs_error_sum": local[3],
            "rank_reference_l1": local[4],
            "rank_relative_l1": local_relative,
            "rank_nonfinite": int(local[5]),
            "rank_mismatched": int(local[6]),
            "rank_elements": int(local[7]),
            "rank_worst_error": {
                "flat_index": int(local_worst[0]),
                "actual": local_worst[1],
                "reference": local_worst[2],
                "allowed_error": local_worst[3],
            },
            "global_max_absolute": maximum,
            "global_relative_l1": relative,
            "global_max_error_ratio": ratio,
            "global_max_tolerance_excess": excess,
            "global_nonfinite": int(nonfinite),
            "global_mismatched": int(mismatched),
            "global_elements": int(elements),
        }
        if "relative_l1" in rule:
            passed = nonfinite == 0 and relative <= float(rule["relative_l1"])
            reports[name]["relative_l1_bound"] = float(rule["relative_l1"])
        else:
            allowed = max(4, 2e-7 * elements) if rule.get("exceptions") else 0
            passed = nonfinite == 0 and mismatched <= allowed
            reports[name]["allowed_mismatches"] = int(allowed)
        reports[name]["pass"] = bool(passed)
    return reports


class TrainingIteration:
    def __init__(
        self,
        config,
        workspace,
        x,
        ids,
        scores,
        dy,
        weights,
        *,
        functional,
        swiglu_limit=None,
        recompute=False,
        forward_weights=None,
        backward_weights=None,
    ):
        """``weights`` are the six BF16 reference weights; ``forward_weights`` /
        ``backward_weights`` (MXFP8: MoK's tuple conventions) override what the
        kernels receive."""
        if (
            x.dtype != torch.bfloat16
            or dy.dtype != torch.bfloat16
            or scores.dtype != torch.float32
            or ids.dtype != torch.int64
            or len(weights) != 6
            or any(
                not isinstance(v, torch.Tensor) or v.dtype != torch.bfloat16
                for v in weights
            )
        ):
            raise ValueError(
                "This workload requires BF16 x/dY/expert tensors, FP32 scores and int64 IDs"
            )
        if (
            x.shape != dy.shape
            or ids.shape != scores.shape
            or ids.shape[0] != x.shape[0]
        ):
            raise ValueError(
                "Source input, routing and upstream-gradient shapes must agree"
            )
        self.config, self.workspace = config, workspace
        self.functional = functional
        self.x, self.ids, self.scores, self.dy = x, ids, scores, dy
        self.weights = weights
        self.forward_weights = forward_weights or weights
        self.backward_weights = backward_weights or weights
        self.swiglu_limit = swiglu_limit
        # ``recompute``: checkpoint step = forward + recompute_forward_context
        # + backward from the recomputed context (asserted bitwise equal).
        self.recompute = bool(recompute)
        self.graph = None
        self.outputs = None

    def _execute(self):
        self.schedule = self.functional.build_schedule(
            self.workspace,
            self.config,
            self.ids,
            num_local_experts=self.weights[3].shape[0],
        )
        y, self.context = self.functional.forward(
            self.config,
            self.workspace,
            self.schedule,
            self.x,
            self.scores,
            *self.forward_weights,
            swiglu_limit=self.swiglu_limit,
        )
        context = self.context
        if self.recompute:
            context = self.functional.recompute_forward_context(
                self.config,
                self.workspace,
                self.schedule,
                self.x,
                self.forward_weights[0],
                self.forward_weights[1],
                self.forward_weights[3],
                self.forward_weights[4],
                swiglu_limit=self.swiglu_limit,
            )
            if not torch.cuda.is_current_stream_capturing():
                # Only the defined rows are comparable (ring rows beyond the
                # retained macrobatch / real source rows are never written).
                routed_rows, shared_rows = context_defined_rows(self.config, context)

                def rows(value, count, transposed):
                    # MXFP8 context entries are (E4M3, scale tiles) pairs; ``x`` and
                    # ``hidden`` are stored transposed, scale tiles per 128-row block.
                    if not isinstance(value, tuple):
                        return (value[:count],)
                    data, scales = value
                    blocks = count // 128
                    if transposed:
                        return (
                            data[:, :count],
                            scales.view(data.shape[0] // 128, -1, 32, 16)[:, :blocks],
                        )
                    return (data[:count], scales[: blocks * (data.shape[1] // 128)])

                def defined(ctx):
                    return (
                        *rows(ctx.x_routed, routed_rows, True),
                        ctx.gate_shared[:shared_rows],
                        *rows(ctx.gate_routed, routed_rows, False),
                        ctx.up_shared[:shared_rows],
                        *rows(ctx.up_routed, routed_rows, False),
                        ctx.hidden_shared[:shared_rows],
                        *rows(ctx.hidden_routed, routed_rows, True),
                    )

                saved, rebuilt = defined(self.context), defined(context)
                if not all(
                    torch.equal(a, b) for a, b in zip(saved, rebuilt, strict=True)
                ):
                    raise RuntimeError("Recomputed context differs from the saved one")
        gradients = self.functional.backward(
            self.config,
            self.workspace,
            self.schedule,
            context,
            self.dy,
            self.x,
            self.scores,
            *self.backward_weights,
            swiglu_limit=self.swiglu_limit,
        )
        self.outputs = (y, *gradients)
        if any(
            v.dtype != (torch.float32 if idx == 2 else torch.bfloat16)
            for idx, v in enumerate(self.outputs)
        ):
            raise RuntimeError("BF16 output/gradient formats changed")
        return self.outputs

    def capture(self):
        if self.graph is not None:
            raise RuntimeError("Iteration is already captured")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._execute()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self._execute()

    def run(self):
        if self.graph is None:
            return self._execute()
        self.graph.replay()
        return self.outputs


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    owns_group = not dist.is_initialized()
    if owns_group:
        # Cold JIT compilation on one rank can hold the others in the first
        # collective for minutes; MOK_PG_TIMEOUT_S widens the watchdog timeout.
        timeout_s = int(os.environ.get("MOK_PG_TIMEOUT_S", "600"))
        dist.init_process_group(
            "nccl", device_id=device, timeout=datetime.timedelta(seconds=timeout_s)
        )
    rank, ep = dist.get_rank(), dist.get_world_size()
    assert ep in (1, 4, 16, 64), "Launch with 1, 4, 16 or 64 ranks"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokens = 512
    topk, local_experts = (2, 4) if ep in (1, 4) else (8, 256 // ep)
    total_experts = ep * local_experts
    mxfp8 = os.environ.get("MOK_TOY_PRECISION", "bf16") == "mxfp8"
    functional = prepare_mok_bf16(
        ep_size=ep, local_experts=local_experts, topk=topk, mxfp8=mxfp8
    )
    gate = MXFP8_GATES if mxfp8 else BF16_GATE
    cases = []
    for hidden, intermediate in ((256, 256), (512, 512)):
        config, workspace = create_mok_bf16_workspace(
            group=dist.group.WORLD,
            device=device,
            num_local_tokens=tokens,
            hidden_size=hidden,
            topk=topk,
            fwd_num_comm_sms=8,
            bwd_num_comm_sms=8,
            minibatch_size=256,
            macrobatch_size=768,
            schedule_capacity_multiplier=2.0,
        )
        options = dict(dtype=torch.bfloat16, device=device)
        torch.manual_seed(32471)
        shared = [
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(intermediate, hidden, **options) / hidden**0.5,
            torch.randn(hidden, intermediate, **options) / intermediate**0.5,
        ]
        global_weights = [
            torch.randn(total_experts, intermediate, hidden, **options) / hidden**0.5,
            torch.randn(total_experts, intermediate, hidden, **options) / hidden**0.5,
            torch.randn(total_experts, hidden, intermediate, **options)
            / intermediate**0.5,
        ]
        weights = (
            *shared,
            *(
                w[rank * local_experts : (rank + 1) * local_experts]
                for w in global_weights
            ),
        )
        x, dy = (
            torch.empty(tokens, hidden, **options),
            torch.empty(tokens, hidden, **options),
        )
        ids = torch.empty(tokens, topk, dtype=torch.int64, device=device)
        scores = torch.empty(tokens, topk, device=device)
        if mxfp8:
            # MoK's conventions: forward (w_fp8, w_sc) pairs; backward gate/up 4-tuples
            # and the down (w_t_fp8, w_t_sc) pair.
            quantized = [mxfp8_quantize(w, True, True) for w in weights[3:]]
            kernel_weights = dict(
                forward_weights=(*weights[:3], *((q[0], q[1]) for q in quantized)),
                backward_weights=(
                    *weights[:3],
                    quantized[0],
                    quantized[1],
                    (quantized[2][2], quantized[2][3]),
                ),
            )

            def expected_outputs(data):
                return reference_mxfp8(data, weights, quantized)

        else:
            kernel_weights = {}

            def expected_outputs(data):
                return reference(data, weights)

        iteration = TrainingIteration(
            config,
            workspace,
            x,
            ids,
            scores,
            dy,
            weights,
            functional=functional,
            **kernel_weights,
        )

        def fill(generation, empty):
            torch.manual_seed(8291 + generation)
            all_x = torch.randn(tokens * ep, hidden, **options) * 0.125
            all_dy = torch.randn_like(all_x) * 0.125
            all_scores = torch.rand(tokens * ep, topk, device=device) + 0.1
            all_scores.div_(all_scores.sum(-1, keepdim=True)).mul_(2.5)
            rows = torch.arange(tokens * ep, device=device) + generation
            all_ids = torch.stack(
                [
                    (rows + slot) % (topk if empty else total_experts)
                    for slot in range(topk)
                ],
                -1,
            )
            section = slice(rank * tokens, (rank + 1) * tokens)
            for destination, value in zip(
                (x, dy, ids, scores), (all_x, all_dy, all_ids, all_scores), strict=True
            ):
                destination.copy_(value[section])
            torch.cuda.synchronize()
            dist.barrier()
            return dict(x=all_x, d_output=all_dy, expert_ids=all_ids, scores=all_scores)

        reports, loads = [], []

        def check(actual, expected, generation, empty):
            torch.cuda.synchronize()
            errors = error_report(actual, expected, gate)
            assert all(value["pass"] for value in errors.values()), (
                rank,
                hidden,
                generation,
                errors,
            )
            loads.append(iteration.schedule.num_tokens.item())
            report = dict(generation=generation, empty=empty, errors=errors)
            reports.append(report)
            print(
                f"rank {rank}: H{hidden} generation={generation} empty={empty} all nine PASS",
                flush=True,
            )

        for generation, empty in ((0, False), (1, True), (2, False)):
            data = fill(generation, empty)
            expected = expected_outputs(data)
            check(iteration.run(), expected, generation, empty)
        iteration.capture()
        saved = []
        for generation, empty in (
            (11, False),
            (11, False),
            (11, False),
            (17, True),
            (23, False),
        ):
            data = fill(generation, empty)
            expected = expected_outputs(data)
            actual = iteration.run()
            check(actual, expected, generation, empty)
            if generation == 11:
                saved.append(tuple(value.clone() for value in actual))
        assert all(
            all(torch.equal(a, b) for a, b in zip(saved[0], other, strict=True))
            for other in saved[1:]
        )
        # Checkpoint step (forward + recompute_forward_context + backward from
        # the recomputed context) reproduces the saved-context step bitwise,
        # eagerly and as a captured graph.
        checkpoint = TrainingIteration(
            config,
            workspace,
            x,
            ids,
            scores,
            dy,
            weights,
            functional=functional,
            recompute=True,
            **kernel_weights,
        )
        fill(11, False)
        for _ in range(2):
            actual = checkpoint.run()
            torch.cuda.synchronize()
            assert all(
                torch.equal(a, b) for a, b in zip(saved[0], actual, strict=True)
            ), (rank, hidden, "checkpoint step differs from the saved-context step")
            if checkpoint.graph is None:
                checkpoint.capture()
        cases.append(
            dict(
                hidden=hidden,
                intermediate=intermediate,
                precision="mxfp8" if mxfp8 else "bf16",
                source_tokens_per_rank=tokens,
                global_source_tokens=tokens * ep,
                topk=topk,
                global_experts=total_experts,
                macro=config.macrobatch_size,
                mini=config.minibatch_size,
                schedule_capacity=workspace.storage.schedule_capacity,
                routed_loads=loads,
                reports=reports,
                three_graph_replays_bitwise_equal=True,
                checkpoint_recompute_bitwise_equal=True,
            )
        )
    if rank == 0:
        print(json.dumps(dict(status="PASS", ep=ep, cases=cases), indent=2))
    if owns_group:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
