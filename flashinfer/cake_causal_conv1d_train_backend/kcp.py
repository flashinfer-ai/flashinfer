"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Context-parallel (KCP) halo forward / backward of the Cake packed width-4
depthwise causal convolution + SiLU (FlashInfer issue #6919): the rank's
fragments of the packed documents are convolved with the three document tokens
before each continuing fragment, received from the ranks that hold them by one
``all_to_all_single`` per direction, and the backward sends the first rows'
derivative back to the predecessor fragments, which fold it into their last
rows' ``dx`` before the one BF16 rounding.  ``cake_causal_conv1d_kcp(x_local,
weight, ctx)`` is the autograd entry; the stable, ``@flashinfer_api``-decorated
facade is :mod:`flashinfer.cake_causal_conv1d_train`.  This module imports
only torch, torch.distributed, tvm_ffi and its own package.

Registry contract (``kcp_jit``, the KCP records beside the zero-history
registry ``cake_jit``, whose tables stay untouched): kinds ``kcp_fwd`` (one
record per arch: co-resident grid, 4 CTAs per SM, 128 threads x 8 channels),
``kcp_bwd`` (the fallback record and the ``_st8`` record selected by
``channels_at_least`` = the zero-history rule; ``workspace_rule``
``partial_rows`` with 4 taps) and ``kcp_bwd_head`` (one record; its grid is
``max(1, n_frag x ceil(C / (threads x vec)))`` computed here and launched
through the record's ``arg_plan`` / ``logical`` / ``ffi_entry`` and ``load``);
logical tensor names are ``cake_jit.LOGICAL_TENSORS`` plus ``hist``,
``hist_row_off``, ``dpre_tail``, ``tail_row_off``, ``dpre_head``,
``head_row_off`` (``x_map`` / ``dy_map`` map to ``x`` / ``dy``); the scalars are
``cake_jit.LOGICAL_SCALARS``.

Numerics against the zero-history kernels of this package: the forward runs
the zero-history forward's instructions with history rows, so ``y`` is bitwise
equal to the no-CP ``y`` of the same documents.  The backward takes the SiLU
derivative on the FP32 preactivation (the issue's requirement), whereas the
shipped zero-history backward differentiates the BF16-rounded preactivation,
so ``dx`` is not the no-CP ``dx`` bit for bit: the two differ by that
rounding's effect on ``dpre`` plus one BF16 output rounding.  Measured on
GB300 (sm_103a) over the single-GPU test rows (``S x C`` up to 12000 x 8192,
random BF16 inputs, FP32 taps of scale 0.5): max ``|dx - dx_noCP|`` 0.03125
(one BF16 ulp at magnitude 4-8), 0 elements beyond ``1e-2 + 1e-2 |dx_noCP|``,
and the same maximum between the two policies' FP32 oracles with no kernel
involved; each kernel stays within one BF16 ulp of its own oracle.  ``dweight``
is the rank's fixed-order FP32 partial; it is reproducible across launches and
within ``5e-2 + 1e-4 |dw|`` of an FP64 oracle, not bitwise the no-CP partial.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import tvm_ffi

from . import (
    _cu_seqlens,
    _device_arch,
    _device_index,
    _output,
    _require_tensor,
    _scalars,
    _tokens,
    _weight,
)
from . import kcp_jit as _loader
from .cake_jit import WIDTH, launch_grid, partial_rows, plan
from .kcp_jit import load, route_modules, run_module, selected_modules

HALO = WIDTH - 1  # history rows, tail rows and head rows per fragment
HEAD_SLOTS = HALO  # send slots per head row (a head token is consumed by at most three predecessor fragments)
KCP_FWD = "kcp_fwd"
KCP_BWD = "kcp_bwd"
KCP_BWD_HEAD = "kcp_bwd_head"


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _table(
    ctx: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """One of the context's halo tables, checked against the kernels' ABI shape."""

    table = _require_tensor(ctx.halo_tables[name], f"ctx.halo_tables[{name!r}]")
    if tuple(table.shape) != shape or table.dtype != dtype:
        raise ValueError(
            f"ctx.halo_tables[{name!r}] must be {dtype} {shape}, got {table.dtype} {tuple(table.shape)}"
        )
    if table.device != device or not table.is_contiguous():
        raise ValueError(
            f"ctx.halo_tables[{name!r}] must be a contiguous tensor on {device}"
        )
    return table


def _pool(
    rows: int, C: int, *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """A fresh row pool; a pool without rows still holds one row so that the kernel has a valid pointer (no table
    entry names it)."""

    return torch.empty((max(rows, 1), C), dtype=dtype, device=device)


def _exchange(
    send: torch.Tensor,
    *,
    recv_rows: int,
    send_splits: tuple[int, ...],
    recv_splits: tuple[int, ...],
    ctx: Any,
) -> tuple[torch.Tensor, Optional[Any]]:
    """One ``all_to_all_single`` of ``send`` (rows in destination-rank order) into a fresh ``[recv_rows, C]`` tensor,
    issued asynchronously on the current stream; with one rank the send buffer is the received pool (no collective).
    Returns ``(recv, work)``; the caller waits ``work`` at the first consumer."""

    if int(ctx.world_size) == 1:
        return send, None
    recv = torch.empty((recv_rows, send.shape[1]), dtype=send.dtype, device=send.device)
    work = dist.all_to_all_single(
        recv,
        send,
        output_split_sizes=list(recv_splits),
        input_split_sizes=list(send_splits),
        group=ctx.group,
        async_op=True,
    )
    return recv, work


def _launch_with_grid(
    name: str, values: dict[str, Any], *, grid: tuple[int, int, int], device_index: int
) -> None:
    """Launch record ``name`` with logical ``values`` and an explicit grid (the head program's grid depends on the
    fragment and channel counts, not on the device); otherwise :func:`cake_jit.run_module`'s argument plan."""

    record = _loader.MODULES[name]
    logical = record["logical"]
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, _loader.device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


def _context(ctx: Any, *, device: torch.device, T_r: int) -> tuple[torch.Tensor, int]:
    """The context's local ``cu_seqlens`` (int32 device ``[n_frag + 1]``) and fragment count, checked against the
    tokens the call received."""

    if int(ctx.T_r) != T_r:
        raise ValueError(
            f"x_local holds {T_r} tokens but ctx was built for T_r={int(ctx.T_r)}"
        )
    if int(ctx.conv_kernel_size) != WIDTH:
        raise ValueError(
            f"ctx was built for a width-{int(ctx.conv_kernel_size)} convolution; this kernel is width {WIDTH}"
        )
    if int(ctx.n_frag) == 0:
        # A rank without fragments (T_r == 0): nothing is launched, the collectives get empty splits.
        if T_r != 0:
            raise ValueError(f"ctx has no fragments but x_local holds {T_r} tokens")
        return _require_tensor(ctx.cu_seqlens_local, "ctx.cu_seqlens_local"), 0
    cu, n_frag = _cu_seqlens(ctx.cu_seqlens_local, device=device)
    if n_frag != int(ctx.n_frag):
        raise ValueError(
            f"ctx.cu_seqlens_local names {n_frag} fragments but ctx.n_frag is {int(ctx.n_frag)}"
        )
    return cu, n_frag


def cake_causal_conv1d_kcp_forward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    *,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP forward: ``y`` of the rank's fragments, and the history pool ``hist`` it used (saved for the backward).

    ``x_local`` BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous, the rank's fragments back to back in the order of
    ``ctx.cu_seqlens_local``; ``weight`` FP32 ``[C, 4]``; ``ctx`` a ``KcpContext`` (its halo plan ``ctx.halo``, the
    device tables ``ctx.halo_tables`` and the process group ``ctx.group``).  Every rank of the group calls this
    function once per layer, in the same order: the forward halo exchange is a collective (a rank without tokens
    joins it with empty splits and returns empty outputs).  ``hist`` is a fresh ``[n_hist_rows, C]`` BF16 tensor
    (``[1, C]`` when the rank receives no row).
    """

    x = _require_tensor(x_local, "x_local")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    cu, n_frag = _context(ctx, device=device, T_r=S)
    halo = ctx.halo
    y = _output(out, "out", shape=tuple(x.shape), dtype=torch.bfloat16, device=device)
    hist_index = _require_tensor(
        ctx.halo_tables["hist_index"], "ctx.halo_tables['hist_index']"
    )
    send = (
        x_flat.index_select(0, hist_index)
        if hist_index.numel()
        else x_flat.new_empty((0, C))
    )
    hist, work = _exchange(
        send,
        recv_rows=int(halo.n_hist_rows),
        send_splits=tuple(halo.fwd_send_rows),
        recv_splits=tuple(halo.fwd_recv_rows),
        ctx=ctx,
    )
    if S == 0:
        if work is not None:
            work.wait()
        return y, hist
    hist_row_off = _table(
        ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    (name,) = route_modules(arch, KCP_FWD)
    device_index = _device_index(device)
    record = _loader.MODULES[name]
    scalars = plan(S, C, n_frag, geometry=record["grid_rule"]["geometry"])
    if scalars["n_ch_tiles"] > launch_grid(record, device_index=device_index)[0]:
        raise ValueError(
            f"C={C} spans more channel tiles than the forward's grid on this device holds"
        )
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "y": y.view(S, C),
        "hist": hist
        if hist.shape[0]
        else _pool(0, C, dtype=torch.bfloat16, device=device),
        "hist_row_off": hist_row_off,
        **_scalars(scalars),
    }
    if work is not None:
        work.wait()
    run_module(name, values, device_index=device_index)
    return y, hist


def cake_causal_conv1d_kcp_backward(
    x_local: torch.Tensor,
    weight: torch.Tensor,
    ctx: Any,
    dy: torch.Tensor,
    hist: torch.Tensor,
    *,
    dx: Optional[torch.Tensor] = None,
    dweight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The KCP backward: ``dx`` of the rank's fragments (the halo contributions of the successor fragments folded in
    before the one BF16 rounding) and the rank's FP32 ``dweight`` partial (fixed order; its sum over the ranks is the
    framework's).

    ``hist`` is the forward's pool.  The head program writes the derivative rows of every continuing fragment's first
    tokens into the backward exchange's send layout, the exchange delivers them to the predecessors as ``dpre_tail``,
    and the main program folds them.  Every rank calls this function once per layer, in the same order.
    """

    x = _require_tensor(x_local, "x_local")
    device = x.device
    arch = _device_arch(device)
    S, C, x_flat = _tokens(x, "x_local", device=device)
    w = _weight(weight, C, device=device)
    _, _, dy_flat = _tokens(dy, "dy", device=device, shape=tuple(x.shape))
    cu, n_frag = _context(ctx, device=device, T_r=S)
    halo = ctx.halo
    hist_t = _require_tensor(hist, "hist")
    if (
        hist_t.ndim != 2
        or int(hist_t.shape[1]) != C
        or hist_t.dtype != torch.bfloat16
        or not hist_t.is_contiguous()
    ):
        raise ValueError(
            f"hist must be the forward's contiguous BF16 [n_hist_rows, {C}] pool, got {hist_t.dtype} {tuple(hist_t.shape)}"
        )
    if hist_t.shape[0] == 0:
        # A valid pointer for the kernels; no table entry names a row of it.
        hist_t = _pool(0, C, dtype=torch.bfloat16, device=device)
    dx_out = _output(
        dx, "dx", shape=tuple(x.shape), dtype=torch.bfloat16, device=device
    )
    dw_out = _output(
        dweight, "dweight", shape=(C, WIDTH), dtype=torch.float32, device=device
    )
    device_index = _device_index(device)
    n_head = int(halo.n_head_rows)
    dpre_head = _pool(n_head, C, dtype=torch.float32, device=device)
    if S > 0 and n_head > 0:
        hist_row_off = _table(
            ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
        )
        head_row_off = _table(
            ctx,
            "head_row_off",
            shape=(n_frag, HALO, HEAD_SLOTS),
            dtype=torch.int32,
            device=device,
        )
        (head_name,) = route_modules(arch, KCP_BWD_HEAD)
        geometry = _loader.MODULES[head_name]["grid_rule"]["geometry"]
        channels_per_cta = int(geometry["threads"]) * int(geometry["vec"])
        grid = (max(1, n_frag * _ceil_div(C, channels_per_cta)), 1, 1)
        _launch_with_grid(
            head_name,
            {
                "x": x_flat,
                "w": w,
                "cu": cu,
                "dy": dy_flat,
                "hist": hist_t,
                "hist_row_off": hist_row_off,
                "dpre_head": dpre_head,
                "head_row_off": head_row_off,
                "S": S,
                "C": C,
                "num_docs": n_frag,
            },
            grid=grid,
            device_index=device_index,
        )
    dpre_tail, work = _exchange(
        dpre_head[:n_head],
        recv_rows=int(halo.n_tail_rows),
        send_splits=tuple(halo.bwd_send_rows),
        recv_splits=tuple(halo.bwd_recv_rows),
        ctx=ctx,
    )
    if S == 0:
        if work is not None:
            work.wait()
        dw_out.zero_()
        return dx_out, dw_out
    hist_row_off = _table(
        ctx, "hist_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    tail_row_off = _table(
        ctx, "tail_row_off", shape=(n_frag, HALO), dtype=torch.int32, device=device
    )
    names = selected_modules(arch, KCP_BWD, C=C, S=S)
    record = _loader.MODULES[names[0]]
    scalars = plan(S, C, n_frag, geometry=record["grid_rule"]["geometry"])
    grid_x = launch_grid(record, device_index=device_index)[0]
    if scalars["n_ch_tiles"] > grid_x:
        raise ValueError(
            f"C={C} spans more channel tiles than the backward's grid on this device holds"
        )
    rows = partial_rows(grid_x, scalars["n_ch_tiles"])
    partial = torch.empty((rows, C, WIDTH), dtype=torch.float32, device=device)
    values: dict[str, Any] = {
        "x": x_flat,
        "w": w,
        "cu": cu,
        "dy": dy_flat,
        "dx": dx_out.view(S, C),
        "dw": dw_out,
        "partial": partial,
        "hist": hist_t,
        "hist_row_off": hist_row_off,
        "dpre_tail": dpre_tail
        if dpre_tail.shape[0]
        else _pool(0, C, dtype=torch.float32, device=device),
        "tail_row_off": tail_row_off,
        **_scalars(scalars),
    }
    if work is not None:
        work.wait()
    for name in names:
        run_module(name, values, device_index=device_index)
    return dx_out, dw_out


class CakeCausalConv1dKcpFunction(torch.autograd.Function):
    """Autograd wrapper: the forward saves ``(x_local, weight)`` and the history pool ``hist`` (no second exchange in
    the backward); the backward runs the head program, the backward halo exchange and the main backward program."""

    @staticmethod
    def forward(fn_ctx, x_local, weight, ctx):
        y, hist = cake_causal_conv1d_kcp_forward(x_local, weight, ctx)
        fn_ctx.save_for_backward(x_local, weight, hist)
        fn_ctx.kcp = ctx
        return y

    @staticmethod
    def backward(fn_ctx, grad_y):
        x_local, weight, hist = fn_ctx.saved_tensors
        if grad_y is None:
            grad_y = torch.zeros_like(x_local)
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx, dweight = cake_causal_conv1d_kcp_backward(
            x_local, weight, fn_ctx.kcp, dy, hist
        )
        return dx, dweight, None


def cake_causal_conv1d_kcp(
    x_local: torch.Tensor, weight: torch.Tensor, ctx: Any
) -> torch.Tensor:
    """Autograd-enabled KCP packed width-4 depthwise causal convolution + SiLU of the rank's fragments.

    Parameters
    ----------
    x_local : torch.Tensor
        BF16 ``[T_r, C]`` (or ``[1, T_r, C]``) contiguous activations of the rank's fragments, back to back in the
        order of ``ctx.cu_seqlens_local``.
    weight : torch.Tensor
        FP32 ``[C, 4]`` contiguous taps; ``weight[c, 3]`` multiplies the current token.
    ctx : KcpContext
        The rank's context (``flashinfer.experimental.cake_kda_train.build_kcp_context``): fragment table, halo
        plan, process group.  Every rank of the group calls this function once per layer, in the same order.

    Returns
    -------
    torch.Tensor
        BF16 ``y`` of ``x_local``'s shape, with the three document tokens before every continuing fragment as
        history (zero history at a document's start).  The backward returns BF16 ``dx`` (the successor fragments'
        contributions folded in before the one rounding) and the rank's FP32 ``dweight`` partial.
    """

    return CakeCausalConv1dKcpFunction.apply(x_local, weight, ctx)


__all__ = [
    "CakeCausalConv1dKcpFunction",
    "cake_causal_conv1d_kcp",
    "cake_causal_conv1d_kcp_backward",
    "cake_causal_conv1d_kcp_forward",
]
