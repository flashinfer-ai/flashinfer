# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
# SPDX-License-Identifier: Apache-2.0
# Derived from sgl-project/sglang at 9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2.
# Modified 2026-11-04: explicit tensor ABI, unified plan and caller-owned workspaces.
"""HC preparation and execution, independent of any model/serving framework."""

import math
from dataclasses import dataclass

import torch

from flashinfer.api_logging import flashinfer_api


def _overlap(a, b):
    if not a.numel() or not b.numel():
        return False
    return (
        a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
        and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size()
    )


def _tensor(tensor, shape, dtype, device, name, alignment=16):
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != tuple(shape)
        or tensor.dtype != dtype
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % alignment
        or tensor.requires_grad
    ):
        raise ValueError(f"{name}: invalid shape, dtype, device, alignment or layout")


def _contract(h, lowrank, eps):
    if type(h) is not int or not 0 < h <= 16384 or h % 512:
        raise ValueError("H must be a positive multiple of 512 <= 16384")
    if (
        type(lowrank) is not int
        or lowrank <= 0
        or lowrank % 64
        or lowrank + 4 > (lowrank + 127) // 128 * 128
    ):
        raise ValueError(
            "L must be 64-aligned with room for four Inject rows in padded scratch"
        )
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("rms_eps must be finite and positive")


def _target_device(device, dtype):
    if device.type != "cuda" or dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("HC requires CUDA BF16/FP16 weights")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        raise ValueError("HC sum-state backend requires SM100 or SM103")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare/bind HC weights outside CUDA Graph capture")
    from .jit import require_sources

    require_sources()


def _device(tensor):
    _target_device(tensor.device, tensor.dtype)


def apply_sum(block_output, residual, alpha, *, out, sum_sq, partial_sums=None):
    if not isinstance(residual, torch.Tensor) or residual.ndim != 2:
        raise ValueError("residual must have shape [T,4*H]")
    rows, width = residual.shape
    h = width // 4
    if width % 4 or not 0 < h <= 16384 or h % 512 or rows >= 2**31:
        raise ValueError("Apply requires C=4, 512-aligned H<=16384 and T<2**31")
    if residual.device.type != "cuda" or residual.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError("Apply requires CUDA BF16/FP16 residual")
    if torch.cuda.get_device_capability(residual.device) not in ((10, 0), (10, 3)):
        raise ValueError("HC sum-state requires SM100 or SM103")
    dtype, device = residual.dtype, residual.device
    specs = [
        (residual, (rows, width), dtype, 32),
        (block_output, (rows, h), dtype, 32),
        (alpha, (rows, 4), torch.float32, 16),
        (out, (rows, width), dtype, 32),
        (sum_sq, (rows, 4), torch.float32, 16),
    ]
    if partial_sums is not None:
        specs.append((partial_sums, (rows, 4, 10), torch.float32, 16))
    for tensor, shape, dt, alignment in specs:
        _tensor(tensor, shape, dt, device, "Apply tensor", alignment)
    tensors = [spec[0] for spec in specs]
    for index, dest in enumerate(tensors[3:], 3):
        if any(_overlap(dest, other) for other in tensors[:index]):
            raise ValueError("Apply outputs must not alias inputs or each other")
    if rows:
        from .jit import module

        mod = module(h, dtype)
        if partial_sums is None:
            mod.apply(block_output, residual, alpha, out, sum_sq)
        else:
            mod.apply_partials(block_output, residual, alpha, out, partial_sums)
            mod.reduce(partial_sums, sum_sq)
    return out, sum_sq


@dataclass(frozen=True, eq=False)
class HCWorkspace:
    """Disjoint buffers for one live mix/core/apply invocation at a fixed T.

    Reuse sequentially on one stream. Concurrent invocations require separate
    workspaces. A captured graph retains these tensor addresses. Do not resize
    or rebind tensor storage while a workspace or captured graph is live.
    """

    plan: object
    rows: int
    deterministic: bool
    mixed: torch.Tensor
    alpha: torch.Tensor
    residual: torch.Tensor
    sum_sq: torch.Tensor
    down: torch.Tensor
    partial: torch.Tensor | None
    logits: torch.Tensor | None
    partial_sums: torch.Tensor | None


class HCSumPlan:
    """Prepared weight views plus shape dispatch; no hidden execution scratch."""

    def __init__(self, packed_down, up, norm, lowrank, rms_eps):
        self.packed_down = packed_down
        self.up = up.detach()
        self.norm = norm.detach().view(-1)
        self.hidden_size = packed_down.shape[2]
        self.lowrank = lowrank
        self.rms_eps = rms_eps
        self.kpad = (lowrank + 127) // 128 * 128
        # Shared [H,4] output ordering for the small-T fused Up/Mix.
        h = self.hidden_size
        padded = up.new_zeros((h, 4, self.kpad))
        padded[:, :, :lowrank].copy_(up.view(4, h, lowrank).transpose(0, 1))
        self.up_padded = padded.view(4 * h, self.kpad)
        self.max_rows = min((2**31 - 1) // 4, (2**31 - 1) * 256 // max(h, lowrank + 8))

    def _rows(self, rows):
        if type(rows) is not int or not 0 <= rows <= self.max_rows:
            raise ValueError("HC token count exceeds supported CUDA launch bounds")

    def _residual(self, residual):
        if not isinstance(residual, torch.Tensor) or residual.ndim != 2:
            raise ValueError("residual must have shape [T,4*H]")
        rows = residual.shape[0]
        self._rows(rows)
        _tensor(
            residual,
            (rows, 4 * self.hidden_size),
            self.up.dtype,
            self.up.device,
            "residual",
            32,
        )
        return rows

    def allocate_workspace(self, rows, *, deterministic=False):
        """Allocate once before capture; repeated mix/apply performs no allocation."""
        self._rows(rows)
        if type(deterministic) is not bool:
            raise ValueError("deterministic must be bool")
        with torch.cuda.device(self.up.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("allocate HC workspaces outside CUDA Graph capture")
        return self._allocate_workspace(rows, deterministic=deterministic)

    def _allocate_workspace(self, rows, *, deterministic, small_down=None):
        # Runtime adapters may allocate outputs in a graph's memory pool while
        # supplying already initialized small-T scratch. No initialization
        # kernel is captured in that case; Mix resets the next sums itself.
        h = self.hidden_size

        def new(shape, dtype=None, zero=False):
            return (torch.zeros if zero else torch.empty)(
                shape, device=self.up.device, dtype=dtype or self.up.dtype
            )

        return HCWorkspace(
            self,
            rows,
            deterministic,
            new((rows, h)),
            new((rows, 4), torch.float32),
            new((rows, 4 * h)),
            new((rows, 4), torch.float32),
            small_down
            if small_down is not None
            else new(
                (rows, self.kpad if rows <= 24 else self.lowrank), zero=rows <= 24
            ),
            new((4, rows, self.packed_down.shape[1]), torch.float32)
            if rows > 24
            else None,
            new((rows, 4 * h)) if rows > 24 else None,
            new((rows, 4, 10), torch.float32) if deterministic else None,
        )

    @flashinfer_api
    def bootstrap(self, residual, *, out_sum=None):
        """Compute initial FP32 sums from stored residuals, not pre-cast updates.

        Pass out_sum for allocation-free replay. Later blocks consume the
        pre-storage-cast sum returned by apply directly, without bootstrapping.

        Parameters
        ----------
        residual : torch.Tensor
            Contiguous BF16/FP16 residual ``[T,4*H]`` matching this plan.
        out_sum : torch.Tensor, optional
            FP32 ``[T,4]`` destination, disjoint from residual and all weights.
            Omit to allocate; use32-byte alignment for the small-T consumer.

        Returns
        -------
        torch.Tensor
            Initial square sums in the supplied or allocated destination.
        """
        rows = self._residual(residual)
        if out_sum is None:
            out_sum = torch.empty(
                (rows, 4), device=residual.device, dtype=torch.float32
            )
        _tensor(out_sum, (rows, 4), torch.float32, residual.device, "out_sum")
        if any(
            _overlap(out_sum, tensor)
            for tensor in (
                residual,
                self.packed_down,
                self.up,
                self.norm,
                self.up_padded,
            )
        ):
            raise ValueError(
                "bootstrap output must not alias residual or prepared weights"
            )
        if rows:
            from .jit import module

            module(self.hidden_size, residual.dtype).bootstrap(residual, out_sum)
        return out_sum

    def _workspace(self, ws, residual, sums):
        rows = self._residual(residual)
        if not isinstance(ws, HCWorkspace) or ws.plan is not self or ws.rows != rows:
            raise ValueError("workspace belongs to a different plan or token count")
        dtype, device, h = self.up.dtype, self.up.device, self.hidden_size
        specs = [
            (ws.mixed, (rows, h), dtype),
            (ws.alpha, (rows, 4), torch.float32),
            (ws.residual, (rows, 4 * h), dtype),
            (ws.sum_sq, (rows, 4), torch.float32),
            (ws.down, (rows, self.kpad if rows <= 24 else self.lowrank), dtype),
        ]
        if rows > 24:
            specs += [
                (ws.partial, (4, rows, self.packed_down.shape[1]), torch.float32),
                (ws.logits, (rows, 4 * h), dtype),
            ]
        if ws.deterministic:
            specs.append((ws.partial_sums, (rows, 4, 10), torch.float32))
        inputs = [residual, self.packed_down, self.up, self.norm, self.up_padded]
        _tensor(
            sums,
            (rows, 4),
            torch.float32,
            device,
            "sum_sq",
            32 if rows <= 24 else 16,
        )
        inputs.append(sums)
        for i, (tensor, shape, dt) in enumerate(specs):
            _tensor(tensor, shape, dt, device, "workspace", 32)
            if any(
                _overlap(tensor, other) for other in inputs + [s[0] for s in specs[:i]]
            ):
                raise ValueError(
                    "workspace must not alias inputs, weights or other outputs"
                )
        return rows

    @flashinfer_api
    def mix(self, residual, sum_sq, workspace):
        """Produce mixed core input and final alpha; reset NEXT sums in-kernel.

        Current sums and workspace.sum_sq must be disjoint. This call must
        precede every atomic apply, including each graph replay.

        Parameters
        ----------
        residual : torch.Tensor
            Contiguous current residual ``[T,4*H]`` matching the prepared weights.
        sum_sq : torch.Tensor
            Read-only FP32 sums ``[T,4]``;32-byte aligned for T<=24,
            otherwise16-byte aligned.
        workspace : HCWorkspace
            This plan's workspace for T, distinct from still-live invocations.

        Returns
        -------
        tuple of torch.Tensor
            Mixed input ``[T,H]`` and final FP32 alpha ``[T,4]``.
        """
        rows = self._workspace(workspace, residual, sum_sq)
        if rows:
            with torch.cuda.device(residual.device):
                if rows <= 24:
                    from .lowm import (
                        SplitKTactic,
                        run_splitk_dense_gate,
                        run_splitk_dense_silu_aux,
                    )

                    run_splitk_dense_silu_aux(
                        residual,
                        self.packed_down.transpose(1, 2),
                        workspace.down[:, : self.lowrank + 4],
                        workspace.alpha,
                        True,
                        SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=3),
                        0.25,
                        self.lowrank,
                        sum_sq=sum_sq,
                        rms_eps=self.rms_eps,
                        branch_packed=True,
                    )
                    run_splitk_dense_gate(
                        workspace.down,
                        self.up_padded.T,
                        residual,
                        workspace.mixed,
                        True,
                        SplitKTactic(
                            mma_m=64,
                            mma_n=8 if rows <= 8 else 16 if rows <= 16 else 32,
                            split_k=1,
                            ab_stages=3,
                        ),
                        0.25,
                        4,
                        norm_weight_permuted=self.norm,
                        sum_sq=sum_sq,
                        next_sum_sq=workspace.sum_sq,
                        rms_eps=self.rms_eps,
                    )
                else:
                    from .large import run_large

                    run_large(residual, sum_sq, self, workspace)
        return workspace.mixed, workspace.alpha

    @flashinfer_api
    def apply(self, block_output, residual, workspace):
        """Apply final alpha and produce residual plus pre-cast FP32 sums.

        Statistics use FP32 fma(alpha, block_output, residual) BEFORE BF16/FP16
        storage rounding. Atomic mode adds to sums reset by mix. Deterministic
        mode overwrites 10 partials/branch then uses a fixed reduction tree.
        Determinism covers repeatability at fixed T, not cross-batch bit identity.

        Parameters
        ----------
        block_output : torch.Tensor
            Caller-computed core output ``[T,H]`` in the residual dtype.
        residual : torch.Tensor
            Current ``[T,4*H]`` residual supplied to the corresponding mix.
        workspace : HCWorkspace
            Live workspace from mix, carrying alpha and next-sum destinations.

        Returns
        -------
        tuple of torch.Tensor
            Next residual ``[T,4*H]`` and its pre-storage-cast FP32 sums ``[T,4]``.
        """
        rows = self._residual(residual)
        if (
            not isinstance(workspace, HCWorkspace)
            or workspace.plan is not self
            or workspace.rows != rows
        ):
            raise ValueError("workspace belongs to a different plan or token count")
        if workspace.deterministic != (workspace.partial_sums is not None):
            raise ValueError("partial statistics must match the workspace mode")
        destinations = [workspace.residual, workspace.sum_sq]
        if workspace.partial_sums is not None:
            destinations.append(workspace.partial_sums)
        for dest in destinations:
            if any(
                _overlap(dest, weight)
                for weight in (self.packed_down, self.up, self.norm, self.up_padded)
            ):
                raise ValueError("Apply destinations must not alias prepared weights")
        return apply_sum(
            block_output,
            residual,
            workspace.alpha,
            out=workspace.residual,
            sum_sq=workspace.sum_sq,
            partial_sums=workspace.partial_sums,
        )


@torch.no_grad()
def bind(packed_down, up, norm, *, lowrank, rms_eps):
    if not isinstance(packed_down, torch.Tensor) or packed_down.ndim != 3:
        raise ValueError("packed_down must have shape [4,Npad,H]")
    h = packed_down.shape[2]
    _contract(h, lowrank, rms_eps)
    _device(packed_down)
    dtype, device = packed_down.dtype, packed_down.device
    npad = (lowrank + 11) // 8 * 8
    for tensor, shape, name in (
        (packed_down, (4, npad, h), "packed_down"),
        (up, (4 * h, lowrank), "up"),
        (norm, (h, 4), "norm"),
    ):
        _tensor(tensor.detach(), shape, dtype, device, name, 32)
    return HCSumPlan(packed_down.detach(), up, norm, lowrank, rms_eps)


@torch.no_grad()
def prepare(down, up, inject, norm, *, rms_eps, device=None):
    if not isinstance(norm, torch.Tensor) or norm.ndim != 2 or norm.shape[0] != 4:
        raise ValueError("norm_weight must have shape [4,H]")
    if not isinstance(down, torch.Tensor) or down.ndim != 2:
        raise ValueError("down_weight must have shape [L,4*H]")
    h, lowrank = norm.shape[1], down.shape[0]
    _contract(h, lowrank, rms_eps)
    source_device, dtype = down.device, down.dtype
    if source_device.type not in ("cpu", "cuda"):
        raise ValueError("checkpoint weights must be on CPU or CUDA")
    if source_device.type == "cpu" and device is None:
        raise ValueError("CPU checkpoint preparation requires an explicit CUDA device")
    target = torch.device(device) if device is not None else source_device
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    _target_device(target, dtype)
    if source_device.type == "cuda" and source_device != target:
        raise ValueError("CUDA checkpoint weights must already be on the target device")
    for tensor, shape, name in (
        (down, (lowrank, 4 * h), "down"),
        (up, (4 * h, lowrank), "up"),
        (inject, (4, 4 * h), "inject"),
        (norm, (4, h), "norm"),
    ):
        expected_device = source_device
        if name == "up":
            if tensor.device not in (source_device, target):
                raise ValueError(
                    "Up weights must be with the checkpoint or on the target"
                )
            expected_device = tensor.device
        _tensor(
            tensor.detach(),
            shape,
            dtype,
            expected_device,
            name,
            1 if expected_device.type == "cpu" else 32 if name == "up" else 16,
        )
    from .weights import _pack_native

    packed, interleaved_norm = _pack_native(down, inject, norm)
    # CPU inputs are folded/packed on CPU. Only final native storage is uploaded;
    # no checkpoint-format Down/Inject tensor is ever staged on the GPU here.
    return bind(
        packed.to(target),
        up.to(target),
        interleaved_norm.to(target),
        lowrank=lowrank,
        rms_eps=rms_eps,
    )
