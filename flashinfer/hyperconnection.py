# SPDX-License-Identifier: Apache-2.0
# Adapted from the HC optimization branch; see experimental/hc_sum_sm100/SOURCES.md.
"""Experimental low-rank HyperConnection with carried FP32 square sums.

This API is separate from DeepSeek mHC and the SM12x normalized-state backend.
The returned plan owns prepared weights; each concurrent execution owns a
workspace. See ``flashinfer/experimental/hc_sum_sm100/README.md`` for the ABI.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Union

import torch

from .api_logging import flashinfer_experimental_api

__all__ = [
    "HCApplyContext",
    "HCSumRuntime",
    "HCSumState",
    "prepare_hc_sum_weights",
    "bind_hc_sum_weights",
    "hc_apply_sum",
    "hc_sum_support_reason",
    "hc_sum_weight_shapes",
    "pack_hc_sum_weights_into",
]


def _same_tensor(left, right):
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.data_ptr() == right.data_ptr()
    )


@dataclass(frozen=True, eq=False)
class HCSumState:
    """Residual plus its pre-storage-cast FP32 square sums.

    Tensor identity checks detect replacement and row reordering by a new
    tensor. They cannot detect communication or other in-place mutations:
    callers must discard this state whenever residual contents/rows change.
    """

    is_experimental: ClassVar[bool] = True

    residual: torch.Tensor
    sum_sq: torch.Tensor

    def matches(self, residual):
        return _same_tensor(self.residual, residual)


@dataclass(frozen=True, eq=False)
class HCApplyContext:
    """One Mix's Apply inputs, destinations and retained scratch ownership.

    Apply must run exactly once after Mix, in stream order. Row partitions may
    each be applied once instead. All output/sum buffers belong to this
    invocation, so later invocations cannot overwrite a carried output.
    """

    is_experimental: ClassVar[bool] = True

    residual: torch.Tensor
    alpha: torch.Tensor
    out: torch.Tensor
    sum_sq: torch.Tensor
    partial_sums: torch.Tensor | None
    _workspace: Any
    # Bootstrap/realignment may create a temporary hidden from the caller.
    # Keep it through the core/Apply handoff, including PDL launches.
    _mix_input_sum: torch.Tensor | None = None

    def with_residual(self, residual):
        if not _same_tensor(self.residual, residual):
            raise ValueError("HC Apply residual differs from its Mix/row partition")
        return self

    def slice_rows(self, size, rank):
        """Contiguous torch.tensor_split partition: the first T % size ranks
        receive one extra row. Pass the same residual partition to Apply when
        a framework partitions the core output independently.
        """
        if (
            type(size) is not int
            or size <= 0
            or type(rank) is not int
            or not 0 <= rank < size
        ):
            raise ValueError("row partition requires size>0 and 0<=rank<size")

        def part(tensor):
            return None if tensor is None else tensor.tensor_split(size, dim=0)[rank]

        return HCApplyContext(
            part(self.residual),
            part(self.alpha),
            part(self.out),
            part(self.sum_sq),
            part(self.partial_sums),
            self._workspace,
            self._mix_input_sum,
        )

    def apply(self, block_output, residual=None):
        if residual is not None:
            self.with_residual(residual)
        out, sums = hc_apply_sum(
            block_output,
            self.residual,
            self.alpha,
            out=self.out,
            sum_sq=self.sum_sq,
            partial_sums=self.partial_sums,
        )
        return HCSumState(out, sums)


class HCSumRuntime:
    """Experimental allocator-backed HC execution for one ordered stream.

    Construct before capture and keep the runtime (and its weights) alive for
    every graph that references it. Concurrent execution lanes need separate
    runtimes or explicit plan workspaces. Outputs belong to each invocation.
    """

    is_experimental = True

    @flashinfer_experimental_api(feature="HCSumRuntime")
    def __init__(self, plan: Any, *, deterministic: bool = False):
        from .experimental.hc_sum_sm100.runtime import HCSumRuntime as BackendRuntime

        self._backend = BackendRuntime(plan, deterministic=deterministic)

    @property
    def plan(self):
        return self._backend.plan

    @property
    def deterministic(self):
        return self._backend.deterministic

    def mix(self, residual: torch.Tensor, prepared: Optional[HCSumState] = None):
        """Return (mixed, context), reusing matching pre-storage-cast sums.

        A missing or non-matching state bootstraps sums from the stored
        residual; callers must discard state after any in-place content/row
        changes, which identity checks cannot detect. Apply instead rejects a
        mismatched residual because its alpha already belongs to specific rows.
        """
        return self._backend.mix(residual, prepared)


@flashinfer_experimental_api
def hc_sum_support_reason(*, hidden_size, lowrank, dtype, device, rms_eps=1e-6):
    """Return None for supported SM100/103 weights, otherwise a reason.

    Checks hardware, weight ABI and packaged CUDA sources. This does not
    allocate, compile, or load JIT modules. A CPU/meta device is
    unsupported for execution; native storage shapes can be queried separately.
    """
    from .experimental.hc_sum_sm100.weights import support_reason

    return support_reason(
        hidden_size=hidden_size,
        lowrank=lowrank,
        dtype=dtype,
        device=device,
        rms_eps=rms_eps,
    )


@flashinfer_experimental_api
def hc_sum_weight_shapes(hidden_size, lowrank):
    """Return native (Down+Inject, Norm) shapes, independent of device."""
    from .experimental.hc_sum_sm100.weights import weight_shapes

    return weight_shapes(hidden_size, lowrank)


@flashinfer_experimental_api
def pack_hc_sum_weights_into(down, inject, norm, *, out_down, out_norm):
    """Pack CPU checkpoint Down[L,4H], Inject[4,4H], Norm[4,H] in place.

    Folding uses Gemma gamma=1+Norm in FP32, then rounds once to BF16/FP16.
    Only native weights are uploaded into caller-owned CUDA output storage;
    CPU outputs are also supported for serialization. No output storage is
    replaced and the function retains no checkpoint tensors. Do not call
    while a plan or captured graph bound to these weights is still in use.
    """
    from .experimental.hc_sum_sm100.weights import pack_into

    return pack_into(down, inject, norm, out_down=out_down, out_norm=out_norm)


@flashinfer_experimental_api
def prepare_hc_sum_weights(
    down_weight: torch.Tensor,
    up_weight: torch.Tensor,
    inject_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Pack checkpoint weights for SM100/SM103 HC inference.

    Shapes are Down ``[L,4*H]``, Up ``[4*H,L]``, Inject ``[4,4*H]``
    and Gemma RMS weight ``[4,H]`` (gamma is ``1 + norm_weight``).
    Supported: BF16/FP16, H a multiple of 512 up to 16384, L a positive
    multiple of 64 with ``L+4 <= ceil(L/128)*128``. Preparation allocates and
    must happen outside CUDA Graph capture. The plan does not retain raw
    Down/Inject/RMS tensors; callers may release those after preparation.
    An Up tensor already on the target GPU shares storage; CPU Up is uploaded.
    Any prepared-weight content change, including an in-place
    update, requires a new plan and graph capture before further execution.
    For CPU checkpoint inputs, supply a CUDA ``device``: folding/packing stays
    on CPU and only final native weights are uploaded. Up may already be on the
    target GPU. CUDA inputs remain supported for callers that already own them;
    use CPU preparation or native binding to avoid raw Down/Inject GPU staging.

    The plan provides ``allocate_workspace``, ``bootstrap``, ``mix`` and
    ``apply``. Inference only; no autograd or cross-batch bitwise guarantee.
    """
    from .experimental.hc_sum_sm100.api import prepare

    return prepare(
        down_weight,
        up_weight,
        inject_weight,
        norm_weight,
        rms_eps=rms_eps,
        device=device,
    )


@flashinfer_experimental_api
def bind_hc_sum_weights(
    packed_down: torch.Tensor,
    up_weight: torch.Tensor,
    packed_norm: torch.Tensor,
    *,
    lowrank: int,
    rms_eps: float = 1e-6,
) -> Any:
    """Bind already folded native HC weights without duplicating Down/Inject.

    ``packed_down`` is contiguous ``[4,round_up(L+4,8),H]``;
    ``packed_norm`` is contiguous ``[H,4]`` with the original Gemma RMS
    weights, not gamma. The rows after ``L+4`` must be zero. No second fold
    is performed. Up remains ``[4*H,L]``; a derived small-T Up layout is
    prepared. Supplied storage remains shared and must stay valid. Replacing,
    moving or modifying weights in place requires a new plan and graph capture.
    """
    from .experimental.hc_sum_sm100.api import bind

    return bind(packed_down, up_weight, packed_norm, lowrank=lowrank, rms_eps=rms_eps)


@flashinfer_experimental_api
def hc_apply_sum(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out: torch.Tensor,
    sum_sq: torch.Tensor,
    partial_sums: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply final FP32 alpha and write the next residual and square sums.

    This tensor-only entry supports row-partitioned framework handoffs without
    retaining a whole mix workspace. Inputs are ``[T,H]``, ``[T,4*H]`` and
    ``[T,4]``. Outputs are disjoint ``[T,4*H]`` and FP32 ``[T,4]``. Optional
    FP32 ``partial_sums[T,4,10]`` selects deterministic fixed-order reduction.
    Atomic mode requires sum_sq to have been cleared by the preceding mix or
    by the caller on every invocation/replay. No allocation or reset occurs.
    Destinations must also be disjoint from prepared weights and other live
    state owned by the caller; this tensor-only entry cannot inspect those.
    """
    from .experimental.hc_sum_sm100.api import apply_sum

    return apply_sum(
        block_output, residual, alpha, out=out, sum_sq=sum_sq, partial_sums=partial_sums
    )
