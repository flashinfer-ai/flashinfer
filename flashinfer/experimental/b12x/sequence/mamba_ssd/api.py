"""Allocation-free prepared Mamba-2 selective scan."""

import torch

from b12x.preparation import plan_from_handle, require_prepared
from ..._lib.gating import default_is_supported
from . import META
from ._tuning import MambaQuery, MambaConfig
from .._shared.tensors import overlaps as _overlap


def _validate_aliases(tensors):
    state, out = tensors[7], tensors[-1]
    if _overlap(out, state) or any(
        _overlap(t, out) or _overlap(t, state) for t in tensors[:7] + tensors[8:14]
    ):
        raise ValueError(
            "Mamba writable state/output must not alias inputs or each other"
        )


def _validate(
    x, dt, A, B, C, D, bias, state, cu, src, dst, fresh, num_seqs, num_tokens, out
):
    tensors = (
        x,
        dt,
        A,
        B,
        C,
        D,
        bias,
        state,
        cu,
        src,
        dst,
        fresh,
        num_seqs,
        num_tokens,
        out,
    )
    if not x.is_cuda or any(t.device != x.device for t in tensors):
        raise ValueError("Mamba operands must share a CUDA device")
    if x.ndim != 3 or B.ndim != 3 or C.shape != B.shape or state.ndim != 4:
        raise ValueError(
            "Mamba requires token/head/channel inputs and slot/head/channel/state storage"
        )
    tokens, heads, dim = x.shape
    if B.shape[0] != tokens or state.shape[1:] != (heads, dim, B.shape[2]):
        raise ValueError("Mamba token and state geometry do not match")
    if (
        dt.shape != (tokens, heads)
        or A.shape != (heads,)
        or D.shape != (heads,)
        or bias.shape != (heads,)
    ):
        raise ValueError("Mamba dt is [tokens, heads]; A, D, and bias are [heads]")
    if (
        x.dtype != torch.bfloat16
        or B.dtype != x.dtype
        or C.dtype != x.dtype
        or A.dtype != torch.float32
    ):
        raise TypeError("Mamba requires BF16 x/B/C and FP32 A")
    if out.shape != x.shape or out.dtype != x.dtype or not out.is_contiguous():
        raise ValueError("Mamba output must be contiguous BF16 with the input shape")
    for t in (x, B, C):
        if (
            t.stride(-1) != 1
            or t.stride(1) != t.shape[2]
            or t.stride(0) < t.shape[1] * t.shape[2]
        ):
            raise ValueError(
                "Mamba input heads and channels must be contiguous and rows nonoverlapping"
            )
    if state.stride()[1:] != (dim * B.shape[2], B.shape[2], 1) or dt.stride(1) != 1:
        raise ValueError("Mamba requires contiguous inner state and dt dimensions")
    if cu.ndim != 1 or cu.numel() < 2 or cu.dtype != torch.int32:
        raise ValueError(
            "Mamba boundaries must be an Int32 vector with at least two entries"
        )
    seqs = cu.numel() - 1
    if (
        src.shape != (seqs,)
        or dst.shape != src.shape
        or src.dtype not in (torch.int32, torch.int64)
        or dst.dtype != src.dtype
    ):
        raise ValueError(
            "Mamba state indices must be matched Int32/Int64 request vectors"
        )
    if fresh.shape != (seqs,) or fresh.dtype != torch.bool:
        raise ValueError("Mamba fresh flags must be a boolean request vector")
    if any(t.numel() != 1 or t.dtype != torch.int32 for t in (num_seqs, num_tokens)):
        raise ValueError("Mamba live counts must be Int32 scalars")
    if any(
        not t.is_contiguous()
        for t in (A, D, bias, cu, src, dst, fresh, num_seqs, num_tokens)
    ):
        raise ValueError("Mamba parameters and metadata must be contiguous")
    return tensors


def query_from_call(
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    state,
    cu_seqlens,
    initial_state_indices,
    final_state_indices,
    state_is_fresh,
    num_seqs,
    num_tokens,
    *,
    out,
    dt_softplus=True,
):
    """Describe fixed capacities and layout without reading device metadata."""
    _validate(
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        state,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        state_is_fresh,
        num_seqs,
        num_tokens,
        out,
    )
    return MambaQuery(
        heads=x.shape[1],
        groups=B.shape[1],
        head_dim=x.shape[2],
        dstate=B.shape[2],
        max_tokens=x.shape[0],
        max_seqs=cu_seqlens.numel() - 1,
        max_slots=state.shape[0],
        x_stride=x.stride(0),
        b_stride=B.stride(0),
        c_stride=C.stride(0),
        dt_stride=dt.stride(0),
        state_stride=state.stride(0),
        state_dtype=str(state.dtype).removeprefix("torch."),
        dt_dtype=str(dt.dtype).removeprefix("torch."),
        d_dtype=str(D.dtype).removeprefix("torch."),
        bias_dtype=str(dt_bias.dtype).removeprefix("torch."),
        index_dtype=str(initial_state_indices.dtype).removeprefix("torch."),
        dt_softplus=dt_softplus,
    )


@torch.library.custom_op(
    "b12x::mamba_ssd_out", mutates_args=("state", "out", "workspace")
)
def _run(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state_indices: torch.Tensor,
    final_state_indices: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor | None,
    handle: int,
) -> None:
    require_prepared(plan_from_handle(handle), "sequence.mamba_ssd", x.device).run(
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        state,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        state_is_fresh,
        num_seqs,
        num_tokens,
        out=out,
        workspace=workspace,
    )


@_run.register_fake
def _run_fake(
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    state,
    cu_seqlens,
    initial_state_indices,
    final_state_indices,
    state_is_fresh,
    num_seqs,
    num_tokens,
    out,
    workspace,
    handle,
):
    return None


def run(
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    state,
    cu_seqlens,
    initial_state_indices,
    final_state_indices,
    state_is_fresh,
    num_seqs,
    num_tokens,
    *,
    out,
    plan,
    workspace=None,
):
    """Scan packed live requests and write final states to distinct destination slots.

    The caller supplies monotone boundaries within the live token count and
    unique destination slots that do not overlap another request's source.
    Fresh requests ignore their source bytes. Empty requests and negative
    source/destination slots perform no writes. Unused output rows are preserved.
    State and parameters remain alive until all captured replays have drained.
    """
    _run(
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        state,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        state_is_fresh,
        num_seqs,
        num_tokens,
        out,
        workspace,
        plan.handle,
    )
    return out


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


def clear_caches():
    from ._kernel import compile_scan
    from ._chunked import compile_chunked
    from ._preparation import compile_pipeline

    compile_scan.cache_clear()
    compile_chunked.cache_clear()
    compile_pipeline.cache_clear()


from ._preparation import plan
