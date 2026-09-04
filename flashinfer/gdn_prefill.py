"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import warnings
import weakref
from typing import Callable, Literal, Optional, Tuple, Union, cast
import torch

from .api_logging import flashinfer_api
from .trace.templates.gdn import gdn_prefill_trace

try:
    from .jit import gdn_noncp as _gdn_noncp

    _GDN_NONCP_AVAILABLE = True
except (ImportError, RuntimeError):
    _gdn_noncp = None
    _GDN_NONCP_AVAILABLE = False
from .utils import get_compute_capability, get_device_name, get_device_sm_count
from .gdn_kernels import (
    chunk_gated_delta_rule_sm90,
    chunk_gated_delta_rule_sm100,
    chunk_gated_delta_rule_sm120,
    cp_delta_rule_dsl_sm90,
    cp_delta_rule_dsl_sm100,
    cp_delta_rule_dsl_sm120,
)
from .gdn_kernels.delta_rule_dsl.varlen_helper import (
    is_integer_dtype,
    should_use_cp_host,
)


_STATE_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


_GDN_NONCP_PREFILL_SEQ_LENS: dict[
    tuple[int, int, int, int], tuple[weakref.ReferenceType[torch.Tensor], tuple[int, ...]]
] = {}


def _gdn_noncp_prefill_seq_lens(
    cu_seqlens: torch.Tensor, total_seq_len: int
) -> tuple[int, ...]:
    """Resolve immutable launch metadata once, outside CUDA Graph capture."""

    key = (
        int(cu_seqlens.device.index or 0),
        int(cu_seqlens.data_ptr()),
        int(cu_seqlens._version),
        int(cu_seqlens.numel()),
    )
    cached = _GDN_NONCP_PREFILL_SEQ_LENS.get(key)
    if cached is not None and cached[0]() is cu_seqlens:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires one eager metadata resolution before CUDA Graph capture"
        )
    offsets = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != total_seq_len
        or any(
            end < start
            for start, end in zip(offsets, offsets[1:], strict=False)
        )
    ):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires monotonic cu_seqlens spanning all input tokens"
        )
    seq_lens = tuple(
        end - start for start, end in zip(offsets, offsets[1:], strict=False)
    )
    _GDN_NONCP_PREFILL_SEQ_LENS[key] = (weakref.ref(cu_seqlens), seq_lens)
    return seq_lens


def _gdn_noncp_assert_state_slots(
    indices: torch.Tensor, pool_size: int, *, name: str, allow_minus_one: bool
) -> None:
    """Validate CUDA-resident state slots without a host synchronization."""

    in_pool = (indices >= 0) & (indices < pool_size)
    valid = ((indices == -1) | in_pool) if allow_minus_one else in_pool
    torch._assert_async(
        valid.all(),
        f"{name} must contain "
        + ("-1 or " if allow_minus_one else "")
        + f"slots in [0, {pool_size})",
    )


def _gdn_noncp_prefill_dtype_name(dtype: torch.dtype) -> str:
    names = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.float8_e5m2: "float8_e5m2",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            f"unsupported GDN non-CP prefill dtype {dtype}"
        ) from exc


def _run_gdn_noncp_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    output: torch.Tensor,
    output_state: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_indices: Optional[torch.Tensor],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Launch one exact manifest-backed GDN non-CP non-CP prefill row."""

    if not _GDN_NONCP_AVAILABLE or _gdn_noncp is None:
        raise RuntimeError("the source-only GDN non-CP GDN backend is not installed")
    if use_qk_l2norm_in_kernel:
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP non-CP prefill requires caller-normalized Q/K"
        )
    tensors = tuple(
        tensor
        for tensor in (
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            output,
            output_state,
            state_checkpoints,
            checkpoint_cu_starts,
            state_indices,
        )
        if tensor is not None
    )
    if q.device.type != "cuda" or any(tensor.device != q.device for tensor in tensors):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires all tensors on one CUDA device"
        )
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != q.dtype for tensor in (k, v, output)
    ):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires matching FP16 or BF16 Q/K/V/output"
        )
    if any(not tensor.is_contiguous() for tensor in (q, k, v, output)):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires contiguous Q/K/V/output"
        )
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or output.ndim != 3:
        raise _gdn_noncp.GDNNonCPUnsupportedError("GDN non-CP prefill tensors must be rank 3")
    total_seq_len, num_q_heads, head_size = map(int, q.shape)
    num_k_heads, num_v_heads = int(k.shape[1]), int(v.shape[1])
    num_o_heads = max(num_q_heads, num_v_heads)
    if (
        total_seq_len <= 0
        or head_size != 128
        or tuple(k.shape) != (total_seq_len, num_k_heads, 128)
        or tuple(v.shape) != (total_seq_len, num_v_heads, 128)
        or tuple(output.shape) != (total_seq_len, num_o_heads, 128)
    ):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires exact [tokens,heads,128] tensors"
        )
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or not cu_seqlens.is_contiguous()
    ):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires contiguous int32/int64 cu_seqlens"
        )
    seq_lens = _gdn_noncp_prefill_seq_lens(cu_seqlens, total_seq_len)
    num_seqs = len(seq_lens)
    gates_present = g is not None and beta is not None

    gate = g
    if gate is None:
        gate = torch.ones(
            (total_seq_len, num_o_heads), dtype=torch.float32, device=q.device
        )
    update_gate = beta
    if update_gate is None:
        update_gate = torch.ones_like(gate)
    for name, tensor in (("g", gate), ("beta", update_gate)):
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != (total_seq_len, num_o_heads)
            or not tensor.is_contiguous()
        ):
            raise _gdn_noncp.GDNNonCPUnsupportedError(
                f"GDN non-CP prefill requires contiguous FP32 {name} [tokens,heads]"
            )

    if output_final_state and output_state is None:
        output_state = torch.empty(
            (num_seqs, num_o_heads, 128, 128),
            dtype=torch.float32,
            device=q.device,
        )
    active_states = [
        tensor
        for tensor in (
            initial_state,
            output_state if output_final_state else None,
            state_checkpoints if checkpoint_every_n_tokens else None,
        )
        if tensor is not None
    ]
    state_dtype = active_states[0].dtype if active_states else torch.float32
    if any(tensor.dtype != state_dtype for tensor in active_states):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires one state dtype across initial/final/checkpoints"
        )
    state_dtype_name = _gdn_noncp_prefill_dtype_name(state_dtype)
    for name, tensor in (
        ("initial_state", initial_state),
        ("output_state", output_state if output_final_state else None),
    ):
        if tensor is None:
            continue
        if (
            tensor.ndim != 4
            or tuple(tensor.shape[1:]) != (num_o_heads, 128, 128)
            or tuple(tensor.stride()[1:]) != (16384, 128, 1)
        ):
            raise _gdn_noncp.GDNNonCPUnsupportedError(
                f"GDN non-CP prefill requires {name} with contiguous [H,V,K] rows"
            )
    if state_indices is not None and (
        state_indices.ndim != 1
        or int(state_indices.numel()) != num_seqs
        or state_indices.dtype not in (torch.int32, torch.int64)
        or not state_indices.is_contiguous()
    ):
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "GDN non-CP prefill requires one contiguous integer state index per sequence"
        )
    if state_indices is not None:
        indexed_pools = tuple(
            tensor
            for tensor in (
                initial_state,
                output_state if output_final_state else None,
            )
            if tensor is not None
        )
        if indexed_pools:
            _gdn_noncp_assert_state_slots(
                state_indices,
                min(int(tensor.shape[0]) for tensor in indexed_pools),
                name="GDN non-CP prefill state_indices",
                allow_minus_one=False,
            )
    if checkpoint_every_n_tokens:
        if (
            state_checkpoints is None
            or checkpoint_cu_starts is None
            or not state_checkpoints.is_contiguous()
            or checkpoint_cu_starts.dtype not in (torch.int32, torch.int64)
            or not checkpoint_cu_starts.is_contiguous()
        ):
            raise _gdn_noncp.GDNNonCPUnsupportedError(
                "GDN non-CP checkpoint prefill requires contiguous state/cumulative buffers"
            )

    major, minor = torch.cuda.get_device_capability(q.device)
    arch = _gdn_noncp.arch_for_compute_capability(major, minor)
    route = _gdn_noncp.select_gdn_noncp_prefill_variant(
        arch=arch,
        io_dtype=_gdn_noncp_prefill_dtype_name(q.dtype),
        state_dtype=state_dtype_name,
        num_seqs=num_seqs,
        total_seq_len=total_seq_len,
        max_seq_len=max(seq_lens),
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_state_indices=state_indices is not None,
        gates_present=gates_present,
        seq_lens=seq_lens,
    )
    entry = _gdn_noncp.load_gdn_noncp_kernel(route.variant_name, arch)
    active_clusters = int(torch.cuda.get_device_properties(q.device).multi_processor_count)
    dvsplit = route.route_id.endswith(".dvsplit")
    total_tiles = num_seqs * num_o_heads * (2 if dvsplit else 1)
    if dvsplit or total_tiles <= 128:
        grid_x = min(active_clusters, total_tiles)
    else:
        max_chunks = max((length + 63) // 64 for length in seq_lens)
        if max_chunks <= 8:
            grid_x = min(128, total_tiles)
        elif active_clusters in (148, 160) and total_tiles == 256:
            grid_x = 128
        else:
            grid_x = min(active_clusters, total_tiles)

    empty_i32 = torch.empty(1, dtype=torch.int32, device=q.device)
    cu_seqlens_i32 = (
        cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    )
    state_indices_i32 = (
        empty_i32
        if state_indices is None
        else state_indices
        if state_indices.dtype == torch.int32
        else state_indices.to(torch.int32)
    )
    empty_state = torch.empty(1, dtype=state_dtype, device=q.device)
    launch_initial_state = initial_state if initial_state is not None else empty_state
    launch_output_state = (
        output_state if output_final_state and output_state is not None else empty_state
    )
    launch_checkpoints = (
        state_checkpoints
        if checkpoint_every_n_tokens and state_checkpoints is not None
        else empty_state
    )
    cu_checkpoints_i32 = (
        empty_i32
        if checkpoint_every_n_tokens == 0 or checkpoint_cu_starts is None
        else checkpoint_cu_starts
        if checkpoint_cu_starts.dtype == torch.int32
        else checkpoint_cu_starts.to(torch.int32)
    )
    tensormap_workspace = torch.empty(grid_x * 512, dtype=torch.uint8, device=q.device)
    entry(
        q,
        k,
        v,
        output,
        gate,
        update_gate,
        cu_seqlens_i32,
        state_indices_i32,
        launch_initial_state,
        launch_output_state,
        launch_checkpoints,
        cu_checkpoints_i32,
        tensormap_workspace,
        int(initial_state.stride(0))
        if initial_state is not None
        else num_o_heads * 16384,
        int(output_state.stride(0))
        if output_final_state and output_state is not None
        else num_o_heads * 16384,
        checkpoint_every_n_tokens,
        scale,
        num_seqs,
        num_q_heads,
        num_v_heads,
        total_tiles,
        grid_x,
        1,
        1,
    )
    if output_final_state:
        assert output_state is not None
        return output, output_state
    return output


def _format_dtype_list(dtypes: tuple[torch.dtype, ...]) -> str:
    return ", ".join(str(dtype).removeprefix("torch.") for dtype in dtypes)


def _cp_delta_rule_rejection_reason(
    *,
    arch_major: int,
    cuda_major: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    output: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
) -> Optional[str]:
    if arch_major == 9:
        if cp_delta_rule_dsl_sm90 is None:
            return "CP delta rule SM90 DSL kernel is unavailable"
    elif arch_major == 10:
        if cuda_major < 13:
            return "CP delta rule SM100 requires CUDA 13 or newer"
        if cp_delta_rule_dsl_sm100 is None:
            return "CP delta rule SM100 DSL kernel is unavailable"
    elif arch_major == 12:
        if cp_delta_rule_dsl_sm120 is None:
            return "CP delta rule SM120 DSL kernel is unavailable"
    else:
        return "CP delta rule is currently implemented only for SM90, SM100, and SM120"
    if (
        checkpoint_every_n_tokens > 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ) and arch_major not in (9, 10, 12):
        return "CP delta rule does not support state checkpointing yet"
    if q.shape[-1] != 128:
        return f"CP delta rule only supports head_size=128, got {q.shape[-1]}"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"CP delta rule only supports fp16/bf16 inputs, got {q.dtype}"
    if k.dtype != q.dtype or v.dtype != q.dtype or output.dtype != q.dtype:
        return "CP delta rule requires q/k/v/output dtypes to match"
    for name, tensor in (("g", g), ("beta", beta)):
        if tensor is not None:
            if tensor.dtype != torch.float32:
                return f"CP delta rule requires {name} to be float32"
            if not tensor.is_contiguous():
                return f"CP delta rule requires {name} to be contiguous"
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("output", output),
    ):
        if tensor is None:
            continue
        if not tensor.is_contiguous():
            return f"CP delta rule requires {name} to be contiguous"
    if initial_state is not None:
        if state_indices is None and not initial_state.is_contiguous():
            return "CP delta rule requires initial_state to be contiguous"
    return None


@flashinfer_api(trace=gdn_prefill_trace)
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_cp: Literal["auto"] | bool = "auto",
    state_indices: Optional[torch.Tensor] = None,
    _cp_chunk_len: Optional[int] = None,
    backend: Literal["auto", "flashinfer", "gdn_noncp"] = "auto",
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule (GDN) attention for prefill.

    Implements the gated delta rule linear attention mechanism for efficient
    training and inference.  Supports both GQA (grouped query attention)
    and GVA (grouped value attention) configurations.

    Parameters
    ----------
    q : torch.Tensor
        Queries of shape ``[total_seq_len, num_q_heads, head_size]``.  Must
        be contiguous and on CUDA.
    k : torch.Tensor
        Keys of shape ``[total_seq_len, num_k_heads, head_size]``.  Must be
        contiguous and on CUDA.
    v : torch.Tensor
        Values of shape ``[total_seq_len, num_v_heads, head_size]``.  Must
        be contiguous and on CUDA.
    g : torch.Tensor, optional
        Forget gate (alpha) of shape ``[total_seq_len, num_sab_heads]``
        where ``num_sab_heads = max(num_q_heads, num_v_heads)``.  Must be
        float32.  Defaults to all ones when ``None``.
    beta : torch.Tensor, optional
        Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
        Must be float32.  Defaults to all ones when ``None``.
    scale : float, optional
        Scale factor for the attention scores.  Defaults to
        ``1 / sqrt(head_size)`` when ``None``.
    initial_state : torch.Tensor, optional
        Initial KV state. Packed, sequence-ordered shape
        ``[num_seqs, num_sab_heads, head_size, head_size]``.  Must be
        float32, bfloat16, float16, float8_e4m3fn, or float8_e5m2. Starts from zero state
        when ``None``.  When ``state_indices`` is given (SM90/SM100/SM103/SM120),
        this is instead the state **pool** ``[N_pool, num_sab_heads,
        head_size, head_size]`` and sequence ``i`` reads its initial state
        from row ``state_indices[i]``; the pool may be non-compact (padded
        first-dimension stride, inner ``[H, V, K]`` block contiguous).
    output_final_state : bool
        Whether to output the final state.  Default: ``False``.
    cu_seqlens : torch.Tensor
        Cumulative sequence lengths of shape ``[num_seqs + 1]``, integer
        dtype on the same CUDA device as ``q``.  Required for
        variable-length sequences (varlen mode); must not be ``None``
        (asserted at the top of the function body).  Internally cast to
        ``int32`` for the SM100/Blackwell CuTe-DSL kernel and to ``int64``
        for the SM90/Hopper C++ kernel, so the caller can pass either
        dtype.
    use_qk_l2norm_in_kernel : bool
        Whether to use QK L2 normalization in kernel.  Default: ``False``.
    output : torch.Tensor, optional
        Pre-allocated output tensor of shape
        ``[total_seq_len, num_o_heads, head_size]`` where ``num_o_heads =
        max(num_q_heads, num_v_heads)``.  Allocated automatically when
        ``None``.
    output_state : torch.Tensor, optional
        Pre-allocated output state tensor. Packed, sequence-ordered shape
        ``[num_seqs, num_sab_heads, head_size, head_size]``. May be float32,
        bfloat16, float16, float8_e4m3fn, or float8_e5m2. Required when
        ``output_final_state=True``.  When ``state_indices`` is given it is
        instead the output state **pool** ``[N_pool, ...]`` and sequence
        ``i``'s final state is written to row ``state_indices[i]`` (in place
        when ``output_state is initial_state``); it must be provided by the
        caller (auto-allocation is rejected, since a compact ``[num_seqs, ...]``
        buffer would be indexed out of bounds by the pool slot ids).
    state_checkpoints : torch.Tensor, optional
        Pre-allocated checkpoint tensor of shape ``[total_checkpoints,
        num_sab_heads, head_size, head_size]``. May be float32, bfloat16,
        float16, float8_e4m3fn, or float8_e5m2. Required when
        ``checkpoint_every_n_tokens > 0``. Context-parallel checkpointing is
        currently supported on SM90, SM100, and SM120.
    checkpoint_cu_starts : torch.Tensor, optional
        Cumulative checkpoint counts of shape ``[num_seqs + 1]``, int64.
        ``checkpoint_cu_starts[i+1] - checkpoint_cu_starts[i]`` is the
        number of checkpoints for sequence ``i`` (= ``seq_len_i //
        checkpoint_every_n_tokens``).  Required when
        ``checkpoint_every_n_tokens > 0``.
    checkpoint_every_n_tokens : int
        Store intermediate state every N tokens.  Must be a multiple of the
        chunk size (64).  ``0`` disables checkpointing (default).
    use_cp : Literal["auto"] | bool, optional:
        Whether to use the SM90/SM120 context-parallel DSL implementation when
        low-parallelism heuristics match. ``"auto"`` enables conservative
        routing, ``True`` requires CP support, and ``False`` disables CP.
        Default: ``"auto"``.
    state_indices : torch.Tensor, optional
        Int32 tensor of shape ``[num_seqs]`` (SM90/SM100/SM103/SM120). When provided,
        ``initial_state`` and ``output_state`` are treated as a state pool whose
        first dimension is indexed by these slot ids rather than laid out in
        sequence order: sequence ``i`` reads its initial state from row
        ``state_indices[i]`` and writes its final state back to the same row
        (in place when ``output_state is initial_state``). This lets callers
        that keep a paged/indexed state pool avoid gathering the active rows
        into a packed buffer and scattering the result back. The pool may be
        non-compact (padded first-dimension stride). ``None`` (default) keeps
        the packed, sequence-ordered layout.

        The ids **must be unique**: as with any indexed scatter, two sequences
        sharing a slot id would concurrently write the same pool row across
        work tiles, leaving that row's final state nondeterministic. Uniqueness
        is a caller precondition (not checked at launch, to avoid a per-call
        host sync); the caller's slot allocator is expected to guarantee it.


    backend : {"auto", "flashinfer", "gdn_noncp"}
        ``auto`` selects GDN non-CP only for an exact frozen non-CP manifest row;
        explicit ``gdn_noncp`` requests fail closed.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        When ``output_final_state=False``, the output tensor of shape
        ``[total_seq_len, num_o_heads, head_size]``.  Otherwise a tuple
        ``(output, final_state)`` where ``final_state`` has shape
        ``[num_seqs, num_sab_heads, head_size, head_size]`` — or, when
        ``state_indices`` is given, the state pool ``[N_pool, ...]`` itself
        (i.e. ``output_state``), whose rows named by ``state_indices`` now
        hold the updated final states.

    Notes
    -----
    - Supports GQA (``num_q_heads > num_k_heads = num_v_heads``) and GVA
      (``num_v_heads > num_q_heads = num_k_heads``).
    - The final state layout is ``[N, H, V, K]``.
    - Requires SM90 (Hopper) or SM100 (Blackwell) architecture.  The SM100
      path requires ``head_size == 128`` and
      ``nvidia-cutlass-dsl[cu13]>=4.4.2`` (``pip install
      flashinfer-python[cu13]``).
    """
    if backend not in ("auto", "flashinfer", "gdn_noncp"):
        raise ValueError(f"unsupported GDN backend: {backend!r}")
    if backend == "gdn_noncp" and (not _GDN_NONCP_AVAILABLE or _gdn_noncp is None):
        raise RuntimeError(
            "the source-only GDN non-CP GDN backend is not installed"
        )
    if backend == "gdn_noncp" and use_cp is True:
        raise _gdn_noncp.GDNNonCPUnsupportedError(
            "forced context-parallel prefill is outside the GDN non-CP non-CP backend"
        )
    if use_cp not in ("auto", True, False):
        raise ValueError(f'use_cp must be "auto", True, or False, got {use_cp!r}')
    if checkpoint_every_n_tokens < 0:
        raise ValueError(
            f"checkpoint_every_n_tokens must be non-negative, "
            f"got {checkpoint_every_n_tokens}"
        )
    if checkpoint_every_n_tokens > 0:
        if checkpoint_every_n_tokens % 64 != 0:
            raise ValueError(
                f"checkpoint_every_n_tokens must be a multiple of the chunk size (64), "
                f"got {checkpoint_every_n_tokens}"
            )
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts must both be provided "
                "when checkpoint_every_n_tokens > 0"
            )
    if checkpoint_every_n_tokens == 0 and (
        state_checkpoints is not None or checkpoint_cu_starts is not None
    ):
        raise ValueError(
            "state_checkpoints and checkpoint_cu_starts must be None "
            "when checkpoint_every_n_tokens == 0"
        )

    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"
    if not is_integer_dtype(cu_seqlens.dtype):
        raise ValueError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    if checkpoint_every_n_tokens > 0:
        assert state_checkpoints is not None and checkpoint_cu_starts is not None
        if state_checkpoints.dtype not in _STATE_DTYPES:
            raise ValueError(
                "state_checkpoints must have dtype "
                f"{_format_dtype_list(_STATE_DTYPES)}, "
                f"got {state_checkpoints.dtype}"
            )
        if state_checkpoints.ndim != 4:
            raise ValueError(
                f"state_checkpoints must be 4D "
                f"[total_checkpoints, num_sab_heads, head_size, head_size], "
                f"got {state_checkpoints.ndim}D"
            )
        if not is_integer_dtype(checkpoint_cu_starts.dtype):
            raise ValueError(
                "checkpoint_cu_starts must have an integer dtype, "
                f"got {checkpoint_cu_starts.dtype}"
            )
        if checkpoint_cu_starts.ndim != 1:
            raise ValueError(
                f"checkpoint_cu_starts must be 1D [num_seqs + 1], "
                f"got {checkpoint_cu_starts.ndim}D"
            )
        if checkpoint_cu_starts.size(0) != num_seqs + 1:
            raise ValueError(
                f"checkpoint_cu_starts must have {num_seqs + 1} elements, "
                f"got {checkpoint_cu_starts.size(0)}"
            )
        expected_shape = (
            state_checkpoints.size(0),
            num_sab_heads,
            head_size,
            head_size,
        )
        if tuple(state_checkpoints.shape[1:]) != expected_shape[1:]:
            raise ValueError(
                f"state_checkpoints shape mismatch: expected "
                f"[*, {num_sab_heads}, {head_size}, {head_size}], "
                f"got {list(state_checkpoints.shape)}"
            )

    # Allocate output if not provided
    if output is None:
        output = torch.empty(
            (total_seq_len, num_o_heads, head_size),
            dtype=q.dtype,
            device=q.device,
        )

    device = q.device
    _scale = scale if scale is not None and scale != 0.0 else 1.0 / math.sqrt(head_size)

    _sm_count = get_device_sm_count(device)
    _cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    _device_capability = get_compute_capability(device)
    _arch_major = _device_capability[0]
    _device_name = get_device_name(device)
    cp_heuristic_matches = _arch_major in (9, 10, 12) and should_use_cp_host(
        num_seqs * num_sab_heads,
        _sm_count,
        _device_name,
        device_capability=_device_capability,
    )
    will_use_cp = backend != "gdn_noncp" and (
        use_cp is True or (use_cp == "auto" and cp_heuristic_matches)
    )
    if state_indices is not None:
        if not is_integer_dtype(state_indices.dtype):
            raise ValueError(
                f"state_indices must have an integer dtype, got {state_indices.dtype}"
            )
        if state_indices.shape != (num_seqs,):
            raise ValueError(
                f"state_indices must have shape {(num_seqs,)}, "
                f"got {tuple(state_indices.shape)}"
            )
        # Reject unsupported dispatch paths rather than silently reading/writing
        # the state in packed, sequence-ordered layout.
        if _arch_major not in (9, 10, 12):
            raise NotImplementedError(
                "state_indices is only supported on the SM90/SM100/SM103/SM120 GDN "
                f"prefill kernels; got compute-capability major {_arch_major}, "
                f"use_cp={use_cp!r}."
            )
        # The kernel writes each final state to output_state[state_indices[i]],
        # so a compact [num_seqs, ...] auto-allocation would be indexed out of
        # bounds by arbitrary pool slot ids. Require the caller to pass the pool.
        if output_final_state and output_state is None:
            raise ValueError(
                "state_indices requires an explicit output_state pool sized like "
                "the state pool ([N_pool, H, V, K]); refusing to auto-allocate a "
                "compact [num_seqs, ...] tensor that would be indexed out of bounds."
            )
    if will_use_cp:
        cp_rejection_reason = _cp_delta_rule_rejection_reason(
            arch_major=_arch_major,
            cuda_major=_cuda_major,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output=output,
            initial_state=initial_state,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            state_indices=state_indices,
        )
        if cp_rejection_reason is not None:
            if use_cp is True:
                raise ValueError(cp_rejection_reason)
            warnings.warn(
                f"CP delta rule heuristic matched but CP dispatch is unavailable: {cp_rejection_reason}; "
                "falling back to non-CP delta rule.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if output_final_state and output_state is None:
                output_state = torch.empty(
                    (num_seqs, num_sab_heads, head_size, head_size),
                    dtype=torch.float32,
                    device=device,
                )
            _g = (
                g
                if g is not None
                else torch.ones(
                    total_seq_len, num_sab_heads, dtype=torch.float32, device=device
                )
            )
            _beta = (
                beta
                if beta is not None
                else torch.ones(
                    total_seq_len, num_sab_heads, dtype=torch.float32, device=device
                )
            )
            cp_delta_rule_dsl = cast(
                Callable[..., None],
                {
                    9: cp_delta_rule_dsl_sm90,
                    10: cp_delta_rule_dsl_sm100,
                    12: cp_delta_rule_dsl_sm120,
                }[_arch_major],
            )
            state_indices_kwargs = (
                {"state_indices": state_indices} if state_indices is not None else {}
            )
            checkpoint_kwargs = (
                {
                    "state_checkpoints": state_checkpoints,
                    "checkpoint_cu_starts": checkpoint_cu_starts,
                    "checkpoint_every_n_tokens": checkpoint_every_n_tokens,
                }
                if _arch_major in (9, 10, 12)
                else {}
            )
            cp_delta_rule_dsl(
                output,
                output_state,
                q,
                k,
                v,
                _g,
                _beta,
                cu_seqlens,
                _scale,
                initial_state=initial_state,
                max_seqlen=total_seq_len,
                cp_chunk_len=_cp_chunk_len,
                **state_indices_kwargs,
                **checkpoint_kwargs,
            )
            if output_final_state:
                return output, output_state
            return output
    if backend != "flashinfer":
        if not _GDN_NONCP_AVAILABLE or _gdn_noncp is None:
            if backend == "gdn_noncp":
                raise RuntimeError(
                    "the source-only GDN non-CP GDN backend is not installed"
                )
        else:
            try:
                return _run_gdn_noncp_prefill(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    scale=_scale,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    output=output,
                    output_state=output_state,
                    state_checkpoints=state_checkpoints,
                    checkpoint_cu_starts=checkpoint_cu_starts,
                    checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                    state_indices=state_indices,
                )
            except _gdn_noncp.GDNNonCPUnsupportedError:
                if backend == "gdn_noncp":
                    raise

    if _arch_major == 10:
        if _cuda_major < 13:
            raise NotImplementedError(
                "Blackwell GDN prefill is only supported on CUDA 13+"
            )
        if chunk_gated_delta_rule_sm100 is None:
            raise NotImplementedError("Blackwell GDN prefill kernel is unavailable")

        # Blackwell SM100 and SM103 path (CuTe DSL kernel)
        assert head_size == 128, (
            f"Blackwell GDN prefill requires head_size=128, got {head_size}"
        )

        # Allocate output_state only when needed
        if not output_final_state:
            output_state = None
        elif output_state is None:
            output_state = torch.empty(
                (num_seqs, num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        _g = (
            g
            if g is not None
            else torch.ones(
                total_seq_len, num_sab_heads, dtype=torch.float32, device=device
            )
        )
        _beta = (
            beta
            if beta is not None
            else torch.ones(
                total_seq_len, num_sab_heads, dtype=torch.float32, device=device
            )
        )

        chunk_gated_delta_rule_sm100(
            q,
            k,
            v,
            _g,
            _beta,
            output,
            cu_seqlens,
            initial_state,
            output_state,
            _scale,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            cu_checkpoints=checkpoint_cu_starts,
            output_checkpoints=state_checkpoints,
            state_indices=state_indices,
        )
    elif _arch_major == 12:
        # SM120 Blackwell path (CuTe DSL kernel)
        if chunk_gated_delta_rule_sm120 is None:
            raise NotImplementedError("SM120 GDN prefill DSL kernel is unavailable")
        if output_state is None:
            output_state_shape = (
                initial_state.shape
                if state_indices is not None and initial_state is not None
                else (num_seqs, num_sab_heads, head_size, head_size)
            )
            output_state = torch.empty(
                output_state_shape, dtype=torch.float32, device=device
            )
        chunk_gated_delta_rule_sm120(
            output,
            output_state,
            q,
            k,
            v,
            initial_state,
            g,
            beta,
            cu_seqlens,
            _scale,
            state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
            state_indices=state_indices,
        )
    elif _arch_major == 9:
        # SM90 Hopper path (CuTe DSL kernel)
        if chunk_gated_delta_rule_sm90 is None:
            raise NotImplementedError("SM90 GDN prefill DSL kernel is unavailable")

        if output_state is None:
            output_state = torch.empty(
                (num_seqs, num_sab_heads, head_size, head_size),
                dtype=torch.float32,
                device=device,
            )

        chunk_gated_delta_rule_sm90(
            output,
            output_state,
            q,
            k,
            v,
            initial_state,
            g,
            beta,
            cu_seqlens,
            _scale,
            state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
            state_indices=state_indices,
        )
    else:
        raise NotImplementedError("GDN prefill DSL kernel is unavailable")

    if output_final_state:
        return output, output_state
    else:
        return output
