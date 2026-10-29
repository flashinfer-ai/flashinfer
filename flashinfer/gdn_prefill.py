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
import operator
import os
import warnings
import collections
import functools
from collections.abc import Sequence
from typing import Callable, Literal, Optional, Tuple, Union, cast
import torch

from .api_logging import flashinfer_api
from .cute_dsl.availability import is_cute_dsl_arch_supported
from .trace.templates.gdn import gdn_prefill_trace

try:
    from .jit import cake_gdn as _cake_gdn

    _CAKE_GDN_AVAILABLE = True
    _HostOffsetsError: type[NotImplementedError] = _cake_gdn.CakeGDNUnsupportedError
except (ImportError, RuntimeError):
    _cake_gdn = None
    _CAKE_GDN_AVAILABLE = False
    # CakeGDNUnsupportedError's base: SM8x checks host offsets without Cake.
    _HostOffsetsError = NotImplementedError
from .utils import get_compute_capability, get_device_name, get_device_sm_count
from .gdn_kernels import (
    _chunk_gated_delta_rule_gdn_cp_sm100,
    chunk_gated_delta_rule_sm80,
    chunk_gated_delta_rule_sm90,
    chunk_gated_delta_rule_sm100,
    chunk_gated_delta_rule_sm120,
    cp_delta_rule_dsl_sm80,
    cp_delta_rule_dsl_sm90,
    cp_delta_rule_dsl_sm100,
    cp_delta_rule_dsl_sm120,
)
from .gdn_kernels.delta_rule_dsl.varlen_helper import (
    is_integer_dtype,
    should_use_cp_host,
    should_use_cp_sm80_host,
    should_use_v32_sm80_host,
)


_STATE_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

_GDN_CP_STATE_DTYPES: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.bfloat16,
    torch.float16,
)


@functools.cache
def _cake_gdn_arch(device_index: int) -> "_cake_gdn.CakeGDNArch":
    """Resolve the Cake GDN architecture of one CUDA device once."""

    major, minor = torch.cuda.get_device_capability(device_index)
    return _cake_gdn.arch_for_compute_capability(major, minor)


@functools.cache
def _cake_gdn_sm_count(device_index: int) -> int:
    """Streaming-multiprocessor count of one CUDA device, resolved once."""

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@functools.cache
def _cake_gdn_tensormap_workspace(
    device_index: int, stream: int, grid_x: int
) -> torch.Tensor:
    """Per-launch tensor-map scratch (512 bytes per CTA), resolved once per device, stream and grid.

    The kernel rewrites the workspace at every launch, so one buffer per stream
    serves every call on that stream in order and keeps the launch path
    allocation-free (no per-call ``torch.empty`` under CUDA Graph capture either).
    """

    return torch.empty(
        grid_x * 512, dtype=torch.uint8, device=torch.device("cuda", device_index)
    )


@functools.cache
def _cake_gdn_sentinel(device_index: int, dtype: torch.dtype) -> torch.Tensor:
    """One-element placeholder passed for absent optional tensors.

    The kernel never reads or writes it (the matching specialization flag is
    off), so one tensor per device and dtype serves every call and keeps the
    launch path allocation-free.
    """

    return torch.empty(1, dtype=dtype, device=torch.device("cuda", device_index))


# Unit gates for the public routes without ``g``/``beta``: the kernel variant
# selected for them still reads FP32 gate tensors of ones, so the host keeps a
# bounded LRU of those tensors per device and [tokens, heads] shape instead of
# filling a new one on every call (one allocation plus one fill kernel per call
# otherwise, which the export allocation probe rejects).  A tensor first built
# under CUDA Graph capture is only filled when that graph replays, so it is
# handed to the capture and not cached.
_CAKE_GDN_UNIT_GATES: "collections.OrderedDict[tuple[int, int, int], torch.Tensor]" = (
    collections.OrderedDict()
)
_CAKE_GDN_UNIT_GATES_LIMIT = 8


def _cake_gdn_unit_gates(
    device_index: int, total_seq_len: int, num_o_heads: int
) -> torch.Tensor:
    """FP32 ones of shape [total_seq_len, num_o_heads], reused across calls."""

    key = (device_index, total_seq_len, num_o_heads)
    cached = _CAKE_GDN_UNIT_GATES.get(key)
    if cached is not None:
        _CAKE_GDN_UNIT_GATES.move_to_end(key)
        return cached
    ones = torch.ones(
        (total_seq_len, num_o_heads),
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )
    if torch.cuda.is_current_stream_capturing():
        return ones
    _CAKE_GDN_UNIT_GATES[key] = ones
    while len(_CAKE_GDN_UNIT_GATES) > _CAKE_GDN_UNIT_GATES_LIMIT:
        _CAKE_GDN_UNIT_GATES.popitem(last=False)
    return ones


# Host-free launch metadata.  The non-CP Cake prefill route reads
# nothing back from the device: ``num_seqs`` and the token count are tensor
# shapes, the kernel variant and the persistent grid follow an upper bound on
# the longest sequence, and the kernel resolves ``cu_seqlens`` (and
# ``checkpoint_cu_starts``) on the device at every launch, clamping every
# offset into ``[0, total_tokens]`` and treating a non-monotonic pair as an
# empty sequence (a documented no-op, never an out-of-bounds access).  There
# is no host-int cache and no cached int32 copy: a metadata buffer the caller
# rewrites in place (a persistent engine buffer, a CUDA Graph replay) is
# always read live.  A record that declares ``metadata_i64`` (every sm_107a
# non-CP prefill record) reads int32 or int64 ``cu_seqlens`` /
# ``state_indices`` / ``checkpoint_cu_starts`` directly, so the caller's
# buffers are passed as they are; every other record takes int32 and an
# int64 buffer is cast per call (``_cake_gdn_prefill_i32``).
# ``cu_seqlens_cpu`` is the optional host copy engines
# already hold (vLLM ``query_start_loc_cpu``, the prefix sums of SGLang's
# ``extend_seq_lens_cpu``): it tightens the host validation and the launch
# bound and enables the ragged-batch launch grid (one CTA per tile when the
# host model of the persistent schedule predicts a shorter critical path for
# the given lengths), all without a synchronization.


def _cake_gdn_prefill_host_offsets(
    cu_seqlens_cpu: Union[Sequence[int], torch.Tensor],
    *,
    num_seqs: int,
    total_tokens: int,
) -> tuple[int, ...]:
    """Validate the host copy of ``cu_seqlens`` and return it as a tuple of ints.

    Accepts a one-dimensional int32/int64 CPU tensor or a sequence of Python
    ints of length ``num_seqs + 1`` whose values are non-decreasing, start at
    zero and end at ``total_tokens``; anything else raises
    ``CakeGDNUnsupportedError`` before any launch.  A CUDA tensor is rejected
    rather than copied: that copy would be the device-to-host synchronization
    this parameter exists to avoid.
    """

    if isinstance(cu_seqlens_cpu, torch.Tensor):
        if (
            cu_seqlens_cpu.device.type != "cpu"
            or cu_seqlens_cpu.ndim != 1
            or cu_seqlens_cpu.dtype not in (torch.int32, torch.int64)
        ):
            raise _HostOffsetsError(
                "cu_seqlens_cpu must be a one-dimensional int32/int64 CPU tensor"
            )
        values = cu_seqlens_cpu.tolist()
    else:
        values = []
        for value in cu_seqlens_cpu:
            if isinstance(value, bool):
                raise _HostOffsetsError(
                    "cu_seqlens_cpu must hold integer offsets, not bools"
                )
            try:
                values.append(operator.index(value))
            except TypeError as exc:
                raise _HostOffsetsError(
                    "cu_seqlens_cpu must be a CPU integer tensor or a sequence of ints"
                ) from exc
    offsets = tuple(int(value) for value in values)
    if len(offsets) != num_seqs + 1:
        raise _HostOffsetsError(
            f"cu_seqlens_cpu must have num_seqs + 1 = {num_seqs + 1} entries, "
            f"got {len(offsets)}"
        )
    if (
        offsets[0] != 0
        or offsets[-1] != total_tokens
        or any(end < start for start, end in zip(offsets, offsets[1:], strict=False))
    ):
        raise _HostOffsetsError(
            "GDN non-CP prefill requires monotonic cu_seqlens spanning all input tokens"
        )
    return offsets


def _cake_gdn_prefill_grid_x(
    total_tiles: int,
    sm_count: int,
    dvsplit: bool,
    bound_tokens: int,
    host_seq_lens: Optional[Sequence[int]] = None,
) -> int:
    """Persistent CTA count for ``total_tiles`` work tiles on ``sm_count`` SMs.

    One copy of the host-known grid rule lives in ``flashinfer/jit/cake_gdn.py``
    (the variant resolver's gate-stream rule reads the same grid); it mirrors
    the Cake generator's grid rule exactly and the exporter's mirror test keeps
    the copies identical.  ``host_seq_lens`` -- the per-sequence lengths known
    only from ``cu_seqlens_cpu`` -- enables the ragged-batch route (one CTA per
    tile when the host model predicts a shorter critical path); without it the
    residency rule decides.  ``bound_tokens`` never changes the grid.  The rule
    costs O(S) host work for S sequences and is memoised (the lengths go in
    as a tuple).
    """

    return _cake_gdn._cake_gdn_prefill_grid_x(
        total_tiles,
        sm_count,
        dvsplit,
        bound_tokens,
        None if host_seq_lens is None else tuple(host_seq_lens),
    )


def _cake_gdn_prefill_i32(values: torch.Tensor) -> torch.Tensor:
    """Int32 launch metadata: pass-through for int32, a per-call cast for int64.

    The cast is one small device kernel on the caller's stream (no host
    synchronization; recorded into the graph under capture) and is never
    cached, so an in-place rewrite of the caller's buffer is always what the
    kernel reads.  Int32 metadata is the zero-copy path.  Records that declare
    ``metadata_i64`` (``CakeGDNRoute.metadata_i64``) never take this path:
    their kernels read int32 or int64 metadata directly.
    """

    return values if values.dtype == torch.int32 else values.to(torch.int32)


def _cake_gdn_prefill_launch_metadata(
    values: torch.Tensor, *, metadata_i64: bool
) -> torch.Tensor:
    """The metadata buffer the kernel reads: as given for a ``metadata_i64`` record, else int32."""

    return values if metadata_i64 else _cake_gdn_prefill_i32(values)


# Per-call device-side slot validation, same default as ``gdn_decode.py``: the
# ``torch._assert_async`` chain expands to four elementwise kernels plus the
# ``_assert_async_cuda_kernel`` and exposes ~40 us of launch gaps around a
# 25-70 us kernel on the indexed state-pool rows (R200 CUPTI per-kernel
# decomposition: 5-6 kernels per call, 41-45 us of inter-kernel gap, while the
# Cake kernel itself ran faster than the CuTe one).  The CuTe path trusts
# caller-provided slots, so the Cake adapter does the same by default; set
# ``FLASHINFER_CAKE_GDN_VALIDATE_SLOTS=1`` to restore the asynchronous
# fail-closed check (used by the invalid-slot tests).
_CAKE_GDN_VALIDATE_SLOTS = (
    os.environ.get("FLASHINFER_CAKE_GDN_VALIDATE_SLOTS", "0") == "1"
)


def _cake_gdn_assert_state_slots(
    indices: torch.Tensor, pool_size: int, *, name: str, allow_minus_one: bool
) -> None:
    """Validate CUDA-resident state slots without a host synchronization.

    Only active when ``FLASHINFER_CAKE_GDN_VALIDATE_SLOTS=1``; see the note above.
    """

    if not _CAKE_GDN_VALIDATE_SLOTS:
        return
    in_pool = (indices >= 0) & (indices < pool_size)
    valid = ((indices == -1) | in_pool) if allow_minus_one else in_pool
    torch._assert_async(
        valid.all(),
        f"{name} must contain "
        + ("-1 or " if allow_minus_one else "")
        + f"slots in [0, {pool_size})",
    )


def _validate_slot_args(
    *,
    q: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    output_final_state: bool,
    checkpoint_every_n_tokens: int,
    output_state_indices: Optional[torch.Tensor],
    checkpoint_state_indices: Optional[torch.Tensor],
    prefix_lens: Optional[torch.Tensor],
    backend: str,
    use_cp: Literal["auto"] | bool,
    cp_chunk_len: Optional[int],
) -> bool:
    """Check the slot-addressing arguments; return whether any is given.

    ``output_state_indices``, ``checkpoint_state_indices`` and ``prefix_lens``
    let a caller keep its states in one pool addressed per sequence (a paged
    recurrent-state cache): read and write rows differ, checkpoints land in
    pool rows, and checkpoint positions are absolute. Only the native non-CP
    kernels implement them.
    """

    given = [
        name
        for name, tensor in (
            ("output_state_indices", output_state_indices),
            ("checkpoint_state_indices", checkpoint_state_indices),
            ("prefix_lens", prefix_lens),
        )
        if tensor is not None
    ]
    if not given:
        return False
    names = ", ".join(given)
    if backend not in ("auto", "flashinfer"):
        raise ValueError(
            f"{names} require the native non-CP kernels "
            f'(backend="auto" or "flashinfer"), got backend={backend!r}'
        )
    if use_cp is True or cp_chunk_len is not None:
        raise ValueError(f"{names} are not supported with context parallelism")
    num_seqs = None if cu_seqlens is None else cu_seqlens.size(0) - 1

    def check_index_tensor(name: str, tensor: torch.Tensor, length) -> None:
        if not is_integer_dtype(tensor.dtype):
            raise ValueError(f"{name} must be int32 or int64, got {tensor.dtype}")
        if tensor.ndim != 1 or (length is not None and tensor.size(0) not in length):
            expected = "" if length is None else f" of {length[0]} entries"
            raise ValueError(
                f"{name} must be 1D{expected}, got shape {tuple(tensor.shape)}"
            )
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    seq_shape = None if num_seqs is None else (num_seqs,)
    if output_state_indices is not None:
        check_index_tensor("output_state_indices", output_state_indices, seq_shape)
        if state_indices is None:
            raise ValueError("output_state_indices requires state_indices")
        if output_state is None or not output_final_state:
            raise ValueError(
                "output_state_indices requires an explicit output_state pool "
                "and output_final_state=True"
            )
    if checkpoint_state_indices is not None:
        check_index_tensor("checkpoint_state_indices", checkpoint_state_indices, None)
        if checkpoint_every_n_tokens <= 0:
            raise ValueError(
                "checkpoint_state_indices requires checkpoint_every_n_tokens > 0"
            )
    if prefix_lens is not None:
        check_index_tensor("prefix_lens", prefix_lens, seq_shape)
        if checkpoint_every_n_tokens <= 0:
            raise ValueError("prefix_lens requires checkpoint_every_n_tokens > 0")
    return True


def _cake_gdn_prefill_dtype_name(dtype: torch.dtype) -> str:
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
        raise _cake_gdn.CakeGDNUnsupportedError(
            f"unsupported GDN non-CP prefill dtype {dtype}"
        ) from exc


def _run_cake_gdn_prefill(
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
    max_seqlen: Optional[int] = None,
    cu_seqlens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Launch one exact manifest-backed GDN non-CP prefill row without a device read.

    ``max_seqlen`` is an upper bound on the longest sequence (validated by the
    public entry); ``cu_seqlens_cpu`` is the optional host copy of
    ``cu_seqlens``.  The launch bound is the exact maximum from the host copy
    when given, else ``max_seqlen``, else the packed token count; the host
    copy's per-sequence lengths also drive the ragged-batch launch grid (the
    variant resolver and the launch read the same lengths).
    """

    if not _CAKE_GDN_AVAILABLE or _cake_gdn is None:
        raise RuntimeError("the source-only Cake GDN backend is not installed")
    if use_qk_l2norm_in_kernel:
        raise _cake_gdn.CakeGDNUnsupportedError(
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
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill requires all tensors on one CUDA device"
        )
    device_index = int(
        q.device.index if q.device.index is not None else torch.cuda.current_device()
    )
    arch = _cake_gdn_arch(device_index)
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        tensor.dtype != q.dtype for tensor in (k, v, output)
    ):
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill requires matching FP16 or BF16 Q/K/V/output"
        )
    if any(not tensor.is_contiguous() for tensor in (q, k, v, output)):
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill requires contiguous Q/K/V/output"
        )
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or output.ndim != 3:
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill tensors must be rank 3"
        )
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
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill requires exact [tokens,heads,128] tensors"
        )
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or not cu_seqlens.is_contiguous()
        or int(cu_seqlens.numel()) < 2
    ):
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill requires contiguous int32/int64 cu_seqlens"
        )
    # Host-known metadata only (see the module note): the sequence count is a
    # shape, the per-sequence lengths are known only through the host copy.
    num_seqs = int(cu_seqlens.numel()) - 1
    seq_lens: Optional[tuple[int, ...]] = None
    if cu_seqlens_cpu is not None:
        offsets = _cake_gdn_prefill_host_offsets(
            cu_seqlens_cpu, num_seqs=num_seqs, total_tokens=total_seq_len
        )
        seq_lens = tuple(
            end - start for start, end in zip(offsets, offsets[1:], strict=False)
        )
    if max_seqlen is not None and (
        type(max_seqlen) is not int or max_seqlen <= 0 or max_seqlen > total_seq_len
    ):
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill max_seqlen must be an int in [1, total tokens]"
        )
    if seq_lens is not None:
        bound_tokens = max(seq_lens)
        if max_seqlen is not None and bound_tokens > max_seqlen:
            raise _cake_gdn.CakeGDNUnsupportedError(
                "max_seqlen is smaller than the longest sequence in cu_seqlens_cpu"
            )
    elif max_seqlen is not None:
        bound_tokens = max_seqlen
    else:
        bound_tokens = total_seq_len
    gates_present = g is not None and beta is not None

    gate = g
    update_gate = beta
    if gate is None or update_gate is None:
        unit = _cake_gdn_unit_gates(q.device.index, total_seq_len, num_o_heads)
        gate = unit if gate is None else gate
        update_gate = unit if update_gate is None else update_gate
    for name, tensor in (("g", gate), ("beta", update_gate)):
        if (
            tensor.dtype != torch.float32
            or tuple(tensor.shape) != (total_seq_len, num_o_heads)
            or not tensor.is_contiguous()
        ):
            raise _cake_gdn.CakeGDNUnsupportedError(
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
        raise _cake_gdn.CakeGDNUnsupportedError(
            "GDN non-CP prefill requires one state dtype across initial/final/checkpoints"
        )
    state_dtype_name = _cake_gdn_prefill_dtype_name(state_dtype)
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
            raise _cake_gdn.CakeGDNUnsupportedError(
                f"GDN non-CP prefill requires {name} with contiguous [H,V,K] rows"
            )
    if state_indices is not None and (
        state_indices.ndim != 1
        or int(state_indices.numel()) != num_seqs
        or state_indices.dtype not in (torch.int32, torch.int64)
        or not state_indices.is_contiguous()
    ):
        raise _cake_gdn.CakeGDNUnsupportedError(
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
            _cake_gdn_assert_state_slots(
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
            or checkpoint_cu_starts.ndim != 1
            or checkpoint_cu_starts.dtype not in (torch.int32, torch.int64)
            or not checkpoint_cu_starts.is_contiguous()
        ):
            raise _cake_gdn.CakeGDNUnsupportedError(
                "GDN non-CP checkpoint prefill requires contiguous state/cumulative buffers"
            )
        # Host shapes only.  With ``cu_seqlens_cpu`` the checkpoint count is
        # exact; otherwise ``state_checkpoints.shape[0]`` is taken as given.
        # The contents of ``checkpoint_cu_starts`` (consistent with
        # ``cu_seqlens`` and the interval) stay the documented caller
        # precondition the kernel guards on the device: verifying them here
        # would be the device-to-host read this route no longer performs.
        if int(checkpoint_cu_starts.numel()) != num_seqs + 1:
            raise _cake_gdn.CakeGDNUnsupportedError(
                "GDN non-CP checkpoint_cu_starts must have num_seqs + 1 entries"
            )
        expected_checkpoints = (
            None
            if seq_lens is None
            else sum(length // checkpoint_every_n_tokens for length in seq_lens)
        )
        if (
            state_checkpoints.ndim != 4
            or tuple(state_checkpoints.shape[1:]) != (num_o_heads, 128, 128)
            or (
                expected_checkpoints is not None
                and int(state_checkpoints.shape[0]) != expected_checkpoints
            )
        ):
            raise _cake_gdn.CakeGDNUnsupportedError(
                "GDN non-CP state_checkpoints must have shape "
                "[sum(floor(seq_len/interval)),H,128,128]"
            )

    route = _cake_gdn.select_cake_gdn_prefill_variant(
        arch=arch,
        io_dtype=_cake_gdn_prefill_dtype_name(q.dtype),
        state_dtype=state_dtype_name,
        num_seqs=num_seqs,
        total_seq_len=total_seq_len,
        max_seq_len=bound_tokens,
        num_q_heads=num_q_heads,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        use_initial_state=initial_state is not None,
        store_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        use_state_indices=state_indices is not None,
        gates_present=gates_present,
        host_seq_lens=seq_lens,
    )
    entry = _cake_gdn.load_cake_gdn_kernel(route.variant_name, arch)
    dvsplit = route.route_id.endswith(".dvsplit")
    total_tiles = num_seqs * num_o_heads * (2 if dvsplit else 1)
    # The device's SM count and, with ``cu_seqlens_cpu``, the host sequence
    # lengths: the rule the resolver evaluated with the architecture's nominal
    # count; this grid is authoritative for the workspace.
    grid_x = _cake_gdn_prefill_grid_x(
        total_tiles,
        _cake_gdn_sm_count(device_index),
        dvsplit,
        bound_tokens,
        seq_lens,
    )

    empty_i32 = _cake_gdn_sentinel(device_index, torch.int32)
    # A ``metadata_i64`` record (every sm_107a record) reads int32 or int64
    # offsets and slots directly: the caller's buffers go to the kernel as
    # they are.  Every other record takes int32; int64 metadata is cast per
    # call on the caller's stream (``_cake_gdn_prefill_i32``), never cached.
    metadata_i64 = route.metadata_i64
    cu_seqlens_i32 = _cake_gdn_prefill_launch_metadata(
        cu_seqlens, metadata_i64=metadata_i64
    )
    state_indices_i32 = (
        empty_i32
        if state_indices is None
        else _cake_gdn_prefill_launch_metadata(state_indices, metadata_i64=metadata_i64)
    )
    empty_state = _cake_gdn_sentinel(device_index, state_dtype)
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
        else _cake_gdn_prefill_launch_metadata(
            checkpoint_cu_starts, metadata_i64=metadata_i64
        )
    )
    tensormap_workspace = _cake_gdn_tensormap_workspace(
        device_index, torch.cuda.current_stream(device_index).cuda_stream, grid_x
    )
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
        # Launch ABI v2: the token count bounds every cu_seqlens read on the device.
        total_seq_len,
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


_CAKE_GDN_CP_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
# Explicit SM107 band for the automatic (``use_cp="auto"``) Cake GDN CP route.
# 52 is the largest ``num_seqs * num_sab_heads`` for which ``should_use_cp_host``
# holds on the 212-SM part (``4 * work < 212``); 4096 is the shortest
# ``max_seqlen`` admitted to the band.
_CAKE_GDN_CP_SM107A_MAX_PARALLEL_STATES = 52
_CAKE_GDN_CP_SM107A_MIN_MAX_SEQLEN = 4096


def _cake_gdn_cp_sm107a_plan_pays(
    *,
    num_seqs: int,
    num_sab_heads: int,
    max_seqlen: int,
    total_tokens: int,
    num_sms: int,
) -> bool:
    """Measured band of the automatic SM107 Cake CP route.

    Cake CP beats the sequential Cake kernel only with enough chunks per
    sequence (at least 6 of at least 512 tokens) or with long chunks (at least
    4096 tokens); the plan is the backend's own physical chunk plan.
    """

    from .gdn_kernels.blackwell import cake_gdn_cp_backend as _cp

    if total_tokens <= 0 or max_seqlen <= 0:
        return False
    source = _cp._choose_chunk_len(
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        max_seqlen=max_seqlen,
        num_heads=num_sab_heads,
        num_sms=num_sms,
    )
    physical = _cp._default_physical_chunk_len(
        source_chunk_len=source,
        total_tokens=total_tokens,
        num_seqs=num_seqs,
        max_seqlen=max_seqlen,
        num_heads=num_sab_heads,
        num_sms=num_sms,
    )
    chunks = -(-max_seqlen // physical)
    return physical >= 4096 or (chunks >= 6 and physical >= 512)


# SM100/SM103 band of the automatic (``use_cp="auto"``) Cake GDN CP route.  The
# same rule is implemented, without sharing code, by the Cake dispatcher
# (``_should_dispatch_cp_band`` in the Cake dispatcher)
# and the Cake exporter plan (``exports/gdn/export.py::_prefill_plan``): BF16/FP16
# rows with FP32 initial+final state, gates, no checkpoints and no state indices
# whose parallel work leaves most SMs idle in the persistent non-CP kernel (the
# CP heuristic ``4 * N * Ho < SMs``) and whose longest sequence carries at least
# 4096 tokens, so the four-stage program's fixed stage costs pay for the shorter
# per-sequence chain.
_CAKE_GDN_CP_SM100_BAND_CAPABILITIES = ((10, 0), (10, 3))
_CAKE_GDN_CP_SM100_BAND_WORK_FACTOR = 4
_CAKE_GDN_CP_SM100_BAND_MIN_MAX_SEQLEN = 4096


def _cake_gdn_cp_band_max_seqlen(
    max_seqlen: Optional[int],
    total_seq_len: int,
    num_seqs: int,
    cu_seqlens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
) -> int:
    """The longest-sequence bound the SM100/SM103 band rule is evaluated on.

    With the caller's ``max_seqlen`` hint (or the exact host prefix ``cu_seqlens_cpu``)
    the rule sees the exact bound, the
    quantity the Cake dispatcher and the export plan read from the host
    sequence lengths.  Without a hint the host cannot read ``cu_seqlens``
    back, so the band uses the balanced-batch estimate
    ``ceil(total_seq_len / num_seqs)`` -- the convention the Cake CP program
    resolves an omitted hint to -- rather than the packed length, which
    would enter the band for any batch whose total reaches 4096 tokens
    (``4 x 2048``) although no sequence does.  The CP launch bound
    (``cp_max_seqlen``) keeps the packed-length fallback, which is correct
    for every batch.
    """

    if max_seqlen is not None:
        return int(max_seqlen)
    if cu_seqlens_cpu is not None:
        # The host prefix (``cu_seqlens_cpu``) carries the exact longest sequence; it is validated
        # against ``cu_seqlens`` by the non-CP route before any launch, and read here without a device sync.
        starts = (
            cu_seqlens_cpu.tolist()
            if isinstance(cu_seqlens_cpu, torch.Tensor)
            else [int(v) for v in cu_seqlens_cpu]
        )
        if len(starts) >= 2:
            longest = max(
                int(b) - int(a) for a, b in zip(starts[:-1], starts[1:], strict=False)
            )
            if longest > 0:
                return longest
    return -(-int(total_seq_len) // int(num_seqs))


def _cake_gdn_cp_sm100_band(
    *,
    device_capability: tuple[int, int],
    num_seqs: int,
    num_sab_heads: int,
    max_seqlen: int,
    num_sms: int,
    io_dtype: torch.dtype,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    output_final_state: bool,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_indices: Optional[torch.Tensor],
) -> bool:
    """Whether an ``auto`` ``cake_gdn`` request on SM100/SM103 runs the Cake CP route."""

    return bool(
        device_capability in _CAKE_GDN_CP_SM100_BAND_CAPABILITIES
        and io_dtype in (torch.bfloat16, torch.float16)
        and initial_state is not None
        and initial_state.dtype == torch.float32
        and output_final_state
        and (output_state is None or output_state.dtype == torch.float32)
        and g is not None
        and beta is not None
        and checkpoint_every_n_tokens == 0
        and state_indices is None
        and num_seqs > 0
        and num_sab_heads > 0
        and _CAKE_GDN_CP_SM100_BAND_WORK_FACTOR * num_seqs * num_sab_heads < num_sms
        and max_seqlen >= _CAKE_GDN_CP_SM100_BAND_MIN_MAX_SEQLEN
    )


def _use_gdn_cp_sm100(
    *,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    cp_chunk_len: Optional[int],
) -> bool:
    """Select GDN CP for the ratified PR4078 domain and FP32 checkpoints."""

    checkpoint_enabled = checkpoint_every_n_tokens > 0
    if checkpoint_enabled:
        if (
            state_checkpoints is None
            or state_checkpoints.dtype != torch.float32
            or checkpoint_cu_starts is None
            or checkpoint_cu_starts.dtype not in (torch.int32, torch.int64)
            or not checkpoint_cu_starts.is_cuda
            or cp_chunk_len not in (None, checkpoint_every_n_tokens)
        ):
            return False
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        return False
    if cp_chunk_len is not None and (cp_chunk_len <= 0 or cp_chunk_len % 64 != 0):
        return False
    return all(
        tensor is None or tensor.dtype in _GDN_CP_STATE_DTYPES
        for tensor in (initial_state, output_state)
    )


def _is_gdn_cp_sm100_supported_shape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Return whether tensor shapes satisfy the generated backend contract."""

    if (
        q.ndim != 3
        or k.ndim != 3
        or v.ndim != 3
        or q.shape[0] != k.shape[0]
        or q.shape[0] != v.shape[0]
        or q.shape[2] != 128
        or k.shape[2] != 128
        or v.shape[2] != 128
    ):
        return False
    hq, hk, hv = int(q.shape[1]), int(k.shape[1]), int(v.shape[1])
    return bool(
        hq > 0
        and hk > 0
        and hv > 0
        and ((hq == hk and hv % hq == 0) or (hk == hv and hq % hk == 0))
    )


def _allocate_sm8x_output_state(
    num_seqs: int,
    num_sab_heads: int,
    head_size: int,
    device,
    initial_state: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
    returned: bool = True,
) -> torch.Tensor:
    """The SM8x final-state buffer; a zero-length sequence never writes its row.

    A returned buffer starts as the initial state, or zero without one, so that
    row is h0 or zero. Scratch the caller never reads stays uninitialized.
    """
    shape = (
        initial_state.shape
        if state_indices is not None and initial_state is not None
        else (num_seqs, num_sab_heads, head_size, head_size)
    )
    if returned and initial_state is None:
        return torch.zeros(shape, dtype=torch.float32, device=device)
    out = torch.empty(shape, dtype=torch.float32, device=device)
    if returned:
        out.copy_(initial_state)
    return out


def _cp_delta_rule_rejection_reason(
    *,
    arch_major: int,
    use_gdn_cp_backend: bool,
    cuda_version: tuple[int, int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    output: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
    cp_chunk_len: Optional[int],
) -> Optional[str]:
    """Why CP cannot serve this call, or ``None`` if it can."""
    if arch_major == 8:
        if cp_delta_rule_dsl_sm80 is None:
            return "CP delta rule SM8x DSL kernel is unavailable"
    elif arch_major == 9:
        if cp_delta_rule_dsl_sm90 is None:
            return "CP delta rule SM90 DSL kernel is unavailable"
    elif arch_major == 10:
        if use_gdn_cp_backend and cuda_version < (12, 8):
            return "GDN CP SM100 kernel requires CUDA 12.8 or newer"
        if not use_gdn_cp_backend and cuda_version < (13, 0):
            return "CP delta rule SM100 DSL kernel requires CUDA 13 or newer"
        if use_gdn_cp_backend and _chunk_gated_delta_rule_gdn_cp_sm100 is None:
            return "GDN CP SM100 kernel is unavailable"
        if not use_gdn_cp_backend and cp_delta_rule_dsl_sm100 is None:
            return "CP delta rule SM100 DSL kernel is unavailable"
    elif arch_major == 12:
        if cp_delta_rule_dsl_sm120 is None:
            return "CP delta rule SM120 DSL kernel is unavailable"
    else:
        return (
            "CP delta rule is currently implemented only for SM8x, SM90, "
            "SM100, and SM120"
        )
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
    if initial_state is not None and not initial_state.is_contiguous():
        if not use_gdn_cp_backend and state_indices is None:
            return "CP delta rule requires initial_state to be contiguous"
    return None


@functools.cache
def _cudnn_gdn_prefill_available() -> bool:
    from .cudnn.linear_attention import _linear_attention_auto_available

    return _linear_attention_auto_available()


def _prefer_cudnn_gdn_prefill(q, v, cu_seqlens) -> bool:
    # B200 raw-gate measurements include the existing fused-gate native path.
    # Higher-head GVA was near parity, so retain its native policy for now.
    return (
        q.is_cuda
        and q.shape[1:] == (4, 128)
        and v.shape[1:] == (8, 128)
        and 128 <= q.shape[0] <= 8192
        and 2 <= cu_seqlens.numel() <= 3
        and get_compute_capability(q.device) == (10, 0)
    )


def _is_cudnn_gdn_auto_eligible(
    q, k, v, g, beta, A_log, dt_bias, initial_state, output_state, output, cu_seqlens
) -> bool:
    """Keep automatic substitution within the measured compact inference contract."""
    if not q.is_cuda or q.ndim != 3 or v.ndim != 3:
        return False
    total, _, dim = q.shape
    heads = v.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    tensors = (
        (q, q.shape, torch.bfloat16),
        (k, q.shape, torch.bfloat16),
        (v, (total, heads, dim), torch.bfloat16),
        (g, (total, heads), torch.bfloat16),
        (beta, (total, heads), torch.bfloat16),
        (A_log, (heads,), torch.float32),
        (dt_bias, (heads,), torch.float32),
        (initial_state, (num_seqs, heads, dim, dim), torch.float32),
        (output_state, (num_seqs, heads, dim, dim), torch.float32),
        (output, (total, heads, dim), torch.bfloat16),
        (cu_seqlens, (num_seqs + 1,), torch.int64),
    )
    for tensor, shape, dtype in tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            return False
    if not torch.is_inference_mode_enabled() and (
        output.is_inference() or output_state.is_inference()
    ):
        return False

    # Contiguous byte ranges also detect aliases made through independent
    # DLPack Storage objects. A declined call keeps its existing native route.
    def span(tensor):
        begin = tensor.data_ptr()
        return begin, begin + tensor.numel() * tensor.element_size()

    written = (span(output), span(output_state))
    read = tuple(
        span(tensor)
        for tensor in (q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens)
    )
    for i, (begin, end) in enumerate(written):
        if any(
            begin < other_end and other_begin < end
            for other_begin, other_end in (*read, *written[:i])
        ):
            return False
    return True


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
    backend: Literal["auto", "flashinfer", "cake_gdn", "cudnn"] = "auto",
    max_seqlen: Optional[int] = None,
    *,
    gate_domain: Literal["linear", "log"] = "linear",
    use_gate_in_kernel: bool = False,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
    cu_seqlens_cpu: Optional[Union[Sequence[int], torch.Tensor]] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    checkpoint_state_indices: Optional[torch.Tensor] = None,
    prefix_lens: Optional[torch.Tensor] = None,
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
        float32 for precomputed alpha; raw gates may also be float16 or
        bfloat16 when ``use_gate_in_kernel=True``. Defaults to all ones
        when ``None``.
    beta : torch.Tensor, optional
        Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
        Must be float32 for precomputed beta; logits may also use the Q
        dtype when ``beta_is_logit=True``. Defaults to all ones when ``None``.
    scale : float, optional
        Scale factor for the attention scores.  Defaults to
        ``1 / sqrt(head_size)`` when ``None``.
    initial_state : torch.Tensor, optional
        Initial KV state. Packed, sequence-ordered shape
        ``[num_seqs, num_sab_heads, head_size, head_size]``.  Must be
        float32, bfloat16, float16, float8_e4m3fn, or float8_e5m2. Starts from zero state
        when ``None``.  When ``state_indices`` is given (SM8x/SM90/SM100/SM103/SM120),
        this is instead the state **pool** ``[N_pool, num_sab_heads,
        head_size, head_size]`` and sequence ``i`` reads its initial state
        from row ``state_indices[i]``; on SM100/SM103 every rank-4 positive,
        non-overlapping stride layout is accepted.
    output_final_state : bool
        Whether to output the final state.  Default: ``False``.
    cu_seqlens : torch.Tensor
        Cumulative sequence lengths of shape ``[num_seqs + 1]``, integer
        int32 or int64 dtype on the same CUDA device as ``q``. Required for
        variable-length sequences (varlen mode); must not be ``None``.
        Repeated adjacent offsets represent legal zero-length sequences.
    use_qk_l2norm_in_kernel : bool
        Whether to L2-normalize each Q/K head with epsilon ``1e-6``.
        Normalization accumulates in float32 and rounds back to the input
        dtype before the chunked kernel. Default: ``False``.
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
        ``i``'s final state is written to row ``state_indices[i]``, or to row
        ``output_state_indices[i]`` when that is given (in place when
        ``output_state is initial_state``); it must be provided by the
        caller (auto-allocation is rejected, since a compact ``[num_seqs, ...]``
        buffer would be indexed out of bounds by the pool slot ids).
        SM100/SM103 accepts the same positive, non-overlapping rank-4 stride
        layouts as ``initial_state``.
    state_checkpoints : torch.Tensor, optional
        Pre-allocated checkpoint tensor of shape ``[total_checkpoints,
        num_sab_heads, head_size, head_size]``. May be float32, bfloat16,
        float16, float8_e4m3fn, or float8_e5m2. Required when
        ``checkpoint_every_n_tokens > 0``. Context-parallel checkpointing is
        currently supported on SM8x, SM90, SM100, and SM120.
    checkpoint_cu_starts : torch.Tensor, optional
        Cumulative checkpoint counts of shape ``[num_seqs + 1]``, int32 or
        int64 on the same CUDA device as ``q``.
        ``checkpoint_cu_starts[i+1] - checkpoint_cu_starts[i]`` is the
        number of checkpoints for sequence ``i``: ``seq_len_i //
        checkpoint_every_n_tokens``, or ``(prefix_lens[i] + seq_len_i) // N -
        prefix_lens[i] // N`` with ``prefix_lens`` (``N =
        checkpoint_every_n_tokens``). With ``checkpoint_state_indices`` only
        ``checkpoint_cu_starts[i]`` is read and ``[num_seqs]`` entries are
        also accepted.  Required when
        ``checkpoint_every_n_tokens > 0``. Without ``checkpoint_state_indices``
        the values must be monotonic and consistent with ``cu_seqlens``; this
        caller precondition is not checked at launch to avoid a device-to-host
        synchronization.
    checkpoint_every_n_tokens : int
        Store intermediate state every N tokens.  Must be a multiple of the
        chunk size (64).  ``0`` disables checkpointing (default).
    use_cp : Literal["auto"] | bool, optional:
        Whether to use the architecture-specific context-parallel DSL
        implementation on SM8x, SM90, SM100/SM103/SM107 or SM120, where it is
        supported and low-parallelism heuristics match. SM100/SM103/SM107 uses
        the generated GDN CP-only four-stage implementation for structurally
        supported shapes; other legal configurations retain the CuTe-DSL
        implementation. ``"auto"`` enables conservative routing, ``True``
        requires CP support, and ``False`` disables CP. Default: ``"auto"``.
    state_indices : torch.Tensor, optional
        Int32 or int64 tensor of shape ``[num_seqs]``
        (SM8x/SM90/SM100/SM103/SM120). When provided,
        ``initial_state`` and ``output_state`` are treated as a state pool whose
        first dimension is indexed by these slot ids rather than laid out in
        sequence order: sequence ``i`` reads its initial state from row
        ``state_indices[i]`` and writes its final state back to the same row
        (in place when ``output_state is initial_state``), or to row
        ``output_state_indices[i]`` when that is given. This lets callers
        that keep a paged/indexed state pool avoid gathering the active rows
        into a packed buffer and scattering the result back. The pool may be
        any positive, non-overlapping rank-4 stride layout on SM100/SM103.
        ``None`` (default) keeps sequence-ordered row mapping without requiring
        the physical view itself to be contiguous.

        Without ``output_state_indices`` the ids **must be unique**: as with
        any indexed scatter, two sequences sharing a slot id would
        concurrently write the same pool row across work tiles, leaving that
        row's final state nondeterministic. Uniqueness is a caller
        precondition (not checked at launch, to avoid a per-call host sync);
        the caller's slot allocator is expected to guarantee it. With
        ``output_state_indices`` read ids may repeat or be ``-1``; only the
        rows a sequence writes must not be read or written by another
        sequence.
        ``backend="cudnn"`` requires frontend 1.31+, int32 indices and
        float32/bfloat16 pools with dense ``[H, V, K]`` rows and a 16-byte
        aligned slot stride. It supports identical input/output pool views
        or disjoint memory, and does not support checkpoints.
    _cp_chunk_len : int, optional
        Internal context-parallel chunk-length override used for testing and
        tuning. ``None`` lets the CP backend select the length automatically;
        an explicit value must be a multiple of 64.
    backend : {"auto", "flashinfer", "cake_gdn", "cudnn"}
        ``auto`` uses the native SM90/SM100/SM120 kernels and context-parallel
        routing, and may select cuDNN for measured SM100 raw-gate inference
        workloads. Automatic cuDNN selection requires both gate-fusion flags,
        BF16 inputs/gates, caller-normalized Q/K, separate compact float32
        input/output states with ``output_final_state=True``, and a caller-owned
        output. Explicit CP, pools and checkpoints retain native
        routing. Requires cuDNN frontend 1.31 GA and CuTe DSL 4.7 or newer.
        Cake kernels require an explicit ``cake_gdn``
        request. Use ``backend="cake_gdn", use_cp=True`` for Cake CP on
        SM100/SM103/SM107; ``use_cp=False`` retains Cake non-CP, and ``"auto"``
        enters Cake CP only inside the measured SM100/SM103 band (BF16/FP16
        I/O, FP32 initial and final state, gates, no checkpoints or state
        indices, ``4 * num_seqs * max(Hq, Hv) < SMs`` and ``max_seqlen >=
        4096``; without a ``max_seqlen`` hint the band estimates the longest
        sequence as ``ceil(total_seq_len / num_seqs)``) or the SM107 band.
        Explicit Cake requests fail for unsupported inputs without falling
        back to another backend. The Cake non-CP route performs no
        device-to-host synchronization and no eager warm-up call before CUDA
        Graph capture (the one-time kernel compile/load per variant is the
        only eager step); see ``cu_seqlens_cpu``.
        The explicit ``cake_gdn`` non-CP route serves ``head_size == 128``
        with BF16 or FP16 I/O (one dtype for Q/K/V/output, caller-normalized
        Q/K, FP32 gates). State dtypes: FP32 with either I/O dtype; BF16,
        FP16 and FP8 (e4m3/e5m2) with BF16 I/O, the low-precision three only
        with an initial and a final state on the DV-split schedule. The
        initial state may be zero (``None``) or given and the final state is
        optional; ``state_indices`` pools need both. Checkpoints are limited
        to the frozen FP16-I/O FP32-state packed contract without an initial
        state and to the SGLang TP4 (H=4, HV=8) BF16-I/O BF16-state indexed
        DV-split contract at 64-token intervals. The frozen
        ``(HEAD_GROUP_LOG2, IS_GQA, NUM_O_HEADS_LOG2)`` geometry set of each
        contract is enumerated by
        :func:`flashinfer.jit.cake_gdn.cake_gdn_prefill_geometries` from
        ``csrc/gdn/cake/manifest.json``; an unlisted geometry, schedule or
        architecture raises ``CakeGDNUnsupportedError``. The BF16-I/O
        FP32-state initial+final contract covers the Qwen3-Next / Qwen3.5
        per-rank geometries for TP1, TP2, TP4 and TP8 in both physical
        schedules (DV-split and full-DV).
        ``cudnn`` runs cuDNN's fused SM100 linear-attention engine through
        :func:`flashinfer.cudnn.cudnn_chunk_gated_delta_rule`.
    max_seqlen : int, optional
        Safe upper bound on the maximum logical sequence length, no larger
        than ``total_seq_len``. CP kernels use this host-side hint to
        bound their per-sequence launch grids without reading ``cu_seqlens``
        back from the GPU. When omitted, CP uses ``total_seq_len``, which is
        correct for any batch; passing the exact maximum of a batched call
        lets CP launch smaller grids.  The SM100/SM103 ``use_cp="auto"``
        band reads the hint (or the exact maximum from ``cu_seqlens_cpu``) as
        the longest sequence and, without either, the balanced-batch estimate
        ``ceil(total_seq_len / num_seqs)``; pass the exact maximum so the band
        decision matches the Cake dispatcher's. ``backend="cake_gdn"`` without
        CP uses the same bound (or the exact maximum from ``cu_seqlens_cpu``)
        to select its kernel variant; a loose bound is always correct and only
        affects whether the single-chunk kernel is eligible.  The persistent
        grid does not depend on the bound: it follows the per-sequence lengths
        of ``cu_seqlens_cpu`` when given (see below), else the residency rule.
    gate_domain : {"linear", "log"}
        Domain of precomputed ``g``: linear alpha (default) or natural-log
        alpha. ``None`` still means no decay. Ignored when use_gate_in_kernel
        is set, since that flag makes ``g`` a raw gate input.
    use_gate_in_kernel : bool
        Interpret ``g`` as raw logits and compute
        ``log(alpha) = -exp(A_log) * softplus(g + dt_bias)``. cuDNN fuses
        this transform; other backends materialize the same FP32 alpha.
    A_log, dt_bias : torch.Tensor, optional
        Per-head ``[num_sab_heads]`` parameters, required with
        use_gate_in_kernel. Float32, float16 or bfloat16, on q's device.
    beta_is_logit : bool
        Apply sigmoid to beta in FP32, round to beta's input dtype, then
        accumulate in FP32. cuDNN fuses the transform; other backends
        materialize it. The default accepts precomputed beta.
    cu_seqlens_cpu : Sequence[int] or torch.Tensor, optional
        Host copy of ``cu_seqlens``: a one-dimensional int32/int64 CPU tensor
        or a sequence of Python ints of length ``num_seqs + 1`` holding the
        same values (vLLM's ``query_start_loc_cpu``, the prefix sums of
        SGLang's ``extend_seq_lens_cpu``). Used by ``backend="cake_gdn"``
        without CP, and on SM8x by the ``use_cp="auto"`` and V32 rules, which
        need the exact longest sequence and keep the fused path without it. When given, the Cake
        route validates it on the host (non-decreasing, first entry ``0``,
        last entry ``total_seq_len``; a mismatch raises before any launch),
        takes the exact longest sequence as the bound for variant selection,
        checks the checkpoint count exactly, and enables the ragged-batch
        launch grid: from the per-sequence lengths the host models the
        persistent kernel's strided tile assignment against one CTA per tile
        and launches one CTA per tile when the modelled critical path shrinks
        by at least 4 % (same kernel binary, bitwise identical outputs);
        equal-length batches keep the residency grid.
        When omitted the route still reads nothing back from the device: the
        bound is ``max_seqlen`` (or ``total_seq_len``), the persistent grid
        follows the residency rule alone and ``cu_seqlens`` is
        resolved by the kernel at every launch, which clamps every offset
        into ``[0, total_seq_len]`` and treats a non-monotonic pair as an
        empty sequence -- a documented no-op, never an out-of-bounds access
        (``checkpoint_cu_starts`` consistency stays a caller precondition).
        Int32 ``cu_seqlens`` reaches the kernel as is; int64 is cast on the
        stream at every call, except on records that read int64 metadata
        directly (every Cake non-CP prefill record on CC 10.7), where int64
        reaches the kernel as is too. Nothing is cached by tensor identity, so
        a buffer rewritten in place is always read live, including on CUDA
        Graph replay.
    output_state_indices : torch.Tensor, optional
        Contiguous int32 or int64 tensor of shape ``[num_seqs]`` on the device
        of ``q``. Requires
        ``state_indices``, an explicit ``output_state`` pool and
        ``output_final_state=True``. Sequence ``i`` then reads its initial
        state from row ``state_indices[i]`` of ``initial_state`` and writes
        its final state to row ``output_state_indices[i]`` of
        ``output_state``, which may be the same pool. A read id of ``-1``
        starts the sequence from the zero state; a write id of ``-1`` skips
        its final-state write. A zero-length sequence writes nothing: its
        write row keeps its contents. Rows written by one sequence must not be
        read or written by another sequence of the same call.
    checkpoint_state_indices : torch.Tensor, optional
        Contiguous int32 or int64 destination rows for checkpoints, on the
        device of ``q``. Requires ``checkpoint_every_n_tokens > 0``.
        ``state_checkpoints`` is then a state pool ``[N_pool, num_sab_heads,
        head_size, head_size]`` whose rows are dense ``[H, V, K]`` blocks (the
        row stride may be padded), so a pool it shares with
        ``initial_state``/``output_state`` needs that layout too. The ``j``-th
        checkpoint of sequence ``i`` is written to row
        ``checkpoint_state_indices[checkpoint_cu_starts[i] + j]``; a row of
        ``-1`` skips that checkpoint. ``checkpoint_cu_starts`` (``[num_seqs]``
        or ``[num_seqs + 1]`` entries) only gives each sequence's first entry,
        so a paged caller can pass its page table directly (see
        :func:`flashinfer.gdn_paged.chunk_gated_delta_rule_paged`).
    prefix_lens : torch.Tensor, optional
        Contiguous int32 or int64 tensor of shape ``[num_seqs]`` on the device
        of ``q``: the number of tokens of each sequence processed before this
        call. Requires ``checkpoint_every_n_tokens > 0``. Checkpoints are then
        taken at absolute token positions: sequence ``i`` stores a checkpoint after
        its ``n``-th new token whenever ``prefix_lens[i] + n`` is a multiple
        of ``checkpoint_every_n_tokens`` (``1 <= n <= seq_len_i``). Prefix
        lengths need not be multiples of the 64-token chunk; a sequence whose
        prefix is not chunk-aligned is chunked from its aligned absolute
        position, so every checkpoint falls on a chunk boundary. Sequence
        ``i`` then has ``(prefix_lens[i] + seq_len_i) // N - prefix_lens[i] //
        N`` checkpoints (``N = checkpoint_every_n_tokens``); size
        ``checkpoint_cu_starts`` accordingly.

        ``output_state_indices``, ``checkpoint_state_indices`` and
        ``prefix_lens`` are served by the native non-CP kernels on SM8x, SM90,
        SM100/SM103/SM107 and SM120 (``backend="auto"`` or
        ``"flashinfer"``); ``use_cp=True``, ``backend="cake_gdn"`` and
        ``backend="cudnn"`` reject them.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        When ``output_final_state=False``, the output tensor of shape
        ``[total_seq_len, num_o_heads, head_size]``.  Otherwise a tuple
        ``(output, final_state)`` where ``final_state`` has shape
        ``[num_seqs, num_sab_heads, head_size, head_size]`` — or, when
        ``state_indices`` is given, the state pool ``[N_pool, ...]`` itself
        (i.e. ``output_state``), whose rows named by ``state_indices`` (by
        ``output_state_indices`` when given; ``-1`` rows are not written) now
        hold the updated final states.

    Notes
    -----
    - Supports GQA (``num_q_heads > num_k_heads = num_v_heads``) and GVA
      (``num_v_heads > num_q_heads = num_k_heads``).
    - The final state layout is ``[N, H, V, K]``.
    - Requires SM8x (Ampere), SM90 (Hopper), SM100 (Blackwell) or SM120. The
      SM100 path requires ``head_size == 128``. On SM100/SM103, ``gdn_cp``
      supports structurally legal equal-head, GQA, and GVA shapes on
      CUDA 12.8, CUDA 12.9, and CUDA 13; SM107 supports the same shapes with
      an ``nvcc`` that can emit ``compute_107a``.
      Other SM100 CP DSL routes require CUDA 13 and
      ``nvidia-cutlass-dsl[cu13]>=4.4.2`` (``pip install
      flashinfer-python[cu13]``).
    - On SM8x the state may not be FP8.
    - ``use_cp="auto"`` selects CP on compute capability 8.0 when
      ``cu_seqlens_cpu`` gives a longest sequence of at least 8192 tokens and
      ``num_seqs * max(num_q_heads, num_v_heads)`` is at most 16.
    """
    if backend not in ("auto", "flashinfer", "cake_gdn", "cudnn"):
        raise ValueError(f"unsupported GDN backend: {backend!r}")
    if backend != "cudnn" and (
        gate_domain != "linear"
        or use_gate_in_kernel
        or beta_is_logit
        or A_log is not None
        or dt_bias is not None
    ):
        from .gdn_kernels.gates import validate_gate_inputs

        validate_gate_inputs(
            q,
            v,
            g,
            beta,
            gate_domain,
            use_gate_in_kernel,
            A_log,
            dt_bias,
            beta_is_logit,
        )
    if (
        backend == "cake_gdn"
        and use_cp is not True
        and (not _CAKE_GDN_AVAILABLE or _cake_gdn is None)
    ):
        raise RuntimeError("the source-only Cake GDN backend is not installed")
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
    slot_args = _validate_slot_args(
        q=q,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        output_state=output_state,
        output_final_state=output_final_state,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        output_state_indices=output_state_indices,
        checkpoint_state_indices=checkpoint_state_indices,
        prefix_lens=prefix_lens,
        backend=backend,
        use_cp=use_cp,
        cp_chunk_len=_cp_chunk_len,
    )
    if slot_args:
        # The native non-CP kernels serve these arguments; never auto-route
        # the call to CP or cuDNN, which would drop them.
        use_cp = False

    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"
    if not is_integer_dtype(cu_seqlens.dtype):
        raise ValueError(
            f"cu_seqlens must have an integer dtype, got {cu_seqlens.dtype}"
        )

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    if num_seqs <= 0:
        raise ValueError("cu_seqlens must contain at least two entries")
    # Without a hint, only the packed length is a bound that holds for
    # every batch; a smaller one leaves the tail of the longest sequence
    # unprocessed.
    cp_max_seqlen = max_seqlen if max_seqlen is not None else total_seq_len
    if type(cp_max_seqlen) is not int or cp_max_seqlen < 0:
        raise ValueError("max_seqlen must be a nonnegative integer")
    if total_seq_len and cp_max_seqlen == 0:
        raise ValueError("max_seqlen must be positive when q is nonempty")
    minimum_cp_max_seqlen = (total_seq_len + num_seqs - 1) // num_seqs
    if cp_max_seqlen < minimum_cp_max_seqlen:
        raise ValueError(
            "max_seqlen cannot be smaller than ceil(total_seq_len / num_seqs)"
        )
    if cp_max_seqlen > total_seq_len:
        raise ValueError("max_seqlen cannot exceed total_seq_len")
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    cudnn_auto = (
        backend == "auto"
        and use_gate_in_kernel
        and beta_is_logit
        and not use_qk_l2norm_in_kernel
        and output_final_state
        and use_cp == "auto"
        and _cp_chunk_len is None
        and state_indices is None
        and checkpoint_every_n_tokens == 0
        and _prefer_cudnn_gdn_prefill(q, v, cu_seqlens)
        and _is_cudnn_gdn_auto_eligible(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            output_state,
            output,
            cu_seqlens,
        )
        and _cudnn_gdn_prefill_available()
    )
    if backend == "cudnn" or cudnn_auto:
        from .cudnn import cudnn_chunk_gated_delta_rule
        from .cudnn.linear_attention import _try_cudnn_auto

        unsupported = [
            name
            for name, requested in (
                ("use_cp", use_cp is True or _cp_chunk_len is not None),
                ("checkpoint_every_n_tokens", checkpoint_every_n_tokens > 0),
            )
            if requested
        ]
        if unsupported:
            raise NotImplementedError(
                'chunk_gated_delta_rule(backend="cudnn") does not support '
                + ", ".join(unsupported)
            )
        options = dict(
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output=output,
            output_state=output_state,
            state_indices=state_indices,
            gate_domain=gate_domain,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            beta_is_logit=beta_is_logit,
        )
        if cudnn_auto:
            # Cache the value FE sees, including scalar tensors accepted by the
            # existing adapter, rather than treating a scalar as operand layout.
            auto_scale = float(scale) if scale is not None else None
            result = _try_cudnn_auto(
                cudnn_chunk_gated_delta_rule, q, k, v, g, beta, auto_scale, **options
            )
            if result is not None:
                return result
        else:
            return cudnn_chunk_gated_delta_rule(q, k, v, g, beta, scale, **options)

    if gate_domain != "linear" or use_gate_in_kernel or beta_is_logit:
        from .gdn_kernels.gates import materialize_gates

        g, beta = materialize_gates(
            g, beta, gate_domain, use_gate_in_kernel, A_log, dt_bias, beta_is_logit
        )
    if prefix_lens is not None and beta is None:
        # Chunk-aligning an unaligned prefix pads the first chunk with identity
        # rows, which the kernels build from a zero beta.
        beta = torch.ones(
            (q.size(0), max(q.size(1), v.size(1))),
            dtype=torch.float32,
            device=q.device,
        )

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
        # With checkpoint_state_indices only each sequence's first entry is
        # read, so a paged caller may pass one start per sequence.
        cu_starts_lengths = (
            (num_seqs + 1,)
            if checkpoint_state_indices is None
            else (num_seqs, num_seqs + 1)
        )
        if checkpoint_cu_starts.size(0) not in cu_starts_lengths:
            expected = " or ".join(str(n) for n in cu_starts_lengths)
            raise ValueError(
                f"checkpoint_cu_starts must have {expected} elements, "
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
        # A strided state_checkpoints would be written through a copy and lost.
        # A checkpoint pool (checkpoint_state_indices) may pad its rows like
        # the state pools, but each row must be one dense [H, V, K] block.
        if not checkpoint_cu_starts.is_contiguous():
            raise RuntimeError("checkpoint_cu_starts must be contiguous")
        if checkpoint_state_indices is None:
            if not state_checkpoints.is_contiguous():
                raise RuntimeError("state_checkpoints must be contiguous")
        else:
            row_size = num_sab_heads * head_size * head_size
            dense_rows = state_checkpoints.stride()[1:] == (
                head_size * head_size,
                head_size,
                1,
            )
            if not dense_rows or state_checkpoints.stride(0) < row_size:
                raise RuntimeError(
                    "state_checkpoints pool rows must be dense [H, V, K] blocks "
                    "with a row stride of at least H * V * K elements"
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
    _cuda_version_parts = torch.version.cuda.split(".") if torch.version.cuda else []
    _cuda_version = (
        (int(_cuda_version_parts[0]), int(_cuda_version_parts[1]))
        if len(_cuda_version_parts) >= 2
        else (0, 0)
    )
    _cuda_major = _cuda_version[0]
    _device_capability = get_compute_capability(device)
    _arch_major = _device_capability[0]
    _device_name = get_device_name(device)
    _parallel_work = num_seqs * num_sab_heads
    # SM8x routes on the exact longest sequence; max_seqlen may be an upper bound.
    sm8x_max_seqlen = None
    if _arch_major == 8 and cu_seqlens_cpu is not None:
        offsets = _cake_gdn_prefill_host_offsets(
            cu_seqlens_cpu, num_seqs=num_seqs, total_tokens=total_seq_len
        )
        sm8x_max_seqlen = max(b - a for a, b in zip(offsets, offsets[1:], strict=False))
        if cp_max_seqlen < sm8x_max_seqlen:
            raise ValueError(
                "max_seqlen is smaller than the longest sequence in cu_seqlens_cpu"
            )
    cp_heuristic_matches = (
        _arch_major in (9, 10, 12)
        and should_use_cp_host(
            _parallel_work,
            _sm_count,
            _device_name,
            device_capability=_device_capability,
        )
    ) or should_use_cp_sm80_host(
        sm8x_max_seqlen,
        _parallel_work,
        _device_capability,
    )
    cake_cp_domain_matches = bool(
        backend == "cake_gdn"
        and _device_capability in _CAKE_GDN_CP_CAPABILITIES
        and _use_gdn_cp_sm100(
            initial_state=initial_state,
            output_state=output_state,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            cp_chunk_len=_cp_chunk_len,
        )
        and _is_gdn_cp_sm100_supported_shape(q, k, v)
    )
    # SM107 only: ``use_cp="auto"`` enters Cake CP inside an explicit shape
    # band (fp32 SIMT fixup route).  SM100/SM103 enter Cake CP through the
    # measured FP32-state band below (``_cake_gdn_cp_sm100_band``) or with
    # ``use_cp=True``.
    cake_cp_sm107a_band = (
        cake_cp_domain_matches
        and use_cp == "auto"
        and _device_capability == (10, 7)
        and num_seqs * num_sab_heads <= _CAKE_GDN_CP_SM107A_MAX_PARALLEL_STATES
        and cp_max_seqlen >= _CAKE_GDN_CP_SM107A_MIN_MAX_SEQLEN
        and _cake_gdn_cp_sm107a_plan_pays(
            num_seqs=num_seqs,
            num_sab_heads=num_sab_heads,
            max_seqlen=cp_max_seqlen,
            total_tokens=total_seq_len,
            num_sms=_sm_count,
        )
    )
    # SM100/SM103: ``use_cp="auto"`` enters Cake CP inside the measured band
    # (long few-sequence FP32-state rows); every other row keeps Cake non-CP.
    cake_cp_sm100_band = (
        cake_cp_domain_matches
        and use_cp == "auto"
        and _cake_gdn_cp_sm100_band(
            device_capability=_device_capability,
            num_seqs=num_seqs,
            num_sab_heads=num_sab_heads,
            max_seqlen=_cake_gdn_cp_band_max_seqlen(
                max_seqlen, total_seq_len, num_seqs, cu_seqlens_cpu=cu_seqlens_cpu
            ),
            num_sms=_sm_count,
            io_dtype=q.dtype,
            initial_state=initial_state,
            output_state=output_state,
            output_final_state=output_final_state,
            g=g,
            beta=beta,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            state_indices=state_indices,
        )
    )
    will_use_cp = (
        use_cp is True
        or (backend != "cake_gdn" and use_cp == "auto" and cp_heuristic_matches)
        or cake_cp_sm107a_band
        or cake_cp_sm100_band
    )
    use_gdn_cp_backend = bool(
        backend == "cake_gdn" and will_use_cp and cake_cp_domain_matches
    )
    if backend == "cake_gdn" and will_use_cp and not use_gdn_cp_backend:
        raise ValueError(
            "Cake GDN CP requires SM100/SM103/SM107 and supported head, state and "
            "checkpoint configurations"
        )
    # The SM80 entry checks the rest of the V32 contract (dtypes, state, no checkpoints).
    offer_v32 = (
        backend != "cake_gdn"
        and not will_use_cp
        and not slot_args
        and should_use_v32_sm80_host(
            sm8x_max_seqlen,
            num_seqs,
            num_q_heads,
            num_v_heads,
            _device_capability,
        )
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
        if _arch_major not in (8, 9, 10, 12):
            raise NotImplementedError(
                "state_indices is only supported on the SM80/SM90/SM100/SM103/SM120 "
                f"GDN prefill kernels; got compute-capability major {_arch_major}, "
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
        # No pool to read from or write to: the slot ids would only index the
        # compact scratch state the SM90/SM120 kernels still store into.
        if initial_state is None and output_state is None:
            state_indices = None
        # Opt-in slot bounds check (FLASHINFER_CAKE_GDN_VALIDATE_SLOTS=1), as Cake does.
        indexed_pools = tuple(
            tensor for tensor in (initial_state, output_state) if tensor is not None
        )
        if output_state_indices is not None:
            # Separate rows: -1 reads the zero state / skips the final write.
            if initial_state is not None:
                _cake_gdn_assert_state_slots(
                    state_indices,
                    int(initial_state.shape[0]),
                    name="GDN prefill state_indices",
                    allow_minus_one=True,
                )
            _cake_gdn_assert_state_slots(
                output_state_indices,
                int(output_state.shape[0]),
                name="GDN prefill output_state_indices",
                allow_minus_one=True,
            )
        elif indexed_pools and (backend != "cake_gdn" or will_use_cp):
            _cake_gdn_assert_state_slots(
                state_indices,
                min(int(tensor.shape[0]) for tensor in indexed_pools),
                name="GDN prefill state_indices",
                allow_minus_one=False,
            )
    if checkpoint_state_indices is not None and state_checkpoints is not None:
        _cake_gdn_assert_state_slots(
            checkpoint_state_indices,
            int(state_checkpoints.shape[0]),
            name="GDN prefill checkpoint_state_indices",
            allow_minus_one=True,
        )

    if (
        use_qk_l2norm_in_kernel
        and _arch_major in (8, 9, 10, 12)
        and (_arch_major != 10 or _cuda_major >= 13)
        and is_cute_dsl_arch_supported(*_device_capability)
    ):
        # The chunked backends consume caller-normalized operands. Honor the
        # public flag before dispatch; unnormalized keys can make the delta
        # recurrence expansive and produce nonfinite outputs. Unsupported
        # configurations must reach the backend checks without compiling here.
        from .gdn_kernels.qk_l2norm import normalize_qk

        q, k = normalize_qk(q, k)
        use_qk_l2norm_in_kernel = False

    if will_use_cp:
        cp_rejection_reason = _cp_delta_rule_rejection_reason(
            arch_major=_arch_major,
            use_gdn_cp_backend=use_gdn_cp_backend,
            cuda_version=_cuda_version,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            output=output,
            initial_state=initial_state,
            output_state=output_state,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            state_indices=state_indices,
            cp_chunk_len=_cp_chunk_len,
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
            if output_final_state and output_state is None and _arch_major == 8:
                output_state = _allocate_sm8x_output_state(
                    num_seqs,
                    num_sab_heads,
                    head_size,
                    device,
                    initial_state,
                    state_indices,
                )
            if output_final_state and output_state is None:
                output_state = torch.empty(
                    (num_seqs, num_sab_heads, head_size, head_size),
                    dtype=torch.float32,
                    device=device,
                )
            if use_gdn_cp_backend:
                gdn_cp_backend = cast(
                    Callable[..., None], _chunk_gated_delta_rule_gdn_cp_sm100
                )
                gdn_cp_backend(
                    output,
                    output_state,
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu_seqlens,
                    _scale,
                    initial_state=initial_state,
                    state_indices=state_indices,
                    state_checkpoints=state_checkpoints,
                    checkpoint_cu_starts=checkpoint_cu_starts,
                    checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                    cp_chunk_len=_cp_chunk_len,
                    max_seqlen=cp_max_seqlen,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )
                if output_final_state:
                    return output, output_state
                return output
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
                    8: cp_delta_rule_dsl_sm80,
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
                if _arch_major in (8, 9, 10, 12)
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
                max_seqlen=cp_max_seqlen,
                cp_chunk_len=_cp_chunk_len,
                **state_indices_kwargs,
                **checkpoint_kwargs,
            )
            if output_final_state:
                return output, output_state
            return output
    # Compiled Cake specializations cover more shapes than have been qualified
    # against the existing prefill kernels. Keep them opt-in until automatic
    # dispatch has a performance-qualified domain.
    if backend == "cake_gdn":
        return _run_cake_gdn_prefill(
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
            max_seqlen=cp_max_seqlen,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

    slot_kwargs = (
        {
            "output_state_indices": output_state_indices,
            "checkpoint_state_indices": checkpoint_state_indices,
            "prefix_lens": prefix_lens,
        }
        if slot_args
        else {}
    )
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
            **slot_kwargs,
        )
    elif _arch_major == 8:
        # SM80 Ampere path (CuTe DSL kernel)
        if chunk_gated_delta_rule_sm80 is None:
            raise NotImplementedError("SM80 GDN prefill DSL kernel is unavailable")
        if output_state is None:
            output_state = _allocate_sm8x_output_state(
                num_seqs,
                num_sab_heads,
                head_size,
                device,
                initial_state,
                state_indices,
                returned=output_final_state,
            )
        chunk_gated_delta_rule_sm80(
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
            _v32=offer_v32,
            **slot_kwargs,
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
            **slot_kwargs,
        )
    elif _arch_major == 9:
        # SM90 Hopper path (CuTe DSL kernel)
        if chunk_gated_delta_rule_sm90 is None:
            raise NotImplementedError("SM90 GDN prefill DSL kernel is unavailable")

        # The kernel always stores the final state, to
        # output_state[state_indices[i]] when indexed, so scratch for a
        # read-only pool must span the pool, as on SM120.
        if output_state is None:
            output_state = torch.empty(
                (
                    initial_state.shape
                    if state_indices is not None and initial_state is not None
                    else (num_seqs, num_sab_heads, head_size, head_size)
                ),
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
            **slot_kwargs,
        )
    else:
        raise NotImplementedError("GDN prefill DSL kernel is unavailable")

    if output_final_state:
        return output, output_state
    else:
        return output
