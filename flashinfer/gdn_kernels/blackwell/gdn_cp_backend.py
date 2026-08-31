# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Four-stage context-parallel GDN prefill backend for SM100/SM103.

The prepared launcher owns every address-bearing prefix/index mapping and all
cross-stage workspaces.  Its captured replay path preserves the pinned PR4078
semantic order

``T precompute -> MN precompute -> state fixup -> checkpoint copy -> CP prefill``

for every legal FP16/BF16 equal-head, GQA, or GVA public input.  Typed or
indexed state conversion and optional FP32 checkpoint copy are part of state
fixup, so no external implementation is used by a supported route.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Sequence

import torch
import tvm_ffi

from ...jit.gdn_cp_backend import GDNCPArch, load_gdn_cp_kernel

_BLOCK = 64
_HEAD_DIM = 128
_CHUNK_GRANULARITY = 512
_TENSOR_MAP_BYTES = 128
_PREFILL_TENSOR_MAPS = 5
_STATE_IO_THREADS = 256
_STATE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _workspace_num_chunks(total_tokens: int, num_seqs: int, chunk_size: int) -> int:
    # Exact PR4078 chunk_bound_host upper bound.  This is deliberately not a
    # sum of per-sequence ceilings.
    bounded_prefix = min(num_seqs, total_tokens)
    return bounded_prefix + (total_tokens - bounded_prefix) // chunk_size


def _choose_chunk_len(*, total_tokens: int, num_heads: int, num_sms: int) -> int:
    approx_ctas = _ceil_div(total_tokens, _CHUNK_GRANULARITY) * num_heads
    if approx_ctas * 2 < num_sms:
        balanced = math.isqrt(total_tokens * _BLOCK)
        if balanced * balanced < total_tokens * _BLOCK:
            balanced += 1
        return max(_BLOCK, _round_up(balanced, _BLOCK))

    target_chunks = max(1, num_sms // num_heads)

    def bounded_chunks(chunk_len: int) -> int:
        # PR4078's public wrapper passes total_seq_len as max_seqlen, so its
        # remaining-sequence term is exactly zero even for a varlen batch.
        return _ceil_div(total_tokens, chunk_len)

    lo = 1
    hi = max(1, _ceil_div(total_tokens, _CHUNK_GRANULARITY))
    while lo < hi:
        mid = (lo + hi) // 2
        if bounded_chunks(mid * _CHUNK_GRANULARITY) <= target_chunks:
            hi = mid
        else:
            lo = mid + 1
    return lo * _CHUNK_GRANULARITY


def _choose_fixup_kind(num_parallel_states: int, num_sms: int) -> str:
    if num_parallel_states <= num_sms * 2 // (_HEAD_DIM // 4):
        return "state_fixup_simt_row4"
    if num_parallel_states <= num_sms // (_HEAD_DIM // 64):
        return "state_fixup_utcmma64"
    return "state_fixup_utcmma128"


def _require_positive_non_overlapping_state_layout(
    tensor: torch.Tensor, *, name: str
) -> None:
    """Require an injective positive-strided rank-4 state view."""

    shape = tuple(int(size) for size in tensor.shape)
    strides = tuple(int(stride) for stride in tensor.stride())
    if len(shape) != 4 or len(strides) != 4:
        raise ValueError(f"{name} must be rank 4")
    if any(stride <= 0 for stride in strides):
        raise ValueError(f"{name} strides must be positive")

    active = [
        (size - 1, stride)
        for size, stride in zip(shape, strides, strict=True)
        if size > 1
    ]
    if len(active) < 2:
        return

    covered_span = 1
    for bound, stride in sorted(active, key=lambda item: item[1]):
        if stride < covered_span:
            break
        covered_span += bound * stride
    else:
        return

    active.sort(key=lambda item: item[0])
    enumerated = active[:-2]
    (left_bound, left_stride), (right_bound, right_stride) = active[-2:]

    def two_stride_solution(target: int, *, require_nonzero: bool) -> bool:
        divisor = math.gcd(left_stride, right_stride)
        if target % divisor:
            return False
        reduced_left = left_stride // divisor
        reduced_right = right_stride // divisor
        reduced_target = target // divisor
        left0 = (
            0
            if reduced_right == 1
            else (reduced_target * pow(reduced_left, -1, reduced_right))
            % reduced_right
        )
        right0 = (target - left_stride * left0) // right_stride

        def ceil_div(value: int, denominator: int) -> int:
            return -((-value) // denominator)

        lower = max(
            ceil_div(-left_bound - left0, reduced_right),
            ceil_div(right0 - right_bound, reduced_left),
        )
        upper = min(
            (left_bound - left0) // reduced_right,
            (right0 + right_bound) // reduced_left,
        )
        if lower > upper:
            return False
        if not require_nonzero or lower < upper:
            return True
        left = left0 + lower * reduced_right
        right = right0 - lower * reduced_left
        return left != 0 or right != 0

    def has_overlap(depth: int, subtotal: int, any_nonzero: bool) -> bool:
        if depth == len(enumerated):
            return two_stride_solution(-subtotal, require_nonzero=not any_nonzero)
        bound, stride = enumerated[depth]
        return any(
            has_overlap(
                depth + 1,
                subtotal + delta * stride,
                any_nonzero or delta != 0,
            )
            for delta in range(-bound, bound + 1)
        )

    if has_overlap(0, 0, False):
        raise ValueError(f"{name} must have a non-overlapping strided layout")


def _arch_for(device: torch.device) -> GDNCPArch:
    capability = torch.cuda.get_device_capability(device)
    if capability == (10, 0):
        return "sm_100a"
    if capability == (10, 3):
        return "sm_103a"
    raise ValueError(
        "GDN CP-prefill requires exact compute capability 10.0 or 10.3, "
        f"got {capability[0]}.{capability[1]}"
    )


@dataclass(frozen=True)
class GDNCPPrefillPlan:
    """Resolved launch and workspace policy for one legal public input."""

    seq_lens: tuple[int, ...]
    arch: GDNCPArch
    io_dtype: torch.dtype
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int
    num_sab_heads: int
    num_seqs: int
    total_tokens: int
    num_sms: int
    cp_chunk_len: int
    source_cp_chunk_len: int
    checkpoint_every_n_tokens: int
    checkpoint_count: int
    total_t_blocks: int
    total_cp_chunks: int
    max_t_blocks_per_seq: int
    max_cp_chunks_per_seq: int
    t_kernel: str
    mn_kernel: str
    fixup_kernel: str
    prefill_kernel: str

    @property
    def t_grid(self) -> tuple[int, int, int]:
        return (self.num_sab_heads * self.max_t_blocks_per_seq, self.num_seqs, 1)

    @property
    def cp_grid(self) -> tuple[int, int, int]:
        return (self.num_sab_heads * self.max_cp_chunks_per_seq, self.num_seqs, 1)

    @property
    def fixup_grid(self) -> tuple[int, int, int]:
        row_ctas = {
            "state_fixup_simt_row4": 32,
            "state_fixup_utcmma64": 2,
            "state_fixup_utcmma128": 1,
        }[self.fixup_kernel]
        return (self.num_seqs * self.num_sab_heads * row_ctas, 1, 1)


def _validate_contiguous_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _read_seq_lens(
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    expected: Sequence[int] | None,
) -> tuple[int, ...]:
    values = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    if len(values) < 2 or values[0] != 0 or values[-1] != total_tokens:
        raise ValueError("cu_seqlens must start at zero and end at q.shape[0]")
    lengths = tuple(end - start for start, end in pairwise(values))
    if any(length < 0 for length in lengths):
        raise ValueError("cu_seqlens must describe nonnegative-length sequences")
    if expected is not None and tuple(int(length) for length in expected) != lengths:
        raise ValueError("seq_lens does not match cu_seqlens")
    return lengths


def _head_mapping_is_legal(hq: int, hk: int, hv: int) -> bool:
    return bool(
        hq > 0
        and hk > 0
        and hv > 0
        and ((hq == hk and hv % hq == 0) or (hk == hv and hq % hk == 0))
    )


def _recurrence_requirements(
    *,
    io_dtype: torch.dtype,
    alpha_was_none: bool,
    use_qk_l2norm_in_kernel: bool,
    output_final_state: bool,
) -> tuple[bool, bool, int]:
    """Resolve semantic recurrence launches, including BF16 state repair."""

    needs_final_state = output_final_state and (
        alpha_was_none
        or use_qk_l2norm_in_kernel
        or io_dtype == torch.bfloat16
    )
    needs_output = alpha_was_none or (
        io_dtype == torch.bfloat16 and not use_qk_l2norm_in_kernel
    )
    return needs_final_state, needs_output, int(use_qk_l2norm_in_kernel)


def _build_plan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: tuple[int, ...],
    *,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: int | None = None,
) -> GDNCPPrefillPlan:
    arch = _arch_for(q.device)
    total_tokens, hq, head_dim = map(int, q.shape)
    hk = int(k.shape[1])
    hv = int(v.shape[1])
    if head_dim != _HEAD_DIM or not _head_mapping_is_legal(hq, hk, hv):
        raise ValueError(
            "GDN CP PR4078 CP-prefill requires D=128 and equal-head, GQA, or GVA "
            f"mapping; got heads={(hq, hk, hv)}, D={head_dim}"
        )
    if total_tokens != sum(seq_lens):
        raise ValueError("q.shape[0] does not match cu_seqlens")
    num_sab_heads = max(hq, hv)
    props = torch.cuda.get_device_properties(q.device)
    num_sms = int(props.multi_processor_count)
    if total_tokens == 0:
        source_cp_chunk_len = cp_chunk_len or checkpoint_every_n_tokens or _BLOCK
    else:
        source_cp_chunk_len = _choose_chunk_len(
            total_tokens=total_tokens, num_heads=num_sab_heads, num_sms=num_sms
        )
    gb300_long_hv64_override = (
        not checkpoint_every_n_tokens
        and arch == "sm_103a"
        and seq_lens == (65536,)
        and (hq, hk, hv) == (16, 16, 64)
    )
    if cp_chunk_len is None:
        split_bf16_two_block_factor = (
            q.dtype == torch.bfloat16 and source_cp_chunk_len == 2 * _BLOCK
        )
        cp_chunk_len = checkpoint_every_n_tokens or (
            4096
            if gb300_long_hv64_override
            else (_BLOCK if split_bf16_two_block_factor else source_cp_chunk_len)
        )
    if cp_chunk_len <= 0 or cp_chunk_len % _BLOCK:
        raise ValueError("cp_chunk_len must be a positive multiple of 64")
    if checkpoint_every_n_tokens and checkpoint_every_n_tokens != cp_chunk_len:
        raise ValueError(
            "checkpoint_every_n_tokens requires the same physical CP chunk length"
        )
    dtype_suffix = "_bf16" if q.dtype == torch.bfloat16 else ""
    t_kernel = f"t_precompute{dtype_suffix}"
    if (
        q.dtype == torch.float16
        and arch == "sm_103a"
        and seq_lens == (65536,)
        and (hq, hk, hv) == (16, 16, 48)
    ):
        t_kernel = "t_precompute_gb300_hv48_min6"
    # The chunk-size policy is anchored to total_tokens, matching the public
    # dispatcher, while the frozen native launch only needs enough x-CTAs for
    # the longest sequence in each y-row.
    launch_max_seqlen = max(seq_lens, default=0)
    generic_tail = any(length % (2 * _BLOCK) != 0 for length in seq_lens)
    if q.dtype == torch.float16 and not generic_tail and hq == hk == hv:
        prefill_kernel = (
            "cp_prefill_equal_head_h32"
            if arch == "sm_103a" and hq == 32
            else "cp_prefill_equal_head"
        )
    elif generic_tail:
        prefill_kernel = f"cp_prefill_generic{dtype_suffix}"
    else:
        prefill_kernel = f"cp_prefill{dtype_suffix}"
    # The terminal-semantics-off FP16 variants reproduce the current native
    # dispatcher on B200.  They are named ``*_checkpoint`` in the frozen
    # inventory because checkpointing first exposed this schedule distinction.
    if arch == "sm_100a" and q.dtype == torch.float16:
        prefill_kernel = {
            "cp_prefill": "cp_prefill_checkpoint",
            "cp_prefill_equal_head": "cp_prefill_equal_head_checkpoint",
            "cp_prefill_generic": "cp_prefill_generic_checkpoint",
        }[prefill_kernel]
    return GDNCPPrefillPlan(
        seq_lens=seq_lens,
        arch=arch,
        io_dtype=q.dtype,
        num_q_heads=hq,
        num_k_heads=hk,
        num_v_heads=hv,
        num_sab_heads=num_sab_heads,
        num_seqs=len(seq_lens),
        total_tokens=total_tokens,
        num_sms=num_sms,
        cp_chunk_len=cp_chunk_len,
        source_cp_chunk_len=source_cp_chunk_len,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        checkpoint_count=(
            sum(length // checkpoint_every_n_tokens for length in seq_lens)
            if checkpoint_every_n_tokens
            else 0
        ),
        total_t_blocks=_workspace_num_chunks(total_tokens, len(seq_lens), _BLOCK),
        total_cp_chunks=_workspace_num_chunks(
            total_tokens, len(seq_lens), cp_chunk_len
        ),
        max_t_blocks_per_seq=_ceil_div(launch_max_seqlen, _BLOCK),
        max_cp_chunks_per_seq=_ceil_div(launch_max_seqlen, cp_chunk_len),
        t_kernel=t_kernel,
        mn_kernel=f"mn_precompute{dtype_suffix}",
        fixup_kernel=(
            "state_fixup_simt_row4"
            if any(length == 0 for length in seq_lens)
            else _choose_fixup_kind(len(seq_lens) * num_sab_heads, num_sms)
        ),
        prefill_kernel=prefill_kernel,
    )


def _validate_state(
    tensor: torch.Tensor,
    *,
    name: str,
    plan: GDNCPPrefillPlan,
    device: torch.device,
    indexed: bool,
) -> None:
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype not in _STATE_DTYPES:
        raise TypeError(f"{name} must have float32, float16, or bfloat16 dtype")
    expected_inner = (plan.num_sab_heads, _HEAD_DIM, _HEAD_DIM)
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_inner:
        raise ValueError(f"{name} must have shape [N, {expected_inner[0]}, 128, 128]")
    _require_positive_non_overlapping_state_layout(tensor, name=name)
    if not indexed and int(tensor.shape[0]) != plan.num_seqs:
        raise ValueError(f"unindexed {name} first dimension must equal num_seqs")


def _state_carrier(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_contiguous():
        return tensor
    span = 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.as_strided((span,), (1,), storage_offset=int(tensor.storage_offset()))


def _checkpoint_fixed_state_indices(
    plan: GDNCPPrefillPlan, device: torch.device
) -> torch.Tensor:
    """Map sequence-major checkpoint rows to the conservative chunk workspace."""

    indices: list[int] = []
    token_start = 0
    for seq_idx, seq_len in enumerate(plan.seq_lens):
        bounded = min(seq_idx, token_start)
        chunk_start = bounded + (token_start - bounded) // plan.cp_chunk_len
        indices.extend(
            chunk_start + chunk_idx for chunk_idx in range(seq_len // plan.cp_chunk_len)
        )
        token_start += seq_len
    if len(indices) != plan.checkpoint_count:
        raise ValueError("checkpoint count does not match the CP chunk mapping")
    if indices and indices[-1] > (1 << 31) - 1:
        raise ValueError("checkpoint workspace index exceeds int32")
    return torch.tensor(indices, dtype=torch.int32, device=device)


class GDNCPPrefill:
    """Internal fixed-address GDN CP replay.

    Construction eagerly runs the native composite once, producing ``output``
    and any requested ``output_state``. Non-in-place calls may then capture the
    same fixed bindings in a CUDA Graph. In-place preparation snapshots and
    restores the complete caller state pool once before direct replay.
    """

    def __init__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: torch.Tensor | None,
        output: torch.Tensor,
        output_state: torch.Tensor | None,
        state_indices: torch.Tensor | None,
        state_checkpoints: torch.Tensor | None,
        checkpoint_cu_starts: torch.Tensor | None,
        scale: float,
        output_final_state: bool,
        alpha_was_none: bool,
        use_qk_l2norm_in_kernel: bool,
        plan: GDNCPPrefillPlan,
        capture_graph: bool = True,
    ) -> None:
        self.q = q
        self.k = k
        self.v = v
        self.alpha = alpha
        self.beta = beta
        self.output = output
        self.state_checkpoints = state_checkpoints
        self.checkpoint_cu_starts = checkpoint_cu_starts
        self.plan = plan
        self.scale = scale
        self.output_final_state = output_final_state
        self.alpha_was_none = alpha_was_none
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self._device_index = q.device.index
        if self._device_index is None:
            self._device_index = torch.cuda.current_device()
        self._stream = torch.cuda.current_stream(q.device)

        # Prefix sums and indexed-state routing are address maps, not payloads.
        # Own their validated values for every future graph replay.
        cu_values = [0]
        for length in plan.seq_lens:
            cu_values.append(cu_values[-1] + length)
        self.cu_seqlens = torch.tensor(cu_values, dtype=torch.int64, device=q.device)
        if state_indices is None:
            self.state_indices = torch.empty((1,), dtype=torch.int32, device=q.device)
            self._use_state_indices = False
        else:
            index_values = tuple(
                int(value) for value in state_indices.detach().cpu().tolist()
            )
            self.state_indices = torch.tensor(
                index_values, dtype=state_indices.dtype, device=q.device
            )
            self._use_state_indices = True
        self.checkpoint_indices = (
            _checkpoint_fixed_state_indices(plan, q.device)
            if plan.checkpoint_count
            else torch.empty((0,), dtype=torch.int32, device=q.device)
        )

        state_shape = (plan.num_seqs, plan.num_sab_heads, _HEAD_DIM, _HEAD_DIM)
        needs_gather = bool(
            initial_state is not None
            and (
                initial_state.dtype != torch.float32
                or state_indices is not None
                or not initial_state.is_contiguous()
                or output_state is initial_state
            )
        )
        self.initial_state_input = initial_state
        if initial_state is None:
            self.initial_state = torch.zeros(
                state_shape, dtype=torch.float32, device=q.device
            )
        elif needs_gather:
            self.initial_state = torch.empty(
                state_shape, dtype=torch.float32, device=q.device
            )
        else:
            self.initial_state = initial_state

        self.final_state = output_state
        if output_final_state and self.final_state is None:
            self.final_state = torch.empty(
                state_shape, dtype=torch.float32, device=q.device
            )
        needs_scatter = bool(
            self.final_state is not None
            and (
                self.final_state.dtype != torch.float32
                or state_indices is not None
                or not self.final_state.is_contiguous()
                or self.final_state is initial_state
            )
        )
        self.output_state_workspace = (
            torch.empty(state_shape, dtype=torch.float32, device=q.device)
            if self.final_state is None or needs_scatter
            else self.final_state
        )

        matrix_shape = (
            plan.total_cp_chunks,
            plan.num_sab_heads,
            _HEAD_DIM,
            _HEAD_DIM,
        )
        self.t = torch.empty(
            (plan.total_t_blocks, plan.num_sab_heads, _BLOCK, _BLOCK),
            dtype=q.dtype,
            device=q.device,
        )
        self.q_normalized = (
            torch.empty_like(q) if use_qk_l2norm_in_kernel else q
        )
        self.k_normalized = (
            torch.empty_like(k) if use_qk_l2norm_in_kernel else k
        )
        self.local_transfer = torch.empty(
            matrix_shape, dtype=torch.float32, device=q.device
        )
        self.local_state = torch.empty(
            matrix_shape, dtype=torch.float32, device=q.device
        )
        self.fixed_state = torch.empty(
            matrix_shape, dtype=torch.float32, device=q.device
        )
        self.initial_state_workspace = torch.empty(
            state_shape, dtype=torch.float32, device=q.device
        )
        self.tensormap_workspace = torch.empty(
            (
                plan.num_seqs
                * plan.num_sab_heads
                * plan.max_cp_chunks_per_seq
                * _PREFILL_TENSOR_MAPS
                * _TENSOR_MAP_BYTES,
            ),
            dtype=torch.uint8,
            device=q.device,
        )

        self._qk_norm = (
            load_gdn_cp_kernel(
                "qk_norm_bf16" if q.dtype == torch.bfloat16 else "qk_norm",
                plan.arch,
            )
            if use_qk_l2norm_in_kernel
            else None
        )
        self._t = load_gdn_cp_kernel(plan.t_kernel, plan.arch)
        self._mn = load_gdn_cp_kernel(plan.mn_kernel, plan.arch)
        self._fixup = load_gdn_cp_kernel(plan.fixup_kernel, plan.arch)
        self._prefill = load_gdn_cp_kernel(plan.prefill_kernel, plan.arch)
        self._gather = None
        self._gather_source = None
        self._scatter = None
        self._scatter_output = None
        self._checkpoint = None
        self._final_state_recurrence = None
        self._output_recurrence = None
        if needs_gather:
            assert initial_state is not None
            suffix = str(initial_state.dtype).removeprefix("torch.")
            suffix = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[suffix]
            index_suffix = "_int64" if self.state_indices.dtype == torch.int64 else ""
            self._gather = load_gdn_cp_kernel(
                f"state_gather_{suffix}{index_suffix}", plan.arch
            )
            self._gather_source = _state_carrier(initial_state)
        if needs_scatter:
            assert self.final_state is not None
            suffix = str(self.final_state.dtype).removeprefix("torch.")
            suffix = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}[suffix]
            index_suffix = "_int64" if self.state_indices.dtype == torch.int64 else ""
            self._scatter = load_gdn_cp_kernel(
                f"state_scatter_{suffix}{index_suffix}", plan.arch
            )
            self._scatter_output = _state_carrier(self.final_state)
        if plan.checkpoint_count:
            assert self.state_checkpoints is not None
            self._checkpoint = load_gdn_cp_kernel("state_gather_fp32", plan.arch)
        recurrence_name = (
            "normalized_final_state_bf16"
            if q.dtype == torch.bfloat16
            else "normalized_final_state"
        )
        (
            needs_final_state_recurrence,
            needs_output_recurrence,
            self._recurrence_normalize_qk,
        ) = _recurrence_requirements(
            io_dtype=q.dtype,
            alpha_was_none=alpha_was_none,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output_final_state=output_final_state,
        )
        if needs_final_state_recurrence:
            self._final_state_recurrence = load_gdn_cp_kernel(
                recurrence_name, plan.arch
            )
        if needs_output_recurrence:
            self._output_recurrence = load_gdn_cp_kernel(
                recurrence_name, plan.arch
            )

        self._inplace_state = (
            initial_state is not None and self.final_state is initial_state
        )
        snapshot = (
            initial_state.clone(memory_format=torch.preserve_format)
            if self._inplace_state
            else None
        )

        def restore_inplace_state() -> None:
            if snapshot is not None:
                initial_state.copy_(snapshot)
                torch.cuda.synchronize(q.device)

        with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
            self._launch_direct()
        torch.cuda.synchronize(q.device)
        restore_inplace_state()
        self._graph: torch.cuda.CUDAGraph | None = None
        if capture_graph and not self._inplace_state:
            self._graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.device(q.device),
                tvm_ffi.use_torch_stream(),
                torch.cuda.graph(self._graph),
            ):
                self._launch_direct()

        self._refresh_retained_tensors()

    def _refresh_retained_tensors(self) -> None:
        retained = (
            self.q,
            self.k,
            self.v,
            self.alpha,
            self.beta,
            self.output,
            self.initial_state_input,
            self.final_state,
            self.cu_seqlens,
            self.state_indices,
            self.checkpoint_indices,
            self.state_checkpoints,
            self.initial_state,
            self.output_state_workspace,
            self.q_normalized,
            self.k_normalized,
            self.t,
            self.local_transfer,
            self.local_state,
            self.fixed_state,
            self.initial_state_workspace,
            self.tensormap_workspace,
            self._gather_source,
            self._scatter_output,
        )
        self._retained_tensors = tuple(
            tensor for tensor in retained if tensor is not None
        )

    def launch_with_bindings(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor | None,
        beta: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        output: torch.Tensor,
        output_state: torch.Tensor | None,
        state_checkpoints: torch.Tensor | None,
    ) -> None:
        """Launch the direct composite with structurally compatible addresses."""

        if self._graph is not None:
            raise RuntimeError("dynamic GDN CP bindings require a direct composite")
        current = torch.cuda.current_stream(q.device)
        if current.cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("GDNCPPrefill must launch on its preparation stream")

        self.q = q
        self.k = k
        self.v = v
        self.output = output
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta

        self.initial_state_input = initial_state
        if initial_state is not None:
            if self._gather is None:
                self.initial_state = initial_state
            else:
                self._gather_source = _state_carrier(initial_state)

        self.final_state = output_state
        if output_state is not None:
            if self._scatter is None:
                self.output_state_workspace = output_state
            else:
                self._scatter_output = _state_carrier(output_state)
        self.state_checkpoints = state_checkpoints

        self._refresh_retained_tensors()
        with torch.cuda.device(q.device), tvm_ffi.use_torch_stream():
            self._launch_direct()
        for tensor in self._retained_tensors:
            tensor.record_stream(self._stream)

    def _launch_direct(self) -> None:
        p = self.plan
        if p.total_tokens:
            if self._qk_norm is not None:
                self._qk_norm(
                    self.q,
                    self.k,
                    self.q_normalized,
                    self.k_normalized,
                    p.num_q_heads,
                    p.num_k_heads,
                    p.total_tokens,
                    max(p.num_q_heads, p.num_k_heads),
                    1,
                )
            self._t(
                self.k_normalized,
                self.beta,
                self.t,
                self.cu_seqlens,
                p.num_k_heads,
                p.num_sab_heads,
                p.total_t_blocks,
                p.num_seqs,
                *p.t_grid,
            )
            self._mn(
                self.k_normalized,
                self.v,
                self.t,
                self.alpha,
                self.local_transfer,
                self.local_state,
                self.cu_seqlens,
                p.cp_chunk_len,
                p.num_k_heads,
                p.num_v_heads,
                p.num_sab_heads,
                p.total_cp_chunks,
                p.num_seqs,
                *p.cp_grid,
            )
        if self._gather is not None:
            assert (
                self.initial_state_input is not None and self._gather_source is not None
            )
            total_values = p.num_seqs * p.num_sab_heads * _HEAD_DIM * _HEAD_DIM
            self._gather(
                self._gather_source,
                self.state_indices,
                self.initial_state,
                int(self.initial_state_input.stride(0)),
                int(self.initial_state_input.stride(1)),
                int(self.initial_state_input.stride(2)),
                int(self.initial_state_input.stride(3)),
                p.num_sab_heads,
                total_values,
                int(self._use_state_indices),
                _ceil_div(total_values, _STATE_IO_THREADS),
                1,
                1,
            )
        self._fixup(
            self.local_transfer,
            self.local_state,
            self.initial_state,
            self.initial_state_workspace,
            self.fixed_state,
            self.output_state_workspace,
            self.cu_seqlens,
            p.cp_chunk_len,
            p.total_cp_chunks,
            p.num_seqs,
            p.num_sab_heads,
            *p.fixup_grid,
        )
        if self._final_state_recurrence is not None:
            self._final_state_recurrence(
                self.q,
                self.k,
                self.v,
                self.alpha,
                self.beta,
                self.cu_seqlens,
                self.initial_state,
                self.output_state_workspace,
                self.output,
                self.scale,
                self._recurrence_normalize_qk,
                0,
                1,
                p.num_q_heads,
                p.num_k_heads,
                p.num_v_heads,
                p.num_sab_heads,
                p.num_sab_heads,
                p.num_seqs,
                1,
            )
        if self._checkpoint is not None:
            assert self.state_checkpoints is not None
            total_values = p.checkpoint_count * p.num_sab_heads * _HEAD_DIM * _HEAD_DIM
            self._checkpoint(
                self.fixed_state,
                self.checkpoint_indices,
                self.state_checkpoints,
                int(self.fixed_state.stride(0)),
                int(self.fixed_state.stride(1)),
                int(self.fixed_state.stride(2)),
                int(self.fixed_state.stride(3)),
                p.num_sab_heads,
                total_values,
                1,
                _ceil_div(total_values, _STATE_IO_THREADS),
                1,
                1,
            )
        if self._scatter is not None:
            assert self.final_state is not None and self._scatter_output is not None
            total_values = p.num_seqs * p.num_sab_heads * _HEAD_DIM * _HEAD_DIM
            self._scatter(
                self.output_state_workspace,
                self.state_indices,
                self._scatter_output,
                int(self.final_state.stride(0)),
                int(self.final_state.stride(1)),
                int(self.final_state.stride(2)),
                int(self.final_state.stride(3)),
                p.num_sab_heads,
                total_values,
                int(self._use_state_indices),
                _ceil_div(total_values, _STATE_IO_THREADS),
                1,
                1,
            )
        if p.total_tokens:
            self._prefill(
                self.q_normalized,
                self.k_normalized,
                self.v,
                self.t,
                self.output,
                self.alpha,
                self.cu_seqlens,
                self.fixed_state,
                self.initial_state_workspace,
                self.tensormap_workspace,
                p.cp_chunk_len,
                p.source_cp_chunk_len,
                p.num_q_heads,
                p.num_k_heads,
                p.num_v_heads,
                p.num_sab_heads,
                self.scale,
                *p.cp_grid,
            )
            if self._output_recurrence is not None:
                self._output_recurrence(
                    self.q,
                    self.k,
                    self.v,
                    self.alpha,
                    self.beta,
                    self.cu_seqlens,
                    self.initial_state,
                    self.output_state_workspace,
                    self.output,
                    self.scale,
                    self._recurrence_normalize_qk,
                    1,
                    0,
                    p.num_q_heads,
                    p.num_k_heads,
                    p.num_v_heads,
                    p.num_sab_heads,
                    p.num_sab_heads,
                    p.num_seqs,
                    1,
                )

    def replay(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replay the captured route once on its preparation stream."""

        current = torch.cuda.current_stream(self.q.device)
        if current.cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("GDNCPPrefill must replay on its preparation stream")
        if self._graph is None:
            with torch.cuda.device(self.q.device), tvm_ffi.use_torch_stream():
                self._launch_direct()
        else:
            self._graph.replay()
        for tensor in self._retained_tensors:
            tensor.record_stream(self._stream)
        return self.output, self.final_state if self.output_final_state else None


def prepare_gdn_cp_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    seq_lens: Sequence[int] | None = None,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: int | None = None,
    scale: float | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    _capture_graph: bool = True,
) -> GDNCPPrefill:
    """Run once and optionally capture one internal fixed-address GDN CP graph."""

    if not q.is_cuda:
        raise ValueError("GDN CP-prefill requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-3 tensors")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q must have float16 or bfloat16 dtype")
    total = int(q.shape[0])
    if total < 0:
        raise ValueError("q token extent must be nonnegative")
    device = q.device
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise TypeError("cu_seqlens must have int32 or int64 dtype")
    if cu_seqlens.ndim != 1 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a contiguous rank-1 tensor")
    if checkpoint_every_n_tokens < 0 or (
        checkpoint_every_n_tokens and checkpoint_every_n_tokens % _BLOCK
    ):
        raise ValueError(
            "checkpoint_every_n_tokens must be zero or a positive multiple of 64"
        )
    if cp_chunk_len is not None and (
        cp_chunk_len <= 0 or cp_chunk_len % _BLOCK
    ):
        raise ValueError("cp_chunk_len must be a positive multiple of 64")
    if not isinstance(use_qk_l2norm_in_kernel, bool):
        raise TypeError("use_qk_l2norm_in_kernel must be a bool")
    lengths = _read_seq_lens(cu_seqlens, total_tokens=total, expected=seq_lens)
    plan = _build_plan(
        q,
        k,
        v,
        lengths,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        cp_chunk_len=cp_chunk_len,
    )

    _validate_contiguous_tensor(
        q, name="q", shape=(total, plan.num_q_heads, 128), dtype=q.dtype, device=device
    )
    _validate_contiguous_tensor(
        k, name="k", shape=(total, plan.num_k_heads, 128), dtype=q.dtype, device=device
    )
    _validate_contiguous_tensor(
        v, name="v", shape=(total, plan.num_v_heads, 128), dtype=q.dtype, device=device
    )
    _validate_contiguous_tensor(
        cu_seqlens,
        name="cu_seqlens",
        shape=(plan.num_seqs + 1,),
        dtype=cu_seqlens.dtype,
        device=device,
    )
    gate_shape = (total, plan.num_sab_heads)
    # Materialized ones feed the four-stage ABI, while raw absence selects the
    # public semantic recurrence mirror for output and any requested state.
    alpha_was_none = alpha is None
    if alpha is None:
        alpha = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _validate_contiguous_tensor(
            alpha, name="alpha", shape=gate_shape, dtype=torch.float32, device=device
        )
    if beta is None:
        beta = torch.ones(gate_shape, dtype=torch.float32, device=device)
    else:
        _validate_contiguous_tensor(
            beta, name="beta", shape=gate_shape, dtype=torch.float32, device=device
        )
    output_shape = (total, plan.num_sab_heads, _HEAD_DIM)
    if output is None:
        output = torch.empty(output_shape, dtype=q.dtype, device=device)
    else:
        _validate_contiguous_tensor(
            output, name="output", shape=output_shape, dtype=q.dtype, device=device
        )

    indexed = state_indices is not None
    if state_indices is not None:
        if state_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("state_indices must have int32 or int64 dtype")
        _validate_contiguous_tensor(
            state_indices,
            name="state_indices",
            shape=(plan.num_seqs,),
            dtype=state_indices.dtype,
            device=device,
        )
    if checkpoint_every_n_tokens:
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts are required when "
                "checkpointing is enabled"
            )
        _validate_contiguous_tensor(
            state_checkpoints,
            name="state_checkpoints",
            shape=(plan.checkpoint_count, plan.num_sab_heads, 128, 128),
            dtype=torch.float32,
            device=device,
        )
        if checkpoint_cu_starts.dtype not in (torch.int32, torch.int64):
            raise TypeError("checkpoint_cu_starts must have int32 or int64 dtype")
        if (
            tuple(checkpoint_cu_starts.shape) != (plan.num_seqs + 1,)
            or not checkpoint_cu_starts.is_contiguous()
        ):
            raise ValueError(
                "checkpoint_cu_starts must be contiguous with shape [num_seqs + 1]"
            )
        if checkpoint_cu_starts.device not in (device, torch.device("cpu")):
            raise ValueError("checkpoint_cu_starts must be on q.device or CPU")
        expected_checkpoint_cu = [0]
        for length in plan.seq_lens:
            expected_checkpoint_cu.append(
                expected_checkpoint_cu[-1] + length // checkpoint_every_n_tokens
            )
        observed_checkpoint_cu = tuple(
            int(value) for value in checkpoint_cu_starts.detach().cpu().tolist()
        )
        if observed_checkpoint_cu != tuple(expected_checkpoint_cu):
            raise ValueError(
                "checkpoint_cu_starts does not match sequence lengths and interval"
            )
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise ValueError(
            "state_checkpoints and checkpoint_cu_starts must be None when "
            "checkpointing is disabled"
        )
    if initial_state is not None:
        _validate_state(
            initial_state,
            name="initial_state",
            plan=plan,
            device=device,
            indexed=indexed,
        )
    if output_state is not None:
        _validate_state(
            output_state,
            name="output_state",
            plan=plan,
            device=device,
            indexed=indexed,
        )
    if not output_final_state and output_state is not None:
        raise ValueError("output_state must be None when output_final_state=False")
    if indexed and output_final_state and output_state is None:
        raise ValueError("indexed final state requires an explicit output_state pool")
    pool_rows = {
        int(tensor.shape[0])
        for tensor in (initial_state, output_state)
        if tensor is not None
    }
    if len(pool_rows) > 1:
        raise ValueError(
            "initial_state and output_state pools must have equal row counts"
        )
    if state_indices is not None:
        if not pool_rows:
            raise ValueError(
                "state_indices requires an initial_state or output_state pool"
            )
        indices = tuple(int(value) for value in state_indices.detach().cpu().tolist())
        pool_size = next(iter(pool_rows))
        if len(set(indices)) != len(indices):
            raise ValueError("state_indices values must be unique")
        if any(index < 0 or index >= pool_size for index in indices):
            raise ValueError("state_indices values must address the state pool")
    if initial_state is not None and output_state is not None:
        aliases = (
            initial_state.untyped_storage().data_ptr()
            == output_state.untyped_storage().data_ptr()
        )
        if aliases and output_state is not initial_state:
            raise ValueError("state storage aliasing must use the same tensor object")
    if output.numel() > 0:
        output_storage = output.untyped_storage().data_ptr()
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("alpha", alpha),
            ("beta", beta),
        ):
            if tensor.untyped_storage().data_ptr() == output_storage:
                raise ValueError(f"output must not alias read-only input {name}")
    resolved_scale = (
        1.0 / math.sqrt(_HEAD_DIM) if scale is None or scale == 0.0 else float(scale)
    )
    if not math.isfinite(resolved_scale):
        raise ValueError("scale must be finite")
    return GDNCPPrefill(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output=output,
        output_state=output_state,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        scale=resolved_scale,
        output_final_state=output_final_state,
        alpha_was_none=alpha_was_none,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        plan=plan,
        capture_graph=_capture_graph,
    )


def _layout_key(tensor: torch.Tensor | None) -> tuple[object, ...]:
    if tensor is None:
        return (None,)
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


_public_prepared: GDNCPPrefill | None = None
_public_key: tuple[object, ...] | None = None
_public_metadata_binding: (
    tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        int | None,
        int | None,
        int | None,
        tuple[
            tuple[int, ...],
            tuple[int, ...] | None,
            tuple[int, ...] | None,
        ],
    ]
    | None
) = None


def _metadata_version(tensor: torch.Tensor | None) -> int | None:
    """Return a version counter when the tensor tracks one."""

    if tensor is None or torch.is_inference(tensor):
        return None
    return int(tensor._version)


def _metadata_signature(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor | None,
    checkpoint_cu_starts: torch.Tensor | None,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...] | None,
    tuple[int, ...] | None,
]:
    """Read the address maps whose payload determines the prepared plan."""

    cu_values = tuple(int(value) for value in cu_seqlens.detach().cpu().tolist())
    state_values = (
        tuple(int(value) for value in state_indices.detach().cpu().tolist())
        if state_indices is not None
        else None
    )
    checkpoint_values = (
        tuple(int(value) for value in checkpoint_cu_starts.detach().cpu().tolist())
        if checkpoint_cu_starts is not None
        else None
    )
    return cu_values, state_values, checkpoint_values


def _reset_gdn_cp_prefill_cache() -> None:
    """Release the internal public-dispatch plan and its owned workspaces."""

    global _public_key, _public_metadata_binding, _public_prepared
    _public_key = None
    _public_metadata_binding = None
    _public_prepared = None


def chunk_gated_delta_rule_gdn_cp_sm100(
    output: torch.Tensor,
    output_state: torch.Tensor | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor | None,
    beta: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    initial_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    output_final_state: bool,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_cu_starts: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    cp_chunk_len: int | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> None:
    """Public dispatcher target; every accepted SM100/SM103 route uses GDN CP."""

    global _public_key, _public_metadata_binding, _public_prepared
    stream = torch.cuda.current_stream(q.device)
    cu_seqlens_version = _metadata_version(cu_seqlens)
    state_indices_version = _metadata_version(state_indices)
    checkpoint_cu_starts_version = _metadata_version(checkpoint_cu_starts)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing:
        if not (
            _public_metadata_binding is not None
            and _public_metadata_binding[0] is cu_seqlens
            and _public_metadata_binding[1] is state_indices
            and _public_metadata_binding[2] is checkpoint_cu_starts
            and _public_metadata_binding[3] == cu_seqlens_version
            and _public_metadata_binding[4] == state_indices_version
            and _public_metadata_binding[5] == checkpoint_cu_starts_version
        ):
            raise RuntimeError(
                "GDN CP-prefill metadata must be warmed with the same "
                "unchanged tensors before CUDA graph capture"
            )
        metadata_signature = _public_metadata_binding[6]
    else:
        metadata_signature = _metadata_signature(
            cu_seqlens,
            state_indices,
            checkpoint_cu_starts,
        )
    key: tuple[object, ...] = (
        *(
            _layout_key(tensor)
            for tensor in (
                output,
                output_state,
                q,
                k,
                v,
                alpha,
                beta,
                initial_state,
                state_checkpoints,
            )
        ),
        _layout_key(cu_seqlens),
        _layout_key(state_indices),
        _layout_key(checkpoint_cu_starts),
        metadata_signature,
        output_state is initial_state and initial_state is not None,
        float(scale),
        bool(output_final_state),
        int(checkpoint_every_n_tokens),
        cp_chunk_len,
        bool(use_qk_l2norm_in_kernel),
        int(stream.cuda_stream),
    )
    if _public_prepared is None or _public_key != key:
        if capturing:
            raise RuntimeError(
                "GDN CP-prefill plan must be warmed before CUDA graph capture"
            )
        _public_prepared = prepare_gdn_cp_prefill(
            q,
            k,
            v,
            alpha,
            beta,
            cu_seqlens,
            initial_state,
            output=output,
            output_state=output_state,
            state_indices=state_indices,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            cp_chunk_len=cp_chunk_len,
            scale=scale,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            _capture_graph=False,
        )
        _public_key = key
        _public_metadata_binding = (
            cu_seqlens,
            state_indices,
            checkpoint_cu_starts,
            cu_seqlens_version,
            state_indices_version,
            checkpoint_cu_starts_version,
            metadata_signature,
        )
        if initial_state is not None and output_state is initial_state:
            _public_prepared.replay()
        return
    if not capturing:
        _public_metadata_binding = (
            cu_seqlens,
            state_indices,
            checkpoint_cu_starts,
            cu_seqlens_version,
            state_indices_version,
            checkpoint_cu_starts_version,
            metadata_signature,
        )
    _public_prepared.launch_with_bindings(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        initial_state=initial_state,
        output=output,
        output_state=output_state,
        state_checkpoints=state_checkpoints,
    )


__all__ = [
    "GDNCPPrefill",
    "GDNCPPrefillPlan",
    "chunk_gated_delta_rule_gdn_cp_sm100",
    "prepare_gdn_cp_prefill",
]
