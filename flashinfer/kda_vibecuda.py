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
"""

"""
Kimi Delta Attention Prefill - VibeCUDA Backend
================================================

The VibeCUDA recurrent-KDA prefill backend: an optimized SM100-family
FlashKDA schedule family selected through ``backend="vibecuda"`` in
:func:`flashinfer.kda.recurrent_kda`. It shares the public prefill contract
with the frozen Cake backend (same eligibility shape, same in-place state
semantics, same packed-sequence ordering policy, same workspace and CUDA
graph model) while reordering three dispatch policies that the frozen
dispatcher leaves serial:

* M128 slab regime. Short and mid shapes (token count <= 8192, or
  <= 65536 with >= 4 heads) run a compile-time slab-specialized M128 image
  whose combined N=160 UMMA-4 issue is split into two smaller issues,
  shortening tensor-pipe residency per chunk on latency-bound chains.
* Split-seq affine prefix. Under-parallelized fixed layouts (<= 32
  (sequence, head) tasks and >= 256 chunks) split each head's serial chunk
  chain into independent token windows; a bf16 register-carry scan composes
  the per-part affine transforms and a correction pass accumulates the
  carry. At two parts the map pass and scan collapse to an exact carry
  copy.
* Device-planned persistent M128. Packed workloads with at least four
  (sequence, head) tasks per SM (an eager-only route, like the Cake
  persistent schedule) plan balanced task bins inside a one-block planner
  kernel and launch exactly ``sm_count`` workers.

Unsupported contract features (state pools, checkpoints, token-row-strided
beta, speculative or grouped-query layouts) stay on the Cake backend; an
explicit ``backend="vibecuda"`` request for them raises instead of
silently rerouting.

JIT targets: CC 10.0 and CC 10.3 devices load exact-architecture
``sm_100a`` and ``sm_103a`` builds so each target can use its measured
architecture-specific schedule (see ``_vibecuda_prefill_target``).
"""

import math
import threading
from typing import Optional, cast

import torch

from . import kda_prefill as _kda_prefill
from .kda_prefill import (
    _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES,
    RecurrentKDAPrefillWorkspace,
    _RecurrentKDAPrefillWorkspaceBase,
    _bind_workspace,
    _check_output_does_not_overlap_inputs,
    _flash_kda_device_sm_count,
    _fixed_cu_seqlens,
    _identity_seq_order,
    _stream_cache_key,
    _validate_prefill_seq_order,
    _workspace_buffer,
)
from .utils import get_compute_capability

_HEAD_DIM = 128
_DESCRIPTOR_STORAGE_BYTES = 7 * 128
_BETA_TMA_MIN_HEADS = 8
_PERSISTENT_MIN_TASKS_PER_WORKER = 4
_PERSISTENT_MAX_WORKERS = 160
_SPLIT_MAX_TASKS = 32
_SPLIT_MIN_CHUNKS = 256
_SPLIT_MIN_CHUNKS_PER_PART = 32


class _VibeCUDAPrefillState:
    """Per-(workspace) VibeCUDA storage: TMA descriptors, beta padding, the
    dummy state, split-seq scratch, persistent planner scratch, and the
    device-sorted sequence-order staging buffer.

    Every buffer is capacity-grown only; contents are recomputed on every
    call. One state object is owned by either the implicit per-(device,
    stream) workspace or an explicit ``RecurrentKDAPrefillWorkspace`` (via
    its ``_vibecuda_state`` slot), so CUDA graph replay keeps stable
    addresses.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.descriptor_storages = {
            variant: torch.empty(
                _DESCRIPTOR_STORAGE_BYTES, dtype=torch.uint8, device=device
            )
            for variant in ("m64", "m128", "persistent")
        }
        self.descriptor_signatures: dict[str, tuple] = {}
        self.dummy_state = torch.empty(1, dtype=torch.bfloat16, device=device)
        self._beta_padding: Optional[torch.Tensor] = None
        self._sorted_seq_order: Optional[torch.Tensor] = None
        self._split_state: Optional[torch.Tensor] = None
        self._split_map_state: Optional[torch.Tensor] = None
        self._split_carry: Optional[torch.Tensor] = None
        self._split_out: Optional[torch.Tensor] = None
        self._split_map_state_bf16: Optional[torch.Tensor] = None
        self._persistent_task_ids: Optional[torch.Tensor] = None
        self._persistent_task_offsets: Optional[torch.Tensor] = None
        self._persistent_choice: Optional[torch.Tensor] = None


def _vibecuda_state(
    workspace: _RecurrentKDAPrefillWorkspaceBase, device: torch.device
) -> _VibeCUDAPrefillState:
    """Resolve (and lazily create) the VibeCUDA state owned by ``workspace``.

    Mirrors the SM120 backend's ``_sm120_state`` composition: the shared
    public workspace type carries one opaque slot per backend so each
    backend owns the buffers only it understands.
    """

    state = workspace._vibecuda_state
    if state is None:
        with workspace._vibecuda_state_lock:
            state = workspace._vibecuda_state
            if state is None:
                state = _VibeCUDAPrefillState(device)
                workspace._vibecuda_state = state
    return cast(_VibeCUDAPrefillState, state)


def _buffer_owner(
    state: _VibeCUDAPrefillState,
) -> _RecurrentKDAPrefillWorkspaceBase:
    return cast(_RecurrentKDAPrefillWorkspaceBase, state)


class _VibeCUDAStreamWorkspace(_RecurrentKDAPrefillWorkspaceBase):
    """Internal eager-only workspace for one CUDA stream."""


_vibecuda_stream_workspaces: dict[tuple[int, int], _VibeCUDAStreamWorkspace] = {}
_vibecuda_stream_workspaces_lock = threading.Lock()


def _get_vibecuda_stream_workspace(device: torch.device) -> _VibeCUDAStreamWorkspace:
    key = _stream_cache_key(device)
    with _vibecuda_stream_workspaces_lock:
        workspace = _vibecuda_stream_workspaces.get(key)
        if workspace is None:
            workspace = _VibeCUDAStreamWorkspace(device)
            _vibecuda_stream_workspaces[key] = workspace
        return workspace


def _vibecuda_beta_tma_source(
    beta: torch.Tensor,
    state: _VibeCUDAPrefillState,
) -> torch.Tensor:
    """Pack beta for the TMA descriptor exactly as the kernels expect.

    The descriptor fetches a full (32 token, 8 head) box, so shapes with
    fewer than 32 tokens or fewer than 8 heads are staged into a padded
    workspace buffer; the padding is re-zeroed and refilled on every call.
    TMA global strides must be multiples of 16 bytes, so head counts whose
    (head, 2-byte) stride is not a multiple of 8 heads are staged through
    the padded buffer as well (padded head columns stay zero-filled).
    """

    batch_size, seq_len, num_heads = beta.shape
    total_tokens = batch_size * seq_len
    beta_flat = beta.reshape(total_tokens, num_heads)
    padded_tokens = max(total_tokens, 32)
    aligned_heads = (
        (num_heads + _BETA_TMA_MIN_HEADS - 1)
        // _BETA_TMA_MIN_HEADS
        * _BETA_TMA_MIN_HEADS
    )
    padded_heads = max(aligned_heads, _BETA_TMA_MIN_HEADS)
    if padded_tokens == total_tokens and padded_heads == num_heads:
        return beta_flat
    padded = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_beta_padding",
        device=beta.device,
        numel=padded_tokens * padded_heads,
        capture_error=(
            "recurrent_kda vibecuda beta TMA workspace is not large enough "
            "for CUDA graph capture; warm the largest padded token/head "
            "shape on this stream before capture"
        ),
    ).view(padded_tokens, padded_heads)
    padded.zero_()
    padded[:total_tokens, :num_heads].copy_(beta_flat)
    return padded


def _vibecuda_split_buffers(
    state: _VibeCUDAPrefillState,
    device: torch.device,
    num_tasks: int,
    num_parts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split-seq scratch (capacity-grown only, refilled every call).

    ``split_state``/``map_state`` hold each part's affine transform (fp32
    end state + the map pass's linear operator), ``carry`` the composed
    per-part prefix states, and ``map_state_bf16`` the register-carry scan's
    bf16 M panels. The scan keeps the carry in registers, so no separate
    gamma buffer is needed.
    """

    capture_error = (
        "recurrent_kda vibecuda split-seq workspace is not large enough for "
        "CUDA graph capture; warm the largest split shape on this stream "
        "before capture"
    )
    split_state = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_split_state",
        device=device,
        numel=num_tasks * num_parts * 16384,
        capture_error=capture_error,
        dtype=torch.float32,
    )
    map_state = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_split_map_state",
        device=device,
        numel=num_tasks * num_parts * 16384,
        capture_error=capture_error,
        dtype=torch.float32,
    )
    carry = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_split_carry",
        device=device,
        numel=num_tasks * (num_parts - 1) * 16384,
        capture_error=capture_error,
        dtype=torch.float32,
    )
    map_state_bf16 = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_split_map_state_bf16",
        device=device,
        numel=num_tasks * num_parts * 16384,
        capture_error=capture_error,
        dtype=torch.bfloat16,
    )
    return split_state, map_state, carry, map_state_bf16


def _vibecuda_split_out_buffer(
    state: _VibeCUDAPrefillState,
    device: torch.device,
    numel: int,
) -> torch.Tensor:
    """Correction-pass scratch output (out-shaped bf16). Rows outside the
    correction window are never read, so no clearing is needed."""

    return _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_split_out",
        device=device,
        numel=numel,
        capture_error=(
            "recurrent_kda vibecuda split-out workspace is not large enough "
            "for CUDA graph capture; warm the largest split shape on this "
            "stream before capture"
        ),
    )


def _vibecuda_persistent_planner_buffers(
    state: _VibeCUDAPrefillState,
    device: torch.device,
    total_tasks: int,
    sm_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capacity-grown int32 bins refilled by the device planner each call."""

    capture_error = (
        "recurrent_kda vibecuda persistent-planner workspace is not large "
        "enough for CUDA graph capture; warm the largest packed shape on "
        "this stream before capture"
    )
    task_ids = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_persistent_task_ids",
        device=device,
        numel=total_tasks,
        capture_error=capture_error,
        dtype=torch.int32,
    )
    task_offsets = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_persistent_task_offsets",
        device=device,
        numel=sm_count + 1,
        capture_error=capture_error,
        dtype=torch.int32,
    )
    choice = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_persistent_choice",
        device=device,
        numel=total_tasks,
        capture_error=capture_error,
        dtype=torch.int32,
    )
    return task_ids, task_offsets, choice


def _vibecuda_sorted_seq_order(
    module,
    state: _VibeCUDAPrefillState,
    cu_seqlens_i64: torch.Tensor,
    stream_ptr: int,
) -> torch.Tensor:
    """Stable descending-length sequence order, sorted on device (no host
    readback; equal lengths keep their original relative order)."""

    num_sequences = cu_seqlens_i64.numel() - 1
    order = _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_sorted_seq_order",
        device=cu_seqlens_i64.device,
        numel=num_sequences,
        capture_error=(
            "recurrent_kda vibecuda sorted seq_order workspace is not large "
            "enough for CUDA graph capture; warm the largest packed shape "
            "on this stream before capture or pass an explicit seq_order"
        ),
        dtype=torch.int32,
    )
    module.sort_seqs_into(cu_seqlens_i64, order, stream_ptr)
    return order


def _vibecuda_kda_prefill_is_eligible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Return whether the call exactly matches the VibeCUDA contract: the
    plain-prefill subset of the frozen eligibility plus contiguous beta
    (the VibeCUDA kernels consume beta through a padded copy instead of the
    token-row-strided direct path) and no serving-only features.
    """

    if not _kda_prefill._flash_kda_prefill_is_eligible(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_spec_tokens=num_spec_tokens,
        num_accepted_tokens=num_accepted_tokens,
        output=output,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        beta_is_logit=beta_is_logit,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    ):
        return False
    if not beta.is_contiguous():
        return False
    if initial_state is not None and not initial_state.is_contiguous():
        return False
    if ssm_state_indices is not None:
        return False
    if checkpoint_every_n_tokens or state_checkpoints is not None:
        return False
    return True


def _get_vibecuda_prefill_module(target):
    from .jit.flash_kda import load_vibecuda_flash_kda_module

    return load_vibecuda_flash_kda_module(target)


def _vibecuda_prefill_target(device: torch.device):
    """Select the VibeCUDA JIT target for ``device``.

    The VibeCUDA schedules use exact-architecture images because the SM100
    and SM103 builds carry independently measured schedule choices.
    """

    compute_capability = get_compute_capability(device)
    if compute_capability == (10, 0):
        return "sm100a"
    if compute_capability == (10, 3):
        return "sm103a"
    raise RuntimeError(
        "The VibeCUDA recurrent KDA prefill backend supports only compute "
        f"capabilities 10.0 and 10.3, got {compute_capability}"
    )


def _run_vibecuda_kda_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: float,
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the VibeCUDA prefill backend (same public contract as
    :func:`flashinfer.kda_prefill._run_flash_kda_prefill`).
    """

    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and prefill_workspace is None:
        raise RuntimeError(
            "CUDA graph capture of recurrent_kda vibecuda prefill requires an "
            "explicit RecurrentKDAPrefillWorkspace warmed with the exact "
            "tensors on the capture stream"
        )
    batch_size, seq_len, num_heads, _ = q.shape
    fixed_layout = cu_seqlens is None
    num_sequences = batch_size if fixed_layout else cu_seqlens.numel() - 1
    target = _vibecuda_prefill_target(q.device)
    compute_capability = get_compute_capability(q.device)
    if compute_capability not in _FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "the vibecuda recurrent-KDA prefill backend requires compute "
            f"capability 10.0 or 10.3, got {compute_capability[0]}."
            f"{compute_capability[1]}"
        )
    sm_count = _flash_kda_device_sm_count(q.device)
    stream_workspace = (
        _get_vibecuda_stream_workspace(q.device) if prefill_workspace is None else None
    )

    # Variant dispatch: the M64 two-CTA schedule is the measured winner for
    # fixed single-sequence H=64; everything else runs the M128 family.
    # (Round-102 A/B: routing sub-persistent packed H=64 to M64 measured
    # 0.583x on the mixed-length case — the value-half prep duplication
    # dominates the short chains. M128 stays the only packed route.)
    variant = (
        "m64" if (fixed_layout and num_sequences == 1 and num_heads == 64) else "m128"
    )

    # Split-seq policy (fixed layouts only): under-parallelized small-BH
    # ultra-long chains split into num_parts contiguous token windows. Every
    # launch stays within one wave: part counts never exceed sm_count and
    # keep at least 32 chunks per part.
    split_parts = 1
    if fixed_layout:
        num_tasks = num_sequences * num_heads
        max_chunks = (seq_len + 31) // 32
        if num_tasks <= _SPLIT_MAX_TASKS and max_chunks >= _SPLIT_MIN_CHUNKS:
            sm_fill_parts = max(2, sm_count // num_tasks)
            split_parts = min(
                sm_count,
                sm_fill_parts,
                max(2, max_chunks // _SPLIT_MIN_CHUNKS_PER_PART),
            )
        if split_parts >= 2:
            variant = "m128"

    # Persistent-M128 policy (packed only; eager-only, like the Cake
    # persistent route): deep workloads whose (seq, head) task count
    # wave-chains the direct route launch exactly sm_count workers with
    # device-planned balanced bins.
    use_persistent = (
        split_parts == 1
        and not fixed_layout
        and prefill_workspace is None
        and sm_count <= _PERSISTENT_MAX_WORKERS
        and num_sequences * num_heads >= _PERSISTENT_MIN_TASKS_PER_WORKER * sm_count
    )
    if use_persistent:
        variant = "persistent"

    if fixed_layout:
        cu_seqlens_i64 = _fixed_cu_seqlens(
            device=q.device, batch_size=batch_size, seq_len=seq_len
        )
    else:
        assert cu_seqlens is not None
        if cu_seqlens.dtype == torch.int32 and capturing:
            raise RuntimeError(
                "packed recurrent_kda vibecuda prefill requires int64 "
                "cu_seqlens during CUDA graph capture; convert it before "
                "capture"
            )
        cu_seqlens_i64 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int64
            else cu_seqlens.to(torch.int64)
        )

    scale_value = 1.0 / math.sqrt(_HEAD_DIM) if scale is None else float(scale)
    if not math.isfinite(scale_value):
        raise ValueError(f"scale must be finite, got {scale_value}")

    if output is None:
        if capturing:
            raise RuntimeError(
                "CUDA graph capture requires a preallocated output tensor "
                "for recurrent_kda vibecuda prefill"
            )
        out_buf = torch.empty_like(q)
    else:
        out_buf = output
    _check_output_does_not_overlap_inputs(
        out_buf, q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state
    )

    use_initial_state = initial_state is not None
    dummy_state: Optional[torch.Tensor] = None
    if initial_state is not None:
        initial_state_arg = initial_state
        final_state_arg = initial_state
        store_final_state = True
        returned_state = initial_state
    else:
        final_state_arg = None
        returned_state = None
        store_final_state = output_final_state

    stream_ptr = int(torch.cuda.current_stream(q.device).cuda_stream)
    explicit_workspace = prefill_workspace is not None
    workspace: _RecurrentKDAPrefillWorkspaceBase = (
        stream_workspace if stream_workspace is not None else prefill_workspace
    )
    assert workspace is not None
    # TVM FFI may release the GIL. Serialize the complete shared-workspace
    # enqueue sequence so two host threads cannot interleave preparation or
    # launch on the same CUDA stream.
    with workspace._lock:
        _bind_workspace(
            workspace,
            device=q.device,
            stream_ptr=stream_ptr,
            capturing=capturing,
            explicit=explicit_workspace,
        )
        state = _vibecuda_state(workspace, q.device)
        if not use_initial_state:
            dummy_state = state.dummy_state
            initial_state_arg = dummy_state
        if output_final_state and initial_state is None:
            if explicit_workspace:
                final_state_arg = _workspace_buffer(
                    workspace=workspace,
                    attribute="_state_scratch",
                    device=q.device,
                    numel=num_sequences * num_heads * _HEAD_DIM * _HEAD_DIM,
                    capture_error=(
                        "recurrent_kda vibecuda final-state workspace is not "
                        "large enough for CUDA graph capture; warm the "
                        "largest shape on this stream before capture"
                    ),
                ).view(num_sequences, num_heads, _HEAD_DIM, _HEAD_DIM)
            else:
                final_state_arg = torch.empty(
                    (num_sequences, num_heads, _HEAD_DIM, _HEAD_DIM),
                    dtype=torch.bfloat16,
                    device=q.device,
                )
            returned_state = final_state_arg
        elif output_final_state:
            returned_state = initial_state

        # Resolve every Python-side artifact first (workspace buffers, JIT
        # module, sequence order): the beta-pad staging issues GPU kernels
        # from Python before the FFI-issued kernel chain, and span-based
        # timers (CUPTI activity span per call) count any idle gap between
        # them. Issuing the pads last keeps the pad kernels and the kernel
        # chain contiguous on the stream.
        split_state = None
        map_state = None
        carry = None
        split_out = None
        map_state_bf16 = None
        if split_parts >= 2:
            split_state, map_state, carry, map_state_bf16 = _vibecuda_split_buffers(
                state,
                q.device,
                num_sequences * num_heads,
                split_parts,
            )
            split_out = _vibecuda_split_out_buffer(state, q.device, out_buf.numel())
        descriptor_storage = state.descriptor_storages[variant]
        module = _get_vibecuda_prefill_module(target)
        if seq_order is None:
            seq_order_i32 = (
                _identity_seq_order(device=q.device, num_sequences=num_sequences)
                if fixed_layout
                else _vibecuda_sorted_seq_order(
                    module=module,
                    state=state,
                    cu_seqlens_i64=cu_seqlens_i64,
                    stream_ptr=stream_ptr,
                )
            )
        else:
            seq_order_i32 = _validate_prefill_seq_order(
                seq_order,
                fixed_layout=fixed_layout,
                num_sequences=num_sequences,
                device=q.device,
            )

        beta_tma = _vibecuda_beta_tma_source(beta, state)
        if split_parts >= 2:
            signature = _kda_prefill._descriptor_signature(
                q=q,
                k=k,
                v=v,
                g=g,
                beta_tma=beta_tma,
                out=out_buf,
                packet_workspace=None,
            ) + (_kda_prefill._tensor_descriptor_signature(split_out),)
        else:
            signature = _kda_prefill._descriptor_signature(
                q=q, k=k, v=v, g=g, beta_tma=beta_tma, out=out_buf
            )
        warmed_signature = state.descriptor_signatures.get(variant)
        if capturing:
            if warmed_signature != signature:
                raise RuntimeError(
                    "RecurrentKDAPrefillWorkspace is not warmed for the "
                    f"exact vibecuda {variant} descriptor signature; eagerly "
                    "invoke the same call on this stream before capture"
                )
            prepare_descriptors = 0
        else:
            prepare_descriptors = int(warmed_signature != signature)
        try:
            if variant == "m64":
                module.run_m64(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
            elif variant == "persistent":
                task_ids, task_offsets, choice = _vibecuda_persistent_planner_buffers(
                    state, q.device, num_sequences * num_heads, sm_count
                )
                module.run_persistent_m128(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    task_ids,
                    task_offsets,
                    choice,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    scale_value,
                    float(lower_bound),
                    sm_count,
                    stream_ptr,
                )
            elif split_parts >= 2:
                module.run_m128_split(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    split_state,
                    map_state,
                    carry,
                    split_out,
                    map_state_bf16,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    scale_value,
                    float(lower_bound),
                    split_parts,
                    stream_ptr,
                )
            else:
                module.run_m128(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    beta_tma,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    float(lower_bound),
                    stream_ptr,
                )
        except Exception:
            if prepare_descriptors:
                state.descriptor_signatures.pop(variant, None)
            raise
        if prepare_descriptors:
            state.descriptor_signatures[variant] = signature
        if capturing and explicit_workspace:
            workspace._captured = True
    return out_buf, (returned_state if output_final_state else None)
