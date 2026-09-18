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
graph model) while reordering five dispatch policies that the frozen
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
  copy. The dead-map progress-flag channel skips the map pass's dead band
  on wide splits.
* Device-planned persistent M128. Packed workloads with at least four
  (sequence, head) tasks per SM (an eager-only route, like the Cake
  persistent schedule) plan balanced task bins inside a one-block planner
  kernel and launch exactly ``sm_count`` workers.
* Head-family fam2 route. Non-split workloads with 12/64/96 heads (12/64/96
  on CC 10.0, 64/96 on CC 10.3) run the compile-time-H single-kernel
  per-(sequence, head) recurrence built on the evolved slab schedule,
  which removes the planner and every split/map/correction launch from the
  per-chunk serial chain.
* Fused N16 route. H12 fixed single-sequence layouts up to 1024 tokens and
  H12 packed layouts with at most six sequences factor raw operands into
  chain workspace in one preparation kernel, then walk the chain at a
  16-token chunk granularity in a single fused host call.

Unsupported contract features (state pools, checkpoints, token-row-strided
beta, speculative or grouped-query layouts) stay on the Cake backend; an
explicit ``backend="vibecuda"`` request for them raises instead of
silently rerouting.

JIT targets: CC 10.0 and CC 10.3 devices load exact-architecture
``sm_100a`` and ``sm_103a`` builds so each target can use its measured
architecture-specific schedule (see ``_vibecuda_prefill_target``).
"""

import math
import os
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
_FN16_CHUNK_TOKENS = 16
# Head-family fam2 support per exact-architecture target (measured winners).
# sm103a keeps the round-200 dispatch: fam2h12 measured only a marginal in-window
# gain on CC 10.3 (R204 same-window A/B) while the split-row fast-plan
# extension is the retained sm103 improvement, so CC 10.3 routes H12 through
# the m128/persistent fallbacks exactly as the retained round-200 build did.
_FAM2_HEADS_BY_TARGET = {"sm100a": (12, 64, 96), "sm103a": (64, 96)}


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
            for variant in (
                "m64",
                "m128",
                "persistent",
                "fam2h12",
                "fam2h64",
                "fam2h96",
                "fn16_prepare",
                "fn16_chain",
            )
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
        self._split_lookback_flags: Optional[torch.Tensor] = None
        self._persistent_task_ids: Optional[torch.Tensor] = None
        self._persistent_task_offsets: Optional[torch.Tensor] = None
        self._persistent_choice: Optional[torch.Tensor] = None
        self._fn16_qd: Optional[torch.Tensor] = None
        self._fn16_kd: Optional[torch.Tensor] = None
        self._fn16_w: Optional[torch.Tensor] = None
        self._fn16_qk: Optional[torch.Tensor] = None
        self._fn16_diag: Optional[torch.Tensor] = None
        # Eager fast-plan cache (see ``_VibeCUDAFastPlan``): metadata-keyed
        # dispatch plans and the per-variant pointer tuples mirroring
        # ``descriptor_signatures`` for the plan-driven enqueue path.
        self._fast_plans: dict[tuple, "_VibeCUDAFastPlan"] = {}
        self._fast_last_ptrs: dict[str, tuple] = {}


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
    direct_heads: bool = False,
) -> torch.Tensor:
    """Pack beta for the TMA descriptor exactly as the kernels expect.

    The descriptor fetches a full (32 token, 8 head) box, so shapes with
    fewer than 32 tokens or fewer than 8 heads are staged into a padded
    workspace buffer. TMA global strides must be multiples of 16 bytes, so
    head counts whose (head, 2-byte) stride is not a multiple of 8 heads
    would need the padded buffer as well — but the direct-read routes
    (M128/slab/split and the fam2/fused-N16 images) fetch beta logits
    straight from the original [T, H] layout with per-lane global reads
    when ``num_heads`` is not a multiple of 8, so those calls pass the
    original tensor and skip the pad copy entirely. The buffer is zeroed
    once at growth time: padding lanes are not math operands for real
    heads (TMA stride separates lanes), so a reused buffer only needs its
    token/head submatrix rewritten on each call.
    """

    batch_size, seq_len, num_heads = beta.shape
    total_tokens = batch_size * seq_len
    beta_flat = beta.reshape(total_tokens, num_heads)
    if direct_heads and num_heads % 8 != 0:
        return beta_flat
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
        zero_on_allocate=True,
    ).view(padded_tokens, padded_heads)
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


def _vibecuda_split_lookback_flags(
    state: _VibeCUDAPrefillState,
    device: torch.device,
    num_tasks: int,
    num_parts: int,
) -> torch.Tensor:
    """Per-part int32 flag channel for the split pipeline.

    The split kernel binding clears the buffer before each consuming pass;
    the map pass doubles it as the dead-map progress flags, and the fused
    correction walk uses it as the lookback flag channel."""

    return _workspace_buffer(
        workspace=_buffer_owner(state),
        attribute="_split_lookback_flags",
        device=device,
        numel=num_tasks * num_parts,
        capture_error=(
            "recurrent_kda vibecuda split lookback workspace is not large "
            "enough for CUDA graph capture; warm the largest split shape on "
            "this stream before capture"
        ),
        dtype=torch.int32,
    )


def _vibecuda_fn16_buffers(
    state: _VibeCUDAPrefillState,
    device: torch.device,
    num_heads: int,
    total_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused-N16 factor workspace (capacity-grown only, refilled by the
    prepare kernel on every call).

    ``qd``/``kd``/``w`` are the [heads, padded_tokens, 128] bf16 factor
    tiles, ``qk`` the [heads, chunks, 16, 16] bf16 pair-gram workspace, and
    ``diag`` the [heads, chunks, 128] fp32 diagonal workspace; they are the
    chain kernel's only operand inputs."""

    padded_tokens = total_chunks * _FN16_CHUNK_TOKENS
    capture_error = (
        "recurrent_kda vibecuda fused-N16 workspace is not large enough for "
        "CUDA graph capture; warm the largest H12 short/packed shape on this "
        "stream before capture"
    )

    def _buffer(attribute: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
        return _workspace_buffer(
            workspace=_buffer_owner(state),
            attribute=attribute,
            device=device,
            numel=numel,
            capture_error=capture_error,
            dtype=dtype,
        )

    qd = _buffer("_fn16_qd", num_heads * padded_tokens * _HEAD_DIM, torch.bfloat16)
    kd = _buffer("_fn16_kd", num_heads * padded_tokens * _HEAD_DIM, torch.bfloat16)
    w = _buffer("_fn16_w", num_heads * padded_tokens * _HEAD_DIM, torch.bfloat16)
    qk = _buffer(
        "_fn16_qk",
        num_heads * total_chunks * _FN16_CHUNK_TOKENS * _FN16_CHUNK_TOKENS,
        torch.bfloat16,
    )
    diag = _buffer("_fn16_diag", num_heads * total_chunks * _HEAD_DIM, torch.float32)
    return qd, kd, w, qk, diag


def _vibecuda_fn16_prepare_ctas(
    num_heads: int, total_chunks: int, sm_count: int
) -> int:
    """Wave-quantized prepare grid for the fused-N16 route.

    H12 sweeps favor chunks-per-CTA 1 for short layouts and 12 for deep
    ones; other head counts use the generic 8/6 policy. The rectangular
    grid is wave-quantized once it exceeds eight worker waves."""

    if num_heads == 12:
        chunks_per_cta = 1 if total_chunks <= 128 else 12
    else:
        chunks_per_cta = 8 if num_heads * total_chunks >= 16384 else 6
    rect = (total_chunks + chunks_per_cta - 1) // chunks_per_cta * num_heads
    if rect < 8 * sm_count:
        return rect
    full = (rect // sm_count) * sm_count
    if full < num_heads or full * 100 < rect * 98:
        return rect
    return full


def _vibecuda_fn16_chain_schedule(total_tasks: int, sm_count: int) -> int:
    """Recurrence-chain schedule selection: 2*tasks > SMs -> s7 (two-
    resident), tasks <= 8 -> s9 (underfilled grid), else s8 (canonical)."""

    if 2 * total_tasks > sm_count:
        return 7
    if total_tasks <= 8:
        return 9
    return 8


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
    # The VibeCUDA schedules bake the gate lower bound into the kernels, so a
    # missing bound is a strict-backend rejection (the Cake schedules
    # approximate it instead).
    if lower_bound is None:
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


# ---------------------------------------------------------------------------
# Eager fast-plan cache (SM100)
#
# Hot packed-prefill calls repeat the same tensor metadata every invocation
# (serving and benchmark loops both call ``recurrent_kda`` with identical
# shapes/dtypes/flags and only fresh device buffers). The generic entry then
# re-derives, per call: the full frozen+vibecuda eligibility conjunction, the
# fn16/split/persistent/fam2 policy chain, the beta-TMA padding decision, and
# the fixed-layout metadata synthesis. Profiling on the packed H96 fam2 row
# attributes ~60us of host enqueue per call to that repeated derivation while
# the GPU idles ahead of the kernel chain.
#
# A successful generic run on an eligible eager target records a
# ``_VibeCUDAFastPlan`` on the implicit per-(device, stream) workspace state,
# keyed ONLY by tensor metadata and scalar flags: shape/stride/dtype/device
# per tensor, cu_seqlens numel/dtype, and the scalar scalar/flag prefix of
# the public signature. Device pointers and cu_seqlens/seq_order VALUES are
# deliberately excluded: eligibility and dispatch are metadata-pure (sequence
# lengths only reach the device sort/persistent gates through counts), so a
# key hit proves the recorded dispatch leg is still the generic result, and
# pointer-dependent work (TMA descriptor staleness, output-overlap checks)
# is still re-derived on every fast call. Cache misses and every opted-out
# feature surface (explicit workspaces, CUDA graph capture, state pools,
# checkpoints, spec/extras) fall through to the generic path unchanged.
# The eager fast-plan cache is SM100-only: CC 10.3 keeps the round-200
# retained dispatch, which always derives eligibility and the schedule through
# the generic path (the sm103 fast-plan re-enactment measured below the
# retained round-200 sm103 gate).
_FAST_PLAN_TARGETS = frozenset({"sm100a"})
_FAST_PLAN_LIMIT = 64
_FAST_PLAN_ENABLED = os.environ.get("KDA_FASTPLAN", "1") != "0"

# Per-device-index fast-path support memo; computing the JIT target needs a
# compute-capability probe, so it is resolved once per device.
_fast_plan_supported_devices: dict[int, bool] = {}


class _VibeCUDAFastPlan:
    """Recorded dispatch artifacts of one successful generic eager call.

    Everything here is derivable from the metadata key plus the physical
    device; none of it depends on input values or pointer identity.
    """

    __slots__ = (
        "variant",
        "module",
        "num_heads",
        "num_sequences",
        "fixed_layout",
        "sm_count",
        "descriptor_storage",
        "cu_is_int64",
        "beta_direct_heads",
        "beta_needs_pad",
        "beta_rows",
        "scale_value",
        "lower_bound_value",
        "use_initial_state",
        "output_final_state",
        "split_parts",
    )

    def __init__(
        self,
        *,
        variant: str,
        module,
        num_heads: int,
        num_sequences: int,
        fixed_layout: bool,
        sm_count: int,
        descriptor_storage: torch.Tensor,
        cu_is_int64: bool,
        beta_direct_heads: bool,
        beta_needs_pad: bool,
        beta_rows: int,
        scale_value: float,
        lower_bound_value: float,
        use_initial_state: bool,
        output_final_state: bool,
        split_parts: int,
    ) -> None:
        self.variant = variant
        self.module = module
        self.num_heads = num_heads
        self.num_sequences = num_sequences
        self.fixed_layout = fixed_layout
        self.sm_count = sm_count
        self.descriptor_storage = descriptor_storage
        self.cu_is_int64 = cu_is_int64
        self.beta_direct_heads = beta_direct_heads
        self.beta_needs_pad = beta_needs_pad
        self.beta_rows = beta_rows
        self.scale_value = scale_value
        self.lower_bound_value = lower_bound_value
        self.use_initial_state = use_initial_state
        self.output_final_state = output_final_state
        # 1 for the non-split routes; >=2 records the split-seq policy so the
        # hit path can re-enact ``run_m128_split`` (its part count is a pure
        # function of the plan-pinned metadata key plus the device).
        self.split_parts = split_parts


def _fast_plan_device_supported(device: torch.device) -> bool:
    index = device.index
    if index is None:
        return False
    supported = _fast_plan_supported_devices.get(index)
    if supported is None:
        try:
            supported = _vibecuda_prefill_target(device) in _FAST_PLAN_TARGETS
        except RuntimeError:
            supported = False
        _fast_plan_supported_devices[index] = supported
    return supported


def _fast_plan_key(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    scale: Optional[float],
    lower_bound: Optional[float],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    beta_is_logit: bool,
    output_final_state: bool,
) -> Optional[tuple]:
    """Hashable metadata envelope proving eligibility/dispatch invariance.

    Covers every field the frozen/vibecuda eligibility conjunction and the
    dispatch policy chain read: per-tensor shape/dtype/device plus one
    contiguity requirement (every tensor an eligible VibeCUDA call accepts —
    Q/K/V/G, beta, A_log, dt_bias, initial_state, output, cu_seqlens — must
    be contiguous, so stride is implied by shape), cu_seqlens numel/dtype
    (values never reach host dispatch), seq_order presence (its metadata is
    re-validated on the hit path), and the scalar gate/flag prefix. Inputs
    with identical keys take the identical generic route, so the recorded
    plan re-enacts a proven generic result. Returns ``None`` when any input
    violates a metadata condition the eligibility requires, forcing the
    generic path (which raises or dispatches as before).
    """

    def sig(t: torch.Tensor) -> tuple:
        # get_device() returns the CUDA index (or -1 for CPU) without
        # materializing a torch.device wrapper per tensor.
        return (t.shape, t.dtype, t.get_device())

    if not (
        q.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
        and g.is_contiguous()
        and beta.is_contiguous()
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
    ):
        return None
    for optional in (initial_state, output, cu_seqlens, seq_order):
        if optional is not None and not optional.is_contiguous():
            return None
    return (
        sig(q),
        sig(k),
        sig(v),
        sig(g),
        sig(beta),
        sig(A_log),
        sig(dt_bias),
        sig(initial_state) if initial_state is not None else None,
        sig(output) if output is not None else None,
        sig(cu_seqlens) if cu_seqlens is not None else None,
        sig(seq_order) if seq_order is not None else None,
        float(scale) if scale is not None else None,
        float(lower_bound) if lower_bound is not None else None,
        bool(use_qk_l2norm_in_kernel),
        bool(use_gate_in_kernel),
        bool(beta_is_logit),
        bool(output_final_state),
    )


def _record_fast_plan(
    *,
    state: _VibeCUDAPrefillState,
    key: tuple,
    variant: str,
    module,
    num_heads: int,
    num_sequences: int,
    fixed_layout: bool,
    sm_count: int,
    descriptor_storage: torch.Tensor,
    cu_is_int64: bool,
    beta_direct_heads: bool,
    beta_needs_pad: bool,
    beta_rows: int,
    scale_value: float,
    lower_bound_value: float,
    use_initial_state: bool,
    output_final_state: bool,
    split_parts: int,
) -> None:
    plans = state._fast_plans
    if len(plans) >= _FAST_PLAN_LIMIT and key not in plans:
        # Metadata sweeps must not grow the cache without bound; dropping
        # every entry only costs each live shape one generic re-record.
        plans.clear()
    plans[key] = _VibeCUDAFastPlan(
        variant=variant,
        module=module,
        num_heads=num_heads,
        num_sequences=num_sequences,
        fixed_layout=fixed_layout,
        sm_count=sm_count,
        descriptor_storage=descriptor_storage,
        cu_is_int64=cu_is_int64,
        beta_direct_heads=beta_direct_heads,
        beta_needs_pad=beta_needs_pad,
        beta_rows=beta_rows,
        scale_value=scale_value,
        lower_bound_value=lower_bound_value,
        use_initial_state=use_initial_state,
        output_final_state=output_final_state,
        split_parts=split_parts,
    )


def _try_run_fast_vibecuda_kda_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[RecurrentKDAPrefillWorkspace],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    beta_is_logit: bool,
) -> Optional[tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Plan-cache probe for the public ``backend=\"vibecuda\"`` facade.

    Returns ``(output, final_state)`` exactly as
    :func:`_run_vibecuda_kda_prefill` when the call metadata hit a recorded
    plan, or ``None`` so the caller falls through to the generic path. The
    probe is side-effect free on a miss.
    """

    if not _FAST_PLAN_ENABLED:
        return None
    # Every feature surface outside the recorded call class keeps the
    # generic path (its errors included): explicit workspaces / CUDA graph
    # capture, spec decode, state pools and source pools, checkpoints.
    if (
        prefill_workspace is not None
        or ssm_state_indices is not None
        or num_spec_tokens is not None
        or num_accepted_tokens is not None
        or initial_state_source is not None
        or initial_state_indices is not None
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
        or checkpoint_every_n_tokens
        or A_log is None
        or dt_bias is None
        or lower_bound is None
        or not isinstance(q, torch.Tensor)
        or not q.is_cuda
        or torch.cuda.is_current_stream_capturing()
        or not _fast_plan_device_supported(q.device)
    ):
        return None
    scale_value = 1.0 / math.sqrt(_HEAD_DIM) if scale is None else float(scale)
    lower_bound_value = float(lower_bound)
    # Non-finite scalars take the generic validation/raise path.
    if not (
        math.isfinite(scale_value)
        and math.isfinite(lower_bound_value)
        and lower_bound_value < 0.0
    ):
        return None
    # The fused-N16 route never records plans; skip the key build for its
    # metadata classes so their per-call cost is unchanged.
    if q.ndim == 4 and q.shape[2] == 12:
        if cu_seqlens is None:
            if q.shape[0] == 1 and q.shape[1] <= 1024:
                return None
        elif cu_seqlens.numel() - 1 <= 6:
            return None
    key = _fast_plan_key(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output=output,
        cu_seqlens=cu_seqlens,
        seq_order=seq_order,
        scale=scale,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        beta_is_logit=beta_is_logit,
        output_final_state=output_final_state,
    )
    workspace = _get_vibecuda_stream_workspace(q.device)
    state = _vibecuda_state(workspace, q.device)
    plan = state._fast_plans.get(key)
    if plan is None:
        return None

    device_index, stream_ptr = _stream_cache_key(q.device)
    batch_size, seq_len, num_heads, _ = q.shape
    out_buf = output if output is not None else torch.empty_like(q)
    use_initial_state = initial_state is not None
    dummy_state: Optional[torch.Tensor] = None
    if use_initial_state:
        initial_state_arg = initial_state
        final_state_arg = initial_state
        store_final_state = True
        returned_state = initial_state
    else:
        dummy_state = state.dummy_state
        initial_state_arg = dummy_state
        if output_final_state:
            final_state_arg = torch.empty(
                (plan.num_sequences, num_heads, _HEAD_DIM, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=q.device,
            )
            returned_state = final_state_arg
            store_final_state = True
        else:
            final_state_arg = None
            returned_state = None
            store_final_state = False
    if plan.beta_needs_pad:
        beta_tma = _vibecuda_beta_tma_source(
            beta, state, direct_heads=plan.beta_direct_heads
        )
    else:
        beta_tma = beta.reshape(plan.beta_rows, num_heads)
    # Lean disjointness audit: eligible-plan tensors are all contiguous, so
    # each occupies exactly [ptr, ptr + numel * element_size); the raise and
    # the message match the generic helper's contract.
    out_ptr = out_buf.data_ptr()
    out_span = out_ptr + out_buf.numel() * out_buf.element_size()
    for name, t in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("initial_state", initial_state),
    ):
        if t is None:
            continue
        t_ptr = t.data_ptr()
        if t_ptr < out_span and out_ptr < t_ptr + t.numel() * t.element_size():
            raise ValueError(
                f"output must not overlap {name} for frozen recurrent_kda prefill"
            )

    if plan.fixed_layout:
        cu_seqlens_i64 = _fixed_cu_seqlens(
            device=q.device, batch_size=batch_size, seq_len=seq_len
        )
    else:
        assert cu_seqlens is not None
        cu_seqlens_i64 = cu_seqlens if plan.cu_is_int64 else cu_seqlens.to(torch.int64)
    module = plan.module
    if seq_order is not None:
        seq_order_i32 = _validate_prefill_seq_order(
            seq_order,
            fixed_layout=False,
            num_sequences=plan.num_sequences,
            device=q.device,
        )
    elif plan.fixed_layout:
        seq_order_i32 = _identity_seq_order(
            device=q.device, num_sequences=plan.num_sequences
        )
    else:
        seq_order_i32 = _vibecuda_sorted_seq_order(
            module=module,
            state=state,
            cu_seqlens_i64=cu_seqlens_i64,
            stream_ptr=stream_ptr,
        )

    variant = plan.variant
    with workspace._lock:
        _bind_workspace(
            workspace,
            device=q.device,
            stream_ptr=stream_ptr,
            capturing=False,
            explicit=False,
        )
        # Pointer-diff staleness on the plan-pinned shape/stride/dtype
        # envelope replaces the full descriptor signature tuple; a TMA
        # descriptor is a pure function of those fields plus the base
        # pointer, so differing pointers (or a missing record) force the
        # binding's descriptor rebuild and equal pointers prove it warm.
        split_buffers: Optional[tuple] = None
        if plan.split_parts >= 2:
            num_tasks = plan.num_sequences * num_heads
            split_buffers = (
                *_vibecuda_split_buffers(state, q.device, num_tasks, plan.split_parts),
                _vibecuda_split_out_buffer(state, q.device, out_buf.numel()),
                _vibecuda_split_lookback_flags(
                    state, q.device, num_tasks, plan.split_parts
                ),
            )
        ptrs: tuple[int, ...] = (
            q.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            g.data_ptr(),
            beta_tma.data_ptr(),
            out_ptr,
        )
        if split_buffers is not None:
            # The split descriptor signature covers the correction-pass
            # scratch output as well; the tuple length gap against the
            # non-split routes keeps their warm records disjoint.
            ptrs += (split_buffers[4].data_ptr(),)
        prepare_descriptors = int(state._fast_last_ptrs.get(variant) != ptrs)
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
                    plan.descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    plan.scale_value,
                    plan.lower_bound_value,
                    stream_ptr,
                )
            elif variant == "persistent":
                task_ids, task_offsets, choice = _vibecuda_persistent_planner_buffers(
                    state, q.device, plan.num_sequences * num_heads, plan.sm_count
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
                    plan.descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    plan.scale_value,
                    plan.lower_bound_value,
                    plan.sm_count,
                    stream_ptr,
                )
            elif variant.startswith("fam2h"):
                module.run_fam2(
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
                    plan.descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    plan.scale_value,
                    plan.lower_bound_value,
                    stream_ptr,
                )
            elif plan.split_parts >= 2:
                # Split-seq re-enactment: same binding and argument order as
                # the generic path; the scratch buffers are the grow-only
                # workspace tensors fetched above.
                assert split_buffers is not None
                (
                    split_state,
                    map_state,
                    carry,
                    map_state_bf16,
                    split_out,
                    lookback_flags,
                ) = split_buffers
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
                    plan.descriptor_storage,
                    split_state,
                    map_state,
                    carry,
                    split_out,
                    map_state_bf16,
                    lookback_flags,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    plan.scale_value,
                    plan.lower_bound_value,
                    plan.split_parts,
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
                    plan.descriptor_storage,
                    prepare_descriptors,
                    num_heads,
                    int(use_initial_state),
                    int(store_final_state),
                    plan.scale_value,
                    plan.lower_bound_value,
                    stream_ptr,
                )
        except Exception:
            state._fast_last_ptrs.pop(variant, None)
            raise
        state._fast_last_ptrs[variant] = ptrs
        # Keep the generic path's full-signature record conservative: a
        # later generic run for this variant (e.g. after plan eviction)
        # must rebuild rather than trust a signature the fast path advanced.
        state.descriptor_signatures.pop(variant, None)
    return out_buf, (returned_state if output_final_state else None)


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

    # Fused-N16 route (H12 short/packed classes, measured wins on both
    # targets): fixed single-sequence layouts up to 1024 tokens, and packed
    # layouts with at most six sequences (the ragged-tail regime where the
    # 16-token chunk halves the serial step and the fused single crossing
    # removes the multi-launch host path). The class predicates are all
    # host-known, so they compose with the split/persistent gates below
    # without reading cu_seqlens values.
    use_fn16 = num_heads == 12 and (
        (fixed_layout and batch_size == 1 and seq_len <= 1024)
        or (not fixed_layout and num_sequences <= 6)
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

    # Head-family fam2 route: compile-time-H single-kernel per-(sequence,
    # head) recurrence on the evolved slab schedule; replaces the M128 and
    # persistent routes for the supported head counts. Measured per target:
    # heads 12/64/96 on CC 10.0, heads 64/96 on CC 10.3 (H12 short classes
    # use the fused-N16 route above on both targets).
    if (
        split_parts == 1
        and num_heads in _FAM2_HEADS_BY_TARGET[target]
        and variant in ("m128", "persistent")
    ):
        variant = f"fam2h{num_heads}"

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

    stream_ptr = _stream_cache_key(q.device)[1]
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
            if not output_final_state:
                final_state_arg = dummy_state
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
        lookback_flags = None
        if split_parts >= 2:
            split_state, map_state, carry, map_state_bf16 = _vibecuda_split_buffers(
                state,
                q.device,
                num_sequences * num_heads,
                split_parts,
            )
            split_out = _vibecuda_split_out_buffer(state, q.device, out_buf.numel())
            lookback_flags = _vibecuda_split_lookback_flags(
                state, q.device, num_sequences * num_heads, split_parts
            )
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

        beta_tma = _vibecuda_beta_tma_source(
            beta,
            state,
            direct_heads=(
                use_fn16 or variant in ("m128", "fam2h12", "fam2h64", "fam2h96")
            ),
        )
        if use_fn16:
            # Fused-N16 route: one host call enqueues the prepare kernel
            # (factors q/k/g/beta into the chain workspace) and the
            # recurrence-chain kernel. The capacity upper bound for the exact
            # 16-token chunk total is padded by one chunk per sequence; the
            # device-side inline planner maps capacity-pad chunks to the last
            # sequence, so capacity-sized buffers are sufficient.
            total_tokens = batch_size * seq_len
            total_chunks = total_tokens // _FN16_CHUNK_TOKENS + num_sequences
            qd, kd_ws, w_ws, qk_ws, diag_ws = _vibecuda_fn16_buffers(
                state, q.device, num_heads, total_chunks
            )
            desc_prepare = state.descriptor_storages["fn16_prepare"]
            desc_chain = state.descriptor_storages["fn16_chain"]
            sig_prepare = tuple(
                _kda_prefill._tensor_descriptor_signature(t)
                for t in (q, k, g, beta, qd, kd_ws, w_ws)
            ) + (total_tokens, total_chunks, num_heads)
            sig_chain = tuple(
                _kda_prefill._tensor_descriptor_signature(t)
                for t in (qd, kd_ws, w_ws, qk_ws, diag_ws, v, out_buf)
            ) + (total_tokens, total_chunks, num_heads)
            prep_desc = int(
                state.descriptor_signatures.get("fn16_prepare") != sig_prepare
            )
            chain_desc = int(state.descriptor_signatures.get("fn16_chain") != sig_chain)
            if capturing and (prep_desc or chain_desc):
                raise RuntimeError(
                    "RecurrentKDAPrefillWorkspace is not warmed for the "
                    "exact vibecuda fused-N16 descriptor signatures; eagerly "
                    "invoke the same call on this stream before capture"
                )
            prepare_ctas = _vibecuda_fn16_prepare_ctas(
                num_heads, total_chunks, sm_count
            )
            schedule = _vibecuda_fn16_chain_schedule(
                num_sequences * num_heads, sm_count
            )
            try:
                module.run_bt16_fused(
                    q,
                    k,
                    g,
                    beta,
                    A_log,
                    dt_bias,
                    cu_seqlens_i64,
                    seq_order_i32,
                    v,
                    initial_state_arg,
                    out_buf,
                    final_state_arg,
                    qd,
                    kd_ws,
                    w_ws,
                    qk_ws,
                    diag_ws,
                    desc_prepare,
                    desc_chain,
                    prep_desc,
                    chain_desc,
                    total_chunks,
                    num_heads,
                    float(lower_bound),
                    prepare_ctas,
                    schedule,
                    int(use_initial_state),
                    int(store_final_state),
                    scale_value,
                    2 * num_sequences * num_heads,
                    stream_ptr,
                )
            except Exception:
                if prep_desc:
                    state.descriptor_signatures.pop("fn16_prepare", None)
                if chain_desc:
                    state.descriptor_signatures.pop("fn16_chain", None)
                raise
            if prep_desc:
                state.descriptor_signatures["fn16_prepare"] = sig_prepare
            if chain_desc:
                state.descriptor_signatures["fn16_chain"] = sig_chain
            if capturing and explicit_workspace:
                workspace._captured = True
            return out_buf, (returned_state if output_final_state else None)
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
            elif variant.startswith("fam2h"):
                module.run_fam2(
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
                    lookback_flags,
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
        # A generic (plan-miss) run may only have refreshed the descriptors
        # for ITS pointers; the fast path's pointer record for this variant
        # is not proven by it, so force the next fast hit to rebuild.
        state._fast_last_ptrs.pop(variant, None)
        if (
            _FAST_PLAN_ENABLED
            and not capturing
            and not explicit_workspace
            and _fast_plan_device_supported(q.device)
        ):
            record_key = _fast_plan_key(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=A_log,
                dt_bias=dt_bias,
                initial_state=initial_state,
                output=output,
                cu_seqlens=cu_seqlens,
                seq_order=seq_order,
                scale=scale,
                lower_bound=lower_bound,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                beta_is_logit=True,
                output_final_state=output_final_state,
            )
            if record_key is not None:
                total_tokens = beta.shape[0] * beta.shape[1]
                padded_tokens = max(total_tokens, 32)
                aligned_heads = (
                    (num_heads + _BETA_TMA_MIN_HEADS - 1)
                    // _BETA_TMA_MIN_HEADS
                    * _BETA_TMA_MIN_HEADS
                )
                padded_heads = max(aligned_heads, _BETA_TMA_MIN_HEADS)
                beta_direct_heads = variant in ("m128", "fam2h12", "fam2h64", "fam2h96")
                beta_needs_pad = not (
                    (beta_direct_heads and num_heads % _BETA_TMA_MIN_HEADS != 0)
                    or (padded_tokens == total_tokens and padded_heads == num_heads)
                )
                _record_fast_plan(
                    state=state,
                    key=record_key,
                    variant=variant,
                    module=module,
                    num_heads=num_heads,
                    num_sequences=num_sequences,
                    fixed_layout=fixed_layout,
                    sm_count=sm_count,
                    descriptor_storage=descriptor_storage,
                    cu_is_int64=(fixed_layout or cu_seqlens.dtype == torch.int64),
                    beta_direct_heads=beta_direct_heads,
                    beta_needs_pad=beta_needs_pad,
                    beta_rows=total_tokens,
                    scale_value=scale_value,
                    lower_bound_value=float(lower_bound),
                    use_initial_state=use_initial_state,
                    output_final_state=output_final_state,
                    split_parts=split_parts,
                )
        if capturing and explicit_workspace:
            workspace._captured = True
    return out_buf, (returned_state if output_final_state else None)
