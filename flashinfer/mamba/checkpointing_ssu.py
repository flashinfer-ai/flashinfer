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

import functools
from typing import Any, Optional

import torch

from ..api_logging import flashinfer_api
from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
    autotuner_initializer_empty,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..jit.mamba.checkpointing_ssu import gen_checkpointing_ssu_module
from ..utils import register_custom_op, register_fake_op


_ALGORITHM_AUTO = 0
_ALGORITHM_MONOLITH = 1
_ALGORITHM_TWO_KERNEL = 2
_AUTOTUNE_CACHE_SCHEMA_VERSION = 3

# A tactic is (main pipeline stages, main CTAs/SM, precompute heads/CTA).
# The all-zero tuple selects the monolithic kernel.
_CheckpointingSSUTactic = tuple[int, int, int]


@functools.cache
def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _get_module(
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    state_scale_dtype: Optional[torch.dtype],
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    num_groups: int,
    philox_rounds: int = 0,
    enable_pdl: bool = False,
):
    return gen_checkpointing_ssu_module(
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale_dtype,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        num_groups,
        philox_rounds,
        enable_pdl,
    ).build_and_load()


def _heads_per_cta_candidates(heads_per_group: int) -> tuple[int, ...]:
    """Return valid divisors on the ``HEADS_PER_GROUP >> k`` launch chain."""
    candidates = []
    value = heads_per_group
    while value > 0:
        if heads_per_group % value == 0:
            candidates.append(value)
        value >>= 1
    return tuple(candidates)


def _make_tactics(heads_per_group: int) -> tuple[_CheckpointingSSUTactic, ...]:
    """Build the explicit ReplaySSM tactic space."""
    return (
        (0, 0, 0),
        *(
            (stages, ctas_per_sm, heads_per_cta)
            for stages in (1, 2)
            for ctas_per_sm in (1, 2, 4, 8, 16)
            for heads_per_cta in _heads_per_cta_candidates(heads_per_group)
        ),
    )


def allocate_checkpointing_ssu_scratch(
    batch_size: int,
    num_heads: int,
    num_predicted_tokens: int,
    max_window: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate CUDA-graph-safe scratch for the two-kernel ReplaySSM path."""
    if batch_size <= 0 or num_heads <= 0:
        raise ValueError("batch_size and num_heads must be positive")
    if not 0 < num_predicted_tokens <= 16:
        raise ValueError("num_predicted_tokens must be in [1, 16]")
    if not 0 < max_window <= 16:
        raise ValueError("max_window must be in [1, 16]")

    token_pad = ((num_predicted_tokens + 15) // 16) * 16
    old_token_pad = ((max_window + 7) // 8) * 8
    cb_scaled = torch.empty(
        batch_size,
        num_heads,
        32,
        token_pad // 2,
        dtype=dtype,
        device=device,
    )
    cumAdt_vec = torch.empty(
        batch_size,
        num_heads,
        token_pad,
        dtype=torch.float32,
        device=device,
    )
    cb_old = torch.empty(
        batch_size,
        num_heads,
        32,
        old_token_pad // 2,
        dtype=dtype,
        device=device,
    )
    return cb_scaled, cumAdt_vec, cb_old


def _profile_cache_capacity(shapes: tuple[tuple[int, ...], ...]) -> int:
    """Use one private padding slot beyond the bucketed dense batch."""
    return shapes[1][0] + 1


def _profile_batch_size(shapes: tuple[tuple[int, ...], ...]) -> int:
    return shapes[1][0]


def _initialize_broadcast_dt(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    base = torch.zeros(shape[:-1], dtype=dtype, device=device)
    return base.unsqueeze(-1).expand(shape)


def _initialize_state_batch_indices(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.arange(1, shape[0] + 1, dtype=dtype, device=device)


def _prepare_checkpointing_ssu_profile_inputs(inputs: list[Any]) -> list[Any]:
    """Install valid cache indices and a mixed replay history."""
    state, x, x_cache = inputs[0], inputs[1], inputs[7]
    ring_start, prev_num_accepted_tokens = inputs[10], inputs[11]
    state_batch_indices = inputs[15]
    batch = x.size(0)
    if state_batch_indices is not None:
        state_batch_indices.copy_(
            torch.arange(1, batch + 1, device=state.device).to(
                state_batch_indices.dtype
            )
        )
    slots = torch.arange(state.size(0), device=state.device)
    ring_start.copy_((slots % x_cache.size(2)).to(ring_start.dtype))
    max_window = x_cache.size(2) - x.size(1)
    prev_num_accepted_tokens.copy_(
        ((slots * max_window) % (max_window + 1)).to(prev_num_accepted_tokens.dtype)
    )
    return inputs


def _checkpointing_ssu_tuning_config(inputs: list[Any]) -> TuningConfig:
    """Profile dense decode buckets with compact private state caches."""
    cache_capacity_inputs = [0, 7, 8, 9, 10, 11]
    if inputs[16] is not None:
        cache_capacity_inputs.append(16)
    constraints = [
        ConstraintSpec(
            input_idx=index,
            dim_idx=0,
            infer_shape=_profile_cache_capacity,
        )
        for index in cache_capacity_inputs
    ]
    if inputs[15] is not None:
        constraints.append(
            ConstraintSpec(
                input_idx=15,
                dim_idx=0,
                infer_shape=_profile_batch_size,
            )
        )

    dynamic_specs: tuple[DynamicTensorSpec, ...] = ()
    tensor_initializers: list[tuple[int, Any]] = []
    if inputs[18] is None:
        batch_inputs = [1, 2, 4, 5, 6, 19, 20, 21]
        if inputs[13] is not None:
            batch_inputs.append(13)
        if inputs[15] is not None:
            batch_inputs.append(15)
        dynamic_specs = (
            DynamicTensorSpec(
                input_idx=tuple(batch_inputs),
                dim_idx=(0,) * len(batch_inputs),
                gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
            ),
        )
        tensor_initializers.extend(
            (
                (2, _initialize_broadcast_dt),
                (6, autotuner_initializer_empty),
                (19, autotuner_initializer_empty),
                (20, autotuner_initializer_empty),
                (21, autotuner_initializer_empty),
            )
        )
        if inputs[15] is not None:
            tensor_initializers.append((15, _initialize_state_batch_indices))

    profile_arena_candidates = (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 19, 20, 21)
    profile_arena_inputs = tuple(
        index
        for index in profile_arena_candidates
        if isinstance(inputs[index], torch.Tensor) and inputs[index].is_contiguous()
    )

    return TuningConfig(
        dynamic_tensor_specs=dynamic_specs,
        constraint_specs=tuple(constraints),
        tensor_initializers=tuple(tensor_initializers),
        use_cold_l2_cache=True,
        use_cuda_graph=True,
        profile_arena_input_indices=profile_arena_inputs,
        inputs_pre_hook=_prepare_checkpointing_ssu_profile_inputs,
    )


class CheckpointingSSURunner(TunableRunner):
    """Runtime ReplaySSM tactic runner following FlashInfer's tuner contract."""

    def __init__(
        self,
        module_base_args: tuple[Any, ...],
        *,
        dt_softplus: bool,
        pad_slot_id: int,
        requested_algorithm: int,
        requested_d_split: int,
        precompute_heads_per_cta: int,
        heads_per_group: int,
    ) -> None:
        self._module_base_args = module_base_args
        self._dt_softplus = dt_softplus
        self._pad_slot_id = pad_slot_id
        self._requested_algorithm = requested_algorithm
        self._requested_d_split = requested_d_split
        self._precompute_heads_per_cta = precompute_heads_per_cta
        self._tactics = _make_tactics(heads_per_group)

    def __hash__(self) -> int:
        return hash(
            (
                self._module_base_args,
                self._dt_softplus,
                self._pad_slot_id,
                self._requested_algorithm,
                self._requested_d_split,
                self._precompute_heads_per_cta,
                tuple(self._tactics),
            )
        )

    @staticmethod
    def _batch(inputs: list[Any]) -> int:
        cu_seqlens = inputs[18]
        return cu_seqlens.numel() - 1 if cu_seqlens is not None else inputs[1].size(0)

    @staticmethod
    def _two_kernel_supported(inputs: list[Any]) -> bool:
        state, x = inputs[0], inputs[1]
        return (
            inputs[19] is not None
            and inputs[20] is not None
            and inputs[21] is not None
            and state.element_size() in (2, 4)
            and x.element_size() == 2
        )

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[_CheckpointingSSUTactic]:
        del profile
        if not self._two_kernel_supported(inputs):
            return [self._tactics[0]]
        return list(self._tactics)

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[Any, ...]:
        stride_signature = tuple(
            None if inputs[index] is None else tuple(inputs[index].stride()[1:])
            for index in (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 19, 20, 21)
        )
        return (
            _AUTOTUNE_CACHE_SCHEMA_VERSION,
            self._module_base_args,
            self._dt_softplus,
            self._pad_slot_id,
            self._requested_d_split,
            tuple(inputs[index] is not None for index in (12, 13, 14, 15, 16, 17, 18)),
            stride_signature,
        )

    def _resolve_fallback_algorithm(self, inputs: list[Any]) -> int:
        if self._requested_algorithm != _ALGORITHM_AUTO:
            return self._requested_algorithm
        state, x = inputs[0], inputs[1]
        if self._two_kernel_supported(inputs) and self._batch(inputs) * state.size(
            1
        ) >= _sm_count(x.device):
            return _ALGORITHM_TWO_KERNEL
        return _ALGORITHM_MONOLITH

    def _resolve_d_split(self, inputs: list[Any], algorithm: int) -> int:
        if self._requested_d_split != 0:
            return self._requested_d_split
        state = inputs[0]
        dim = state.size(2)
        d_split = 1
        if (
            algorithm == _ALGORITHM_MONOLITH
            and state.dtype == torch.float32
            and dim % 2 == 0
            and dim // 2 >= 32
            and self._batch(inputs) * state.size(1) <= 8 * _sm_count(state.device)
        ):
            d_split = 2
        return d_split

    def forward(
        self,
        inputs: list[Any],
        tactic: _CheckpointingSSUTactic | int = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if do_preparation:
            _get_module(*self._module_base_args)
            return

        if tactic == -1:
            algorithm = self._resolve_fallback_algorithm(inputs)
            precompute_heads_per_cta = self._precompute_heads_per_cta
            main_pipeline_stages = main_ctas_per_sm = 0
        else:
            if not isinstance(tactic, tuple) or tactic not in self._tactics:
                raise ValueError(f"Unknown checkpointing SSU tactic: {tactic}")
            main_pipeline_stages, main_ctas_per_sm, precompute_heads_per_cta = tactic
            algorithm = (
                _ALGORITHM_MONOLITH
                if tactic == self._tactics[0]
                else _ALGORITHM_TWO_KERNEL
            )

        two_kernel = algorithm == _ALGORITHM_TWO_KERNEL
        if two_kernel and not self._two_kernel_supported(inputs):
            raise ValueError(
                "two-kernel checkpointing SSU requires its scratch trio, "
                "2-byte input, and 2- or 4-byte state"
            )
        d_split = self._resolve_d_split(inputs, algorithm)
        module = _get_module(*self._module_base_args)
        module.checkpointing_ssu(
            *inputs[:15],
            self._dt_softplus,
            inputs[15],
            self._pad_slot_id,
            inputs[16],
            inputs[17],
            d_split,
            inputs[18],
            inputs[19] if two_kernel else None,
            inputs[20] if two_kernel else None,
            inputs[21] if two_kernel else None,
            precompute_heads_per_cta,
            main_pipeline_stages,
            main_ctas_per_sm,
        )


@register_custom_op(
    "flashinfer::checkpointing_ssu",
    mutates_args=(
        "state",
        "out",
        "x_cache",
        "B_cache",
        "dt_cache",
        "state_scale",
        # Two-kernel scratch — the precompute writes them.
        "cb_scaled",
        "cumAdt_vec",
        "cb_old",
    ),
)
def _checkpointing_ssu(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    d_split: int,
    cu_seqlens: Optional[torch.Tensor],
    cb_scaled: Optional[torch.Tensor],
    cumAdt_vec: Optional[torch.Tensor],
    cb_old: Optional[torch.Tensor],
    precompute_heads_per_cta: int,
    main_pipeline_stages: int,
    main_ctas_per_sm: int,
    algorithm: int,
    enable_pdl: bool,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    num_groups: int,
) -> None:
    """Internal function registered with torch.library for torch.compile() support."""
    module_base_args = (
        state_dtype,
        input_dtype,
        dt_dtype,
        weight_dtype,
        matrixA_dtype,
        stateIndex_dtype,
        state_scale.dtype if state_scale is not None else None,
        dim,
        dstate,
        npredicted,
        max_window,
        heads_per_group,
        num_groups,
        philox_rounds,
        enable_pdl,
    )
    inputs = [
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        D,
        z,
        dt_bias,
        state_batch_indices,
        state_scale,
        rand_seed,
        cu_seqlens,
        cb_scaled,
        cumAdt_vec,
        cb_old,
    ]
    runner = CheckpointingSSURunner(
        module_base_args,
        dt_softplus=dt_softplus,
        pad_slot_id=pad_slot_id,
        requested_algorithm=algorithm,
        requested_d_split=d_split,
        precompute_heads_per_cta=precompute_heads_per_cta,
        heads_per_group=heads_per_group,
    )

    # Explicit launch controls are an opt-out used by tests and benchmark
    # sweeps.  Pure "auto" calls with two-kernel scratch follow the same
    # TunableRunner/choose_one contract as other FlashInfer tuned operations.
    tune = (
        algorithm == _ALGORITHM_AUTO
        and cb_scaled is not None
        and cumAdt_vec is not None
        and cb_old is not None
        and precompute_heads_per_cta == 0
        and main_pipeline_stages == 0
        and main_ctas_per_sm == 0
        and cu_seqlens is None
    )
    if tune:
        runner, tactic = AutoTuner.get().choose_one(
            "checkpointing_ssu",
            [runner],
            _checkpointing_ssu_tuning_config(inputs),
            inputs,
        )
        runner(inputs, tactic=tactic)
    else:
        runner(inputs, tactic=-1)


@register_fake_op("flashinfer::checkpointing_ssu")
def _checkpointing_ssu_fake(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    dt_softplus: bool,
    state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    state_scale: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    d_split: int,
    cu_seqlens: Optional[torch.Tensor],
    cb_scaled: Optional[torch.Tensor],
    cumAdt_vec: Optional[torch.Tensor],
    cb_old: Optional[torch.Tensor],
    precompute_heads_per_cta: int,
    main_pipeline_stages: int,
    main_ctas_per_sm: int,
    algorithm: int,
    enable_pdl: bool,
    philox_rounds: int,
    state_dtype: torch.dtype,
    input_dtype: torch.dtype,
    dt_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    matrixA_dtype: torch.dtype,
    stateIndex_dtype: torch.dtype,
    dim: int,
    dstate: int,
    npredicted: int,
    max_window: int,
    heads_per_group: int,
    num_groups: int,
) -> None:
    """Fake implementation for torch.compile() meta tensor propagation."""
    pass


@flashinfer_api
def checkpointing_ssu(
    state: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    state_batch_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = -1,
    state_scale: Optional[torch.Tensor] = None,
    rand_seed: Optional[torch.Tensor] = None,
    philox_rounds: int = 10,
    d_split: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    enable_pdl: bool = False,
    cb_scaled: Optional[torch.Tensor] = None,
    cumAdt_vec: Optional[torch.Tensor] = None,
    cb_old: Optional[torch.Tensor] = None,
    precompute_heads_per_cta: int = 0,
    algorithm: str = "auto",
) -> torch.Tensor:
    """Checkpointing SSU with MTP replay using matmul-based parallel token processing.

    Parameters
    ----------
    state : torch.Tensor
        SSM state, shape (state_cache_size, nheads, dim, dstate). Updated in-place.
    x_cache : torch.Tensor
        Ring of cached x, shape (state_cache_size, nheads, RING_BUFFER_LEN, dim).
        RING_BUFFER_LEN is implicit (= size(2)); the LOGICAL replay window is
        max_window = RING_BUFFER_LEN - T (flush rule pnat + 2T > RING_BUFFER_LEN).
    B_cache : torch.Tensor
        Ring of cached B, shape (state_cache_size, ngroups, RING_BUFFER_LEN, dstate).
    dt_cache : torch.Tensor
        Ring of cached processed dt, shape (state_cache_size, nheads,
        RING_BUFFER_LEN), f32.  Replay decays are recomputed from it (no
        cumAdt is cached — prefix sums are not ring-shift-invariant).
    ring_start : torch.Tensor
        Ring head per slot (oldest live row), shape (state_cache_size,), int32.
        The HOST owns bookkeeping: advance by the replayed count on flush.
    prev_num_accepted_tokens : torch.Tensor
        Number of old tokens to replay, shape (state_cache_size,), int32.
    x : torch.Tensor
        New token inputs, shape (batch, T, nheads, dim).
    dt : torch.Tensor
        Delta time, shape (batch, T, nheads, dim) with tie_hdim (stride[-1]=0).
        Accepted in native dtype (e.g. bf16) — converted to f32 internally.
    A : torch.Tensor
        Decay rate, shape (nheads, dim, dstate) with tie_hdim.
    B : torch.Tensor
        Input projection, shape (batch, T, ngroups, dstate).
    C : torch.Tensor
        Output projection, shape (batch, T, ngroups, dstate).
    out : torch.Tensor
        Preallocated output, shape (batch, T, nheads, dim).
    D : Optional[torch.Tensor]
        Skip connection, shape (nheads, dim).
    z : Optional[torch.Tensor]
        Gate, shape (batch, T, nheads, dim).
    dt_bias : Optional[torch.Tensor]
        Bias added to dt, shape (nheads, dim) with tie_hdim.
    dt_softplus : bool
        Whether to apply softplus to dt.
    state_batch_indices : Optional[torch.Tensor]
        Maps batch index to cache slot, shape (batch,), int32 | int64.
    pad_slot_id : int
        Sentinel value for padded entries.
    state_scale : Optional[torch.Tensor]
        Block-scale decode factors for quantized state, shape (state_cache_size, nheads, dim), f32.
    rand_seed : Optional[torch.Tensor]
        Single-element int64 CUDA tensor for stochastic rounding seed.
    philox_rounds : int
        Philox PRNG rounds for stochastic rounding (default 10).
    d_split : Optional[int]
        Per-head DIM split factor.  This is only exposed for benchmarking.
        Do not use it cause it will make things slow.
    cu_seqlens : Optional[torch.Tensor]
        Cumulative sequence boundaries for packed variable-length input, as a
        one-dimensional int32 CUDA tensor of shape ``(batch + 1,)``.  It must
        start at 0, be monotonically nondecreasing, and end at ``total_tokens``.
        When provided, ``x`` must have shape
        ``(1, total_tokens, nheads, dim)`` and ``max_seqlen`` is required.
    max_seqlen : Optional[int]
        Upper bound on every packed sequence length.  In variable-length mode
        this is the JIT-specialized predicted-token count used to derive the
        logical replay window from the ring-buffer size.  Must be ``None`` when
        ``cu_seqlens`` is not provided.
    precompute_heads_per_cta : int
        Two-kernel PRECOMPUTE head-tiling: heads per precompute CTA.  0 (default) uses the
        launcher's co-residency heuristic; >0 overrides it (must divide nheads/ngroups,
        snapped to the HEADS_PER_GROUP>>k chain).  Tuning knob — two-kernel path only.
    algorithm : str
        Kernel selection: ``"auto"`` (default), ``"monolith"``, or ``"two-kernel"``.
        With the scratch trio and no explicit tuning knobs, ``"auto"`` uses
        FlashInfer's cached autotuner tactic.  Inside an ``autotune(True)``
        context, it profiles monolithic against every supported combination of
        precompute heads/CTA, main pipeline stages, and main CTAs/SM.  Without a
        cached tactic it retains the production fallback: use the split when
        ``batch * nheads >= sm_count`` and otherwise use monolithic.
        ``"two-kernel"`` forces the split (scratch trio required), while
        ``"monolith"`` forces the monolithic kernel (scratch ignored).
        Benches and tests that must pin a path should force it.
    enable_pdl : bool
        When True the kernel is launched with
        `cudaLaunchAttributeProgrammaticStreamSerialization`, enabling the
        in-kernel `griddepcontrol.{wait,launch_dependents}` PTX to gate on
        the upstream (e.g. conv1d) and signal the downstream kernel.
        Caller's responsibility: upstream/downstream kernels must also be
        PDL-paired for the wait/signal to have effect.  Defaults to False.
    cb_scaled : Optional[torch.Tensor]
        Pre-allocated input-dtype (same as ``x``) scratch for the precomputed
        new-token CB matrix, fragment-native layout
        (batch, nheads, WARP_SIZE, MMA_FRAG_SIZE) — each (batch, head)'s CB is
        one m16n8k16 MMA A-fragment stored as [warp lane, register].  Providing
        it (together with ``cumAdt_vec`` / ``cb_old``) makes the
        **two-kernel** (precompute + main) path available — ``algorithm`` decides
        whether it runs; leaving all four ``None`` always runs the monolithic
        kernel.  Caller-allocated so the path is CUDA-graph-safe (no in-wrapper
        allocation, like ``out``).
    cumAdt_vec : Optional[torch.Tensor]
        Pre-allocated fp32 scratch for the per-head raw cumAdt vector, shape
        (batch, nheads, T_pad); the main kernel exponentiates it on the fly to
        get the decay/β factor.  Must be provided iff ``cb_scaled`` is.
    cb_old : Optional[torch.Tensor]
        Pre-allocated input-dtype (same as ``x``) scratch for the precomputed
        old-token CB matrix, fragment-native layout
        (batch, nheads, WARP_SIZE, K_old // 2) where
        K_old = next_multiple_of_8(max_window) — the m16n8k{K_old} MMA
        A-fragment consumed on the no-write (replay) path, stored as
        [warp lane, register].  Must be provided iff ``cb_scaled`` is.

    Returns
    -------
    out : torch.Tensor
        Output tensor, shape (batch, T, nheads, dim).
    """
    # Validate quantized state ↔ state_scale combo.
    # int8 and fp8_e4m3fn use a per-(cache, head, dim) decode-scale tensor
    # (QUANT_MAX = 127 and 448 respectively).  Non-quantized dtypes must NOT
    # pass one (the kernel hardcodes the dispatch on whether `state_scale_t`
    # is `void`).
    _quantized_state_dtypes = (torch.int8, torch.float8_e4m3fn)
    if state.dtype in _quantized_state_dtypes:
        assert state_scale is not None, (
            f"state.dtype={state.dtype} requires a state_scale tensor "
            f"of shape (cache, nheads, dim) and dtype float32"
        )
        cache_size, nheads_state, dim_state = (
            state.size(0),
            state.size(1),
            state.size(2),
        )
        assert state_scale.shape == (cache_size, nheads_state, dim_state), (
            f"state_scale shape mismatch: expected "
            f"{(cache_size, nheads_state, dim_state)}, got {tuple(state_scale.shape)}"
        )
        assert state_scale.dtype == torch.float32, (
            f"state_scale must be float32 (got {state_scale.dtype})"
        )
        assert state_scale.is_cuda, "state_scale must be a CUDA tensor"
        # The 8-bit replay path uses a per-warp M-shard layout
        # (Layout<_4, _1>) that requires per-warp M = D_PER_CTA / 4 ≥ 16
        # (m16n8 atom M).  → D_PER_CTA ≥ 64 → d_split == 1.  d_split == 2
        # would give D_PER_CTA = 32 = 8 per warp, which doesn't fit the atom.
        assert d_split == 1 or d_split is None, (
            f"8-bit state (int8/fp8) requires d_split=1 (got d_split={d_split}); "
            f"the M-shard-per-warp layout needs D_PER_CTA / 4 >= 16."
        )
    else:
        assert state_scale is None, (
            f"state_scale must be None for non-quantized state.dtype={state.dtype}"
            f" (allowed quantized dtypes: {_quantized_state_dtypes})"
        )

    # Validate rand_seed / philox_rounds
    if rand_seed is not None:
        assert isinstance(rand_seed, torch.Tensor), (
            "rand_seed must be a CUDA int64 tensor"
        )
        assert rand_seed.numel() == 1, (
            f"rand_seed must be single-element, got {rand_seed.numel()}"
        )
        assert rand_seed.dtype == torch.int64, (
            f"rand_seed must be int64, got {rand_seed.dtype}"
        )
        assert rand_seed.is_cuda, "rand_seed must be a CUDA tensor"
        assert philox_rounds > 0, (
            f"philox_rounds must be > 0 with rand_seed, got {philox_rounds}"
        )
    else:
        philox_rounds = 0

    # Extract JIT specialization keys
    dim = state.size(2)
    dstate = state.size(3)
    # Varlen: inputs are packed (1, total_tokens, ...) — `x.size(1)` is no
    # longer a JIT key (it varies per call).  The caller must promise an
    # upper bound on every cu_seqlens[i+1] - cu_seqlens[i] via `max_seqlen`,
    # which becomes the JIT-stamped NPREDICTED.  REQUIRED under the ring
    # contract: RING_BUFFER_LEN = max_window + NPREDICTED, so without an
    # explicit T the split of the ring row count is underdetermined.
    if cu_seqlens is not None:
        assert x.dim() == 4 and x.size(0) == 1, (
            f"varlen mode: x must be (1, total_tokens, nheads, dim), got shape {tuple(x.shape)}"
        )
        assert cu_seqlens.dim() == 1 and cu_seqlens.dtype == torch.int32, (
            f"cu_seqlens must be a 1D int32 CUDA tensor, got shape "
            f"{tuple(cu_seqlens.shape)} dtype {cu_seqlens.dtype}"
        )
        assert cu_seqlens.is_cuda, "cu_seqlens must be a CUDA tensor"
        # The persistent main's meta ring packs (bos << 8 | seq_len) into one
        # int32 (kernel_checkpointing_ssu_main.cuh meta_cu), capping the packed
        # token offset at 2^23 - 1.
        assert x.size(1) < (1 << 23), (
            f"varlen total_tokens={x.size(1)} exceeds the packed meta_cu bos "
            f"capacity (must be < {1 << 23})"
        )
        assert max_seqlen is not None, (
            "varlen mode requires max_seqlen under the ring contract "
            "(RING_BUFFER_LEN = max_window + max_seqlen is otherwise ambiguous)"
        )
        npredicted = max_seqlen
    else:
        assert max_seqlen is None, (
            "max_seqlen is only valid with cu_seqlens (varlen mode); for "
            "non-varlen the JIT key is taken from x.size(1)"
        )
        npredicted = x.size(1)
    # LOGICAL replay window from the implicit ring length (ReplaySSM contract).
    max_window = x_cache.size(2) - npredicted
    assert max_window <= 16, (
        f"checkpointing_ssu supports at most 16 cache tokens (max_window), got {max_window}"
    )
    assert npredicted <= max_window, (
        f"npredicted ({npredicted}) must be <= max_window ({max_window})"
    )

    # ── Monolith vs two-kernel split (auto unless forced) ──
    # The split is AVAILABLE iff the caller provides the scratch quartet —
    # graph-safe, the caller pre-allocates like `out` (no wrapper allocation).
    # All three or none: cb_scaled (C5) + cumAdt_vec (β) are produced on both
    # paths; cb_old (C6) is consumed on the no-write path, which the wrapper
    # can't predict per-slot.  (Old decay is recomputed in-registers by the
    # main from the dt ring — no scratch carries it.)
    # The launcher routes on params.cb_scaled != nullptr.
    scratch_provided = cb_scaled is not None
    if scratch_provided != (cumAdt_vec is not None) or scratch_provided != (
        cb_old is not None
    ):
        raise ValueError(
            "cb_scaled, cumAdt_vec, and cb_old must be provided together "
            f"(they make the two-kernel path available); got "
            f"cb_scaled set={cb_scaled is not None}, "
            f"cumAdt_vec set={cumAdt_vec is not None}, cb_old set={cb_old is not None}"
        )
    nheads = state.size(1)
    assert algorithm in ("auto", "monolith", "two-kernel"), (
        f"algorithm must be one of 'auto', 'monolith', 'two-kernel'; got {algorithm!r}"
    )
    algorithm_int = {
        "auto": _ALGORITHM_AUTO,
        "monolith": _ALGORITHM_MONOLITH,
        "two-kernel": _ALGORITHM_TWO_KERNEL,
    }[algorithm]
    if algorithm_int == _ALGORITHM_TWO_KERNEL and not scratch_provided:
        raise ValueError(
            "algorithm='two-kernel' requires the cb_scaled/cumAdt_vec/cb_old/"
            "scratch trio (got none) — allocate them or use 'auto'/'monolith'"
        )

    # ── d_split selection (v12 §59) ──
    # Auto-heuristic, measured on B200 (mixed-batch bench): d_split=2 pays
    # only when BOTH hold —
    #   (a) f32 state: the per-CTA state load (dim/d_split × dstate × 4 B) is
    #       the small-batch latency pole; halving it cut mixed b1 13 %
    #       (5.73 → 4.99 µs) and won through b64.  2-byte state is half as
    #       long already — splitting only buys duplicated B/C/x traffic and
    #       idle output-MMA warps (bf16 regressed at every batch size).
    #   (b) the d_split=1 grid (batch × nheads CTAs) underfills the GPU.
    #       Crossover measured between b64 (win) and b128 (loss) at
    #       nheads=16 on 148 SMs → threshold 8 × SM count.
    # d_split=4 is deferred to v12.x (needs warp-count restructure for
    # output MMA).
    requested_d_split = 0 if d_split is None else d_split
    if requested_d_split != 0:
        assert requested_d_split in (1, 2), (
            "d_split must be in {1, 2} for v12 "
            f"(d_split=4 deferred), got {requested_d_split}"
        )
        assert dim % requested_d_split == 0, (
            f"dim={dim} must be divisible by d_split={requested_d_split}"
        )
        assert dim // requested_d_split >= 32, (
            f"d_split={requested_d_split} gives "
            f"D_PER_CTA={dim // requested_d_split} < 32 "
            "(output MMA m16n8 floor with _1×4 warp layout)"
        )

    stateIndex_dtype = torch.int32
    if state_batch_indices is not None:
        stateIndex_dtype = state_batch_indices.dtype

    # HEADS_PER_GROUP is JIT-stamped (was a runtime `dispatchRatio` over 7
    # candidate values).  Stamping it as a constexpr means each .so compiles
    # exactly one specialization instead of seven — ~7x faster per JIT.
    # The kernel asserts `params.nheads / params.ngroups == HEADS_PER_GROUP`
    # before launch.
    ngroups = B.size(-2)
    assert nheads % ngroups == 0, (
        f"nheads ({nheads}) must be divisible by ngroups ({ngroups})"
    )
    heads_per_group = nheads // ngroups

    # D and dt_bias share a single JIT `weight_dtype` specialization.  If
    # both are present, the dtypes must match — otherwise the kernel will
    # read one of them as the wrong type.
    if D is not None and dt_bias is not None:
        assert D.dtype == dt_bias.dtype, (
            f"D.dtype ({D.dtype}) and dt_bias.dtype ({dt_bias.dtype}) must match"
        )
    weight_dtype = (
        D.dtype
        if D is not None
        else (dt_bias.dtype if dt_bias is not None else dt.dtype)
    )

    _checkpointing_ssu(
        state,
        x,
        dt,
        A,
        B,
        C,
        out,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        D,
        z,
        dt_bias,
        dt_softplus,
        state_batch_indices,
        pad_slot_id,
        state_scale,
        rand_seed,
        requested_d_split,
        cu_seqlens,
        cb_scaled,
        cumAdt_vec,
        cb_old,
        precompute_heads_per_cta,
        0,
        0,
        algorithm_int,
        enable_pdl,
        philox_rounds=philox_rounds,
        state_dtype=state.dtype,
        input_dtype=x.dtype,
        dt_dtype=dt.dtype,
        weight_dtype=weight_dtype,
        matrixA_dtype=A.dtype,
        stateIndex_dtype=stateIndex_dtype,
        dim=dim,
        dstate=dstate,
        npredicted=npredicted,
        max_window=max_window,
        heads_per_group=heads_per_group,
        num_groups=ngroups,
    )
    return out
