# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/integration/ep_moe.py @ 042b6aae (2026-06-30) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Conservative expert-parallel MoE entrypoints for replicated inputs.

This module deliberately owns a contract separate from :mod:`tp_moe`:

* every EP rank receives the same BF16 activations and global top-k routes;
* ``expert_map`` maps global expert ids to this rank's local weight ids and
  uses ``-1`` for non-local experts;
* the W4A16 route-pack kernel filters non-local routes and produces a
  zero-filled rank-local output partial; and
* the caller is responsible for summing those partials across the EP group.

There are no collectives here.  In particular, this is not the all-to-all or
batched-expert activation contract used by DeepEP-style integrations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from flashinfer.experimental.sm12x._lib.intrinsics import align_up
from flashinfer.experimental.sm12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from flashinfer.experimental.sm12x._lib.utils import get_num_sm
from flashinfer.experimental.sm12x.moe._shared.execution import MoEWeightPreparationPlan
from flashinfer.experimental.sm12x.moe._shared.kernels.activations import (
    moe_activation_w1_rows,
)
from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.host import (
    _W4A16_ALLOWED_ROUTED_SIZES,
    max_packed_route_slots,
    packed_gemm_scratch_elements,
    route_pack_token_capacity,
)

# NOTE(one-time port): upstream, ep_moe and tp_moe were siblings in
# b12x/integration/. The prepared-weights payload plumbing they share should
# eventually hoist into moe/_shared/weights.py; until then this is the one
# sanctioned intra-group op->op reach-through.
from ..fused_moe._impl import (
    SM12XFP4ExpertWeights,
    _normalize_swiglu_params,
    _prepared_payload_for_runtime,
)


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:
        # Tensors created inside inference mode do not expose a version
        # counter. They still retain the fixed-storage contract below.
        return None


@dataclass(frozen=True, kw_only=True)
class EPExpertMap:
    """A validated, static global-to-local expert map.

    Preparing the map may synchronize with the device to validate its values.
    Binding and execution only perform allocation-free metadata/version checks.
    """

    tensor: torch.Tensor
    global_num_experts: int
    local_num_experts: int
    device: torch.device
    _data_ptr: int
    _version: int | None

    def validate_static(self) -> None:
        if self.tensor.data_ptr() != self._data_ptr:
            raise RuntimeError("EP expert_map storage changed after preparation")
        version = _tensor_version(self.tensor)
        if self._version is not None and version != self._version:
            raise RuntimeError("EP expert_map was mutated after preparation")
        if self.tensor.device != self.device:
            raise RuntimeError("EP expert_map device changed after preparation")


def prepare_ep_expert_map(
    expert_map: torch.Tensor,
    *,
    local_num_experts: int,
    global_num_experts: int | None = None,
    device: torch.device | str | None = None,
) -> EPExpertMap:
    """Validate and freeze one EP rank's global-to-local expert map.

    Every local id must occur exactly once.  This matches vLLM's linear and
    round-robin expert placement maps and prevents a malformed map from
    indexing outside the rank-local weight allocation.
    """

    if not isinstance(expert_map, torch.Tensor):
        raise TypeError("expert_map must be a torch.Tensor")
    if expert_map.dtype != torch.int32:
        raise TypeError("expert_map must have dtype torch.int32")
    if expert_map.ndim != 1 or not expert_map.is_contiguous():
        raise ValueError("expert_map must be a contiguous rank-1 tensor")
    if expert_map.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EP expert_map must be prepared before CUDA graph capture")

    expected_device = expert_map.device if device is None else torch.device(device)
    if expert_map.device != expected_device:
        raise ValueError(
            f"expert_map must be on {expected_device}, got {expert_map.device}"
        )

    global_e = int(expert_map.numel())
    if global_num_experts is not None and global_e != int(global_num_experts):
        raise ValueError(
            "expert_map global expert count mismatch: "
            f"expected {int(global_num_experts)}, got {global_e}"
        )
    local_e = int(local_num_experts)
    if global_e <= 0:
        raise ValueError("global_num_experts must be positive")
    if local_e <= 0:
        raise ValueError("local_num_experts must be positive")
    if local_e > global_e:
        raise ValueError(
            f"local_num_experts={local_e} exceeds global_num_experts={global_e}"
        )

    values = expert_map.detach().cpu().tolist()
    invalid = [value for value in values if value < -1 or value >= local_e]
    if invalid:
        raise ValueError(
            "expert_map values must be -1 or valid local expert ids; "
            f"found {invalid[0]} for local_num_experts={local_e}"
        )
    mapped = sorted(value for value in values if value >= 0)
    expected = list(range(local_e))
    if mapped != expected:
        raise ValueError(
            "expert_map must map every local expert id exactly once: "
            f"expected {expected}, got {mapped}"
        )

    return EPExpertMap(
        tensor=expert_map,
        global_num_experts=global_e,
        local_num_experts=local_e,
        device=expected_device,
        _data_ptr=expert_map.data_ptr(),
        _version=_tensor_version(expert_map),
    )


@dataclass(frozen=True, kw_only=True)
class EPMoEScratchCaps:
    """Capacity and weight metadata for replicated-input W4A16 EP."""

    max_tokens: int
    num_topk: int
    global_num_experts: int
    device: torch.device | str
    weight_plan: MoEWeightPreparationPlan
    apply_router_weight_on_input: bool = False
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_tokens", max(int(self.max_tokens), 1))
        object.__setattr__(self, "num_topk", max(int(self.num_topk), 1))
        object.__setattr__(self, "global_num_experts", int(self.global_num_experts))
        object.__setattr__(self, "device", torch.device(self.device))
        object.__setattr__(
            self,
            "apply_router_weight_on_input",
            bool(self.apply_router_weight_on_input),
        )
        if not isinstance(self.weight_plan, MoEWeightPreparationPlan):
            raise TypeError("weight_plan must be a MoEWeightPreparationPlan")
        if self.global_num_experts <= 0:
            raise ValueError("global_num_experts must be positive")
        if self.weight_plan.num_experts <= 0:
            raise ValueError("the local weight plan must contain at least one expert")
        if self.weight_plan.num_experts > self.global_num_experts:
            raise ValueError(
                "local expert count cannot exceed global expert count: "
                f"local={self.weight_plan.num_experts}, "
                f"global={self.global_num_experts}"
            )
        if "w4a16" not in self.weight_plan.quant_modes:
            raise ValueError("replicated-input EP requires a W4A16 weight plan")
        if self.weight_plan.io_dtype != "bfloat16":
            raise TypeError(
                "replicated-input W4A16 EP requires BF16 activations, got "
                f"{self.weight_plan.io_dtype!r}"
            )
        limit, alpha, beta = _normalize_swiglu_params(
            self.weight_plan.activation,
            self.swiglu_limit,
            self.swiglu_alpha,
            self.swiglu_beta,
        )
        object.__setattr__(self, "swiglu_limit", limit)
        object.__setattr__(self, "swiglu_alpha", alpha)
        object.__setattr__(self, "swiglu_beta", beta)

    @property
    def local_num_experts(self) -> int:
        return self.weight_plan.num_experts


@dataclass(frozen=True)
class _EPBufferSpec:
    name: str
    elements: int
    dtype: torch.dtype
    offset_bytes: int

    @property
    def nbytes(self) -> int:
        return int(self.elements) * int(self.dtype.itemsize)


def _make_buffer_layout(
    specs: tuple[tuple[str, int, torch.dtype], ...],
) -> tuple[tuple[_EPBufferSpec, ...], int]:
    offset = 0
    layout = []
    for name, elements, dtype in specs:
        offset = align_up(offset, 16)
        spec = _EPBufferSpec(
            name=name,
            elements=max(int(elements), 1),
            dtype=dtype,
            offset_bytes=offset,
        )
        layout.append(spec)
        offset += spec.nbytes
    return tuple(layout), max(align_up(offset, 16), 1)


def _map_scratch_views(
    scratch: torch.Tensor,
    layout: tuple[_EPBufferSpec, ...],
) -> dict[str, torch.Tensor]:
    views: dict[str, torch.Tensor] = {}
    for spec in layout:
        byte_view = scratch.narrow(0, spec.offset_bytes, spec.nbytes)
        views[spec.name] = byte_view.view(spec.dtype)[: spec.elements]
    return views


@dataclass(frozen=True)
class EPMoEScratchPlan:
    caps: EPMoEScratchCaps
    _layout: tuple[_EPBufferSpec, ...]
    _nbytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def bind(
        self,
        *,
        scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        a: torch.Tensor,
        experts: SM12XFP4ExpertWeights,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: EPExpertMap,
        output: torch.Tensor,
        fast_math: bool = True,
    ) -> "EPMoEFP4Binding":
        if not isinstance(experts, SM12XFP4ExpertWeights):
            raise TypeError("experts must come from prepare_sm12x_fp4_moe_weights")
        if experts.plan != self.caps.weight_plan:
            raise ValueError("experts do not match the EP scratch weight plan")
        if not isinstance(expert_map, EPExpertMap):
            raise TypeError("expert_map must come from prepare_ep_expert_map")
        expert_map.validate_static()
        if expert_map.global_num_experts != self.caps.global_num_experts:
            raise ValueError("expert_map global expert count does not match EP plan")
        if expert_map.local_num_experts != self.caps.local_num_experts:
            raise ValueError("expert_map local expert count does not match EP weights")
        if expert_map.device != self.caps.device:
            raise ValueError("expert_map device does not match EP plan")

        if a.ndim != 2:
            raise ValueError(f"a must be rank 2, got shape {tuple(a.shape)}")
        m, k = map(int, a.shape)
        if m > self.caps.max_tokens:
            raise ValueError(
                f"input tokens {m} exceed EP scratch capacity {self.caps.max_tokens}"
            )
        if k != self.caps.weight_plan.hidden_size:
            raise ValueError(
                f"input hidden size {k} does not match {self.caps.weight_plan.hidden_size}"
            )
        if a.dtype != torch.bfloat16:
            raise TypeError(f"EP activations must be torch.bfloat16, got {a.dtype}")
        if a.device != self.caps.device or not a.is_contiguous():
            raise ValueError("a must be contiguous on the EP plan device")
        expected_routes = (m, self.caps.num_topk)
        if tuple(topk_ids.shape) != expected_routes:
            raise ValueError(
                f"topk_ids must have shape {expected_routes}, got {tuple(topk_ids.shape)}"
            )
        if tuple(topk_weights.shape) != expected_routes:
            raise ValueError(
                "topk_weights must have shape "
                f"{expected_routes}, got {tuple(topk_weights.shape)}"
            )
        if topk_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("topk_ids must have dtype torch.int32 or torch.int64")
        if topk_weights.dtype != torch.float32:
            raise TypeError("topk_weights must have dtype torch.float32")
        for name, tensor in (("topk_ids", topk_ids), ("topk_weights", topk_weights)):
            if tensor.device != self.caps.device or not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous on the EP plan device")
        if output.shape != a.shape or output.dtype != a.dtype:
            raise ValueError("output must have the same shape and dtype as a")
        if output.device != a.device or not output.is_contiguous():
            raise ValueError("output must be contiguous on the EP plan device")

        storage = scratch_tensor(
            scratch,
            self._scratch_specs,
            owner="replicated-input EP MoE",
        )
        views = _map_scratch_views(storage, self._layout)
        return EPMoEFP4Binding(
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            expert_map=expert_map,
            output=output,
            max_tokens=self.caps.max_tokens,
            num_topk=self.caps.num_topk,
            apply_router_weight_on_input=self.caps.apply_router_weight_on_input,
            fast_math=bool(fast_math),
            swiglu_limit=self.caps.swiglu_limit,
            swiglu_alpha=float(self.caps.swiglu_alpha),
            swiglu_beta=float(self.caps.swiglu_beta),
            intermediate_cache13=views["intermediate_cache13"],
            intermediate_cache2=views["intermediate_cache2"],
            fc1_c_tmp=views["fc1_c_tmp"],
            fc2_c_tmp=views["fc2_c_tmp"],
            packed_route_indices=views["packed_route_indices"],
            block_expert_ids=views["block_expert_ids"],
            packed_route_count=views["packed_route_count"],
            expert_offsets=views["expert_offsets"],
        )


def plan_ep_moe_scratch(caps: EPMoEScratchCaps) -> EPMoEScratchPlan:
    """Plan fixed-capacity scratch for replicated-input W4A16 EP."""

    if not isinstance(caps, EPMoEScratchCaps):
        raise TypeError("caps must be an EPMoEScratchCaps")
    routed_rows = int(caps.max_tokens) * int(caps.num_topk)
    route_capacity_rows = (
        route_pack_token_capacity(caps.max_tokens, caps.num_topk) * caps.num_topk
    )
    hidden_size = int(caps.weight_plan.hidden_size)
    intermediate_size = int(caps.weight_plan.intermediate_size)
    fc1_cols = moe_activation_w1_rows(
        caps.weight_plan.activation,
        intermediate_size,
    )
    sms = max(int(get_num_sm(caps.device)), 1)
    route_slots = 1
    route_blocks = 1
    fc1_tmp = 1
    fc2_tmp = 1
    for block_size in _W4A16_ALLOWED_ROUTED_SIZES:
        slots = max_packed_route_slots(
            route_capacity_rows,
            int(block_size),
            caps.global_num_experts,
        )
        blocks = (slots + int(block_size) - 1) // int(block_size)
        route_slots = max(route_slots, slots)
        route_blocks = max(route_blocks, blocks)
        fc1_tmp = max(
            fc1_tmp,
            packed_gemm_scratch_elements(
                size_n=fc1_cols,
                route_slots=slots,
                moe_block_size=int(block_size),
                sms=sms,
            ),
        )
        fc2_tmp = max(
            fc2_tmp,
            packed_gemm_scratch_elements(
                size_n=hidden_size,
                route_slots=slots,
                moe_block_size=int(block_size),
                sms=sms,
            ),
        )

    layout, nbytes = _make_buffer_layout(
        (
            (
                "intermediate_cache13",
                routed_rows * max(fc1_cols, hidden_size),
                torch.bfloat16,
            ),
            (
                "intermediate_cache2",
                routed_rows * intermediate_size,
                torch.bfloat16,
            ),
            ("fc1_c_tmp", fc1_tmp, torch.float32),
            ("fc2_c_tmp", fc2_tmp, torch.float32),
            ("packed_route_indices", route_slots, torch.int32),
            ("block_expert_ids", route_blocks, torch.int32),
            ("packed_route_count", 1, torch.int32),
            ("expert_offsets", caps.global_num_experts + 1, torch.int32),
        )
    )
    scratch_specs = (
        scratch_buffer_spec(
            "ep_moe.scratch",
            nbytes=nbytes,
            device=caps.device,
        ),
    )
    return EPMoEScratchPlan(
        caps=caps,
        _layout=layout,
        _nbytes=nbytes,
        _scratch_specs=scratch_specs,
    )


@dataclass(frozen=True, kw_only=True)
class EPMoEFP4Binding:
    """One allocation-free replicated-input EP launch binding."""

    a: torch.Tensor
    experts: SM12XFP4ExpertWeights
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    expert_map: EPExpertMap
    output: torch.Tensor
    max_tokens: int
    num_topk: int
    apply_router_weight_on_input: bool
    fast_math: bool
    swiglu_limit: float | None
    swiglu_alpha: float
    swiglu_beta: float
    intermediate_cache13: torch.Tensor
    intermediate_cache2: torch.Tensor
    fc1_c_tmp: torch.Tensor
    fc2_c_tmp: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor

    def run(self) -> torch.Tensor:
        return sm12x_ep_moe_fp4(binding=self)


def sm12x_ep_moe_fp4(*, binding: EPMoEFP4Binding) -> torch.Tensor:
    """Execute one replicated-input EP rank and return its local partial."""

    if not isinstance(binding, EPMoEFP4Binding):
        raise TypeError("binding must be an EPMoEFP4Binding")
    binding.expert_map.validate_static()
    prepared = _prepared_payload_for_runtime(
        binding.experts,
        quant_mode="w4a16",
        source_format=binding.experts.source_format,
        activation=binding.experts.activation,
        w13_layout=binding.experts.w13_layout,
        dtype=binding.a.dtype,
        hidden_size=int(binding.a.shape[1]),
    )
    if prepared is None:
        raise RuntimeError("the EP W4A16 weight representation was not materialized")

    from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.kernel import (
        run_w4a16_moe,
    )

    return run_w4a16_moe(
        binding.a,
        prepared,
        binding.topk_weights,
        binding.topk_ids,
        activation=binding.experts.activation,
        intermediate_cache13=binding.intermediate_cache13,
        intermediate_cache2=binding.intermediate_cache2,
        output=binding.output,
        fc1_c_tmp=binding.fc1_c_tmp,
        fc2_c_tmp=binding.fc2_c_tmp,
        packed_route_indices=binding.packed_route_indices,
        block_expert_ids=binding.block_expert_ids,
        packed_route_count=binding.packed_route_count,
        expert_offsets=binding.expert_offsets,
        expert_map=binding.expert_map.tensor,
        apply_router_weight_on_input=binding.apply_router_weight_on_input,
        fast_math=binding.fast_math,
        swiglu_limit=binding.swiglu_limit,
        swiglu_alpha=binding.swiglu_alpha,
        swiglu_beta=binding.swiglu_beta,
    )


__all__ = [
    "EPExpertMap",
    "EPMoEFP4Binding",
    "EPMoEScratchCaps",
    "EPMoEScratchPlan",
    "sm12x_ep_moe_fp4",
    "plan_ep_moe_scratch",
    "prepare_ep_expert_map",
]
