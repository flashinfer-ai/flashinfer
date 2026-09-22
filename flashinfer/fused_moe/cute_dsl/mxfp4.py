# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Planned MXFP4 W4A8 routed MoE; minimum architecture SM100 (B300: SM103).

Planning binds caller-owned buffers and compiles the selected offline tactic.
Execution reads their current contents, including runtime SiTU parameters.
Use a separate plan/output/workspace for concurrently executing calls.

Rank-local layouts come from explicit metadata (``Mxfp4MoEParallelLayout`` or
the explicit local expert interval), never from tensor shapes. Expert
parallelism owns a contiguous global expert interval; MoE tensor parallelism
owns an intermediate-dimension shard of every expert. Hybrid layouts are
rejected. Every rank output is a partial sum whose reduction is external.
"""

from dataclasses import dataclass
from math import prod
from typing import Optional

import cuda.bindings.driver as cuda
import torch

from ...tllm_enums import ActivationType
from .fused_moe import _moe_core_impl, validate_w4a8_inputs
from .moe_utils import get_max_num_tiles
from .mxfp4_routing import _plan_route_preprocess
from .tuner import DEFAULT_BLACKWELL_MOE_TACTIC, canonicalize_w4a8_tactic


_PARALLEL_MODES = ("single", "expert_parallel", "moe_tensor_parallel")


@dataclass(frozen=True)
class Mxfp4MoEParallelLayout:
    """Explicit rank placement; ``size`` ranks, this process is ``rank``.

    ``mode`` is ``"single"`` (one rank owns everything), ``"expert_parallel"``
    (rank ``r`` owns global experts ``[r*E/size, (r+1)*E/size)`` with the
    full intermediate dimension) or ``"moe_tensor_parallel"`` (every rank owns
    all experts and intermediate columns ``[r*I/size, (r+1)*I/size)`` of each).
    Hybrid expert/tensor parallelism is not representable; ``from_sizes``
    rejects it. Global ``num_experts`` and ``intermediate_size`` are resolved
    into rank-local values by ``resolve_mxfp4_moe_layout``.
    """

    mode: str = "single"
    size: int = 1
    rank: int = 0

    def __post_init__(self):
        if self.mode not in _PARALLEL_MODES:
            raise ValueError(f"parallel mode must be one of {_PARALLEL_MODES}")
        if not isinstance(self.size, int) or isinstance(self.size, bool):
            raise ValueError("parallel size must be an int")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError("parallel rank must be an int")
        if self.size < 1:
            raise ValueError("parallel size must be positive")
        if not 0 <= self.rank < self.size:
            raise ValueError(f"parallel rank must be in [0, {self.size})")
        if self.mode == "single" and (self.size, self.rank) != (1, 0):
            raise ValueError("single layout requires size 1 and rank 0")

    @classmethod
    def from_sizes(
        cls,
        *,
        ep_size: int = 1,
        ep_rank: int = 0,
        moe_tp_size: int = 1,
        moe_tp_rank: int = 0,
    ) -> "Mxfp4MoEParallelLayout":
        """Build a layout from EP/TP sizes; both above one is a hybrid error."""
        for name, size, rank in (
            ("ep", ep_size, ep_rank),
            ("moe_tp", moe_tp_size, moe_tp_rank),
        ):
            if size < 1 or not 0 <= rank < size:
                raise ValueError(
                    f"require {name}_size >= 1 and 0 <= {name}_rank < size"
                )
        if ep_size > 1 and moe_tp_size > 1:
            raise ValueError(
                "hybrid expert/tensor parallelism is unsupported: "
                f"ep_size={ep_size} and moe_tp_size={moe_tp_size} both exceed 1"
            )
        if ep_size > 1:
            return cls("expert_parallel", ep_size, ep_rank)
        if moe_tp_size > 1:
            return cls("moe_tensor_parallel", moe_tp_size, moe_tp_rank)
        return cls()


@dataclass(frozen=True)
class Mxfp4MoERankLayout:
    """Resolved rank-local geometry; ``num_experts``/``intermediate_size`` are global.

    ``parallel_size``/``parallel_rank`` are ``None`` when an explicit expert
    interval does not coincide with a uniform expert-parallel rank slice.
    """

    mode: str
    num_experts: int
    intermediate_size: int
    num_local_experts: int
    local_expert_offset: int
    intermediate_shard: int
    parallel_size: Optional[int] = None
    parallel_rank: Optional[int] = None

    @property
    def gemm1_n(self) -> int:
        """Rank-local GEMM1 output width: interleaved up and gate rows."""
        return 2 * self.intermediate_shard

    @property
    def gemm2_k(self) -> int:
        """Rank-local GEMM2 contraction length."""
        return self.intermediate_shard


def resolve_mxfp4_moe_layout(
    num_experts: int,
    intermediate_size: int,
    *,
    parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
    num_local_experts: Optional[int] = None,
    local_expert_offset: Optional[int] = None,
) -> Mxfp4MoERankLayout:
    """Derive and validate rank-local geometry from explicit metadata only.

    ``parallel_layout`` is the uniform EP/TP form. ``num_local_experts`` and
    ``local_expert_offset`` are the explicit expert-interval form, which is
    expert parallelism (or single when the interval is every expert). When
    both forms are given they must agree. Nothing is inferred from tensors.
    """
    if num_experts <= 0 or intermediate_size <= 0:
        raise ValueError("num_experts and intermediate_size must be positive")
    if parallel_layout is None:
        local = num_experts if num_local_experts is None else num_local_experts
        offset = 0 if local_expert_offset is None else local_expert_offset
        if local <= 0 or offset < 0 or offset + local > num_experts:
            raise ValueError(
                "local experts must form a nonempty contiguous global expert interval"
            )
        if local == num_experts and offset == 0:
            return Mxfp4MoERankLayout(
                "single",
                num_experts,
                intermediate_size,
                local,
                0,
                intermediate_size,
                1,
                0,
            )
        uniform = num_experts % local == 0 and offset % local == 0
        return Mxfp4MoERankLayout(
            "expert_parallel",
            num_experts,
            intermediate_size,
            local,
            offset,
            intermediate_size,
            num_experts // local if uniform else None,
            offset // local if uniform else None,
        )
    if not isinstance(parallel_layout, Mxfp4MoEParallelLayout):
        raise TypeError("parallel_layout must be an Mxfp4MoEParallelLayout")
    mode, size, rank = parallel_layout.mode, parallel_layout.size, parallel_layout.rank
    if mode == "expert_parallel":
        if num_experts % size:
            raise ValueError(
                f"expert parallelism requires num_experts ({num_experts}) divisible "
                f"by ep size ({size})"
            )
        local, offset, shard = (
            num_experts // size,
            rank * (num_experts // size),
            (intermediate_size),
        )
    elif mode == "moe_tensor_parallel":
        if intermediate_size % size or (intermediate_size // size) % 128:
            raise ValueError(
                f"MoE tensor parallelism requires intermediate_size ({intermediate_size}) "
                f"divisible by moe_tp size ({size}) into a multiple of 128"
            )
        local, offset, shard = num_experts, 0, intermediate_size // size
    else:
        local, offset, shard = num_experts, 0, intermediate_size
    if num_local_experts is not None and num_local_experts != local:
        raise ValueError(
            f"num_local_experts={num_local_experts} is inconsistent with "
            f"{mode} size {size} rank {rank}, which owns {local} local experts"
        )
    if local_expert_offset is not None and local_expert_offset != offset:
        raise ValueError(
            f"local_expert_offset={local_expert_offset} is inconsistent with "
            f"{mode} size {size} rank {rank}, whose offset is {offset}"
        )
    return Mxfp4MoERankLayout(
        mode, num_experts, intermediate_size, local, offset, shard, size, rank
    )


@dataclass(frozen=True)
class Mxfp4MoECapability:
    """Support verdict plus the resolved rank-local layout when it resolves."""

    supported: bool
    reason: str
    cuda_graph: bool
    layout: Optional[Mxfp4MoERankLayout] = None


def mxfp4_moe_capability(
    *,
    gpu_arch: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: Optional[int] = None,
    parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
    quantization: str = "mxfp4_w4a8",
    activation_type: ActivationType = ActivationType.Situ,
    cuda_graph: bool = True,
) -> Mxfp4MoECapability:
    """Query support from explicit metadata without allocating or using CUDA.

    ``gpu_arch`` is 100 for SM100 or 103 for SM103. Weight scales are UE8M0
    with group size 32; activation storage is E4M3 and output is BF16.
    ``num_experts`` and ``intermediate_size`` are global model values; the
    rank-local expert interval and intermediate shard are derived from
    ``parallel_layout`` and/or the explicit ``num_local_experts`` and
    ``local_expert_offset`` (see ``resolve_mxfp4_moe_layout``). The result
    carries that resolved layout; CUDA Graph capture is supported for every
    supported configuration in both parallel modes.
    """
    reason = ""
    layout = None
    if gpu_arch not in (100, 103):
        reason = "MXFP4 W4A8 requires SM100 or SM103"
    elif quantization != "mxfp4_w4a8":
        reason = "quantization must be explicitly mxfp4_w4a8"
    elif activation_type not in (ActivationType.Situ, ActivationType.Swiglu):
        reason = "planned MXFP4 supports SiTU and SwiGLU"
    elif min(hidden_size, intermediate_size) <= 0 or (
        hidden_size % 128 or intermediate_size % 128
    ):
        reason = "hidden and intermediate dimensions must be positive multiples of 128"
    elif not (1 <= top_k <= num_experts <= 1024):
        reason = "require 1 <= top_k <= num_experts <= 1024"
    elif top_k > 32:
        reason = "top_k must not exceed 32"
    else:
        try:
            layout = resolve_mxfp4_moe_layout(
                num_experts,
                intermediate_size,
                parallel_layout=parallel_layout,
                num_local_experts=num_local_experts,
                local_expert_offset=local_expert_offset,
            )
        except (TypeError, ValueError) as error:
            reason = str(error)
    return Mxfp4MoECapability(not reason, reason, not reason, layout)


@dataclass(frozen=True)
class _WorkspaceField:
    name: str
    shape: tuple
    dtype: torch.dtype
    offset: int
    nbytes: int


def _align(size: int) -> int:
    return (size + 255) // 256 * 256


def _byte_interval(tensor):
    first = tensor.data_ptr()
    span = 1 + sum(
        (dim - 1) * stride
        for dim, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return first, first + span * tensor.element_size()


def _overlap(left, right):
    return max(left[0], right[0]) < min(left[1], right[1])


class Mxfp4MoEPlan:
    """Executable with fixed buffer addresses, prepared outside graph capture.

    Update bound activation, routing, beta and scale tensors in-place before
    calling ``run`` or replaying a captured graph. No weights are copied.
    The output is this rank's partial sum: its local experts under expert
    parallelism, or its intermediate shard under MoE tensor parallelism.
    """

    def __init__(
        self, *, kwargs, workspace, topk_ids, topk_weights, route_ids, route_weights
    ):
        # Keep every bound tensor alive alongside the raw launch pointers.
        self._kwargs = kwargs
        self.workspace = workspace
        self.output = kwargs["moe_output"]
        self._topk_ids = topk_ids
        self._topk_weights = topk_weights
        self._route_ids = route_ids
        self._route_weights = route_weights
        self._route_preprocess = None
        self.device = self.output.device
        self._packed_weight_view = (
            topk_ids.view(torch.bfloat16)[:, ::2] if topk_weights is None else None
        )

    def _prepare_routing(self):
        if self._topk_weights is None:
            torch.bitwise_right_shift(self._topk_ids, 16, out=self._route_ids)
            self._route_weights.copy_(self._packed_weight_view)
        elif self._route_weights is not self._topk_weights:
            self._route_weights.copy_(self._topk_weights)

    def _prepare(self):
        # The existing path validates and warms the exact pointers/callables
        # retained below. No stream is retained: run resolves the caller's.
        launches = {}
        with torch.cuda.device(self.device):
            self._prepare_routing()
            _moe_core_impl(**self._kwargs, _prepared_launches=launches)
        self._sort, self._sort_args = launches["sort"]
        self._gather, self._gather_args, self._gather_kwargs = launches["gather"]
        self._memset, self._memset_args = launches["memset"]
        self._finalize, self._finalize_args = launches["finalize"]
        if self.output.shape[0] <= 16:
            self._route_preprocess = _plan_route_preprocess(
                self._topk_ids,
                self._topk_weights,
                route_ids=self._route_ids,
                route_weights=self._route_weights,
                output=self.output,
                moe_sort_buffers=(
                    None
                    if self._kwargs["enable_pdl"]
                    else self._kwargs["moe_sort_buffers"]
                ),
                num_experts=self._kwargs["num_experts"],
                num_local_experts=self._kwargs["num_local_experts"],
                local_expert_offset=self._kwargs["local_expert_offset"],
                tile_size=self._kwargs["tile_size"],
                _single_tile_per_expert=self._kwargs.get(
                    "_enable_decode_specialization", False
                ),
            )
            # Preprocessing warmup clears output. Finish the complete MoE so
            # plan retains its existing valid-output postcondition.
            self.run()

    def run(self) -> torch.Tensor:
        """Enqueue on the caller's current stream and return the bound output.

        All GPU buffers and compiled kernels were prepared by ``plan``.
        This method performs no tuning, allocation, or host synchronization.
        """
        with torch.cuda.device(self.device):
            stream_ptr = torch.cuda.current_stream().cuda_stream
            stream = cuda.CUstream(stream_ptr)
            if self._route_preprocess is None:
                self._prepare_routing()
            else:
                self._route_preprocess.run(stream)
            if (
                self._route_preprocess is None
                or not self._route_preprocess.sorts_tokens
            ):
                self._sort(*self._sort_args, stream_ptr)
            self._gather(*self._gather_args, stream=stream, **self._gather_kwargs)
            if self._route_preprocess is None:
                self._memset(*self._memset_args, stream_ptr)
            self._finalize(*self._finalize_args, stream=stream)
        return self.output


class CuteDslMxfp4MoEWrapper:
    """MXFP4 runner with offline tactics and explicit caller-owned workspace.

    The wrapper is metadata only. ``get_workspace_size(T)`` may be called
    before any CUDA allocation. ``plan`` compiles and performs warmup
    execution using valid caller inputs; it must run outside CUDA Graph
    capture. Its returned plan is used both for prefill and decode.

    ``offline_tactics`` maps token-count upper bounds to W4A8 tactic tuples.
    The smallest covering bucket is selected from host shape metadata. No
    serving-time tuning or device-to-host routing inspection is performed.
    An omitted table uses the conservative existing Blackwell tactic.

    ``num_experts`` and ``intermediate_size`` are the global model values.
    The rank-local layout is explicit: ``parallel_layout`` (uniform EP or MoE
    TP) and/or the expert interval ``num_local_experts``/``local_expert_offset``
    (the expert-parallel form). Both forms may be given if they agree. The
    derived ``layout`` fixes the weight shapes ``plan`` accepts; shapes never
    select the mode. Every rank computes a partial output; reduce externally.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_local_experts: Optional[int] = None,
        local_expert_offset: Optional[int] = None,
        parallel_layout: Optional[Mxfp4MoEParallelLayout] = None,
        activation_type: ActivationType = ActivationType.Situ,
        quantization: str = "mxfp4_w4a8",
        enable_pdl: bool = False,
        offline_tactics: Optional[dict] = None,
    ):
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = ActivationType(activation_type)
        self.quantization = quantization
        self.enable_pdl = enable_pdl
        supported = mxfp4_moe_capability(
            gpu_arch=103,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            parallel_layout=parallel_layout,
            quantization=quantization,
            activation_type=self.activation_type,
        )
        if not supported.supported:
            raise ValueError(supported.reason)
        self.parallel_layout = parallel_layout
        self.layout = supported.layout
        self.num_local_experts = self.layout.num_local_experts
        self.local_expert_offset = self.layout.local_expert_offset
        self.intermediate_shard = self.layout.intermediate_shard
        self._offline_tactics = sorted(
            (int(limit), canonicalize_w4a8_tactic(tactic))
            for limit, tactic in (offline_tactics or {}).items()
        )
        if any(limit <= 0 for limit, _ in self._offline_tactics):
            raise ValueError("offline tactic bucket bounds must be positive")

    @property
    def parallel_mode(self) -> str:
        return self.layout.mode

    def _metadata(self):
        return dict(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            parallel_layout=self.parallel_layout,
            quantization=self.quantization,
            activation_type=self.activation_type,
        )

    def _tactic(self, num_tokens):
        for limit, tactic in self._offline_tactics:
            if num_tokens <= limit:
                return tactic
        return DEFAULT_BLACKWELL_MOE_TACTIC

    def _workspace_fields(self, num_tokens):
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        tile = self._tactic(num_tokens)[0]
        tiles = get_max_num_tiles(num_tokens, self.top_k, self.num_local_experts, tile)
        rows = tiles * tile
        specs = [
            ("out_tile_idx_to_expert_idx", (tiles,), torch.int32, 4),
            ("out_tile_idx_to_mn_limit", (tiles,), torch.int32, 4),
            (
                "out_expanded_idx_to_permuted_idx",
                (num_tokens, self.top_k),
                torch.int32,
                4,
            ),
            ("out_permuted_idx_to_expanded_idx", (rows,), torch.int32, 4),
            ("out_total_num_padded_tokens", (1,), torch.int32, 4),
            ("out_num_non_exiting_tiles", (1,), torch.int32, 4),
            ("gemm1_out", (rows, self.intermediate_shard), torch.float8_e4m3fn, 1),
            (
                "gemm1_out_scale",
                (32, 4, rows // 128, 4, self.intermediate_shard // 128, 1),
                torch.uint8,
                1,
            ),
            ("route_ids", (num_tokens, self.top_k), torch.int32, 4),
            ("route_weights", (num_tokens, self.top_k), torch.float32, 4),
            ("w1_alpha", (self.num_local_experts,), torch.float32, 4),
            ("w2_alpha", (self.num_local_experts,), torch.float32, 4),
        ]
        if num_tokens > 1024:
            specs.append(("out_expert_counts", (2 * self.num_experts,), torch.int32, 4))
        fields, offset = [], 0
        for name, shape, dtype, itemsize in specs:
            offset = _align(offset)
            size = prod(shape) * itemsize
            fields.append(_WorkspaceField(name, shape, dtype, offset, size))
            offset += size
        return fields, _align(offset)

    def get_workspace_size(self, num_tokens: int) -> int:
        """Return required workspace bytes for any routing at this token count.

        The size follows the rank-local layout: the GEMM1 intermediate region
        uses ``intermediate_shard`` columns and the per-expert regions use
        ``num_local_experts``.
        """
        return self._workspace_fields(num_tokens)[1]

    def plan(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: Optional[torch.Tensor],
        w1: torch.Tensor,
        w1_sf: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        *,
        beta: Optional[torch.Tensor] = None,
        linear_beta: Optional[torch.Tensor] = None,
        workspace: torch.Tensor,
        output: torch.Tensor,
    ) -> Mxfp4MoEPlan:
        """Bind buffers and prepare kernels; all tensor contents must be valid.

        Weights use ``prepare_cute_dsl_mxfp4_weights`` layouts for this rank's
        shard: ``[num_local_experts, 2*intermediate_shard, H/2]`` W1 and
        ``[num_local_experts, H, intermediate_shard/2]`` W2, as produced by
        ``shard_cute_dsl_mxfp4_weights`` for the resolved layout. Shapes are
        validated against that layout and never used to select it. ``x_sf`` is
        linear UE8M0 bytes [T,H/32]. ``beta`` and optional ``linear_beta`` are
        contiguous CUDA FP32 tensors with one value or one per local expert.
        Their values must be finite and positive; they are read on the device
        at execution, so changing them requires no recompilation.

        Routing may be separate int32 IDs and BF16/FP32 weights, or packed
        int32 (expert ID in high 16 bits, BF16 weight in low 16 bits) when
        ``topk_weights=None``. IDs are global, must be in ``[0, num_experts)``,
        and must be distinct within each token. Output and workspace must be
        distinct from all inputs; workspace is a contiguous uint8 tensor whose
        address is aligned to 256 bytes.
        """
        if x.device.type != "cuda":
            raise ValueError("plan requires CUDA tensors")
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("plan must be called before CUDA Graph capture")
            major, minor = torch.cuda.get_device_capability(x.device)
            capability = mxfp4_moe_capability(
                gpu_arch=major * 10 + minor, **self._metadata()
            )
            if not capability.supported:
                raise ValueError(capability.reason)
            num_tokens = x.shape[0]
            layout = self.layout
            expected = {
                "x": (x, (num_tokens, self.hidden_size), torch.float8_e4m3fn),
                "x_sf": (x_sf, (num_tokens, self.hidden_size // 32), torch.uint8),
                "topk_ids": (topk_ids, (num_tokens, self.top_k), torch.int32),
                "w1": (
                    w1,
                    (
                        layout.num_local_experts,
                        2 * layout.intermediate_shard,
                        self.hidden_size // 2,
                    ),
                    torch.uint8,
                ),
                "w2": (
                    w2,
                    (
                        layout.num_local_experts,
                        self.hidden_size,
                        layout.intermediate_shard // 2,
                    ),
                    torch.uint8,
                ),
                "output": (output, (num_tokens, self.hidden_size), torch.bfloat16),
            }
            for name, (tensor, shape, dtype) in expected.items():
                if (
                    tensor.device != x.device
                    or tensor.dtype != dtype
                    or tuple(tensor.shape) != shape
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be contiguous {dtype} {shape} on {x.device} "
                        f"for parallel mode {layout.mode} ({layout.num_local_experts} "
                        f"local experts at offset {layout.local_expert_offset}, "
                        f"intermediate shard {layout.intermediate_shard}); got "
                        f"{tensor.dtype} {tuple(tensor.shape)} on {tensor.device}"
                    )
            for name, tensor, rows, columns in (
                ("w1_sf", w1_sf, 2 * layout.intermediate_shard, self.hidden_size),
                ("w2_sf", w2_sf, self.hidden_size, layout.intermediate_shard),
            ):
                expected_shape = (
                    32,
                    4,
                    rows // 128,
                    4,
                    columns // 128,
                    layout.num_local_experts,
                )
                if (
                    tensor.device != x.device
                    or tensor.dtype != torch.uint8
                    or tuple(tensor.shape) != expected_shape
                ):
                    raise ValueError(
                        f"{name} must be the prepared uint8 MMA scale layout "
                        f"{expected_shape} on {x.device} for parallel mode "
                        f"{layout.mode}; got {tensor.dtype} {tuple(tensor.shape)} "
                        f"on {tensor.device}"
                    )
            if topk_weights is not None and (
                topk_weights.device != x.device
                or topk_weights.dtype not in (torch.bfloat16, torch.float32)
                or tuple(topk_weights.shape) != (num_tokens, self.top_k)
                or not topk_weights.is_contiguous()
            ):
                raise ValueError(
                    "topk_weights must be contiguous CUDA BF16/FP32 [T,top_k]"
                )
            if self.activation_type == ActivationType.Situ and beta is None:
                raise ValueError("SiTU requires runtime beta")
            if self.activation_type != ActivationType.Situ and (
                beta is not None or linear_beta is not None
            ):
                raise ValueError("SiTU parameters require ActivationType.Situ")
            for name, tensor in (("beta", beta), ("linear_beta", linear_beta)):
                if tensor is not None and (
                    tensor.device != x.device
                    or tensor.dtype != torch.float32
                    or tensor.ndim != 1
                    or tensor.numel() not in (1, self.num_local_experts)
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be CUDA FP32 [1] or [num_local_experts]"
                    )
            fields, size = self._workspace_fields(num_tokens)
            if (
                workspace.device != x.device
                or workspace.dtype != torch.uint8
                or workspace.ndim != 1
                or not workspace.is_contiguous()
                or workspace.numel() < size
                or workspace.data_ptr() % 256
            ):
                raise ValueError(
                    f"workspace requires at least {size} aligned CUDA uint8 bytes"
                )
            workspace_interval = (workspace.data_ptr(), workspace.data_ptr() + size)
            output_interval = _byte_interval(output)
            if _overlap(workspace_interval, output_interval):
                raise ValueError("output must not overlap workspace")
            for name, tensor in (
                ("x", x),
                ("x_sf", x_sf),
                ("topk_ids", topk_ids),
                ("topk_weights", topk_weights),
                ("w1", w1),
                ("w1_sf", w1_sf),
                ("w2", w2),
                ("w2_sf", w2_sf),
                ("beta", beta),
                ("linear_beta", linear_beta),
            ):
                if tensor is not None and (
                    _overlap(workspace_interval, _byte_interval(tensor))
                    or _overlap(output_interval, _byte_interval(tensor))
                ):
                    raise ValueError(f"output/workspace must not overlap {name}")
            buffers = {
                f.name: workspace.narrow(0, f.offset, f.nbytes)
                .view(f.dtype)
                .view(f.shape)
                for f in fields
            }
            buffers["w1_alpha"].fill_(1.0)
            buffers["w2_alpha"].fill_(1.0)
            route_weights = (
                topk_weights
                if topk_weights is not None and topk_weights.dtype == torch.float32
                else buffers["route_weights"]
            )
            validate_w4a8_inputs(x, x_sf, route_weights, w1, w1_sf, w2, w2_sf)
            if w1_sf.device != x.device or w2_sf.device != x.device:
                raise ValueError("weight scales must be on the input device")
            tile, gemm1, gemm2 = self._tactic(num_tokens)
            # The public routing contract requires distinct IDs per token,
            # so an expert has at most T rows. Restrict this specialization
            # to the qualified B300 SiTU decode tactic.
            decode_specialization = (
                (major, minor) == (10, 3)
                and 1 <= num_tokens <= 16
                and self.activation_type == ActivationType.Situ
                and not self.enable_pdl
                and tile == 128
                and gemm1 == ((128, 128), (1, 1), False)
                and gemm2 == ((128, 128), (1, 1), False)
            )
            route_ids = buffers["route_ids"] if topk_weights is None else topk_ids
            kwargs = dict(
                x=x,
                x_sf=x_sf,
                token_selected_experts=route_ids,
                token_final_scales=route_weights,
                w1_weight=w1,
                w1_weight_sf=w1_sf,
                w1_alpha=buffers["w1_alpha"],
                fc2_input_scale=None,
                w2_weight=w2,
                w2_weight_sf=w2_sf,
                w2_alpha=buffers["w2_alpha"],
                num_experts=self.num_experts,
                top_k=self.top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_offset,
                tile_size=tile,
                gemm1_mma_tiler_mn=gemm1[0],
                gemm1_cluster_shape_mn=gemm1[1],
                gemm2_mma_tiler_mn=gemm2[0],
                gemm2_cluster_shape_mn=gemm2[1],
                moe_sort_buffers={
                    name: value
                    for name, value in buffers.items()
                    if name.startswith("out_")
                },
                gemm1_out=buffers["gemm1_out"],
                gemm1_out_scale=buffers["gemm1_out_scale"],
                moe_output=output,
                output_dtype=torch.bfloat16,
                use_async_memset=False,
                use_fused_finalize=True,
                enable_pdl=self.enable_pdl,
                activation_type=self.activation_type.value,
                situ_beta=beta,
                situ_linear_beta=linear_beta,
                _enable_decode_specialization=decode_specialization,
            )
            plan = Mxfp4MoEPlan(
                kwargs=kwargs,
                workspace=workspace,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                route_ids=route_ids,
                route_weights=route_weights,
            )
            plan._prepare()
            return plan
