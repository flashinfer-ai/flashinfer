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
import math
import threading
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import (
    cutlass_fused_moe_trace,
    trtllm_bf16_moe_trace,
    trtllm_bf16_routed_moe_trace,
    trtllm_fp4_block_scale_moe_trace_dispatch,
    trtllm_fp4_block_scale_routed_moe_trace,
    trtllm_fp8_block_scale_moe_trace_dispatch,
    trtllm_fp8_block_scale_routed_moe_trace,
    trtllm_fp8_per_tensor_scale_moe_trace,
    trtllm_fp8_per_tensor_scale_routed_moe_trace,
    trtllm_mxint4_block_scale_moe_trace,
)
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from flashinfer.autotuner.initializers import (
    autotuner_initializer_empty,
    autotuner_initializer_ones,
    autotuner_initializer_rand,
    autotuner_initializer_randn,
    autotuner_initializer_zeros,
)
from ..jit import (
    setup_cubin_loader,
)
from ..jit.core import logger
from ..jit.cpp_ext import is_cuda_version_at_least
from ..jit.fused_moe import (
    gen_cutlass_fused_moe_sm89_module,
    gen_cutlass_fused_moe_sm90_module,
    gen_cutlass_fused_moe_sm100_module,
    gen_cutlass_fused_moe_sm103_module,
    gen_cutlass_fused_moe_sm120_module,
    gen_trtllm_gen_fused_moe_sm100_module,
)
from ..tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    RoutingInputMode,
    RoutingMethodType,
    WeightLayout,
    deduce_trtllm_gen_tensor_dtype,
    trtllm_gen_dtype_has_scale,
)
from ..utils import (
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
    register_custom_op,
    register_fake_op,
)
from .da_moe import DA_MAX_EXPERTS, DABody

# These helpers moved to prepare.py; keep aliases here for backward compatibility.
from .prepare import (
    interleave_moe_scales_for_sm90_mixed_gemm as interleave_moe_scales_for_sm90_mixed_gemm,
    interleave_moe_weights_for_sm90_mixed_gemm as interleave_moe_weights_for_sm90_mixed_gemm,
)
from .utils import (
    get_hybrid_num_tokens_buckets,
    make_hybrid_bucket_mapper,
    make_random_topk_ids,
)

if TYPE_CHECKING:
    from flashinfer.fused_moe.da_config import TrtllmDaConfig


# RoutingInputMode (the FusedMoE launcher's routing-input ABI enum) lives in
# flashinfer.tllm_enums with the other kernel-ABI enums; it is imported above
# and re-exported here for compatibility (``core.RoutingInputMode``).


@dataclass(frozen=True)
class TrtllmMoERoutingMetadataSlot:
    """Graph-stable routing metadata consumed by bodies sharing one tile-N."""

    # Routing tile-N that determines every buffer extent in this slot.
    tile_n: int
    # Device scalar containing the live padded permutation size.
    # Native FFI[0]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.total_num_padded_tokens.
    total_num_padded_tokens: torch.Tensor
    # Expanded token-slot to permuted-row mapping.
    # Native FFI[1]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.expanded_idx_to_permuted_idx.
    expanded_idx_to_permuted_idx: torch.Tensor
    # Permuted-row to original token mapping with the backend guard element.
    # Native FFI[2]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.permuted_idx_to_token_idx.
    permuted_idx_to_token_idx: torch.Tensor
    # Live BF16 or FP32 routing weights in token/top-k layout.
    # Native FFI[3]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.expert_weights.
    expert_weights: torch.Tensor
    # Routing kernel histogram scratch sized for the expert specialization.
    # Native FFI[4]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.expert_count_histogram.
    expert_count_histogram: torch.Tensor
    # Live token count produced for every expert.
    # Native FFI[5]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.num_tokens_per_expert.
    num_tokens_per_expert: torch.Tensor
    # Grouped-GEMM CTA-to-expert batch mapping.
    # Native FFI[6]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.cta_idx_xy_to_batch_idx.
    cta_idx_xy_to_batch_idx: torch.Tensor
    # Grouped-GEMM CTA M/N limit mapping.
    # Native FFI[7]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.cta_idx_xy_to_mn_limit.
    cta_idx_xy_to_mn_limit: torch.Tensor
    # Device scalar containing the number of live grouped-GEMM CTAs.
    # Native FFI[8]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:RoutingMetadataBuffers.num_non_exiting_ctas.
    num_non_exiting_ctas: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return tensors in the fixed native nine-slot ABI order."""
        return (
            self.total_num_padded_tokens,
            self.expanded_idx_to_permuted_idx,
            self.permuted_idx_to_token_idx,
            self.expert_weights,
            self.expert_count_histogram,
            self.num_tokens_per_expert,
            self.cta_idx_xy_to_batch_idx,
            self.cta_idx_xy_to_mn_limit,
            self.num_non_exiting_ctas,
        )


@dataclass(frozen=True)
class TrtllmMoERoutingMetadata:
    """Prepared single- or multi-tile routing metadata with immutable capacity."""

    # Routing representation whose live tensors populate this storage.
    routing_input_mode: RoutingInputMode
    # Global expert domain used by routing validation.
    num_experts: int
    # Routed experts selected for each token.
    top_k: int
    # First global expert ID owned by this rank.
    local_expert_offset: int
    # Number of experts owned by this rank.
    num_local_experts: int
    # One graph-stable output slot per sorted unique tile-N.
    slots: tuple[TrtllmMoERoutingMetadataSlot, ...]

    def flat_tensors(self) -> list[torch.Tensor]:
        """Flatten every tile slot into the native in-place population ABI."""
        return [tensor for slot in self.slots for tensor in slot.tensors()]

    @property
    def tile_ns(self) -> tuple[int, ...]:
        """Return the immutable routing tiles represented by this storage."""
        return tuple(slot.tile_n for slot in self.slots)


@dataclass(frozen=True)
class TRTLLMCanonicalRouting:
    """Graph-stable outputs and scratch for one real FromLogits router launch."""

    # Native int16 expert IDs emitted by the ordinary router and consumed during DA replay.
    # Native FFI[0]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:CanonicalRoutingBuffers.routing_replay_ids.
    routing_replay_ids: torch.Tensor
    # Canonical BF16 expert weights produced by the real routing method.
    # Native FFI[1]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:CanonicalRoutingBuffers.expert_weights.
    expert_weights: torch.Tensor
    # Conventional packed scratch retained only to satisfy the ordinary router/body ABI.
    # Native FFI[2]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:CanonicalRoutingBuffers.packed_scratch.
    packed_router_scratch: torch.Tensor
    # Remaining native tensors retaining every pointer used by the router launch.
    # Native FFI[3:11]; sync with csrc/trtllm_fused_moe_kernel_launcher.cu:CanonicalRoutingBuffers.num_tokens_per_expert.
    scratch: tuple[torch.Tensor, ...]
    # Tile used only to size the router's temporary permutation storage.
    tile_n: int

    def tensors(self) -> list[torch.Tensor]:
        """Return tensors in the native eleven-slot canonical-routing ABI order."""
        return [
            self.routing_replay_ids,
            self.expert_weights,
            self.packed_router_scratch,
            *self.scratch,
        ]


@dataclass(frozen=True)
class TrtllmDaBodyCaptureStream:
    """Own one reusable native stream reserved for direct DA body capture."""

    # CUDA device whose runtime context owns the native stream.
    device_index: int
    # Opaque native stream handle passed unchanged through the body-capture FFI.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:trtllm_moe_create_da_body_capture_stream.
    handle: int
    # Non-owning PyTorch view used to make the private native stream current for launches.
    external_stream: torch.cuda.ExternalStream

    @classmethod
    def create(cls, runtime: Any, device: torch.device) -> "TrtllmDaBodyCaptureStream":
        """Create a native capture stream and tie its destruction to Python ownership."""
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        handle = int(runtime.create_da_body_capture_stream(device_index))
        owner = cls(
            device_index=device_index,
            handle=handle,
            external_stream=torch.cuda.ExternalStream(handle, device=device),
        )
        weakref.finalize(
            owner, runtime.destroy_da_body_capture_stream, device_index, handle
        )
        return owner


@functools.cache
def _get_trtllm_da_body_capture_stream(
    device_index: int,
) -> TrtllmDaBodyCaptureStream:
    """Return the single process-wide DA body-capture stream for one CUDA device."""
    runtime = get_trtllm_moe_sm100_module()
    return TrtllmDaBodyCaptureStream.create(runtime, torch.device("cuda", device_index))


@functools.cache
def _get_trtllm_da_body_capture_lock(device_index: int) -> threading.RLock:
    """Serialize direct child-graph capture on the one reusable per-device stream."""
    _ = device_index
    return threading.RLock()


@dataclass(frozen=True)
class TrtllmDaBodyWorkspace:
    """Maximum typed body buffers shared by mutually exclusive SWITCH bodies."""

    # Maximum-sized dtype-specific tensors reused by every mutually exclusive body.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:BF16DABodyBuffers.gemm1_output and peer records.
    tensors: tuple[torch.Tensor, ...]
    # Shared per-device native stream used only for serialized child-graph body capture.
    capture_stream: TrtllmDaBodyCaptureStream


@dataclass(frozen=True)
class TrtllmDaProfileWorkspace:
    """Prepared buffers for one tactic and one cold-L2 profiling lane."""

    # Concrete body whose tactic and tile determine the prepared kernel ABI.
    body: DABody
    # Single-tile metadata collection populated from native replay IDs on every invocation.
    routing_metadata: TrtllmMoERoutingMetadata
    # Dtype-specific buffers retaining every pointer used by the profiled body.
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TrtllmDaResources:
    """Production graph-stable workspace owned by one concurrent replay lane."""

    # Published plan generation for which every retained buffer was prepared.
    generation: int
    # One fused multi-tile metadata allocation shared by all deduplicated bodies.
    routing_metadata: TrtllmMoERoutingMetadata
    # One maximum typed workspace shared by all mutually exclusive SWITCH bodies.
    body_workspace: TrtllmDaBodyWorkspace
    # Device scalar written by the selector with the replay-selected body index.
    selected_body: torch.Tensor
    # Stable FromLogits router outputs, or None for caller-precomputed routing.
    canonical_routing: Optional[TRTLLMCanonicalRouting] = None


@dataclass(frozen=True)
class TrtllmDaSwitchCaptureState:
    """Named Python view of the native CUDA SWITCH capture-state ABI."""

    # Number of fixed scalar fields before the variable child-graph tail.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:DASwitchCaptureState.kHeaderSize.
    HEADER_SIZE: ClassVar[int] = 5
    # Index of the conditional child-graph count in the native header.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:DASwitchCaptureState.kBodyCountIndex.
    BODY_COUNT_INDEX: ClassVar[int] = 4
    # Minimum number of unique bodies required to retain a SWITCH.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:DASwitchCaptureState.kMinimumBodyCount.
    MINIMUM_BODY_COUNT: ClassVar[int] = 2

    # Complete native state passed back unchanged when the outer graph is joined.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:DASwitchCaptureState.capture_id.
    native: tuple[int, ...]
    # CUDA-owned child graph handles populated with exact dtype-specific bodies.
    # Sync with csrc/trtllm_fused_moe_kernel_launcher.cu:DASwitchCaptureState.body_graphs.
    body_graph_handles: tuple[int, ...]

    @classmethod
    def from_native(cls, state: Sequence[int]) -> "TrtllmDaSwitchCaptureState":
        """Decode and validate the stable native state returned by begin capture."""
        if len(state) < cls.HEADER_SIZE:
            raise ValueError("DA SWITCH capture state is incomplete")
        body_count = int(state[cls.BODY_COUNT_INDEX])
        if body_count < cls.MINIMUM_BODY_COUNT:
            raise ValueError("DA SWITCH capture state requires at least two bodies")
        if len(state) != cls.HEADER_SIZE + body_count:
            raise ValueError("DA SWITCH capture state has an invalid body count")
        native = tuple(int(value) for value in state)
        return cls(native=native, body_graph_handles=native[cls.HEADER_SIZE :])

    def to_native(self) -> list[int]:
        """Return a TVM-FFI-compatible copy of the complete native state."""
        return list(self.native)

    @property
    def conditional_node_handle(self) -> int:
        """Return the native terminal node used to serialize one workspace lane."""
        return self.native[1]


@functools.cache
def _moe_topk_ids_init(num_experts: int, *, packed: bool = True):
    """Return a top-k-id initializer for a given expert count.

    ``PackedPrecomputed`` profiling needs ``(expert_id << 16) | bf16(weight)``,
    while ``UnpackedPrecomputed`` profiling needs plain expert IDs. Cache the
    closure for object identity preservation in rebuilt tuning configs.
    """

    def _init(
        shapes: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        expert_ids = make_random_topk_ids(
            num_experts=num_experts,
            num_tokens=math.prod(shapes[:-1]),
            top_k=shapes[-1],
            device=device,
        ).view(shapes)
        if not packed:
            return expert_ids
        expert_weights = torch.ones(shapes, dtype=torch.bfloat16, device=device).view(
            torch.int16
        )
        return (expert_ids << 16) | expert_weights

    return _init


@functools.cache
def is_trtllm_moe_supported(
    dtype_weights: DtypeTrtllmGen,
    dtype_act: DtypeTrtllmGen,
    quant_method: Optional[str] = None,
) -> bool:
    arch = get_compute_capability(torch.cuda.current_device())
    if arch[0] < 10:
        return False
    if dtype_weights not in [
        DtypeTrtllmGen.Bfloat16,
        DtypeTrtllmGen.E4m3,
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.MxE2m1,
    ]:
        return False
    if (
        dtype_weights == DtypeTrtllmGen.Bfloat16
        and dtype_act != DtypeTrtllmGen.Bfloat16
    ):
        return False
    if dtype_weights == DtypeTrtllmGen.E4m3 and dtype_act != DtypeTrtllmGen.E4m3:
        return False
    if dtype_weights == DtypeTrtllmGen.E2m1 and dtype_act != DtypeTrtllmGen.E2m1:
        return False
    if dtype_weights == DtypeTrtllmGen.MxE2m1 and dtype_act not in [
        DtypeTrtllmGen.MxE2m1,
        DtypeTrtllmGen.MxE4m3,
        DtypeTrtllmGen.Bfloat16,
    ]:
        return False
    return True


def _maybe_get_cached_w3_w1_permute_indices(
    _cache_permute_indices,
    dst_w3_w1_weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: Union[None, int] = None,
    is_gated_act_gemm: bool = True,
) -> torch.Tensor:
    # Include every parameter that changes the generated permutation.
    cache_key = (
        "w3_w1",
        dst_w3_w1_weight.shape,
        epilogue_tile_m,
        num_elts_per_sf,
        is_gated_act_gemm,
    )
    if cache_key not in _cache_permute_indices:
        # Get permute indices and chain them together
        if is_gated_act_gemm:
            permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(dst_w3_w1_weight)
        else:
            permute0 = torch.arange(dst_w3_w1_weight.shape[0], dtype=torch.long)
        if num_elts_per_sf is None:
            permute1 = get_shuffle_matrix_a_row_indices(
                dst_w3_w1_weight, epilogue_tile_m=epilogue_tile_m
            )
        else:
            permute1 = get_shuffle_matrix_sf_a_row_indices(
                dst_w3_w1_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf,
            )
        # Memoize permute indices as recompute is **very** costly
        _cache_permute_indices[cache_key] = permute0[permute1].to(
            dst_w3_w1_weight.device
        )
    permute_indices = _cache_permute_indices[cache_key]
    return permute_indices


def get_w2_permute_indices_with_cache(
    _cache_permute_indices,
    dst_w2_weight: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: Union[None, int] = None,
    is_gated_act_gemm: bool | None = None,
) -> torch.Tensor:
    # Keep gated and non-gated preparation in separate cache namespaces. The
    # row mapping is currently identical, but the cached tensor is device-resident
    # and must not be shared across activation-specific preparation lifetimes.
    cache_key = (
        "w2",
        dst_w2_weight.shape,
        epilogue_tile_m,
        num_elts_per_sf,
        is_gated_act_gemm,
    )
    if cache_key not in _cache_permute_indices:
        if num_elts_per_sf is None:
            permute_indices = get_shuffle_matrix_a_row_indices(
                dst_w2_weight, epilogue_tile_m
            ).to(dst_w2_weight.device)
        else:
            permute_indices = get_shuffle_matrix_sf_a_row_indices(
                dst_w2_weight,
                epilogue_tile_m=epilogue_tile_m,
                num_elts_per_sf=num_elts_per_sf,
            ).to(dst_w2_weight.device)
        # Memoize permute indices as recompute is **very** costly
        _cache_permute_indices[cache_key] = permute_indices
    permute_indices = _cache_permute_indices[cache_key]
    return permute_indices


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    assert x.dim() == 2, f"x should be a 2D tensor, not {x.dim()}"
    M, K = x.shape
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[: (M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2 :]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x: torch.Tensor) -> torch.Tensor:
    r"""Reorder rows of a weight tensor for the TensorRT-LLM gated-activation GEMM layout.

    Pure-PyTorch reimplementation of the TensorRT-LLM ``reorderRowsForGatedActGemm``
    helper.  Used to pre-permute the up/gate weight matrix so that the fused
    gated-activation kernels can access the two halves with a single contiguous
    load.

    Parameters
    ----------
    x : torch.Tensor
        Weight tensor whose rows will be permuted.  Any dtype is accepted; only
        the row dimension is reordered.

    Returns
    -------
    torch.Tensor
        Row-permuted copy of ``x`` (materialized as a new contiguous tensor;
        PyTorch advanced indexing always copies, never aliases).
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


def convert_to_block_layout(input_tensor: torch.Tensor, blockK: int) -> torch.Tensor:
    r"""Reshape a 2-D tensor into a 3-D block layout.

    Splits the inner ``K`` dimension into ``K // blockK`` blocks of size
    ``blockK`` and transposes so the block dimension is outermost.  This is the
    canonical layout consumed by TensorRT-LLM block-scaled MoE kernels.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor of shape ``(M, K)``.
    blockK : int
        Block size along the ``K`` dimension.  ``K`` must be divisible by
        ``blockK``.

    Returns
    -------
    torch.Tensor
        Reshaped contiguous tensor of shape ``(K // blockK, M, blockK)``.
    """
    M, K = input_tensor.shape
    assert K % blockK == 0, "K must be divisible by blockK"
    return input_tensor.view(M, K // blockK, blockK).permute(1, 0, 2).contiguous()


@functools.cache
def get_cutlass_fused_moe_module(backend: str = "100", use_fast_build: bool = False):
    if backend in ("120", "121"):
        module = gen_cutlass_fused_moe_sm120_module(use_fast_build).build_and_load()
    elif backend == "103":
        module = gen_cutlass_fused_moe_sm103_module(use_fast_build).build_and_load()
    elif backend in ("100", "107", "110"):
        module = gen_cutlass_fused_moe_sm100_module(use_fast_build).build_and_load()
    elif backend == "90":
        module = gen_cutlass_fused_moe_sm90_module(use_fast_build).build_and_load()
    elif backend == "89":
        module = gen_cutlass_fused_moe_sm89_module(use_fast_build).build_and_load()
    else:
        raise ValueError(f"Invalid backend: {backend}")

    # Set DeepGEMM JIT include directories after module is loaded
    from ..jit import env as jit_env

    deepgemm_include_dir = str(
        jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "tensorrt_llm"
    )
    module.set_deepgemm_jit_include_dirs([deepgemm_include_dir])

    class MoERunner(TunableRunner):
        # avoid overhead of creating a new runner in forward pass
        runner_dict: Dict[
            Tuple[
                torch.dtype,
                torch.dtype,
                torch.dtype,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
            ],
            Any,
        ] = dict()
        tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    (0,),
                    (0,),
                    get_hybrid_num_tokens_buckets(8192),
                    make_hybrid_bucket_mapper(8192),
                ),
            )
        )

        def __init__(
            self,
            x_dtype: torch.dtype,
            weight_dtype: torch.dtype,
            output_dtype: torch.dtype,
            top_k: int,
            tp_size: int,
            tp_rank: int,
            ep_size: int,
            ep_rank: int,
            cluster_size: int,
            cluster_rank: int,
            enable_alltoall: bool,
            use_deepseek_fp8_block_scale: bool,
            use_w4_group_scaling: bool,
            use_mxfp8_act_scaling: bool,
            min_latency_mode: bool,
            enable_pdl: bool,
            activation_type: ActivationType,
            use_packed_weights: bool,
            use_fused_finalize: bool,
            use_wfp4afp8_humming: bool,
        ):
            self.x_dtype = x_dtype
            self.weight_dtype = weight_dtype
            self.output_dtype = output_dtype
            self.top_k = top_k
            self.tp_size = tp_size
            self.tp_rank = tp_rank
            self.ep_size = ep_size
            self.ep_rank = ep_rank
            self.cluster_size = cluster_size
            self.cluster_rank = cluster_rank
            self.enable_alltoall = enable_alltoall
            self.use_deepseek_fp8_block_scale = use_deepseek_fp8_block_scale
            self.use_w4_group_scaling = use_w4_group_scaling
            self.use_mxfp8_act_scaling = use_mxfp8_act_scaling
            self.use_wfp4afp8_humming = use_wfp4afp8_humming
            self.min_latency_mode = min_latency_mode
            self.enable_pdl = enable_pdl
            self.use_packed_weights = use_packed_weights
            self.use_fused_finalize = use_fused_finalize
            instance_key = (
                x_dtype,
                weight_dtype,
                output_dtype,
                use_deepseek_fp8_block_scale,
                use_w4_group_scaling,
                use_mxfp8_act_scaling,
                use_packed_weights,
                use_fused_finalize,
                use_wfp4afp8_humming,
            )
            self.activation_type = activation_type
            # Set by tuning flow to indicate which GEMM stage (1 or 2) to filter tactics for
            self.gemm_idx_for_tuning: Optional[int] = None

            if instance_key not in MoERunner.runner_dict:
                MoERunner.runner_dict[instance_key] = module.init(
                    x_dtype,
                    weight_dtype,
                    output_dtype,
                    use_deepseek_fp8_block_scale,
                    use_w4_group_scaling,
                    use_mxfp8_act_scaling,
                    use_packed_weights,
                    use_fused_finalize,
                    use_wfp4afp8_humming,
                )

            self.fused_moe_runner = MoERunner.runner_dict[instance_key]

        def get_cache_key_extras(self, _inputs: List[torch.Tensor]) -> tuple:
            # Stage profiling passes only activation and weight tensors, so the
            # profile key captures their shapes but not constructor-fixed options
            # such as top-k, parallel ranks, quantization mode, or activation.
            # The in-memory runner hash distinguishes instances, but it is
            # intentionally excluded from persisted file keys. Include those
            # options here to prevent runners with identical tensor profiles from
            # reusing incompatible saved tactics.
            return (
                self.x_dtype,
                self.weight_dtype,
                self.output_dtype,
                self.top_k,
                self.tp_size,
                self.tp_rank,
                self.ep_size,
                self.ep_rank,
                self.cluster_size,
                self.cluster_rank,
                self.enable_alltoall,
                self.use_deepseek_fp8_block_scale,
                self.use_w4_group_scaling,
                self.use_mxfp8_act_scaling,
                self.use_wfp4afp8_humming,
                self.min_latency_mode,
                self.enable_pdl,
                int(self.activation_type),
                self.use_packed_weights,
                self.use_fused_finalize,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            # Prefer filtering tactics by GEMM stage to avoid invalid combos during tuning
            try:
                gemm1_count = self.fused_moe_runner.get_gemm1_tactic_count()
                gemm2_count = self.fused_moe_runner.get_gemm2_tactic_count()
                total = gemm1_count + gemm2_count
            except Exception:
                return list(range(self.fused_moe_runner.get_tactic_num()))

            stage = getattr(self, "gemm_idx_for_tuning", None)
            if stage == 1:
                all_tactics = list(range(gemm1_count))
            elif stage == 2:
                all_tactics = list(range(gemm1_count, gemm1_count + gemm2_count))
            else:
                all_tactics = list(range(total))

            # Pre-filter tactics with zero occupancy on the current device.
            # This eliminates tactics that would fail during profiling with
            # "GPU lacks the shared memory resources" errors — notably, SM89 (Ada)
            # tile configs used as fallback for pure FP8 MoE on SM120 (Blackwell CC 12.0)
            # where native SM120 FP8 MoE GEMM kernels are not yet available.
            try:
                get_occ = self.fused_moe_runner.get_tactic_occupancy
            except AttributeError:
                # get_tactic_occupancy not available in this build; skip pre-filtering
                return all_tactics if all_tactics else [-1]

            valid_tactics = []
            for t in all_tactics:
                try:
                    if get_occ(t) > 0:
                        valid_tactics.append(t)
                except Exception as e:
                    # If the query fails unexpectedly, include the tactic and let
                    # the autotuner handle any errors during profiling.
                    logger.warning(
                        "get_tactic_occupancy failed for tactic %d: %s; including in autotuner",
                        t,
                        e,
                    )
                    valid_tactics.append(t)
            # Fall back to all tactics if occupancy check eliminated everything
            # (e.g., on an unexpected architecture where all tactics report 0).
            # If all_tactics itself is empty (zero-tactic stage), return [-1] as
            # a sentinel so the autotuner contract is never violated with an empty list.
            if not all_tactics:
                return [-1]
            valid_tactics = valid_tactics if valid_tactics else all_tactics

            if not self.use_w4_group_scaling:
                return valid_tactics

            if stage not in (1, 2):
                return valid_tactics

            x, fc1_expert_weights, _, fc2_expert_weights, _ = inputs
            if stage == 1:
                gemm_n = int(fc1_expert_weights.shape[1])
                gemm_k = int(x.shape[1])
            else:
                gemm_n = int(fc2_expert_weights.shape[1])
                if fc2_expert_weights.dtype == torch.uint8:
                    gemm_k = int(fc2_expert_weights.shape[2]) * 2
                elif fc2_expert_weights.dtype == torch.int64:
                    gemm_k = int(fc2_expert_weights.shape[2]) * 16
                else:
                    gemm_k = int(fc2_expert_weights.shape[2])

            try:
                get_valid_tactics_for_shape = (
                    self.fused_moe_runner.get_valid_tactics_for_shape
                )
                shape_valid_tactics = set(
                    int(t)
                    for t in get_valid_tactics_for_shape(
                        int(stage), int(gemm_n), int(gemm_k)
                    )
                )
            except AttributeError:
                return valid_tactics
            except Exception as e:
                logger.warning(
                    "get_valid_tactics_for_shape failed for stage %s, N=%d, K=%d: %s; "
                    "including occupancy-valid tactics in autotuner",
                    stage,
                    gemm_n,
                    gemm_k,
                    e,
                )
                return valid_tactics

            filtered_tactics = [t for t in valid_tactics if t in shape_valid_tactics]
            return filtered_tactics if filtered_tactics else valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: Any = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            (
                x,
                fc1_expert_weights,
                fc1_expert_biases,
                fc2_expert_weights,
                fc2_expert_biases,
            ) = inputs
            self.fused_moe_runner.run_gemm_profile(
                x,
                fc1_expert_weights,
                fc1_expert_biases,
                fc2_expert_weights,
                fc2_expert_biases,
                self.top_k,
                self.tp_size,
                self.tp_rank,
                self.ep_size,
                self.ep_rank,
                self.cluster_size,
                self.cluster_rank,
                self.enable_alltoall,
                self.min_latency_mode,
                kwargs["gemm_idx"],
                tactic,
                do_preparation,
                self.enable_pdl,
                self.activation_type,
            )

        @classmethod
        @functools.lru_cache(maxsize=None)
        def refine_tuning_config(cls, tune_max_num_tokens: int):
            cls.tuning_config = TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        (0,),
                        (0,),
                        get_hybrid_num_tokens_buckets(tune_max_num_tokens),
                        make_hybrid_bucket_mapper(tune_max_num_tokens),
                    ),
                )
            )

    @register_custom_op(
        "flashinfer::cutlass_fused_moe",
        mutates_args=("output", "workspace_buffer"),
    )
    def cutlass_fused_moe(
        output: torch.Tensor,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        situ_beta: Optional[torch.Tensor] = None,
        situ_linear_beta: Optional[torch.Tensor] = None,
        swizzled_input_sf: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
        enable_pdl: Optional[bool] = None,
        activation_type: ActivationType = ActivationType.Swiglu,
        use_packed_weights: bool = False,
        use_fused_finalize: bool = True,
        use_wfp4afp8_humming: bool = False,
        profile_ids: Optional[List[int]] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(input.device)

        # allocate workspace for profiling
        moe_runner = MoERunner(
            x_dtype=input.dtype,
            weight_dtype=fc1_expert_weights.dtype,
            output_dtype=output_dtype,
            top_k=token_selected_experts.size(1),
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=cluster_size,
            cluster_rank=cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            enable_pdl=enable_pdl,
            activation_type=activation_type,
            use_packed_weights=use_packed_weights,
            use_fused_finalize=use_fused_finalize,
            use_wfp4afp8_humming=use_wfp4afp8_humming,
        )

        if profile_ids is None:
            tuner = AutoTuner.get()
            MoERunner.refine_tuning_config(tune_max_num_tokens)

            # Limit tactics to GEMM1 during tuning
            moe_runner.gemm_idx_for_tuning = 1
            _, gemm_tactic_1 = tuner.choose_one(
                "trtllm::fused_moe::gemm1",
                [moe_runner],
                MoERunner.tuning_config,
                [
                    input,
                    fc1_expert_weights,
                    fc1_expert_biases,
                    fc2_expert_weights,
                    fc2_expert_biases,
                ],
                gemm_idx=1,
            )

            # Limit tactics to GEMM2 during tuning
            moe_runner.gemm_idx_for_tuning = 2
            _, gemm_tactic_2 = tuner.choose_one(
                "trtllm::fused_moe::gemm2",
                [moe_runner],
                MoERunner.tuning_config,
                [
                    input,
                    fc1_expert_weights,
                    fc1_expert_biases,
                    fc2_expert_weights,
                    fc2_expert_biases,
                ],
                gemm_idx=2,
            )
        else:
            if len(profile_ids) != 2:
                raise ValueError(
                    "profile_ids must contain [gemm1_profile, gemm2_profile]"
                )
            gemm_tactic_1, gemm_tactic_2 = profile_ids

        run_moe = (
            moe_runner.fused_moe_runner.run_moe_min_latency
            if min_latency_mode
            else moe_runner.fused_moe_runner.run_moe
        )
        num_active_experts_per_node = torch.empty(
            (1,), dtype=torch.int32, device=input.device
        )
        experts_to_token_score = torch.empty(
            (fc2_expert_weights.shape[0], input.shape[0]),
            dtype=torch.float32,
            device=input.device,
        )
        active_expert_global_ids = torch.empty(
            (fc2_expert_weights.shape[0],),
            dtype=torch.int32,
            device=input.device,
        )
        min_latency_output = (
            [
                num_active_experts_per_node,
                experts_to_token_score,
                active_expert_global_ids,
            ]
            if min_latency_mode
            else []
        )
        run_moe(
            output,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            quant_scales,
            input_sf,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
            swizzled_input_sf,
            *min_latency_output,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            enable_alltoall,
            min_latency_mode,
            [gemm_tactic_1, gemm_tactic_2],
            enable_pdl,
            activation_type,
            workspace_buffer,
        )

        return (
            output
            if min_latency_mode
            else [
                output,
                num_active_experts_per_node,
                experts_to_token_score,
                active_expert_global_ids,
            ]
        )

    @register_fake_op("flashinfer::cutlass_fused_moe")
    def _fake_cutlass_fused_moe(
        output: torch.Tensor,
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc1_expert_biases: Optional[torch.Tensor],
        fc2_expert_weights: torch.Tensor,
        fc2_expert_biases: Optional[torch.Tensor],
        output_dtype: torch.dtype,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor] = None,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        situ_beta: Optional[torch.Tensor] = None,
        situ_linear_beta: Optional[torch.Tensor] = None,
        swizzled_input_sf: bool = True,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        cluster_size: int = 1,
        cluster_rank: int = 0,
        enable_alltoall: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        min_latency_mode: bool = False,
        tune_max_num_tokens: int = 8192,
        enable_pdl: Optional[bool] = None,
        activation_type: ActivationType = ActivationType.Swiglu,
        use_packed_weights: bool = False,
        use_fused_finalize: bool = True,
        use_wfp4afp8_humming: bool = False,
        profile_ids: Optional[List[int]] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        seq_len = input.shape[0]
        hidden_size = fc2_expert_weights.shape[1]

        if min_latency_mode:
            num_experts_on_rank = fc2_expert_weights.shape[0]
            output_shape = [seq_len * num_experts_on_rank, hidden_size]
            experts_to_token_score_shape = [num_experts_on_rank, seq_len]
            active_expert_global_ids_shape = [num_experts_on_rank]
            return [
                input.new_empty(output_shape, dtype=output_dtype),
                input.new_empty([1], dtype=torch.int32),
                input.new_empty(experts_to_token_score_shape, dtype=torch.float32),
                input.new_empty(active_expert_global_ids_shape, dtype=torch.int32),
            ]
        else:
            return [input.new_empty([seq_len, hidden_size], dtype=output_dtype)]

    def _cutlass_fused_moe_workspace_size(
        max_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts_total: int,
        top_k: int,
        *,
        x_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        output_dtype: torch.dtype = torch.bfloat16,
        activation_type: ActivationType = ActivationType.Swiglu,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        min_latency_mode: bool = False,
        use_deepseek_fp8_block_scale: bool = False,
        use_w4_group_scaling: bool = False,
        use_mxfp8_act_scaling: bool = False,
        use_fused_finalize: bool = True,
        use_packed_weights: bool = False,
        use_wfp4afp8_humming: bool = False,
    ) -> int:
        enable_pdl = device_support_pdl(torch.device("cuda"))
        moe_runner = MoERunner(
            x_dtype=x_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            top_k=top_k,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            cluster_size=1,
            cluster_rank=0,
            enable_alltoall=False,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            enable_pdl=enable_pdl,
            activation_type=activation_type,
            use_packed_weights=use_packed_weights,
            use_fused_finalize=use_fused_finalize,
            use_wfp4afp8_humming=use_wfp4afp8_humming,
        )
        return int(
            moe_runner.fused_moe_runner.get_workspace_size(
                max_num_tokens,
                hidden_size,
                intermediate_size,
                num_experts_total,
                top_k,
                tp_size,
                tp_rank,
                ep_size,
                ep_rank,
                min_latency_mode,
                activation_type,
            )
        )

    # Register the module
    return SimpleNamespace(
        MoERunner=MoERunner,
        cutlass_fused_moe=cutlass_fused_moe,
        cutlass_fused_moe_workspace_size=_cutlass_fused_moe_workspace_size,
        interleave_moe_weights_for_sm90_mixed_gemm=(
            module.interleave_moe_weights_for_sm90_mixed_gemm
        ),
    )


# ref: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/custom_ops/torch_custom_ops.py#L121
@flashinfer_api(trace=cutlass_fused_moe_trace)
def cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_packed_weights: bool = False,
    use_wfp4afp8_humming: bool = False,
    tune_max_num_tokens: int = 8192,
    enable_pdl: Optional[bool] = None,
    activation_type: ActivationType = ActivationType.Swiglu,
    swizzled_input_sf: bool = True,
    use_fused_finalize: bool = True,
    profile_ids: Optional[List[int]] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    *,
    situ_beta: Optional[torch.Tensor] = None,
    situ_linear_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a Mixture of Experts (MoE) layer using CUTLASS backend.

    This function implements a fused MoE layer that combines expert selection, expert computation,
    and output combination into a single operation. It uses CUTLASS for efficient matrix multiplication
    and supports various data types and parallelism strategies.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape [num_tokens, hidden_size].
        Support float, float16, bfloat16, float8_e4m3fn and nvfp4.
        For FP8, the input must be quantized.
        For NVFP4, both quantized and non-quantized inputs are supported.

    token_selected_experts : torch.Tensor
        Indices of selected experts for each token.

    token_final_scales : torch.Tensor
        Scaling factors for each token's expert outputs.

    fc1_expert_weights : torch.Tensor
        GEMM1 weights for each expert.

    fc2_expert_weights : torch.Tensor
        GEMM2 weights for each expert.

    output_dtype : torch.dtype
        Desired output data type.

    quant_scales : List[torch.Tensor]
        Quantization scales for the operation.

        NVFP4:
            - gemm1 activation global scale
            - gemm1 weights block scales
            - gemm1 dequant scale
            - gemm2 activation global scale
            - gemm2 weights block scales
            - gemm2 dequant scale

        FP8:
            - gemm1 dequant scale
            - gemm2 activation quant scale
            - gemm2 dequant scale
            - gemm1 input dequant scale

        Humming FP8 x MXFP4 (``use_wfp4afp8_humming=True``):
            - gemm1 folded weight block scales
            - gemm1 per-local-expert residual scale, including the fixed ``2^6`` compensation
            - reserved scalar or per-local-expert gemm2 activation scale
            - gemm2 folded weight block scales
            - gemm2 per-local-expert residual scale, including the fixed ``2^6`` compensation

    fc1_expert_biases : Optional[torch.Tensor]
        GEMM1 biases for each expert.

    fc2_expert_biases : Optional[torch.Tensor]
        GEMM1 biases for each expert.

    input_sf : Optional[torch.Tensor]
        Input scaling factor for quantization.

    swiglu_alpha : Optional[torch.Tensor]
        Swiglu alpha for swiglu activation.

    swiglu_beta : Optional[torch.Tensor]
        Swiglu beta for swiglu activation.

    swiglu_limit : Optional[torch.Tensor]
        Swiglu limit for swiglu activation.

    situ_beta : Optional[torch.Tensor]
        Per-expert ``beta`` tanh scale for the ``Situ`` activation (float32,
        ``[num_experts_on_rank]``). ``None`` uses ``DEFAULT_SITU_BETA``.

    situ_linear_beta : Optional[torch.Tensor]
        Per-expert ``linear_beta`` tanh scale for the ``Situ`` activation (float32,
        ``[num_experts_on_rank]``). ``None`` uses ``DEFAULT_SITU_LINEAR_BETA``.

    tp_size : int = 1
        Tensor parallelism size. Defaults to 1.

    tp_rank : int = 0
        Tensor parallelism rank. Defaults to 0.

    ep_size : int = 1
        Expert parallelism size. Defaults to 1.

    ep_rank : int = 0
        Expert parallelism rank. Defaults to 0.

    cluster_size : int = 1
        Cluster size. Defaults to 1.

    cluster_rank : int = 0
        Cluster rank. Defaults to 0.

    output : Optional[torch.Tensor] = None
        The output tensor, if not provided, will be allocated internally.

    enable_alltoall : bool = False
        Whether to enable all-to-all communication for expert outputs. Defaults to False.

    use_deepseek_fp8_block_scale : bool = False
        Whether to use FP8 block scaling. Defaults to False.

    use_w4_group_scaling : bool = False
        Whether to use W4A8 group scaling. Defaults to False.

    use_mxfp8_act_scaling : bool = False
        Whether to use MXFP8 activation scaling. Defaults to False.

    min_latency_mode : bool = False
        Whether to use minimum latency mode. Defaults to False.

    use_packed_weights : bool = False
        Whether to use packed uint4x2 weights passed as packed uint8 values. Defaults to False.

    use_wfp4afp8_humming : bool = False
        Selects the Humming-style MXFP4-weight x FP8-activation Hopper path with pre-MMA E8M0
        scale fusion. This flag is separate from W4A16 because both paths use uint8 FP4 weight
        storage and ``use_w4_group_scaling=True``.

    tune_max_num_tokens : int = 8192
        Maximum number of tokens for tuning. Defaults to 8192.

    enable_pdl : Optional[bool]
        Whether to launch the kernel with Programmatic Dependent Launch (PDL).
        ``None`` (default) lets the runtime pick a safe value based on the device
        and surrounding stream operations; pass ``True`` to force PDL when every
        adjacent kernel on the stream also supports it, or ``False`` to disable.

    activation_type: ActivationType = ActivationType.Swiglu
        Activation to apply on for GEMM1, note that Relu2 means non-gated GEMM1

    swizzled_input_sf : bool = True
        Whether the input scaling factor (input_sf) is in swizzled layout. Defaults to True.
        Set to False when input_sf is in linear layout, e.g. after FP4 allgather/alltoall
        communication where the scaling factors are received in linear (non-swizzled) format.
        Only relevant when input_sf is not None.

    use_fused_finalize : bool = True
        Whether to fuse the top-k expert reduction ("finalize") into the GEMM2 epilogue.
        Defaults to True for best performance. The fused epilogue reduces expert outputs via
        non-associative atomics, so results are not deterministic run-to-run. Set to
        False to use the non-fused, deterministic finalize path.

    profile_ids : Optional[List[int]]
        Optional ``[gemm1_profile, gemm2_profile]`` override. Both values are absolute indices in
        the runner's combined tactic list; ``-1`` keeps the default tactic for that GEMM.

    workspace_buffer : Optional[torch.Tensor]
        Pre-allocated scratch buffer reused across calls to eliminate per-call workspace
        allocation (which can reach 10-20 GiB at large batch size). If ``None`` (default),
        the workspace is allocated and freed on every call. To opt in:

        1. Query the required size once at model-load time::

               ws_bytes = cutlass_fused_moe_workspace_size(
                   max_num_tokens, hidden_size, intermediate_size,
                   num_experts_total, top_k, x_dtype=..., weight_dtype=...,
                   use_fused_finalize=<same as here>, device=input.device)
               workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=input.device)

        2. Pass ``workspace_buffer=workspace`` on every forward call.

        The buffer must be a 1-D ``torch.uint8`` or ``torch.int8`` tensor on the same
        CUDA device as the input, with at least ``ws_bytes`` bytes and 128-byte-aligned
        data pointer (``torch.empty`` on CUDA satisfies all of these). Allocate one buffer
        per CUDA stream context; overlapping micro-batches on separate streams each need
        their own buffer. A buffer sized for the maximum token count is valid for all
        smaller counts on the same call.

    Returns
    -------
    out: torch.Tensor
        Output tensor of shape [seq_len, hidden_size].


    Raises
    ------
    NotImplementedError:
        If any of the following features are requested but not implemented:
            - Minimum Latency Mode

    Note
    ----
    - The function supports various data types including FP32, FP16, BF16, FP8, and NVFP4.
    - It implements both tensor parallelism and expert parallelism.
    - Currently, some advanced features like FP8 block scaling and minimum latency mode
        are not implemented for Blackwell architecture.
    """
    major, minor = get_compute_capability(input.device)
    device_arch = f"{major * 10 + minor}"

    if use_wfp4afp8_humming and device_arch != "90":
        raise NotImplementedError(
            "Humming-style MXFP4 x FP8 fused MoE is only implemented for SM90."
        )

    if min_latency_mode:
        raise NotImplementedError("min latency mode not yet implemented for Blackwell.")

    if use_deepseek_fp8_block_scale:
        if device_arch != "90":
            raise NotImplementedError(
                "FP8 block scaling not yet implemented for Blackwell."
            )
        elif not is_cuda_version_at_least("12.8"):
            raise NotImplementedError("FP8 block scaling requires CUDA 12.8 or newer.")

    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)

    num_rows = input.shape[0]
    if min_latency_mode:
        num_rows *= fc2_expert_weights.shape[0]
    hidden_size = fc2_expert_weights.shape[1]
    output_shape = (num_rows, hidden_size)

    if output is None:
        output = torch.empty(output_shape, dtype=output_dtype, device=input.device)
    else:
        check_shape_dtype_device(
            output, output_shape, output_dtype, input.device, "output"
        )

    # Module loading and runner construction inspect the current CUDA device.
    # Keep the Python-side context aligned with the input; the C++ runner also
    # installs its own guard for execution and workspace allocation.
    with torch.cuda.device(input.device):
        return get_cutlass_fused_moe_module(device_arch).cutlass_fused_moe(
            output,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            fc1_expert_biases,
            fc2_expert_weights,
            fc2_expert_biases,
            output_dtype,
            quant_scales,
            input_sf,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            situ_beta,
            situ_linear_beta,
            swizzled_input_sf,
            tp_size,
            tp_rank,
            ep_size,
            ep_rank,
            cluster_size,
            cluster_rank,
            use_packed_weights=use_packed_weights,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            min_latency_mode=min_latency_mode,
            tune_max_num_tokens=tune_max_num_tokens,
            enable_pdl=enable_pdl,
            activation_type=activation_type,
            use_fused_finalize=use_fused_finalize,
            use_wfp4afp8_humming=use_wfp4afp8_humming,
            profile_ids=profile_ids,
            workspace_buffer=workspace_buffer,
        )


def cutlass_fused_moe_workspace_size(
    max_num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts_total: int,
    top_k: int,
    *,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype = torch.bfloat16,
    activation_type: ActivationType = ActivationType.Swiglu,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    min_latency_mode: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    use_fused_finalize: bool = True,
    use_packed_weights: bool = False,
    use_wfp4afp8_humming: bool = False,
    device: Optional[torch.device] = None,
) -> int:
    """Return the workspace buffer size in bytes required by :func:`cutlass_fused_moe`.

    Allocate the returned number of bytes once at model-load time (as a 1-D
    ``torch.uint8`` tensor) and pass the buffer as ``workspace_buffer=`` on
    every :func:`cutlass_fused_moe` call to eliminate the per-call
    multi-GiB scratch allocation.

    Parameters
    ----------
    max_num_tokens : int
        Maximum ``input.shape[0]`` that will be seen at runtime.  The buffer
        is monotonically sized by this value, so a buffer allocated for the
        maximum shape is valid for all smaller shapes on the same call.
    hidden_size : int
        Logical hidden dimension.
    intermediate_size : int
        Logical, unpacked intermediate dimension. For packed weights this is
        ``fc2_expert_weights.shape[2]`` multiplied by the format's packing
        factor, not the packed storage dimension itself.
    num_experts_total : int
        Global expert count across all EP ranks. This must equal
        ``fc2_expert_weights.shape[0] * ep_size``.
    top_k : int
        Number of selected experts per token.
    x_dtype, weight_dtype : torch.dtype
        Input and weight dtypes, used to select the kernel runner.
    output_dtype : torch.dtype, optional
        Output dtype (default: ``torch.bfloat16``).
    device : torch.device, optional
        CUDA device on which the workspace will be used. Defaults to the
        current CUDA device.

    Note
    ----
    Allocate one workspace buffer per CUDA stream context.  Overlapping
    micro-batches on separate streams each need their own buffer.
    """
    if max_num_tokens <= 0:
        raise ValueError(f"max_num_tokens must be positive, got {max_num_tokens}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if intermediate_size <= 0:
        raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
    if num_experts_total <= 0:
        raise ValueError(f"num_experts_total must be positive, got {num_experts_total}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")
    if num_experts_total % ep_size != 0:
        raise ValueError(
            f"num_experts_total ({num_experts_total}) must be divisible by "
            f"ep_size ({ep_size})"
        )

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"device must be a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

    with torch.cuda.device(device):
        major, minor = get_compute_capability(device)
        device_arch = f"{major * 10 + minor}"
        return get_cutlass_fused_moe_module(
            device_arch
        ).cutlass_fused_moe_workspace_size(
            max_num_tokens,
            hidden_size,
            intermediate_size,
            num_experts_total,
            top_k,
            x_dtype=x_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            activation_type=activation_type,
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            min_latency_mode=min_latency_mode,
            use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
            use_w4_group_scaling=use_w4_group_scaling,
            use_mxfp8_act_scaling=use_mxfp8_act_scaling,
            use_fused_finalize=use_fused_finalize,
            use_packed_weights=use_packed_weights,
            use_wfp4afp8_humming=use_wfp4afp8_humming,
        )


# trtllmgen-moe-fp8


@dataclass
class MoeRunnerInputs:
    """MoERunner inputs.

    Field order defines the flat-list index used by the autotuner.
    """

    output: torch.Tensor
    routing_logits: Optional[torch.Tensor]
    topk_ids: Optional[torch.Tensor]
    expert_weights: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    gemm1_lora_delta: Optional[torch.Tensor]
    per_token_scale: Optional[torch.Tensor]

    _FIELDS = (
        "output",
        "routing_logits",
        "topk_ids",
        "expert_weights",
        "hidden_states",
        "hidden_states_scale",
        "gemm1_lora_delta",
        "per_token_scale",
    )

    # Index of the dynamic dimension for each field.
    # hidden_states_scale is excluded: its layout differs by op (fp8 DeepSeekFp8
    # uses [hidden_size//128, num_tokens] while fp4/MxFp8 uses [num_tokens, ...]),
    # so _make_tuning_config infers it from the actual tensor at runtime.
    _DYNAMIC_DIM = {
        "output": 0,
        "routing_logits": 0,
        "topk_ids": 0,
        "expert_weights": 0,
        "hidden_states": 0,
        "gemm1_lora_delta": 0,
        "per_token_scale": 0,
    }

    def to_list(self) -> List[Optional[torch.Tensor]]:
        return [getattr(self, name) for name in MoeRunnerInputs._FIELDS]

    @classmethod
    def from_list(cls, lst: List) -> "MoeRunnerInputs":
        return cls(**{name: lst[i] for i, name in enumerate(cls._FIELDS)})

    @classmethod
    def idx(cls, name: str) -> int:
        return cls._FIELDS.index(name)


# Backward-compatible alias: this class was previously named ``MoEInputs``.
# Renamed to ``MoeRunnerInputs`` to disambiguate from the unified-API input
# grouping (the ``MoEActivationPack`` / ``MoEWeightPack`` lifetime split) — see
# PR #3093 review G6.  Old name kept working for out-of-tree importers and tests.
MoEInputs = MoeRunnerInputs


def _alloc_trtllm_moe_output(
    num_tokens: int,
    hidden_size: int,
    do_finalize: bool,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Allocate the finalized-output buffer for a trtllm-gen MoE op.
    When `do_finalize` is false, return a zero-width `(num_tokens, 0)`
    placeholder instead: the leading `num_tokens` dim is preserved for
    shape checks and the autotuner's token bucketing.
    """
    return torch.empty(
        num_tokens, hidden_size if do_finalize else 0, dtype=dtype, device=device
    )


def _fake_trtllm_moe_output(
    hidden_states: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    do_finalize: bool,
    output: Optional[torch.Tensor] = None,
    expert_weights: Optional[torch.Tensor] = None,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> List[torch.Tensor]:
    """Model the native TRTLLM MoE result contract for FakeTensor tracing."""
    num_tokens = hidden_states.shape[0]
    if do_finalize:
        finalized = (
            output
            if output is not None and output.shape[1] == hidden_size
            else hidden_states.new_empty(
                (num_tokens, hidden_size), dtype=torch.bfloat16
            )
        )
        if gemm1_lora_delta is None:
            return [finalized]
    else:
        # Routing-dependent expert padding makes the first dimension dynamic.
        gemm2_rows = torch.library.get_ctx().new_dynamic_size()
        finalized = hidden_states.new_empty(
            (gemm2_rows, hidden_size), dtype=torch.bfloat16
        )

    total_top_k = top_k + num_fused_shared_experts
    expanded_idx_to_permuted_idx = hidden_states.new_empty(
        (num_tokens * total_top_k,), dtype=torch.int32
    )
    if not do_finalize:
        weights = (
            expert_weights
            if expert_weights is not None and expert_weights.numel() > 0
            else hidden_states.new_empty(
                (num_tokens, total_top_k), dtype=torch.bfloat16
            )
        )
        result = [finalized, weights, expanded_idx_to_permuted_idx]
    else:
        result = [finalized, expanded_idx_to_permuted_idx]

    if gemm1_lora_delta is not None:
        gemm1_rows = torch.library.get_ctx().new_dynamic_size()
        result.append(
            hidden_states.new_empty(
                (gemm1_rows, intermediate_size), dtype=torch.bfloat16
            )
        )
    return result


def _unpack_trtllm_moe_output(
    intermediate_output,
    output: torch.Tensor,
    do_finalize: bool,
    gemm1_lora_delta: Optional[torch.Tensor],
    expert_weights: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Translate the ``Array<Tensor>`` returned by ``FusedMoeLauncher::run`` to
    the Python-facing ``List[torch.Tensor]``.

    A slot the launcher borrowed from the caller rather than allocated comes back
    empty, and calling ``from_dlpack`` on it raises "invalid capsule". That is the
    case for ``output``, which the caller always provides, and for
    ``expert_weights`` whenever the caller passed a buffer down. For those two we
    return the caller's own tensor instead of unpacking the slot.
    """
    if do_finalize and gemm1_lora_delta is None:
        return [output]
    elif do_finalize and gemm1_lora_delta is not None:
        return [
            output,
            torch.from_dlpack(intermediate_output[1]),  # expanded_idx_to_permuted_idx
            torch.from_dlpack(intermediate_output[2]),  # gemm1_output
        ]

    # do_finalize=False: index 1 is expert_weights.  Only convert it when the
    # launcher owned (allocated) the buffer -- converting a borrowed slot would
    # dlpack an empty Tensor and raise "invalid capsule".
    weights = (
        expert_weights
        if expert_weights is not None and expert_weights.numel() > 0
        else torch.from_dlpack(intermediate_output[1])
    )
    result = [
        torch.from_dlpack(intermediate_output[0]),  # gemm2_output
        weights,  # expert_weights
        torch.from_dlpack(intermediate_output[2]),  # expanded_idx_to_permuted_idx
    ]
    if gemm1_lora_delta is not None:
        result.append(torch.from_dlpack(intermediate_output[3]))  # gemm1_output
    return result


def _enabled_trtllm_da_config() -> Optional["TrtllmDaConfig"]:
    """Resolve the complete DA configuration only when its master switch is enabled."""
    from flashinfer.fused_moe.da_config import (
        TrtllmDaConfig,
        is_trtllm_da_enabled,
    )

    if not is_trtllm_da_enabled():
        return None
    return TrtllmDaConfig.from_environment()


def get_trtllm_moe_sm100_module():
    device = torch.device("cuda", torch.cuda.current_device())
    enable_rubin = get_compute_capability(device) == (10, 7)
    return _get_trtllm_moe_sm100_module_impl(enable_rubin)


@functools.cache
def _get_trtllm_moe_sm100_module_impl(enable_rubin: bool):
    module = gen_trtllm_gen_fused_moe_sm100_module(enable_rubin=enable_rubin)
    moe_op = module.build_and_load()
    setup_cubin_loader(str(module.get_library_path()))

    class MoERunner(TunableRunner):
        # Cache valid tactics to reduce the overhead of re-querying the kernel.
        # TODO(siyuan): directly cache the runners
        valid_tactics_dict: dict = dict()

        def __init__(
            self,
            top_k: int,
            num_local_experts: int,
            dtype_act: DtypeTrtllmGen,
            dtype_weights: DtypeTrtllmGen,
            fp8_quantization_type: Fp8QuantizationType,
            hidden_size: int,
            intermediate_size: int,
            activation_type: int = ActivationType.Swiglu.value,
            use_shuffled_weight: bool = False,
            weight_layout: int = WeightLayout.MajorK,
            use_packed_weights: bool = False,
            use_per_token_scaling: bool = False,
            num_experts: Optional[int] = None,
            num_fused_shared_experts: int = 0,
        ):
            self.num_local_experts = num_local_experts
            self.top_k = top_k
            # Fused shared experts widen the per-token expert count and the local
            # expert count seen by the kernel. Keep top_k / num_local_experts raw
            # (forward() adds the shared experts via the C++ op), but record the
            # fused count so valid-tactic enumeration matches prepare_moe().
            self.num_fused_shared_experts = num_fused_shared_experts or 0
            self.dtype_act = dtype_act
            self.dtype_weights = dtype_weights
            self.fp8_quantization_type = fp8_quantization_type
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.activation_type = ActivationType(activation_type)
            self.use_shuffled_weight = use_shuffled_weight
            self.weight_layout = WeightLayout(weight_layout)
            self.use_packed_weights = use_packed_weights
            self.use_per_token_scaling = use_per_token_scaling
            self.num_experts = (
                num_experts if num_experts is not None else num_local_experts
            )

        def _make_tuning_config(
            self,
            moe_inputs: "MoeRunnerInputs",
            tune_max_num_tokens: int = 8192,
            routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
            **kwargs,
        ) -> TuningConfig:
            """Build a TuningConfig for this runner instance.

            Args:
                moe_inputs: Input parameters for this call.
                tune_max_num_tokens: Upper bound for the num_tokens tuning buckets.
                routing_input_mode: Routing representation used by the launcher.
                **kwargs: Extra TuningConfig kwargs (e.g. use_cold_l2_cache).
            """

            spec = {
                "output": autotuner_initializer_empty,
                "hidden_states": autotuner_initializer_randn,
            }
            if moe_inputs.routing_logits is not None:
                spec["routing_logits"] = autotuner_initializer_rand
            if moe_inputs.topk_ids is not None:
                spec["topk_ids"] = _moe_topk_ids_init(
                    self.num_experts,
                    packed=routing_input_mode != RoutingInputMode.UnpackedPrecomputed,
                )
            if moe_inputs.expert_weights is not None:
                spec["expert_weights"] = autotuner_initializer_ones
            if moe_inputs.hidden_states_scale is not None:
                spec["hidden_states_scale"] = autotuner_initializer_ones
            if moe_inputs.gemm1_lora_delta is not None:
                spec["gemm1_lora_delta"] = autotuner_initializer_zeros
            if moe_inputs.per_token_scale is not None:
                spec["per_token_scale"] = autotuner_initializer_ones

            sorted_inputs = sorted(
                (MoeRunnerInputs.idx(name), name, init) for name, init in spec.items()
            )
            input_idx = tuple(i for i, _, _ in sorted_inputs)

            num_tokens = moe_inputs.hidden_states.shape[0]

            def _dynamic_dim(name: str) -> int:
                if name == "hidden_states_scale":
                    # DeepSeekFp8 uses [hidden_size//128, num_tokens];
                    # all others (MxFp8, fp4, …) use [num_tokens, ...].
                    t = moe_inputs.hidden_states_scale
                    if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                        assert t.shape == (self.hidden_size // 128, num_tokens), (
                            f"hidden_states_scale shape {tuple(t.shape)} does not match "
                            f"expected DeepSeekFp8 layout "
                            f"(hidden_size//128={self.hidden_size // 128}, num_tokens={num_tokens})"
                        )
                        return 1
                    assert t.shape[0] == num_tokens, (
                        f"hidden_states_scale shape {tuple(t.shape)} does not match "
                        f"expected layout (num_tokens={num_tokens}, ...)"
                    )
                    return 0
                return MoeRunnerInputs._DYNAMIC_DIM[name]

            dim_idx = tuple(_dynamic_dim(name) for _, name, _ in sorted_inputs)
            tensor_initializers = tuple((idx, init) for idx, _, init in sorted_inputs)
            value_aware_names = {"topk_ids", "expert_weights"}
            kwargs.setdefault(
                "value_aware_input_indices",
                tuple(
                    idx for idx, name, _ in sorted_inputs if name in value_aware_names
                ),
            )
            kwargs.setdefault("profile_arena_input_indices", input_idx)

            return TuningConfig(
                dynamic_tensor_specs=(
                    DynamicTensorSpec(
                        input_idx,
                        dim_idx,
                        get_hybrid_num_tokens_buckets(tune_max_num_tokens, 1),
                        make_hybrid_bucket_mapper(tune_max_num_tokens),
                    ),
                ),
                tensor_initializers=tensor_initializers,
                **kwargs,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            moe_inputs = MoeRunnerInputs.from_list(inputs)
            num_tokens = moe_inputs.hidden_states.shape[0]

            has_gemm1_lora_delta = moe_inputs.gemm1_lora_delta is not None

            # Enumerate valid tactics for the fused (routed + shared) expert
            # dimensions so they match what prepare_moe() validates against at
            # runtime (effectiveTopK / effectiveLocalExperts). nfse defaults to 0,
            # so non-shared-expert paths are unaffected. Including nfse in the key
            # also prevents cache collisions across different shared-expert counts.
            nfse = self.num_fused_shared_experts
            instance_key = (
                self.dtype_act,
                self.dtype_weights,
                self.fp8_quantization_type,
                self.top_k + nfse,
                self.hidden_size,
                self.intermediate_size,
                self.num_local_experts + nfse,
                self.activation_type,
                self.use_shuffled_weight,
                self.weight_layout,
                self.use_per_token_scaling,
                num_tokens,
                has_gemm1_lora_delta,
            )
            if instance_key not in MoERunner.valid_tactics_dict:
                try:
                    valid_tactics = moe_op.trtllm_get_valid_moe_configs(*instance_key)
                except Exception as e:
                    logger.debug(
                        f"[Autotuner]: Failed to get valid tactics for {instance_key}. Error occurred: {e}"
                    )
                    return []
                MoERunner.valid_tactics_dict[instance_key] = valid_tactics
            return MoERunner.valid_tactics_dict[instance_key]

        def get_factorized_tactic_space(
            self,
            inputs: List[torch.Tensor],
        ):
            """Return C++-declared legal FC1/FC2 factors and tile-local anchors."""
            from flashinfer.fused_moe.da_tuner import (
                FactorizedTactic,
                FactorizedTacticSpace,
            )

            moe_inputs = MoeRunnerInputs.from_list(inputs)
            rows = moe_op.trtllm_get_valid_moe_factorizations(
                self.dtype_act,
                self.dtype_weights,
                self.fp8_quantization_type,
                self.top_k + self.num_fused_shared_experts,
                self.hidden_size,
                self.intermediate_size,
                self.num_local_experts + self.num_fused_shared_experts,
                self.activation_type,
                self.use_shuffled_weight,
                self.weight_layout,
                self.use_per_token_scaling,
                moe_inputs.hidden_states.shape[0],
                moe_inputs.gemm1_lora_delta is not None,
            )
            tactics = []
            anchors = {}
            for tile_n, config, fc1, fc2, is_anchor in rows:
                identity = (int(tile_n), int(config))
                tactics.append(
                    FactorizedTactic(
                        tactic=identity,
                        tile_n=int(tile_n),
                        fc1=int(fc1),
                        fc2=int(fc2),
                    )
                )
                if is_anchor:
                    anchors[int(tile_n)] = identity
            return FactorizedTacticSpace(tactics, anchors)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: Any = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            moe_inputs = MoeRunnerInputs.from_list(inputs)
            output = moe_inputs.output
            routing_logits = moe_inputs.routing_logits
            topk_ids = moe_inputs.topk_ids
            expert_weights = moe_inputs.expert_weights
            topk_weights = expert_weights
            hidden_states = moe_inputs.hidden_states
            # The generic helper identifies TRTLLM dtypes whose ABI normally
            # consumes an auxiliary scale tensor (FP4 and MX formats). Plain
            # E4m3 returns false, but DeepSeek block-FP8 is an exception: it
            # requires the real per-1x128-block scales from the activation pack.
            hidden_states_scale = (
                moe_inputs.hidden_states_scale
                if (
                    trtllm_gen_dtype_has_scale(self.dtype_act)
                    or self.fp8_quantization_type
                    in (
                        Fp8QuantizationType.DeepSeekFp8,
                        Fp8QuantizationType.PerChannelFp8,
                    )
                )
                else None
            )
            da_routing_metadata = kwargs.get("da_routing_metadata", ())
            da_body_workspace = kwargs.get("da_body_workspace", ())
            prepare_da_body = do_preparation and bool(da_routing_metadata)

            num_tokens = hidden_states.shape[0]
            # sanity checks to ensure that dynamic tensors have the correct shapes
            assert output.shape[0] == num_tokens, (
                "output's first dimension must be batch size."
            )
            if routing_logits is not None:
                assert routing_logits.shape[0] == num_tokens, (
                    "routing_logits's first dimension must be batch size."
                )
            # topk_ids/expert_weights can be empty(0) when routing_logits is provided,
            # or real tensors when pre-computed routing is used.
            if topk_ids is not None and topk_ids.numel() > 0:
                assert topk_ids.shape[0] == num_tokens, (
                    "topk_ids's first dimension must be batch size."
                )
            if expert_weights is not None and expert_weights.numel() > 0:
                assert expert_weights.shape[0] == num_tokens, (
                    "expert_weights's first dimension must be batch size."
                )
            assert hidden_states.shape[0] == num_tokens, (
                "hidden_states's first dimension must be batch size."
            )
            if hidden_states_scale is not None:
                assert hidden_states_scale.dim() == 2, (
                    "hidden_states_scale must be a 2D tensor"
                )
                if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                    assert hidden_states_scale.shape[1] == num_tokens, (
                        f"DeepSeekFp8 hidden_states_scale shape {tuple(hidden_states_scale.shape)} "
                        f"expects num_tokens={num_tokens} at dim 1"
                    )
                else:
                    assert hidden_states_scale.shape[0] == num_tokens, (
                        f"hidden_states_scale shape {tuple(hidden_states_scale.shape)} "
                        f"expects num_tokens={num_tokens} at dim 0"
                    )
            # Choose the appropriate operation based on data types
            if self.dtype_weights == DtypeTrtllmGen.Bfloat16:
                # BF16 operations
                result = moe_op.trtllm_bf16_moe(
                    kwargs["routing_input_mode"],
                    routing_logits,
                    kwargs["routing_bias"],
                    topk_ids,
                    expert_weights,
                    hidden_states,
                    kwargs["gemm1_weights"],
                    kwargs["gemm2_weights"],
                    moe_inputs.gemm1_lora_delta,
                    kwargs.get("gemm1_alpha"),
                    kwargs.get("gemm1_beta"),
                    kwargs.get("gemm1_clamp_limit"),
                    output,
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["use_shuffled_weight"],
                    kwargs["weight_layout"],
                    kwargs["do_finalize"],
                    kwargs["enable_pdl"],
                    [-1, -1] if tactic == -1 else tactic,
                    self.activation_type,
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                    list(da_routing_metadata),
                    list(da_body_workspace),
                    prepare_da_body,
                )
                if prepare_da_body or da_routing_metadata:
                    return list(result)
            elif (
                self.dtype_act == DtypeTrtllmGen.E4m3
                and self.dtype_weights == DtypeTrtllmGen.E4m3
            ) or (
                self.dtype_act == DtypeTrtllmGen.MxE4m3
                and self.dtype_weights == DtypeTrtllmGen.MxE4m3
            ):
                # FP8 operations
                if (
                    self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
                    or self.fp8_quantization_type == Fp8QuantizationType.MxFp8
                ):
                    # FP8 block scale
                    result = moe_op.trtllm_fp8_block_scale_moe(
                        kwargs["routing_input_mode"],
                        routing_logits,
                        topk_ids,
                        topk_weights,
                        kwargs["routing_bias"],
                        hidden_states,
                        hidden_states_scale,
                        kwargs["gemm1_weights"],
                        kwargs["gemm1_weights_scale"],
                        moe_inputs.gemm1_lora_delta,
                        kwargs.get("gemm1_alpha"),
                        kwargs.get("gemm1_beta"),
                        kwargs.get("gemm1_clamp_limit"),
                        kwargs["gemm2_weights"],
                        kwargs["gemm2_weights_scale"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs.get("num_fused_shared_experts", 0),
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["routing_method_type"],
                        kwargs["use_shuffled_weight"],
                        kwargs["weight_layout"],
                        kwargs["do_finalize"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                        self.fp8_quantization_type,
                        self.activation_type,
                        kwargs.get("norm_topk_prob", True),
                        kwargs.get("routing_replay_out"),
                        list(da_routing_metadata),
                        list(da_body_workspace),
                        prepare_da_body,
                    )
                elif self.fp8_quantization_type == Fp8QuantizationType.PerChannelFp8:
                    # FP8 per-token activation and per-channel weight scales.
                    result = moe_op.trtllm_fp8_per_channel_scale_moe(
                        routing_logits,
                        topk_ids,
                        topk_weights,
                        kwargs["routing_bias"],
                        hidden_states,
                        hidden_states_scale,
                        kwargs["gemm1_weights"],
                        kwargs["gemm1_per_channel_weight_scale"],
                        kwargs["output1_scale_scalar"],
                        kwargs["output1_scale_gate_scalar"],
                        kwargs["gemm2_weights"],
                        kwargs["gemm2_per_channel_weight_scale"],
                        kwargs["output2_scale_scalar"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["use_routing_scales_on_input"],
                        kwargs["routing_method_type"],
                        kwargs["do_finalize"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                        self.activation_type,
                        kwargs.get("norm_topk_prob", True),
                    )
                elif routing_logits is None:
                    # FP8 per tensor scale, pre-computed routing.
                    result = moe_op.trtllm_fp8_per_tensor_scale_routed_moe(
                        kwargs["routing_input_mode"],
                        topk_ids,
                        topk_weights,
                        kwargs["routing_bias"],
                        hidden_states,
                        kwargs["gemm1_weights"],
                        kwargs["output1_scales_scalar"],
                        kwargs["output1_scales_gate_scalar"],
                        kwargs["gemm2_weights"],
                        kwargs["output2_scales_scalar"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["use_routing_scales_on_input"],
                        kwargs["routing_method_type"],
                        kwargs["do_finalize"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                        self.activation_type,
                        kwargs.get("norm_topk_prob", True),
                        kwargs.get("routing_replay_out"),
                        list(da_routing_metadata),
                        list(da_body_workspace),
                        prepare_da_body,
                    )
                else:
                    # FP8 per tensor scale
                    result = moe_op.trtllm_fp8_per_tensor_scale_moe(
                        routing_logits,
                        kwargs["routing_bias"],
                        hidden_states,
                        kwargs["gemm1_weights"],
                        kwargs["output1_scales_scalar"],
                        kwargs["output1_scales_gate_scalar"],
                        kwargs["gemm2_weights"],
                        kwargs["output2_scales_scalar"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["use_routing_scales_on_input"],
                        kwargs["routing_method_type"],
                        kwargs["do_finalize"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                        self.activation_type,
                        kwargs.get("norm_topk_prob", True),
                        kwargs.get("routing_replay_out"),
                        list(da_routing_metadata),
                        list(da_body_workspace),
                        prepare_da_body,
                    )
                    # Unlike the routed per-tensor entry point, the FromLogits
                    # ABI does not accept the caller's expert_weights buffer;
                    # the launcher owns and returns that tensor.
                    expert_weights = None
                if prepare_da_body or da_routing_metadata:
                    return list(result)
            elif (
                self.dtype_act == DtypeTrtllmGen.Bfloat16
                and self.dtype_weights == DtypeTrtllmGen.MxInt4
            ):
                result = moe_op.trtllm_mxint4_block_scale_moe(
                    routing_logits,
                    kwargs["routing_bias"],
                    topk_ids,
                    expert_weights,
                    hidden_states,
                    kwargs["gemm1_weights"],
                    kwargs["gemm1_weights_scale"],
                    kwargs["gemm1_alpha"],
                    kwargs["gemm1_beta"],
                    kwargs["gemm1_clamp_limit"],
                    moe_inputs.gemm1_lora_delta,
                    kwargs["gemm2_weights"],
                    kwargs["gemm2_weights_scale"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["do_finalize"],
                    kwargs["enable_pdl"],
                    output,
                    [-1, -1] if tactic == -1 else tactic,
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                    list(da_routing_metadata),
                    list(da_body_workspace),
                    prepare_da_body,
                )
                if prepare_da_body or da_routing_metadata:
                    return list(result)
            else:
                result = moe_op.trtllm_fp4_block_scale_moe(
                    kwargs.get("routing_input_mode", RoutingInputMode.FromLogits),
                    routing_logits,
                    topk_ids,
                    topk_weights,
                    kwargs["routing_bias"],
                    hidden_states,
                    hidden_states_scale,  # hidden_states_scale
                    kwargs["gemm1_weights"],
                    kwargs["gemm1_weights_scale"],
                    kwargs["gemm1_bias"],
                    moe_inputs.gemm1_lora_delta,
                    kwargs["gemm1_alpha"],
                    kwargs["gemm1_beta"],
                    kwargs["gemm1_clamp_limit"],
                    kwargs["gemm2_weights"],
                    kwargs["gemm2_weights_scale"],
                    kwargs["gemm2_bias"],
                    kwargs["output1_scale_scalar"],
                    kwargs["output1_scale_gate_scalar"],
                    kwargs["output2_scale_scalar"],
                    kwargs["per_token_scale"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs.get("num_fused_shared_experts", 0),
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["do_finalize"],
                    kwargs["enable_pdl"],
                    self.activation_type,
                    output,
                    [-1, -1] if tactic == -1 else tactic,
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                    list(da_routing_metadata),
                    list(da_body_workspace),
                    prepare_da_body,
                )
                if prepare_da_body or da_routing_metadata:
                    return list(result)

            return _unpack_trtllm_moe_output(
                result,
                output,
                kwargs["do_finalize"],
                moe_inputs.gemm1_lora_delta,
                expert_weights,
            )

    class DABodyRunner:
        """Compose one ordinary MoERunner with prepared-metadata body execution."""

        def __init__(self, moe_runner: MoERunner) -> None:
            """Retain the dtype-agnostic ordinary runner used by every typed adapter."""
            # Ordinary full-operation runner whose dtype branch preserves the native ABI.
            self._moe_runner = moe_runner

        @property
        def moe_runner(self) -> MoERunner:
            """Return the composed ordinary runner."""
            return self._moe_runner

        def prepare_body(
            self,
            inputs: List[torch.Tensor],
            body,
            routing_metadata: TrtllmMoERoutingMetadataSlot,
            **kwargs,
        ) -> tuple[torch.Tensor, ...]:
            """Allocate one body's graph-stable buffers outside CUDA Graph capture."""
            if body.tile_n != routing_metadata.tile_n:
                raise ValueError("DA body and routing metadata tile_n must match")
            prepared = self._moe_runner.forward(
                inputs,
                tactic=[body.tile_n, body.tactic],
                do_preparation=True,
                da_routing_metadata=routing_metadata.tensors(),
                **kwargs,
            )
            return tuple(prepared)

        def prepare_max_body_workspace(
            self,
            inputs: List[torch.Tensor],
            bodies: Sequence[Any],
            routing_metadata_by_tile: Mapping[int, TrtllmMoERoutingMetadataSlot],
            **kwargs,
        ) -> tuple[torch.Tensor, ...]:
            """Retain one field-wise maximum typed workspace across candidate bodies."""
            if not bodies:
                raise ValueError(
                    "A shared DA body workspace requires at least one body"
                )

            # Materialize each tactic's exact native requirements, then retain only the largest
            # allocation for every field in this dtype-specific ABI. Mutually exclusive bodies may
            # safely bind typed views to the resulting common pointer set.
            candidates = [
                self.prepare_body(
                    inputs,
                    body,
                    routing_metadata_by_tile[body.tile_n],
                    **kwargs,
                )
                for body in bodies
            ]
            field_count = len(candidates[0])
            if any(len(candidate) != field_count for candidate in candidates):
                raise RuntimeError(
                    "One dtype-specific DA plan exposed multiple body ABIs"
                )

            # Dtype and device are ABI properties; choose the largest byte capacity per field while
            # allowing tactic-dependent logical shapes and scratch sizes.
            maximum_fields = []
            for field_index in range(field_count):
                fields = [candidate[field_index] for candidate in candidates]
                first = fields[0]
                if any(
                    field.dtype != first.dtype or field.device != first.device
                    for field in fields[1:]
                ):
                    raise RuntimeError(
                        "DA body workspace field ABI changed across tactics"
                    )
                maximum_fields.append(max(fields, key=lambda field: field.numel()))
            return tuple(maximum_fields)

        def forward_from_metadata(
            self,
            inputs: List[torch.Tensor],
            body,
            routing_metadata: TrtllmMoERoutingMetadataSlot,
            body_workspace: Sequence[torch.Tensor],
            **kwargs,
        ) -> None:
            """Launch one fixed body without routing, allocation, or DA policy lookup."""
            if body.tile_n != routing_metadata.tile_n:
                raise ValueError("DA body and routing metadata tile_n must match")
            self._moe_runner.forward(
                inputs,
                tactic=[body.tile_n, body.tactic],
                da_routing_metadata=routing_metadata.tensors(),
                da_body_workspace=tuple(body_workspace),
                **kwargs,
            )

    class DAProfileRunner(TunableRunner):
        """Measure native replay preamble and typed body as one complete DA operation."""

        def __init__(
            self,
            body_runner: DABodyRunner,
            packed_router_scratch: torch.Tensor,
            canonical_expert_weights: torch.Tensor,
        ) -> None:
            """Compose profiling around one dtype-agnostic prepared-body runner."""
            # Typed body capability shared with production capture.
            self._body_runner = body_runner
            # Conventional placeholder satisfying dtype-specific launcher validation only.
            self._packed_router_scratch = packed_router_scratch
            # Conventional placeholder paired with packed entries at the body FFI boundary.
            self._canonical_expert_weights = canonical_expert_weights
            # Prepared workspaces keyed by tactic and exact cold-L2 lane bindings.
            self._workspace_cache: dict[tuple[Any, ...], TrtllmDaProfileWorkspace] = {}

        def get_valid_tactics(
            self, inputs: list[torch.Tensor], profile: OptimizationProfile
        ) -> list[Any]:
            """Delegate tactic enumeration to the composed ordinary runner."""
            return self._body_runner.moe_runner.get_valid_tactics(inputs, profile)

        def get_factorized_tactic_space(self, inputs: list[torch.Tensor]):
            """Delegate legal factorization enumeration to the ordinary runner."""
            return self._body_runner.moe_runner.get_factorized_tactic_space(inputs)

        @staticmethod
        def _binding_key(inputs: list[Any], tactic: tuple[int, int]) -> tuple[Any, ...]:
            """Return the immutable cache key for one tactic and profiling lane."""
            tensor_bindings = tuple(
                (value.data_ptr(), tuple(value.shape))
                for value in inputs
                if isinstance(value, torch.Tensor)
            )
            return tactic, tensor_bindings

        def _body_inputs(self, inputs: list[Any]) -> list[Any]:
            """Substitute conventional placeholders that prepared bodies never consume."""
            body_inputs = list(inputs)
            body_inputs[MoeRunnerInputs.idx("routing_logits")] = None
            body_inputs[MoeRunnerInputs.idx("topk_ids")] = self._packed_router_scratch
            body_inputs[MoeRunnerInputs.idx("expert_weights")] = (
                self._canonical_expert_weights
            )
            return body_inputs

        def _prepare_workspace(
            self, inputs: list[Any], tactic: tuple[int, int], **kwargs
        ) -> TrtllmDaProfileWorkspace:
            """Prepare one lane before CUDA Graph timing and retain its exact pointers."""
            key = self._binding_key(inputs, tactic)
            cached = self._workspace_cache.get(key)
            if cached is not None:
                return cached
            moe_inputs = MoeRunnerInputs.from_list(inputs)
            if moe_inputs.topk_ids is None or moe_inputs.expert_weights is None:
                raise RuntimeError(
                    "DA profiling requires native IDs and routing weights"
                )
            body = DABody(tile_n=tactic[0], tactic=tactic[1])
            metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
                moe_inputs.topk_ids,
                num_experts=kwargs["num_experts"],
                top_k=self._body_runner.moe_runner.top_k,
                local_expert_offset=kwargs["local_expert_offset"],
                num_local_experts=kwargs["local_num_experts"],
                tile_ns=(body.tile_n,),
                routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
                topk_weights=moe_inputs.expert_weights,
            )
            body_kwargs = dict(kwargs)
            body_kwargs["routing_input_mode"] = RoutingInputMode.PackedPrecomputed
            tensors = self._body_runner.prepare_body(
                self._body_inputs(inputs), body, metadata.slots[0], **body_kwargs
            )
            workspace = TrtllmDaProfileWorkspace(body, metadata, tuple(tensors))
            self._workspace_cache[key] = workspace
            return workspace

        def prepare_batches(
            self, input_batches: Sequence[list[Any]], tactic: Sequence[int], **kwargs
        ) -> None:
            """Prepare every cold-L2 lane before the autotuner enters graph capture."""
            identity = tuple(int(value) for value in tactic)
            if len(identity) != 2 or identity[1] < 0:
                raise RuntimeError(
                    "DA profiling requires a concrete (tile_n, tactic) pair"
                )
            for inputs in input_batches:
                self._prepare_workspace(inputs, identity, **kwargs)

        def forward(
            self,
            inputs: list[Any],
            tactic: Any = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> None:
            """Launch native-ID preamble and one exact typed body for full-op timing."""
            identity = tuple(int(value) for value in tactic)
            if len(identity) != 2 or identity[1] < 0:
                raise RuntimeError(
                    "DA profiling requires a concrete (tile_n, tactic) pair"
                )
            workspace = self._prepare_workspace(inputs, identity, **kwargs)
            if do_preparation:
                return
            moe_inputs = MoeRunnerInputs.from_list(inputs)
            if moe_inputs.topk_ids is None or moe_inputs.expert_weights is None:
                raise RuntimeError(
                    "DA profiling requires native IDs and routing weights"
                )
            populate_trtllm_moe_routing_metadata_(
                workspace.routing_metadata,
                moe_inputs.topk_ids,
                moe_inputs.expert_weights,
            )
            body_kwargs = dict(kwargs)
            body_kwargs["routing_input_mode"] = RoutingInputMode.PackedPrecomputed
            self._body_runner.forward_from_metadata(
                self._body_inputs(inputs),
                workspace.body,
                workspace.routing_metadata.slots[0],
                workspace.tensors,
                **body_kwargs,
            )

    @register_custom_op(
        "flashinfer::trtllm_bf16_moe",
        mutates_args=("routing_replay_out",),
    )
    def trtllm_bf16_moe_op(
        routing_input_mode: int,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm1_lora_delta: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
        gemm1_alpha: Optional[torch.Tensor] = None,
        gemm1_beta: Optional[torch.Tensor] = None,
        gemm1_clamp_limit: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        assert routing_logits is not None or topk_ids is not None, (
            "either routing_logits or topk_ids must be provided"
        )
        _validate_bf16_gemm1_activation_params(
            activation_type,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            local_num_experts,
            hidden_states.device,
        )
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)

        # Use AutoTuner to select the best tactic
        tuner = AutoTuner.get()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        if output is None:
            output = _alloc_trtllm_moe_output(
                num_tokens, hidden_size, do_finalize, hidden_states.device
            )
        elif do_finalize:
            check_shape_dtype_device(
                output,
                (num_tokens, hidden_size),
                torch.bfloat16,
                hidden_states.device,
                "output",
            )
        if routing_logits is not None:
            # When routing_logits is provided, we must pass topk_ids/expert_weights with no allocation
            topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
            expert_weights = torch.empty(
                0, dtype=torch.bfloat16, device=hidden_states.device
            )
        else:
            # When routing_logits is provided, we either have topk_ids/expert_weights,
            # packed into a single tensor as topk_id
            # or have them individually as topk_ids and expert_weights respectively
            topk_ids = topk_ids
            expert_weights = (
                expert_weights
                if expert_weights is not None
                else torch.empty(0, dtype=torch.bfloat16, device=hidden_states.device)
            )

        dtype_act = DtypeTrtllmGen.Bfloat16
        dtype_weights = DtypeTrtllmGen.Bfloat16

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=weight_layout,
            use_shuffled_weight=use_shuffled_weight,
            activation_type=activation_type,
            num_experts=num_experts,
        )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_lora_delta=gemm1_lora_delta,
            per_token_scale=None,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            routing_input_mode=RoutingInputMode(routing_input_mode),
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

        runner_kwargs = {
            "routing_bias": routing_bias,
            "routing_input_mode": routing_input_mode,
            "gemm1_weights": gemm1_weights,
            "gemm2_weights": gemm2_weights,
            "gemm1_alpha": gemm1_alpha,
            "gemm1_beta": gemm1_beta,
            "gemm1_clamp_limit": gemm1_clamp_limit,
            "num_experts": num_experts,
            "n_group": n_group,
            "topk_group": topk_group,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": local_num_experts,
            "routed_scaling_factor": routed_scaling_factor,
            "routing_method_type": routing_method_type,
            "use_shuffled_weight": use_shuffled_weight,
            "weight_layout": weight_layout,
            "do_finalize": do_finalize,
            "enable_pdl": enable_pdl,
            "activation_type": activation_type,
            "norm_topk_prob": norm_topk_prob,
            "routing_replay_out": routing_replay_out,
        }
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_bf16_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            **runner_kwargs,
        )

        def run_selected_tactic(tactic: Any) -> List[torch.Tensor]:
            """Launch the unchanged monolithic BF16 operation with one tactic."""
            # Preserve the exact ordinary BF16 FFI ABI for baseline, eager, and singleton paths.
            intermediate_output = moe_op.trtllm_bf16_moe(
                routing_input_mode,
                routing_logits,
                routing_bias,
                topk_ids,
                expert_weights,
                hidden_states,
                gemm1_weights,
                gemm2_weights,
                gemm1_lora_delta,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                output,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                routing_method_type,
                use_shuffled_weight,
                weight_layout,
                do_finalize,
                enable_pdl,
                [-1, -1] if tactic == -1 else tactic,
                activation_type,
                norm_topk_prob,
                routing_replay_out,
                [],
                [],
                False,
            )
            # Reconstruct the established public result independently of DA plan mode.
            return _unpack_trtllm_moe_output(
                intermediate_output,
                output,
                do_finalize,
                gemm1_lora_delta,
                expert_weights,
            )

        routing_mode = RoutingInputMode(routing_input_mode)
        # When do_finalize=False, the FC2 output format is determined on device based on runtime
        # expert distribution. Therefore it is not eligible for DA until we can canonicalize
        # output format. A LoRA call returns body-specific intermediate buffers whose pointers
        # cannot vary behind one public graph output, so it retains the ordinary multi-output ABI.
        # TODO(da-moe): Prepare graph-stable LoRA auxiliary outputs before admitting this ABI.
        da_eligible = (
            routing_mode
            in (
                RoutingInputMode.PackedPrecomputed,
                RoutingInputMode.UnpackedPrecomputed,
            )
            and topk_ids is not None
            and do_finalize
            and gemm1_lora_delta is None
            and 0 < num_experts <= DA_MAX_EXPERTS
        )
        if not da_eligible:
            return run_selected_tactic(tactic)

        from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic

        da_config = _enabled_trtllm_da_config()
        if da_config is None:
            return run_selected_tactic(tactic)
        return run_dist_aware_tactic(
            custom_op="flashinfer::trtllm_bf16_moe",
            tuner=tuner,
            config=da_config,
            runner=moe_runner,
            runtime=TrtllmDaRuntime(moe_runner),
            tuning_config=tuning_config,
            inputs=moe_inputs.to_list(),
            runner_kwargs=runner_kwargs,
            baseline_tactic=tactic,
            routing_input_mode=routing_input_mode,
            routing_id_index=MoeRunnerInputs.idx("topk_ids"),
            routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=local_num_experts,
            top_k=top_k,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            run_fixed_tactic=run_selected_tactic,
            finish_switch=lambda: _unpack_trtllm_moe_output(
                [], output, do_finalize, gemm1_lora_delta, expert_weights
            ),
        )

    @register_fake_op("flashinfer::trtllm_bf16_moe")
    def _fake_trtllm_bf16_moe(
        routing_input_mode: int,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm1_lora_delta: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
        gemm1_alpha: Optional[torch.Tensor] = None,
        gemm1_beta: Optional[torch.Tensor] = None,
        gemm1_clamp_limit: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        # Acknowledge the declared mutation-only argument without reading device data in fake mode.
        _ = routing_replay_out
        return _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=hidden_states.shape[1],
            intermediate_size=intermediate_size,
            top_k=top_k,
            do_finalize=do_finalize,
            output=output,
            expert_weights=expert_weights,
            gemm1_lora_delta=gemm1_lora_delta,
        )

    @register_custom_op(
        "flashinfer::trtllm_fp8_per_tensor_scale_moe",
        mutates_args=("routing_replay_out",),
    )
    def trtllm_fp8_per_tensor_scale_moe_op(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        # Use AutoTuner to select the best tactic
        tuner = AutoTuner.get()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        if output is None:
            output = _alloc_trtllm_moe_output(
                num_tokens, hidden_size, do_finalize, hidden_states.device
            )
        elif do_finalize:
            check_shape_dtype_device(
                output,
                (num_tokens, hidden_size),
                torch.bfloat16,
                hidden_states.device,
                "output",
            )
        topk_ids = torch.empty(
            num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
        )
        topk_weights = torch.empty(
            num_tokens, top_k, dtype=torch.bfloat16, device=hidden_states.device
        )

        dtype_act = DtypeTrtllmGen.E4m3  # FP8 activation
        dtype_weights = DtypeTrtllmGen.E4m3  # FP8 weights

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,  # per_tensor mode
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
            activation_type=activation_type,
            use_per_token_scaling=use_routing_scales_on_input,
            num_experts=num_experts,
        )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=topk_weights,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

        runner_kwargs = {
            "routing_input_mode": RoutingInputMode.FromLogits,
            "routing_bias": routing_bias,
            "gemm1_weights": gemm1_weights,
            "output1_scales_scalar": output1_scales_scalar,
            "output1_scales_gate_scalar": output1_scales_gate_scalar,
            "gemm2_weights": gemm2_weights,
            "output2_scales_scalar": output2_scales_scalar,
            "num_experts": num_experts,
            "n_group": n_group,
            "topk_group": topk_group,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": local_num_experts,
            "routed_scaling_factor": routed_scaling_factor,
            "use_routing_scales_on_input": use_routing_scales_on_input,
            "routing_method_type": routing_method_type,
            "do_finalize": do_finalize,
            "enable_pdl": enable_pdl,
            "activation_type": activation_type,
            "norm_topk_prob": norm_topk_prob,
            "routing_replay_out": routing_replay_out,
        }
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp8_per_tensor_scale_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            **runner_kwargs,
        )

        def run_selected_tactic(tactic: Any) -> List[torch.Tensor]:
            """Launch the unchanged FromLogits FP8 per-tensor operation."""
            # Preserve the exact logits-routing FFI ABI for every fixed-tactic dispatch mode.
            intermediate_output = moe_op.trtllm_fp8_per_tensor_scale_moe(
                routing_logits,
                routing_bias,
                hidden_states,
                gemm1_weights,
                output1_scales_scalar,
                output1_scales_gate_scalar,
                gemm2_weights,
                output2_scales_scalar,
                output,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                use_routing_scales_on_input,
                routing_method_type,
                do_finalize,
                enable_pdl,
                [-1, -1] if tactic == -1 else tactic,
                activation_type,
                norm_topk_prob,
                routing_replay_out,
                [],
                [],
                False,
            )
            # Convert the native result back to the established public output contract.
            return _unpack_trtllm_moe_output(
                intermediate_output, output, do_finalize, None
            )

        # When do_finalize=False, the FC2 output format is determined on device based on runtime
        # expert distribution. Therefore it is not eligible for DA until we can canonicalize
        # output format. The exact launcher owns dtype, optional-operand, and top-k validation.
        da_eligible = do_finalize and 0 < num_experts <= DA_MAX_EXPERTS
        if not da_eligible:
            return run_selected_tactic(tactic)

        from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic

        da_config = _enabled_trtllm_da_config()
        if da_config is None:
            return run_selected_tactic(tactic)
        return run_dist_aware_tactic(
            custom_op="flashinfer::trtllm_fp8_per_tensor_scale_moe",
            tuner=tuner,
            config=da_config,
            runner=moe_runner,
            runtime=TrtllmDaRuntime(moe_runner),
            tuning_config=tuning_config,
            inputs=moe_inputs.to_list(),
            runner_kwargs=runner_kwargs,
            baseline_tactic=tactic,
            routing_input_mode=RoutingInputMode.FromLogits,
            routing_id_index=MoeRunnerInputs.idx("routing_logits"),
            routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
            routing_precomputed_id_index=MoeRunnerInputs.idx("topk_ids"),
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=local_num_experts,
            top_k=top_k,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            run_fixed_tactic=run_selected_tactic,
            finish_switch=lambda: [output],
        )

    @register_fake_op("flashinfer::trtllm_fp8_per_tensor_scale_moe")
    def _fake_trtllm_fp8_per_tensor_scale_moe(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ):
        # Acknowledge the declared mutation-only argument without reading device data in fake mode.
        _ = routing_replay_out
        return _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=hidden_states.shape[1],
            intermediate_size=intermediate_size,
            top_k=top_k,
            do_finalize=do_finalize,
            output=output,
        )

    @register_custom_op(
        "flashinfer::trtllm_fp8_per_tensor_scale_routed_moe",
        mutates_args=("routing_replay_out",),
    )
    def trtllm_fp8_per_tensor_scale_routed_moe_op(
        routing_input_mode: int,
        topk_ids: torch.Tensor,
        expert_weights: torch.Tensor | None,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        routing_replay_out: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        assert topk_ids.dtype == torch.int32, "topk_ids must be an int32 tensor."
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        # Use AutoTuner to select the best tactic
        tuner = AutoTuner.get()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        if output is None:
            output = _alloc_trtllm_moe_output(
                num_tokens, hidden_size, do_finalize, hidden_states.device
            )
        elif do_finalize:
            check_shape_dtype_device(
                output,
                (num_tokens, hidden_size),
                torch.bfloat16,
                hidden_states.device,
                "output",
            )
        if expert_weights is None:
            expert_weights = torch.empty(
                0, dtype=torch.bfloat16, device=hidden_states.device
            )

        dtype_act = DtypeTrtllmGen.E4m3  # FP8 activation
        dtype_weights = DtypeTrtllmGen.E4m3  # FP8 weights

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,  # per_tensor mode
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
            activation_type=activation_type,
            use_per_token_scaling=use_routing_scales_on_input,
            num_experts=num_experts,
        )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=None,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            routing_input_mode=RoutingInputMode(routing_input_mode),
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

        runner_kwargs = {
            "routing_input_mode": routing_input_mode,
            "routing_bias": routing_bias,
            "gemm1_weights": gemm1_weights,
            "output1_scales_scalar": output1_scales_scalar,
            "output1_scales_gate_scalar": output1_scales_gate_scalar,
            "gemm2_weights": gemm2_weights,
            "output2_scales_scalar": output2_scales_scalar,
            "num_experts": num_experts,
            "n_group": n_group,
            "topk_group": topk_group,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": local_num_experts,
            "routed_scaling_factor": routed_scaling_factor,
            "use_routing_scales_on_input": use_routing_scales_on_input,
            "routing_method_type": routing_method_type,
            "do_finalize": do_finalize,
            "enable_pdl": enable_pdl,
            "activation_type": activation_type,
            "norm_topk_prob": True,
            "routing_replay_out": routing_replay_out,
        }
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp8_per_tensor_scale_routed_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            **runner_kwargs,
        )

        def run_selected_tactic(tactic: Any) -> List[torch.Tensor]:
            """Launch the FP8 per-tensor operation with one fixed tactic."""
            intermediate_output = moe_op.trtllm_fp8_per_tensor_scale_routed_moe(
                routing_input_mode,
                topk_ids,
                expert_weights,
                routing_bias,
                hidden_states,
                gemm1_weights,
                output1_scales_scalar,
                output1_scales_gate_scalar,
                gemm2_weights,
                output2_scales_scalar,
                output,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                use_routing_scales_on_input,
                routing_method_type,
                do_finalize,
                enable_pdl,
                [-1, -1] if tactic == -1 else tactic,
                activation_type,
                True,
                routing_replay_out,
                [],
                [],
                False,
            )
            return _unpack_trtllm_moe_output(
                intermediate_output, output, do_finalize, None, expert_weights
            )

        routing_mode = RoutingInputMode(routing_input_mode)
        # When do_finalize=False, the FC2 output format is determined on device based on runtime
        # expert distribution. Therefore it is not eligible for DA until we can canonicalize
        # output format. FP32 Llama4 weights also need the ordinary launcher's BF16 token-scale
        # conversion, whose scratch buffer is not part of the prepared-body ABI.
        da_eligible = (
            routing_mode
            in (
                RoutingInputMode.PackedPrecomputed,
                RoutingInputMode.UnpackedPrecomputed,
            )
            and do_finalize
            and not (
                routing_mode is RoutingInputMode.UnpackedPrecomputed
                and use_routing_scales_on_input
                and expert_weights.dtype == torch.float32
            )
            and 0 < num_experts <= DA_MAX_EXPERTS
        )
        if not da_eligible:
            return run_selected_tactic(tactic)

        from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic

        da_config = _enabled_trtllm_da_config()
        if da_config is None:
            return run_selected_tactic(tactic)
        return run_dist_aware_tactic(
            custom_op="flashinfer::trtllm_fp8_per_tensor_scale_routed_moe",
            tuner=tuner,
            config=da_config,
            runner=moe_runner,
            runtime=TrtllmDaRuntime(moe_runner),
            tuning_config=tuning_config,
            inputs=moe_inputs.to_list(),
            runner_kwargs=runner_kwargs,
            baseline_tactic=tactic,
            routing_input_mode=routing_input_mode,
            routing_id_index=MoeRunnerInputs.idx("topk_ids"),
            routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=local_num_experts,
            top_k=top_k,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            run_fixed_tactic=run_selected_tactic,
            finish_switch=lambda: _unpack_trtllm_moe_output(
                [], output, do_finalize, None, expert_weights
            ),
        )

    @register_fake_op("flashinfer::trtllm_fp8_per_tensor_scale_routed_moe")
    def _fake_trtllm_fp8_per_tensor_scale_routed_moe(
        routing_input_mode: int,
        topk_ids: torch.Tensor,
        expert_weights: torch.Tensor | None,
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        output1_scales_scalar: torch.Tensor,
        output1_scales_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        output2_scales_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        routing_replay_out: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ):
        # Acknowledge the declared mutation-only argument without reading device data in fake mode.
        _ = routing_replay_out
        return _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=hidden_states.shape[1],
            intermediate_size=intermediate_size,
            top_k=top_k,
            do_finalize=do_finalize,
            output=output,
            expert_weights=expert_weights,
        )

    @register_custom_op(
        "flashinfer::trtllm_fp8_per_channel_scale_moe",
        mutates_args=(),
    )
    def trtllm_fp8_per_channel_scale_moe_op(
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_per_channel_weight_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_per_channel_weight_scale: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
    ) -> List[torch.Tensor]:
        if routing_logits is None:
            assert topk_ids is not None, (
                "either topk_ids or routing_logits must be provided."
            )
            assert topk_ids.dtype == torch.int32, "topk_ids must be an int32 tensor."
            routing_dtype = torch.bfloat16
        else:
            routing_dtype = routing_logits.dtype

        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        tuner = AutoTuner.get()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        output = _alloc_trtllm_moe_output(
            num_tokens, hidden_size, do_finalize, hidden_states.device
        )

        if routing_logits is not None:
            topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
            expert_weights = torch.empty(
                0, dtype=routing_dtype, device=hidden_states.device
            )
        else:
            expert_weights = (
                expert_weights
                if expert_weights is not None
                else torch.empty(0, dtype=routing_dtype, device=hidden_states.device)
            )

        if hidden_states.dtype == torch.bfloat16:
            dtype_act = DtypeTrtllmGen.Bfloat16
        elif hidden_states.dtype == torch.float16:
            dtype_act = DtypeTrtllmGen.Fp16
        elif hidden_states.dtype == torch.float8_e4m3fn:
            dtype_act = DtypeTrtllmGen.E4m3
        else:
            raise ValueError(
                "FP8 per-channel MoE hidden_states must have dtype "
                "torch.bfloat16, torch.float16, or torch.float8_e4m3fn, got "
                f"{hidden_states.dtype}."
            )

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=DtypeTrtllmGen.E4m3,
            fp8_quantization_type=Fp8QuantizationType.PerChannelFp8,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
            activation_type=activation_type,
            num_experts=num_experts,
        )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp8_per_channel_scale_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            routing_bias=routing_bias,
            gemm1_weights=gemm1_weights,
            gemm1_per_channel_weight_scale=gemm1_per_channel_weight_scale,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=output1_scale_gate_scalar,
            gemm2_weights=gemm2_weights,
            gemm2_per_channel_weight_scale=gemm2_per_channel_weight_scale,
            output2_scale_scalar=output2_scale_scalar,
            num_experts=num_experts,
            n_group=n_group,
            topk_group=topk_group,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
            routed_scaling_factor=routed_scaling_factor,
            use_routing_scales_on_input=use_routing_scales_on_input,
            routing_method_type=routing_method_type,
            do_finalize=do_finalize,
            enable_pdl=enable_pdl,
            activation_type=activation_type,
        )
        intermediate_output = moe_op.trtllm_fp8_per_channel_scale_moe(
            routing_logits,
            topk_ids,
            expert_weights,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_per_channel_weight_scale,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            gemm2_weights,
            gemm2_per_channel_weight_scale,
            output2_scale_scalar,
            output,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            use_routing_scales_on_input,
            routing_method_type,
            do_finalize,
            enable_pdl,
            [-1, -1] if tactic == -1 else tactic,
            activation_type,
            norm_topk_prob,
        )
        return _unpack_trtllm_moe_output(
            intermediate_output, output, do_finalize, None, expert_weights
        )

    @register_fake_op("flashinfer::trtllm_fp8_per_channel_scale_moe")
    def _fake_trtllm_fp8_per_channel_scale_moe(
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_per_channel_weight_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_per_channel_weight_scale: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        use_routing_scales_on_input: bool,
        routing_method_type: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
    ):
        seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        return [hidden_states.new_empty([seq_len, hidden_size], dtype=torch.bfloat16)]

    @register_custom_op(
        "flashinfer::trtllm_fp8_block_scale_moe",
        mutates_args=("routing_replay_out",),
    )
    def trtllm_fp8_block_scale_moe_op(
        routing_input_mode: int,
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_lora_delta: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
        num_fused_shared_experts: int = 0,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        # Determine routing mode: compute from logits or use pre-computed
        if routing_logits is None:
            assert topk_ids is not None, (
                "either topk_ids or routing_logits must be provided."
            )
            assert topk_ids.dtype == torch.int32, "topk_ids must be an int32 tensor."
        routing_dtype = torch.bfloat16

        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)

        # Use AutoTuner to select the best tactic
        tuner = AutoTuner.get()

        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]

        if output is None:
            output = _alloc_trtllm_moe_output(
                num_tokens, hidden_size, do_finalize, hidden_states.device
            )
        elif do_finalize:
            check_shape_dtype_device(
                output,
                (num_tokens, hidden_size),
                torch.bfloat16,
                hidden_states.device,
                "output",
            )

        if routing_logits is not None:
            # When routing_logits is provided, allocate empty buffers (kernel will fill them)
            topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
            expert_weights = torch.empty(
                0, dtype=routing_dtype, device=hidden_states.device
            )
        else:
            # When routing_logits is None, we have pre-computed routing:
            # - packed format: topk_ids contains ``(expert_id << 16) | weight``
            # - unpacked format: separate topk_ids and expert_weights
            topk_ids = topk_ids
            expert_weights = (
                expert_weights
                if expert_weights is not None
                else torch.empty(0, dtype=routing_dtype, device=hidden_states.device)
            )

        dtype_act = (
            DtypeTrtllmGen.E4m3
            if fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
            else DtypeTrtllmGen.MxE4m3
        )  # FP8 activation
        dtype_weights = (
            DtypeTrtllmGen.E4m3
            if fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
            else DtypeTrtllmGen.MxE4m3
        )  # FP8 weights
        _validate_fp8_block_scale_gemm1_activation_params(
            fp8_quantization_type,
            activation_type,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
        )

        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=local_num_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=fp8_quantization_type,  # block_scale mode
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            weight_layout=weight_layout,
            use_shuffled_weight=use_shuffled_weight,
            num_experts=num_experts,
            num_fused_shared_experts=num_fused_shared_experts,
        )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=gemm1_lora_delta,
            per_token_scale=None,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            routing_input_mode=RoutingInputMode(routing_input_mode),
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

        runner_kwargs = {
            "routing_input_mode": routing_input_mode,
            "routing_bias": routing_bias,
            "gemm1_weights": gemm1_weights,
            "gemm1_weights_scale": gemm1_weights_scale,
            "gemm1_alpha": gemm1_alpha,
            "gemm1_beta": gemm1_beta,
            "gemm1_clamp_limit": gemm1_clamp_limit,
            "gemm2_weights": gemm2_weights,
            "gemm2_weights_scale": gemm2_weights_scale,
            "num_experts": num_experts,
            "n_group": n_group,
            "topk_group": topk_group,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": local_num_experts,
            "routed_scaling_factor": routed_scaling_factor,
            "routing_method_type": routing_method_type,
            "use_shuffled_weight": use_shuffled_weight,
            "weight_layout": weight_layout,
            "do_finalize": do_finalize,
            "enable_pdl": enable_pdl,
            "num_fused_shared_experts": num_fused_shared_experts,
            "norm_topk_prob": norm_topk_prob,
            "routing_replay_out": routing_replay_out,
        }
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp8_block_scale_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            **runner_kwargs,
        )
        _nfse = num_fused_shared_experts if num_fused_shared_experts is not None else 0

        def run_selected_tactic(tactic: Any) -> List[torch.Tensor]:
            """Launch the unchanged FP8 block-scale operation with one tactic."""
            # Preserve the exact DeepSeek/MXFP8 ordinary ABI selected by runner configuration.
            intermediate_output = moe_op.trtllm_fp8_block_scale_moe(
                routing_input_mode,
                routing_logits,
                topk_ids,
                expert_weights,
                routing_bias,
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm1_lora_delta,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                gemm2_weights,
                gemm2_weights_scale,
                output,
                num_experts,
                top_k,
                _nfse,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routed_scaling_factor,
                routing_method_type,
                use_shuffled_weight,
                weight_layout,
                do_finalize,
                enable_pdl,
                [-1, -1] if tactic == -1 else tactic,
                fp8_quantization_type,
                activation_type,
                norm_topk_prob,
                routing_replay_out,
                [],
                [],
                False,
            )
            # Reconstruct the public result without exposing dtype-specific body workspaces.
            return _unpack_trtllm_moe_output(
                intermediate_output,
                output,
                do_finalize,
                gemm1_lora_delta,
                expert_weights,
            )

        routing_mode = RoutingInputMode(routing_input_mode)
        # When do_finalize=False, the FC2 output format is determined on device based on runtime
        # expert distribution. Therefore it is not eligible for DA until we can canonicalize
        # output format. Generic block-scale DA also needs precomputed routing, excludes fused
        # shared experts, and retains LoRA on the ordinary body-specific multi-output ABI.
        # TODO(da-moe): Prepare graph-stable LoRA auxiliary outputs before admitting this ABI.
        da_eligible = (
            routing_mode
            in (
                RoutingInputMode.PackedPrecomputed,
                RoutingInputMode.UnpackedPrecomputed,
            )
            and topk_ids is not None
            and do_finalize
            and gemm1_lora_delta is None
            and _nfse == 0
            and 0 < num_experts <= DA_MAX_EXPERTS
        )
        if not da_eligible:
            return run_selected_tactic(tactic)

        from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic

        da_config = _enabled_trtllm_da_config()
        if da_config is None:
            return run_selected_tactic(tactic)
        return run_dist_aware_tactic(
            custom_op="flashinfer::trtllm_fp8_block_scale_moe",
            tuner=tuner,
            config=da_config,
            runner=moe_runner,
            runtime=TrtllmDaRuntime(moe_runner),
            tuning_config=tuning_config,
            inputs=moe_inputs.to_list(),
            runner_kwargs=runner_kwargs,
            baseline_tactic=tactic,
            routing_input_mode=routing_input_mode,
            routing_id_index=MoeRunnerInputs.idx("topk_ids"),
            routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=local_num_experts,
            top_k=top_k,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            run_fixed_tactic=run_selected_tactic,
            finish_switch=lambda: _unpack_trtllm_moe_output(
                [], output, do_finalize, gemm1_lora_delta, expert_weights
            ),
        )

    @register_fake_op("flashinfer::trtllm_fp8_block_scale_moe")
    def _fake_trtllm_fp8_block_scale_moe(
        routing_input_mode: int,
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_lora_delta: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int = 0,
        use_shuffled_weight: bool = False,
        weight_layout: int = 0,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        tune_max_num_tokens: int = 8192,
        fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
        num_fused_shared_experts: int = 0,
        activation_type: int = ActivationType.Swiglu.value,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        # Acknowledge mutation-only and fallback-only controls without executing the native op.
        _ = routing_replay_out
        return _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=hidden_states.shape[1],
            intermediate_size=intermediate_size,
            top_k=top_k,
            do_finalize=do_finalize,
            output=output,
            expert_weights=expert_weights,
            gemm1_lora_delta=gemm1_lora_delta,
            num_fused_shared_experts=num_fused_shared_experts,
        )

    @register_custom_op(
        "flashinfer::trtllm_fp4_block_scale_moe",
        mutates_args=("routing_replay_out",),
    )
    def trtllm_fp4_block_scale_moe_op(
        routing_input_mode: int,
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: Optional[torch.Tensor],
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_lora_delta: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: Optional[torch.Tensor],
        output1_scale_gate_scalar: Optional[torch.Tensor],
        output2_scale_scalar: Optional[torch.Tensor],
        per_token_scale: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        num_local_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        enable_pdl: Optional[bool] = None,
        activation_type: int = ActivationType.Swiglu.value,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
        num_fused_shared_experts: int = 0,
    ) -> List[torch.Tensor]:
        if routing_logits is None:
            assert topk_ids is not None, (
                "either topk_ids or routing_logits must be provided."
            )
            assert topk_ids.dtype == torch.int32, "topk_ids must be an int32 tensor."
        # The trtllm-gen routing kernel always emits expert weights as bfloat16
        # (routingData.mDtypeOutput is hard-set to Bfloat16 for every routing
        # method in csrc/trtllm_fused_moe_runner.cu), independent of the
        # routing_logits dtype. This buffer is returned verbatim to the caller
        # when do_finalize=False, so it must be bfloat16 regardless of
        # routing_logits.dtype (e.g. fp32 DeepSeekV3 logits); otherwise the
        # returned expert_weights mislabels bf16 data as fp32. See #3595.
        routing_dtype = torch.bfloat16
        hidden_size = hidden_states.shape[-1]
        if hidden_states.dtype == torch.uint8:
            hidden_size = hidden_size * 2
        num_tokens = hidden_states.shape[0]

        # workspace buffers required by trtllm-gen
        # For Mode 3 (UnpackedPrecomputed), topk_ids and topk_weights are user-provided INPUTS
        if routing_input_mode == RoutingInputMode.UnpackedPrecomputed:
            assert num_fused_shared_experts == 0, (
                "num_fused_shared_experts > 0 is not supported with pre-computed routing"
            )
            assert topk_ids is not None, (
                "topk_ids must be provided for UnpackedPrecomputed mode"
            )
            assert topk_weights is not None, (
                "topk_weights must be provided for UnpackedPrecomputed mode"
            )
            assert topk_weights.dtype in (torch.bfloat16, torch.float32), (
                f"topk_weights must be bfloat16 or float32, got {topk_weights.dtype}."
            )
        else:
            # For Mode 1 (FromLogits) and Mode 2 (PackedPrecomputed), allocate OUTPUT buffers.
            # The routing kernel writes top_k + num_fused_shared_experts slots per token
            # (fused shared experts are appended after the routed top-k).
            if topk_ids is None:
                topk_ids = torch.empty(
                    num_tokens,
                    top_k + num_fused_shared_experts,
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            if topk_weights is None:
                # FP4BlockScaleLauncher borrows this buffer instead of allocating
                # FusedMoeLauncher::expert_weights. Keep it non-empty so
                # do_finalize=False can return valid weights; the routing kernel
                # fills it for both FromLogits and PackedPrecomputed.
                topk_weights = torch.empty(
                    num_tokens,
                    top_k + num_fused_shared_experts,
                    dtype=routing_dtype,
                    device=hidden_states.device,
                )
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        if output is None:
            output = _alloc_trtllm_moe_output(
                num_tokens, hidden_size, do_finalize, hidden_states.device
            )
        elif do_finalize:
            check_shape_dtype_device(
                output, None, torch.bfloat16, hidden_states.device, "output"
            )
            assert output.shape[0] == num_tokens, (
                f"output.shape[0]={output.shape[0]} must be equal to {num_tokens}"
            )
            assert output.shape[1] <= hidden_size, (
                f"output.shape[1]={output.shape[1]} must be less than or equal to {hidden_size}"
            )

        tuner = AutoTuner.get()
        dtype_act = deduce_trtllm_gen_tensor_dtype(hidden_states, hidden_states_scale)
        dtype_weights = deduce_trtllm_gen_tensor_dtype(
            gemm1_weights, gemm1_weights_scale
        )
        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=num_local_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
            use_per_token_scaling=per_token_scale is not None,
            num_experts=num_experts,
            num_fused_shared_experts=num_fused_shared_experts,
        )
        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=topk_weights,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=gemm1_lora_delta,
            per_token_scale=per_token_scale,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            routing_input_mode=RoutingInputMode(routing_input_mode),
            use_cold_l2_cache=True,
            use_cuda_graph=True,
        )

        runner_kwargs = {
            "routing_input_mode": routing_input_mode,
            "num_experts": num_experts,
            "routing_bias": routing_bias,
            "gemm1_weights": gemm1_weights,
            "gemm1_weights_scale": gemm1_weights_scale,
            "gemm1_bias": gemm1_bias,
            "gemm1_alpha": gemm1_alpha,
            "gemm1_beta": gemm1_beta,
            "gemm1_clamp_limit": gemm1_clamp_limit,
            "gemm2_weights": gemm2_weights,
            "gemm2_weights_scale": gemm2_weights_scale,
            "gemm2_bias": gemm2_bias,
            "output1_scale_scalar": output1_scale_scalar,
            "output1_scale_gate_scalar": output1_scale_gate_scalar,
            "output2_scale_scalar": output2_scale_scalar,
            "per_token_scale": per_token_scale,
            "n_group": n_group,
            "topk_group": topk_group,
            "local_expert_offset": local_expert_offset,
            "routed_scaling_factor": routed_scaling_factor,
            "routing_method_type": routing_method_type,
            "enable_pdl": enable_pdl,
            "do_finalize": do_finalize,
            "activation_type": activation_type,
            "num_fused_shared_experts": num_fused_shared_experts,
            "norm_topk_prob": norm_topk_prob,
            "routing_replay_out": routing_replay_out,
        }
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_fp4_block_scale_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            **runner_kwargs,
        )

        def run_selected_tactic(tactic: Any) -> List[torch.Tensor]:
            """Launch the unchanged monolithic FP4 operation with one fixed tactic."""
            intermediate_output = moe_op.trtllm_fp4_block_scale_moe(
                routing_input_mode,
                routing_logits,
                topk_ids,
                topk_weights,
                routing_bias,
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm1_bias,
                gemm1_lora_delta,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                gemm2_weights,
                gemm2_weights_scale,
                gemm2_bias,
                output1_scale_scalar,
                output1_scale_gate_scalar,
                output2_scale_scalar,
                per_token_scale,
                num_experts,
                top_k,
                num_fused_shared_experts,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                num_local_experts,
                routed_scaling_factor,
                routing_method_type,
                do_finalize,
                enable_pdl,
                activation_type,
                output,
                [-1, -1] if tactic == -1 else tactic,
                norm_topk_prob,
                routing_replay_out,
                [],
                [],
                False,
            )
            # FP4 always borrows the caller's topk_weights buffer (the launcher has
            # no allocate branch), so it is always the source for expert_weights.
            return _unpack_trtllm_moe_output(
                intermediate_output,
                output,
                do_finalize,
                gemm1_lora_delta,
                topk_weights,
            )

        # When do_finalize=False, the FC2 output format is determined on device based on runtime
        # expert distribution. Therefore it is not eligible for DA until we can canonicalize
        # output format. FP4 DA also needs precomputed IDs/weights, excludes fused shared experts,
        # and retains LoRA on the ordinary body-specific multi-output ABI.
        # TODO(da-moe): Prepare graph-stable LoRA auxiliary outputs before admitting this ABI.
        da_eligible = (
            RoutingInputMode(routing_input_mode)
            in (
                RoutingInputMode.PackedPrecomputed,
                RoutingInputMode.UnpackedPrecomputed,
            )
            and topk_ids is not None
            and topk_weights is not None
            and do_finalize
            and gemm1_lora_delta is None
            and num_fused_shared_experts == 0
            and 0 < num_experts <= DA_MAX_EXPERTS
        )
        if not da_eligible:
            return run_selected_tactic(tactic)

        from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic

        da_config = _enabled_trtllm_da_config()
        if da_config is None:
            return run_selected_tactic(tactic)

        flat_inputs = moe_inputs.to_list()
        routing_id_index = MoeRunnerInputs.idx("topk_ids")
        routing_weight_index = MoeRunnerInputs.idx("expert_weights")
        runtime = TrtllmDaRuntime(moe_runner)
        return run_dist_aware_tactic(
            custom_op="flashinfer::trtllm_fp4_block_scale_moe",
            tuner=tuner,
            config=da_config,
            runner=moe_runner,
            runtime=runtime,
            tuning_config=tuning_config,
            inputs=flat_inputs,
            runner_kwargs=runner_kwargs,
            baseline_tactic=tactic,
            routing_input_mode=routing_input_mode,
            routing_id_index=routing_id_index,
            routing_weight_index=routing_weight_index,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            top_k=top_k,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            run_fixed_tactic=run_selected_tactic,
            finish_switch=lambda: _unpack_trtllm_moe_output(
                [], output, do_finalize, gemm1_lora_delta, topk_weights
            ),
        )

    @register_fake_op("flashinfer::trtllm_fp4_block_scale_moe")
    def _fake_trtllm_fp4_block_scale_moe(
        routing_input_mode: int,
        routing_logits: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        topk_weights: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: Optional[torch.Tensor],
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_lora_delta: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: Optional[torch.Tensor],
        output1_scale_gate_scalar: Optional[torch.Tensor],
        output2_scale_scalar: Optional[torch.Tensor],
        per_token_scale: Optional[torch.Tensor],
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        num_local_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        enable_pdl: Optional[bool] = None,
        activation_type: int = ActivationType.Swiglu.value,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
        num_fused_shared_experts: int = 0,
    ):
        # Acknowledge mutation-only and fallback-only controls without executing the native op.
        _ = routing_replay_out
        return _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=gemm2_weights.shape[1],
            intermediate_size=intermediate_size,
            top_k=top_k,
            do_finalize=do_finalize,
            output=output,
            expert_weights=topk_weights,
            gemm1_lora_delta=gemm1_lora_delta,
            num_fused_shared_experts=num_fused_shared_experts,
        )

    @register_custom_op(
        "flashinfer::trtllm_mxint4_block_scale_moe",
        mutates_args=("routing_replay_out",),
    )
    def trtllm_mxint4_block_scale_moe_op(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm1_lora_delta: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        num_local_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        assert routing_logits is not None or topk_ids is not None, (
            "either routing_logits or topk_ids must be provided"
        )
        hidden_size = hidden_states.shape[-1]
        if hidden_states.dtype == torch.uint8:
            hidden_size = hidden_size * 2
        num_tokens = hidden_states.shape[0]

        if routing_logits is not None:
            # When routing_logits is provided, we must pass topk_ids/expert_weights with no allocation
            topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
            expert_weights = torch.empty(
                0, dtype=torch.bfloat16, device=hidden_states.device
            )
        else:
            # When routing_logits is provided, we either have topk_ids/expert_weights,
            # packed into a single tensor as topk_id
            # or have them individually as topk_ids and expert_weights respectively
            topk_ids = topk_ids
            expert_weights = (
                expert_weights
                if expert_weights is not None
                else torch.empty(0, dtype=torch.bfloat16, device=hidden_states.device)
            )
        if enable_pdl is None:
            enable_pdl = device_support_pdl(hidden_states.device)
        if output is None:
            output = _alloc_trtllm_moe_output(
                num_tokens, hidden_size, do_finalize, hidden_states.device
            )

        tuner = AutoTuner.get()
        dtype_act = DtypeTrtllmGen.Bfloat16
        dtype_weights = DtypeTrtllmGen.MxInt4
        moe_runner = MoERunner(
            top_k=top_k,
            num_local_experts=num_local_experts,
            dtype_act=dtype_act,
            dtype_weights=dtype_weights,
            fp8_quantization_type=Fp8QuantizationType.NoneFp8,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=ActivationType.Swiglu,
            weight_layout=WeightLayout.BlockMajorK,
            use_shuffled_weight=True,
            num_experts=num_experts,
        )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_lora_delta=gemm1_lora_delta,
            per_token_scale=None,
        )
        tuning_config = moe_runner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=tune_max_num_tokens,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

        routing_input_mode = (
            RoutingInputMode.FromLogits
            if routing_logits is not None
            else RoutingInputMode.PackedPrecomputed
        )
        runner_kwargs = {
            "routing_input_mode": routing_input_mode,
            "num_experts": num_experts,
            "routing_bias": routing_bias,
            "gemm1_weights": gemm1_weights,
            "gemm1_weights_scale": gemm1_weights_scale,
            "gemm1_alpha": gemm1_alpha,
            "gemm1_beta": gemm1_beta,
            "gemm1_clamp_limit": gemm1_clamp_limit,
            "gemm2_weights": gemm2_weights,
            "gemm2_weights_scale": gemm2_weights_scale,
            "n_group": n_group,
            "topk_group": topk_group,
            "local_expert_offset": local_expert_offset,
            "local_num_experts": num_local_experts,
            "routed_scaling_factor": routed_scaling_factor,
            "routing_method_type": routing_method_type,
            "do_finalize": do_finalize,
            "enable_pdl": enable_pdl,
            "norm_topk_prob": norm_topk_prob,
            "routing_replay_out": routing_replay_out,
        }
        _, tactic = tuner.choose_one(
            "flashinfer::trtllm_mxint4_block_scale_moe",
            [moe_runner],
            tuning_config,
            moe_inputs.to_list(),
            **runner_kwargs,
        )

        def run_selected_tactic(tactic: Any) -> List[torch.Tensor]:
            """Launch the unchanged MXINT4 operation with one complete tactic."""
            # Preserve the exact MXINT4 ordinary FFI ABI for every fixed-tactic path.
            intermediate_output = moe_op.trtllm_mxint4_block_scale_moe(
                routing_logits,
                routing_bias,
                topk_ids,
                expert_weights,
                hidden_states,
                gemm1_weights,
                gemm1_weights_scale,
                gemm1_alpha,
                gemm1_beta,
                gemm1_clamp_limit,
                gemm1_lora_delta,
                gemm2_weights,
                gemm2_weights_scale,
                num_experts,
                top_k,
                n_group,
                topk_group,
                intermediate_size,
                local_expert_offset,
                num_local_experts,
                routed_scaling_factor,
                routing_method_type,
                do_finalize,
                enable_pdl,
                output,
                [-1, -1] if tactic == -1 else tactic,
                norm_topk_prob,
                routing_replay_out,
                [],
                [],
                False,
            )
            # Reconstruct the established public result after the native launch completes.
            return _unpack_trtllm_moe_output(
                intermediate_output,
                output,
                do_finalize,
                gemm1_lora_delta,
                expert_weights,
            )

        # When do_finalize=False, the FC2 output format is determined on device based on runtime
        # expert distribution. Therefore it is not eligible for DA until we can canonicalize
        # output format. MXINT4 DA currently consumes its packed precomputed routing ABI and
        # retains LoRA on the ordinary body-specific multi-output ABI.
        # TODO(da-moe): Prepare graph-stable LoRA auxiliary outputs before admitting this ABI.
        da_eligible = (
            routing_input_mode is RoutingInputMode.PackedPrecomputed
            and topk_ids is not None
            and do_finalize
            and gemm1_lora_delta is None
            and 0 < num_experts <= DA_MAX_EXPERTS
        )
        if not da_eligible:
            return run_selected_tactic(tactic)

        from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic

        da_config = _enabled_trtllm_da_config()
        if da_config is None:
            return run_selected_tactic(tactic)
        return run_dist_aware_tactic(
            custom_op="flashinfer::trtllm_mxint4_block_scale_moe",
            tuner=tuner,
            config=da_config,
            runner=moe_runner,
            runtime=TrtllmDaRuntime(moe_runner),
            tuning_config=tuning_config,
            inputs=moe_inputs.to_list(),
            runner_kwargs=runner_kwargs,
            baseline_tactic=tactic,
            routing_input_mode=routing_input_mode,
            routing_id_index=MoeRunnerInputs.idx("topk_ids"),
            routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            top_k=top_k,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            run_fixed_tactic=run_selected_tactic,
            finish_switch=lambda: _unpack_trtllm_moe_output(
                [], output, do_finalize, gemm1_lora_delta, expert_weights
            ),
        )

    @register_fake_op("flashinfer::trtllm_mxint4_block_scale_moe")
    def _fake_trtllm_mxint4_block_scale_moe(
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        expert_weights: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm1_lora_delta: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        num_local_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool = True,
        enable_pdl: Optional[bool] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        norm_topk_prob: bool = True,
        routing_replay_out: Optional[torch.Tensor] = None,
    ):
        # Acknowledge the declared mutation-only argument without reading device data in fake mode.
        _ = routing_replay_out
        return _fake_trtllm_moe_output(
            hidden_states,
            hidden_size=hidden_states.shape[1],
            intermediate_size=intermediate_size,
            top_k=top_k,
            do_finalize=do_finalize,
            output=output,
            expert_weights=expert_weights,
            gemm1_lora_delta=gemm1_lora_delta,
        )

    return SimpleNamespace(
        trtllm_bf16_moe=trtllm_bf16_moe_op,
        trtllm_fp8_per_tensor_scale_moe=trtllm_fp8_per_tensor_scale_moe_op,
        trtllm_fp8_per_tensor_scale_routed_moe=trtllm_fp8_per_tensor_scale_routed_moe_op,
        trtllm_fp8_per_channel_scale_moe=trtllm_fp8_per_channel_scale_moe_op,
        trtllm_fp8_block_scale_moe=trtllm_fp8_block_scale_moe_op,
        trtllm_fp4_block_scale_moe=trtllm_fp4_block_scale_moe_op,
        trtllm_mxint4_block_scale_moe=trtllm_mxint4_block_scale_moe_op,
        allocate_routing_metadata_multi_tile=(
            moe_op.trtllm_moe_allocate_routing_metadata_multi_tile
        ),
        max_da_multi_tile_tokens=moe_op.trtllm_moe_max_da_multi_tile_tokens,
        populate_routing_metadata_multi_tile=(
            moe_op.trtllm_moe_populate_routing_metadata_multi_tile
        ),
        allocate_canonical_routing=moe_op.trtllm_moe_allocate_canonical_routing,
        canonicalize_routing=moe_op.trtllm_moe_canonicalize_routing,
        begin_da_switch_capture=moe_op.trtllm_moe_begin_da_switch_capture,
        inspect_da_workspace_lane=moe_op.trtllm_moe_inspect_da_workspace_lane,
        create_da_body_capture_stream=(moe_op.trtllm_moe_create_da_body_capture_stream),
        destroy_da_body_capture_stream=(
            moe_op.trtllm_moe_destroy_da_body_capture_stream
        ),
        begin_da_body_capture=moe_op.trtllm_moe_begin_da_body_capture,
        end_da_body_capture=moe_op.trtllm_moe_end_da_body_capture,
        finish_da_switch_capture=moe_op.trtllm_moe_finish_da_switch_capture,
        # Canonical tactic-aware TunableRunner (closes over the raw moe_op and
        # trtllm_get_valid_moe_configs).  Exposed so the unified MoE API's
        # TrtllmFp4RoutedRunner can delegate to it instead of re-deriving the
        # raw op's positional call.
        MoERunner=MoERunner,
        DABodyRunner=DABodyRunner,
        DAProfileRunner=DAProfileRunner,
    )


def allocate_trtllm_moe_canonical_routing(
    routing_logits: torch.Tensor, *, top_k: int, tile_n: int
) -> TRTLLMCanonicalRouting:
    """Allocate stable real-router outputs and scratch without launching routing."""
    runtime = get_trtllm_moe_sm100_module()
    tensors = [
        _torch_view_of_ffi_tensor(tensor)
        for tensor in runtime.allocate_canonical_routing(routing_logits, top_k, tile_n)
    ]
    if len(tensors) != 11:
        raise RuntimeError(
            "Native canonical routing allocation returned an invalid ABI"
        )
    return TRTLLMCanonicalRouting(
        routing_replay_ids=tensors[0],
        expert_weights=tensors[1],
        packed_router_scratch=tensors[2],
        scratch=tuple(tensors[3:]),
        tile_n=tile_n,
    )


def _torch_view_of_ffi_tensor(tensor: Any) -> torch.Tensor:
    """Return a zero-copy Torch view for one tensor crossing the TVM-FFI boundary."""
    if isinstance(tensor, torch.Tensor):
        return tensor
    return torch.from_dlpack(tensor)


def canonicalize_trtllm_moe_routing_(
    canonical: TRTLLMCanonicalRouting,
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    *,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    use_routing_scales_on_input: bool,
    use_deep_seek_fp8: bool,
    norm_topk_prob: bool,
    enable_pdl: bool,
) -> None:
    """Run the real router into stable packed and native replay representations."""
    # Reuse the graph-stable canonical allocation; routing content changes without changing any
    # address captured by later fixed-body profiling or framework replay.
    runtime = get_trtllm_moe_sm100_module()
    # Forward the complete public routing policy so native canonicalization matches ordinary MoE.
    runtime.canonicalize_routing(
        routing_logits,
        routing_bias,
        hidden_states,
        canonical.tensors(),
        top_k,
        n_group,
        topk_group,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_routing_scales_on_input,
        use_deep_seek_fp8,
        norm_topk_prob,
        enable_pdl,
        canonical.tile_n,
    )


def trtllm_moe_allocate_routing_metadata(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    num_local_experts: int,
    tile_n: int,
    routing_input_mode: RoutingInputMode,
    topk_weights: Optional[torch.Tensor] = None,
) -> TrtllmMoERoutingMetadataSlot:
    """Allocate one tile's graph-stable routing metadata as a reference slot."""
    # Delegate to the fused multi-tile allocator so one- and many-tile callers share one ABI.
    metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
        topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_ns=(tile_n,),
        routing_input_mode=routing_input_mode,
        topk_weights=topk_weights,
    )
    # A one-tile request is guaranteed to produce exactly one canonicalized slot.
    return metadata.slots[0]


def trtllm_moe_allocate_routing_metadata_multi_tile(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    num_local_experts: int,
    tile_ns: Sequence[int],
    routing_input_mode: RoutingInputMode,
    topk_weights: Optional[torch.Tensor] = None,
) -> TrtllmMoERoutingMetadata:
    """Allocate graph-stable metadata for up to eight sorted unique tiles."""
    # Canonicalize ordering before native allocation because body-to-slot lookup is tile based and
    # CUDA Graph storage must remain deterministic.
    canonical_tile_ns = tuple(sorted(set(int(tile_n) for tile_n in tile_ns)))
    if len(canonical_tile_ns) != len(tile_ns):
        raise ValueError("tile_ns must contain unique values")
    runtime = get_trtllm_moe_sm100_module()
    # Native code returns a flat repeated nine-tensor ABI at the TVM-FFI boundary.
    flat = [
        _torch_view_of_ffi_tensor(tensor)
        for tensor in runtime.allocate_routing_metadata_multi_tile(
            topk_ids,
            num_experts,
            top_k,
            local_expert_offset,
            num_local_experts,
            list(canonical_tile_ns),
            int(routing_input_mode),
            topk_weights,
        )
    ]
    tensors_per_slot = 9
    if len(flat) != tensors_per_slot * len(canonical_tile_ns):
        raise RuntimeError("Native routing metadata allocation returned an invalid ABI")
    # Decode the flat boundary exactly once into named Python records used by preparation/capture.
    slots = []
    for index, tile_n in enumerate(canonical_tile_ns):
        offset = index * tensors_per_slot
        slots.append(
            TrtllmMoERoutingMetadataSlot(
                tile_n=tile_n,
                total_num_padded_tokens=flat[offset],
                expanded_idx_to_permuted_idx=flat[offset + 1],
                permuted_idx_to_token_idx=flat[offset + 2],
                expert_weights=flat[offset + 3],
                expert_count_histogram=flat[offset + 4],
                num_tokens_per_expert=flat[offset + 5],
                cta_idx_xy_to_batch_idx=flat[offset + 6],
                cta_idx_xy_to_mn_limit=flat[offset + 7],
                num_non_exiting_ctas=flat[offset + 8],
            )
        )
    return TrtllmMoERoutingMetadata(
        routing_input_mode=routing_input_mode,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        slots=tuple(slots),
    )


def populate_trtllm_moe_routing_metadata_(
    metadata: TrtllmMoERoutingMetadata,
    topk_ids: torch.Tensor,
    topk_weights: Optional[torch.Tensor] = None,
) -> None:
    """Populate prepared slots in place with one fused live-input kernel."""
    runtime = get_trtllm_moe_sm100_module()
    runtime.populate_routing_metadata_multi_tile(
        topk_ids,
        metadata.num_experts,
        metadata.top_k,
        metadata.local_expert_offset,
        metadata.num_local_experts,
        list(metadata.tile_ns),
        metadata.flat_tensors(),
        int(metadata.routing_input_mode),
        topk_weights,
    )


class TrtllmDaRuntime:
    """Prepare and capture production TRTLLM DA bodies around one MoERunner."""

    def __init__(self, moe_runner: Any) -> None:
        """Compose the dtype-agnostic ordinary runner with its DA body capability."""
        runtime = get_trtllm_moe_sm100_module()
        # Ordinary full-operation runner retained for fallback and fixed-body capture.
        self._moe_runner = moe_runner
        # Prepared-metadata capability that delegates to the runner's exact dtype branch.
        self._body_runner = runtime.DABodyRunner(moe_runner)

    @property
    def moe_runner(self) -> Any:
        """Return the composed ordinary full-operation runner."""
        return self._moe_runner

    def max_multi_tile_tokens(self, num_experts: int) -> int:
        """Return the native fused-preamble token bound for one expert domain."""
        runtime = get_trtllm_moe_sm100_module()
        return int(runtime.max_da_multi_tile_tokens(num_experts))

    def prepare_from_logits_profile(
        self,
        inputs: List[torch.Tensor],
        runner_kwargs: Mapping[str, Any],
        tile_n: int,
    ) -> tuple[TRTLLMCanonicalRouting, List[torch.Tensor], dict[str, Any]]:
        """Allocate canonical router outputs and expose native replay profiling inputs."""
        # Allocate stable router outputs once; later realizations overwrite contents in place.
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        if moe_inputs.routing_logits is None:
            raise ValueError("FromLogits profiling requires routing logits")
        canonical = allocate_trtllm_moe_canonical_routing(
            moe_inputs.routing_logits,
            top_k=self._moe_runner.top_k,
            tile_n=tile_n,
        )
        # Replace logits-only slots with native replay IDs and weights while preserving every other
        # exact body argument and its storage.
        profile_inputs = list(inputs)
        profile_inputs[MoeRunnerInputs.idx("routing_logits")] = None
        profile_inputs[MoeRunnerInputs.idx("topk_ids")] = canonical.routing_replay_ids
        profile_inputs[MoeRunnerInputs.idx("expert_weights")] = canonical.expert_weights
        profile_kwargs = dict(runner_kwargs)
        profile_kwargs["routing_input_mode"] = RoutingInputMode.UnpackedPrecomputed
        return canonical, profile_inputs, profile_kwargs

    def make_from_logits_body_inputs(
        self, canonical: TRTLLMCanonicalRouting, inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Return conventional placeholders for prepared bodies that consume metadata."""
        body_inputs = list(inputs)
        body_inputs[MoeRunnerInputs.idx("routing_logits")] = None
        body_inputs[MoeRunnerInputs.idx("topk_ids")] = canonical.packed_router_scratch
        body_inputs[MoeRunnerInputs.idx("expert_weights")] = canonical.expert_weights
        return body_inputs

    def make_from_logits_profile_runner(
        self, canonical: TRTLLMCanonicalRouting
    ) -> TunableRunner:
        """Compose the native replay preamble with dtype-specific prepared bodies."""
        runtime = get_trtllm_moe_sm100_module()
        return runtime.DAProfileRunner(
            self._body_runner,
            canonical.packed_router_scratch,
            canonical.expert_weights,
        )

    def make_from_logits_profile_tuning_config(
        self, profile_inputs: List[torch.Tensor], num_tokens: int
    ) -> TuningConfig:
        """Build the fixed-body cold-L2 profile config for canonical routed pairs."""
        return self._moe_runner._make_tuning_config(
            MoeRunnerInputs.from_list(profile_inputs),
            tune_max_num_tokens=num_tokens,
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )

    def refresh_canonical_routing(
        self,
        canonical: TRTLLMCanonicalRouting,
        inputs: List[torch.Tensor],
        runner_kwargs: Mapping[str, Any],
    ) -> None:
        """Refresh stable conventional and native replay outputs from live routing logits."""
        # Resolve the live logits input while retaining the preallocated canonical destinations.
        moe_inputs = MoeRunnerInputs.from_list(inputs)
        if moe_inputs.routing_logits is None:
            raise ValueError("Canonical routing requires live routing logits")
        # Forward the runner's exact routing policy so canonicalization matches ordinary dispatch.
        canonicalize_trtllm_moe_routing_(
            canonical,
            moe_inputs.routing_logits,
            runner_kwargs.get("routing_bias"),
            moe_inputs.hidden_states,
            top_k=self._moe_runner.top_k,
            n_group=runner_kwargs.get("n_group"),
            topk_group=runner_kwargs.get("topk_group"),
            local_expert_offset=runner_kwargs["local_expert_offset"],
            local_num_experts=runner_kwargs["local_num_experts"],
            routed_scaling_factor=runner_kwargs.get("routed_scaling_factor"),
            routing_method_type=runner_kwargs["routing_method_type"],
            use_routing_scales_on_input=runner_kwargs.get(
                "use_routing_scales_on_input", False
            ),
            use_deep_seek_fp8=(
                self._moe_runner.fp8_quantization_type
                == Fp8QuantizationType.DeepSeekFp8
            ),
            norm_topk_prob=runner_kwargs.get("norm_topk_prob", True),
            enable_pdl=runner_kwargs["enable_pdl"],
        )

    def prepare(
        self,
        plan: Any,
        inputs: List[torch.Tensor],
        topk_ids: torch.Tensor,
        *,
        num_experts: int,
        top_k: int,
        local_expert_offset: int,
        num_local_experts: int,
        routing_input_mode: RoutingInputMode,
        topk_weights: Optional[torch.Tensor] = None,
        **runner_kwargs,
    ) -> TrtllmDaResources:
        """Allocate routing and body resources during noncapturing framework warmup."""
        # Canonicalize logits routing to an unpacked stable pair before allocating metadata; direct
        # precomputed modes reuse the caller's live tensors unchanged.
        tile_ns = tuple(sorted({body.tile_n for body in plan.bodies}))
        canonical_routing = None
        body_inputs = inputs
        body_routing_mode = routing_input_mode
        body_input_mode = routing_input_mode
        if routing_input_mode == RoutingInputMode.FromLogits:
            canonical_kwargs = dict(runner_kwargs)
            canonical_kwargs["local_expert_offset"] = local_expert_offset
            canonical_kwargs["local_num_experts"] = num_local_experts
            canonical_routing, body_inputs, runner_kwargs = (
                self.prepare_from_logits_profile(inputs, canonical_kwargs, tile_ns[0])
            )
            self.refresh_canonical_routing(canonical_routing, inputs, runner_kwargs)
            body_inputs = self.make_from_logits_body_inputs(canonical_routing, inputs)
            topk_ids = canonical_routing.routing_replay_ids
            topk_weights = canonical_routing.expert_weights
            body_routing_mode = RoutingInputMode.UnpackedPrecomputed
            body_input_mode = RoutingInputMode.PackedPrecomputed
        # Allocate and prime every unique tile in one fused metadata topology shared by bodies.
        routing_metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
            topk_ids,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            num_local_experts=num_local_experts,
            tile_ns=tile_ns,
            routing_input_mode=body_routing_mode,
            topk_weights=topk_weights,
        )
        populate_trtllm_moe_routing_metadata_(routing_metadata, topk_ids, topk_weights)
        slots = {slot.tile_n: slot for slot in routing_metadata.slots}
        body_kwargs = dict(runner_kwargs)
        body_kwargs.update(
            routing_input_mode=body_input_mode,
            num_experts=num_experts,
            local_expert_offset=local_expert_offset,
        )
        # Retain the field-wise maximum allocation required by any exact-ABI body. Conditional
        # bodies are mutually exclusive, so every body may reuse this one stable pointer set.
        device_index = topk_ids.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        capture_stream = _get_trtllm_da_body_capture_stream(device_index)
        body_workspace = TrtllmDaBodyWorkspace(
            tensors=self._body_runner.prepare_max_body_workspace(
                body_inputs, plan.bodies, slots, **body_kwargs
            ),
            capture_stream=capture_stream,
        )
        return TrtllmDaResources(
            generation=plan.generation,
            routing_metadata=routing_metadata,
            body_workspace=body_workspace,
            selected_body=torch.full(
                (1,), -1, dtype=torch.int32, device=topk_ids.device
            ),
            canonical_routing=canonical_routing,
        )

    def capture_switch(
        self,
        plan: Any,
        resources: TrtllmDaResources,
        inputs: List[torch.Tensor],
        topk_ids: torch.Tensor,
        *,
        expected_capture_id: int,
        previous_conditional_node_handle: int,
        topk_weights: Optional[torch.Tensor] = None,
        **runner_kwargs,
    ) -> Optional[tuple[Any, int]]:
        """Inject one serial lane invocation or return None before graph mutation."""
        from flashinfer.fused_moe.da_moe import DAGraphTopology

        # Prove capture identity and transitive ordering before canonicalization writes any shared
        # lane storage. A failed proof leaves this invocation pristine for ordinary fallback.
        runtime = get_trtllm_moe_sm100_module()
        lane_inspection = tuple(
            int(value)
            for value in runtime.inspect_da_workspace_lane(
                topk_ids,
                expected_capture_id,
                previous_conditional_node_handle,
            )
        )
        if len(lane_inspection) != 2:
            raise RuntimeError("Incomplete native DA workspace-lane inspection")
        if not lane_inspection[1]:
            return None

        # Refresh canonical routing inside the outer capture when logits are the live framework
        # input, then expose its stable unpacked pair to every child body.
        metadata = resources.routing_metadata
        slots = {slot.tile_n: slot for slot in metadata.slots}
        body_inputs = inputs
        if resources.canonical_routing is not None:
            self.refresh_canonical_routing(
                resources.canonical_routing, inputs, runner_kwargs
            )
            body_inputs = self.make_from_logits_body_inputs(
                resources.canonical_routing, inputs
            )
            topk_ids = resources.canonical_routing.routing_replay_ids
            topk_weights = resources.canonical_routing.expert_weights
        # Body callbacks consume the metadata's canonical routing representation rather than the
        # original wrapper representation.
        body_kwargs = dict(runner_kwargs)
        body_kwargs.update(
            routing_input_mode=(
                RoutingInputMode.PackedPrecomputed
                if resources.canonical_routing is not None
                else metadata.routing_input_mode
            ),
            num_experts=metadata.num_experts,
            local_expert_offset=metadata.local_expert_offset,
        )
        # Begin the outer selector/preamble/SWITCH mutation and decode its cross-language child
        # graph handles before capturing any dtype-specific body.
        capture_state = TrtllmDaSwitchCaptureState.from_native(
            runtime.begin_da_switch_capture(
                topk_ids,
                metadata.num_experts,
                metadata.top_k,
                metadata.local_expert_offset,
                metadata.num_local_experts,
                list(metadata.tile_ns),
                metadata.flat_tensors(),
                int(metadata.routing_input_mode),
                topk_weights,
                plan.exemplar_spectra,
                plan.exemplar_body_indices,
                plan.num_selector_exemplars,
                resources.selected_body,
                len(plan.bodies),
                expected_capture_id,
                previous_conditional_node_handle,
            )
        )
        device_index = topk_ids.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        # Record each exact-ABI body directly into its CUDA-owned conditional child graph. The
        # bounded per-device stream is serialized across operation domains and capture threads.
        with _get_trtllm_da_body_capture_lock(device_index):
            workspace = resources.body_workspace
            for body, body_graph_handle in zip(
                plan.bodies, capture_state.body_graph_handles, strict=True
            ):
                stream_handle = workspace.capture_stream.handle
                runtime.begin_da_body_capture(
                    device_index, stream_handle, body_graph_handle
                )
                with torch.cuda.stream(workspace.capture_stream.external_stream):
                    self._body_runner.forward_from_metadata(
                        body_inputs,
                        body,
                        slots[body.tile_n],
                        workspace.tensors,
                        **body_kwargs,
                    )
                runtime.end_da_body_capture(
                    device_index, stream_handle, body_graph_handle
                )
        # Join the populated SWITCH to the outer capture and decode runtime-inspected topology.
        topology = runtime.finish_da_switch_capture(topk_ids, capture_state.to_native())
        return DAGraphTopology.from_native(
            topology
        ), capture_state.conditional_node_handle


def _validate_bf16_gemm1_activation_params(
    activation_type: int,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    local_num_experts: int,
    device: torch.device,
) -> None:
    if gemm1_alpha is None and gemm1_beta is None and gemm1_clamp_limit is None:
        return
    if int(activation_type) != int(ActivationType.Swiglu):
        raise ValueError(
            "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported "
            "for ActivationType.Swiglu."
        )
    for name, tensor in (
        ("gemm1_alpha", gemm1_alpha),
        ("gemm1_beta", gemm1_beta),
        ("gemm1_clamp_limit", gemm1_clamp_limit),
    ):
        if tensor is not None:
            check_shape_dtype_device(
                tensor,
                (local_num_experts,),
                torch.float32,
                device,
                name,
            )


def _validate_routing_replay_out(
    routing_replay_out: Optional[torch.Tensor],
    top_k: int,
    num_fused_shared_experts: int = 0,
) -> None:
    """Validate routing_replay_out tensor properties before passing to C++ kernels."""
    if routing_replay_out is None:
        return
    if num_fused_shared_experts > 0:
        # Replay records at stride top_k + nfse, mismatching the [num_tokens, top_k] layout.
        raise ValueError(
            "routing_replay_out is not supported with num_fused_shared_experts > 0"
        )
    if routing_replay_out.dtype != torch.int16:
        raise ValueError(
            f"routing_replay_out must be int16, got {routing_replay_out.dtype}"
        )
    if routing_replay_out.ndim != 2:
        raise ValueError(
            f"routing_replay_out must be 2D [num_tokens, top_k], got {routing_replay_out.ndim}D"
        )
    if routing_replay_out.shape[1] != top_k:
        raise ValueError(
            f"routing_replay_out dim1 must equal top_k={top_k}, got {routing_replay_out.shape[1]}"
        )
    if not routing_replay_out.is_contiguous():
        raise ValueError("routing_replay_out must be contiguous (packed row-major)")


def _validate_fp8_block_scale_gemm1_activation_params(
    fp8_quantization_type: Fp8QuantizationType,
    activation_type: int,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
) -> None:
    if gemm1_alpha is None and gemm1_beta is None and gemm1_clamp_limit is None:
        return
    if Fp8QuantizationType(fp8_quantization_type) not in (
        Fp8QuantizationType.MxFp8,
        Fp8QuantizationType.DeepSeekFp8,
    ):
        raise ValueError(
            "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported "
            "for Fp8QuantizationType.MxFp8 and Fp8QuantizationType.DeepSeekFp8 in "
            f"FP8 block scale MoE, got {Fp8QuantizationType(fp8_quantization_type)}."
        )
    if int(activation_type) != int(ActivationType.Swiglu):
        raise ValueError(
            "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported "
            "for ActivationType.Swiglu."
        )


@flashinfer_api(trace=trtllm_bf16_moe_trace)
def trtllm_bf16_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float] = None,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = True,
    weight_layout: int = WeightLayout.BlockMajorK,
    do_finalize: bool = True,
    enable_pdl: bool = True,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""BF16 MoE operation with autotuning support.

    Implements a bfloat16 Mixture of Experts layer using the TensorRT-LLM backend
    with automatic performance tuning for optimal tile-size selection.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` tensor of routing logits.  ``float32`` or
        ``bfloat16``.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` tensor of routing bias.  ``float32`` or
        ``bfloat16``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states.  Must be
        ``bfloat16``.
    gemm1_weights : torch.Tensor
        ``[num_experts, M // 128, hidden_size // 128, 128]`` first-layer
        weights, ``bfloat16``.  ``M`` equals ``2 * intermediate_size`` for
        gated activations and ``intermediate_size`` for non-gated
        activations.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size // 128, intermediate_size, 128]``
        second-layer weights, ``bfloat16``.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing (may be ``None`` for some methods).
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    use_shuffled_weight : bool
        Whether to use the shuffled weight layout (default ``True``).
    weight_layout : int
        Weight layout for ``gemm1_weights`` / ``gemm2_weights``; matches
        :class:`flashinfer.tllm_enums.WeightLayout`.  This BF16 MoE entry
        point requires ``BlockMajorK`` — passing any other value raises a
        runtime error.  Default ``WeightLayout.BlockMajorK``.

        - ``0`` ``MajorK`` — K-major, logical shape ``[Mn, K]``.
          *Not supported by this function.*
        - ``1`` ``MajorMn`` — M-major (A) / N-major (B), logical shape
          ``[K, Mn]``.  *Not supported by this function.*
        - ``2`` ``BlockMajorK`` — Blocked along K, logical shape
          ``[K / blockK, Mn, blockK]`` (``blockK`` is fixed at 128 B).
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : bool
        Whether to enable Programmatic Dependent Launch.  Auto-enabled for
        SM90+ when ``True``.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    activation_type : int
        Activation type (default ``3`` — Swiglu).  ``3`` Swiglu;
        ``6`` Relu2 (non-gated).
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities (default ``True``).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.  Column
        order matches ``topk_indices``.  When ``None`` (default) the
        kernel skips the write entirely.  The buffer may be larger than
        ``num_tokens`` for CUDA-graph pre-allocation; only rows
        ``[0, num_tokens)`` are written.
    gemm1_alpha : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 CUDA per-expert SwiGLU OA
        alpha parameter.  Supported with ``ActivationType.Swiglu``.  Any
        subset of ``gemm1_alpha``, ``gemm1_beta``, ``gemm1_clamp_limit``
        can be provided independently.  When ``None`` (default),
        ``alpha=1.0`` is used.  Let GEMM1 output be split as ``X1``
        (linear/up half) and ``X2`` (gate half).  The fused activation
        output is ``X2 * sigmoid(alpha * X2) * (X1 + beta)``.  Pass raw
        BF16-path values; no host-side scalar dequant-scale conversion is
        applied.
    gemm1_beta : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 CUDA per-expert SwiGLU OA
        beta parameter.  Supported with ``ActivationType.Swiglu``.  When
        ``None`` (default), ``beta=0.0`` is used.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 CUDA per-expert clamp
        limit.  Supported with ``ActivationType.Swiglu``.  When provided,
        ``X1 = clamp(X1, -limit, limit)`` and
        ``X2 = clamp(X2, max=limit)``.  When ``None`` (default), no clamp
        is applied.
    output : Optional[torch.Tensor]
        Optional in-place output tensor of shape ``[seq_len, hidden_size]``.
        Allocated internally when ``None`` (default).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        If ``do_finalize`` is ``True`` returns the final MoE output (deprecated
        scalar return; will become ``[output]`` in v0.8.0).  Otherwise returns
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
    """
    _validate_routing_replay_out(routing_replay_out, top_k)
    _validate_bf16_gemm1_activation_params(
        activation_type,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        local_num_experts,
        hidden_states.device,
    )
    result = get_trtllm_moe_sm100_module().trtllm_bf16_moe(
        RoutingInputMode.FromLogits,
        routing_logits,
        routing_bias,
        None,  # topk_ids
        None,  # expert_weights
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        None,  # gemm1_lora_delta — LoRA only supported with routed API to enforce consistent routing behavior
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        norm_topk_prob,
        routing_replay_out,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        output,
    )

    if do_finalize:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


def _split_precomputed_routing(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], "RoutingInputMode"]:
    """Split a routed-MoE ``topk_ids`` argument into its kernel-level inputs."""
    if isinstance(topk_ids, tuple):
        topk_ids_tensor, topk_weights = topk_ids
        return topk_ids_tensor, topk_weights, RoutingInputMode.UnpackedPrecomputed
    return topk_ids, None, RoutingInputMode.PackedPrecomputed


@flashinfer_api(trace=trtllm_bf16_routed_moe_trace)
def trtllm_bf16_routed_moe(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float] = None,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = True,
    weight_layout: int = WeightLayout.BlockMajorK,
    do_finalize: bool = True,
    enable_pdl: bool = True,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    r"""Pre-routed BF16 MoE operation with autotuning support.

    Like :func:`trtllm_bf16_moe`, but takes pre-computed routing instead of
    routing logits, either as a packed ``topk_ids`` tensor or as a
    ``(topk_ids, topk_weights)`` pair.

    Parameters
    ----------
    topk_ids : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``[seq_len, top_k]`` int32 tensor of packed expert indices and
        weights.  Format ``(expert_id << 16) | (weight_bf16.view(int16))``.
        Alternatively a ``(topk_ids, topk_weights)`` pair of plain ``int32``
        indices and ``bfloat16`` or ``float32`` weights.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states, ``bfloat16``.
    gemm1_weights : torch.Tensor
        ``[num_experts, M // 128, hidden_size // 128, 128]`` first-layer
        weights, ``bfloat16``.  ``M`` equals ``2 * intermediate_size`` for
        gated activations and ``intermediate_size`` for non-gated activations.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size // 128, intermediate_size, 128]``
        second-layer weights, ``bfloat16``.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing (may be ``None`` for some methods).
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    use_shuffled_weight : bool
        Whether to use the shuffled weight layout (default ``True``).
    weight_layout : int
        Weight layout for ``gemm1_weights`` / ``gemm2_weights``; matches
        :class:`flashinfer.tllm_enums.WeightLayout`.  This BF16 MoE entry
        point requires ``BlockMajorK`` — passing any other value raises a
        runtime error.  Default ``WeightLayout.BlockMajorK``.

        - ``0`` ``MajorK`` — K-major, logical shape ``[Mn, K]``.
          *Not supported by this function.*
        - ``1`` ``MajorMn`` — M-major (A) / N-major (B), logical shape
          ``[K, Mn]``.  *Not supported by this function.*
        - ``2`` ``BlockMajorK`` — Blocked along K, logical shape
          ``[K / blockK, Mn, blockK]`` (``blockK`` is fixed at 128 B).
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : bool
        Whether to enable Programmatic Dependent Launch (default ``True``).
    gemm1_lora_delta : Optional[torch.Tensor]
        Optional LoRA delta for GEMM1.  When provided the gated activation
        output is also returned for downstream LoRA adapters.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    activation_type : int
        Activation type (default ``3`` — Swiglu).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.  Column
        order matches ``topk_indices``.  When ``None`` (default) the
        kernel skips the write entirely.  The buffer may be larger than
        ``num_tokens`` for CUDA-graph pre-allocation; only rows
        ``[0, num_tokens)`` are written.
    gemm1_alpha : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 CUDA per-expert SwiGLU OA
        alpha parameter.  Supported with ``ActivationType.Swiglu``.  Any
        subset of ``gemm1_alpha``, ``gemm1_beta``, ``gemm1_clamp_limit``
        can be provided independently.  When ``None`` (default),
        ``alpha=1.0`` is used.  Let GEMM1 output be split as ``X1``
        (linear/up half) and ``X2`` (gate half).  The fused activation
        output is ``X2 * sigmoid(alpha * X2) * (X1 + beta)``.  Pass raw
        BF16-path values; no host-side scalar dequant-scale conversion is
        applied.
    gemm1_beta : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 CUDA per-expert SwiGLU OA
        beta parameter.  Supported with ``ActivationType.Swiglu``.  When
        ``None`` (default), ``beta=0.0`` is used.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 CUDA per-expert clamp
        limit.  Supported with ``ActivationType.Swiglu``.  When provided,
        ``X1 = clamp(X1, -limit, limit)`` and
        ``X2 = clamp(X2, max=limit)``.  When ``None`` (default), no clamp
        is applied.
    output : Optional[torch.Tensor]
        Optional in-place output tensor of shape ``[seq_len, hidden_size]``.
        Allocated internally when ``None`` (default).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Return shape depends on ``do_finalize`` and ``gemm1_lora_delta``.

        =============  ==================  =========================================================================
        do_finalize    gemm1_lora_delta    Returned tensors
        =============  ==================  =========================================================================
        ``True``       ``None``            ``output`` (deprecated scalar return; becomes ``[output]`` in v0.8.0)
        ``True``       ``Tensor``          ``[output, expanded_idx_to_permuted_idx, gemm1_activation_output]``
        ``False``      ``None``            ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``
        ``False``      ``Tensor``          ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx, gemm1_activation_output]``
        =============  ==================  =========================================================================
    """
    _validate_routing_replay_out(routing_replay_out, top_k)
    _validate_bf16_gemm1_activation_params(
        activation_type,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        local_num_experts,
        hidden_states.device,
    )
    topk_ids_tensor, topk_weights, routing_mode = _split_precomputed_routing(topk_ids)

    result = get_trtllm_moe_sm100_module().trtllm_bf16_moe(
        routing_mode,
        None,
        None,
        topk_ids_tensor,
        topk_weights,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        gemm1_lora_delta,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        True,  # norm_topk_prob: not used for pre-computed routing
        routing_replay_out,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        output,
    )

    if do_finalize and gemm1_lora_delta is None:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api(trace=trtllm_fp8_per_tensor_scale_moe_trace)
def trtllm_fp8_per_tensor_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""FP8 per-tensor-scale MoE operation.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` tensor of routing logits, ``float32`` or
        ``bfloat16``.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` tensor of routing bias, ``bfloat16`` or
        ``float32``.  May be ``None``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states.
        ``float8_e4m3fn``, ``float16``, or ``bfloat16``.
    gemm1_weights : torch.Tensor
        ``[num_experts, M, hidden_size]`` first-layer weights.  ``M`` is
        ``2 * intermediate_size`` for gated activations and ``intermediate_size``
        for non-gated activations.
    output1_scales_scalar : torch.Tensor
        ``[local_num_experts]`` first-layer output scales.
    output1_scales_gate_scalar : torch.Tensor
        ``[local_num_experts]`` first-layer gate scales.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` second-layer weights.
    output2_scales_scalar : torch.Tensor
        ``[local_num_experts]`` second-layer output scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    use_routing_scales_on_input : bool
        Whether to use routing scales on input.
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.  ``None`` (default)
        lets the runtime auto-select on SM90+.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    activation_type : int
        Activation type (default ``3`` — Swiglu).  ``0`` Gelu; ``3`` Swiglu;
        ``4`` Geglu; ``6`` Relu2; ``9`` Identity.
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities (default ``True``).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.  Column
        order matches ``topk_indices``.  When ``None`` (default) the
        kernel skips the write entirely.  The buffer may be larger than
        ``num_tokens`` for CUDA-graph pre-allocation; only rows
        ``[0, num_tokens)`` are written.
    output : Optional[torch.Tensor]
        Optional in-place output tensor of shape ``[seq_len, hidden_size]``.
        Allocated internally when ``None`` (default).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Final MoE output when ``do_finalize`` is ``True``, otherwise
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
    """
    _validate_routing_replay_out(routing_replay_out, top_k)
    result = get_trtllm_moe_sm100_module().trtllm_fp8_per_tensor_scale_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        gemm1_weights,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights,
        output2_scales_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        use_routing_scales_on_input,
        routing_method_type,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        norm_topk_prob,
        routing_replay_out,
        output,
    )

    if do_finalize:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api(trace=trtllm_fp8_per_tensor_scale_routed_moe_trace)
def trtllm_fp8_per_tensor_scale_routed_moe(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    routing_replay_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""Pre-routed FP8 per-tensor-scale MoE operation.

    Like :func:`trtllm_fp8_per_tensor_scale_moe`, but consumes pre-computed
    routing instead of routing logits, either as a packed ``(expert_id, weight)``
    tensor or as a ``(topk_ids, topk_weights)`` pair. Use this entry point for
    distributed MoE where routing (top-k selection, including EPLB
    redundant-expert placement) happens in an external DP/EP dispatch, or for
    CUDA-graph capture (avoids the CPU-GPU sync from logits processing).

    Parameters
    ----------
    topk_ids : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``[seq_len, top_k]`` int32 tensor of packed expert indices and weights
        with format ``(expert_id << 16) | (weight_bf16.view(int16))``.
        Alternatively a ``(topk_ids, topk_weights)`` pair of plain ``int32``
        indices and ``bfloat16`` or ``float32`` weights.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` tensor of routing bias (may be ``None``).
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states.
    gemm1_weights : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size]`` first-layer weights.
    output1_scales_scalar : torch.Tensor
        ``[local_num_experts]`` first-layer output scales.
    output1_scales_gate_scalar : torch.Tensor
        ``[local_num_experts]`` first-layer gate scales.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` second-layer weights.
    output2_scales_scalar : torch.Tensor
        ``[local_num_experts]`` second-layer output scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    use_routing_scales_on_input : bool
        Whether to use routing scales on input (Llama4-style).
    routing_method_type : int
        Routing method (default ``0``).  Matches
        :class:`flashinfer.tllm_enums.RoutingMethodType`; see
        :func:`trtllm_fp8_per_tensor_scale_moe` for the full list.
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.  ``None`` (default)
        lets the runtime auto-select on SM90+.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    activation_type : int
        Activation type (default ``3`` — Swiglu).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.
    output : Optional[torch.Tensor]
        Optional in-place output tensor of shape ``[seq_len, hidden_size]``.
        Allocated internally when ``None`` (default).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Final MoE output when ``do_finalize`` is ``True``, otherwise
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
    """
    _validate_routing_replay_out(routing_replay_out, top_k)
    topk_ids_tensor, topk_weights, routing_mode = _split_precomputed_routing(topk_ids)
    result = get_trtllm_moe_sm100_module().trtllm_fp8_per_tensor_scale_routed_moe(
        routing_mode,
        topk_ids_tensor,
        topk_weights,
        routing_bias,
        hidden_states,
        gemm1_weights,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights,
        output2_scales_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        use_routing_scales_on_input,
        routing_method_type,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        routing_replay_out,
        output,
    )

    if do_finalize:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api
def trtllm_fp8_per_channel_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """FP8 per-channel scale MoE operation.

    Args:
        routing_logits: [seq_len, num_experts] tensor of routing logits
        routing_bias: [num_experts] tensor of routing bias
        hidden_states: [seq_len, hidden_size] tensor of input hidden states
        hidden_states_scale: [seq_len, 1] FP32 per-token dequantization multipliers
        gemm1_weights: [num_experts, M, hidden_size] FP8 first layer weights,
            where M is 2*intermediate_size for gated activations and
            intermediate_size otherwise
        gemm1_per_channel_weight_scale: [local_num_experts, M] per-channel
            weight dequantization multipliers for gemm1, in the same shuffled row
            order as gemm1_weights
        output1_scale_scalar: [local_num_experts] per-expert output scales for gemm1
        output1_scale_gate_scalar: [local_num_experts] per-expert gate scales for gemm1
        gemm2_weights: [num_experts, hidden_size, intermediate_size] FP8 second layer weights
        gemm2_per_channel_weight_scale: [local_num_experts, hidden_size]
            per-channel dequantization multipliers for gemm2, in the same shuffled
            row order as gemm2_weights
        output2_scale_scalar: [local_num_experts] per-expert output scales for gemm2
        num_experts: Total number of experts
        top_k: Number of experts to route to per token
        n_group: Number of expert groups
        topk_group: Number of groups to consider for top-k routing
        intermediate_size: Size of intermediate layer
        local_expert_offset: Offset of local experts in global expert space
        local_num_experts: Number of experts handled by this device
        routed_scaling_factor: Scaling factor for routing
        use_routing_scales_on_input: Whether to use routing scales on input
        routing_method_type: Type of routing method to use (default: 0)
        do_finalize: Whether to finalize the output (default: True).
        enable_pdl: Whether to enable Programmatic Dependent Launch (PDL). Auto-enabled for >= sm90.
        tune_max_num_tokens: Maximum number of tokens for tuning. (default: 8192)
        activation_type: Type of activation function (default: 3 - Swiglu)
        norm_topk_prob: Whether to normalize the top-k probabilities (default: True)

    Returns:
        when do_finalize=True, returns the final MoE output.
        otherwise, returns the intermediate results (gemm2_output, expert_weights, expanded_idx_to_permuted_idx).
    """
    result = get_trtllm_moe_sm100_module().trtllm_fp8_per_channel_scale_moe(
        routing_logits,
        None,
        None,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_per_channel_weight_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        gemm2_weights,
        gemm2_per_channel_weight_scale,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        use_routing_scales_on_input,
        routing_method_type,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        norm_topk_prob,
    )

    if do_finalize:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api
def trtllm_fp8_per_channel_scale_routed_moe(
    topk_ids: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_per_channel_weight_scale: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_per_channel_weight_scale: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool = False,
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""FP8 per-token activation/per-channel weight MoE with pre-computed routing.

    Parameters
    ----------
    topk_ids : torch.Tensor
        ``[seq_len, top_k]`` int32 tensor of packed expert indices and weights
        with format ``(expert_id << 16) | (weight_bf16.view(int16))``.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` tensor of routing bias. May be ``None``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states.
    hidden_states_scale : torch.Tensor
        ``[seq_len, 1]`` FP32 per-token dequantization multipliers.
    gemm1_weights : torch.Tensor
        ``[num_experts, M, hidden_size]`` FP8 first-layer weights, where ``M`` is
        ``2 * intermediate_size`` for gated activations and ``intermediate_size``
        otherwise.
    gemm1_per_channel_weight_scale : torch.Tensor
        ``[local_num_experts, M]`` per-channel weight dequantization multipliers
        for GEMM1, in the same shuffled row order as ``gemm1_weights``.
    output1_scale_scalar : torch.Tensor
        ``[local_num_experts]`` per-expert output scales for GEMM1.
    output1_scale_gate_scalar : torch.Tensor
        ``[local_num_experts]`` per-expert gate scales for GEMM1.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` FP8 second-layer weights.
    gemm2_per_channel_weight_scale : torch.Tensor
        ``[local_num_experts, hidden_size]`` per-channel dequantization
        multipliers for GEMM2, in the same shuffled row order as ``gemm2_weights``.
    output2_scale_scalar : torch.Tensor
        ``[local_num_experts]`` per-expert output scales for GEMM2.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    use_routing_scales_on_input : bool
        Whether to apply routing scales to the input (default ``False``).
    routing_method_type : int
        Routing method (default ``0``). Matches
        :class:`flashinfer.tllm_enums.RoutingMethodType`.
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch. ``None`` lets the
        runtime auto-select on SM90+.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    activation_type : int
        Activation type (default ``3`` — Swiglu).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Final MoE output when ``do_finalize`` is ``True``; otherwise
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
    """
    result = get_trtllm_moe_sm100_module().trtllm_fp8_per_channel_scale_moe(
        None,  # routing_logits
        topk_ids,
        None,  # expert_weights
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_per_channel_weight_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        gemm2_weights,
        gemm2_per_channel_weight_scale,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        use_routing_scales_on_input,
        routing_method_type,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        True,  # norm_topk_prob: not used for pre-computed routing
    )

    if do_finalize:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api(trace=trtllm_fp8_block_scale_moe_trace_dispatch)
def trtllm_fp8_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
    num_fused_shared_experts: Optional[int] = None,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""FP8 block-scaled MoE operation.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` tensor of routing logits, ``float32`` or
        ``bfloat16``.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` tensor of routing bias, ``bfloat16`` or
        ``float32``.  May be ``None``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states.
        ``float16``, ``bfloat16``, or ``float8_e4m3fn`` (block scale must
        match: see ``hidden_states_scale``).
    hidden_states_scale : torch.Tensor
        ``[hidden_size // 128, seq_len]`` tensor of hidden-states block scales.
    gemm1_weights : torch.Tensor
        First-layer weights.  ``[num_experts, M, hidden_size]`` when
        ``weight_layout == WeightLayout.MajorK`` (``0``), or
        ``[num_experts, M // 128, hidden_size, 128]`` when
        ``weight_layout == WeightLayout.BlockMajorK`` (``2``).  ``M`` is
        ``2 * intermediate_size`` for gated activations and
        ``intermediate_size`` for non-gated activations.
    gemm1_weights_scale : torch.Tensor
        ``[num_experts, 2*intermediate_size // (32 if mxfp8 else 128), hidden_size // (32 if mxfp8 else 128)]``
        first-layer block scales.
    gemm2_weights : torch.Tensor
        Second-layer weights.  ``[num_experts, hidden_size, intermediate_size]``
        when ``weight_layout == WeightLayout.MajorK``, or
        ``[num_experts, hidden_size // 128, intermediate_size, 128]`` when
        ``weight_layout == WeightLayout.BlockMajorK``.
    gemm2_weights_scale : torch.Tensor
        ``[num_experts, hidden_size // (32 if mxfp8 else 128), intermediate_size // (32 if mxfp8 else 128)]``
        second-layer block scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    routing_method_type : int
        Routing method (default ``0``).  See :func:`trtllm_bf16_moe`.
    use_shuffled_weight : bool
        Whether to use the shuffled weight layout (default ``False``).
    weight_layout : int
        Weight layout for ``gemm1_weights`` / ``gemm2_weights``; matches
        :class:`flashinfer.tllm_enums.WeightLayout`.  Allowed values for
        this function depend on ``fp8_quantization_type``: ``DeepSeekFp8``
        accepts ``MajorK`` or ``BlockMajorK``; ``MxFp8`` requires
        ``MajorK``.  Default ``0`` (``MajorK``).

        - ``0`` ``MajorK`` — K-major, logical shape ``[Mn, K]``.
        - ``1`` ``MajorMn`` — M-major (A) / N-major (B), logical shape
          ``[K, Mn]``.  *Not supported by this function.*
        - ``2`` ``BlockMajorK`` — Blocked along K, logical shape
          ``[K / blockK, Mn, blockK]`` (``blockK`` is fixed at 128 B).
          *Only valid when ``fp8_quantization_type`` is ``DeepSeekFp8``.*
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.  ``None`` (default)
        lets the runtime auto-select on SM90+.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    fp8_quantization_type : Fp8QuantizationType
        FP8 quantization scheme (default ``Fp8QuantizationType.DeepSeekFp8``).
    num_fused_shared_experts : Optional[int]
        Number of shared experts to fuse into the MoE kernel (default
        ``None`` / ``0``).  When ``> 0``, the weight tensors must have
        ``num_experts + num_fused_shared_experts`` in the expert dimension.
        Expert parallelism (EP) is not yet supported together with fused shared
        experts: when this is ``> 0`` you must pass ``local_expert_offset == 0``
        and ``local_num_experts == num_experts`` (all routed experts local),
        otherwise a ``ValueError`` is raised.
    activation_type : int
        Activation type (default ``3`` — Swiglu).  ``3`` Swiglu; ``4`` Geglu;
        ``6`` Relu2; ``9`` Identity.
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities (default ``True``).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.  Column order
        matches ``topk_indices``.  When ``None`` (default) the kernel skips
        the write entirely.  The buffer may be larger than ``num_tokens`` for
        CUDA-graph pre-allocation; only rows ``[0, num_tokens)`` are written.
    gemm1_alpha : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 per-expert SwiGLU OA alpha
        parameter.  Supported for ``Fp8QuantizationType.MxFp8`` and
        ``Fp8QuantizationType.DeepSeekFp8`` with ``ActivationType.Swiglu``.  Any
        subset of ``gemm1_alpha``, ``gemm1_beta``, ``gemm1_clamp_limit``
        can be provided independently.  When ``None`` (default),
        ``alpha=1.0`` is used.  Let GEMM1 output be split as ``X1``
        (linear/up half) and ``X2`` (gate half).  The activation
        output is ``X2 * sigmoid(alpha * X2) * (X1 + beta)``.  Pass raw
        values; neither block-scale recipe carries a scalar dequant scale, so
        no host-side conversion is applied.
    gemm1_beta : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 per-expert SwiGLU OA beta
        parameter.  Supported for ``Fp8QuantizationType.MxFp8`` and
        ``Fp8QuantizationType.DeepSeekFp8`` with ``ActivationType.Swiglu``.
        When ``None`` (default), ``beta=0.0`` is used.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 per-expert clamp limit.
        Supported for ``Fp8QuantizationType.MxFp8`` and
        ``Fp8QuantizationType.DeepSeekFp8`` with ``ActivationType.Swiglu``.
        When provided, ``X1 = clamp(X1, -limit, limit)`` and
        ``X2 = clamp(X2, max=limit)``.  When ``None`` (default), no clamp
        is applied.
    output : Optional[torch.Tensor]
        Optional in-place output tensor of shape ``[seq_len, hidden_size]``.
        Allocated internally when ``None`` (default).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Final MoE output when ``do_finalize`` is ``True``, otherwise
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
    """
    # Fused shared experts do not yet support expert parallelism (EP). The routing
    # kernel maps a shared expert's global id (num_experts + k) to a weight row as
    # (global_id - local_expert_offset), which only lands at the intended local slot
    # when local_expert_offset == 0 and local_num_experts == num_experts. Reject EP
    # configurations explicitly instead of silently producing wrong results.
    nfse = num_fused_shared_experts or 0
    if nfse > 0 and (local_expert_offset != 0 or local_num_experts != num_experts):
        raise ValueError(
            "Fused shared experts (num_fused_shared_experts > 0) do not yet support "
            "expert parallelism: require local_expert_offset == 0 and "
            "local_num_experts == num_experts. Got "
            f"num_fused_shared_experts={nfse}, local_expert_offset={local_expert_offset}, "
            f"local_num_experts={local_num_experts}, num_experts={num_experts}."
        )
    # Only the DeepSeekV3 routing path implements fused shared experts.
    if nfse > 0 and routing_method_type != RoutingMethodType.DeepSeekV3:
        raise ValueError(
            "Fused shared experts (num_fused_shared_experts > 0) are only supported "
            f"with DeepSeekV3 routing; got routing_method_type={routing_method_type}."
        )
    _validate_routing_replay_out(routing_replay_out, top_k, nfse)
    _validate_fp8_block_scale_gemm1_activation_params(
        fp8_quantization_type,
        activation_type,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
    )
    result = get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        RoutingInputMode.FromLogits,
        routing_logits,
        None,  # topk_ids - will be computed from routing_logits
        None,  # expert_weights - will be computed from routing_logits
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        None,  # gemm1_lora_delta — LoRA only supported with routed API
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        fp8_quantization_type,
        num_fused_shared_experts if num_fused_shared_experts is not None else 0,
        activation_type,
        norm_topk_prob,
        routing_replay_out,
    )

    if do_finalize:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api(trace=trtllm_fp8_block_scale_routed_moe_trace)
def trtllm_fp8_block_scale_routed_moe(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
    activation_type: int = ActivationType.Swiglu.value,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""Pre-routed FP8 block-scaled MoE operation.

    Like :func:`trtllm_fp8_block_scale_moe`, but consumes pre-computed routing
    instead of routing logits, either as a packed ``(expert_id, weight)``
    tensor or as a ``(topk_ids, topk_weights)`` pair.  Use this entry
    point for CUDA-graph capture (avoids the CPU-GPU sync from logits
    processing) or distributed MoE where routing happens elsewhere.

    Parameters
    ----------
    topk_ids : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        ``[seq_len, top_k]`` int32 tensor of packed expert indices and weights
        with format ``(expert_id << 16) | (weight_bf16.view(int16))``.
        Alternatively a ``(topk_ids, topk_weights)`` pair of plain ``int32``
        indices and ``bfloat16`` or ``float32`` weights.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` tensor of routing bias, ``bfloat16`` or
        ``float32``.  May be ``None``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` tensor of input hidden states.
        ``float16``, ``bfloat16``, or ``float8_e4m3fn`` (block scale must
        match: see ``hidden_states_scale``).
    hidden_states_scale : torch.Tensor
        ``[hidden_size // (32 if mxfp8 else 128), seq_len]`` block scales for
        the hidden states.
    gemm1_weights : torch.Tensor
        ``[num_experts, M, hidden_size]`` first-layer weights where ``M`` is
        ``2 * intermediate_size`` for gated activations and
        ``intermediate_size`` for non-gated.
    gemm1_weights_scale : torch.Tensor
        ``[num_experts, 2*intermediate_size // (32 if mxfp8 else 128), hidden_size // (32 if mxfp8 else 128)]``
        first-layer block scales.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` second-layer weights.
    gemm2_weights_scale : torch.Tensor
        ``[num_experts, hidden_size // (32 if mxfp8 else 128), intermediate_size // (32 if mxfp8 else 128)]``
        second-layer block scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    use_shuffled_weight : bool
        Whether to use the shuffled weight layout (default ``False``).
    weight_layout : int
        Weight layout for ``gemm1_weights`` / ``gemm2_weights``; matches
        :class:`flashinfer.tllm_enums.WeightLayout`.  Allowed values for
        this function depend on ``fp8_quantization_type``: ``DeepSeekFp8``
        accepts ``MajorK`` or ``BlockMajorK``; ``MxFp8`` requires
        ``MajorK``.  Default ``0`` (``MajorK``).

        - ``0`` ``MajorK`` — K-major, logical shape ``[Mn, K]``.
        - ``1`` ``MajorMn`` — M-major (A) / N-major (B), logical shape
          ``[K, Mn]``.  *Not supported by this function.*
        - ``2`` ``BlockMajorK`` — Blocked along K, logical shape
          ``[K / blockK, Mn, blockK]`` (``blockK`` is fixed at 128 B).
          *Only valid when ``fp8_quantization_type`` is ``DeepSeekFp8``.*
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.  ``None`` (default)
        lets the runtime auto-select on SM90+.
    gemm1_lora_delta : Optional[torch.Tensor]
        Optional MoE LoRA delta of shape
        ``[num_tokens, top_k, 2 * intermediate_size]``, ``bfloat16``.  When
        set for MXFP8 it is added to FC1 before the fused gated activation and
        the post-activation FC1 output is appended to the return list.
    output : Optional[torch.Tensor]
        Optional in-place output tensor of shape ``[seq_len, hidden_size]``.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    fp8_quantization_type : Fp8QuantizationType
        FP8 quantization scheme (default ``Fp8QuantizationType.DeepSeekFp8``).
    activation_type : int
        Activation type (default ``3`` — Swiglu).  ``3`` Swiglu; ``4`` Geglu;
        ``6`` Relu2; ``9`` Identity.
    gemm1_alpha : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 per-expert SwiGLU OA alpha
        parameter.  Supported for ``Fp8QuantizationType.MxFp8`` and
        ``Fp8QuantizationType.DeepSeekFp8`` with ``ActivationType.Swiglu``.  Any
        subset of ``gemm1_alpha``, ``gemm1_beta``, ``gemm1_clamp_limit``
        can be provided independently.  When ``None`` (default),
        ``alpha=1.0`` is used.  Let GEMM1 output be split as ``X1``
        (linear/up half) and ``X2`` (gate half).  The activation
        output is ``X2 * sigmoid(alpha * X2) * (X1 + beta)``.  Pass raw
        values; neither block-scale recipe carries a scalar dequant scale, so
        no host-side conversion is applied.
    gemm1_beta : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 per-expert SwiGLU OA beta
        parameter.  Supported for ``Fp8QuantizationType.MxFp8`` and
        ``Fp8QuantizationType.DeepSeekFp8`` with ``ActivationType.Swiglu``.
        When ``None`` (default), ``beta=0.0`` is used.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional ``[local_num_experts]`` float32 per-expert clamp limit.
        Supported for ``Fp8QuantizationType.MxFp8`` and
        ``Fp8QuantizationType.DeepSeekFp8`` with ``ActivationType.Swiglu``.
        When provided, ``X1 = clamp(X1, -limit, limit)`` and
        ``X2 = clamp(X2, max=limit)``.  When ``None`` (default), no clamp
        is applied.

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Return shape depends on ``do_finalize`` and ``gemm1_lora_delta``;
        see :func:`trtllm_bf16_routed_moe` for the table.
    """
    _validate_fp8_block_scale_gemm1_activation_params(
        fp8_quantization_type,
        activation_type,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
    )
    topk_ids_tensor, topk_weights, routing_mode = _split_precomputed_routing(topk_ids)

    result = get_trtllm_moe_sm100_module().trtllm_fp8_block_scale_moe(
        routing_mode,
        None,  # routing_logits
        topk_ids_tensor,
        topk_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_lora_delta,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        output,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        do_finalize,
        enable_pdl,
        tune_max_num_tokens,
        fp8_quantization_type,
        0,  # num_fused_shared_experts: not supported on the pre-routed path
        activation_type,
        True,  # norm_topk_prob: not used for pre-computed routing
    )

    if do_finalize and gemm1_lora_delta is None:
        logger.warning_once(
            "the single torch.Tensor return type is deprecated and will be replaced with List[torch.Tensor] in the v0.8.0."
        )
        return result[0]
    else:
        return result


@flashinfer_api(trace=trtllm_fp4_block_scale_moe_trace_dispatch)
def trtllm_fp4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = ActivationType.Swiglu.value,
    per_token_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    num_fused_shared_experts: Optional[int] = None,
) -> List[torch.Tensor]:
    r"""FP4 block-scaled MoE operation.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` tensor of routing logits.  ``float32`` or
        ``bfloat16``.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` tensor of routing bias, ``bfloat16`` or
        ``float32`` (independent of ``routing_logits``'s dtype).  May be
        ``None``.
    hidden_states : torch.Tensor
        Hidden states of shape ``[seq_len, hidden_size // 2]`` (NVFP4) or
        ``[seq_len, hidden_size]`` (MXFP8 / bfloat16).  Supports bfloat16,
        MXFP8, and NVFP4 (packed into uint8).
    hidden_states_scale : Optional[torch.Tensor]
        Block scales for MXFP8 / NVFP4 hidden states of shape
        ``[seq_len, hidden_size // (32 if mxfp8 else 16)]``.  Dtype is float8.
    gemm1_weights : torch.Tensor
        ``[num_experts, M, hidden_size // 2]`` packed FP4 FC1 weights, dtype
        ``uint8``.  ``M`` is ``2 * intermediate_size`` for gated activations and
        ``intermediate_size`` for non-gated ones (``6`` Relu2, ``9`` Identity).
    gemm1_weights_scale : torch.Tensor
        ``[num_experts, M, hidden_size // (32 if mxfp4 else 16)]`` FC1 weight
        block scales, dtype float8, with the same ``M`` as ``gemm1_weights``.
    gemm1_bias : Optional[torch.Tensor]
        ``[num_experts, M]`` FC1 bias, ``float32``, with the same ``M`` as
        ``gemm1_weights``.
    gemm1_alpha : Optional[torch.Tensor]
        ``[num_experts]`` swiglu alpha, ``float32``.
        For SiTU this is ``[local_num_experts]``, finite and positive;
        ``None`` materializes per-expert ``alpha=1``.

    gemm1_beta : Optional[torch.Tensor]
        ``[num_experts]`` swiglu beta, ``float32``.
        For SiTU this is ``[local_num_experts]``, finite and positive;
        ``None`` materializes per-expert ``beta=1``.
    gemm1_clamp_limit : Optional[torch.Tensor]
        ``[num_experts]`` swiglu clamp limit, ``float32``.
        For SiTU a provided limit is per-local-expert, finite, and positive;
        it clamps ``x0`` to ``[-limit, limit]`` and ``x1`` from above.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` packed FP4 FC2
        weights, dtype ``uint8``.
    gemm2_weights_scale : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size // (32 if mxfp4 else 16)]``
        FC2 weight block scales, dtype float8.
    gemm2_bias : Optional[torch.Tensor]
        ``[num_experts, hidden_size]`` FC2 bias, ``float32``.
    output1_scale_scalar : Optional[torch.Tensor]
        ``[local_num_experts]`` scaling factors for the first-layer activation
        output.
    output1_scale_gate_scalar : Optional[torch.Tensor]
        ``[local_num_experts]`` scaling factors for the first-layer gate
        output.
    output2_scale_scalar : Optional[torch.Tensor]
        ``[local_num_experts]`` scaling factors for the second-layer output.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.
    activation_type : int
        Activation type (default ``3`` — Swiglu).  ``3`` Swiglu; ``4`` Geglu;
        ``6`` Relu2; ``9`` Identity.
        ``10`` SiTU uses ``beta*tanh(x0/beta) * alpha*tanh(x1/alpha)*sigmoid(x1)``.
    per_token_scale : Optional[torch.Tensor]
        ``[seq_len]`` per-token scaling factors, ``float32``.
    output : Optional[torch.Tensor]
        Optional in-place ``[seq_len, hidden_size]`` output tensor.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities (default ``True``).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.  Column
        order matches ``topk_indices``.  When ``None`` (default) the
        kernel skips the write entirely.  The buffer may be larger than
        ``num_tokens`` for CUDA-graph pre-allocation; only rows
        ``[0, num_tokens)`` are written.
    num_fused_shared_experts : Optional[int]
        Number of shared experts to fuse into the MoE kernel (default
        ``None`` / ``0``).  When ``> 0``, every per-expert tensor
        (``gemm1_weights``, ``gemm1_weights_scale``, ``gemm2_weights``,
        ``gemm2_weights_scale``, ``output*_scale_scalar``, biases) must have
        ``num_experts + num_fused_shared_experts`` rows in the expert
        dimension — the shared-expert weights are appended after the routed
        ones.  Every token is unconditionally routed to the shared experts
        with weight ``1.0``. With ``do_finalize=False``, the returned
        ``expert_weights`` and ``expanded_idx_to_permuted_idx`` cover
        ``top_k + num_fused_shared_experts`` slots per token.

    Returns
    -------
    List[torch.Tensor]
        ``[output]`` when ``do_finalize`` is ``True``, otherwise
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
        The ``expert_weights`` tensor is always ``bfloat16`` (the routing
        kernel emits bf16 weights for every routing method), regardless of
        the ``routing_logits`` dtype — including the ``do_finalize=False``
        path and fp32 ``DeepSeekV3`` logits.
    """
    # Fused shared experts do not yet support expert parallelism (EP). The routing
    # kernel maps a shared expert's global id (num_experts + k) to a weight row as
    # (global_id - local_expert_offset), which only lands at the intended local slot
    # when local_expert_offset == 0 and local_num_experts == num_experts. Reject EP
    # configurations explicitly instead of silently producing wrong results.
    nsfe = num_fused_shared_experts or 0
    if nsfe > 0 and (local_expert_offset != 0 or local_num_experts != num_experts):
        raise ValueError(
            "Fused shared experts (num_fused_shared_experts > 0) do not yet support "
            "expert parallelism: require local_expert_offset == 0 and "
            "local_num_experts == num_experts. Got "
            f"num_fused_shared_experts={nsfe}, local_expert_offset={local_expert_offset}, "
            f"local_num_experts={local_num_experts}, num_experts={num_experts}."
        )
    # Only the DeepSeekV3 routing path implements fused shared experts.
    if nsfe > 0 and routing_method_type != RoutingMethodType.DeepSeekV3:
        raise ValueError(
            "Fused shared experts (num_fused_shared_experts > 0) are only supported "
            f"with DeepSeekV3 routing; got routing_method_type={routing_method_type}."
        )
    _validate_routing_replay_out(routing_replay_out, top_k, nsfe)
    return get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(
        RoutingInputMode.FromLogits,
        routing_logits,
        None,
        None,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        None,  # gemm1_lora_delta: not supported for the non-routed entry
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        per_token_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        activation_type,
        output,
        tune_max_num_tokens,
        norm_topk_prob,
        routing_replay_out,
        nsfe,
    )


@flashinfer_api(trace=trtllm_fp4_block_scale_routed_moe_trace)
def trtllm_fp4_block_scale_routed_moe(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = ActivationType.Swiglu.value,
    per_token_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """FP4 block scale MoE operation with pre-computed routing.

    This function supports two pre-computed routing formats:
    1. Packed format: ``topk_ids`` is a single int32 tensor with
       ``(expert_id << 16) | weight`` entries (high 16 bits = int16 expert
       id, low 16 bits = float16/bfloat16 weight, matching
       ``PackedScoreIdx`` in ``include/flashinfer/trtllm/fused_moe/RoutingKernel.h``).
    2. Unpacked format: ``topk_ids`` is a tuple ``(topk_ids, topk_weights)``.

    Parameters
    ----------
    topk_ids : Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
        Pre-computed routing decision.  Either a single int32 tensor of shape
        ``[seq_len, top_k]`` in packed format ``(expert_id << 16) | weight`` or
        a tuple ``(ids, weights)`` where ``ids`` is int32 of shape
        ``[seq_len, top_k]`` (plain expert indices) and ``weights`` is
        ``bfloat16`` or ``float32`` of the same shape (routing weights). The
        weights are consumed at their native dtype (no cast), so passing the
        ``float32`` weights emitted by typical routers is copy-free.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` routing bias, ``bfloat16`` or ``float32``.  May be
        ``None``.
    hidden_states : torch.Tensor
        Hidden states of shape ``[seq_len, hidden_size // 2]`` (NVFP4) or
        ``[seq_len, hidden_size]`` (MXFP8 / bfloat16).  Supports bfloat16,
        MXFP8 (``float8_e4m3fn``), and NVFP4 (packed into ``uint8``).
    hidden_states_scale : Optional[torch.Tensor]
        ``[seq_len, hidden_size // (32 if mxfp8 else 16)]`` block scales of
        the hidden states, float8.
    gemm1_weights : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size // 2]`` packed
        FP4 FC1 weights, ``uint8``.
    gemm1_weights_scale : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size // (32 if mxfp4 else 16)]``
        FC1 weight block scales, float8.
    gemm1_bias : Optional[torch.Tensor]
        ``[num_experts, 2 * intermediate_size]`` FC1 bias, float32.
    gemm1_alpha : Optional[torch.Tensor]
        ``[num_experts]`` swiglu alpha, float32.
        For SiTU this is ``[local_num_experts]``, finite and positive;
        ``None`` materializes per-expert ``alpha=1``.

    gemm1_beta : Optional[torch.Tensor]
        ``[num_experts]`` swiglu beta, float32.
        For SiTU this is ``[local_num_experts]``, finite and positive;
        ``None`` materializes per-expert ``beta=1``.
    gemm1_clamp_limit : Optional[torch.Tensor]
        ``[num_experts]`` swiglu clamp limit, float32.
        For SiTU a provided limit is per-local-expert, finite, and positive;
        it clamps ``x0`` to ``[-limit, limit]`` and ``x1`` from above.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` packed FP4 FC2
        weights, ``uint8``.
    gemm2_weights_scale : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size // (32 if mxfp4 else 16)]``
        FC2 weight block scales, float8.
    gemm2_bias : Optional[torch.Tensor]
        ``[num_experts, hidden_size]`` FC2 bias, float32.
    output1_scale_scalar : Optional[torch.Tensor]
        ``[local_num_experts]`` scaling factors for the first-layer activation
        output.
    output1_scale_gate_scalar : Optional[torch.Tensor]
        ``[local_num_experts]`` scaling factors for the first-layer gate
        output.
    output2_scale_scalar : Optional[torch.Tensor]
        ``[local_num_experts]`` scaling factors for the second-layer output.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.
    activation_type : int
        Activation type (default ``3`` — Swiglu).
        ``10`` SiTU uses ``beta*tanh(x0/beta) * alpha*tanh(x1/alpha)*sigmoid(x1)``.
    per_token_scale : Optional[torch.Tensor]
        ``[seq_len]`` per-token scaling factors, float32.
    output : Optional[torch.Tensor]
        Optional in-place ``[seq_len, hidden_size]`` output tensor.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    gemm1_lora_delta : Optional[torch.Tensor]
        Optional MoE LoRA delta of shape
        ``[num_tokens, top_k, 2 * intermediate_size]``, ``bfloat16``.  When
        set it is added to FC1 before the fused gated activation and the
        post-activation FC1 output is appended to the return list.

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Return shape depends on ``do_finalize`` and ``gemm1_lora_delta``;
        see :func:`trtllm_bf16_routed_moe` for the table.
    """
    topk_ids_tensor, topk_weights, routing_mode = _split_precomputed_routing(topk_ids)

    # The kernel folds dequantScaleAb into scaleC and applies it to the bias
    # when the input is Fp8 or NvFp4 and DeepSeekFp8 is not used (see trtllm-gen
    # getDoesScaleAb()); pre-divide lora_delta to compensate.
    if (
        gemm1_lora_delta is not None
        and output1_scale_gate_scalar is not None
        and hidden_states.dtype == torch.uint8
    ):
        if routing_mode == RoutingInputMode.UnpackedPrecomputed:
            # topk_ids_tensor: [num_tokens, top_k] int32 of plain expert IDs.
            expert_idx = topk_ids_tensor.to(torch.int64)
        else:
            # Packed format: high 16 bits = expert_id, low 16 bits = packed weight.
            expert_idx = (topk_ids_tensor.to(torch.int32) >> 16).to(torch.int64)
        # topk_ids carry GLOBAL expert ids, but output1_scale_gate_scalar is
        # [local_num_experts]. Convert to the local row (global - offset).
        local_idx = (expert_idx - local_expert_offset).clamp(0, local_num_experts - 1)
        inv_dequant_ab = (1.0 / output1_scale_gate_scalar.to(torch.float32))[local_idx]
        gemm1_lora_delta = (
            gemm1_lora_delta.to(torch.float32) * inv_dequant_ab[..., None]
        ).to(gemm1_lora_delta.dtype)

    return get_trtllm_moe_sm100_module().trtllm_fp4_block_scale_moe(
        routing_mode,
        None,
        topk_ids_tensor,
        topk_weights,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_lora_delta,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        per_token_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        activation_type,
        output,
        tune_max_num_tokens,
        True,  # norm_topk_prob: not used for pre-computed routing
        None,  # routing_replay_out: not used for pre-computed routing
        0,  # num_fused_shared_experts: not used for pre-computed routing
    )


@flashinfer_api(trace=trtllm_mxint4_block_scale_moe_trace)
def trtllm_mxint4_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    r"""MXINT4 block-scaled MoE operation.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` routing logits, ``float32`` or ``bfloat16``.
    routing_bias : Optional[torch.Tensor]
        ``[num_experts]`` routing bias, ``bfloat16`` or ``float32``.  May be
        ``None``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` input hidden states, ``bfloat16``.
    gemm1_weights : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size // 2]`` packed
        MXINT4 FC1 weights, ``uint8``.
    gemm1_weights_scale : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size // 32]`` FC1
        weight block scales, ``bfloat16``.
    gemm1_alpha : Optional[torch.Tensor]
        ``[num_experts]`` swiglu alpha, ``float32``.
    gemm1_beta : Optional[torch.Tensor]
        ``[num_experts]`` swiglu beta, ``float32``.
    gemm1_clamp_limit : Optional[torch.Tensor]
        ``[num_experts]`` swiglu clamp limit, ``float32``.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size]`` packed MXINT4 FC2
        weights, ``uint8``.
    gemm2_weights_scale : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size // 32]`` FC2 weight
        block scales, ``bfloat16``.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        Size of the intermediate layer.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Scaling factor for routing.
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    do_finalize : bool
        Whether to finalize the output (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.
    output : Optional[torch.Tensor]
        Optional in-place ``[seq_len, hidden_size]`` output tensor.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).
    norm_topk_prob : bool
        Whether to normalize the top-k probabilities (default ``True``).
    routing_replay_out : Optional[torch.Tensor]
        Optional ``int16`` tensor of shape ``(num_tokens_or_larger, top_k)``
        used to capture the selected expert IDs during routing.  Column
        order matches ``topk_indices``.  When ``None`` (default) the
        kernel skips the write entirely.  The buffer may be larger than
        ``num_tokens`` for CUDA-graph pre-allocation; only rows
        ``[0, num_tokens)`` are written.

    Returns
    -------
    List[torch.Tensor]
        ``[output]`` when ``do_finalize`` is ``True``, otherwise
        ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``.
    """
    _validate_routing_replay_out(routing_replay_out, top_k)
    return get_trtllm_moe_sm100_module().trtllm_mxint4_block_scale_moe(
        routing_logits,
        routing_bias,
        None,  # topk_ids
        None,  # expert_weights
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        None,  # gemm1_lora_delta — LoRA only supported with routed API to enforce consistent routing behavior
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        output,
        tune_max_num_tokens,
        norm_topk_prob,
        routing_replay_out,
    )


@flashinfer_api
def trtllm_mxint4_block_scale_routed_moe(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
) -> List[torch.Tensor]:
    """MxInt4 block-scale MoE with pre-computed routing.

    Same FC1/FC2 kernel and LoRA contract as :func:`trtllm_mxint4_block_scale_moe`,
    but the caller supplies pre-computed top-k routing instead of raw routing
    logits.  This skips the routing kernel's top-k computation and reuses the
    BF16-routed packed-int32 contract for ``topk_ids``.

    Parameters
    ----------
    topk_ids : torch.Tensor
        ``[seq_len, top_k]`` int32 tensor of packed expert indices and
        weights: ``(expert_id << 16) | (weight_bf16.view(int16))``.
    hidden_states : torch.Tensor
        ``[seq_len, hidden_size]`` ``bfloat16`` input activations.
    gemm1_weights : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size // 2]`` packed
        MXINT4 weights, ``uint8``.
    gemm1_weights_scale : torch.Tensor
        ``[num_experts, 2 * intermediate_size, hidden_size // 32]`` FC1
        weight scales, ``bfloat16``.
    gemm1_alpha : Optional[torch.Tensor]
        ``[num_experts]`` swiglu alpha, ``float32``.
    gemm1_beta : Optional[torch.Tensor]
        ``[num_experts]`` swiglu beta, ``float32``.
    gemm1_clamp_limit : Optional[torch.Tensor]
        ``[num_experts]`` swiglu clamp limit, ``float32``.
    gemm2_weights : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size // 2]`` packed
        MXINT4 weights, ``uint8``.
    gemm2_weights_scale : torch.Tensor
        ``[num_experts, hidden_size, intermediate_size // 32]`` FC2 weight
        scales, ``bfloat16``.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts to route to per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Number of groups to consider for top-k routing.
    intermediate_size : int
        FC1/FC2 inner dimension.
    local_expert_offset : int
        Offset of local experts in the global expert space.
    local_num_experts : int
        Number of experts handled by this device.
    routed_scaling_factor : Optional[float]
        Optional output scaling factor.
    routing_method_type : int
        Routing method (default ``0``).  Selects the routing-kernel
        pipeline; matches :class:`flashinfer.tllm_enums.RoutingMethodType`.

        - ``0`` ``Default`` — Softmax → TopK.
        - ``1`` ``Renormalize`` — TopK → Softmax.
        - ``2`` ``DeepSeekV3`` — Sigmoid → RoutingBiasAdd → Top-2 in group →
          Top-``topk_group`` groups → Top-``top_k`` experts from the
          selected groups.
        - ``3`` ``Llama4`` — Top-1 → Sigmoid.
        - ``4`` ``RenormalizeNaive`` — Softmax → TopK → Renormalize (Qwen3
          style).
        - ``5`` ``TopK`` — TopK only (no softmax/sigmoid).
        - ``6`` ``SigmoidRenorm`` — Sigmoid → TopK → Renormalize (divide by
          the sum of the top-K weights).
        - ``7`` ``MiniMax2`` — Sigmoid + Bias → TopK → ScaledSumNormalize
          (``routeScale = 1.0``, ``epsilon = 1e-20``).
        - ``8`` ``Sigmoid`` — Sigmoid → TopK (no renormalization).
        - ``9`` ``TopKSigmoid`` — TopK → Sigmoid (no renormalization).
        - ``10`` ``Unspecified`` — reserved.
    do_finalize : bool
        Whether to run the finalize stage (default ``True``).
    enable_pdl : Optional[bool]
        Whether to enable Programmatic Dependent Launch.
    gemm1_lora_delta : Optional[torch.Tensor]
        Optional MoE LoRA delta of shape
        ``[num_tokens, top_k, 2 * intermediate_size]``, ``bfloat16``, in
        concatenated gate/up layout.  When set, added to FC1 before SwiGLU
        and the post-activation buffer is appended to the return list.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.
    tune_max_num_tokens : int
        Maximum number of tokens for autotuning (default ``8192``).

    Returns
    -------
    List[torch.Tensor]
        Return shape depends on ``do_finalize`` and ``gemm1_lora_delta``.

        =============  ==================  =========================================================================
        do_finalize    gemm1_lora_delta    Returned tensors
        =============  ==================  =========================================================================
        ``True``       ``None``            ``[output]``
        ``True``       ``Tensor``          ``[output, expanded_idx_to_permuted_idx, gemm1_activation_output]``
        ``False``      ``None``            ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``
        ``False``      ``Tensor``          ``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx, gemm1_activation_output]``
        =============  ==================  =========================================================================
    """
    return get_trtllm_moe_sm100_module().trtllm_mxint4_block_scale_moe(
        None,  # routing_logits
        None,  # routing_bias
        topk_ids,
        None,  # expert_weights
        hidden_states,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm1_lora_delta,
        gemm2_weights,
        gemm2_weights_scale,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        do_finalize,
        enable_pdl,
        output,
        tune_max_num_tokens,
        True,  # norm_topk_prob: not used for pre-computed routing
    )
