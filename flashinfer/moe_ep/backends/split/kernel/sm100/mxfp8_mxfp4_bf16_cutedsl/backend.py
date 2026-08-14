"""CuTeDSL W4A8 split kernel — MXFP8-quantized dispatch output x MXFP4 weights.

Post-dispatch inner compute for the split EP path: the comm backend delivers
BF16 tokens, this backend MXFP8-quantizes them locally (linear block-32 UE8M0
scales) and runs the SM100 ``cute_dsl_fused_moe_mxfp8_mxfp4`` mixed-precision
kernel over this rank's MXFP4 expert shard.

Routing synthesis mirrors ``..fused_moe.bridge``: the LL EXPERT_MAJOR layout
pre-assigns rows to experts by position, so compute runs at ``top_k=1`` with
weight 1 (EP ``combine`` owns the real reweight + reduction); the RANK_MAJOR
and HIGH_THROUGHPUT layouts carry received routing, so compute runs at the
real ``top_k`` with non-local picks masked to weight 0 and ``combine`` just
sums per-rank partials.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from ......config import BootstrapConfig, EpAlgorithm, EpLayout, FleetParams
from ......core.kernel.base import SplitKernelBackend, SplitKernelContext
from ......core.validation.common import MoEEpConfigError, validate_mega_arch
from ......weights import MoEWeightPack
from .config import Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig
from .weights import TransformedSplitWeights, preprocess_split_weights

if TYPE_CHECKING:
    import torch

from ......core.kernel.registry import register_split_kernel

# Per-token row widths (bf16 elements) the nccl_ep LL dispatch device kernel
# accepts; anything else hits 'Unsupported hidden' (low_latency.cu). Probed
# empirically on nccl4py 0.3.1 (jobs 2390737/2390761): 3072 is REJECTED even
# though nixl_ep supports it, and sub-2048 widths are all rejected. The kernel
# also rejects 1-byte payload dtypes outright, hence the bf16 view below.
_NCCL_EP_LL_BF16_WIDTHS = (2048, 2560, 4096, 5120, 6144, 7168, 8192)


def packed_dispatch_width(hidden: int) -> int:
    """Smallest transport-supported bf16 send width holding the packed row.

    The packed row is ``hidden`` fp8 payload bytes + ``hidden/32`` UE8M0
    scale bytes, viewed as bf16 (so ``hidden`` must be divisible by 64 for an
    exact element count) and zero-padded up to a supported width.
    """
    if hidden % 64:
        raise ValueError(
            f"mxfp8_dispatch requires hidden % 64 == 0 for an exact bf16 "
            f"packed-row width, got {hidden}"
        )
    need = (hidden + hidden // 32) // 2
    for width in _NCCL_EP_LL_BF16_WIDTHS:
        if width >= need:
            return width
    raise ValueError(
        f"packed MXFP8 row for hidden={hidden} needs {need} bf16 elements, "
        f"above the largest transport-supported width "
        f"{_NCCL_EP_LL_BF16_WIDTHS[-1]}"
    )


@register_split_kernel("sm100_mxfp8_mxfp4_bf16_cutedsl")
class Mxfp8Mxfp4CutedslSplitKernelBackend(SplitKernelBackend):
    def __init__(self, config: Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig) -> None:
        super().__init__(config)
        if not isinstance(config, Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig):
            raise TypeError(
                "Mxfp8Mxfp4CutedslSplitKernelBackend expects "
                "Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_SplitConfig, got "
                f"{type(config).__name__}"
            )
        self._kernel_config = config
        self._rank: Optional[int] = None
        # One wrapper per top_k (EXPERT_MAJOR synthesizes top_k=1; RANK_MAJOR/
        # HT run the received top_k) — the wrapper bakes top_k in at build.
        self._wrappers: Dict[int, Any] = {}

    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_mxfp8_mxfp4_bf16_cutedsl"

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch()  # same SM100-family gate as the cutedsl mega kernels
        from flashinfer.cute_dsl import is_cute_dsl_available

        if not is_cute_dsl_available():
            raise MoEEpConfigError(
                "sm100_mxfp8_mxfp4_bf16_cutedsl requires the nvidia-cutlass-dsl "
                "package (CuTeDSL) to be importable."
            )
        if fleet_params.num_experts % bootstrap.world_size != 0:
            raise MoEEpConfigError(
                f"FleetParams.num_experts ({fleet_params.num_experts}) must be "
                f"divisible by world_size ({bootstrap.world_size})."
            )
        self._rank = bootstrap.rank

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedSplitWeights:
        transformed = preprocess_split_weights(weights)
        if fleet_params.num_experts % transformed.num_local_experts != 0:
            raise MoEEpConfigError(
                f"local expert count {transformed.num_local_experts} does not "
                f"divide FleetParams.num_experts ({fleet_params.num_experts}); "
                "each rank must own an equal shard."
            )
        if fleet_params.token_hidden_size != transformed.hidden_size:
            raise MoEEpConfigError(
                f"weight hidden size {transformed.hidden_size} != "
                f"FleetParams.token_hidden_size ({fleet_params.token_hidden_size})."
            )
        self._transformed_weights = transformed
        return transformed

    def _require_ready(self) -> TransformedSplitWeights:
        if self._transformed_weights is None:
            raise RuntimeError(
                "preprocess_weights() must run before compute() — the layer "
                "calls it at construction; direct users must call it explicitly."
            )
        if self._rank is None:
            raise RuntimeError(
                "validate_init() must run before compute() to bind the EP rank."
            )
        return self._transformed_weights

    def _ensure_wrapper(self, top_k: int, num_experts: int):
        wrapper = self._wrappers.get(top_k)
        if wrapper is None:
            from flashinfer.fused_moe.cute_dsl.fused_moe_mxfp8_mxfp4 import (
                CuteDslMxfp8Mxfp4MoEWrapper,
            )

            tw = self._transformed_weights
            wrapper = CuteDslMxfp8Mxfp4MoEWrapper(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=tw.hidden_size,
                intermediate_size=tw.intermediate_size,
                num_local_experts=tw.num_local_experts,
                local_expert_offset=self._rank * tw.num_local_experts,
                enable_pdl=self._kernel_config.enable_pdl,
            )
            self._wrappers[top_k] = wrapper
        return wrapper

    def pack_dispatch_payload(self, x: "torch.Tensor") -> "torch.Tensor":
        """Quantize tokens pre-dispatch when ``mxfp8_dispatch`` is enabled.

        Packs the ``[M, H]`` fp8 payload and the ``[M, H/32]`` linear UE8M0
        scale bytes into one row per token, zero-padded to the nearest
        transport-supported width and viewed as bf16 (the LL dispatch kernel
        rejects 1-byte payload dtypes and off-whitelist widths). Wire bytes
        vs plain BF16: 0.57x at H=7168, 0.625x at H=4096/8192; no saving at
        H<=2048 (the whitelist floor). Per-token rows quantize identically
        before or after dispatch, so results match the default path bit for
        bit.
        """
        if not self._kernel_config.mxfp8_dispatch:
            return x
        import torch

        if x.dim() != 2 or x.dtype != torch.bfloat16:
            raise ValueError(
                "mxfp8_dispatch expects 2D BF16 [num_tokens, hidden] tokens, "
                f"got {x.dtype} shape {tuple(x.shape)}"
            )
        from flashinfer.quantization.fp8_quantization import mxfp8_quantize

        m, hidden = x.shape
        send_width = packed_dispatch_width(hidden)
        q, sf = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
        packed = torch.zeros(m, 2 * send_width, dtype=torch.uint8, device=x.device)
        packed[:, :hidden] = q.view(torch.uint8)
        packed[:, hidden : hidden + hidden // 32] = sf.view(torch.uint8).reshape(
            m, hidden // 32
        )
        return packed.view(torch.bfloat16)

    def _unpack_or_quantize(
        self, flat: "torch.Tensor", hidden: int
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return (fp8 values ``[M, H]``, uint8 scale bytes ``[M, H/32]``)."""
        import torch

        m = flat.shape[0]
        if self._kernel_config.mxfp8_dispatch:
            send_width = packed_dispatch_width(hidden)
            if flat.dtype != torch.bfloat16 or flat.shape[1] != send_width:
                raise ValueError(
                    "mxfp8_dispatch compute expects packed bf16-viewed rows of "
                    f"width {send_width} ({hidden} payload + {hidden // 32} "
                    f"scale bytes, zero-padded), got {flat.dtype} width "
                    f"{flat.shape[1]}"
                )
            packed_bytes = flat.contiguous().view(torch.uint8)
            x_q = packed_bytes[:, :hidden].contiguous().view(torch.float8_e4m3fn)
            x_sf = packed_bytes[:, hidden : hidden + hidden // 32].contiguous()
            return x_q, x_sf
        if flat.dtype != torch.bfloat16 or flat.shape[1] != hidden:
            raise ValueError(
                "sm100_mxfp8_mxfp4_bf16_cutedsl consumes BF16 dispatch tokens "
                f"of width {hidden} (quantized locally to MXFP8), got "
                f"{flat.dtype} width {flat.shape[1]}"
            )
        from flashinfer.quantization.fp8_quantization import mxfp8_quantize

        x_q, x_sf = mxfp8_quantize(flat.contiguous(), is_sf_swizzled_layout=False)
        return x_q, x_sf.view(torch.uint8).reshape(m, hidden // 32).contiguous()

    def compute(self, ctx: SplitKernelContext) -> "torch.Tensor":
        import torch

        tw = self._require_ready()
        expert_tensors = ctx.expert_tensors
        if expert_tensors.dim() != 3:
            raise ValueError(
                "compute expects a 3D dispatch tensor, got shape "
                f"{tuple(expert_tensors.shape)}"
            )
        fleet_params = ctx.fleet_params
        offset = self._rank * tw.num_local_experts
        hidden = tw.hidden_size
        dim0, dim1, _row_width = expert_tensors.shape
        flat = expert_tensors.reshape(dim0 * dim1, _row_width)
        m = flat.shape[0]
        device = flat.device

        is_ht = fleet_params.algorithm is EpAlgorithm.HIGH_THROUGHPUT
        if is_ht or fleet_params.layout is EpLayout.RANK_MAJOR:
            if ctx.recv_topk_idx is None or ctx.recv_topk_weights is None:
                raise RuntimeError(
                    f"{'HT' if is_ht else 'RANK_MAJOR'} compute requires dispatch "
                    "to return recv_topk_idx / recv_topk_weights; got None."
                )
            # Received LOCAL expert ids (-1 = non-local pick): convert to
            # global ids for the kernel and mask non-local picks to weight 0
            # (pointed at a valid local expert so indexing stays in range).
            idx = ctx.recv_topk_idx.to(torch.int64)
            weights = ctx.recv_topk_weights.to(torch.float32)
            if idx.shape != weights.shape or idx.shape[0] != m:
                raise ValueError(
                    f"recv_topk_idx/weights must share shape [M={m}, top_k]; "
                    f"got {tuple(idx.shape)} / {tuple(weights.shape)}."
                )
            is_local = (idx >= 0) & (idx < tw.num_local_experts)
            selected_experts = (
                torch.where(is_local, idx + offset, torch.full_like(idx, offset))
                .to(torch.int32)
                .contiguous()
            )
            final_scales = torch.where(
                is_local, weights, torch.zeros_like(weights)
            ).contiguous()
            top_k = selected_experts.shape[1]
        else:
            # EXPERT_MAJOR: row -> its own (single) expert, weight 1.0; the
            # real topk_weights are applied by EP combine.
            row_expert = torch.arange(dim0, device=device, dtype=torch.int32)
            selected_experts = (
                row_expert.repeat_interleave(dim1).reshape(m, 1) + offset
            ).contiguous()
            final_scales = torch.ones(m, 1, dtype=torch.float32, device=device)
            top_k = 1

        x_q, x_sf = self._unpack_or_quantize(flat, hidden)

        wrapper = self._ensure_wrapper(top_k, fleet_params.num_experts)
        out = wrapper.run(
            x_q,
            x_sf,
            selected_experts,
            final_scales,
            tw.w1_weight,
            tw.w1_weight_sf,
            tw.w1_alpha,
            tw.w2_weight,
            tw.w2_weight_sf,
            tw.w2_alpha,
            tactic=self._kernel_config.tactic,
        )
        return out.view(dim0, dim1, hidden)
