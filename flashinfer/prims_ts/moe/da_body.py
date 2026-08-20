"""Typed routing and workspace records for PrimsTS MoE body execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Mapping,
    Protocol,
    Self,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
)

import torch

from flashinfer.fused_moe.factorized import MoeTactic

if TYPE_CHECKING:
    from flashinfer.fused_moe.core import TrtllmMoERoutingMetadataSlot
    from flashinfer.fused_moe.da_moe import DABody


@dataclass(frozen=True)
class PrimsTsRoutingMetadata:
    """Typed PrimsTS view of one shared native routing-metadata slot."""

    # Device scalar containing the live padded permutation size.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.total_num_padded_tokens.
    total_num_padded_tokens: torch.Tensor
    # Expanded token-slot to permuted-row mapping consumed by finalize.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.expanded_idx_to_permuted_idx.
    expanded_idx_to_permuted_idx: torch.Tensor
    # Permuted-row to original token mapping consumed by both GEMMs.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.permuted_idx_to_token_idx.
    permuted_idx_to_token_idx: torch.Tensor
    # Graph-live routed weights consumed by finalize.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.expert_weights.
    expert_weights: torch.Tensor
    # Per-expert histogram retained by the shared routing ABI.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.expert_count_histogram.
    expert_count_histogram: torch.Tensor
    # Per-expert token counts retained by the shared routing ABI.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.num_tokens_per_expert.
    num_tokens_per_expert: torch.Tensor
    # Grouped-GEMM CTA-to-expert mapping consumed by PrimsTS bodies.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.cta_idx_xy_to_batch_idx.
    tile_idx: torch.Tensor
    # Grouped-GEMM CTA row limits consumed by PrimsTS bodies.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.cta_idx_xy_to_mn_limit.
    mn_limit: torch.Tensor
    # Device scalar containing the number of live grouped-GEMM CTAs.
    # sync with flashinfer/fused_moe/core.py:TrtllmMoERoutingMetadataSlot.num_non_exiting_ctas.
    num_non_exiting_ctas: torch.Tensor

    @classmethod
    def from_sequence(cls, tensors: Sequence[torch.Tensor]) -> PrimsTsRoutingMetadata:
        """Decode the fixed shared routing ABI into named PrimsTS fields."""
        if len(tensors) != 9:
            raise ValueError("PrimsTS routing metadata requires nine tensors")
        return cls(*tensors)

    def body_view(self) -> PrimsTsBodyRouting:
        """Project shared metadata onto the fields consumed by one PrimsTS body."""
        return PrimsTsBodyRouting(
            expert_weights=self.expert_weights,
            expanded_idx_to_permuted_idx=self.expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx=self.permuted_idx_to_token_idx,
            tile_idx=self.tile_idx,
            mn_limit=self.mn_limit,
            num_non_exiting_ctas=self.num_non_exiting_ctas,
            total_num_padded_tokens=self.total_num_padded_tokens,
        )


@dataclass(frozen=True)
class PrimsTsBodyRouting:
    """Routing fields consumed directly by every PrimsTS FC1/FC2 body ABI."""

    # Routed weights consumed by the shared finalize kernel.
    expert_weights: torch.Tensor
    # Expanded token-slot to permuted-row mapping consumed by finalize.
    expanded_idx_to_permuted_idx: torch.Tensor
    # Permuted-row to original token mapping consumed by both GEMMs.
    permuted_idx_to_token_idx: torch.Tensor
    # Grouped-GEMM CTA-to-expert mapping consumed by PrimsTS bodies.
    tile_idx: torch.Tensor
    # Grouped-GEMM CTA row limits consumed by PrimsTS bodies.
    mn_limit: torch.Tensor
    # Device scalar containing the number of live grouped-GEMM CTAs.
    num_non_exiting_ctas: torch.Tensor
    # Device scalar containing the live padded permutation size.
    total_num_padded_tokens: torch.Tensor


@dataclass(frozen=True)
class PrimsTsNvfp4BodyWorkspace:
    """Typed graph-stable body workspace for one PrimsTS NVFP4 ABI."""

    # Quantized FC1 output used by config rows with a fused quantizing epilogue.
    gemm1_output_quantized: torch.Tensor
    # BF16 FC1 output used by config rows with explicit per-token FC2 quantization.
    gemm1_output_bf16: torch.Tensor
    # FC1 block scales consumed by the NVFP4 FC2 body.
    gemm1_output_scale: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor
    # Optional per-token quantized FC1 activation consumed by FC2.
    activation_output: torch.Tensor
    # Optional per-token FC1 activation scales consumed by FC2.
    activation_output_scale: torch.Tensor
    # Optional per-token scaling factor consumed by FC2.
    per_token_scale_fc2: torch.Tensor
    # Optional BF16 copy of live FP32 unpacked routing weights consumed by finalize.
    expert_weights_bf16: torch.Tensor

    @classmethod
    def from_sequence(
        cls, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsNvfp4BodyWorkspace:
        """Decode one maximum NVFP4 workspace without assuming another dtype ABI."""
        if len(tensors) != 8:
            raise ValueError("PrimsTS NVFP4 body workspace requires eight tensors")
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact NVFP4 adapter ABI order."""
        return (
            self.gemm1_output_quantized,
            self.gemm1_output_bf16,
            self.gemm1_output_scale,
            self.gemm2_output,
            self.activation_output,
            self.activation_output_scale,
            self.per_token_scale_fc2,
            self.expert_weights_bf16,
        )


@dataclass(frozen=True)
class PrimsTsBf16BodyWorkspace:
    """Typed graph-stable body workspace for the PrimsTS BF16 ABI."""

    # FC1 activation output consumed by the BF16 FC2 body.
    gemm1_output: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor

    @classmethod
    def from_sequence(cls, tensors: Sequence[torch.Tensor]) -> PrimsTsBf16BodyWorkspace:
        """Decode one maximum BF16 workspace without assuming another dtype ABI."""
        if len(tensors) != 2:
            raise ValueError("PrimsTS BF16 body workspace requires two tensors")
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact BF16 adapter ABI order."""
        return self.gemm1_output, self.gemm2_output


@dataclass(frozen=True)
class PrimsTsMxfp4Mxfp8BodyWorkspace:
    """Typed graph-stable body workspace for the PrimsTS MXFP4xMXFP8 ABI."""

    # FC1 quantized activation output consumed by the MXFP8 FC2 body.
    gemm1_output: torch.Tensor
    # FC1 block scales consumed by the MXFP8 FC2 body.
    gemm1_output_scale: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor

    @classmethod
    def from_sequence(
        cls, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsMxfp4Mxfp8BodyWorkspace:
        """Decode one maximum MXFP4xMXFP8 workspace in its exact ABI order."""
        if len(tensors) != 3:
            raise ValueError(
                "PrimsTS MXFP4xMXFP8 body workspace requires three tensors"
            )
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact MXFP4xMXFP8 adapter ABI order."""
        return self.gemm1_output, self.gemm1_output_scale, self.gemm2_output


@dataclass(frozen=True)
class PrimsTsMxfp4Bf16BodyWorkspace:
    """Typed graph-stable body workspace for the PrimsTS MXFP4xBF16 ABI."""

    # FC1 BF16 activation output consumed by the BF16 FC2 body.
    gemm1_output: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor

    @classmethod
    def from_sequence(
        cls, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsMxfp4Bf16BodyWorkspace:
        """Decode one maximum MXFP4xBF16 workspace in its exact ABI order."""
        if len(tensors) != 2:
            raise ValueError("PrimsTS MXFP4xBF16 body workspace requires two tensors")
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact MXFP4xBF16 adapter ABI order."""
        return self.gemm1_output, self.gemm2_output


@dataclass(frozen=True)
class PrimsTsFp8PerTensorBodyWorkspace:
    """Typed graph-stable body workspace for the PrimsTS FP8 per-tensor ABI."""

    # FC1 FP8 activation output consumed by the FP8 FC2 body.
    gemm1_output: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor

    @classmethod
    def from_sequence(
        cls, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsFp8PerTensorBodyWorkspace:
        """Decode one maximum FP8 per-tensor workspace in its exact ABI order."""
        if len(tensors) != 2:
            raise ValueError(
                "PrimsTS FP8 per-tensor body workspace requires two tensors"
            )
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact FP8 per-tensor adapter ABI order."""
        return self.gemm1_output, self.gemm2_output


@dataclass(frozen=True)
class PrimsTsDeepSeekFp8BodyWorkspace:
    """Typed graph-stable body workspace for the PrimsTS DeepSeek FP8 ABI."""

    # Unactivated FC1 output consumed by the explicit DeepSeek activation kernel.
    gemm1_output: torch.Tensor
    # FC1 accumulator scales consumed by the explicit activation kernel.
    gemm1_output_scale: torch.Tensor
    # Quantized activation output consumed by FC2.
    activation_output: torch.Tensor
    # Quantized activation scales consumed by FC2.
    activation_output_scale: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor

    @classmethod
    def from_sequence(
        cls, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsDeepSeekFp8BodyWorkspace:
        """Decode one maximum DeepSeek FP8 workspace in its exact ABI order."""
        if len(tensors) != 5:
            raise ValueError(
                "PrimsTS DeepSeek FP8 body workspace requires five tensors"
            )
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact DeepSeek FP8 adapter ABI order."""
        return (
            self.gemm1_output,
            self.gemm1_output_scale,
            self.activation_output,
            self.activation_output_scale,
            self.gemm2_output,
        )


@dataclass(frozen=True)
class PrimsTsMxfp8BodyWorkspace:
    """Typed graph-stable body workspace for the PrimsTS MXFP8 ABI."""

    # Quantized FC1 activation output consumed by FC2.
    gemm1_output: torch.Tensor
    # Quantized FC1 activation scales consumed by FC2.
    gemm1_output_scale: torch.Tensor
    # Router-provided activation alias retained by the native MXFP8 ABI.
    activation_output: torch.Tensor
    # Router-provided activation-scale alias retained by the native MXFP8 ABI.
    activation_output_scale: torch.Tensor
    # FC2 expert output consumed by the shared finalize kernel.
    gemm2_output: torch.Tensor
    # Graph-stable padded view of the live MXFP8 activation scales.
    hidden_states_scale_padded: torch.Tensor

    @classmethod
    def from_sequence(
        cls, tensors: Sequence[torch.Tensor]
    ) -> PrimsTsMxfp8BodyWorkspace:
        """Decode one maximum MXFP8 workspace in its exact ABI order."""
        if len(tensors) != 6:
            raise ValueError("PrimsTS MXFP8 body workspace requires six tensors")
        return cls(*tensors)

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return fields in the exact MXFP8 adapter ABI order."""
        return (
            self.gemm1_output,
            self.gemm1_output_scale,
            self.activation_output,
            self.activation_output_scale,
            self.gemm2_output,
            self.hidden_states_scale_padded,
        )


class PrimsTsBodyWorkspace(Protocol):
    """Define the typed workspace record contract shared by PrimsTS runners."""

    @classmethod
    def from_sequence(cls, tensors: Sequence[torch.Tensor]) -> Self:
        """Decode one dtype-specific workspace from its public tensor ABI."""
        ...

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return workspace fields in the dtype-specific body ABI order."""
        ...


WorkspaceT = TypeVar("WorkspaceT", bound=PrimsTsBodyWorkspace)


@dataclass(frozen=True)
class PrimsTsBodyExecution(Generic[WorkspaceT]):
    """Pair graph-live routing fields with one dtype-specific body workspace."""

    # Routing values and metadata consumed by the body and finalize kernels.
    routing: PrimsTsBodyRouting
    # Exact dtype-specific buffers consumed and produced by the body kernels.
    workspace: WorkspaceT


class PrimsTsBodySource(Protocol[WorkspaceT]):
    """Resolve ordinary routing or an already-prepared DA body execution."""

    # True when the source owns graph-stable storage that must not be reallocated.
    preallocated: bool

    def resolve(
        self, factory: Callable[[], PrimsTsBodyExecution[WorkspaceT]]
    ) -> PrimsTsBodyExecution[WorkspaceT]:
        """Return the execution record supplied by this source."""
        ...


@dataclass(frozen=True)
class PrimsTsOrdinaryBodySource(Generic[WorkspaceT]):
    """Resolve body inputs by invoking the ordinary dtype-specific router."""

    # Ordinary routing owns any body allocations returned by the router.
    preallocated: ClassVar[bool] = False

    def resolve(
        self, factory: Callable[[], PrimsTsBodyExecution[WorkspaceT]]
    ) -> PrimsTsBodyExecution[WorkspaceT]:
        """Invoke the ordinary routing and allocation factory exactly once."""
        return factory()


@dataclass(frozen=True)
class PrimsTsPreparedBodySource(Generic[WorkspaceT]):
    """Supply graph-live metadata and graph-stable storage prepared by DA."""

    # Typed execution record retained by one DA workspace lane.
    execution: PrimsTsBodyExecution[WorkspaceT]
    # Prepared DA bodies must reuse the supplied graph-stable buffers.
    preallocated: ClassVar[bool] = True

    def resolve(
        self, factory: Callable[[], PrimsTsBodyExecution[WorkspaceT]]
    ) -> PrimsTsBodyExecution[WorkspaceT]:
        """Return prepared body inputs without invoking ordinary routing."""
        del factory
        return self.execution


class PrimsTsPreparedBodyRunner(Protocol):
    """Describe the ordinary-runner capabilities required by the DA adapter."""

    def prepare_body_workspace(
        self,
        inputs: list[torch.Tensor],
        tactic: MoeTactic,
        **kwargs: Any,
    ) -> Sequence[torch.Tensor]:
        """Allocate one tactic's exact workspace outside CUDA Graph capture."""
        ...

    def run_prepared_body(
        self,
        inputs: list[torch.Tensor],
        tactic: MoeTactic,
        routing_metadata: Sequence[torch.Tensor],
        body_workspace: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        """Run one tactic from graph-live routing and preallocated storage."""
        ...


class PrimsTsDaBodyRunner:
    """Compose an ordinary PrimsTS runner with prepared-metadata body execution."""

    def __init__(self, moe_runner: PrimsTsPreparedBodyRunner) -> None:
        """Retain the ordinary dtype-specific runner used by this DA adapter."""
        # Ordinary full-operation runner with explicit workspace and prepared-body capabilities.
        self._moe_runner = moe_runner

    @property
    def moe_runner(self) -> PrimsTsPreparedBodyRunner:
        """Return the composed ordinary PrimsTS runner."""
        return self._moe_runner

    def prepare_body(
        self,
        inputs: list[torch.Tensor],
        body: DABody,
        routing_metadata: TrtllmMoERoutingMetadataSlot,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Allocate one body's exact typed workspace outside CUDA Graph capture."""
        if body.tile_n != routing_metadata.tile_n:
            raise ValueError("DA body and routing metadata tile_n must match")
        prepared = self._moe_runner.prepare_body_workspace(
            inputs,
            tactic=[body.tile_n, body.tactic],
            **kwargs,
        )
        return tuple(prepared)

    def prepare_max_body_workspace(
        self,
        inputs: list[torch.Tensor],
        bodies: Sequence[DABody],
        routing_metadata_by_tile: Mapping[int, TrtllmMoERoutingMetadataSlot],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Retain one field-wise maximum typed workspace across candidate bodies."""
        if not bodies:
            raise ValueError("A shared DA body workspace requires at least one body")

        # Materialize every tactic's exact PrimsTS ABI before selecting shared backing storage.
        # SWITCH bodies are mutually exclusive, so the largest allocation per field is sufficient.
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
            raise RuntimeError("One dtype-specific DA plan exposed multiple body ABIs")

        # Dtype and device are fixed ABI properties. Logical shapes may vary by tactic, while
        # every captured body receives a typed view into the selected maximum-capacity field.
        maximum_fields = []
        for field_index in range(field_count):
            fields = [candidate[field_index] for candidate in candidates]
            first = fields[0]
            if any(
                field.dtype != first.dtype or field.device != first.device
                for field in fields[1:]
            ):
                raise RuntimeError("DA body workspace field ABI changed across tactics")
            maximum_fields.append(max(fields, key=lambda field: field.numel()))
        return tuple(maximum_fields)

    def forward_from_metadata(
        self,
        inputs: list[torch.Tensor],
        body: DABody,
        routing_metadata: TrtllmMoERoutingMetadataSlot,
        body_workspace: Sequence[torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Launch one PrimsTS body without routing, allocation, or DA policy lookup."""
        if body.tile_n != routing_metadata.tile_n:
            raise ValueError("DA body and routing metadata tile_n must match")
        self._moe_runner.run_prepared_body(
            inputs,
            tactic=[body.tile_n, body.tactic],
            routing_metadata=routing_metadata.tensors(),
            body_workspace=tuple(body_workspace),
            **kwargs,
        )
