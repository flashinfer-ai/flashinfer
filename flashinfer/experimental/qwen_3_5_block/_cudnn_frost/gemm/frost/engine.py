# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FROST GEMM engine: JIT-compiled fused matmul chains for cuDNN graphs."""

from typing import TYPE_CHECKING, List

from flashinfer.experimental.qwen_3_5_block._cudnn_frost import behavior_note
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.engines.base import BaseEngine, CompiledPlan, ExecutionContext, PlanConfig
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.workspace import Workspace

if TYPE_CHECKING:
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost._pygraph import pygraph


class _FrostGemmPlan(CompiledPlan):
    """A compiled fused-GEMM kernel plus the graph binding it was compiled for."""

    takes_variant_pack = True

    def __init__(self, compiled):
        self._compiled = compiled
        self._tensors = list(compiled.binding.bound_tensors())
        self._operand_indices = None
        self._lowered = getattr(compiled, "lowered", None)
        self._launch = self._lowered

    def get_workspace_size(self) -> int:
        return int(getattr(self._compiled, "workspace_bytes", 0) or 0)

    def get_workspace_size_for_shapes(self, override_uids, override_shapes) -> int:
        """Workspace for an override-shape execute (split-K partials follow the overridden A and B)."""
        sized = getattr(self._compiled, "workspace_bytes_for_overrides", None)
        if sized is None or not self.get_workspace_size():
            return self.get_workspace_size()
        return sized(override_uids, override_shapes)

    def execute(self, graph, variant_pack, ctx: ExecutionContext) -> None:
        indices = self._operand_indices
        if indices is None:
            try:
                indices = self._operand_indices = [variant_pack.index_of(t.get_uid()) for t in self._tensors]
            except KeyError as exc:
                raise ValueError(f"frost_gemm: tensor uid {exc} is bound by the kernel but is not an operand of this graph") from exc
        operands = variant_pack.operands(indices)
        graph_order = None
        borrowed = variant_pack.graph_described
        if borrowed:
            flags = tuple(i in borrowed for i in indices)
            graph_order = flags if any(flags) else None
        required = self.get_workspace_size()
        if required:
            sized = getattr(self._compiled, "workspace_bytes_for_operands", None)
            if sized is not None:
                required = sized(operands, graph_order)
        workspace = Workspace.over(variant_pack, required, "frost_gemm") if required else None
        launch = self._launch
        if launch is not None:
            launch(operands, graph_order, stream=ctx.stream, workspace=workspace)
        elif workspace is not None:
            self._compiled(dict(zip(self._tensors, operands)), workspace, stream=ctx.stream)
        else:
            self._compiled(dict(zip(self._tensors, operands)), stream=ctx.stream)


class FrostGemmEngine(BaseEngine):
    """Fused matmul chains (dense, block-scaled, MoE grouped) on SM100/SM103."""

    name = "frost_gemm"
    behavior_notes = (behavior_note.RUNTIME_COMPILATION,)  # JIT-compiled at build_plans()

    def check_support(self, graph: "pygraph") -> None:
        from .compiler import probe_supported
        from .graph_analyzer import _graph_has_gemm

        if not _graph_has_gemm(graph):
            raise NotImplementedError("frost_gemm: the graph has no matmul / moe_grouped_matmul node")
        try:
            probe_supported(graph)
        except (NotImplementedError, ValueError) as exc:
            raise NotImplementedError(f"frost_gemm: {exc}") from exc

    def knobs_to_public(self, knobs) -> dict:
        from .knobs import GemmKnobs

        if knobs is None:
            return {}
        if isinstance(knobs, dict):
            return dict(knobs)
        if isinstance(knobs, GemmKnobs):
            return knobs.to_public()
        return super().knobs_to_public(knobs)

    def knobs_from_public(self, public: dict):
        from .knobs import GemmKnobs

        return GemmKnobs.from_public(public) if public else None

    def build_plan(self, graph: "pygraph", plan: PlanConfig, ctx: ExecutionContext = None) -> CompiledPlan:
        from .graph_analyzer import build_gemm_plan
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import build_device

        knobs = plan.knobs if plan is not None else None
        handle = ctx.handle if ctx is not None else None
        device = handle.device.ordinal if hasattr(handle, "device") else None
        try:
            if isinstance(knobs, dict):  # a replayed public record, not yet converted
                knobs = self.knobs_from_public(knobs)  # a malformed record is a decline, not an abort
            with build_device(device):
                return _FrostGemmPlan(build_gemm_plan(graph, knobs=knobs))
        except (NotImplementedError, ValueError) as exc:
            raise NotImplementedError(f"frost_gemm: {exc}") from exc


def FrostGemmEngines(ids):
    """The gemm engines the manifest asked for, with the ids it assigned."""
    out = []
    for cls in (FrostGemmEngine,):
        if cls.name in ids:
            engine = cls()
            engine.engine_id = ids[cls.name]
            out.append(engine)
    return out
