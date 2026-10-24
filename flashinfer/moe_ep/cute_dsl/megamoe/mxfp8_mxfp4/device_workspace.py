# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed device views over the opaque MegaMoE workspaces.

``workspace.py`` is the sole owner of byte offsets and extents.  This module
only validates an allocation against that plan and turns each declared region
into a zero-copy Torch view.  CuTe conversion is deliberately lazy so importing
the workspace/frontend path on a CPU-only host does not import CUTLASS or CuTe.

Single-rank dispatch uses ``expert_send_count`` as its expert-count vector.
There is intentionally no fabricated expert-offset allocation: the persistent
scheduler derives the padded data, scale, and token-block prefixes from those
counts, just as the host scheduler model does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .workspace import RegionLayout, RegionSpec, WorkspacePlan, round_up


SCHEDULER_COUNTER_REGIONS = {
    # Dispatch publishes token arrivals here; FC1 scheduler records wait on it.
    "fc1_ready_counters": "l1_arrival_count",
    # FC1 epilogues publish completed handoff tiles here; FC2 waits on it.
    "fc2_ready_counters": "fc1_done_counter",
}


def _torch_dtype(spec: RegionSpec) -> torch.dtype:
    ordinary = {
        "uint8": torch.uint8,
        "int32": torch.int32,
        "int64": torch.int64,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if spec.dtype in ordinary:
        return ordinary[spec.dtype]
    attr = {
        "float8_e4m3": "float8_e4m3fn",
        "float8_e8m0": "float8_e8m0fnu",
    }.get(spec.dtype)
    dtype = getattr(torch, attr, None) if attr is not None else None
    if dtype is None:
        raise RuntimeError(
            f"Torch does not provide the dtype required by {spec.name}: {spec.dtype}"
        )
    return dtype


def _validate_byte_workspace(
    tensor: torch.Tensor, layout: RegionLayout, name: str
) -> None:
    if tensor.dtype is not torch.uint8:
        raise TypeError(f"{name} workspace must have dtype torch.uint8")
    if tensor.ndim != 1 or not tensor.is_contiguous():
        raise ValueError(f"{name} workspace must be a contiguous 1D tensor")
    if tensor.numel() != layout.total_bytes:
        raise ValueError(
            f"{name} workspace has {tensor.numel()} bytes; "
            f"the plan requires exactly {layout.total_bytes}"
        )
    base = tensor.data_ptr()
    for region in layout.regions:
        address = base + region.offset
        if address % region.spec.align:
            raise ValueError(
                f"{name}.{region.spec.name} address is not "
                f"{region.spec.align}-byte aligned"
            )


def _partition(
    workspace: torch.Tensor, layout: RegionLayout
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for region in layout.regions:
        raw = workspace.narrow(0, region.offset, region.spec.nbytes)
        result[region.spec.name] = raw.view(_torch_dtype(region.spec)).reshape(
            region.spec.shape
        )
    return result


@dataclass(frozen=True)
class ExpertPoolOffsets:
    """Host oracle for scheduler-derived expert pool prefixes.

    Each tuple has ``expert_count + 1`` entries and ends at the total extent.
    It is a contract/debugging object, not device storage.
    """

    data_rows: tuple[int, ...]
    scale_rows: tuple[int, ...]
    token_blocks: tuple[int, ...]


class DeviceWorkspacePartition:
    """Zero-copy Torch/CuTe views for one workspace plan.

    ``shared_workspace`` remains optional during single-rank bring-up because
    the current integration allocates only the local pool.  Passing it enables
    the shared-region views without changing any local offsets.
    """

    def __init__(
        self,
        plan: WorkspacePlan,
        local_workspace: torch.Tensor,
        route_terms: torch.Tensor,
        *,
        shared_workspace: torch.Tensor | None = None,
    ) -> None:
        _validate_byte_workspace(local_workspace, plan.local, "local")
        if shared_workspace is not None:
            _validate_byte_workspace(shared_workspace, plan.shared, "shared")
            if shared_workspace.device != local_workspace.device:
                raise ValueError("local and shared workspaces must be on one device")

        config = plan.config
        expected_route_shape = (
            config.max_tokens_per_rank,
            config.num_topk,
            config.hidden,
        )
        if route_terms.dtype is not torch.bfloat16:
            raise TypeError("route_terms must have dtype torch.bfloat16")
        if tuple(route_terms.shape) != expected_route_shape:
            raise ValueError(
                f"route_terms must have shape {expected_route_shape}, "
                f"got {tuple(route_terms.shape)}"
            )
        if not route_terms.is_contiguous():
            raise ValueError("route_terms must be contiguous")
        if route_terms.device != local_workspace.device:
            raise ValueError("route_terms and workspace must be on one device")

        self.plan = plan
        self.local_workspace = local_workspace
        self.shared_workspace = shared_workspace
        self.route_terms = route_terms
        self.local = _partition(local_workspace, plan.local)
        self.shared = (
            _partition(shared_workspace, plan.shared)
            if shared_workspace is not None
            else {}
        )

        # Semantic views used by dispatch/scheduler code.  These aliases are
        # still backed by declared workspace bytes; none creates storage.
        self.expert_counts = self.local["expert_send_count"]
        self.expert_recv_count = self.shared.get("expert_recv_count")
        self.expert_recv_count_sum = self.shared.get(
            "expert_recv_count_sum", self.expert_counts
        )
        self.fc1_ready_counters = self.local[
            SCHEDULER_COUNTER_REGIONS["fc1_ready_counters"]
        ]
        self.fc2_ready_counters = self.local[
            SCHEDULER_COUNTER_REGIONS["fc2_ready_counters"]
        ]
        self.token_metadata = self.local["token_src_metadata"]
        self.token_metadata_i64 = self.token_metadata.view(torch.int64).reshape(-1)

        sf_bytes = self.local["l1_sf_buffer"].view(torch.uint8)
        sf_cols = config.sf_uint32_per_token * 4
        sf_dtype = _torch_dtype(plan.local.region("fc1_output_sf").spec)
        self.activation_scales = sf_bytes.view(sf_dtype).reshape(
            config.pool_sf_capacity, sf_cols
        )

    @classmethod
    def from_launch_regions(
        cls,
        plan: WorkspacePlan,
        regions: Mapping[str, torch.Tensor],
    ) -> "DeviceWorkspacePartition":
        """Build from the stable frontend ``workspace_regions`` mapping."""

        missing = tuple(
            name for name in ("local", "shared", "route_terms") if name not in regions
        )
        if missing:
            raise ValueError("workspace regions are missing: " + ", ".join(missing))
        return cls(
            plan,
            regions["local"],
            regions["route_terms"],
            shared_workspace=regions["shared"],
        )

    @property
    def torch_views(self) -> dict[str, torch.Tensor]:
        """All launch views, including stable semantic aliases."""

        views = dict(self.local)
        views.update({f"shared.{name}": value for name, value in self.shared.items()})
        views.update(
            {
                "route_terms": self.route_terms,
                "expert_counts": self.expert_counts,
                **(
                    {"expert_recv_count": self.expert_recv_count}
                    if self.expert_recv_count is not None
                    else {}
                ),
                "expert_recv_count_sum": self.expert_recv_count_sum,
                "fc1_ready_counters": self.fc1_ready_counters,
                "fc2_ready_counters": self.fc2_ready_counters,
                "token_metadata": self.token_metadata,
                "token_metadata_i64": self.token_metadata_i64,
                "activation_scales": self.activation_scales,
            }
        )
        return views

    @property
    def zero_prefix_views(self) -> tuple[torch.Tensor, ...]:
        """Byte slices that must be reset before each persistent launch."""

        prefixes = [
            self.local_workspace.narrow(0, 0, self.plan.local.zero_prefix_bytes)
        ]
        if self.shared_workspace is not None:
            prefixes.append(
                self.shared_workspace.narrow(0, 0, self.plan.shared.zero_prefix_bytes)
            )
        return tuple(prefixes)

    def zero_prefixes_(self) -> "DeviceWorkspacePartition":
        """Reset exactly the counter/readiness prefixes on the current stream."""

        for prefix in self.zero_prefix_views:
            prefix.zero_()
        return self

    def validate_zero_prefixes(self) -> None:
        """Synchronizing debug check that every required prefix byte is zero."""

        for index, prefix in enumerate(self.zero_prefix_views):
            if bool(torch.count_nonzero(prefix).item()):
                kind = "local" if index == 0 else "shared"
                raise ValueError(f"{kind} workspace counter prefix is not zero")

    def expert_pool_offsets(
        self, counts: Sequence[int] | torch.Tensor
    ) -> ExpertPoolOffsets:
        """Validate counts and mirror the device scheduler's prefix arithmetic."""

        if isinstance(counts, torch.Tensor):
            if counts.device.type != "cpu":
                raise ValueError("expert_pool_offsets is a host-only validation helper")
            if counts.ndim != 1:
                raise ValueError("expert counts must be one-dimensional")
            values = tuple(int(value) for value in counts.tolist())
        else:
            values = tuple(int(value) for value in counts)

        config = self.plan.config
        if len(values) != config.num_experts_per_rank:
            raise ValueError(
                f"expected {config.num_experts_per_rank} expert counts, "
                f"got {len(values)}"
            )
        if any(value < 0 for value in values):
            raise ValueError("expert counts must be non-negative")

        data = [0]
        scales = [0]
        blocks = [0]
        for value in values:
            data.append(data[-1] + round_up(value, config.token_padding_block))
            scales.append(scales[-1] + round_up(value, config.sf_padding_block))
            blocks.append(
                blocks[-1]
                + (value + config.cluster_tile_tokens - 1) // config.cluster_tile_tokens
            )
        if data[-1] > config.pool_token_capacity:
            raise ValueError("expert counts exceed the token-pool capacity")
        if scales[-1] > config.pool_sf_capacity:
            raise ValueError("expert counts exceed the scale-pool capacity")
        if blocks[-1] + config.num_experts_per_rank > config.fc1_done_slots:
            raise ValueError("expert counts exceed the readiness-counter capacity")
        return ExpertPoolOffsets(tuple(data), tuple(scales), tuple(blocks))

    def to_cute(self) -> dict[str, Any]:
        """Convert every Torch view lazily; CuTe is never imported at module load."""

        if self.local_workspace.device.type != "cuda":
            raise ValueError("CuTe views require CUDA workspace tensors")
        try:
            from cutlass.cute.runtime import from_dlpack
            from cutlass.torch import get_leading_dim
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("CUTLASS CuTe DSL is required for device views") from exc

        alignments = {
            region.spec.name: region.spec.align for region in self.plan.local.regions
        }
        alignments.update(
            {
                f"shared.{region.spec.name}": region.spec.align
                for region in self.plan.shared.regions
            }
        )
        alias_sources = {
            "expert_counts": "expert_send_count",
            "expert_recv_count": "shared.expert_recv_count",
            "expert_recv_count_sum": (
                "shared.expert_recv_count_sum" if self.shared else "expert_send_count"
            ),
            **SCHEDULER_COUNTER_REGIONS,
            "token_metadata": "token_src_metadata",
            "token_metadata_i64": "token_src_metadata",
            "activation_scales": "l1_sf_buffer",
        }
        result: dict[str, Any] = {}
        for name, tensor in self.torch_views.items():
            source_name = alias_sources.get(name, name)
            assumed_align = alignments.get(source_name, max(tensor.element_size(), 1))
            result[name] = from_dlpack(
                tensor, assumed_align=assumed_align
            ).mark_layout_dynamic(leading_dim=get_leading_dim(tensor))
        return result


__all__ = [
    "DeviceWorkspacePartition",
    "ExpertPoolOffsets",
    "SCHEDULER_COUNTER_REGIONS",
]
