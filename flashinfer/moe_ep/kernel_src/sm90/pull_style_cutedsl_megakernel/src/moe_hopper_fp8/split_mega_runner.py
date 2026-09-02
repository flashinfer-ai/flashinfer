# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host contracts for a PR357-style Hopper MXFP4 MegaMoE split pipeline.

This module deliberately stops at the boundary that can be expressed and
tested without pretending that two ordinary stream launches are a PR357
pipeline.  It owns:

* validation of the independently selected K1/K2 tactics and SM budgets;
* construction of ``split_role="k1"`` and ``split_role="k2"`` kernels;
* a byte-for-byte check that both kernels interpret the shared handoff
  workspaces identically; and
* role-specific compile requests carrying the Green Context stream and the
  partition-local ``max_active_clusters`` value.

The CUDA Graph/Green Context executor is intentionally a separate object.  A
caller must supply a verified executor before launching K1 and K2 concurrently;
there is no sequential fallback in this module because such a fallback would
not satisfy the PR357-style execution contract.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Protocol, Sequence, Tuple

import torch


class SplitMegaConfigurationError(ValueError):
    """Raised when a tactic pair cannot share the split MegaMoE ABI."""


class SplitMegaWorkspaceMismatch(RuntimeError):
    """Raised when K1 and K2 disagree about an opaque workspace layout."""


class SplitMegaExecutorRequired(RuntimeError):
    """Raised when a caller tries to launch without a verified graph executor."""


class SplitMegaTacticLike(Protocol):
    """Structural tactic ABI consumed by :class:`SplitMegaPlan`.

    The split runtime must remain importable without the benchmark/reference
    runners.  In particular, ``runner_fc12.ImplDesc`` transitively imports
    quantization backends that are not part of the production vendor slice.
    Full-repository callers may still pass that descriptor because it matches
    this protocol; production callers can use :class:`SplitMegaTactic`.
    """

    mma_tiler_mnk: Tuple[int, int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    use_2cta_instrs: bool
    force_static_sched: bool
    clc_bundle_size: Optional[int]
    num_sched_stages: Optional[int]
    load_balance_mode: str
    group_hint: Optional[int]
    in_kernel_fc2_reduce: bool
    token_back_mode: str
    epi_flag_batch: int | Tuple[int, int]
    flag_batch: int


@dataclass(frozen=True)
class SplitMegaTactic:
    """Lightweight, production-safe K1 or K2 compile-time tactic.

    This contains only fields used by the split kernel builder.  It deliberately
    has no Torch, CuTeDSL, reference-runner, or other quantization-backend
    dependency, so a deployment can construct a split plan from the four
    vendored production packages alone.
    """

    mma_tiler_mnk: Tuple[int, int, int] = (128, 64, 128)
    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    use_2cta_instrs: bool = False
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    in_kernel_fc2_reduce: bool = False
    token_back_mode: Literal[
        "epi_warps", "standalone_warps", "reuse_dispatch_warps"
    ] = "epi_warps"
    epi_flag_batch: int | Tuple[int, int] = (1, 1)
    flag_batch: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("mma_tiler_mnk", self.mma_tiler_mnk),
            ("cluster_shape_mnk", self.cluster_shape_mnk),
        ):
            if (
                not isinstance(value, tuple)
                or len(value) != 3
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
            ):
                raise SplitMegaConfigurationError(
                    f"{name} must be an integer triple, got {value!r}."
                )
        if self.load_balance_mode not in ("static", "atomic_counter"):
            raise SplitMegaConfigurationError(
                "load_balance_mode must be 'static' or 'atomic_counter', got "
                f"{self.load_balance_mode!r}."
            )
        if self.token_back_mode not in (
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ):
            raise SplitMegaConfigurationError(
                f"unsupported token_back_mode {self.token_back_mode!r}."
            )
        for name, value in (
            ("group_hint", self.group_hint),
            ("clc_bundle_size", self.clc_bundle_size),
            ("num_sched_stages", self.num_sched_stages),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise SplitMegaConfigurationError(
                    f"{name} must be a positive integer when set, got {value!r}."
                )
        _require_positive_int("flag_batch", self.flag_batch)
        batches = (
            (self.epi_flag_batch, self.epi_flag_batch)
            if isinstance(self.epi_flag_batch, int)
            and not isinstance(self.epi_flag_batch, bool)
            else self.epi_flag_batch
        )
        if not isinstance(batches, tuple) or len(batches) != 2:
            raise SplitMegaConfigurationError(
                "epi_flag_batch must be an integer or an (fc1, fc2) pair, got "
                f"{self.epi_flag_batch!r}."
            )
        for role, value in zip(("fc1", "fc2"), batches):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                or value > 32
            ):
                raise SplitMegaConfigurationError(
                    f"epi_flag_batch[{role}] must be in [1, 32], got {value!r}."
                )

    @property
    def token_back_by_dispatch(self) -> bool:
        return self.token_back_mode != "epi_warps"


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SplitMegaConfigurationError(
            f"{name} must be a positive integer, got {value!r}."
        )
    return value


def _cluster_size(impl: SplitMegaTacticLike) -> int:
    cm, cn, ck = impl.cluster_shape_mnk
    if ck != 1:
        raise SplitMegaConfigurationError(
            f"split MegaMoE requires cluster K=1, got {impl.cluster_shape_mnk}."
        )
    return cm * cn


@dataclass(frozen=True)
class SplitMegaPlan:
    """Compile-time K1/K2 tactics and their Green Context SM partitions.

    M, N, K, scheduler stages, and group hints are intentionally independent.
    Different token-N tiles use a tactic-independent completion block; the
    experimental multi-CTA cluster path remains same-N until its handoff
    mapping is generalized as well.
    """

    fc1_impl: SplitMegaTacticLike
    fc2_impl: SplitMegaTacticLike
    k1_sm_count: int
    k2_sm_count: int

    def __post_init__(self) -> None:
        k1_sms = _require_positive_int("k1_sm_count", self.k1_sm_count)
        k2_sms = _require_positive_int("k2_sm_count", self.k2_sm_count)

        for role, impl in (("K1", self.fc1_impl), ("K2", self.fc2_impl)):
            m, n, k = impl.mma_tiler_mnk
            if m not in (128, 256):
                raise SplitMegaConfigurationError(
                    f"{role} requires Hopper swap-AB tile M=128 or 256, got {m}."
                )
            if k not in (128, 256):
                raise SplitMegaConfigurationError(
                    f"{role} requires MXFP4 tile K=128 or 256, got {k}."
                )
            if n not in (16, 32, 64, 128):
                raise SplitMegaConfigurationError(
                    f"{role} requires token tile N in (16, 32, 64, 128), got {n}."
                )
            if impl.cluster_shape_mnk not in (
                (1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)
            ):
                raise SplitMegaConfigurationError(
                    f"{role} has unsupported Hopper cluster shape "
                    f"{impl.cluster_shape_mnk}."
                )
            if impl.use_2cta_instrs:
                raise SplitMegaConfigurationError(
                    f"{role} split MegaMoE currently supports only Hopper 1CTA MMA."
                )
            if not impl.force_static_sched:
                raise SplitMegaConfigurationError(
                    f"{role} split MegaMoE currently requires force_static_sched=True."
                )
            if impl.load_balance_mode != "static":
                raise SplitMegaConfigurationError(
                    f"{role} split MegaMoE currently requires "
                    "load_balance_mode='static'; K1/K2 execute concurrently "
                    "and cannot share one atomic task counter."
                )
            if impl.in_kernel_fc2_reduce:
                raise SplitMegaConfigurationError(
                    f"{role} must leave top-k reduction to standalone K3."
                )
            if impl.token_back_mode != "epi_warps":
                raise SplitMegaConfigurationError(
                    f"{role} must use direct epilogue combine, not dispatch token-back."
                )

        if self.fc1_impl.cluster_shape_mnk != self.fc2_impl.cluster_shape_mnk:
            raise SplitMegaConfigurationError(
                "K1 and K2 must use the same cluster shape for the shared "
                "pool/counter ABI; got "
                f"{self.fc1_impl.cluster_shape_mnk} and "
                f"{self.fc2_impl.cluster_shape_mnk}."
            )
        fc1_n = self.fc1_impl.mma_tiler_mnk[1]
        fc2_n = self.fc2_impl.mma_tiler_mnk[1]
        if fc1_n != fc2_n:
            if self.fc1_impl.cluster_shape_mnk != (1, 1, 1):
                raise SplitMegaConfigurationError(
                    "independent K1/K2 token-N currently requires the Hopper "
                    "1CTA cluster shape (1, 1, 1)."
                )
            if max(fc1_n, fc2_n) % min(fc1_n, fc2_n) != 0:
                raise SplitMegaConfigurationError(
                    "independent K1/K2 token-N tiles must have an integral "
                    f"ratio, got {fc1_n} and {fc2_n}."
                )
        cluster_size = _cluster_size(self.fc1_impl)
        for role, sm_count in (("K1", k1_sms), ("K2", k2_sms)):
            if sm_count % cluster_size != 0:
                raise SplitMegaConfigurationError(
                    f"{role} SM count {sm_count} is not divisible by cluster "
                    f"size {cluster_size}."
                )

    @property
    def token_padding_block(self) -> int:
        return max(
            self.fc1_impl.mma_tiler_mnk[1],
            self.fc2_impl.mma_tiler_mnk[1],
        ) * self.fc1_impl.cluster_shape_mnk[1]

    @property
    def handoff_token_n(self) -> Optional[int]:
        """Canonical completion block, or None for the legacy same-N ABI."""

        fc1_n = self.fc1_impl.mma_tiler_mnk[1]
        fc2_n = self.fc2_impl.mma_tiler_mnk[1]
        return max(fc1_n, fc2_n) if fc1_n != fc2_n else None

    @property
    def workspace_counter_tile_tokens(self) -> int:
        """Conservative counter capacity shared by both tactic layouts."""

        return min(
            self.fc1_impl.mma_tiler_mnk[1],
            self.fc2_impl.mma_tiler_mnk[1],
        ) * self.fc1_impl.cluster_shape_mnk[1]

    def impl_for(self, role: str) -> SplitMegaTacticLike:
        if role == "k1":
            return self.fc1_impl
        if role == "k2":
            return self.fc2_impl
        raise SplitMegaConfigurationError(
            f"role must be 'k1' or 'k2', got {role!r}."
        )

    def sm_count_for(self, role: str) -> int:
        if role == "k1":
            return self.k1_sm_count
        if role == "k2":
            return self.k2_sm_count
        raise SplitMegaConfigurationError(
            f"role must be 'k1' or 'k2', got {role!r}."
        )

    def max_active_clusters_for(self, role: str) -> int:
        return self.sm_count_for(role) // _cluster_size(self.impl_for(role))


def _dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype))


@dataclass(frozen=True)
class WorkspaceRegionContract:
    name: str
    byte_offset: int
    byte_size: int
    alignment: int
    dtype: str
    shape: Tuple[Any, ...]
    stride: Tuple[Any, ...]


@dataclass(frozen=True)
class SplitMegaWorkspaceContract:
    """Normalized local/shared workspace ABI for one split kernel role."""

    local_total_bytes: int
    shared_total_bytes: int
    local_zero_i32_count: int
    shared_zero_i32_count: int
    counter_epoch_banks: int
    local_counter_bank_spans: Tuple[Tuple[int, int], ...]
    shared_counter_bank_spans: Tuple[Tuple[int, int], ...]
    local_regions: Tuple[WorkspaceRegionContract, ...]
    shared_regions: Tuple[WorkspaceRegionContract, ...]

    @staticmethod
    def _regions(kernel: Any, space: str) -> Tuple[WorkspaceRegionContract, ...]:
        specs: Sequence[Any] = getattr(kernel, f"_{space}_region_specs")
        offsets: Mapping[str, int] = getattr(kernel, f"_{space}_offsets")
        return tuple(
            WorkspaceRegionContract(
                name=spec.name,
                byte_offset=int(offsets[spec.name]),
                byte_size=int(spec.nbytes),
                alignment=int(spec.align),
                dtype=_dtype_name(spec.cute_dtype),
                shape=tuple(spec.shape),
                stride=tuple(spec.stride_row_major),
            )
            for spec in specs
        )

    @classmethod
    def from_kernel(cls, kernel: Any) -> "SplitMegaWorkspaceContract":
        local_total, shared_total = kernel.get_workspace_sizes()
        counter_epoch_banks = int(getattr(kernel, "counter_epoch_banks", 1))
        local_spans = getattr(
            kernel, "local_counter_bank_spans",
            ((0, int(kernel.local_zero_i32_count) * 4),),
        )
        shared_spans = getattr(
            kernel, "shared_counter_bank_spans",
            ((0, int(kernel.shared_zero_i32_count) * 4),),
        )
        return cls(
            local_total_bytes=int(local_total),
            shared_total_bytes=int(shared_total),
            local_zero_i32_count=int(kernel.local_zero_i32_count),
            shared_zero_i32_count=int(kernel.shared_zero_i32_count),
            counter_epoch_banks=counter_epoch_banks,
            local_counter_bank_spans=tuple(local_spans),
            shared_counter_bank_spans=tuple(shared_spans),
            local_regions=cls._regions(kernel, "local"),
            shared_regions=cls._regions(kernel, "shared"),
        )

    @classmethod
    def require_compatible(
        cls, k1_kernel: Any, k2_kernel: Any
    ) -> "SplitMegaWorkspaceContract":
        k1 = cls.from_kernel(k1_kernel)
        k2 = cls.from_kernel(k2_kernel)
        if k1 != k2:
            differences = []
            for field_name in (
                "local_total_bytes",
                "shared_total_bytes",
                "local_zero_i32_count",
                "shared_zero_i32_count",
                "counter_epoch_banks",
                "local_counter_bank_spans",
                "shared_counter_bank_spans",
                "local_regions",
                "shared_regions",
            ):
                if getattr(k1, field_name) != getattr(k2, field_name):
                    differences.append(field_name)
            raise SplitMegaWorkspaceMismatch(
                "K1/K2 workspace ABI mismatch in: " + ", ".join(differences)
            )
        return k1

    def region(self, space: str, name: str) -> WorkspaceRegionContract:
        if space == "local":
            regions = self.local_regions
        elif space == "shared":
            regions = self.shared_regions
        else:
            raise ValueError(f"space must be 'local' or 'shared', got {space!r}.")
        for region in regions:
            if region.name == name:
                return region
        raise KeyError(f"{space} workspace has no region {name!r}.")

    def counter_bank_span(
        self, space: str, bank: int
    ) -> Tuple[int, int]:
        if (
            isinstance(bank, bool)
            or not isinstance(bank, int)
            or not 0 <= bank < self.counter_epoch_banks
        ):
            raise SplitMegaConfigurationError(
                "counter bank must select an existing bank, got "
                f"bank={bank!r}, banks={self.counter_epoch_banks}."
            )
        if space == "local":
            spans = self.local_counter_bank_spans
        elif space == "shared":
            spans = self.shared_counter_bank_spans
        else:
            raise ValueError(f"space must be 'local' or 'shared', got {space!r}.")
        if len(spans) != self.counter_epoch_banks:
            raise SplitMegaWorkspaceMismatch(
                f"{space} counter bank span count {len(spans)} does not match "
                f"counter_epoch_banks={self.counter_epoch_banks}."
            )
        return spans[bank]

    def counter_region(
        self, space: str, logical_name: str, bank: int
    ) -> WorkspaceRegionContract:
        self.counter_bank_span(space, bank)
        physical_name = logical_name if bank == 0 else f"{logical_name}__bank{bank}"
        return self.region(space, physical_name)


# These ordered tuples are the public compile/runtime ABI for the two split
# roles. Keep them intentionally smaller than the legacy fused __call__:
# K1 owns dispatch + FC1, while K2 owns FC2 + combine. In particular, neither
# role carries the other role's weights/scales merely to satisfy a shared host
# launcher signature.
_K1_RUNTIME_ARGUMENT_NAMES = (
    "activation",
    "activation_sf",
    "topk_idx",
    "topk_weights",
    "fc1_weight",
    "fc1_weight_sf",
    "fc1_weight_dequant_scale",
    "local_workspace",
    "shared_workspace",
    "peer_rank_ptr_mapper_host",
)

_K2_RUNTIME_ARGUMENT_NAMES = (
    "fc2_weight",
    "fc2_weight_sf",
    "fc2_weight_dequant_scale",
    "local_workspace",
    "shared_workspace",
    "peer_rank_ptr_mapper_host",
)

# Backward-compatible union used by CPU fixtures and callers that construct one
# shared argument dictionary. Extra entries in that dictionary are harmless;
# compile_requests filters it through the role tuple above.
_RUNTIME_ARGUMENT_NAMES = frozenset(
    _K1_RUNTIME_ARGUMENT_NAMES + _K2_RUNTIME_ARGUMENT_NAMES
)


def _runtime_argument_names_for_role(role: str) -> Tuple[str, ...]:
    if role == "k1":
        return _K1_RUNTIME_ARGUMENT_NAMES
    if role == "k2":
        return _K2_RUNTIME_ARGUMENT_NAMES
    raise SplitMegaConfigurationError(
        f"role must be 'k1' or 'k2', got {role!r}."
    )


def _split_entrypoint(kernel: Any, role: str) -> Any:
    """Resolve and audit the narrow CuteDSL entrypoint for one role."""

    entry_name = f"split_{role}_entry"
    entry = getattr(kernel, entry_name, None)
    if entry is None or not callable(entry):
        raise SplitMegaConfigurationError(
            f"{role.upper()} kernel must expose callable {entry_name}()."
        )

    expected = (
        *_runtime_argument_names_for_role(role),
        "max_active_clusters",
        "stream",
    )
    try:
        signature = inspect.signature(entry)
    except (TypeError, ValueError) as exc:
        raise SplitMegaConfigurationError(
            f"cannot inspect {role.upper()} entrypoint ABI: {exc}"
        ) from exc
    actual = tuple(signature.parameters)
    if actual != expected:
        raise SplitMegaConfigurationError(
            f"{role.upper()} entrypoint ABI mismatch: expected {expected}, "
            f"got {actual}."
        )
    return entry


@dataclass(frozen=True)
class SplitMegaCompileRequest:
    """One role's explicit input to ``cute.compile``."""

    role: str
    kernel: Any
    max_active_clusters: int
    kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = set(_runtime_argument_names_for_role(self.role))
        expected.update(("max_active_clusters", "stream"))
        actual = set(self.kwargs)
        unexpected = actual.difference(expected | {"options"})
        missing = expected.difference(actual)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if unexpected:
                details.append("unexpected=" + ",".join(sorted(unexpected)))
            raise SplitMegaConfigurationError(
                f"{self.role.upper()} compile ABI mismatch: "
                + "; ".join(details)
            )
        if self.kwargs["max_active_clusters"] != self.max_active_clusters:
            raise SplitMegaConfigurationError(
                f"{self.role.upper()} max_active_clusters field/kwarg mismatch."
            )

    def compile(self, cute_module: Any) -> Any:
        return cute_module.compile(self.kernel, **dict(self.kwargs))


class SplitMegaGraphExecutor(Protocol):
    """Interface implemented by the verified Green Context graph layer."""

    def launch(
        self,
        pair: "SplitMegaMxfp4KernelPair",
        k1_request: SplitMegaCompileRequest,
        k2_request: SplitMegaCompileRequest,
        *,
        k3: Any,
    ) -> Any:
        ...


@dataclass
class SplitMegaMxfp4KernelPair:
    """Constructed K1/K2 roles sharing one checked handoff workspace ABI."""

    plan: SplitMegaPlan
    k1_kernel: Any
    k2_kernel: Any
    workspace: SplitMegaWorkspaceContract
    hidden: int
    num_topk: int
    apply_topk_in_fc1: bool
    counter_epoch_bank: int = 0

    def get_workspace_sizes(self) -> Tuple[int, int]:
        return (
            self.workspace.local_total_bytes,
            self.workspace.shared_total_bytes,
        )

    def selected_counter_bank_span(self, space: str) -> Tuple[int, int]:
        return self.workspace.counter_bank_span(
            space, self.counter_epoch_bank
        )

    def compile_requests(
        self,
        runtime_arguments: Mapping[str, Any],
        *,
        k1_stream: Any,
        k2_stream: Any,
        options: Optional[str] = None,
    ) -> Tuple[SplitMegaCompileRequest, SplitMegaCompileRequest]:
        """Create K1/K2 compile requests for two distinct Green streams."""
        if k1_stream is k2_stream or k1_stream == k2_stream:
            raise SplitMegaConfigurationError(
                "K1 and K2 require distinct Green Context streams."
            )

        def request(role: str, kernel: Any, stream: Any) -> SplitMegaCompileRequest:
            argument_names = _runtime_argument_names_for_role(role)
            missing = sorted(set(argument_names).difference(runtime_arguments))
            if missing:
                raise SplitMegaConfigurationError(
                    f"missing {role.upper()} runtime arguments: "
                    + ", ".join(missing)
                )
            kwargs = {
                name: runtime_arguments[name]
                for name in argument_names
            }
            kwargs["max_active_clusters"] = self.plan.max_active_clusters_for(role)
            kwargs["stream"] = stream
            if options is not None:
                kwargs["options"] = options
            return SplitMegaCompileRequest(
                role=role,
                kernel=_split_entrypoint(kernel, role),
                max_active_clusters=self.plan.max_active_clusters_for(role),
                kwargs=kwargs,
            )

        return (
            request("k1", self.k1_kernel, k1_stream),
            request("k2", self.k2_kernel, k2_stream),
        )

    def combine_quant_view(self, shared_workspace: torch.Tensor) -> torch.Tensor:
        """Return K3's BF16 ``(token, topk, hidden)`` staging view."""
        region = self.workspace.region("shared", "combine_quant")
        if region.dtype not in ("BFloat16", "cutlass.BFloat16"):
            raise SplitMegaWorkspaceMismatch(
                "Hopper split K3 currently requires a BF16 combine_quant "
                f"region, got {region.dtype}."
            )
        if shared_workspace.dtype is not torch.uint8 or shared_workspace.ndim != 1:
            raise SplitMegaConfigurationError(
                "shared_workspace must be a flat torch.uint8 tensor."
            )
        if shared_workspace.numel() < self.workspace.shared_total_bytes:
            raise SplitMegaConfigurationError(
                f"shared_workspace has {shared_workspace.numel()} bytes, needs "
                f"{self.workspace.shared_total_bytes}."
            )
        return (
            shared_workspace.narrow(
                0, region.byte_offset, region.byte_size
            )
            .view(torch.bfloat16)
            .reshape(region.shape)
        )

    def join_counter_view(self, shared_workspace: torch.Tensor) -> torch.Tensor:
        """Return the peer-visible split K2 completion epoch counter."""
        region = self.workspace.counter_region(
            "shared", "split_k2_join_count", self.counter_epoch_bank
        )
        if region.dtype not in ("Int32", "cutlass.Int32"):
            raise SplitMegaWorkspaceMismatch(
                "split K2/K3 join requires an Int32 counter, got "
                f"{region.dtype}."
            )
        if region.shape != (1,) or region.byte_size != 4:
            raise SplitMegaWorkspaceMismatch(
                "split K2/K3 join counter must be exactly one Int32, got "
                f"shape={region.shape}, bytes={region.byte_size}."
            )
        if shared_workspace.dtype is not torch.uint8 or shared_workspace.ndim != 1:
            raise SplitMegaConfigurationError(
                "shared_workspace must be a flat torch.uint8 tensor."
            )
        if shared_workspace.numel() < self.workspace.shared_total_bytes:
            raise SplitMegaConfigurationError(
                f"shared_workspace has {shared_workspace.numel()} bytes, needs "
                f"{self.workspace.shared_total_bytes}."
            )
        return (
            shared_workspace.narrow(0, region.byte_offset, region.byte_size)
            .view(torch.int32)
            .reshape(region.shape)
        )

    def reset_barrier_signal_view(
        self, shared_workspace: torch.Tensor
    ) -> torch.Tensor:
        """Return the persistent two-slot NVLink barrier signal."""

        region = self.workspace.region("shared", "nvlink_barrier_signal")
        if region.dtype not in ("Int32", "cutlass.Int32") or region.shape != (2,):
            raise SplitMegaWorkspaceMismatch(
                "split K0 reset barrier signal must be Int32[2], got "
                f"dtype={region.dtype}, shape={region.shape}."
            )
        return (
            shared_workspace.narrow(0, region.byte_offset, region.byte_size)
            .view(torch.int32)
            .reshape(region.shape)
        )

    def reset_barrier_phase_view(
        self, local_workspace: torch.Tensor
    ) -> torch.Tensor:
        """Return the persistent local phase counter used by K0 and K1."""

        region = self.workspace.region("local", "nvlink_barrier_counter")
        if region.dtype not in ("Int32", "cutlass.Int32") or region.shape != (1,):
            raise SplitMegaWorkspaceMismatch(
                "split K0 reset phase must be one Int32, got "
                f"dtype={region.dtype}, shape={region.shape}."
            )
        return (
            local_workspace.narrow(0, region.byte_offset, region.byte_size)
            .view(torch.int32)
            .reshape(region.shape)
        )

    def make_k3(self, *, sm_arch: str) -> Any:
        """Construct standalone TopkReduce; compilation belongs to executor."""
        from moe_nvfp4_swapab.topk_reduce import TopkReduce

        combine_format = self.k1_kernel.combine_format
        return TopkReduce(
            self.hidden,
            self.num_topk,
            combine_format,
            sm_arch=sm_arch,
        )

    def launch(
        self,
        executor: Optional[SplitMegaGraphExecutor],
        k1_request: SplitMegaCompileRequest,
        k2_request: SplitMegaCompileRequest,
        *,
        k3: Any,
    ) -> Any:
        """Delegate only to an explicitly supplied, verified graph executor."""
        if executor is None:
            raise SplitMegaExecutorRequired(
                "PR357-style launch requires a verified Green Context CUDA "
                "Graph executor; sequential K1->K2 fallback is intentionally disabled."
            )
        return executor.launch(self, k1_request, k2_request, k3=k3)


def build_mxfp4_split_kernel_pair(
    problem: Any,
    plan: SplitMegaPlan,
    *,
    rank: int,
    kind: str = "fp8_e4m3",
    fp8_scale_mode: str = "mxfp4_hybrid",
    fp8_accum_mode: str = "1xacc",
    apply_topk_in_fc1: bool = True,
    kernel_class: Optional[type] = None,
    ab_dtype: Any = None,
    sf_padding_block: Optional[int] = None,
    counter_epoch_banks: int = 1,
    counter_epoch_bank: int = 0,
) -> SplitMegaMxfp4KernelPair:
    """Instantiate purpose-built split K1/K2 kernels with independent tactics.

    ``kernel_class``, ``ab_dtype`` and ``sf_padding_block`` are injectable to
    keep the host contract unit-testable without compiling a CUDA kernel.
    Production callers should leave them unset.
    """
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise SplitMegaConfigurationError(f"rank must be an integer, got {rank!r}.")
    if rank < 0 or rank >= int(problem.world_size):
        raise SplitMegaConfigurationError(
            f"rank {rank} is outside world_size={problem.world_size}."
        )
    if (
        isinstance(counter_epoch_banks, bool)
        or not isinstance(counter_epoch_banks, int)
        or counter_epoch_banks not in (1, 2)
    ):
        raise SplitMegaConfigurationError(
            "counter_epoch_banks must be 1 or 2, got "
            f"{counter_epoch_banks!r}."
        )
    if (
        isinstance(counter_epoch_bank, bool)
        or not isinstance(counter_epoch_bank, int)
        or not 0 <= counter_epoch_bank < counter_epoch_banks
    ):
        raise SplitMegaConfigurationError(
            "counter_epoch_bank must select an existing bank, got "
            f"bank={counter_epoch_bank!r}, banks={counter_epoch_banks}."
        )

    if kind not in ("fp8_e4m3", "mxfp8_e4m3"):
        raise SplitMegaConfigurationError(
            "MXFP4 split MegaMoE requires an E4M3 activation kind."
        )
    if fp8_scale_mode != "mxfp4_hybrid":
        raise SplitMegaConfigurationError(
            "PR357-style MXFP4 split currently requires mxfp4_hybrid "
            "(FC1 per-token, FC2 K64) scaling."
        )
    if fp8_accum_mode != "1xacc":
        raise SplitMegaConfigurationError(
            "MXFP4 split MegaMoE currently requires fp8_accum_mode='1xacc'."
        )

    if kernel_class is None:
        from moe_hopper_fp8.megamoe_kernel_fp8 import (
            Sm90MegaMoESwapABMxfp4Fp8Kernel,
        )

        kernel_class = Sm90MegaMoESwapABMxfp4Fp8Kernel
    if ab_dtype is None:
        from moe_hopper_fp8.hopper_moe_utils import fp8_kind_to_cutlass_dtype

        ab_dtype = fp8_kind_to_cutlass_dtype(kind)
    if sf_padding_block is None:
        from common.megamoe_constants import SfPaddingBlock

        sf_padding_block = SfPaddingBlock

    static_expert_shape = (
        int(problem.num_experts_per_rank),
        int(problem.intermediate),
        int(problem.hidden),
    )

    def build(role: str) -> Any:
        impl = plan.impl_for(role)
        max_active_clusters = plan.max_active_clusters_for(role)
        group_hint = (
            impl.group_hint
            if impl.group_hint is not None
            else 3 * max_active_clusters
        )
        kwargs = dict(
            mma_tiler_mnk=impl.mma_tiler_mnk,
            cluster_shape_mnk=impl.cluster_shape_mnk,
            use_2cta_instrs=impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=plan.token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=impl.force_static_sched,
            clc_bundle_size=impl.clc_bundle_size,
            num_sched_stages=impl.num_sched_stages,
            ab_dtype=ab_dtype,
            fp8_scale_mode=fp8_scale_mode,
            fp8_accum_mode=fp8_accum_mode,
            world_size=int(problem.world_size),
            local_rank=rank,
            num_topk=int(problem.num_topk),
            max_tokens_per_rank=int(problem.num_tokens_per_rank),
            hidden=int(problem.hidden),
            fc2_in_kernel_topk_reduce=False,
            apply_topk_in_fc1=apply_topk_in_fc1,
            # PR4688 exposes the three-mode token-back API publicly; the
            # donor's boolean spelling is the equivalent epilogue-warp mode.
            token_back_mode="epi_warps",
            epi_flag_batch=impl.epi_flag_batch,
            flag_batch=impl.flag_batch,
            gate_up_clamp=getattr(problem, "gate_up_clamp", None),
            split_role=role,
            split_fc1_tile_m=(
                plan.fc1_impl.mma_tiler_mnk[0] if role == "k2" else None
            ),
            split_fc1_token_n=plan.fc1_impl.mma_tiler_mnk[1],
            split_handoff_token_n=plan.handoff_token_n,
            split_workspace_counter_tile_tokens=(
                plan.workspace_counter_tile_tokens
            ),
            split_counter_epoch_banks=counter_epoch_banks,
            split_counter_epoch_bank=counter_epoch_bank,
        )
        return kernel_class(**kwargs)

    k1_kernel = build("k1")
    k2_kernel = build("k2")
    workspace = SplitMegaWorkspaceContract.require_compatible(
        k1_kernel, k2_kernel
    )
    return SplitMegaMxfp4KernelPair(
        plan=plan,
        k1_kernel=k1_kernel,
        k2_kernel=k2_kernel,
        workspace=workspace,
        hidden=int(problem.hidden),
        num_topk=int(problem.num_topk),
        apply_topk_in_fc1=bool(apply_topk_in_fc1),
        counter_epoch_bank=counter_epoch_bank,
    )


__all__ = [
    "SplitMegaCompileRequest",
    "SplitMegaConfigurationError",
    "SplitMegaExecutorRequired",
    "SplitMegaGraphExecutor",
    "SplitMegaMxfp4KernelPair",
    "SplitMegaPlan",
    "SplitMegaTactic",
    "SplitMegaTacticLike",
    "SplitMegaWorkspaceContract",
    "SplitMegaWorkspaceMismatch",
    "WorkspaceRegionContract",
    "build_mxfp4_split_kernel_pair",
]
