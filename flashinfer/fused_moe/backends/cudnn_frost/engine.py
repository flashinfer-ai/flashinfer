# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Shared host engine for generated Frost MoE kernels.

Dtype adapters own tensor contracts, scale layouts and native executor construction.
The engine owns fixed compiler/artifact metadata and bounded bucket preparation;
each packed call owns its launch state independently of these lookup caches.
Like the surrounding MoE runners, use one engine per thread and stream.
"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import torch

from ....autotuner import TuningConfig
from ...utils import map_to_hybrid_bucket
from . import runtime
from .cache import LRUCache, retain_graph_resources
from .compiler import (
    capture_compiler_snapshot,
    require_moe_assembler,
    use_compiler_snapshot,
)


class PreparationMetadata(NamedTuple):
    first: tuple
    second: tuple
    fma_tactic: Any
    plan_key: tuple
    minimum_tokens: int = 0
    maximum_tokens: int = 0
    validate_fma_sources: Callable | None = None


class _SealedKernel(NamedTuple):
    """Keep a selected kernel's tactic bound to its engine's compiler snapshot."""

    original: Any
    tactic: tuple

    def __getattr__(self, name):
        return getattr(self.original, name)


class SelectionSnapshot(NamedTuple):
    """Selectors bound to a detached catalog and fixed compiler identity."""

    select_kernels: Callable
    select_fma: Callable
    validate_fma_sources: Callable | None = None


def seal_kernels(kernels):
    """Detach metadata and seal every tactic inside the compiler snapshot."""
    return tuple(_SealedKernel(deepcopy(kernel), kernel.tactic) for kernel in kernels)


def make_kernel_selector(
    first_catalog,
    second_catalog,
    fused_catalog,
    tables,
    arch,
    name,
    sm_count,
):
    """Return shape selection over captured catalogs and shortlist profiles."""
    from .catalog import can_implement, select_kernels as select_catalog

    def matching(pool, rows, hidden, intermediate, experts, first):
        n, k = (intermediate, hidden) if first else (hidden, intermediate)
        dimensions = dict(s=rows, n=n, k=k, experts=experts, groups=experts)
        return tuple(
            kernel
            for kernel in pool
            if kernel.arch == arch
            and (not first or kernel.activation == name)
            and can_implement(kernel, dimensions)
        )

    def select_kernels(shape):
        tokens, hidden, intermediate, experts, topk = shape
        first = matching(
            first_catalog, tokens * topk, hidden, intermediate, experts, True
        )
        second = matching(
            second_catalog, tokens * topk, hidden, intermediate, experts, False
        )
        profiles = tables.get((arch, name, experts, hidden, intermediate, topk))
        fused = matching(
            fused_catalog, tokens * topk, hidden, intermediate, experts, True
        )
        return select_catalog(
            first,
            second,
            fused,
            profiles,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            sm_count=sm_count,
        )

    return select_kernels


def capture_fma_selection(module, name, arch):
    """Bind FMA availability/tactic, checking source drift only for cold plans."""
    if arch != "sm_107a":
        return lambda shape: None, None
    digest = module.source_digest
    digest_function = digest.func if isinstance(digest, partial) else digest
    fresh_digest = (
        partial(digest_function.__wrapped__, *digest.args, **(digest.keywords or {}))
        if isinstance(digest, partial)
        else digest_function.__wrapped__
    )
    # Explicit recapture must not reuse a previously generated FMA body or
    # its validation receipt. Old native plans still own their loaded kernels.
    digest_function.cache_clear()
    for key in ("_requirements", "_source"):
        digest_function.__wrapped__.__globals__[key].cache_clear()
    try:
        module.check_support(name)
    except NotImplementedError:
        return lambda shape: None, None
    expected = digest()
    tactic = module.tactic(name)
    supported = module.supported

    def select_fma(shape):
        return tactic if supported(*shape, name) else None

    def validate_sources():
        if fresh_digest() != expected:
            raise RuntimeError(
                "Frost FMA sources changed after engine build; refresh or rebuild "
                "the runner outside CUDA Graph capture"
            )

    return select_fma, validate_sources


class PackedInputs(list):
    """Keep a call's bucket workspace and executors alongside its tensors."""

    def __init__(self, values, launch_state, tuning_config=None):
        super().__init__(values)
        self.launch_state = launch_state
        self._tuning_config = tuning_config

    @property
    def tuning_config(self):
        """Create profiling hooks only when the autotuner requests them."""
        if self._tuning_config is None:
            # Keep valid quantized bytes/scales and the real routing distribution.
            # Capture tensor references without a cycle back to this packed call.
            values = self[1:]

            def preserve_inputs(tensors):
                tensors[1:] = values
                return tensors

            self._tuning_config = TuningConfig(
                use_cuda_graph=True,
                cuda_graph_profile_replays=3,
                inputs_pre_hook=preserve_inputs,
            )
        return self._tuning_config


def allocate_workspace(required: int, device, workspace_pool):
    """Reuse geometric storage while old calls and graphs own their slices."""
    capacity = min((size for size in workspace_pool if size >= required), default=0)
    if not capacity:
        capacity = 1 << (max(required, 1) - 1).bit_length()
        workspace_pool[capacity] = torch.empty(
            capacity, dtype=torch.uint8, device=device
        )
    return workspace_pool[capacity][:required]


class FrostEngine:
    """Reuse fixed native executors with bucket kernel choices and workspace.

    Construction and explicit refresh bind the assembler and artifact catalog.
    Shared cache changes are adopted only on explicit refresh or reconstruction.
    Actual compilation checks its pinned identity before and after compiling;
    already loaded plans remain usable.
    """

    def __init__(
        self,
        device,
        *,
        artifact_roots: Callable,
        artifact_version: Callable,
        refresh_artifacts: Callable,
        policy_version: str,
        tactic_tag: str,
        build_plan: Callable,
        select_kernels: Callable | None = None,
        select_fma: Callable | None = None,
        capture_selection: Callable | None = None,
        require_assembler: bool = False,
        token_ceiling: int | None = None,
    ):
        if token_ceiling is not None and token_ceiling < 1:
            raise ValueError("Frost token ceiling must be positive")
        self.device = torch.device(device)
        self.arch = runtime._arch_for(self.device)
        self.arch_number = int(self.arch.removeprefix("sm_").removesuffix("a"))
        self.sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self._artifact_roots = artifact_roots
        self._artifact_version = artifact_version
        self._refresh_artifacts = refresh_artifacts
        self._policy_version = policy_version
        self._tactic_tag = tactic_tag
        self._legacy_selection: SelectionSnapshot | None = None
        if capture_selection is None:
            if select_kernels is None or select_fma is None:
                raise ValueError("Frost engine requires its captured selectors")
            self._legacy_selection = SelectionSnapshot(select_kernels, select_fma)
        self._capture_selection = capture_selection
        self._build_plan = build_plan
        self._require_assembler = require_assembler
        self._token_ceiling = token_ceiling
        self._generation = 0
        self._metadata = LRUCache(maxsize=128)
        self.plans = LRUCache()
        self.executors = LRUCache(maxsize=128)
        self.workspace_pool: dict[int, torch.Tensor] = {}
        self._capture_snapshot()

    def _capture_snapshot(self):
        context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            self._capture_device_snapshot()

    def _capture_device_snapshot(self):
        # Read and validate the new binding before dropping any lookup entries.
        # No tensors, native modules or graph ownership are kept in this key.
        compiler = capture_compiler_snapshot()
        with use_compiler_snapshot(compiler):
            if self._require_assembler:
                require_moe_assembler()
            roots = tuple(self._artifact_roots())
            version = self._artifact_version()
            digest = runtime._tactic_digest(self._policy_version)
            selection = (
                self._legacy_selection
                if self._capture_selection is None
                else self._capture_selection(roots, self.arch, self.sm_count)
            )
            if selection is None:
                raise ValueError("Frost engine requires its captured selectors")
        self.compiler_snapshot = compiler
        self.artifact_roots = roots
        self.artifact_version = version
        self._selection = selection
        self._select_kernels = selection.select_kernels
        self._select_fma = selection.select_fma
        self._generation += 1
        device_key = ()
        if self.device.type == "cuda" and torch.cuda.is_available():
            from .device import device_facts

            index = (
                torch.cuda.current_device()
                if self.device.index is None
                else self.device.index
            )
            device_key = device_facts(index)
        self._snapshot_key = (
            self._generation,
            roots,
            version,
            self.arch,
            self.sm_count,
            digest,
            device_key,
        )
        self._metadata.clear()
        self.plans.clear()
        self.executors.clear()

    def refresh(self):
        """Rebind the compiler and artifact catalog for subsequent preparations.

        Only lookup ownership is released. Previously packed calls and captured
        graphs continue to own their original native modules and workspace.
        """
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Refresh cuDNN Frost MoE outside CUDA Graph capture")
        self._refresh_artifacts()
        self._capture_snapshot()

    def _bucket_bounds(self, bucket: int) -> tuple[int, int]:
        """Find actual members of a standard bucket, including a partial ceiling.

        The mapper is monotone below the ceiling. Its final capped value can
        drop below a power-of-two bucket (T=5/6 -> 8, T=7 -> 7), so search that
        prefix and handle the ceiling separately. This runs only on a cold
        bucket catalog, without enumerating its token counts.
        """
        ceiling = self._token_ceiling
        if ceiling is None:
            return bucket, bucket
        low, high = 1, ceiling
        while low < high:
            middle = (low + high) // 2
            if map_to_hybrid_bucket(middle, ceiling) < bucket:
                low = middle + 1
            else:
                high = middle
        first = low
        if first == ceiling or map_to_hybrid_bucket(first, ceiling) != bucket:
            if bucket != ceiling:
                raise ValueError(f"Unknown Frost token bucket: {bucket}")
            return ceiling, ceiling
        low, high = first, ceiling
        while low < high:
            middle = (low + high) // 2
            if map_to_hybrid_bucket(middle, ceiling) <= bucket:
                low = middle + 1
            else:
                high = middle
        return first, ceiling if bucket == ceiling else low - 1

    @staticmethod
    def _covers_bucket(kernel, minimum: int, maximum: int, topk: int) -> bool:
        """Keep only generated kernels valid at every expanded-row count."""
        rule = kernel.contract.get("s")
        if minimum == maximum:
            return runtime._dimension_matches(minimum * topk, rule)
        if not isinstance(rule, dict):
            return False
        multiple = int(rule.get("multiple_of", 1))
        return (
            runtime._dimension_matches(minimum * topk, rule)
            and runtime._dimension_matches(maximum * topk, rule)
            and multiple > 0
            and topk % multiple == 0
        )

    def prepare(self, shape: tuple) -> PreparationMetadata:
        """Reuse one catalog and workspace capacity for the entire token bucket."""
        bucket = (
            shape[0]
            if self._token_ceiling is None
            else map_to_hybrid_bucket(shape[0], self._token_ceiling)
        )
        bucket_shape = (bucket, *shape[1:])
        metadata = self._metadata.get(bucket_shape)
        if metadata is None:
            minimum, maximum = self._bucket_bounds(bucket)
            # A partial ceiling may leave the mapper's bucket ID above every
            # actual member. Select and allocate at the largest legal member.
            selection_shape = (maximum, *shape[1:])
            with use_compiler_snapshot(self.compiler_snapshot):
                first, second = self._select_kernels(selection_shape)
                if self._token_ceiling is not None:
                    first = tuple(
                        kernel
                        for kernel in first
                        if self._covers_bucket(kernel, minimum, maximum, shape[4])
                    )
                    second = tuple(
                        kernel
                        for kernel in second
                        if self._covers_bucket(kernel, minimum, maximum, shape[4])
                    )
                fma_tactic = (
                    self._select_fma(selection_shape) if first and second else None
                )
                if fma_tactic is not None and self._token_ceiling is not None:
                    # The built-in FMA scopes are contiguous (T<=4 or T=1).
                    for tokens in (minimum, maximum):
                        if self._select_fma((tokens, *shape[1:])) != fma_tactic:
                            fma_tactic = None
                            break
                first = tuple(
                    kernel
                    if isinstance(kernel, _SealedKernel)
                    else _SealedKernel(kernel, kernel.tactic)
                    for kernel in first
                )
                second = tuple(
                    kernel
                    if isinstance(kernel, _SealedKernel)
                    else _SealedKernel(kernel, kernel.tactic)
                    for kernel in second
                )
            metadata = PreparationMetadata(
                first,
                second,
                fma_tactic,
                bucket_shape
                + (
                    self._snapshot_key,
                    tuple(a.tactic for a in first),
                    tuple(b.tactic for b in second),
                    fma_tactic,
                ),
                minimum,
                maximum,
                self._selection.validate_fma_sources,
            )
            self._metadata[bucket_shape] = metadata
        return metadata

    def state(self, shape: tuple, metadata: PreparationMetadata | None = None):
        """Reuse a bucket workspace and fixed native executors across token counts."""
        if metadata is None:
            bucket = (
                shape[0]
                if self._token_ceiling is None
                else map_to_hybrid_bucket(shape[0], self._token_ceiling)
            )
            bucket_shape = (bucket, *shape[1:])
            state = self.plans.get(bucket_shape)
            if state is not None:
                return state
            metadata = self.prepare(shape)
        else:
            # Explicit metadata callers must not reuse receipts from an older
            # binding. Ordinary packing goes straight to the bucket state cache.
            bucket_shape = metadata.plan_key[: len(shape)]
            if (
                bucket_shape[1:] != shape[1:]
                or not metadata.minimum_tokens <= shape[0] <= metadata.maximum_tokens
                or metadata.plan_key[len(shape)] != self._snapshot_key
            ):
                raise ValueError(
                    "Frost preparation changed; prepare the input packs again"
                )
        if not metadata.first or not metadata.second:
            raise ValueError("No legal cuDNN Frost stage candidates for this problem")
        state = self.plans.get(bucket_shape)
        if state is None:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Prepare cuDNN Frost MoE outside CUDA Graph capture"
                    )
                with use_compiler_snapshot(self.compiler_snapshot):
                    capacity_shape = (metadata.maximum_tokens, *shape[1:])
                    state = self._build_plan(
                        capacity_shape, metadata, self.workspace_pool, self.executors
                    )
                # Tuning identity belongs to the packed state, including after
                # refresh. Reading the current engine would relabel old calls.
                state.preparation_key = metadata.plan_key
            self.plans[bucket_shape] = state
        return state

    def valid_tactics(self, inputs, *, shape=None, launch_state=None):
        """Return packed-state tactics or metadata-only profiling candidates."""
        state = (
            launch_state
            if launch_state is not None
            else getattr(inputs, "launch_state", None)
        )
        if state is not None:
            return list(state.launches)
        if shape is None:
            raise ValueError("Frost profiling inputs require their exact shape")
        metadata = self.prepare(shape)
        tactics = [
            (self._tactic_tag, a.tactic, b.tactic)
            for a, b in product(metadata.first, metadata.second)
        ]
        if metadata.fma_tactic is not None:
            tactics.append(metadata.fma_tactic)
        return tactics

    def forward(self, inputs, tactic=-1, *, launch_state=None):
        """Launch the state owned by this packed call, including after refresh."""
        state = (
            launch_state
            if launch_state is not None
            else getattr(inputs, "launch_state", None)
        )
        if state is None:
            raise ValueError(
                "Preserve packed cuDNN Frost inputs or pass their launch_state"
            )
        if tactic == -1:
            tactic = next(iter(state.launches))
        try:
            launch = state.launches[tactic]
        except KeyError:
            raise ValueError(
                f"Unknown or stale cuDNN Frost MoE tactic: {tactic!r}"
            ) from None
        retain_graph_resources(state, inputs)
        launch(*inputs, state.workspace)
        return inputs[0]


class FrostRunnerMixin:
    """Shared runner operations; dtype adapters prepare the packed tensors."""

    if TYPE_CHECKING:
        # Supplied by the dtype adapter and its MoERunner base at runtime.
        _engine: Any
        config: Any
        _input_support: Callable

        def _require_built(self) -> None: ...

    def _accepts_shape(self, act):
        return self._input_support(self.config, act, self._engine.arch_number)

    def refresh(self):
        """Refresh fixed compiler/artifact metadata without releasing live calls."""
        self._require_built()
        self._engine.refresh()

    def forward(
        self,
        inputs,
        tactic: Any = -1,
        do_preparation=False,
        *,
        launch_state=None,
        **kwargs: Any,
    ):
        self._require_built()
        return self._engine.forward(inputs, tactic, launch_state=launch_state)
