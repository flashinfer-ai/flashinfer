"""MoEEpMegaLayer — fused mega-MoE kernel path."""

from __future__ import annotations

import dataclasses
import warnings
import weakref
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from ..config import BootstrapConfig, FleetParams
from ..core.kernel.registry import create_mega_kernel
from ..core.runtime import (
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from ..core.validation.common import (
    MoEEpConfigError,
    ensure_bootstrap_dist_validated,
    validate_bootstrap_world_size,
    validate_fleet_weights,
)
from ..weights import MoEWeightPack
from .config import MegaConfig

if TYPE_CHECKING:
    from ..core.kernel.base import MegaKernelBackend
    from ..tensors import MoEEpTensors


class MoEEpMegaWorkspace:
    """Capacity-profile handle owned by one MegaMoE layer.

    Create handles with :meth:`MoEEpMegaLayer.create_workspace`; do not
    construct them directly. The selected workspace keeps stable addresses
    until :meth:`destroy` or layer destruction. A stable returned-output
    address additionally requires ``return_workspace_view=True`` on a backend
    that supports output views.

    Creation and destruction are EP collectives. Before destroying a handle,
    callers must synchronize all work and retire every CUDA graph that uses it.
    """

    def __init__(
        self,
        layer: "MoEEpMegaLayer",
        fleet_params: FleetParams,
        backend_workspace: Any,
        *,
        kernel: MegaKernelBackend | None = None,
    ) -> None:
        self._layer_ref = weakref.ref(layer)
        self._fleet_params = fleet_params
        self._backend_workspace = backend_workspace
        self._kernel = kernel if kernel is not None else layer._kernel
        self._destroyed = False

    @property
    def max_tokens_per_rank(self) -> int:
        """Maximum live-token count accepted by this handle."""

        return self._fleet_params.max_tokens_per_rank

    @property
    def live_tokens(self) -> int | None:
        """Offline workload bucket; does not limit accepted input shapes."""
        return self._fleet_params.tuning_live_tokens

    @property
    def is_destroyed(self) -> bool:
        """Whether this handle has released its backing workspace."""

        return self._destroyed

    def destroy(self) -> None:
        """Collectively release this profile; idempotent on every rank."""

        if self._destroyed:
            return
        layer = self._layer_ref()
        if layer is None:
            raise MoEEpConfigError(
                "owning MegaMoE layer no longer exists; collective workspace "
                "destruction cannot be completed"
            )
        layer._destroy_workspace_handle(self)


class MoEEpMegaLayer(nn.Module):
    """Fused EP mega kernel — no separate dispatch/combine transport.

    Memory invariant: the source ``MoEWeightPack`` is released as soon as the
    kernel's transformed weights exist — the transformed tensors own the
    memory. Retaining the pack would hold a per-layer dequant copy (multiple
    GB at large-model geometry) across every MoE layer and OOM at model load.
    When ``backend.transformed_weights`` is supplied, the source pack is never
    stored at all.

    CUDA graphs: call :meth:`warmup` on ALL EP ranks first, then capture
    ``forward``. Under capture the output tensor returned at capture time is
    the one the graph writes on every replay — consume that same tensor
    across replays (standard graph practice). Lazy compile/alloc/autotune
    paths raise if they would fire mid-capture instead of corrupting it.
    """

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        weights: MoEWeightPack,
        backend: MegaConfig,
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._mega_config = backend
        self._megakernel_config = backend.megakernel

        ensure_moe_ep_cuda_device(bootstrap)

        self._kernel = create_mega_kernel(self._megakernel_config)
        self._kernel.bind_ep_bootstrap(bootstrap)
        self._runtime = None
        self._workspace: Any = None
        # Check backend geometry and compiler capabilities before collective
        # runtime initialization.
        self._kernel.validate_init(bootstrap, fleet_params)
        if bootstrap.auto_bootstrap:
            self._runtime = bootstrap_moe_ep_runtime(
                bootstrap,
                self._kernel.runtime_requirements(bootstrap),
            )

        validate_bootstrap_world_size(bootstrap)

        if backend.transformed_weights is None:
            validate_fleet_weights(weights, fleet_params, bootstrap.world_size)

        self._weights: Optional[MoEWeightPack] = (
            weights if backend.transformed_weights is None else None
        )
        self._transformed: Optional[Any] = None
        self._default_workspace_handle: MoEEpMegaWorkspace | None = None
        # Strong tracking is intentional: workspace create/destroy is an EP
        # collective and therefore must never be triggered by rank-local GC.
        # Dict insertion order gives every EP rank the same reverse-allocation
        # collective teardown order. One live profile per capacity also avoids
        # presenting pooled aliases as independent concurrency lanes.
        self._workspaces: dict[tuple[int, int | None], MoEEpMegaWorkspace] = {}
        self._preprocessing_count = 0
        self._destroyed = False
        self._bootstrap_validated = False
        self._forward_signatures: dict[int, tuple[Any, ...]] = {}
        self._staged_kernel: MegaKernelBackend | None = None
        self._staged_workspace: Any = None
        self._staged_transformed: Any = None

        if backend.transformed_weights is not None:
            self._transformed = backend.transformed_weights
            self._kernel.validate_transformed_weights(
                self._transformed,
                self._bootstrap,
                self._fleet_params,
            )
        elif backend.preprocess_weights:
            self._preprocess_weights()

    def _preprocess_weights(self) -> None:
        if self._transformed is not None:
            return
        assert self._weights is not None, (
            "source weight pack was released but no transformed weights exist"
        )
        self._transformed = self._kernel.preprocess_weights(
            self._weights, self._fleet_params
        )
        self._preprocessing_count += 1
        self._weights = None

    def _ensure_workspace(self) -> Any:
        if self._default_workspace_handle is not None:
            return self._default_workspace_handle._backend_workspace
        if self._workspace is None:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise MoEEpConfigError(
                    "mega workspace allocation (symmetric heap) cannot run "
                    "during CUDA graph capture; call warmup() on all EP ranks "
                    "before capturing"
                )
            self._workspace = self._kernel.prepare_workspace(
                self._bootstrap, self._fleet_params
            )
        return self._workspace

    def live_token_buckets(self, max_tokens_per_rank: int) -> list[int]:
        """Read offline live-token profiles before capture (same cache on all ranks)."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError("Read live-token profiles before CUDA graph capture")
        return self._kernel.live_token_buckets(
            dataclasses.replace(
                self._fleet_params,
                max_tokens_per_rank=max_tokens_per_rank,
                tuning_live_tokens=None,
            )
        )

    def create_workspace(
        self,
        max_tokens_per_rank: int,
        *,
        live_tokens: int | None = None,
        knobs: dict[str, Any] | None = None,
    ) -> MoEEpMegaWorkspace:
        """Allocate one reusable capacity profile for this layer.

        The handle reuses this layer's transformed weights. ``knobs=None``
        inherits the layer's configuration. A dictionary selects fixed knobs
        for this handle and is copied before allocation. The layer must use
        fixed knobs, and its default workspace must not already be allocated
        when overriding that capacity.

        ``live_tokens`` selects an offline workload profile without shrinking
        capacity. Explicit knobs take precedence over that profile's cache.
        Call collectively with matching arguments on all EP ranks before
        capture. Only one live handle per (capacity, live_tokens) is allowed.
        """

        if isinstance(max_tokens_per_rank, bool) or not isinstance(
            max_tokens_per_rank, int
        ):
            raise TypeError("max_tokens_per_rank must be an int")
        if max_tokens_per_rank <= 0:
            raise ValueError("max_tokens_per_rank must be positive")
        if self._destroyed:
            raise MoEEpConfigError("MegaMoE layer has been destroyed")
        if getattr(self._megakernel_config, "knobs", None) == "auto":
            raise MoEEpConfigError(
                "create_workspace() requires fixed/offline-tuned knobs; "
                "knobs='auto' has mutable per-capacity tuning state"
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE workspace allocation cannot run during CUDA graph capture"
            )
        if live_tokens is not None and not self._kernel.supports_live_token_tuning:
            raise MoEEpConfigError("This backend does not support live-token tuning")
        profile_key = (max_tokens_per_rank, live_tokens)
        if profile_key in self._workspaces:
            raise MoEEpConfigError(
                "a MegaMoE workspace profile for max_tokens_per_rank="
                f"{max_tokens_per_rank}, live_tokens={live_tokens} already exists on this layer"
            )
        is_default_capacity = (
            live_tokens is None
            and max_tokens_per_rank == self._fleet_params.max_tokens_per_rank
        )
        kernel = self._kernel
        if knobs is not None:
            if not isinstance(knobs, dict):
                raise TypeError("workspace knobs must be a dictionary or None")
            if (
                not dataclasses.is_dataclass(self._megakernel_config)
                or isinstance(self._megakernel_config, type)
                or not hasattr(self._megakernel_config, "knobs")
            ):
                raise MoEEpConfigError("this MegaMoE backend does not support knobs")
            if is_default_capacity and self._workspace is not None:
                raise MoEEpConfigError(
                    "cannot override knobs for an already allocated default workspace"
                )
            kernel = create_mega_kernel(
                dataclasses.replace(self._megakernel_config, knobs=deepcopy(knobs))
            )
            kernel.bind_ep_bootstrap(self._bootstrap)
        fleet_params = dataclasses.replace(
            self._fleet_params,
            max_tokens_per_rank=max_tokens_per_rank,
            tuning_live_tokens=live_tokens,
        )
        kernel.validate_init(self._bootstrap, fleet_params)
        if is_default_capacity and self._workspace is not None:
            # Transfer the already-lazy-allocated default profile into the
            # explicit handle without acquiring a second pooled reference.
            backend_workspace = self._workspace
            self._workspace = None
        else:
            backend_workspace = kernel.prepare_workspace(
                self._bootstrap,
                fleet_params,
            )
        handle = MoEEpMegaWorkspace(
            self, fleet_params, backend_workspace, kernel=kernel
        )
        self._workspaces[profile_key] = handle
        if is_default_capacity:
            self._default_workspace_handle = handle
        return handle

    def _resolve_workspace(
        self,
        workspace: MoEEpMegaWorkspace | None,
    ) -> tuple[FleetParams, Any, MegaKernelBackend]:
        if workspace is None:
            if self._destroyed:
                raise MoEEpConfigError("MegaMoE layer has been destroyed")
            if self._default_workspace_handle is None:
                return self._fleet_params, self._ensure_workspace(), self._kernel
            workspace = self._default_workspace_handle
        if not isinstance(workspace, MoEEpMegaWorkspace):
            raise TypeError(
                "workspace must be created by MoEEpMegaLayer.create_workspace()"
            )
        if workspace._layer_ref() is not self:
            raise MoEEpConfigError("MegaMoE workspace belongs to a different layer")
        if workspace._destroyed or workspace._backend_workspace is None:
            raise MoEEpConfigError("MegaMoE workspace has been destroyed")
        if (
            self._workspaces.get((workspace.max_tokens_per_rank, workspace.live_tokens))
            is not workspace
        ):
            raise MoEEpConfigError(
                "MegaMoE workspace was not created by this layer or is no longer active"
            )
        return workspace._fleet_params, workspace._backend_workspace, workspace._kernel

    def _destroy_workspace_handle(self, workspace: MoEEpMegaWorkspace) -> None:
        if workspace._destroyed:
            return
        if (
            self._workspaces.get((workspace.max_tokens_per_rank, workspace.live_tokens))
            is not workspace
        ):
            raise MoEEpConfigError(
                "MegaMoE workspace was not created by this layer or is no longer active"
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE workspace destruction cannot run during CUDA graph capture"
            )
        backend_workspace = workspace._backend_workspace
        if backend_workspace is not None:
            workspace._kernel.destroy(backend_workspace)
            self._forward_signatures.pop(id(backend_workspace), None)
            if self._staged_workspace is backend_workspace:
                self._staged_kernel = None
                self._staged_workspace = None
                self._staged_transformed = None
        # Commit the closed state only after the collective backend release
        # succeeds.  A failed release remains retryable on every rank.
        workspace._destroyed = True
        workspace._backend_workspace = None
        if self._default_workspace_handle is workspace:
            self._default_workspace_handle = None
        if (
            self._workspaces.get((workspace.max_tokens_per_rank, workspace.live_tokens))
            is workspace
        ):
            del self._workspaces[(workspace.max_tokens_per_rank, workspace.live_tokens)]

    @property
    def supports_output_view(self) -> bool:
        """Whether ``forward(return_workspace_view=True)`` is supported."""
        return self._kernel.supports_output_view

    @property
    def output_buffer(self) -> torch.Tensor:
        """Stable zero-copy output owned by the default workspace."""
        _, backend_workspace, kernel = self._resolve_workspace(None)
        return kernel.workspace_output(backend_workspace)

    def _get_transformed_weights(self) -> Any:
        """Return this layer's one-time backend weight transformation."""
        if self._transformed is None:
            if not self._mega_config.preprocess_weights:
                raise MoEEpConfigError(
                    "preprocess_weights=False requires "
                    "MegaConfig.transformed_weights at init"
                )
            self._preprocess_weights()
        assert self._transformed is not None
        return self._transformed

    def warmup(
        self,
        t: Optional["MoEEpTensors"] = None,
        *,
        workspace: MoEEpMegaWorkspace | None = None,
    ) -> None:
        """Run one full eager forward so ``forward`` becomes graph-capturable.

        Forces every lazy host-side step — workspace allocation (symmetric
        heap), ``cute.compile``, the ``knobs="auto"`` autotune sweep, and one
        real kernel launch (module load) — then synchronizes the device.

        COLLECTIVE: call on ALL EP ranks together before any rank starts
        capturing (the kernel has cross-rank device-side barriers, and the
        lazy steps include collective symmetric-heap allocation).

        ``t`` defaults to a max-shape dummy batch. Pass a real batch when
        ``quantize_input=False`` — pre-quantized activations and scales
        cannot be fabricated here.
        """
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE warmup cannot run during CUDA graph capture"
            )
        uses_dummy_input = t is None
        if t is None:
            if not self._mega_config.quantize_input:
                raise MoEEpConfigError(
                    "warmup() cannot build a dummy pre-quantized batch; pass "
                    "MoEEpTensors explicitly when quantize_input=False"
                )
        fleet_params, _backend_workspace, _kernel = self._resolve_workspace(workspace)
        if t is None:
            from ..tensors import MoEEpTensors

            fp = fleet_params
            device = torch.device("cuda", torch.cuda.current_device())
            # Every mega kernel config declares top_k: int; the MegaConfig
            # field is duck-typed `object` (kernel-specific config union).
            top_k = int(self._megakernel_config.top_k)  # type: ignore[attr-defined]
            num_tokens = fp.max_tokens_per_rank
            t = MoEEpTensors(
                hidden_states=torch.zeros(
                    num_tokens,
                    fp.token_hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                ),
                # Distinct in-range experts per row (top_k <= num_experts is
                # validated at init), spread across all experts.
                topk_ids=(
                    torch.arange(num_tokens * top_k, device=device) % fp.num_experts
                ).view(num_tokens, top_k),
                topk_weights=torch.full(
                    (num_tokens, top_k),
                    1.0 / top_k,
                    dtype=torch.float32,
                    device=device,
                ),
            )
        if self._kernel.supports_unfinalized_output:
            self.forward_unfinalized(t, workspace=workspace)
        else:
            # Output-view-capable backends capture their internal graph against a
            # stable workspace address. Warming that path is what makes a later
            # outer CUDA Graph capture safe.
            self.forward(
                t,
                workspace=workspace,
                return_workspace_view=(
                    self.supports_output_view
                    and t.output is None
                    and getattr(self._megakernel_config, "knobs", None) != "auto"
                ),
            )
        torch.cuda.synchronize()
        if uses_dummy_input:
            # The synthetic routing ids are int64, but some backends accept
            # multiple input dtypes. Let the first real batch establish the
            # steady-state signature instead of pinning it to the dummy batch.
            self._forward_signatures.pop(id(_backend_workspace), None)

    def get_tuned_knobs(self) -> dict[str, Any]:
        """Return the selected knobs after a successful ``knobs="auto"`` warmup.

        The returned dictionary is a copy and can be passed as ``knobs`` when
        constructing a fixed layer with the same kernel configuration and
        capacity. This call does not allocate, tune or read a cache file.
        It raises if the layer is destroyed or has no automatic tuning result.
        """
        if self._destroyed:
            raise MoEEpConfigError("MegaMoE layer has been destroyed")
        return self._kernel.get_tuned_knobs()

    def _resolve_quantize_input(self, t: "MoEEpTensors") -> bool:
        if not self._mega_config.quantize_input:
            return False
        if t.hidden_states.dtype != torch.bfloat16:
            raise MoEEpConfigError(
                f"MegaConfig.quantize_input=True expects bf16 hidden_states; "
                f"got {t.hidden_states.dtype}. Set quantize_input=False and provide "
                f"MoEEpTensors.scales for pre-quantized activations."
            )
        return True

    @staticmethod
    def _input_signature(t: "MoEEpTensors") -> tuple[Any, ...]:
        return (
            t.hidden_states.device,
            t.hidden_states.dtype,
            t.hidden_states.ndim,
            t.hidden_states.shape[1] if t.hidden_states.ndim > 1 else None,
            t.topk_ids.device,
            t.topk_ids.dtype,
            t.topk_ids.ndim,
            t.topk_ids.shape[1] if t.topk_ids.ndim > 1 else None,
            t.topk_weights.device,
            t.topk_weights.dtype,
            t.topk_weights.ndim,
            t.topk_weights.shape[1] if t.topk_weights.ndim > 1 else None,
        )

    def _prepare_inputs(
        self,
        t: "MoEEpTensors",
        *,
        workspace: MoEEpMegaWorkspace | None = None,
        compile_tokens_per_rank: int | None = None,
    ) -> tuple[FleetParams, Any, MegaKernelBackend, bool, Any]:
        if (
            compile_tokens_per_rank is not None
            and compile_tokens_per_rank < t.num_tokens
        ):
            raise MoEEpConfigError(
                "compile_tokens_per_rank cannot be smaller than the live token count"
            )
        if not self._bootstrap_validated:
            ensure_bootstrap_dist_validated(self._bootstrap)
            self._bootstrap_validated = True

        if workspace is None:
            if self._destroyed:
                raise MoEEpConfigError("MegaMoE layer has been destroyed")
            fleet_params = self._fleet_params
            if self._default_workspace_handle is None:
                backend_workspace = self._workspace
                kernel = self._kernel
            else:
                backend_workspace = self._default_workspace_handle._backend_workspace
                kernel = self._default_workspace_handle._kernel
        else:
            fleet_params, backend_workspace, kernel = self._resolve_workspace(workspace)
        quantize_input = self._resolve_quantize_input(t)
        signature = self._input_signature(t)
        validation_key = None if backend_workspace is None else id(backend_workspace)
        previous_signature = (
            None
            if validation_key is None
            else self._forward_signatures.get(validation_key)
        )
        if previous_signature is None:
            kernel.validate_forward(
                t,
                fleet_params,
                quantize_input=quantize_input,
            )
        elif signature != previous_signature:
            raise MoEEpConfigError(
                "MegaMoE steady-state input signature changed; the backend "
                "requires stable device, dtype, rank, hidden size, and top-k"
            )
        elif t.num_tokens > fleet_params.max_tokens_per_rank:
            raise MoEEpConfigError(
                f"{t.num_tokens} tokens exceed max_tokens_per_rank="
                f"{fleet_params.max_tokens_per_rank}"
            )

        if backend_workspace is None:
            backend_workspace = self._ensure_workspace()
            validation_key = id(backend_workspace)
        if previous_signature is None:
            assert validation_key is not None
            self._forward_signatures[validation_key] = signature

        transformed_weights = self._get_transformed_weights()
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            kernel.validate_capture_ready(
                backend_workspace,
                transformed_weights,
            )

        caller_output = t.output
        if caller_output is not None and (
            caller_output.dtype != torch.bfloat16
            or caller_output.device != t.hidden_states.device
            or caller_output.ndim != 2
            or caller_output.shape[0] < t.num_tokens
            or caller_output.shape[1] != fleet_params.token_hidden_size
        ):
            raise MoEEpConfigError(
                "MegaMoE caller output must be a bf16 CUDA tensor with shape "
                f"at least ({t.num_tokens}, {fleet_params.token_hidden_size}); "
                f"got shape={tuple(caller_output.shape)}, "
                f"dtype={caller_output.dtype}, device={caller_output.device}"
            )

        kernel.set_compile_tokens_per_rank(
            backend_workspace,
            compile_tokens_per_rank,
        )
        return (
            fleet_params,
            backend_workspace,
            kernel,
            quantize_input,
            transformed_weights,
        )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        *,
        workspace: MoEEpMegaWorkspace | None = None,
        compile_tokens_per_rank: int | None = None,
    ) -> None:
        """Validate and stage one invocation without launching the kernel.

        ``compile_tokens_per_rank`` selects a collective graph/kernel bucket
        without changing the selected workspace capacity. Every EP rank must
        use the same bucket for an invocation.
        """

        (
            _,
            backend_workspace,
            kernel,
            quantize_input,
            transformed_weights,
        ) = self._prepare_inputs(
            t,
            workspace=workspace,
            compile_tokens_per_rank=compile_tokens_per_rank,
        )
        kernel.stage_inputs(
            t,
            backend_workspace,
            quantize_input=quantize_input,
        )
        self._staged_kernel = kernel
        self._staged_workspace = backend_workspace
        self._staged_transformed = transformed_weights

    def compute_staged(self, *, output: torch.Tensor | None) -> torch.Tensor:
        """Launch the kernel using inputs staged by :meth:`stage_inputs`."""

        if (
            self._staged_kernel is None
            or self._staged_workspace is None
            or self._staged_transformed is None
        ):
            raise MoEEpConfigError(
                "compute_staged() requires a prior warmup/stage_inputs() call"
            )
        return self._staged_kernel.compute(
            self._staged_workspace,
            self._staged_transformed,
            output=output,
        )

    def forward(
        self,
        t: "MoEEpTensors",
        *,
        workspace: MoEEpMegaWorkspace | None = None,
        return_workspace_view: bool = False,
    ) -> torch.Tensor:
        """Run MegaMoE and return either an owned tensor or a workspace view.

        Passing ``workspace`` selects its capacity and fixed knobs; it does not
        change output ownership. The default always returns an owned output.
        ``return_workspace_view=True`` explicitly opts into a borrowed view on
        backends that support one. A view remains valid under stream ordering
        until the next launch reuses the pooled physical workspace.
        """
        if self._kernel.supports_unfinalized_output:
            raise MoEEpConfigError("finalize_mode=NONE requires forward_unfinalized()")
        if return_workspace_view and t.output is not None:
            raise MoEEpConfigError(
                "return_workspace_view=True cannot be combined with t.output"
            )
        if return_workspace_view and not self.supports_output_view:
            raise MoEEpConfigError(
                "return_workspace_view=True is not supported by this MegaMoE backend"
            )

        if workspace is None:
            if self._destroyed:
                raise MoEEpConfigError("MegaMoE layer has been destroyed")
            fleet_params = self._fleet_params
        else:
            fleet_params, _, _ = self._resolve_workspace(workspace)
        if t.num_tokens > fleet_params.max_tokens_per_rank:
            raise MoEEpConfigError(
                f"{t.num_tokens} tokens exceed max_tokens_per_rank="
                f"{fleet_params.max_tokens_per_rank}"
            )
        (
            fleet_params,
            backend_workspace,
            kernel,
            quantize_input,
            transformed_weights,
        ) = self._prepare_inputs(t, workspace=workspace)
        output = t.output
        if output is None and not return_workspace_view:
            # Owned-output allocation must stay ahead of the staging round
            # (allocator work between stage and compute can sync the device
            # mid-round).
            output = torch.empty(
                t.num_tokens,
                fleet_params.token_hidden_size,
                dtype=torch.bfloat16,
                device=t.hidden_states.device,
            )
        kernel.stage_inputs(
            t,
            backend_workspace,
            quantize_input=quantize_input,
        )
        self._staged_kernel = kernel
        self._staged_workspace = backend_workspace
        self._staged_transformed = transformed_weights
        return self.compute_staged(output=output)

    def forward_unfinalized(
        self,
        t: "MoEEpTensors",
        *,
        workspace: MoEEpMegaWorkspace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return GEMM2 rows, remaining routing weights, and a route-to-row map.

        Requires ``finalize_mode=FinalizeMode.NONE``. Results are contiguous BF16
        ``[T * top_k, hidden]`` rows, FP32 ``[T, top_k]`` weights, and an int32
        ``[T, top_k]`` identity map whose entries index the first result.
        Weights are one when ``apply_routing_weights_before_fc2=True`` already
        applied routing.
        These tensors match the unfinalized TRTLLM MoE output layout.

        All results borrow pooled workspace storage: consume them on the current
        stream before any layer reuses that workspace. EP communication completes
        before caller finalization; zero-token ranks still participate in the
        full collective launch. Warm up on all ranks before capture.
        """
        if not self._kernel.supports_unfinalized_output:
            raise MoEEpConfigError(
                "forward_unfinalized() requires finalize_mode=FinalizeMode.NONE"
            )
        _, backend_workspace, kernel, quantize_input, transformed_weights = (
            self._prepare_inputs(t, workspace=workspace)
        )
        kernel.stage_inputs(t, backend_workspace, quantize_input=quantize_input)
        return kernel.compute_unfinalized(backend_workspace, transformed_weights)

    def destroy(self) -> None:
        """Collectively release all profiles and runtime resources.

        Call this explicitly, in the same order on every EP rank, after all
        CUDA work and captured graphs that reference the layer have retired.
        """

        if self._destroyed:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise MoEEpConfigError(
                "MegaMoE layer destruction cannot run during CUDA graph capture"
            )
        for workspace in reversed(tuple(self._workspaces.values())):
            self._destroy_workspace_handle(workspace)
        if self._workspace is not None:
            self._kernel.destroy(self._workspace)
            self._workspace = None
        self._staged_kernel = None
        self._staged_workspace = None
        self._staged_transformed = None
        if self._runtime is not None:
            finalize_moe_ep_runtime(self._runtime)
            self._runtime = None
        self._destroyed = True

    def __del__(self) -> None:
        if not getattr(self, "_destroyed", True):
            warnings.warn(
                "MoEEpMegaLayer owns collective resources and was not explicitly "
                "destroyed; call layer.destroy() collectively on all EP ranks",
                ResourceWarning,
                stacklevel=2,
            )
