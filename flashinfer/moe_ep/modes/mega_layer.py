"""MoEEpMegaLayer — fused mega-MoE kernel path."""

from __future__ import annotations

import contextlib
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
    from ..tensors import MoEEpTensors


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
        if bootstrap.auto_bootstrap:
            self._runtime = bootstrap_moe_ep_runtime(
                bootstrap,
                self._kernel.runtime_requirements(bootstrap),
            )

        validate_bootstrap_world_size(bootstrap)
        self._kernel.validate_init(bootstrap, fleet_params)

        if backend.transformed_weights is None:
            validate_fleet_weights(weights, fleet_params, bootstrap.world_size)

        self._weights: Optional[MoEWeightPack] = (
            weights if backend.transformed_weights is None else None
        )
        self._transformed: Optional[Any] = None
        self._workspace: Any = None
        self._bootstrap_validated = False
        self._forward_validated = False
        self._forward_signature: tuple[Any, ...] | None = None

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
        self._weights = None

    def _ensure_workspace(self) -> Any:
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

    @property
    def output_buffer(self) -> torch.Tensor:
        """Stable zero-copy output owned by the pooled mega workspace."""
        return self._kernel.workspace_output(self._ensure_workspace())

    def warmup(self, t: Optional["MoEEpTensors"] = None) -> None:
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
        if t is None:
            if not self._mega_config.quantize_input:
                raise MoEEpConfigError(
                    "warmup() cannot build a dummy pre-quantized batch; pass "
                    "MoEEpTensors explicitly when quantize_input=False"
                )
            from ..tensors import MoEEpTensors

            fp = self._fleet_params
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
        self.forward(t)
        torch.cuda.synchronize()

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

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        *,
        compile_tokens_per_rank: int | None = None,
    ) -> None:
        """Validate and stage one iteration without launching the mega kernel.

        Keeping staging separate lets frameworks capture its fixed-shape GPU
        work while replaying a backend-owned compute graph eagerly. This is
        useful when nesting that graph would discard backend-specific launch
        scheduling such as Green Context partitioning.

        ``compile_tokens_per_rank`` is a collective hint: when provided, every
        EP rank must pass the same padded row count. It selects a graph/kernel
        specialization without changing the workspace capacity.
        """
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
        quantize_input = self._resolve_quantize_input(t)
        signature = (
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

        if not self._forward_validated:
            self._kernel.validate_forward(
                t,
                self._fleet_params,
                quantize_input=quantize_input,
            )
            self._forward_validated = True
            self._forward_signature = signature
        elif signature != self._forward_signature:
            raise MoEEpConfigError(
                "MegaMoE steady-state input signature changed; the backend "
                "requires stable device, dtype, rank, hidden size, and top-k"
            )
        elif t.num_tokens > self._fleet_params.max_tokens_per_rank:
            raise MoEEpConfigError(
                f"{t.num_tokens} tokens exceed MegaMoE capacity "
                f"{self._fleet_params.max_tokens_per_rank}"
            )

        if self._transformed is None:
            if not self._mega_config.preprocess_weights:
                raise MoEEpConfigError(
                    "preprocess_weights=False requires "
                    "MegaConfig.transformed_weights at init"
                )
            self._preprocess_weights()
        assert self._transformed is not None

        workspace = self._ensure_workspace()

        caller_output = t.output
        if caller_output is not None and (
            caller_output.dtype != torch.bfloat16
            or caller_output.device != t.hidden_states.device
            or caller_output.ndim != 2
            or caller_output.shape[0] < t.num_tokens
            or caller_output.shape[1] != self._fleet_params.token_hidden_size
        ):
            raise MoEEpConfigError(
                "MegaMoE caller output must be a bf16 CUDA tensor with shape "
                f"at least ({t.num_tokens}, {self._fleet_params.token_hidden_size}); "
                f"got shape={tuple(caller_output.shape)}, "
                f"dtype={caller_output.dtype}, device={caller_output.device}"
            )
        self._kernel.set_compile_tokens_per_rank(
            workspace, compile_tokens_per_rank
        )
        self._kernel.stage_inputs(
            t,
            workspace,
            quantize_input=quantize_input,
        )

    def compute_staged(self, *, output: torch.Tensor | None) -> torch.Tensor:
        """Launch the mega kernel using inputs staged by :meth:`stage_inputs`."""
        if self._workspace is None or self._transformed is None:
            raise MoEEpConfigError(
                "compute_staged() requires a prior warmup/stage_inputs() call"
            )
        return self._kernel.compute(
            self._workspace,
            self._transformed,
            output=output,
        )

    def forward(self, t: "MoEEpTensors") -> torch.Tensor:
        self.stage_inputs(t)
        result = self.compute_staged(output=t.output)
        caller_output = t.output
        if caller_output is not None:
            return result
        y = torch.empty(
            t.num_tokens,
            self._fleet_params.token_hidden_size,
            dtype=torch.bfloat16,
            device=t.hidden_states.device,
        )
        y.copy_(result)
        return y

    def destroy(self) -> None:
        if self._workspace is not None:
            self._kernel.destroy(self._workspace)
            self._workspace = None
        if self._runtime is not None:
            finalize_moe_ep_runtime(self._runtime)
            self._runtime = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()
