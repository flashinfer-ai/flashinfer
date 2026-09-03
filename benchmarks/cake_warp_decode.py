#!/usr/bin/env python3
"""SM103 Cake warp-decode correctness and cold-L2 CUPTI benchmark harness.

The harness deliberately prepares one TRTLLM NVFP4 physical representation and
passes those exact tensor objects to the Cake launcher and the public
``trtllm_fp4_block_scale_routed_moe`` baseline.  Correctness and benchmark
preparation are kept outside timed or CUDA Graph capture regions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import weakref
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    ActivationType,
    BackendOptions,
    CakeWarpDecodeConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    SwiGLU,
    TrtllmFp4Config,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.jit.cake_fused_moe_warp_decode import (
    get_cake_fused_moe_warp_decode_module,
)
from flashinfer.testing.utils import bench_gpu_time


ATOL = 1e-2
RTOL = 1e-2
MAX_TOKENS = 32
RETIREMENT_DELAY_CYCLES = 50_000_000
PAIR_PATTERNS = (
    ("ABBA", ("exported", "baseline", "baseline", "exported")),
    ("BAAB", ("baseline", "exported", "exported", "baseline")),
)
# A failed same-address re-prepare can leave receipt ownership ambiguous: the
# binding may have rejected the call before revoking the old generation, or
# after synchronizing and erasing it.  Hold the backing allocation for process
# lifetime on that exceptional path so neither case can be recycled unsafely.
_WORKSPACE_QUARANTINE: dict[int, torch.Tensor] = {}


def _quarantine_workspace(workspace: torch.Tensor) -> int:
    identity = id(workspace)
    existing = _WORKSPACE_QUARANTINE.get(identity)
    if existing is not None and existing is not workspace:
        raise AssertionError("workspace quarantine identity collision")
    _WORKSPACE_QUARANTINE[identity] = workspace
    return identity


def _unquarantine_workspace(identity: int, workspace: torch.Tensor) -> None:
    retained = _WORKSPACE_QUARANTINE.pop(identity, None)
    if retained is not workspace:
        raise AssertionError("workspace quarantine ownership mismatch")


def _release_workspace_receipt_fail_closed(
    module: Any, workspace: torch.Tensor, receipt: int
) -> None:
    """Release once, retaining storage permanently if retirement is uncertain."""
    identity = _quarantine_workspace(workspace)
    try:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    except Exception:
        raise
    else:
        _unquarantine_workspace(identity, workspace)


def _finalize_workspace_receipt(
    module: Any, workspace: torch.Tensor, receipt: int
) -> None:
    with suppress(Exception):
        _release_workspace_receipt_fail_closed(module, workspace, receipt)
        # Finalizers cannot report errors usefully. The helper has retained the
        # allocation so its address cannot be recycled beneath uncertain work.


def _detach_and_release_workspace_receipt(releaser: weakref.finalize) -> None:
    detached = releaser.detach()
    if detached is None:
        raise AssertionError("workspace receipt finalizer lost ownership")
    _, callback, args, kwargs = detached
    if callback is not _finalize_workspace_receipt or kwargs or len(args) != 3:
        raise AssertionError("workspace receipt finalizer has an invalid callback")
    module, workspace, receipt = args
    _release_workspace_receipt_fail_closed(module, workspace, receipt)


@dataclass(frozen=True)
class Geometry:
    """One exact warp-decode model geometry."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    selector_boundaries: tuple[int, ...]


GEOMETRIES = (
    Geometry("e512_i512_k10", 2048, 512, 512, 10, (1, 2, 22, 23, 32)),
    Geometry("e60_i1536_k4", 2048, 1536, 60, 4, (1, 7, 8, 10, 11, 12, 16, 17, 32)),
)


def _selector_bucket(geometry: Geometry, num_tokens: int) -> str:
    """Name the fixed schedule/route-packer bucket exercised by a row."""
    if geometry.num_experts == 512:
        if num_tokens == 1:
            return "static_direct"
        if num_tokens <= 22:
            return "persistent_direct"
        return "persistent_general_route_packer"

    if num_tokens <= 7:
        return "persistent_direct"
    if num_tokens <= 10:
        return "persistent_direct_padded_sfb"
    if num_tokens == 11:
        return "persistent_e64_scan1_padded_sfb"
    if num_tokens <= 16:
        return "persistent_e64_scan2"
    return "persistent_general_route_packer"


@dataclass
class PhysicalFixture:
    """Graph-stable TRTLLM NVFP4 physical tensors for one geometry."""

    geometry: Geometry
    hidden_states_q: torch.Tensor
    hidden_states_scale: torch.Tensor
    initial_topk_ids: torch.Tensor
    initial_topk_weights: torch.Tensor
    mutated_topk_ids: torch.Tensor
    mutated_topk_weights: torch.Tensor
    weight_view: Mapping[str, torch.Tensor]

    def case(self, num_tokens: int, *, mutated_routes: bool = False) -> "PhysicalCase":
        if not 1 <= num_tokens <= MAX_TOKENS:
            raise ValueError(f"num_tokens must be in [1, {MAX_TOKENS}]")
        ids = self.mutated_topk_ids if mutated_routes else self.initial_topk_ids
        weights = (
            self.mutated_topk_weights if mutated_routes else self.initial_topk_weights
        )
        return PhysicalCase(
            geometry=self.geometry,
            num_tokens=num_tokens,
            hidden_states_q=self.hidden_states_q[:num_tokens],
            hidden_states_scale=self.hidden_states_scale[:num_tokens],
            topk_ids=ids[:num_tokens],
            topk_weights=weights[:num_tokens],
            weight_view=self.weight_view,
        )

    def stage_routes(self, num_tokens: int, *, mutated: bool) -> "PhysicalCase":
        """Copy routing into stable initial slots used by an already-captured graph."""
        if mutated:
            self.initial_topk_ids[:num_tokens].copy_(self.mutated_topk_ids[:num_tokens])
            self.initial_topk_weights[:num_tokens].copy_(
                self.mutated_topk_weights[:num_tokens]
            )
        else:
            ids, weights = _make_routing(self.geometry, mutated=False)
            self.initial_topk_ids[:num_tokens].copy_(ids[:num_tokens])
            self.initial_topk_weights[:num_tokens].copy_(weights[:num_tokens])
        return self.case(num_tokens)


@dataclass(frozen=True)
class PhysicalCase:
    """The exact physical inputs shared by every implementation for one T."""

    geometry: Geometry
    num_tokens: int
    hidden_states_q: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    weight_view: Mapping[str, torch.Tensor]

    @property
    def device(self) -> torch.device:
        return self.hidden_states_q.device

    def benchmark_inputs(self) -> tuple[torch.Tensor, ...]:
        """Expose live tensors so bench_gpu_time selects the correct CUDA device."""
        view = self.weight_view
        return (
            self.hidden_states_q,
            self.hidden_states_scale,
            self.topk_ids,
            self.topk_weights,
            view["gemm1_weights"],
            view["gemm1_weights_scale"],
            view["gemm2_weights"],
            view["gemm2_weights_scale"],
            view["output1_scale_scalar"],
            view["output1_scale_gate_scalar"],
            view["output2_scale_scalar"],
        )


@dataclass
class PreparedCall:
    """A launch-only callable whose initialization has already completed."""

    name: str
    output: torch.Tensor
    invoke: Callable[[], torch.Tensor]
    workspace: Any = None
    workspace_receipt: Optional[int] = None
    receipt_releaser: Optional[weakref.finalize] = None

    def close(self) -> None:
        """Release a public workspace receipt exactly once, if this call owns one."""
        if self.receipt_releaser is not None and self.receipt_releaser.alive:
            _detach_and_release_workspace_receipt(self.receipt_releaser)
            self.workspace_receipt = None


def _make_routing(
    geometry: Geometry, *, mutated: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    tokens = torch.arange(MAX_TOKENS, device=device, dtype=torch.int64)[:, None]
    ranks = torch.arange(geometry.top_k, device=device, dtype=torch.int64)[None, :]
    ids = ((tokens * 17 + ranks * 29) % (geometry.num_experts - 1) + 1).to(torch.int32)
    ids[:, 0] = 0
    raw_weights = (geometry.top_k + 1 - ranks).expand(MAX_TOKENS, -1).float()
    raw_weights = raw_weights + (tokens % 5).float() * 0.03125
    if mutated:
        ids = ((tokens * 31 + ranks * 37 + 11) % (geometry.num_experts - 1) + 1).to(
            torch.int32
        )
        ids[:, 0] = 0
        raw_weights = torch.flip(raw_weights, dims=(1,))
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    return ids.contiguous(), weights.contiguous()


def _prepare_fixture(geometry: Geometry, seed: int) -> PhysicalFixture:
    """Use the public TRTLLM preparation API to produce one physical fixture."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden = torch.randn(
        MAX_TOKENS, geometry.hidden_size, device=device, dtype=torch.bfloat16
    )
    w1 = (
        torch.randn(
            geometry.num_experts,
            2 * geometry.intermediate_size,
            geometry.hidden_size,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(
            geometry.num_experts,
            geometry.hidden_size,
            geometry.intermediate_size,
            device=device,
        )
        * 0.02
    ).to(torch.bfloat16)
    hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(
        hidden, variant=QuantVariant.NVFP4
    )
    weight_view = TrtllmFp4Config.prepare_weights(
        w1,
        w2,
        variant=QuantVariant.NVFP4,
        num_local_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        device=device,
    )
    initial_ids, initial_weights = _make_routing(geometry, mutated=False)
    mutated_ids, mutated_weights = _make_routing(geometry, mutated=True)
    return PhysicalFixture(
        geometry=geometry,
        hidden_states_q=hidden_q,
        hidden_states_scale=hidden_scale,
        initial_topk_ids=initial_ids,
        initial_topk_weights=initial_weights,
        mutated_topk_ids=mutated_ids,
        mutated_topk_weights=mutated_weights,
        weight_view=weight_view,
    )


def _normalize_result(result: Any) -> torch.Tensor:
    if isinstance(result, (list, tuple)):
        if not result:
            raise RuntimeError("MoE implementation returned an empty result sequence")
        result = result[0]
    if not isinstance(result, torch.Tensor):
        raise TypeError(
            f"MoE implementation returned {type(result).__name__}, not Tensor"
        )
    return result


def _prepare_baseline(case: PhysicalCase) -> PreparedCall:
    geometry = case.geometry
    view = case.weight_view
    output = torch.empty(
        case.num_tokens,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=case.device,
    )

    def invoke() -> torch.Tensor:
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(case.topk_ids, case.topk_weights),
            routing_bias=None,
            hidden_states=case.hidden_states_q,
            hidden_states_scale=case.hidden_states_scale,
            gemm1_weights=view["gemm1_weights"],
            gemm1_weights_scale=view["gemm1_weights_scale"],
            gemm1_bias=None,
            gemm1_alpha=view.get("gemm1_alpha"),
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=view["gemm2_weights"],
            gemm2_weights_scale=view["gemm2_weights_scale"],
            gemm2_bias=None,
            output1_scale_scalar=view["output1_scale_scalar"],
            output1_scale_gate_scalar=view["output1_scale_gate_scalar"],
            output2_scale_scalar=view["output2_scale_scalar"],
            num_experts=geometry.num_experts,
            top_k=geometry.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=geometry.intermediate_size,
            local_expert_offset=0,
            local_num_experts=geometry.num_experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.TopK.value,
            do_finalize=True,
            enable_pdl=True,
            activation_type=ActivationType.Swiglu.value,
            per_token_scale=None,
            output=output,
            tune_max_num_tokens=MAX_TOKENS,
        )
        result_tensor = _normalize_result(result)
        if result_tensor.data_ptr() != output.data_ptr():
            raise RuntimeError("TRTLLM baseline did not honor its caller output tensor")
        return output

    return PreparedCall("flashinfer_trtllm_nvfp4", output, invoke)


def _cake_shape(case: PhysicalCase) -> tuple[int, int, int, int, int]:
    geometry = case.geometry
    return (
        case.num_tokens,
        geometry.hidden_size,
        geometry.intermediate_size,
        geometry.num_experts,
        geometry.top_k,
    )


def _invoke_cake(
    module: Any,
    case: PhysicalCase,
    output: torch.Tensor,
    workspace: torch.Tensor,
    workspace_receipt: int,
) -> torch.Tensor:
    view = case.weight_view
    module.cake_fused_moe_warp_decode(
        output,
        workspace,
        case.hidden_states_q,
        case.hidden_states_scale,
        case.topk_ids,
        case.topk_weights,
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        view["output1_scale_scalar"],
        view["output1_scale_gate_scalar"],
        view["output2_scale_scalar"],
        workspace_receipt,
        True,
    )
    return output


def _prepare_cake(case: PhysicalCase) -> PreparedCall:
    geometry = case.geometry
    module = get_cake_fused_moe_warp_decode_module(device=case.device)
    shape = _cake_shape(case)
    workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    if workspace_size <= 0:
        raise RuntimeError("Cake workspace query returned a non-positive byte count")
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=case.device)
    output = torch.empty(
        case.num_tokens,
        geometry.hidden_size,
        dtype=torch.bfloat16,
        device=case.device,
    )
    workspace_receipt = int(
        module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
    )
    if workspace_receipt <= 0:
        raise RuntimeError("Cake workspace preparation returned an invalid receipt")

    def invoke() -> torch.Tensor:
        return _invoke_cake(
            module,
            case,
            output,
            workspace,
            workspace_receipt,
        )

    receipt_releaser = weakref.finalize(
        output,
        _finalize_workspace_receipt,
        module,
        workspace,
        workspace_receipt,
    )
    receipt_releaser.atexit = False
    return PreparedCall(
        "cake_warp_decode",
        output,
        invoke,
        workspace,
        workspace_receipt,
        receipt_releaser,
    )


def _distinct_nondefault_streams(
    device: torch.device,
) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
    current = torch.cuda.current_stream(device)
    prepare_stream = torch.cuda.Stream(device=device)
    run_stream = torch.cuda.Stream(device=device)
    handles = {
        int(current.cuda_stream),
        int(prepare_stream.cuda_stream),
        int(run_stream.cuda_stream),
    }
    if len(handles) != 3:
        raise RuntimeError("failed to create distinct prepare and run CUDA streams")
    prepare_stream.wait_stream(current)
    run_stream.wait_stream(current)
    return prepare_stream, run_stream


def _diagnostic(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise TypeError(
            "warp-decode correctness compares BF16 outputs; got "
            f"{actual.dtype} and {expected.dtype}"
        )
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    abs_error = (actual_f32 - expected_f32).abs()
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "max_abs": float(abs_error.max().item()),
        "mean_abs": float(abs_error.mean().item()),
        "finite": bool(torch.isfinite(actual_f32).all().item()),
    }


def _correctness_case(fixture: PhysicalFixture, num_tokens: int) -> dict[str, Any]:
    """Check parity and reuse after preparing and launching on distinct streams."""
    case = fixture.stage_routes(num_tokens, mutated=False)
    prepare_stream, run_stream = _distinct_nondefault_streams(case.device)
    with torch.cuda.stream(prepare_stream):
        cake = _prepare_cake(case)
    with torch.cuda.stream(run_stream):
        baseline = _prepare_baseline(case)
        output_ptr = cake.output.data_ptr()
        workspace_ptr = cake.workspace.data_ptr()
        workspace_bytes = cake.workspace.numel()
        first = cake.invoke().clone()
        expected = baseline.invoke().clone()
        cake.output.fill_(float("nan"))
        second = cake.invoke().clone()
    run_stream.synchronize()

    first_diagnostic = _diagnostic(first, expected)
    second_diagnostic = _diagnostic(second, expected)
    _diagnostic(second, first)
    if cake.output.data_ptr() != output_ptr:
        raise AssertionError("Cake replaced the caller-owned output tensor")
    if (
        cake.workspace.data_ptr() != workspace_ptr
        or cake.workspace.numel() != workspace_bytes
    ):
        raise AssertionError("Cake replaced or resized the caller-owned workspace")
    cake.close()
    return {
        "geometry": fixture.geometry.name,
        "num_tokens": num_tokens,
        "selector_bucket": _selector_bucket(fixture.geometry, num_tokens),
        "prepare_stream": "non_default",
        "run_stream": "different_non_default",
        "cross_stream_prepare_run": True,
        "workspace_receipt_valid": cake.workspace_receipt is not None,
        "workspace_receipt_released": cake.workspace_receipt is None,
        "output_reused": True,
        "workspace_reused": True,
        "first_launch": first_diagnostic,
        "second_launch": second_diagnostic,
        "status": "pass",
    }


def _expect_receipt_rejected(
    module: Any,
    case: PhysicalCase,
    output: torch.Tensor,
    workspace: torch.Tensor,
    receipt: int,
    *,
    label: str,
) -> None:
    try:
        _invoke_cake(module, case, output, workspace, receipt)
    except Exception as error:
        if "current preparation receipt" not in str(error):
            raise AssertionError(f"{label} failed for an unexpected reason") from error
    else:
        raise AssertionError(f"{label} was accepted")


def _expect_release_rejected(
    module: Any,
    receipt: int,
    *,
    label: str,
    message: str,
) -> None:
    try:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    except Exception as error:
        if message not in str(error):
            raise AssertionError(f"{label} failed for an unexpected reason") from error
    else:
        raise AssertionError(f"{label} was accepted")


def _same_address_receipt_case(fixture: PhysicalFixture) -> dict[str, Any]:
    """Model allocator reuse with a new tensor generation at one device address."""
    num_tokens = fixture.geometry.selector_boundaries[0]
    case = fixture.stage_routes(num_tokens, mutated=False)
    shape = _cake_shape(case)
    prepare_stream, run_stream = _distinct_nondefault_streams(case.device)
    with torch.cuda.stream(prepare_stream):
        module = get_cake_fused_moe_warp_decode_module(device=case.device)
        workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
        workspace_owner = torch.empty(
            workspace_size, dtype=torch.uint8, device=case.device
        )
        workspace_replacement = workspace_owner.view(-1)
        if workspace_replacement is workspace_owner:
            raise AssertionError("workspace replacement must be a new tensor object")
        if workspace_replacement.data_ptr() != workspace_owner.data_ptr():
            raise AssertionError("workspace replacement did not preserve its address")
        output = torch.empty(
            case.num_tokens,
            fixture.geometry.hidden_size,
            dtype=torch.bfloat16,
            device=case.device,
        )
        first_receipt = int(
            module.cake_fused_moe_warp_decode_prepare_workspace(workspace_owner, *shape)
        )
        first_releaser = weakref.finalize(
            output,
            _finalize_workspace_receipt,
            module,
            workspace_owner,
            first_receipt,
        )
        first_releaser.atexit = False
        # A successful same-address re-prepare synchronously retires this
        # generation. Disarm the fallback before crossing that transition: if
        # re-prepare itself fails after revocation, a finalizer cannot issue an
        # ambiguous stale second release while the harness unwinds.
        quarantine_identity = _quarantine_workspace(workspace_owner)
        first_releaser.detach()
        replacement_receipt = int(
            module.cake_fused_moe_warp_decode_prepare_workspace(
                workspace_replacement, *shape
            )
        )
        replacement_releaser = weakref.finalize(
            output,
            _finalize_workspace_receipt,
            module,
            workspace_replacement,
            replacement_receipt,
        )
        replacement_releaser.atexit = False
    if first_receipt <= 0 or replacement_receipt <= 0:
        raise RuntimeError("Cake workspace preparation returned an invalid receipt")
    if first_receipt == replacement_receipt:
        raise AssertionError("same-address workspace generations reused a receipt")
    _unquarantine_workspace(quarantine_identity, workspace_owner)
    _expect_release_rejected(
        module,
        0,
        label="non-positive workspace receipt release",
        message="positive preparation receipt",
    )
    _expect_release_rejected(
        module,
        first_receipt,
        label="stale same-address workspace receipt release",
        message="unknown or already released receipt",
    )

    with torch.cuda.stream(run_stream):
        baseline = _prepare_baseline(case)
        _expect_receipt_rejected(
            module,
            case,
            output,
            workspace_replacement,
            first_receipt,
            label="stale same-address workspace receipt",
        )
        actual = _invoke_cake(
            module,
            case,
            output,
            workspace_replacement,
            replacement_receipt,
        ).clone()
        expected = baseline.invoke().clone()
    run_stream.synchronize()
    replacement_diagnostic = _diagnostic(actual, expected)
    _detach_and_release_workspace_receipt(replacement_releaser)
    _expect_release_rejected(
        module,
        replacement_receipt,
        label="repeated workspace receipt release",
        message="unknown or already released receipt",
    )
    with torch.cuda.stream(run_stream):
        _expect_receipt_rejected(
            module,
            case,
            output,
            workspace_replacement,
            replacement_receipt,
            label="released workspace receipt",
        )

    return {
        "geometry": fixture.geometry.name,
        "num_tokens": num_tokens,
        "selector_bucket": _selector_bucket(fixture.geometry, num_tokens),
        "same_address_new_tensor_generation": True,
        "receipt_advanced": True,
        "stale_receipt_rejected": True,
        "released_receipt_rejected": True,
        "nonpositive_release_rejected": True,
        "stale_release_rejected": True,
        "double_release_rejected": True,
        "replacement_launch": replacement_diagnostic,
        "cross_stream_prepare_run": True,
        "workspace_receipts_released": True,
        "status": "pass",
    }


def _workspace_retirement_case(
    fixture: PhysicalFixture,
) -> dict[str, Any]:
    """Re-prepare and release without caller-side stream synchronization."""
    num_tokens = fixture.geometry.selector_boundaries[-1]
    case = fixture.stage_routes(num_tokens, mutated=False)
    shape = _cake_shape(case)
    prepare_stream, first_stream = _distinct_nondefault_streams(case.device)
    second_stream = torch.cuda.Stream(device=case.device)
    if int(second_stream.cuda_stream) in {
        int(prepare_stream.cuda_stream),
        int(first_stream.cuda_stream),
    }:
        raise RuntimeError("failed to create a third distinct CUDA stream")
    second_stream.wait_stream(torch.cuda.current_stream(case.device))
    module = get_cake_fused_moe_warp_decode_module(device=case.device)
    workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=case.device)
    output = torch.empty(
        case.num_tokens,
        fixture.geometry.hidden_size,
        dtype=torch.bfloat16,
        device=case.device,
    )
    first_receipt: Optional[int] = None
    with torch.cuda.stream(prepare_stream):
        first_receipt = int(
            module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
        )
    replacement_receipt: Optional[int] = None
    try:
        with torch.cuda.stream(first_stream):
            _invoke_cake(module, case, output, workspace, first_receipt)

        # Do not synchronize first_stream here. Re-preparation must retire the
        # recorded submission before it clears and reinitializes the workspace.
        with torch.cuda.stream(second_stream):
            previous_receipt = first_receipt
            # The C++ transition can fail either before or after revoking the
            # old generation. Conservatively disarm cleanup before crossing it;
            # on any exception the failing validation process retains storage
            # until exit instead of attempting an unknowable second release.
            quarantine_identity = _quarantine_workspace(workspace)
            first_receipt = None
            replacement_receipt = int(
                module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
            )
            if replacement_receipt <= 0 or replacement_receipt == previous_receipt:
                raise AssertionError(
                    "workspace re-preparation did not advance its receipt"
                )
            _unquarantine_workspace(quarantine_identity, workspace)
            replacement = _invoke_cake(
                module, case, output, workspace, replacement_receipt
            ).clone()

        # Release immediately, again without synchronizing second_stream. The
        # binding must wait for its completion event before retiring the state.
        retiring_receipt = replacement_receipt
        replacement_receipt = None
        _release_workspace_receipt_fail_closed(module, workspace, retiring_receipt)
        with torch.cuda.stream(second_stream):
            expected = _prepare_baseline(case).invoke().clone()
        second_stream.synchronize()
        replacement_diagnostic = _diagnostic(replacement, expected)
        return {
            "geometry": fixture.geometry.name,
            "num_tokens": num_tokens,
            "selector_bucket": _selector_bucket(fixture.geometry, num_tokens),
            "reprepare_without_caller_stream_sync": True,
            "release_without_caller_stream_sync": True,
            "receipt_advanced": True,
            "replacement_launch": replacement_diagnostic,
            "status": "pass",
        }
    finally:
        if first_receipt is not None:
            retiring_receipt = first_receipt
            first_receipt = None
            _release_workspace_receipt_fail_closed(module, workspace, retiring_receipt)
        if replacement_receipt is not None:
            retiring_receipt = replacement_receipt
            replacement_receipt = None
            _release_workspace_receipt_fail_closed(module, workspace, retiring_receipt)


def _layer_graph_case(fixture: PhysicalFixture) -> dict[str, Any]:
    """Exercise the public MoELayer winner path and graph capture end to end."""
    num_tokens = fixture.geometry.selector_boundaries[1]
    case = fixture.stage_routes(num_tokens, mutated=False)
    geometry = fixture.geometry
    activations = MoEActivationPack(
        hidden_states_q=case.hidden_states_q,
        hidden_states_scale=case.hidden_states_scale,
        topk_ids=case.topk_ids,
        topk_weights=case.topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cake", dict(case.weight_view))
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=geometry.num_experts,
            top_k=geometry.top_k,
            method=RoutingMethodType.TopK,
        ),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(intermediate_size=geometry.intermediate_size),
        activation=SwiGLU(),
        backend=BackendOptions((CakeWarpDecodeConfig(),)),
        execution=ExecutionConfig(
            tune_max_num_tokens=MAX_TOKENS,
            enable_pdl=True,
        ),
    )
    layer = MoELayer(config, device=case.device)
    layer.tuner.clear_cache()
    layer.tuner.reset_statistics()
    warmup_stream, capture_stream = _distinct_nondefault_streams(case.device)
    replay_stream = torch.cuda.Stream(device=case.device)
    if int(replay_stream.cuda_stream) in {
        int(warmup_stream.cuda_stream),
        int(capture_stream.cuda_stream),
    }:
        raise RuntimeError("failed to create a distinct CUDA Graph replay stream")
    replay_stream.wait_stream(torch.cuda.current_stream(case.device))

    # The first public call runs the real one-backend winner-selection path.
    # Its timing helper uses its own warmup/capture streams, which is the exact
    # framework path that a permanently stream-claimed workspace cannot serve.
    with autotune(True, tuning_buckets=(num_tokens,)), torch.cuda.stream(warmup_stream):
        eager = layer(activations, weights).clone()
        expected_eager = _prepare_baseline(case).invoke().clone()
    tuned_total = layer.tuner.stats.tuned_op_total_configs.get("moe_cake", 0)
    tuned_successful = layer.tuner.stats.tuned_op_successful_configs.get("moe_cake", 0)
    if tuned_total < 1 or tuned_successful < 1:
        raise AssertionError(
            "Cake MoELayer validation did not execute successful autotune profiling"
        )
    warmup_stream.synchronize()
    eager_diagnostic = _diagnostic(eager, expected_eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output = layer(activations, weights)
    with torch.cuda.stream(replay_stream):
        graph.replay()
        replayed = captured_output.clone()
        expected_replay = _prepare_baseline(case).invoke().clone()
    replay_stream.synchronize()
    replay_diagnostic = _diagnostic(replayed, expected_replay)
    return {
        "geometry": geometry.name,
        "num_tokens": num_tokens,
        "selector_bucket": _selector_bucket(geometry, num_tokens),
        "winner_backend": layer.winner_backend,
        "winner_selection_exercised": True,
        "autotune_profiling_exercised": True,
        "autotune_total_configs": tuned_total,
        "autotune_successful_configs": tuned_successful,
        "capture_stream": "non_default",
        "replay_stream": "different_non_default",
        "graph_replay_cross_stream": True,
        "eager": eager_diagnostic,
        "replay": replay_diagnostic,
        "status": "pass",
    }


def run_sanitizer(
    geometries: Sequence[Geometry],
    *,
    seed: int,
    sanitizer_tokens: Optional[Sequence[int]],
) -> dict[str, Any]:
    """Issue launch-only boundary cases for compute-sanitizer wrappers."""
    rows: list[dict[str, Any]] = []
    for geometry_index, geometry in enumerate(geometries):
        fixture = _prepare_fixture(geometry, seed + geometry_index)
        tokens = (
            tuple(sanitizer_tokens)
            if sanitizer_tokens is not None
            else geometry.selector_boundaries
        )
        for num_tokens in tokens:
            case = fixture.stage_routes(num_tokens, mutated=False)
            prepare_stream, run_stream = _distinct_nondefault_streams(case.device)
            with torch.cuda.stream(prepare_stream):
                cake = _prepare_cake(case)
            with torch.cuda.stream(run_stream):
                cake.invoke()
            run_stream.synchronize()
            cake.close()
            rows.append(
                {
                    "geometry": geometry.name,
                    "num_tokens": num_tokens,
                    "selector_bucket": _selector_bucket(geometry, num_tokens),
                    "prepare_stream": "non_default",
                    "run_stream": "different_non_default",
                    "cross_stream_prepare_run": True,
                    "workspace_receipt_released": True,
                    "status": "launched",
                }
            )
    return {"mode": "sanitizer", "rows": rows, "status": "pass"}


def _graph_mutation_case(
    fixture: PhysicalFixture, num_tokens: int, mutation_index: int
) -> dict[str, Any]:
    """Capture once, mutate live routing/model tensors, then replay and compare."""
    case = fixture.stage_routes(num_tokens, mutated=False)
    prepare_stream, capture_stream = _distinct_nondefault_streams(case.device)
    replay_stream = torch.cuda.Stream(device=case.device)
    if int(replay_stream.cuda_stream) in {
        int(prepare_stream.cuda_stream),
        int(capture_stream.cuda_stream),
    }:
        raise RuntimeError("failed to create a distinct CUDA Graph replay stream")
    replay_stream.wait_stream(torch.cuda.current_stream(case.device))
    with torch.cuda.stream(prepare_stream):
        cake = _prepare_cake(case)
    with torch.cuda.stream(capture_stream):
        baseline = _prepare_baseline(case)
        warmup = cake.invoke().clone()
        before = baseline.invoke().clone()
    capture_stream.synchronize()
    warmup_diagnostic = _diagnostic(warmup, before)

    output_ptr = cake.output.data_ptr()
    workspace_ptr = cake.workspace.data_ptr()
    workspace_bytes = cake.workspace.numel()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        cake.invoke()

    with torch.cuda.stream(capture_stream):
        cake.output.fill_(float("nan"))
        graph.replay()
        initial_replay = cake.output.clone()
    capture_stream.synchronize()
    initial_replay_diagnostic = _diagnostic(initial_replay, before)

    mutated_expert = 0
    replay_completion = torch.cuda.Event(enable_timing=False)
    with torch.cuda.stream(replay_stream):
        fixture.stage_routes(num_tokens, mutated=True)
        # Make the replay perturbation observable even when random NVFP4
        # quantization collapses a small multi-expert delta at BF16 output
        # precision. Keep valid, unique top-k ids but route all weight through
        # expert zero, whose packed weights and scales are zeroed below.
        case.topk_weights.zero_()
        case.topk_weights[:, 0].fill_(1.0)
        for name in (
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
        ):
            fixture.weight_view[name][mutated_expert].zero_()
        cake.output.fill_(float("nan"))
        torch.cuda._sleep(RETIREMENT_DELAY_CYCLES)
        graph.replay()
        replay_completion.record()
        replayed = cake.output.clone()
        expected = baseline.invoke().clone()

    if replay_completion.query():
        raise AssertionError(
            "deterministic replay delay completed before workspace retirement"
        )
    # No replay-stream synchronization precedes release. The marker proves the
    # replay was still in flight immediately before close, while the host block
    # duration and destructive overwrite below prove that close waited for the
    # graph-owned completion event before permitting storage reuse.
    retirement_start_ns = time.monotonic_ns()
    cake.close()
    retirement_block_ms = (time.monotonic_ns() - retirement_start_ns) / 1e6
    module = get_cake_fused_moe_warp_decode_module(device=case.device)
    replacement_stream = torch.cuda.Stream(device=case.device)
    with torch.cuda.stream(replacement_stream):
        cake.workspace.fill_(0xA5)
    replacement_stream.synchronize()
    replay_stream.synchronize()

    # Reinitialize the deliberately overwritten storage and run a full launch
    # at the same address after the delayed replay has been validated.
    replacement_receipt: Optional[int] = None
    try:
        with torch.cuda.stream(replacement_stream):
            replacement_receipt = int(
                module.cake_fused_moe_warp_decode_prepare_workspace(
                    cake.workspace, *_cake_shape(case)
                )
            )
            if replacement_receipt <= 0:
                raise AssertionError("post-graph workspace re-preparation failed")
            replacement_output = torch.empty_like(cake.output)
            replacement = _invoke_cake(
                module,
                case,
                replacement_output,
                cake.workspace,
                replacement_receipt,
            ).clone()
        retiring_receipt = replacement_receipt
        replacement_receipt = None
        _release_workspace_receipt_fail_closed(module, cake.workspace, retiring_receipt)
    finally:
        if replacement_receipt is not None:
            retiring_receipt = replacement_receipt
            replacement_receipt = None
            _release_workspace_receipt_fail_closed(
                module, cake.workspace, retiring_receipt
            )
    replacement_stream.synchronize()

    diagnostic = _diagnostic(replayed, expected)
    replacement_diagnostic = _diagnostic(replacement, expected)
    mutation_delta = float((before.float() - expected.float()).abs().max().item())
    if mutation_delta == 0.0:
        raise AssertionError("Graph mutation case did not change the reference output")
    if cake.output.data_ptr() != output_ptr:
        raise AssertionError("CUDA Graph replay replaced the caller output tensor")
    if (
        cake.workspace.data_ptr() != workspace_ptr
        or cake.workspace.numel() != workspace_bytes
    ):
        raise AssertionError("CUDA Graph replay replaced the caller workspace")
    return {
        "geometry": fixture.geometry.name,
        "num_tokens": num_tokens,
        "selector_bucket": _selector_bucket(fixture.geometry, num_tokens),
        "prepare_stream": "non_default",
        "capture_stream": "different_non_default",
        "replay_stream": "third_non_default",
        "cross_stream_prepare_capture": True,
        "cross_stream_graph_replay": True,
        "release_without_replay_stream_sync": True,
        "replay_in_flight_before_release": True,
        "workspace_overwritten_after_release": True,
        "retirement_block_ms": retirement_block_ms,
        "deterministic_replay_delay_cycles": RETIREMENT_DELAY_CYCLES,
        "replacement_launch": replacement_diagnostic,
        "workspace_receipt_valid": cake.workspace_receipt is not None,
        "workspace_receipt_released": cake.workspace_receipt is None,
        "mutated": [
            "topk_ids",
            "topk_weights",
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
        ],
        "mutated_expert": mutated_expert,
        "reference_mutation_max_abs": mutation_delta,
        "warmup": warmup_diagnostic,
        "initial_replay": initial_replay_diagnostic,
        "mutated_replay": diagnostic,
        "output_reused": True,
        "workspace_reused": True,
        "status": "pass",
    }


def run_correctness(geometries: Sequence[Geometry], seed: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    workspace_retirement_rows: list[dict[str, Any]] = []
    layer_graph_rows: list[dict[str, Any]] = []
    for geometry_index, geometry in enumerate(geometries):
        fixture = _prepare_fixture(geometry, seed + geometry_index)
        for num_tokens in range(1, MAX_TOKENS + 1):
            rows.append(_correctness_case(fixture, num_tokens))
        receipt_rows.append(_same_address_receipt_case(fixture))
        workspace_retirement_rows.append(_workspace_retirement_case(fixture))
        layer_graph_rows.append(_layer_graph_case(fixture))
        for mutation_index, num_tokens in enumerate(geometry.selector_boundaries):
            graph_rows.append(_graph_mutation_case(fixture, num_tokens, mutation_index))
    return {
        "mode": "correctness",
        "tolerance": {"atol": ATOL, "rtol": RTOL, "comparison_dtype": "bfloat16"},
        "matrix_rows": rows,
        "workspace_receipt_rows": receipt_rows,
        "workspace_retirement_rows": workspace_retirement_rows,
        "layer_graph_rows": layer_graph_rows,
        "graph_mutation_rows": graph_rows,
        "status": "pass",
    }


def _require_cupti() -> str:
    try:
        from cupti import cupti  # noqa: F401

        cupti_version = version("cupti-python")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "cupti-python >= 13 is required; benchmark fallback is not permitted"
        ) from error
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(
            f"cupti-python >= 13 is required, found {cupti_version}; "
            "benchmark fallback is not permitted"
        )
    return cupti_version


def _validate_prepared_call(call: PreparedCall, case: PhysicalCase) -> None:
    expected_shape = (case.num_tokens, case.geometry.hidden_size)
    if call.output.dtype != torch.bfloat16:
        raise TypeError(f"{call.name} output must be BF16, got {call.output.dtype}")
    if tuple(call.output.shape) != expected_shape:
        raise ValueError(
            f"{call.name} output shape {tuple(call.output.shape)} != {expected_shape}"
        )
    if call.output.device != case.device or not call.output.is_contiguous():
        raise ValueError(f"{call.name} output must be contiguous on {case.device}")


def _benchmark_call(
    call: PreparedCall,
    case: PhysicalCase,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    _validate_prepared_call(call, case)

    def launch_with_live_inputs(*_live_inputs: torch.Tensor) -> torch.Tensor:
        return call.invoke()

    call.invoke()
    torch.cuda.synchronize(case.device)
    samples = bench_gpu_time(
        fn=launch_with_live_inputs,
        input_args=case.benchmark_inputs(),
        enable_cupti=True,
        cold_l2_cache=True,
        use_cuda_graph=False,
        dry_run_iters=warmup,
        repeat_iters=repetitions,
    )
    if not samples or any(
        not math.isfinite(float(sample)) or float(sample) <= 0.0 for sample in samples
    ):
        raise RuntimeError(
            f"{call.name} returned non-positive or non-finite CUPTI samples"
        )
    return {
        "name": call.name,
        "median_ms": float(statistics.median(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "samples": len(samples),
    }


def _paired_benchmark_round(
    exported: PreparedCall,
    baseline: PreparedCall,
    case: PhysicalCase,
    *,
    round_index: int,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    """Measure one exported/baseline ABBA or BAAB CUPTI round."""
    pattern, order = PAIR_PATTERNS[round_index % len(PAIR_PATTERNS)]
    calls = {"exported": exported, "baseline": baseline}
    measurements: list[dict[str, Any]] = []
    per_arm: dict[str, list[float]] = {"exported": [], "baseline": []}
    for position, arm in enumerate(order):
        measurement = _benchmark_call(
            calls[arm],
            case,
            warmup=warmup,
            repetitions=repetitions,
        )
        measurements.append(
            {
                "position": position,
                "arm": arm,
                "measurement": measurement,
            }
        )
        per_arm[arm].append(float(measurement["median_ms"]))
    exported_ms = float(statistics.median(per_arm["exported"]))
    baseline_ms = float(statistics.median(per_arm["baseline"]))
    return {
        "round_index": round_index,
        "pattern": pattern,
        "order": list(order),
        "measurements": measurements,
        "exported_median_ms": exported_ms,
        "baseline_median_ms": baseline_ms,
        "exported_baseline_ratio": exported_ms / baseline_ms,
    }


def _paired_arm_summary(
    name: str,
    rounds: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    session_medians = [
        float(item["measurement"]["median_ms"])
        for round_record in rounds
        for item in round_record["measurements"]
        if item["arm"] == name
    ]
    return {
        "name": name,
        "median_ms": float(statistics.median(session_medians)),
        "session_count": len(session_medians),
        "session_medians_ms": session_medians,
    }


def run_benchmark(
    geometries: Sequence[Geometry],
    *,
    seed: int,
    benchmark_tokens: Optional[Sequence[int]],
    warmup: int,
    repetitions: int,
    paired_rounds: int,
) -> dict[str, Any]:
    cupti_version = _require_cupti()
    rows: list[dict[str, Any]] = []
    for geometry_index, geometry in enumerate(geometries):
        fixture = _prepare_fixture(geometry, seed + geometry_index)
        tokens = (
            tuple(benchmark_tokens)
            if benchmark_tokens is not None
            else geometry.selector_boundaries
        )
        for num_tokens in tokens:
            case = fixture.stage_routes(num_tokens, mutated=False)
            prepare_stream, benchmark_stream = _distinct_nondefault_streams(case.device)
            with torch.cuda.stream(prepare_stream):
                exported = _prepare_cake(case)
            output_ptr = exported.output.data_ptr()
            workspace_ptr = exported.workspace.data_ptr()
            workspace_bytes = exported.workspace.numel()
            with torch.cuda.stream(benchmark_stream):
                baseline = _prepare_baseline(case)
                exported_output = exported.invoke().clone()
                baseline_output = baseline.invoke().clone()
                torch.cuda.synchronize(case.device)
                parity = _diagnostic(exported_output, baseline_output)
                row: dict[str, Any] = {
                    "geometry": geometry.name,
                    "num_tokens": num_tokens,
                    "selector_bucket": _selector_bucket(geometry, num_tokens),
                    "prepare_stream": "non_default",
                    "benchmark_stream": "different_non_default",
                    "cross_stream_prepare_run": True,
                    "exported_parity": parity,
                    "exported": None,
                    "flashinfer_baseline": None,
                    "paired_rounds": [],
                    "ratios": {},
                    "status": "pass",
                }
                pair_records = [
                    _paired_benchmark_round(
                        exported,
                        baseline,
                        case,
                        round_index=round_index,
                        warmup=warmup,
                        repetitions=repetitions,
                    )
                    for round_index in range(paired_rounds)
                ]
                exported_summary = _paired_arm_summary("exported", pair_records)
                baseline_summary = _paired_arm_summary("baseline", pair_records)
                row["exported"] = exported_summary
                row["flashinfer_baseline"] = baseline_summary
                row["paired_rounds"] = pair_records
                row["ratios"] = {
                    "exported_over_flashinfer_baseline": (
                        float(exported_summary["median_ms"])
                        / float(baseline_summary["median_ms"])
                    ),
                    "worst_round_exported_over_flashinfer_baseline": max(
                        float(record["exported_baseline_ratio"])
                        for record in pair_records
                    ),
                }
            benchmark_stream.synchronize()
            if exported.output.data_ptr() != output_ptr:
                raise AssertionError("benchmark replaced the caller output tensor")
            if (
                exported.workspace.data_ptr() != workspace_ptr
                or exported.workspace.numel() != workspace_bytes
            ):
                raise AssertionError("benchmark replaced the caller workspace")
            exported.close()
            row["output_reused"] = True
            row["workspace_reused"] = True
            row["workspace_receipt_released"] = True
            rows.append(row)
    return {
        "mode": "benchmark",
        "timing": {
            "backend": "cupti",
            "cupti_python": cupti_version,
            "cold_l2_cache": True,
            "use_cuda_graph": False,
            "warmup": warmup,
            "repetitions": repetitions,
            "paired_rounds": paired_rounds,
            "pair_patterns": [name for name, _ in PAIR_PATTERNS],
        },
        "rows": rows,
        "status": "pass",
    }


def _selected_geometries(name: str) -> tuple[Geometry, ...]:
    if name == "all":
        return GEOMETRIES
    selected = tuple(geometry for geometry in GEOMETRIES if geometry.name == name)
    if not selected:
        raise ValueError(f"unknown geometry {name!r}")
    return selected


def _validate_environment(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("the warp-decode harness requires a CUDA device")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) != (10, 3):
        raise RuntimeError(
            f"the warp-decode harness requires exact SM103, got SM{major}{minor}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("correctness", "benchmark", "sanitizer", "all"),
        default="correctness",
    )
    parser.add_argument(
        "--geometry",
        choices=("all", *(geometry.name for geometry in GEOMETRIES)),
        default="all",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--benchmark-tokens",
        type=int,
        nargs="+",
        help="token counts for benchmark mode; defaults to selector boundaries",
    )
    parser.add_argument(
        "--sanitizer-tokens",
        type=int,
        nargs="+",
        help="token counts for sanitizer mode; defaults to selector boundaries",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--paired-rounds",
        type=int,
        default=2,
        help="even number of alternating ABBA/BAAB exported/baseline rounds",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.benchmark_tokens is not None and any(
        token < 1 or token > MAX_TOKENS for token in args.benchmark_tokens
    ):
        parser.error(f"--benchmark-tokens values must be in [1, {MAX_TOKENS}]")
    if args.sanitizer_tokens is not None and any(
        token < 1 or token > MAX_TOKENS for token in args.sanitizer_tokens
    ):
        parser.error(f"--sanitizer-tokens values must be in [1, {MAX_TOKENS}]")
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("--warmup and --repetitions must be positive")
    if args.paired_rounds < 2 or args.paired_rounds % 2:
        parser.error("--paired-rounds must be an even integer of at least 2")
    return args


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    _validate_environment(device)
    geometries = _selected_geometries(args.geometry)
    report: dict[str, Any] = {
        "device": str(device),
        "compute_capability": "sm_103",
        "geometry": args.geometry,
        "seed": args.seed,
    }
    if args.mode in ("correctness", "all"):
        report["correctness"] = run_correctness(geometries, args.seed)
    if args.mode in ("benchmark", "all"):
        report["benchmark"] = run_benchmark(
            geometries,
            seed=args.seed,
            benchmark_tokens=args.benchmark_tokens,
            warmup=args.warmup,
            repetitions=args.repetitions,
            paired_rounds=args.paired_rounds,
        )
    if args.mode == "sanitizer":
        report["sanitizer"] = run_sanitizer(
            geometries,
            seed=args.seed,
            sanitizer_tokens=args.sanitizer_tokens,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    benchmark = report.get("benchmark")
    if benchmark is not None and benchmark["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
