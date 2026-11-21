# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Specialized kernel ``fp8_block_prefill_sm103_qwen35``: FP8 block-scale MoE
prefill on SM103.

Not a public API: ``MoELayer`` reaches it through its ``specialized`` backend
(:class:`flashinfer.fused_moe.SpecializedMoEConfig`).

One expert geometry, Qwen3.5-397B-A17B-FP8 prefill without expert parallelism,
is served by a CuTe DSL pipeline
(``fused_moe/cute_dsl/blackwell/fp8_block_scale_moe_prefill.py``): 512 local
experts, top-10 RenormalizeNaive routing from bf16 logits (no bias, groups or
scaling factor), hidden 4096, intermediate 1024, the default SwiGLU,
DeepSeek-FP8 block scales with bf16 output and ``do_finalize=True``, on exact
SM103 with the CuTe DSL, for token counts in the package-data allowlist
``workloads.json`` (prefill chunks). The pipeline reads the ``trtllm_fp8_block``
weight view (:class:`TrtllmFp8BlockConfig`: unshuffled ``MajorK`` E4M3 weights
with FP32 128x128 block scales).

Two levels of checks keep every other call on ``MoELayer``'s other candidates
(``trtllm_fp8_block`` in the default list):

* configuration (:func:`config_rejection`, at ``MoELayer`` construction):
  anything but the served configuration on SM103 with the CuTe DSL makes the
  backend skip this kernel;
* per call (:func:`admit`): kill switch, the activation / routing / weight-view
  tensor contract, token count, device and readiness. Never raises.

Build: the programs (13, none depends on the token count) are compiled and
launched once on a dummy problem by the first eager call admitted, outside any
capture. Under CUDA-graph capture a call is admitted only when the programs
exist and this binding already admitted the same token count eagerly, i.e. when
``MoELayer`` already selected its winner for that call outside capture;
otherwise the call stays on the other candidates and nothing compiles inside a
capture.

The numerics are the trtllm-gen DeepSeek-FP8 data path: GEMM1's output and the
SwiGLU output are FP8 E4M3 with one scale per (token, 128 columns), scale
max(amax / 448, FLT_MIN), fp32 accumulation, bf16 output.

Nothing here imports the CuTe DSL at module scope: the kernel module is loaded
on the first call admitted.
"""

from __future__ import annotations

import functools
import json
import sys
import threading
import time
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional, Set

import torch

from ......jit.core import logger
from .....api import QuantFormat, RoutingInputMode, SwiGLU
from ...kernel import (
    KILL_SWITCH_ENV,
    SpecializedMoEKernel,
    is_capturing,
    kill_switch_engaged,
)

NAME = "fp8_block_prefill_sm103_qwen35"
WEIGHT_VIEW = "trtllm_fp8_block"

# Logged once per process at the first dispatch (grep-able).
DISPATCH_MARKER = (
    f"flashinfer specialized kernel: MoELayer specialized/{NAME} "
    "sm103 e512 h4096 i1024 top10 prefill"
)
# Prefix of the once-per-reason rejection lines (debug level); deliberately does
# not contain the dispatch marker, so grepping the marker never matches a reject.
FALLBACK_MARKER = (
    f"flashinfer specialized kernel fallback (MoELayer specialized/{NAME})"
)

# Compile-time geometry of the pipeline.
NUM_EXPERTS = 512
TOP_K = 10
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
BLOCK = 128
COMPUTE_CAPABILITY = (10, 3)
# Upper bound the pipeline's workspace is sized for.
MAX_TOKENS = 32768

# Dispatch surface (package data next to this module).
_WORKLOAD_FILE = "workloads.json"
# The CuTe DSL kernel module, imported on the first build only.
_KERNEL_MODULE = (
    __name__.rsplit(".", 5)[0] + ".cute_dsl.blackwell.fp8_block_scale_moe_prefill"
)
# Weight-view keys the pipeline reads; any other tensor in the view (SwiGLU
# alpha / beta / clamp, biases) changes the math, so such a view is rejected.
_WEIGHT_KEYS = (
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
)

# Tokens of the dummy problem launched once per device after compilation, so the
# first launch of every program (module load, kernel attributes) happens outside
# CUDA-graph capture.
_WARM_TOKENS = 16

_lock = threading.Lock()
# Per device index: "ready" | "failed".
_device_state: Dict[int, str] = {}
_capability: Dict[int, tuple] = {}
_marker_logged = [False]
_announced_reasons: set = set()

# Diagnostics (see :func:`stats`).
STATS: Dict[str, Any] = {
    "calls": 0,
    "dispatch_count": 0,
    "capture_dispatch_count": 0,
    "dispatched_num_tokens": {},
    "rejects": {},
    "precompile_seconds": {},
    "precompile_errors": [],
    "run_errors": [],
}


def _reject(reason: str, detail: str = "") -> bool:
    rejects = STATS["rejects"]
    rejects[reason] = rejects.get(reason, 0) + 1
    if reason not in _announced_reasons:
        _announced_reasons.add(reason)
        logger.debug(
            f"{FALLBACK_MARKER}: other MoELayer candidates, reason={reason}"
            + (f" ({detail})" if detail else "")
            + "; first occurrence in this process, later ones are counted in stats()"
        )
    return False


@functools.cache
def _load_allowlist() -> Optional[dict]:
    """The dispatch surface; ``None`` => never dispatch."""
    try:
        payload = json.loads(
            resources.files(__package__).joinpath(_WORKLOAD_FILE).read_text()
        )
        tokens = payload["num_tokens"]
        lo, hi = int(tokens["min"]), int(tokens["max"])
        exclude = frozenset(int(v) for v in tokens.get("exclude", ()))
        capability = tuple(int(v) for v in payload["compute_capability"])
        if not 1 <= lo <= hi <= MAX_TOKENS:
            raise ValueError(f"num_tokens range [{lo}, {hi}] outside [1, {MAX_TOKENS}]")
        if capability != COMPUTE_CAPABILITY:
            raise ValueError(f"unexpected compute capability {capability}")
        return {"min": lo, "max": hi, "exclude": exclude}
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning(
            f"MoELayer specialized/{NAME}: token-count allowlist unavailable "
            f"({type(exc).__name__}: {str(exc)[:200]}); the kernel never runs"
        )
        return None


def _dense(t: Any, shape: tuple, dtype: torch.dtype) -> bool:
    return (
        isinstance(t, torch.Tensor)
        and t.dtype == dtype
        and tuple(t.shape) == shape
        and t.is_contiguous()
        and t.data_ptr() % 16 == 0
    )


@functools.cache
def cute_dsl_available() -> bool:
    from ......cute_dsl.availability import is_cute_dsl_available

    return is_cute_dsl_available()


def _capability_for(index: int, capturing: bool) -> Optional[tuple]:
    capability = _capability.get(index)
    if capability is None and not capturing:
        capability = tuple(torch.cuda.get_device_capability(index))
        _capability[index] = capability
    return capability


def config_rejection(config: Any, arch: int) -> Optional[str]:
    """Why ``config`` (a ``MoEConfig``) on compute capability ``arch``
    (``major * 10 + minor``) is not the served configuration, or ``None``.

    The quantization pair, output format, activation class, fused shared experts
    and expert parallelism are checked earlier from the kernel's declarations;
    this covers the rest of the geometry."""
    from ......tllm_enums import RoutingMethodType

    if arch != COMPUTE_CAPABILITY[0] * 10 + COMPUTE_CAPABILITY[1]:
        return f"requires exact SM103, got SM{arch}"
    routing, experts, quant = config.routing, config.experts, config.quant
    local_num_experts = (
        routing.num_experts
        if experts.local_num_experts is None
        else experts.local_num_experts
    )
    if (
        routing.num_experts != NUM_EXPERTS
        or local_num_experts != NUM_EXPERTS
        or experts.local_expert_offset != 0
        or routing.top_k != TOP_K
        or experts.intermediate_size != INTERMEDIATE_SIZE
        or experts.num_fused_shared_experts != 0
    ):
        return (
            f"serves only num_experts={NUM_EXPERTS} (all local), top_k={TOP_K}, "
            f"intermediate_size={INTERMEDIATE_SIZE} without shared experts; got "
            f"num_experts={routing.num_experts}, local_num_experts="
            f"{local_num_experts}, local_expert_offset={experts.local_expert_offset}"
            f", top_k={routing.top_k}, intermediate_size={experts.intermediate_size}"
            f", num_fused_shared_experts={experts.num_fused_shared_experts}"
        )
    if int(routing.method) != int(RoutingMethodType.RenormalizeNaive):
        return f"serves only RenormalizeNaive routing, got {routing.method!r}"
    if routing.n_group not in (None, 0) or routing.topk_group not in (None, 0):
        return "does not support grouped routing (n_group / topk_group)"
    if routing.routed_scaling_factor not in (None, 1.0):
        return (
            f"does not support routed_scaling_factor={routing.routed_scaling_factor!r}"
        )
    if quant.pair != (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8):
        return f"serves only DeepSeekFp8 x DeepSeekFp8, got {quant!r}"
    if quant.swizzled_scale_factors is True:
        return "consumes only the linear activation block-scale layout"
    if config.activation != SwiGLU():
        return f"serves only the default SwiGLU(), got {config.activation!r}"
    if not config.finalize.do_finalize:
        return "requires do_finalize=True"
    if not cute_dsl_available():
        return "requires the CuTe DSL (nvidia-cutlass-dsl)"
    return None


def _contract_rejection(act: Any, view: Mapping[str, Any]) -> Optional[str]:
    """Reject reason of the per-call tensor contract (dtypes, shapes, contiguity,
    16-byte alignment, no extra weight-view tensors), or ``None``."""
    if act.routing_input_mode != RoutingInputMode.FromLogits:
        return "routing_input_mode"
    if act.routing_bias is not None or act.per_token_scale is not None:
        return "routing_options"
    hidden_states = act.hidden_states_q
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return "input_layout"
    num_tokens = int(hidden_states.shape[0])
    if not (
        _dense(act.routing_logits, (num_tokens, NUM_EXPERTS), torch.bfloat16)
        and _dense(hidden_states, (num_tokens, HIDDEN_SIZE), torch.float8_e4m3fn)
        and _dense(
            act.hidden_states_scale,
            (HIDDEN_SIZE // BLOCK, num_tokens),
            torch.float32,
        )
    ):
        return "input_layout"
    if any(view.get(key) is not None for key in view.keys() if key not in _WEIGHT_KEYS):
        return "weight_view_extras"
    if not (
        _dense(
            view.get("gemm1_weights"),
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            torch.float8_e4m3fn,
        )
        and _dense(
            view.get("gemm1_weights_scale"),
            (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE // BLOCK, HIDDEN_SIZE // BLOCK),
            torch.float32,
        )
        and _dense(
            view.get("gemm2_weights"),
            (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            torch.float8_e4m3fn,
        )
        and _dense(
            view.get("gemm2_weights_scale"),
            (NUM_EXPERTS, HIDDEN_SIZE // BLOCK, INTERMEDIATE_SIZE // BLOCK),
            torch.float32,
        )
    ):
        return "weight_layout"
    return None


def check_contract(act: Any, view: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a pack or weight view breaks the tensor contract
    (direct runner use; ``MoELayer`` filters such calls through :func:`admit`)."""
    reason = _contract_rejection(act, view)
    if reason is not None:
        raise ValueError(
            f"MoELayer specialized/{NAME}: call outside the served tensor contract "
            f"({reason})"
        )


def admit(
    act: Any,
    view: Mapping[str, Any],
    device: torch.device,
    eager_num_tokens: Set[int],
) -> bool:
    """True when this call may run the specialized pipeline. Order: kill switch,
    tensor contract, token count, device / arch, readiness. ``eager_num_tokens``
    is the calling binding's set of token counts it admitted outside capture
    (updated here). Never raises."""
    STATS["calls"] += 1
    try:
        if kill_switch_engaged():
            return _reject("kill_switch", f"{KILL_SWITCH_ENV}=1")
        reason = _contract_rejection(act, view)
        if reason is not None:
            return _reject(reason)
        num_tokens = int(act.hidden_states_q.shape[0])
        allowlist = _load_allowlist()
        if allowlist is None:
            return _reject("allowlist_unavailable")
        if (
            not allowlist["min"] <= num_tokens <= allowlist["max"]
            or num_tokens in allowlist["exclude"]
        ):
            return _reject(
                "num_tokens",
                f"num_tokens={num_tokens} outside "
                f"[{allowlist['min']}, {allowlist['max']}]",
            )
        # -- device last
        tensors = [
            act.hidden_states_q,
            act.routing_logits,
            act.hidden_states_scale,
            *(view[key] for key in _WEIGHT_KEYS),
        ]
        if device.type != "cuda" or any(t.device != device for t in tensors):
            return _reject("device")
        if torch.cuda.current_device() != device.index:
            return _reject("device")
        capturing = is_capturing()
        capability = _capability_for(device.index, capturing)
        if capability is None:
            return _reject("capture_unready")
        if capability != COMPUTE_CAPABILITY:
            return _reject("arch", f"compute capability {capability}")
        state = _device_state.get(device.index)
        if state == "failed":
            return _reject("precompile_failed", "see stats()['precompile_errors']")
        if capturing:
            if state is None:
                return _reject(
                    "capture_unready",
                    "first call of this configuration under CUDA-graph capture; an "
                    "eager call before capture builds the specialized kernels",
                )
            if num_tokens not in eager_num_tokens:
                return _reject(
                    "capture_unselected",
                    f"num_tokens={num_tokens} first seen under CUDA-graph capture; "
                    "MoELayer selects its winner from an eager call",
                )
            return True
        if state is None and not prepare(device, view):
            return _reject("precompile_failed")
        eager_num_tokens.add(num_tokens)
        return True
    except Exception as exc:  # a guard failure must never break the layer
        return _reject("guard_error", f"{type(exc).__name__}: {str(exc)[:500]}")


def prepare(device: torch.device, view: Mapping[str, Any]) -> bool:
    """Compile every program for ``device`` and launch both program sets (compact
    and padded) once on a dummy problem (eagerly, synchronized) so no later call
    compiles, loads a module or sets kernel attributes. Never runs under
    CUDA-graph capture. Returns whether the device is ready."""
    index = device.index
    state = _device_state.get(index)
    if state is not None:
        return state == "ready"
    with _lock:
        state = _device_state.get(index)
        if state is not None:
            return state == "ready"
        start = time.perf_counter()
        try:
            from .....cute_dsl.blackwell import fp8_block_scale_moe_prefill as kernels

            with torch.cuda.device(device):
                kernels.compile_kernels(device)
                t = _WARM_TOKENS
                logits = torch.zeros(
                    t, NUM_EXPERTS, dtype=torch.bfloat16, device=device
                )
                hidden = torch.zeros(
                    t, HIDDEN_SIZE, dtype=torch.uint8, device=device
                ).view(torch.float8_e4m3fn)
                hidden_scale = torch.ones(
                    HIDDEN_SIZE // BLOCK, t, dtype=torch.float32, device=device
                )
                out = torch.empty(t, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
                # Both program sets: the compact one (T <= 8192) and the padded one
                # (larger T, e.g. the 32768-token bucket captured first).
                for compact in (True, False):
                    kernels.run(
                        logits,
                        hidden,
                        hidden_scale,
                        *(view[key] for key in _WEIGHT_KEYS),
                        out,
                        compact=compact,
                    )
                    torch.cuda.current_stream(device).synchronize()
                    # all-zero activations give an exactly zero output
                    if bool(out.any()):
                        raise RuntimeError(
                            "warm-up launch of the all-zero dummy problem produced "
                            f"a nonzero output ({'compact' if compact else 'padded'} "
                            "programs)"
                        )
        except Exception as exc:  # the other candidates keep the layer working
            message = f"{type(exc).__name__}: {exc}"
            STATS["precompile_errors"].append(
                {"device": index, "error": message[:2000]}
            )
            _device_state[index] = "failed"
            logger.warning(
                f"MoELayer specialized/{NAME}: programs unavailable on cuda:{index}, "
                f"using the other candidates ({message[:2000]})"
            )
            return False
        seconds = time.perf_counter() - start
        STATS["precompile_seconds"][index] = round(seconds, 3)
        _device_state[index] = "ready"
        logger.info(
            f"MoELayer specialized/{NAME}: compiled the programs on cuda:{index} "
            f"({kernels.NUM_PROGRAMS} programs, {seconds:.1f} s)"
        )
        return True


def is_ready(device: torch.device) -> bool:
    return _device_state.get(device.index) == "ready"


def run(inputs: list) -> torch.Tensor:
    """Run the pipeline on the packed inputs ``[routing_logits, hidden_states,
    hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights,
    gemm2_weights_scale, output]`` (programs built for their device) into
    ``output``. Raises on a launch failure, after disabling the kernel on the
    device for the rest of the process."""
    from .....cute_dsl.blackwell import fp8_block_scale_moe_prefill as kernels

    hidden_states = inputs[1]
    num_tokens = int(hidden_states.shape[0])
    capturing = is_capturing()
    try:
        kernels.run(*inputs)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        STATS["run_errors"].append(
            {"num_tokens": num_tokens, "capturing": capturing, "error": message[:2000]}
        )
        _device_state[hidden_states.device.index] = "failed"
        logger.warning(
            f"MoELayer specialized/{NAME}: pipeline failed (num_tokens={num_tokens}), "
            f"disabled for this process ({message[:2000]})"
        )
        raise
    STATS["dispatch_count"] += 1
    if capturing:
        STATS["capture_dispatch_count"] += 1
    seen = STATS["dispatched_num_tokens"]
    seen[num_tokens] = seen.get(num_tokens, 0) + 1
    if not _marker_logged[0]:
        _marker_logged[0] = True
        allowlist = _load_allowlist() or {}
        logger.info(
            f"{DISPATCH_MARKER}: first dispatch num_tokens={num_tokens} "
            f"cuda_graph_capture={capturing} "
            f"num_tokens_allowlist=[{allowlist.get('min')}, {allowlist.get('max')}] "
            f"({KILL_SWITCH_ENV}=1 keeps MoELayer on its other candidates)"
        )
    return inputs[-1]


def stats() -> Dict[str, Any]:
    """Dispatch / compile diagnostics of the kernel.

    Side-effect free and cheap, so it can be polled from any thread of any process:
    it never imports the kernel module (or the CuTe DSL), never builds, never
    touches CUDA and never reads the allowlist file; fields that need those report
    what this process already has (``0`` / ``None`` when the kernel was never used).

    ``calls``: per-call admission checks (``MoELayer`` asks once per call);
    ``dispatch_count`` / ``capture_dispatch_count``: launches of the pipeline
    (eager + capture / capture only; ``MoELayer``'s one-off timing of candidates
    included); ``rejects``: rejected calls by reason; ``compiled_variants``: CuTe
    DSL programs compiled in this process (all devices);
    ``distinct_kernels_for_allowlist``: programs that cover every allowlisted
    token count on one device (none depends on the token count); ``precompiled``:
    devices whose programs are built and warm.
    """
    kernels = sys.modules.get(_KERNEL_MODULE)
    allowlist = None
    cache_info = getattr(_load_allowlist, "cache_info", None)
    if cache_info is not None and cache_info().currsize:
        allowlist = _load_allowlist()  # cached: no file read
    compiled, per_device, max_ws = 0, None, None
    if kernels is not None:
        try:
            compiled = kernels.compiled_program_count()
            per_device = kernels.NUM_PROGRAMS
            if allowlist:
                max_ws = kernels.workspace_bytes(allowlist["max"])
        except Exception:
            pass
    # C-level copies first: the serving thread may add keys while a probe reads.
    device_state = dict(_device_state)
    return {
        "calls": STATS["calls"],
        "dispatch_count": STATS["dispatch_count"],
        "capture_dispatch_count": STATS["capture_dispatch_count"],
        "dispatched_num_tokens": dict(
            sorted(dict(STATS["dispatched_num_tokens"]).items())
        ),
        "rejects": dict(STATS["rejects"]),
        "kernel_module_loaded": kernels is not None,
        "compiled_variants": compiled,
        "distinct_kernels_for_allowlist": per_device,
        "compile_cache_key": "device index (token count is a runtime argument)",
        "precompiled": sorted(i for i, st in device_state.items() if st == "ready"),
        "failed_devices": sorted(i for i, st in device_state.items() if st == "failed"),
        "allowlist": None
        if allowlist is None
        else {
            "min": allowlist["min"],
            "max": allowlist["max"],
            "exclude": sorted(allowlist["exclude"]),
        },
        "workspace_bytes_at_max_tokens": max_ws,
        "precompile_seconds": dict(STATS["precompile_seconds"]),
        "precompile_errors": list(STATS["precompile_errors"]),
        "run_errors": list(STATS["run_errors"]),
    }


def _reset_for_tests() -> None:
    """Forget per-process state (counters, readiness) -- tests only. Compiled
    programs stay cached in the kernel module."""
    _device_state.clear()
    _marker_logged[0] = False
    _announced_reasons.clear()
    STATS["calls"] = 0
    STATS["dispatch_count"] = 0
    STATS["capture_dispatch_count"] = 0
    STATS["dispatched_num_tokens"] = {}
    STATS["rejects"] = {}
    STATS["precompile_seconds"] = {}
    STATS["precompile_errors"] = []
    STATS["run_errors"] = []


class Fp8BlockPrefillSm103Qwen35Kernel(SpecializedMoEKernel):
    """FP8 block-scale MoE prefill of Qwen3.5-397B-A17B-FP8 on exact SM103 as one
    CuTe DSL pipeline.

    Serves one configuration: 512 local experts, top-10 ``RenormalizeNaive``
    routing from ``bfloat16`` logits (``RoutingInputMode.FromLogits``, no bias),
    hidden size 4096, intermediate size 1024, the default :class:`SwiGLU`,
    ``DeepSeekFp8`` x ``DeepSeekFp8`` with bf16 output and ``do_finalize=True``;
    calls of 1792 to 32768 tokens (``workloads.json``). It reads the
    ``trtllm_fp8_block`` weight view and computes the trtllm-gen DeepSeek-FP8
    data path: GEMM1 and SwiGLU outputs re-quantized to FP8 per (token, 128
    columns), fp32 accumulation, bf16 output.
    """

    name = NAME
    weight_view = WEIGHT_VIEW
    quant_variants = ((QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),)
    routing_modes = (RoutingInputMode.FromLogits,)
    activation_classes = (SwiGLU,)
    compute_capabilities = (COMPUTE_CAPABILITY[0] * 10 + COMPUTE_CAPABILITY[1],)

    @classmethod
    def config_rejection(
        cls, config: Any, arch: int, device: torch.device
    ) -> Optional[str]:
        return config_rejection(config, arch)

    def accepts(self, act: Any, view: Mapping[str, Any]) -> bool:
        return admit(act, view, self.device, self.eager_num_tokens)

    def pack_inputs(self, act: Any, view: Mapping[str, Any]) -> List[Any]:
        check_contract(act, view)
        output = torch.empty(
            (act.num_tokens, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=act.hidden_states_q.device,
        )
        return [
            act.routing_logits,
            act.hidden_states_q,
            act.hidden_states_scale,
            view["gemm1_weights"],
            view["gemm1_weights_scale"],
            view["gemm2_weights"],
            view["gemm2_weights_scale"],
            output,
        ]

    def forward(self, inputs: List[Any]) -> torch.Tensor:
        device = inputs[1].device
        if not is_ready(device):
            if is_capturing():
                raise RuntimeError(
                    f"MoELayer specialized/{NAME}: the programs are not built for "
                    f"{device}; run an eager call before CUDA-graph capture."
                )
            view = dict(zip(_WEIGHT_KEYS, inputs[3:7], strict=True))
            if not prepare(device, view):
                raise RuntimeError(
                    f"MoELayer specialized/{NAME}: the programs failed to build on "
                    f"{device}; see its stats()['precompile_errors']."
                )
        try:
            return run(inputs)
        except Exception:
            # A failed launch disabled the kernel on this device (later calls run
            # on the other candidates); this call falls back to the stock
            # trtllm-gen kernel, which rewrites the whole output.
            return self._run_trtllm_gen(inputs)

    def _run_trtllm_gen(self, inputs: List[Any]) -> torch.Tensor:
        from ......tllm_enums import (
            ActivationType,
            Fp8QuantizationType,
            RoutingMethodType,
            WeightLayout,
        )
        from .....core import trtllm_fp8_block_scale_moe

        logits, hidden, hidden_scale, w1, s1, w2, s2, output = inputs
        trtllm_fp8_block_scale_moe(
            routing_logits=logits,
            routing_bias=None,
            hidden_states=hidden,
            hidden_states_scale=hidden_scale,
            gemm1_weights=w1,
            gemm1_weights_scale=s1,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=None,
            routing_method_type=int(RoutingMethodType.RenormalizeNaive),
            use_shuffled_weight=False,
            weight_layout=int(WeightLayout.MajorK),
            enable_pdl=self.config.execution.enable_pdl,
            tune_max_num_tokens=self.config.execution.tune_max_num_tokens,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            activation_type=ActivationType.Swiglu.value,
            output=output,
        )
        return output

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        return stats()
