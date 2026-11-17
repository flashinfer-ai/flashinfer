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

"""Specialized kernel ``bf16_decode_sm103_glm52``: BF16 MoE decode on SM103.

Not a public API: ``MoELayer`` reaches it through its ``specialized`` backend
(:class:`flashinfer.fused_moe.SpecializedMoEConfig`).

One expert geometry, the GLM-5.2 MTP layer without expert parallelism, is
served by fused CUDA kernels (``csrc/fused_moe/bf16_moe_decode_sm103.cu``, JIT
module ``flashinfer/jit/bf16_moe_decode_sm103.py``): 256 local experts,
DeepSeekV3 routing with one expert group (sigmoid scores, top-8 by score + bias,
weights renormalized and scaled by ``routed_scaling_factor``) from fp32 logits
and an fp32 bias, hidden 6144, intermediate 256, the default SwiGLU, BF16 x
BF16 with bf16 output and ``do_finalize=True``, on SM103 GPUs with 150..152
SMs, for the decode token counts in the package-data allowlist
``workloads.json``. The kernels read the ``trtllm_bf16_routed`` weight view
(:class:`TrtllmBf16Config`: shuffled ``BlockMajorK`` bf16 weights).

Two levels of checks keep every other call on ``MoELayer``'s other candidates
(``trtllm_bf16_routed`` in the default list):

* configuration (:func:`config_rejection`, at ``MoELayer`` construction): any
  other configuration or device makes the backend skip this kernel;
* per call (:func:`admit`): kill switch, the activation / routing / weight-view
  tensor contract, token count, device and readiness. Never raises.

Build: the module is loaded (JIT-compiled, or taken from the AOT cache) and
launched once per allowlisted token count on a dummy problem by the first eager
call admitted, outside any capture. Under CUDA-graph capture a call is admitted
only when the kernels are loaded and this binding already admitted the same
token count eagerly, i.e. when ``MoELayer`` already timed its candidates for
that call outside capture; nothing is built inside a capture.

The kernels keep cross-CTA state in device globals of the loaded module (one
instance per device), so calls on one device must be stream-ordered, as a
serving step's layers are.

Numerics: the trtllm-gen BF16 data path -- fp32 accumulation for both GEMMs and
for the weighted reduction over the top-k experts; routing weights, the SwiGLU
output and each expert's GEMM2 output rounded to bf16 where trtllm-gen stores
them; bf16 output.
"""

from __future__ import annotations

import functools
import json
import math
import threading
import time
from importlib import resources
from typing import Any, Dict, List, Mapping, Optional, Set

import torch

from ......jit.core import logger
from ......tllm_enums import RoutingMethodType
from .....api import QuantFormat, RoutingInputMode, SwiGLU
from ...kernel import (
    KILL_SWITCH_ENV,
    SpecializedMoEKernel,
    is_capturing,
    kill_switch_engaged,
)

NAME = "bf16_decode_sm103_glm52"
WEIGHT_VIEW = "trtllm_bf16_routed"

# Logged once per process at the first dispatch (grep-able).
DISPATCH_MARKER = (
    f"flashinfer specialized kernel: MoELayer specialized/{NAME} "
    "sm103 e256 h6144 i256 top8 decode"
)
# Prefix of the once-per-reason rejection lines (debug level); deliberately does
# not contain the dispatch marker, so grepping the marker never matches a reject.
FALLBACK_MARKER = (
    f"flashinfer specialized kernel fallback (MoELayer specialized/{NAME})"
)

# Compile-time geometry of the kernels.
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 256
BLOCK_K = 64  # BlockMajorK blocks of 128 bytes of bf16
COMPUTE_CAPABILITY = (10, 3)
# The kernels' work split assumes this many SMs (one CTA per SM).
SM_COUNT_RANGE = (150, 152)
# Upper bound of the allowlist: one launch serves at most 16 tokens.
MAX_TOKENS = 16

GEMM1_SHAPE = (NUM_EXPERTS, HIDDEN_SIZE // BLOCK_K, 2 * INTERMEDIATE_SIZE, BLOCK_K)
GEMM2_SHAPE = (NUM_EXPERTS, INTERMEDIATE_SIZE // BLOCK_K, HIDDEN_SIZE, BLOCK_K)
# Weight-view keys the kernels read; any other tensor in the view (SwiGLU alpha /
# beta / clamp) changes the math, so such a view is rejected.
_WEIGHT_KEYS = ("gemm1_weights", "gemm2_weights")

# Dispatch surface (package data next to this module).
_WORKLOAD_FILE = "workloads.json"
# The loaded JIT module (one per process; its device globals are per device).
_module: list = []

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
        values = frozenset(int(v) for v in tokens)
        capability = tuple(int(v) for v in payload["compute_capability"])
        if not values or min(values) < 1 or max(values) > MAX_TOKENS:
            raise ValueError(f"num_tokens {sorted(values)} outside [1, {MAX_TOKENS}]")
        if capability != COMPUTE_CAPABILITY:
            raise ValueError(f"unexpected compute capability {capability}")
        return {"num_tokens": values}
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


def _load_module():
    if not _module:
        from ......jit.bf16_moe_decode_sm103 import gen_bf16_moe_decode_sm103_module

        _module.append(gen_bf16_moe_decode_sm103_module().build_and_load())
    return _module[0]


def _capability_for(index: int, capturing: bool) -> Optional[tuple]:
    capability = _capability.get(index)
    if capability is None and not capturing:
        capability = tuple(torch.cuda.get_device_capability(index))
        _capability[index] = capability
    return capability


def _scale(routed_scaling_factor: Optional[float]) -> float:
    # trtllm-gen reads an absent factor as 1.0
    return 1.0 if routed_scaling_factor is None else float(routed_scaling_factor)


def _prepare(
    device: torch.device,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    token_counts: frozenset,
) -> bool:
    """Load the kernel module, check the device and launch every allowlisted token
    count once on a dummy problem (eagerly, synchronized) so no later call builds,
    loads a module or sets kernel attributes. Never runs under CUDA-graph capture."""
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
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            if not SM_COUNT_RANGE[0] <= sms <= SM_COUNT_RANGE[1]:
                raise RuntimeError(
                    f"{sms} SMs; the work split needs {SM_COUNT_RANGE[0]}..{SM_COUNT_RANGE[1]}"
                )
            module = _load_module()
            with torch.cuda.device(device):
                # all-zero activations give an exactly zero output whatever the
                # routing (the dummy inputs of a server's warm-up); the output is
                # pre-filled with NaN, so every element must be written
                for t in sorted(token_counts):
                    logits = torch.zeros(
                        t, NUM_EXPERTS, dtype=torch.float32, device=device
                    )
                    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device=device)
                    hidden = torch.zeros(
                        t, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
                    )
                    out = torch.full(
                        (t, HIDDEN_SIZE),
                        float("nan"),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    module.run(
                        logits, bias, hidden, gemm1_weights, gemm2_weights, out, 1.0
                    )
                    torch.cuda.current_stream(device).synchronize()
                    if bool(out.isnan().any()) or bool(out.any()):
                        raise RuntimeError(
                            "warm-up launch of the all-zero dummy problem did not "
                            f"produce an all-zero output (num_tokens={t})"
                        )
        except Exception as exc:  # the other candidates keep the layer working
            message = f"{type(exc).__name__}: {exc}"
            STATS["precompile_errors"].append(
                {"device": index, "error": message[:2000]}
            )
            _device_state[index] = "failed"
            logger.warning(
                f"MoELayer specialized/{NAME}: kernels unavailable on cuda:{index}, "
                f"using the other candidates ({message[:2000]})"
            )
            return False
        seconds = time.perf_counter() - start
        STATS["precompile_seconds"][index] = round(seconds, 3)
        _device_state[index] = "ready"
        logger.info(
            f"MoELayer specialized/{NAME}: loaded the kernels on cuda:{index} "
            f"({seconds:.1f} s)"
        )
        return True


def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    routed_scaling_factor: Optional[float],
    output: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run the kernels (after :func:`admit` returned True) into ``output``.

    Returns the output, or ``None`` when the launch failed before producing it
    (the caller then runs the trtllm-gen path, which rewrites the whole output)."""
    num_tokens = int(hidden_states.shape[0])
    capturing = is_capturing()
    try:
        # serving passes the routing bias as a parameter that requires grad
        _load_module().run(
            routing_logits,
            routing_bias.detach(),
            hidden_states,
            gemm1_weights,
            gemm2_weights,
            output,
            _scale(routed_scaling_factor),
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        STATS["run_errors"].append(
            {"num_tokens": num_tokens, "capturing": capturing, "error": message[:2000]}
        )
        _device_state[hidden_states.device.index] = "failed"
        logger.warning(
            f"MoELayer specialized/{NAME}: kernel failed (num_tokens={num_tokens}), "
            f"disabled for this process; using trtllm-gen ({message[:2000]})"
        )
        return None
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
            f"num_tokens_allowlist={sorted(allowlist.get('num_tokens', ()))} "
            f"({KILL_SWITCH_ENV}=1 keeps MoELayer on its other candidates)"
        )
    return output


def config_rejection(
    config: Any, arch: int, device: Optional[torch.device] = None
) -> Optional[str]:
    """Why ``config`` (a ``MoEConfig``) on compute capability ``arch``
    (``major * 10 + minor``) and ``device`` is not the served configuration, or
    ``None``. The quantization pair, activation class, fused shared experts and
    expert parallelism are checked earlier from the kernel's declarations; this
    covers the rest of the geometry."""
    if arch != COMPUTE_CAPABILITY[0] * 10 + COMPUTE_CAPABILITY[1]:
        return f"requires exact SM103, got SM{arch}"
    routing, experts = config.routing, config.experts
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
            f"intermediate_size={INTERMEDIATE_SIZE} without shared experts"
        )
    if int(routing.method) != int(RoutingMethodType.DeepSeekV3):
        return f"serves only DeepSeekV3 routing, got {routing.method!r}"
    if routing.n_group != 1 or routing.topk_group != 1:
        return "serves only one expert group (n_group = topk_group = 1)"
    scale = _scale(routing.routed_scaling_factor)
    if not (math.isfinite(scale) and scale > 0.0):
        return f"needs a finite positive routed_scaling_factor, got {scale!r}"
    if config.activation != SwiGLU():
        return f"serves only the default SwiGLU(), got {config.activation!r}"
    if not config.finalize.do_finalize:
        return "requires do_finalize=True"
    # -- device last: the only check that queries the device
    if device is not None and device.type == "cuda":
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        if not SM_COUNT_RANGE[0] <= sms <= SM_COUNT_RANGE[1]:
            return (
                f"needs {SM_COUNT_RANGE[0]}..{SM_COUNT_RANGE[1]} SMs, "
                f"{device} has {sms}"
            )
    return None


def _contract_rejection(act: Any, view: Mapping[str, Any]) -> Optional[str]:
    """Reject reason of the per-call tensor contract (dtypes, shapes, contiguity,
    16-byte alignment, no extra weight-view tensors), or ``None``."""
    if act.routing_input_mode != RoutingInputMode.FromLogits:
        return "routing_input_mode"
    if act.per_token_scale is not None:
        return "routing_options"
    hidden_states = act.hidden_states_q
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 2:
        return "input_layout"
    num_tokens = int(hidden_states.shape[0])
    if not (
        _dense(act.routing_logits, (num_tokens, NUM_EXPERTS), torch.float32)
        and _dense(act.routing_bias, (NUM_EXPERTS,), torch.float32)
        and _dense(hidden_states, (num_tokens, HIDDEN_SIZE), torch.bfloat16)
    ):
        return "input_layout"
    if any(view.get(key) is not None for key in view.keys() if key not in _WEIGHT_KEYS):
        return "weight_view_extras"
    if not (
        _dense(view.get("gemm1_weights"), GEMM1_SHAPE, torch.bfloat16)
        and _dense(view.get("gemm2_weights"), GEMM2_SHAPE, torch.bfloat16)
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
    """True when this ``MoELayer`` call may run the kernels. Order: kill switch,
    tensor contract, token count, device / arch, readiness and CUDA-graph
    capture. ``eager_num_tokens`` is the calling binding's set of token counts it
    admitted outside capture (updated here): under capture only those are
    admitted, i.e. counts for which ``MoELayer`` already selected its winner
    eagerly. The first admitted eager call builds the kernels. Never raises."""
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
        if num_tokens not in allowlist["num_tokens"]:
            return _reject("num_tokens", f"num_tokens={num_tokens} not in allowlist")
        # -- device last
        tensors = [
            act.hidden_states_q,
            act.routing_logits,
            act.routing_bias,
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
                    "eager call before capture builds the kernels",
                )
            if num_tokens not in eager_num_tokens:
                return _reject(
                    "capture_unselected",
                    f"num_tokens={num_tokens} first seen under CUDA-graph capture; "
                    "MoELayer selects its winner from an eager call",
                )
            return True
        if state is None and not _prepare(
            device,
            view["gemm1_weights"],
            view["gemm2_weights"],
            allowlist["num_tokens"],
        ):
            return _reject("precompile_failed")
        eager_num_tokens.add(num_tokens)
        return True
    except Exception as exc:  # a guard failure must never break the layer
        return _reject("guard_error", f"{type(exc).__name__}: {str(exc)[:500]}")


def is_ready(device: torch.device) -> bool:
    return _device_state.get(device.index) == "ready"


def stats() -> Dict[str, Any]:
    """Dispatch / build diagnostics of the kernel.

    Side-effect free and cheap, so it can be polled from any thread of any process:
    it never builds or loads the CUDA module, never touches CUDA and never reads the
    allowlist file; fields that need those report what this process already has
    (``0`` / ``None`` when the kernel was never used).

    ``calls``: per-call admission checks (``MoELayer`` asks once per call);
    ``dispatch_count`` / ``capture_dispatch_count``: launches of the kernels
    (eager + capture / capture only; ``MoELayer``'s one-off timing of candidates
    included); ``rejects``: rejected calls by reason; ``compiled_variants``:
    loaded kernel modules in this process (one module holds every token-count
    specialization); ``distinct_kernels_for_allowlist``: kernel specializations the
    allowlisted token counts launch; ``precompiled``: devices whose kernels are
    loaded and warm.
    """
    allowlist = None
    cache_info = getattr(_load_allowlist, "cache_info", None)
    if cache_info is not None and cache_info().currsize:
        allowlist = _load_allowlist()  # cached: no file read
    per_device = None
    if allowlist:
        # one, two, six and ten tokens have their own kernels; other counts share
        # the generic one
        per_device = len(
            {t if t in (1, 2, 6, 10) else 0 for t in allowlist["num_tokens"]}
        )
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
        "kernel_module_loaded": bool(_module),
        "compiled_variants": len(_module),
        "distinct_kernels_for_allowlist": per_device,
        "compile_cache_key": "one module (token count is a runtime argument)",
        "precompiled": sorted(i for i, st in device_state.items() if st == "ready"),
        "failed_devices": sorted(i for i, st in device_state.items() if st == "failed"),
        "allowlist": None
        if allowlist is None
        else {"num_tokens": sorted(allowlist["num_tokens"])},
        "precompile_seconds": dict(STATS["precompile_seconds"]),
        "precompile_errors": list(STATS["precompile_errors"]),
        "run_errors": list(STATS["run_errors"]),
    }


def _reset_for_tests() -> None:
    """Forget per-process state (counters, readiness) -- tests only. The loaded
    module stays cached."""
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


class Bf16DecodeSm103Glm52Kernel(SpecializedMoEKernel):
    """BF16 MoE decode of the GLM-5.2 MTP layer on SM103 as fused CUDA kernels.

    Serves one configuration: 256 local experts, ``DeepSeekV3`` top-8 routing
    with one expert group from ``float32`` logits and a ``float32`` bias
    (``RoutingInputMode.FromLogits``), hidden size 6144, intermediate size 256,
    the default :class:`SwiGLU`, ``BF16`` x ``BF16`` with bf16 output and
    ``do_finalize=True``, on SM103 GPUs with 150..152 SMs; calls of the token
    counts in ``workloads.json``. It reads the ``trtllm_bf16_routed`` weight
    view and computes the trtllm-gen BF16 data path (routing weights, SwiGLU and
    GEMM2 outputs rounded to bf16, fp32 accumulation).
    """

    name = NAME
    weight_view = WEIGHT_VIEW
    quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
    routing_modes = (RoutingInputMode.FromLogits,)
    activation_classes = (SwiGLU,)
    compute_capabilities = (COMPUTE_CAPABILITY[0] * 10 + COMPUTE_CAPABILITY[1],)

    @classmethod
    def config_rejection(
        cls, config: Any, arch: int, device: torch.device
    ) -> Optional[str]:
        return config_rejection(config, arch, device)

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
            act.routing_bias,
            act.hidden_states_q,
            view["gemm1_weights"],
            view["gemm2_weights"],
            output,
        ]

    def forward(self, inputs: List[Any]) -> torch.Tensor:
        logits, bias, hidden, w1, w2, output = inputs
        device = hidden.device
        if not is_ready(device):
            if is_capturing():
                raise RuntimeError(
                    f"MoELayer specialized/{NAME}: the kernels are not loaded on "
                    f"{device}; run an eager call before CUDA-graph capture."
                )
            allowlist = _load_allowlist() or {"num_tokens": frozenset()}
            if not _prepare(device, w1, w2, allowlist["num_tokens"]):
                raise RuntimeError(
                    f"MoELayer specialized/{NAME}: the kernels failed to load on "
                    f"{device}; see its stats()['precompile_errors']."
                )
        scale = self.config.routing.routed_scaling_factor
        result = run(logits, bias, hidden, w1, w2, scale, output)
        if result is not None:
            return result
        # A failed launch disabled the kernel on this device (later calls run on
        # the other candidates); this call falls back to the stock trtllm-gen
        # kernel, which rewrites the whole output.
        return self._run_trtllm_gen(inputs)

    def _run_trtllm_gen(self, inputs: List[Any]) -> torch.Tensor:
        from ......tllm_enums import ActivationType, WeightLayout
        from .....core import trtllm_bf16_moe

        logits, bias, hidden, w1, w2, output = inputs
        trtllm_bf16_moe(
            routing_logits=logits,
            routing_bias=bias,
            hidden_states=hidden,
            gemm1_weights=w1,
            gemm2_weights=w2,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            n_group=1,
            topk_group=1,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=self.config.routing.routed_scaling_factor,
            routing_method_type=int(RoutingMethodType.DeepSeekV3),
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            do_finalize=True,
            enable_pdl=self.config.execution.enable_pdl is not False,
            tune_max_num_tokens=self.config.execution.tune_max_num_tokens,
            activation_type=ActivationType.Swiglu.value,
            output=output,
        )
        return output

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        return stats()

    @classmethod
    def aot_jit_specs(cls, sm_capabilities: Mapping[str, bool]) -> List[Any]:
        if not sm_capabilities.get("sm103a_exact", False):
            return []
        from ......jit.bf16_moe_decode_sm103 import gen_bf16_moe_decode_sm103_module

        return [gen_bf16_moe_decode_sm103_module()]
