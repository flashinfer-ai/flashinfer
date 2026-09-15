"""Independent BF16 MoE engine built from cuDNN Frost FC1 and FC2 source kernels.

The existing CUTLASS *weight layout* is canonical row-major [up, gate]/down;
we consume that view without invoking or modifying the CUTLASS runner.
Each instance owns its workspace and must be confined to one thread/stream.
"""

from __future__ import annotations

import functools
from itertools import product
from pathlib import Path
from typing import Any

import torch

from ...autotuner import TuningConfig
from ...fused_moe.api import QuantFormat, RoutingInputMode, SwiGLU
from ...fused_moe.runners import MoERunner, _validate_prerouted_inputs
from ...utils import get_compute_capability
from . import fc2, runtime
from .capabilities import require_compiler
from .activations import ACTIVATIONS, activation_name
from .support import large_bf16_moe

_WEIGHT_KEYS = frozenset(("fc1_expert_weights", "fc2_expert_weights"))
_TAG = "cudnn_frost-bf16-moe-v2"


@functools.lru_cache(maxsize=1)
def _module(arch):
    from ...jit.core import gen_jit_spec, sm107a_nvcc_flags

    if arch != "sm_107a":
        raise ValueError("cuDNN Frost BF16 MoE kernels require SM107a")

    return gen_jit_spec(
        f"cudnn_frost_bf16_moe_v2_{arch}",
        [Path(__file__).parent / "csrc" / "moe.cu"],
        extra_cuda_cflags=sm107a_nvcc_flags,
    ).build_and_load()


def _kernels(rows, hidden, intermediate, experts, device, activation=None):
    name = activation_name(SwiGLU() if activation is None else activation)
    values = dict(s=rows, n=intermediate, k=hidden, experts=experts, groups=experts)
    arch = runtime._arch_for(device)
    first = tuple(
        k
        for k in runtime._discover(runtime._artifact_roots())
        if k.arch == arch
        and k.activation == name
        and all(
            runtime._dimension_matches(v, k.contract.get(d)) for d, v in values.items()
        )
    )
    second = fc2.matching_kernels(rows, hidden, intermediate, experts, device)
    return first, second


def _selected_kernels(tokens, hidden, intermediate, experts, topk, device, activation):
    from .shortlist import select

    first, second = _kernels(
        tokens * topk, hidden, intermediate, experts, device, activation
    )
    return select(
        runtime._artifact_roots(),
        runtime._arch_for(device),
        activation_name(activation),
        tokens,
        hidden,
        intermediate,
        experts,
        topk,
        first,
        second,
    )


class _Inputs(list):
    def __init__(self, tensors, state, tuning_config):
        super().__init__(tensors)
        self.launch_state = state
        self.tuning_config = tuning_config


class _Plans:
    def __init__(
        self,
        tokens,
        hidden,
        intermediate,
        experts,
        topk,
        device,
        first,
        second,
        workspace_pool,
    ):
        self.plans = {}
        self.launches = {}
        required = 0
        module = _module(runtime._arch_for(device))
        for a, b in product(first, second):
            key = (_TAG, a.tactic, b.tactic)
            plan = module.make_plan(
                runtime._load_kernel(a, device),
                runtime._load_kernel(b, device),
                tokens,
                hidden,
                intermediate,
                experts,
                topk,
                device.index,
                a.workspace_bytes,
                b.workspace_bytes,
                a.gated,
                [
                    {"output": -1, "scale": 0, "gate_scale": 1, "linear_scale": 2}[name]
                    for name in a.launch_tail
                ],
                a.swap_ab,
                b.swap_ab,
            )
            # Retain Module ownership: its run Function borrows the native plan.
            self.plans[key] = plan
            self.launches[key] = plan["run"]
            required = max(required, plan["workspace_size"]())
        # Exact-shape host plans must not imply one large GPU allocation per
        # token count. Share geometrically grown storage on this runner's stream.
        # Older graph-captured plans keep their original allocation alive.
        capacity = min((size for size in workspace_pool if size >= required), default=0)
        if not capacity:
            capacity = 1 << (max(required, 1) - 1).bit_length()
            workspace_pool[capacity] = torch.empty(
                capacity, dtype=torch.uint8, device=device
            )
        self.workspace = workspace_pool[capacity][:required]


class CudnnFrostBf16MoeRunner(MoERunner):
    """A full MoE candidate, not a GEMM1 override in another backend."""

    backend_key = "cudnn_frost_bf16"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
    supported_activation_classes = tuple(ACTIVATIONS.values())
    supports_expert_parallelism = False

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._plans = {}
        self._workspace_pool = {}

    def _check_support(self):
        super()._check_support()
        if self.device.type != "cuda":
            raise NotImplementedError("cuDNN Frost BF16 MoE kernels require SM107a")
        major, minor = get_compute_capability(self.device)
        if (major, minor) != (10, 7):
            raise NotImplementedError("cuDNN Frost BF16 MoE kernels require SM107a")
        name = activation_name(self.config.activation)
        if not self.config.finalize.do_finalize:
            raise NotImplementedError("cuDNN Frost BF16 MoE requires finalized output")
        if (
            self.config.quant.per_token_scale
            or self.config.quant.swizzled_scale_factors
        ):
            raise NotImplementedError(
                "cuDNN Frost BF16 MoE does not consume quantization scales"
            )
        arch = f"sm_{major}{minor}a"
        roots = runtime._artifact_roots()
        sources = tuple(
            sorted(
                {
                    (kernel.source_path, kernel.source_sha256)
                    for kernel in (*runtime._discover(roots), *fc2.discover(roots))
                    if kernel.arch == arch
                    and (
                        not isinstance(kernel, runtime.CudnnFrostGroupedGemm1Kernel)
                        or kernel.activation == name
                    )
                }
            )
        )
        require_compiler(arch, sources)

    def _build(self):
        # Shape-dependent native resources are prepared at pack time, never in capture.
        pass

    def _validate_pack(self, act, weights):
        if act.routing_input_mode not in self.supported_routing_modes:
            raise NotImplementedError(
                "cuDNN Frost BF16 MoE requires precomputed routing"
            )
        x = act.hidden_states_q
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise ValueError("cuDNN Frost BF16 MoE requires BF16 x[T,H]")
        if act.hidden_states_scale is not None or act.per_token_scale is not None:
            raise ValueError("cuDNN Frost BF16 MoE does not consume activation scales")
        t, h = x.shape
        k, e = self.config.routing.top_k, self.config.routing.num_experts
        i = self.config.experts.intermediate_size
        if not (0 < t <= min(1 << 20, self.config.execution.tune_max_num_tokens)):
            raise ValueError(
                "cuDNN Frost BF16 MoE token count exceeds supported bounds"
            )
        if not (0 < k <= e and t * k < 2**31):
            raise ValueError("cuDNN Frost BF16 MoE requires int32 expanded row indices")
        _validate_prerouted_inputs(
            act,
            t,
            k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )
        view = weights.get_view("cutlass_bf16")
        # Do not silently discard bias, per-expert activation overrides, etc.
        if set(view) != _WEIGHT_KEYS:
            raise ValueError(
                "cuDNN Frost BF16 MoE requires plain weights without overrides"
            )
        w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
        if tuple(w1.shape) != (
            e,
            (2 if self.config.activation.is_gated else 1) * i,
            h,
        ) or tuple(w2.shape) != (e, h, i):
            raise ValueError("cuDNN Frost BF16 MoE weight geometry mismatch")
        if any(w.dtype != torch.bfloat16 for w in (w1, w2)):
            raise ValueError("cuDNN Frost BF16 MoE weights must be BF16")
        if any(v.device != self.device or not v.is_contiguous() for v in (x, w1, w2)):
            raise ValueError(
                "cuDNN Frost BF16 MoE needs contiguous tensors on the runner device"
            )
        if any(v.data_ptr() % 16 for v in (x, w1, w2)):
            raise ValueError("cuDNN Frost BF16 MoE requires 16B-aligned data")
        return t, h, i, e, k, w1, w2

    def accepts(self, act, weights):
        """Per-call auto eligibility, including both artifacts and semantic overrides."""
        major, minor = get_compute_capability(self.device)
        if not large_bf16_moe(self.config, act, major * 10 + minor):
            return False
        try:
            t, h, i, e, k, _, _ = self._validate_pack(act, weights)
        except (KeyError, ValueError, TypeError, NotImplementedError):
            return False
        first, second = _selected_kernels(
            t, h, i, e, k, self.device, self.config.activation
        )
        return bool(first and second)

    def pack_inputs(self, act, weights):
        self._require_built()
        t, h, i, e, k, w1, w2 = self._validate_pack(act, weights)
        first, second = _selected_kernels(
            t, h, i, e, k, self.device, self.config.activation
        )
        if not first or not second:
            raise ValueError("No matching cuDNN Frost FC1/FC2 source kernels")
        key = (
            t,
            h,
            i,
            e,
            k,
            tuple(a.tactic for a in first),
            tuple(b.tactic for b in second),
        )
        if key not in self._plans:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Prepare cuDNN Frost MoE outside CUDA Graph capture"
                    )
                self._plans[key] = _Plans(
                    t, h, i, e, k, self.device, first, second, self._workspace_pool
                )
        output = torch.empty_like(act.hidden_states_q)
        # Exact shapes: a fixed native plan cannot execute a rounded profile.
        # Reuse this call's routing distribution when profiling, instead of random
        # int32 expert ids. The hook is bound to the pack, not mutable runner state.
        ids, scores = act.topk_ids, act.topk_weights

        def preserve_routing(tensors):
            tensors[2], tensors[3] = ids, scores
            return tensors

        # Warm the captured graph and measure sustained replay, as the layer's
        # cross-backend selection does. A single first replay can mis-rank the
        # compound plans even when their steady-state gap is substantial.
        tuning = TuningConfig(
            use_cuda_graph=True,
            cuda_graph_profile_replays=3,
            inputs_pre_hook=preserve_routing,
        )
        return _Inputs(
            [output, act.hidden_states_q, ids, scores, w1, w2], self._plans[key], tuning
        )

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        state = self.launch_state_for(inputs)
        if state is not None:
            return list(state.launches)
        # The autotuner synthesizes a plain tensor list for a profile, so the
        # original pack's launch metadata is not available here. Reapply the
        # same shape shortlist without compiling or building any native plans.
        tokens, hidden = inputs[1].shape
        first, second = _selected_kernels(
            tokens,
            hidden,
            self.config.experts.intermediate_size,
            self.config.routing.num_experts,
            self.config.routing.top_k,
            self.device,
            self.config.activation,
        )
        return [(_TAG, a.tactic, b.tactic) for a, b in product(first, second)]

    def get_cache_key_extras(self, inputs):
        return super().get_cache_key_extras(inputs) + (
            tuple(self.get_valid_tactics(inputs, None)),
        )

    def validate_tactic(self, inputs, tactic):
        return tactic == -1 or tactic in self.get_valid_tactics(inputs, None)

    def forward(
        self, inputs, tactic: Any = -1, do_preparation=False, *, launch_state=None
    ):
        self._require_built()
        state = launch_state or self.launch_state_for(inputs)
        if state is None:
            raise ValueError(
                "Preserve packed cuDNN Frost inputs or pass their launch_state"
            )
        if tactic == -1:
            tactic = next(iter(state.launches))
        if tactic not in state.launches:
            raise ValueError(f"Unknown or stale cuDNN Frost MoE tactic: {tactic!r}")
        state.launches[tactic](*inputs, state.workspace)
        return inputs[0]


def automatic_candidate(config, device):
    """Construct only a semantically supported candidate; no compilation here."""
    runner = CudnnFrostBf16MoeRunner(config, device)
    try:
        runner.check_support()
    except (NotImplementedError, ValueError):
        return None
    runner.build()
    return runner
