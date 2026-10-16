"""Independent BF16 MoE engine built from cuDNN Frost FC1 and FC2 source kernels.

The existing CUTLASS *weight layout* is canonical row-major [up, gate]/down;
we consume that view without invoking or modifying the CUTLASS runner.
Each instance owns its workspace and must be confined to one thread/stream.
"""

from __future__ import annotations

import functools
from itertools import product

import torch

from .....fused_moe.api import QuantFormat, RoutingInputMode, SwiGLU
from .....fused_moe.runners import MoERunner, _validate_prerouted_inputs
from .....utils import get_compute_capability
from . import fc2, fma, runtime
from .support import large_bf16_moe
from ..cache import (
    require_graph_resource_retention,
)
from ..engine import (
    FrostEngine,
    FrostRunnerMixin,
    PackedInputs as _Inputs,
    allocate_workspace,
    SelectionSnapshot,
    capture_fma_selection,
    make_kernel_selector,
    seal_kernels,
)
from ..capabilities import require_compiler
from ..activations import ACTIVATIONS, activation_name
from ..support import SUPPORTED_ARCHS, weight_view
from ..compiler import require_moe_assembler
from ..shortlist import capture_profiles
from ..shortlist import _read
from ..catalog import can_implement, select_kernels as select_catalog
from ..heuristics import POLICY_VERSION
from ..tuning import prepared_state, ranked_tactics

_WEIGHT_KEYS = frozenset(("fc1_expert_weights", "fc2_expert_weights"))
_TAG = "cudnn_frost-bf16-moe-v2"
_UNSUPPORTED = "cuDNN Frost BF16 MoE kernels require SM107a or SM120a"


@functools.lru_cache(maxsize=None)
def _module(arch):
    from .....jit.cudnn_frost import gen_cudnn_frost_moe_module

    return gen_cudnn_frost_moe_module("bf16", arch).build_and_load()


def _kernels(rows, hidden, intermediate, experts, device, activation=None):
    name = activation_name(SwiGLU() if activation is None else activation)
    return _matching_kernels(
        runtime._artifact_roots(),
        runtime._arch_for(device),
        rows,
        hidden,
        intermediate,
        experts,
        name,
    )


def _matching_kernels(roots, arch, rows, hidden, intermediate, experts, name):
    values = dict(s=rows, n=intermediate, k=hidden, experts=experts, groups=experts)
    first = tuple(
        k
        for k in runtime._discover(roots)
        if k.arch == arch and k.activation == name and can_implement(k, values)
    )
    values = dict(s=rows, n=hidden, k=intermediate, experts=experts, groups=experts)
    second = tuple(
        k for k in fc2.discover(roots) if k.arch == arch and can_implement(k, values)
    )
    return first, second


def _selected_kernels(tokens, hidden, intermediate, experts, topk, device, activation):
    if not 0 < tokens <= 1 << 20:
        return (), ()
    return _selected_kernels_cached(
        runtime._artifact_roots(),
        runtime._artifact_cache_version,
        runtime._tactic_digest(POLICY_VERSION),
        runtime._arch_for(device),
        torch.cuda.get_device_properties(device).multi_processor_count,
        tokens,
        hidden,
        intermediate,
        experts,
        topk,
        activation_name(activation),
    )


@functools.lru_cache(maxsize=128)
def _selected_kernels_cached(
    roots,
    artifact_version,
    compiler_key,
    arch,
    sm_count,
    tokens,
    hidden,
    intermediate,
    experts,
    topk,
    name,
):
    # Admission and packing share sealed artifact metadata, never runtime tensors
    # or pointers. Explicit artifact refresh advances the version in this key.
    first, second = _matching_kernels(
        roots, arch, tokens * topk, hidden, intermediate, experts, name
    )
    profiles = _read(roots).get((arch, name, experts, hidden, intermediate, topk), {})
    return select_catalog(
        first,
        second,
        (),
        profiles,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        sm_count=sm_count,
    )


def _fma_tactic(tokens, hidden, intermediate, experts, topk, activation, device):
    # The FMA kernels are specialized and validated for SM107a only.
    if (
        not fma.supported(
            tokens, hidden, intermediate, experts, topk, activation_name(activation)
        )
        or runtime._arch_for(device) != "sm_107a"
    ):
        return None
    try:
        fma.check_support(activation_name(activation))
    except NotImplementedError:
        # An older DSL can still run the existing Tensor Core candidates.
        return None
    return fma.tactic(activation_name(activation))


def _capture_selection(roots, arch, sm_count, activation):
    name = activation_name(activation)
    first = seal_kernels(runtime._discover(roots))
    second = seal_kernels(fc2.discover(roots))
    select_fma, validate_fma_sources = capture_fma_selection(fma, name, arch)
    return SelectionSnapshot(
        make_kernel_selector(
            first, second, (), capture_profiles(roots), arch, name, sm_count
        ),
        select_fma,
        validate_fma_sources,
    )


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
        executor_cache,
        fma_tactic=None,
        validate_fma_sources=None,
    ):
        first = tuple(runtime.bind_kernel(k, device) for k in first)
        second = tuple(runtime.bind_kernel(k, device) for k in second)
        self.first, self.second = first, second
        self.plans = {}
        self.launches = {}
        required = 0
        module = _module(runtime._arch_for(device))
        for a, b in product(first, second):
            key = (_TAG, a.tactic, b.tactic)
            executor_key = (hidden, intermediate, experts, topk, False, key)
            plan = executor_cache.get(executor_key)
            if plan is None:
                plan = module.make_runner(
                    [runtime._load_kernel(a, device)],
                    [runtime._load_kernel(b, device)],
                    hidden,
                    intermediate,
                    experts,
                    topk,
                    device.index,
                    a.workspace_bytes,
                    b.workspace_bytes,
                    a.gated,
                    [
                        {"output": -1, "scale": 0, "gate_scale": 1, "linear_scale": 2}[
                            name
                        ]
                        for name in a.launch_tail
                    ],
                    a.swap_ab,
                    b.swap_ab,
                    False,
                )
                executor_cache[executor_key] = plan
            # Packed states retain Modules whose run Functions borrow them.
            self.plans[key] = plan
            self.launches[key] = plan["run"]
            required = max(required, plan["workspace_size"](tokens))
        if fma_tactic is not None:
            executor_key = (hidden, intermediate, experts, topk, False, fma_tactic)
            plan = executor_cache.get(executor_key)
            if plan is None:
                if validate_fma_sources is not None:
                    validate_fma_sources()
                variants = [
                    fma.build(
                        t, hidden, intermediate, experts, topk, device, fma_tactic[1]
                    )
                    for t in range(1, 5)
                    if fma.supported(
                        t, hidden, intermediate, experts, topk, fma_tactic[1]
                    )
                ]
                plan = module.make_runner(
                    [pair[0] for pair in variants],
                    [pair[1] for pair in variants],
                    hidden,
                    intermediate,
                    experts,
                    topk,
                    device.index,
                    0,
                    0,
                    first[0].gated,
                    [],
                    False,
                    False,
                    True,
                )
                if validate_fma_sources is not None:
                    validate_fma_sources()
                executor_cache[executor_key] = plan
            self.plans[fma_tactic], self.launches[fma_tactic] = plan, plan["run"]
            required = max(required, plan["workspace_size"](tokens))
        # A bucket owns capacity storage; native executors use the call's actual
        # token count. Older packed calls and graphs retain their allocation.
        self.workspace = allocate_workspace(required, device, workspace_pool)


class CudnnFrostBf16MoeRunner(FrostRunnerMixin, MoERunner):
    """A full MoE candidate, not a GEMM1 override in another backend."""

    backend_key = "cudnn_frost_bf16"
    weight_view_aliases = ("cutlass_bf16",)
    _input_support = staticmethod(large_bf16_moe)
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
        self._engine = None
        self._plans, self._workspace_pool = {}, {}

    def _check_support(self):
        super()._check_support()
        if self.device.type != "cuda":
            raise NotImplementedError(_UNSUPPORTED)
        major, minor = get_compute_capability(self.device)
        if f"sm_{major}{minor}a" not in SUPPORTED_ARCHS:
            raise NotImplementedError(_UNSUPPORTED)
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
        require_graph_resource_retention()
        if arch == "sm_107a":
            require_moe_assembler()
        require_compiler(arch, sources)

    def _build(self):
        device, activation = self.device, self.config.activation
        self._engine = FrostEngine(
            device,
            token_ceiling=self.config.execution.tune_max_num_tokens,
            artifact_roots=runtime._artifact_roots,
            artifact_version=lambda: runtime._artifact_cache_version,
            policy_version=POLICY_VERSION,
            tactic_tag=_TAG,
            require_assembler=get_compute_capability(device) == (10, 7),
            refresh_artifacts=runtime.clear_artifact_cache,
            capture_selection=lambda roots, arch, sm_count: _capture_selection(
                roots, arch, sm_count, activation
            ),
            build_plan=lambda shape, metadata, workspace_pool, executor_cache: _Plans(
                *shape,
                device,
                metadata.first,
                metadata.second,
                workspace_pool,
                executor_cache,
                metadata.fma_tactic,
                metadata.validate_fma_sources,
            ),
        )
        self._plans = self._engine.plans
        self._workspace_pool = self._engine.workspace_pool

    def pack_inputs(self, act, weights):
        self._require_built()
        self._validate_pack_contract(act)
        x = act.hidden_states_q
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise ValueError("cuDNN Frost BF16 MoE requires BF16 x[T,H]")
        if act.hidden_states_scale is not None:
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
        view = weight_view(weights, self.backend_key, "cutlass_bf16")
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
        shape = (t, h, i, e, k)
        state = self._engine.state(shape)
        output = torch.empty_like(x)
        ids, scores = act.topk_ids, act.topk_weights
        return _Inputs([output, x, ids, scores, w1, w2], state)

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        state = prepared_state(self, inputs)
        if state is not None:
            return ranked_tactics(state, inputs, _TAG)
        # The autotuner synthesizes a plain tensor list for a profile, so the
        # original pack's launch metadata is not available here. Reapply the
        # same shape shortlist without compiling or building any native plans.
        tokens, hidden = inputs[1].shape
        return self._engine.valid_tactics(
            inputs,
            shape=(
                tokens,
                hidden,
                self.config.experts.intermediate_size,
                self.config.routing.num_experts,
                self.config.routing.top_k,
            ),
        )

    def _input_cache_key_extras(self, inputs):
        # Profiling uses plain lists; keep its key tied to the full prepared
        # launch pool, just like PackedInputs, rather than the ranked shortlist.
        state = prepared_state(self, inputs)
        if state is not None:
            key, tactics = state.preparation_key, tuple(state.launches)
        else:
            t, h = inputs[1].shape
            metadata = self._engine.prepare(
                (
                    t,
                    h,
                    self.config.experts.intermediate_size,
                    self.config.routing.num_experts,
                    self.config.routing.top_k,
                )
            )
            key, tactics = (
                metadata.plan_key,
                tuple(self.get_valid_tactics(inputs, None)),
            )
        return (POLICY_VERSION, key, tactics)

    def validate_tactic(self, inputs, tactic):
        state = prepared_state(self, inputs)
        return tactic == -1 or tactic in (
            state.launches
            if state is not None
            else self.get_valid_tactics(inputs, None)
        )
