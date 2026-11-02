# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Full NVFP4 MoE candidate composed from frozen Frost grouped GEMMs.

The canonical ``cutlass_nvfp4`` weight view is consumed directly. Routing,
segmented scale packing, intermediate NVFP4 quantization, and finalization are
owned by this runner; no other backend executes any part of its pipeline.
"""

from __future__ import annotations

import functools
from itertools import product

import torch

from .....fused_moe.api import QuantFormat, RoutingInputMode
from .....fused_moe.runners import MoERunner, _validate_prerouted_inputs
from .....utils import get_compute_capability
from .. import runtime as common
from ..activations import ACTIVATIONS, activation_name
from ..support import SUPPORTED_ARCHS, weight_view
from .support import is_eligible
from ..compiler import require_moe_assembler
from ..cache import (
    require_graph_resource_retention,
    TensorCache,
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
from ..shortlist import _read, capture_profiles
from ..heuristics import POLICY_VERSION
from ..catalog import can_implement, select_kernels as select_catalog
from ..tuning import prepared_state, ranked_tactics
from . import runtime, fma

_WEIGHT_KEYS = (
    "fc1_expert_weights",
    "fc2_expert_weights",
    "fc1_act_global_scale",
    "fc1_weight_block_scale",
    "fc1_dequant_scale",
    "fc2_act_global_scale",
    "fc2_weight_block_scale",
    "fc2_dequant_scale",
)
_TAG = "cudnn_frost-nvfp4-moe-v3"
_UNSUPPORTED = "cuDNN Frost NVFP4 MoE kernels require SM107a or SM120a"


def _tensor_version(tensor):
    # Only scalar validation uses versions; inference scalars have no counter.
    # Their values must continue to satisfy the validated numerical contract.
    try:
        return tensor._version
    except RuntimeError:
        return None


@functools.lru_cache(maxsize=None)
def _module(arch):
    from .....jit.cudnn_frost import gen_cudnn_frost_moe_module

    return gen_cudnn_frost_moe_module("nvfp4", arch).build_and_load()


def _artifact_roots():
    return _cached_artifact_roots()


@functools.cache
def _cached_artifact_roots():
    return (common.artifact_root("nvfp4"),)


def _kernels(
    rows, hidden, intermediate, experts, device, activation, *, quantized_output=False
):
    return _matching_kernels(
        _artifact_roots(),
        common._arch_for(device),
        rows,
        hidden,
        intermediate,
        experts,
        activation_name(activation),
        quantized_output=quantized_output,
    )


def _matching_kernels(
    roots, arch, rows, hidden, intermediate, experts, name, *, quantized_output=False
):
    first, second = [], []
    for root in roots:
        for kernel in runtime.discover(root, quantized_output=quantized_output):
            n, k = (intermediate, hidden) if kernel.fc1 else (hidden, intermediate)
            dims = dict(s=rows, n=n, k=k, experts=experts, groups=experts)
            if kernel.arch != arch or not can_implement(kernel, dims):
                continue
            if kernel.fc1:
                if kernel.activation == name:
                    first.append(kernel)
            else:
                second.append(kernel)
    return tuple(first), tuple(second)


def _selected_kernels(tokens, hidden, intermediate, experts, topk, device, activation):
    if not 0 < tokens <= 1 << 20:
        return (), ()
    return _selected_kernels_cached(
        _artifact_roots(),
        runtime._artifact_cache_version,
        common._tactic_digest(POLICY_VERSION),
        common._arch_for(device),
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
    # Admission and packing share metadata only; per-call validation stays outside.
    profiles = _read(roots).get((arch, name, experts, hidden, intermediate, topk), {})
    first, second = _matching_kernels(
        roots, arch, tokens * topk, hidden, intermediate, experts, name
    )
    if not first or not second:
        return (), ()
    fused, _ = _matching_kernels(
        roots,
        arch,
        tokens * topk,
        hidden,
        intermediate,
        experts,
        name,
        quantized_output=True,
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


def _fma_tactic(tokens, hidden, intermediate, experts, topk, activation):
    if not fma.supported(
        tokens, hidden, intermediate, experts, topk, activation_name(activation)
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
    normal = seal_kernels(kernel for root in roots for kernel in runtime.discover(root))
    fused = seal_kernels(
        kernel
        for root in roots
        for kernel in runtime.discover(root, quantized_output=True)
    )
    select_fma, validate_fma_sources = capture_fma_selection(fma, name, arch)
    return SelectionSnapshot(
        make_kernel_selector(
            tuple(kernel for kernel in normal if kernel.fc1),
            tuple(kernel for kernel in normal if not kernel.fc1),
            fused,
            capture_profiles(roots),
            arch,
            name,
            sm_count,
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
        swizzled=False,
        fma_tactic=None,
        validate_fma_sources=None,
    ):
        self.plans, self.launches = {}, {}
        first = tuple(common.bind_kernel(k, device) for k in first)
        second = tuple(common.bind_kernel(k, device) for k in second)
        self.first, self.second = first, second
        required = 0
        module = _module(common._arch_for(device))
        for a, b in product(first, second):
            key = (_TAG, a.tactic, b.tactic)
            executor_key = (hidden, intermediate, experts, topk, swizzled, key)
            plan = executor_cache.get(executor_key)
            if plan is None:
                plan = module.make_runner(
                    [common._load_kernel(a, device)],
                    [common._load_kernel(b, device)],
                    hidden,
                    intermediate,
                    experts,
                    topk,
                    device.index,
                    a.workspace_bytes,
                    b.workspace_bytes,
                    a.gated,
                    [runtime._TAIL_SLOTS[name] for name in a.launch_tail],
                    [runtime._TAIL_SLOTS[name] for name in b.launch_tail],
                    a.swap_ab,
                    b.swap_ab,
                    swizzled,
                    False,
                )
                executor_cache[executor_key] = plan
            # Packed states retain Modules whose run Functions borrow them.
            self.plans[key], self.launches[key] = plan, plan["run"]
            required = max(required, plan["workspace_size"](tokens))
        if fma_tactic is not None:
            executor_key = (hidden, intermediate, experts, topk, swizzled, fma_tactic)
            plan = executor_cache.get(executor_key)
            if plan is None:
                if validate_fma_sources is not None:
                    validate_fma_sources()
                variants = [
                    fma.build(
                        t,
                        hidden,
                        intermediate,
                        experts,
                        topk,
                        device,
                        swizzled,
                        fma_tactic[1],
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
                    [],
                    False,
                    False,
                    swizzled,
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


def prepared_stage_inputs(inputs, *, tactic=None):
    """Expose grouped inputs from a prepared call for offline stage measurements.

    FMA plans reject this helper because they never group tokens by expert.
    Executes routing, FC1, and intermediate quantization. Fused FC1 plans do
    not materialize ``fc1_output`` and omit that view from the result. The
    returned views borrow the plan workspace and are overwritten by another
    invocation; copy them before running the complete pipeline. Prepare and call
    this measurement helper outside CUDA Graph capture.
    """
    state = inputs.launch_state
    plan = state.plans[next(iter(state.plans)) if tactic is None else tactic]
    plan["prepare_stages"](*inputs, state.workspace)
    xp, mp, qp, sp1, sp2, op, sf_rows = plan["stage_layout"](inputs[1].shape[0])
    t, packed_h = inputs[1].shape
    e, _, packed_i = inputs[5].shape
    h, i = packed_h * 2, packed_i * 2
    s = t * inputs[2].shape[1]
    raw = state.workspace

    def view(start, count, dtype, shape):
        size = torch.empty((), dtype=dtype).element_size()
        return raw[start : start + count * size].view(dtype).view(shape)

    result = {
        "fc1_tokens": view(xp, s * h // 2, torch.uint8, (s, h // 2)),
        "fc1_token_scales": view(sp1, sf_rows * h // 16, torch.uint8, (-1,)),
        "fc2_tokens": view(qp, s * i // 2, torch.uint8, (s, i // 2)),
        "fc2_token_scales": view(sp2, sf_rows * i // 16, torch.uint8, (-1,)),
        "offsets": view(op, e + 1, torch.int32, (e + 1,)),
    }
    if mp >= 0:
        result["fc1_output"] = view(mp, s * i, torch.bfloat16, (s, i))
    return result


class CudnnFrostNvfp4MoeRunner(FrostRunnerMixin, MoERunner):
    """Measured Tensor Core choices and an independent small-token FMA tactic.

    Weight data and block scales are read directly, including inference tensors.
    Their contents, activations and routing may change in place between graph
    replays, with updates ordered before execution on the replay stream.
    Captured graphs require stable tensor pointers, shapes and layouts. FP32
    GEMM descales are applied before activation (FC1) or BF16 output (FC2).
    The FC2 activation global scale controls intermediate NVFP4 quantization.
    As in CUTLASS, FC1's activation global is already represented by the
    caller's prequantized input and is not reapplied during routing.

    One runner owns one stream's workspace. Distinct shapes and tactics reuse
    it sequentially; concurrent streams must use separate runners or layers.
    """

    backend_key = "cudnn_frost_nvfp4"
    weight_view_aliases = ("cutlass_nvfp4",)
    _input_support = staticmethod(is_eligible)
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = ((QuantFormat.NVFP4, QuantFormat.NVFP4),)
    supported_activation_classes = tuple(ACTIVATIONS.values())
    supports_expert_parallelism = False

    def __init__(self, config, device):
        super().__init__()
        self.config, self.device = config, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._engine = None
        self._plans, self._workspace_pool = {}, {}
        self._validated_scales = TensorCache(maxsize=32)

    def _check_support(self):
        super()._check_support()
        if self.device.type != "cuda":
            raise NotImplementedError(_UNSUPPORTED)
        major, minor = get_compute_capability(self.device)
        arch = f"sm_{major}{minor}a"
        if arch not in SUPPORTED_ARCHS:
            raise NotImplementedError(_UNSUPPORTED)
        name = activation_name(self.config.activation)
        if not self.config.finalize.do_finalize or self.config.quant.per_token_scale:
            raise NotImplementedError(
                "cuDNN Frost NVFP4 MoE requires finalized output without per-token scales"
            )
        sources = tuple(
            sorted(
                {
                    (kernel.source_path, kernel.source_sha256)
                    for root in _artifact_roots()
                    for quantized in (False, True)
                    for kernel in runtime.discover(root, quantized_output=quantized)
                    if kernel.arch == arch
                    and (not kernel.fc1 or kernel.activation == name)
                }
            )
        )
        require_graph_resource_retention()
        # Only the SM107 sources patch tensor maps and need the assembler floor.
        if arch == "sm_107a":
            require_moe_assembler()
        require_compiler(arch, sources)

    def _build(self):
        device, activation = self.device, self.config.activation
        swizzled = self.config.quant.swizzled_scale_factors is True
        self._engine = FrostEngine(
            device,
            token_ceiling=self.config.execution.tune_max_num_tokens,
            artifact_roots=_artifact_roots,
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
                swizzled,
                metadata.fma_tactic,
                metadata.validate_fma_sources,
            ),
        )
        self._plans = self._engine.plans
        self._workspace_pool = self._engine.workspace_pool

    def _check_scale(self, value, *, positive):
        version = _tensor_version(value)
        if self._validated_scales.get(value, version, context=positive) is not None:
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Validate NVFP4 global scales outside CUDA Graph capture"
                )
            valid = torch.isfinite(value)
            if positive:
                valid = valid & (value > 0)
            if not bool(valid.all().item()):
                raise ValueError(
                    "cuDNN Frost NVFP4 requires finite scales and positive quantization globals"
                )
        self._validated_scales.put(value, version, True, context=positive)

    def pack_inputs(self, act, weights):
        self._require_built()
        self._validate_pack_contract(act)
        x, xsf = act.hidden_states_q, act.hidden_states_scale
        if x.ndim != 2 or x.dtype != torch.uint8:
            raise ValueError("cuDNN Frost NVFP4 MoE requires packed uint8 x[T,H/2]")
        t, packed_h = x.shape
        h = packed_h * 2
        k, e, i = (
            self.config.routing.top_k,
            self.config.routing.num_experts,
            self.config.experts.intermediate_size,
        )
        if not (0 < t <= min(1 << 20, self.config.execution.tune_max_num_tokens)):
            raise ValueError("cuDNN Frost NVFP4 token count exceeds supported bounds")
        if not (
            0 < k <= e <= 1024
            and t * k < 2**31 - 128 * e
            and 0 < h <= 1 << 20
            and 0 < i <= 1 << 20
        ):
            raise ValueError("cuDNN Frost NVFP4 requires bounded int32 geometry")
        if h % 128 or i % 128:
            raise ValueError("cuDNN Frost NVFP4 requires H/I divisible by 128")
        _validate_prerouted_inputs(
            act,
            t,
            k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )
        view = weight_view(weights, self.backend_key, "cutlass_nvfp4")
        if set(view) != set(_WEIGHT_KEYS):
            raise ValueError(
                "cuDNN Frost NVFP4 requires plain canonical weights without overrides"
            )
        w1, w2, a1, sf1, alpha1, a2, sf2, alpha2 = (view[name] for name in _WEIGHT_KEYS)
        mult = 2 if self.config.activation.is_gated else 1
        expected = (
            (w1, torch.uint8, (e, mult * i, h // 2)),
            (w2, torch.uint8, (e, h, i // 2)),
            (sf2, torch.uint8, (e, h, i // 16)),
            (alpha1, torch.float32, (e,)),
            (alpha2, torch.float32, (e,)),
        )
        if (
            any(
                v.dtype != dtype or tuple(v.shape) != shape
                for v, dtype, shape in expected
            )
            or sf1.dtype != torch.uint8
            or tuple(sf1.shape)
            not in (
                (e, mult * i, h // 16),
                (mult, e, i, h // 16),
            )
        ):
            raise ValueError(
                "cuDNN Frost NVFP4 canonical weight/scale geometry mismatch"
            )
        if any(
            value.dtype != torch.float32 or tuple(value.shape) not in ((), (1,))
            for value in (a1, a2)
        ):
            raise ValueError("NVFP4 activation global scales must be float32 scalars")
        swizzled = self.config.quant.swizzled_scale_factors is True
        shape = ((t + 127) // 128 * 128 * h // 16,) if swizzled else (t, h // 16)
        dtypes = (torch.uint8,) if swizzled else (torch.uint8, torch.float8_e4m3fn)
        if xsf is None or xsf.dtype not in dtypes or tuple(xsf.shape) != shape:
            raise ValueError("cuDNN Frost NVFP4 activation E4M3 scale layout mismatch")
        tensors = (x, xsf, w1, w2, sf1, sf2)
        if any(
            v.device != self.device or not v.is_contiguous() or v.data_ptr() % 16
            for v in tensors
        ) or any(
            v.device != self.device or not v.is_contiguous() or v.data_ptr() % 4
            for v in (a1, a2, alpha1, alpha2)
        ):
            raise ValueError(
                "cuDNN Frost NVFP4 requires aligned contiguous tensors on the runner device"
            )
        self._check_scale(a1, positive=True)
        self._check_scale(a2, positive=True)
        self._check_scale(alpha1, positive=False)
        self._check_scale(alpha2, positive=False)
        xsf = xsf.view(torch.uint8)
        a1 = a1.reshape(1)
        a2 = a2.reshape(1)
        shape = (t, h, i, e, k)
        state = self._engine.state(shape)
        output = torch.empty((t, h), dtype=torch.bfloat16, device=self.device)
        ids, scores = act.topk_ids, act.topk_weights

        return _Inputs(
            [
                output,
                x,
                ids,
                scores,
                w1,
                w2,
                sf1,
                sf2,
                xsf,
                a1,
                alpha1,
                a2,
                alpha2,
            ],
            state,
        )

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        state = prepared_state(self, inputs, hidden_multiplier=2)
        if state is not None:
            return ranked_tactics(state, inputs, _TAG)
        tokens, packed_hidden = inputs[1].shape
        hidden = 2 * packed_hidden
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
        state = prepared_state(self, inputs, hidden_multiplier=2)
        if state is not None:
            key, tactics = state.preparation_key, tuple(state.launches)
        else:
            t, packed_h = inputs[1].shape
            h = packed_h * 2
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
        state = prepared_state(self, inputs, hidden_multiplier=2)
        return tactic == -1 or tactic in (
            state.launches
            if state is not None
            else self.get_valid_tactics(inputs, None)
        )
