# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Full MXFP8 MoE candidate composed from frozen Frost grouped GEMMs.

The canonical ``cutlass_mxfp8`` weight view is consumed directly. Routing,
segmented scale packing, intermediate MXFP8 quantization, and finalization are
owned by this runner; no other backend executes any part of its pipeline.
"""

from __future__ import annotations

import functools
from itertools import product
from typing import Any, NamedTuple

import torch

from .....autotuner import TuningConfig
from .....fused_moe.api import QuantFormat, RoutingInputMode
from .....fused_moe.runners import MoERunner, _validate_prerouted_inputs
from .....utils import get_compute_capability
from .. import runtime as common
from ..activations import ACTIVATIONS, activation_name
from ..support import weight_view
from ..compiler import require_moe_assembler
from ..cache import (
    LRUCache,
    call_preparation,
    reuse_validation,
    require_graph_resource_retention,
    retain_graph_resources,
    TensorCache,
)
from ..capabilities import require_compiler
from ..shortlist import _read, select
from ..heuristics import POLICY_VERSION, select_stages
from ..tuning import prepared_state, ranked_tactics
from . import runtime, fma
from .support import is_eligible

_WEIGHT_KEYS = (
    "fc1_expert_weights",
    "fc2_expert_weights",
    "fc1_expert_scales",
    "fc2_expert_scales",
    "fc1_input_scale",
    "fc2_input_scale",
)
_TAG = "cudnn_frost-mxfp8-moe-v2"


def _tensor_version(tensor):
    # Inference tensors deliberately have no version counter. Like other
    # prepared-weight caches, those tensors are immutable for the runner's
    # lifetime; ordinary tensors invalidate preparation on in-place writes.
    try:
        return tensor._version
    except RuntimeError:
        return None


@functools.lru_cache(maxsize=1)
def _module(arch):
    from .....jit.cudnn_frost import gen_cudnn_frost_moe_module

    return gen_cudnn_frost_moe_module("mxfp8", arch).build_and_load()


def _artifact_roots():
    return _cached_artifact_roots()


@functools.cache
def _cached_artifact_roots():
    return (common.artifact_root("mxfp8"),)


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
            if kernel.arch != arch or not all(
                common._dimension_matches(v, kernel.contract.get(d))
                for d, v in dims.items()
            ):
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
    if not profiles or tokens > max(profiles):
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
        return select_stages(
            first,
            second,
            fused,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            sm_count=sm_count,
        )
    bucket = min((n for n in profiles if n >= tokens), default=max(profiles))
    if any(len(ids) != 2 for ids in profiles[bucket]):
        return (), ()
    first, second = _matching_kernels(
        roots, arch, tokens * topk, hidden, intermediate, experts, name
    )
    first, second = select(
        roots, arch, name, tokens, hidden, intermediate, experts, topk, first, second
    )
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
    tiles = {kernel.tactic_metadata["tile"] for kernel in first}
    # Retain the measured unfused choices: fusion can trade mainloop overlap
    # for fewer launches. Include available data-store modes in full-pipeline tuning;
    # the fused scale output uses STG in both variants.
    variants = tuple(
        kernel for kernel in fused if kernel.tactic_metadata["tile"] in tiles
    )
    return first + variants, second


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


class _Inputs(list):
    def __init__(self, tensors, state, tuning_config):
        super().__init__(tensors)
        self.launch_state = state
        self.tuning_config = tuning_config


class _PreparationMetadata(NamedTuple):
    first: tuple
    second: tuple
    fma_tactic: Any
    plan_key: tuple


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
        swizzled=False,
        fma_tactic=None,
    ):
        self.plans, self.launches = {}, {}
        self.first, self.second = first, second
        required = 0
        module = _module(common._arch_for(device))
        for a, b in product(first, second):
            key = (_TAG, a.tactic, b.tactic)
            plan = module.make_plan(
                common._load_kernel(a, device),
                common._load_kernel(b, device),
                tokens,
                hidden,
                intermediate,
                experts,
                topk,
                device.index,
                a.workspace_bytes,
                b.workspace_bytes,
                a.gated,
                [runtime._TAIL_SLOTS[name] for name in a.launch_tail],
                a.swap_ab,
                b.swap_ab,
                swizzled,
                False,
            )
            self.plans[key], self.launches[key] = plan, plan["run"]
            required = max(required, plan["workspace_size"]())
        if fma_tactic is not None:
            fc1, fc2 = fma.build(
                tokens,
                hidden,
                intermediate,
                experts,
                topk,
                device,
                swizzled,
                fma_tactic[1],
            )
            plan = module.make_plan(
                fc1,
                fc2,
                tokens,
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
                swizzled,
                True,
            )
            self.plans[fma_tactic], self.launches[fma_tactic] = plan, plan["run"]
            required = max(required, plan["workspace_size"]())
        capacity = min((size for size in workspace_pool if size >= required), default=0)
        if not capacity:
            capacity = 1 << (max(required, 1) - 1).bit_length()
            workspace_pool[capacity] = torch.empty(
                capacity, dtype=torch.uint8, device=device
            )
        self.workspace = workspace_pool[capacity][:required]


def prepared_stage_inputs(inputs, *, tactic=None):
    """Expose grouped inputs from a prepared call for offline stage measurements.

    Executes routing, FC1, and intermediate quantization. Fused plans omit
    the BF16 ``fc1_output`` view. The
    returned views borrow the plan workspace and are overwritten by another
    invocation; copy them before running the complete pipeline. Prepare and call
    this measurement helper outside CUDA Graph capture. FMA plans reject this
    helper because their intermediate tensors remain in token/slot order.
    """
    state = inputs.launch_state
    plan = state.plans[next(iter(state.plans)) if tactic is None else tactic]
    plan["prepare_stages"](*inputs, state.workspace)
    xp, mp, qp, sp1, sp2, op, sf_rows = plan["stage_layout"]()
    t, h = inputs[1].shape
    e, _, i = inputs[5].shape
    s = t * inputs[2].shape[1]
    raw = state.workspace

    def view(start, count, dtype, shape):
        size = torch.empty((), dtype=dtype).element_size()
        return raw[start : start + count * size].view(dtype).view(shape)

    result = {
        "fc1_tokens": view(xp, s * h, torch.float8_e4m3fn, (s, h)),
        "fc1_token_scales": view(sp1, sf_rows * h // 32, torch.uint8, (-1,)),
        "fc2_tokens": view(qp, s * i, torch.float8_e4m3fn, (s, i)),
        "fc2_token_scales": view(sp2, sf_rows * i // 32, torch.uint8, (-1,)),
        "offsets": view(op, e + 1, torch.int32, (e + 1,)),
    }

    if mp >= 0:
        result["fc1_output"] = view(mp, s * i, torch.bfloat16, (s, i))
    return result


class CudnnFrostMxfp8MoeRunner(MoERunner):
    """Measured FC1/FC2 choices and their fused FC1 quantization variants.

    Prepared weight scales are static during CUDA Graph replay. Updating an
    ordinary scale tensor and calling ``pack_inputs`` again refreshes its
    packed view; recapture graphs after such updates. Weights created inside
    ``torch.inference_mode`` have no version counter and must remain immutable;
    replace their tensors to prepare new values. Activations, activation scales,
    routing IDs and scores may change in place between graph replays.

    One runner owns one stream's workspace. Distinct shapes and tactics reuse
    it sequentially; concurrent streams must use separate runners or layers.
    """

    requires_exact_shape = True
    uses_call_preparation = True
    backend_key = "cudnn_frost_mxfp8"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = ((QuantFormat.MXFP8, QuantFormat.MXFP8),)
    supported_activation_classes = tuple(ACTIVATIONS.values())
    supports_expert_parallelism = False

    def __init__(self, config, device):
        super().__init__()
        self.config, self.device = config, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._plans, self._workspace_pool = LRUCache(), {}
        self._preparation_metadata = LRUCache(maxsize=128)
        self._identity_scales = TensorCache(maxsize=32)
        self._fc1_scale_views = TensorCache()

    def _check_support(self):
        super()._check_support()
        if self.device.type != "cuda" or get_compute_capability(self.device) != (10, 7):
            raise NotImplementedError("cuDNN Frost MXFP8 MoE kernels require SM107a")
        name = activation_name(self.config.activation)
        if not self.config.finalize.do_finalize or self.config.quant.per_token_scale:
            raise NotImplementedError(
                "cuDNN Frost MXFP8 MoE requires finalized output without per-token scales"
            )
        sources = tuple(
            sorted(
                {
                    (kernel.source_path, kernel.source_sha256)
                    for root in _artifact_roots()
                    for quantized in (False, True)
                    for kernel in runtime.discover(root, quantized_output=quantized)
                    if kernel.arch == "sm_107a"
                    and (not kernel.fc1 or kernel.activation == name)
                }
            )
        )
        require_graph_resource_retention()
        require_moe_assembler()
        require_compiler("sm_107a", sources)

    def _build(self):
        pass

    def _check_identity_scale(self, value):
        # Canonical preparation creates static identity global scales. The
        # frozen FC1 applies activation internally, so arbitrary global GEMM
        # multipliers cannot be silently substituted by an output multiplier.
        version = _tensor_version(value)
        if self._identity_scales.get(value, version) is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Validate MXFP8 global scales outside CUDA Graph capture"
            )
        if not bool(torch.all(value == 1).item()):
            raise ValueError(
                "cuDNN Frost MXFP8 requires identity per-expert global scales"
            )
        self._identity_scales.put(value, version, True)

    @reuse_validation
    def _validate_pack(self, act, weights):
        require_moe_assembler()
        self._validate_pack_contract(act)
        x, xsf = act.hidden_states_q, act.hidden_states_scale
        if x.ndim != 2 or x.dtype != torch.float8_e4m3fn:
            raise ValueError("cuDNN Frost MXFP8 MoE requires E4M3 x[T,H]")
        t, h = x.shape
        k, e, i = (
            self.config.routing.top_k,
            self.config.routing.num_experts,
            self.config.experts.intermediate_size,
        )
        if not (0 < t <= min(1 << 20, self.config.execution.tune_max_num_tokens)):
            raise ValueError("cuDNN Frost MXFP8 token count exceeds supported bounds")
        if not (
            0 < k <= e <= 1024
            and t * k < 2**31 - 128 * e
            and 0 < h <= 1 << 20
            and 0 < i <= 1 << 20
        ):
            raise ValueError("cuDNN Frost MXFP8 requires bounded int32 geometry")
        if h % 128 or i % 128:
            raise ValueError("cuDNN Frost MXFP8 requires H/I divisible by 128")
        _validate_prerouted_inputs(
            act,
            t,
            k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )
        view = weight_view(weights, self.backend_key, "cutlass_mxfp8")
        if set(view) != set(_WEIGHT_KEYS):
            raise ValueError(
                "cuDNN Frost MXFP8 requires plain canonical weights without overrides"
            )
        w1, w2, sf1, sf2, a1, a2 = (view[name] for name in _WEIGHT_KEYS)
        mult = 2 if self.config.activation.is_gated else 1
        expected = (
            (w1, torch.float8_e4m3fn, (e, mult * i, h)),
            (w2, torch.float8_e4m3fn, (e, h, i)),
            (sf1, torch.int32, (e, mult * i, h // 128)),
            (sf2, torch.int32, (e, h, i // 128)),
            (a1, torch.float32, (e,)),
            (a2, torch.float32, (e,)),
        )
        if any(
            v.dtype != dtype or tuple(v.shape) != shape for v, dtype, shape in expected
        ):
            raise ValueError(
                "cuDNN Frost MXFP8 canonical weight/scale geometry mismatch"
            )
        swizzled = self.config.quant.swizzled_scale_factors is True
        shape = ((t + 127) // 128 * 128 * h // 32,) if swizzled else (t, h // 32)
        dtypes = (torch.uint8,) if swizzled else (torch.uint8, torch.float8_e4m3fn)
        if xsf is None or xsf.dtype not in dtypes or tuple(xsf.shape) != shape:
            raise ValueError("cuDNN Frost MXFP8 activation E8M0 scale layout mismatch")
        tensors = (x, xsf, w1, w2, sf1, sf2, a1, a2)
        if any(
            v.device != self.device or not v.is_contiguous() or v.data_ptr() % 16
            for v in tensors
        ):
            raise ValueError(
                "cuDNN Frost MXFP8 requires aligned contiguous tensors on the runner device"
            )
        with torch.cuda.device(self.device):
            self._check_identity_scale(a1)
            self._check_identity_scale(a2)
        return t, h, i, e, k, w1, w2, sf1, sf2, xsf.view(torch.uint8)

    def _prepare_metadata(self, t, h, i, e, k):
        # Cache sealed selection metadata, never this call's tensors or pointers.
        # Compiler and artifact changes must qualify native-plan lookups too.
        selection_key = (
            _artifact_roots(),
            runtime._artifact_cache_version,
            common._tactic_digest(POLICY_VERSION),
            common._arch_for(self.device),
            torch.cuda.get_device_properties(self.device).multi_processor_count,
            t,
            h,
            i,
            e,
            k,
            activation_name(self.config.activation),
        )
        metadata = self._preparation_metadata.get(selection_key)
        if metadata is None:
            first, second = _selected_kernels(
                t, h, i, e, k, self.device, self.config.activation
            )
            fma_tactic = (
                _fma_tactic(t, h, i, e, k, self.config.activation)
                if first and second
                else None
            )
            plan_key = (
                selection_key,
                tuple(a.tactic for a in first),
                tuple(b.tactic for b in second),
                fma_tactic,
            )
            metadata = _PreparationMetadata(first, second, fma_tactic, plan_key)
            self._preparation_metadata[selection_key] = metadata
        return metadata

    @call_preparation
    def accepts(self, act, weights):
        major, minor = get_compute_capability(self.device)
        if not is_eligible(self.config, act, major * 10 + minor):
            return False
        try:
            t, h, i, e, k, *_ = self._validate_pack(act, weights)
        except (KeyError, ValueError, TypeError, NotImplementedError):
            return False
        first, second = _selected_kernels(
            t, h, i, e, k, self.device, self.config.activation
        )
        return bool(first and second)

    @call_preparation
    def pack_inputs(self, act, weights):
        self._require_built()
        t, h, i, e, k, w1, w2, sf1, sf2, xsf = self._validate_pack(act, weights)
        mult = 2 if self.config.activation.is_gated else 1
        version = _tensor_version(sf1)
        cached = self._fc1_scale_views.get(sf1, version)
        if cached is None:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Prepare MXFP8 weight scales outside CUDA Graph capture"
                    )
                # Frozen SF descriptors have contiguous expert strides. Data
                # descriptors accept canonical interleaved up/gate weights.
                packed_sf1 = (
                    sf1.reshape(e, mult, i, h // 128).transpose(0, 1).contiguous()
                )
            cached = packed_sf1
            self._fc1_scale_views.put(sf1, version, cached)
        sf1 = cached
        first, second, fma_tactic, key = self._prepare_metadata(t, h, i, e, k)
        if not first or not second:
            raise ValueError("No legal cuDNN Frost stage candidates for this problem")
        if key not in self._plans:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Prepare cuDNN Frost MoE outside CUDA Graph capture"
                    )
                self._plans[key] = _Plans(
                    t,
                    h,
                    i,
                    e,
                    k,
                    self.device,
                    first,
                    second,
                    self._workspace_pool,
                    self.config.quant.swizzled_scale_factors is True,
                    fma_tactic,
                )
        output = torch.empty((t, h), dtype=torch.bfloat16, device=self.device)
        ids, scores = act.topk_ids, act.topk_weights

        def preserve_routing(tensors):
            # Autotuning must keep valid MXFP8 bytes/scales and the real route
            # distribution; independently random scale bytes may encode NaN.
            tensors[1:] = [act.hidden_states_q, ids, scores, w1, w2, sf1, sf2, xsf]
            return tensors

        tuning = TuningConfig(
            use_cuda_graph=True,
            cuda_graph_profile_replays=3,
            inputs_pre_hook=preserve_routing,
        )
        return _Inputs(
            [output, act.hidden_states_q, ids, scores, w1, w2, sf1, sf2, xsf],
            self._plans[key],
            tuning,
        )

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        state = prepared_state(self, inputs)
        if state is not None:
            return ranked_tactics(state, inputs, _TAG)
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
        tactics = [(_TAG, a.tactic, b.tactic) for a, b in product(first, second)]
        fma_tactic = _fma_tactic(
            tokens,
            hidden,
            self.config.experts.intermediate_size,
            self.config.routing.num_experts,
            self.config.routing.top_k,
            self.config.activation,
        )
        if fma_tactic is not None:
            tactics.append(fma_tactic)
        return tactics

    def _input_cache_key_extras(self, inputs):
        state = prepared_state(self, inputs)
        return (
            POLICY_VERSION,
            (
                tuple(state.launches)
                if state is not None
                else tuple(self.get_valid_tactics(inputs, None))
            ),
        )

    def validate_tactic(self, inputs, tactic):
        state = prepared_state(self, inputs)
        return tactic == -1 or tactic in (
            state.launches
            if state is not None
            else self.get_valid_tactics(inputs, None)
        )

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
        state = launch_state or self.launch_state_for(inputs)
        if state is None:
            raise ValueError(
                "Preserve packed cuDNN Frost inputs or pass their launch_state"
            )
        if tactic == -1:
            tactic = next(iter(state.launches))
        if tactic not in state.launches:
            raise ValueError(f"Unknown or stale cuDNN Frost MoE tactic: {tactic!r}")
        retain_graph_resources(state, inputs)
        state.launches[tactic](*inputs, state.workspace)
        return inputs[0]


def automatic_candidate(config, device):
    runner = CudnnFrostMxfp8MoeRunner(config, device)
    try:
        runner.check_support()
    except (NotImplementedError, ValueError):
        return None
    runner.build()
    return runner
