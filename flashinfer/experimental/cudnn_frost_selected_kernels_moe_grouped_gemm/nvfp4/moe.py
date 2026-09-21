# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Full NVFP4 MoE candidate composed from frozen Frost grouped GEMMs.

The canonical ``cutlass_nvfp4`` weight view is consumed directly. Routing,
segmented scale packing, intermediate NVFP4 quantization, and finalization are
owned by this runner; no other backend executes any part of its pipeline.
"""

from __future__ import annotations

import functools
from itertools import product
from pathlib import Path
from typing import Any

import torch

from ....autotuner import TuningConfig
from ....fused_moe.api import QuantFormat, RoutingInputMode
from ....fused_moe.runners import MoERunner, _validate_prerouted_inputs
from ....utils import get_compute_capability
from .. import runtime as common
from ..activations import ACTIVATIONS, activation_name
from ..capabilities import require_compiler
from ..shortlist import _read, select
from . import runtime
from .support import is_eligible

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
_TAG = "cudnn_frost-nvfp4-moe-v1"


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
    from ....jit.core import gen_jit_spec, sm107a_nvcc_flags

    if arch != "sm_107a":
        raise ValueError("cuDNN Frost NVFP4 MoE kernels require SM107a")
    return gen_jit_spec(
        f"cudnn_frost_nvfp4_moe_v1_{arch}",
        [Path(__file__).parent.parent / "csrc" / "moe_nvfp4.cu"],
        extra_cuda_cflags=sm107a_nvcc_flags,
    ).build_and_load()


def _artifact_roots():
    return (common.artifact_root("nvfp4"),)


def _kernels(rows, hidden, intermediate, experts, device, activation):
    name = activation_name(activation)
    arch = common._arch_for(device)
    first, second = [], []
    for root in _artifact_roots():
        for kernel in runtime.discover(root):
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
    roots, arch, name = (
        _artifact_roots(),
        common._arch_for(device),
        activation_name(activation),
    )
    profiles = _read(roots).get((arch, name, experts, hidden, intermediate, topk), {})
    # Reuse the common next-measured-token bucket, while refusing the explicit
    # legacy path that returns an unfiltered pool when no profile table exists.
    if not profiles or not 0 < tokens <= 12288:
        return (), ()
    bucket = min((n for n in profiles if n >= tokens), default=max(profiles))
    if any(len(ids) != 2 for ids in profiles[bucket]):
        return (), ()
    first, second = _kernels(
        tokens * topk, hidden, intermediate, experts, device, activation
    )
    return select(
        roots, arch, name, tokens, hidden, intermediate, experts, topk, first, second
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
        swizzled=False,
    ):
        self.plans, self.launches = {}, {}
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
                [runtime._TAIL_SLOTS[name] for name in b.launch_tail],
                a.swap_ab,
                b.swap_ab,
                swizzled,
            )
            self.plans[key], self.launches[key] = plan, plan["run"]
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

    Executes routing, FC1, and the exact native intermediate quantizer. The
    returned views borrow the plan workspace and are overwritten by another
    invocation; copy them before running the complete pipeline. Prepare and call
    this measurement helper outside CUDA Graph capture.
    """
    state = inputs.launch_state
    plan = state.plans[next(iter(state.plans)) if tactic is None else tactic]
    plan["prepare_stages"](*inputs, state.workspace)
    xp, mp, qp, sp1, sp2, op, sf_rows = plan["stage_layout"]()
    t, packed_h = inputs[1].shape
    e, _, packed_i = inputs[5].shape
    h, i = packed_h * 2, packed_i * 2
    s = t * inputs[2].shape[1]
    raw = state.workspace

    def view(start, count, dtype, shape):
        size = torch.empty((), dtype=dtype).element_size()
        return raw[start : start + count * size].view(dtype).view(shape)

    return {
        "fc1_tokens": view(xp, s * h // 2, torch.uint8, (s, h // 2)),
        "fc1_token_scales": view(sp1, sf_rows * h // 16, torch.uint8, (-1,)),
        "fc1_output": view(mp, s * i, torch.bfloat16, (s, i)),
        "fc2_tokens": view(qp, s * i // 2, torch.uint8, (s, i // 2)),
        "fc2_token_scales": view(sp2, sf_rows * i // 16, torch.uint8, (-1,)),
        "offsets": view(op, e, torch.int32, (e,)),
    }


class CudnnFrostNvfp4MoeRunner(MoERunner):
    """Four measured FC1/FC2 combinations per supported problem-size bucket.

    Prepared weight scales are static during CUDA Graph replay. Updating an
    ordinary block-scale tensor and calling ``pack_inputs`` again refreshes its
    packed view; recapture graphs after such updates. Weights created inside
    ``torch.inference_mode`` have no version counter and must remain immutable;
    replace their tensors to prepare new values. Activations, activation scales,
    routing IDs and scores may change in place between graph replays. FP32
    GEMM descales are applied before activation (FC1) or BF16 output (FC2).
    The FC2 activation global scale controls intermediate NVFP4 quantization.
    As in CUTLASS, FC1's activation global is already represented by the
    caller's prequantized input and is not reapplied during routing.

    One runner owns one stream's workspace. Distinct shapes and tactics reuse
    it sequentially; concurrent streams must use separate runners or layers.
    """

    backend_key = "cudnn_frost_nvfp4"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = ((QuantFormat.NVFP4, QuantFormat.NVFP4),)
    supported_activation_classes = tuple(ACTIVATIONS.values())
    supports_expert_parallelism = False

    def __init__(self, config, device):
        super().__init__()
        self.config, self.device = config, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._plans, self._workspace_pool, self._validated_scales = {}, {}, {}
        self._fc1_scale_views = {}

    def _check_support(self):
        super()._check_support()
        if self.device.type != "cuda" or get_compute_capability(self.device) != (10, 7):
            raise NotImplementedError("cuDNN Frost NVFP4 MoE kernels require SM107a")
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
                    for kernel in runtime.discover(root)
                    if kernel.arch == "sm_107a"
                    and (not kernel.fc1 or kernel.activation == name)
                }
            )
        )
        require_compiler("sm_107a", sources)

    def _build(self):
        pass

    def _check_scale(self, value, *, positive):
        key = (id(value), positive)
        version = _tensor_version(value)
        cached = self._validated_scales.get(key)
        if cached is not None and cached[0] is value and cached[1] == version:
            return
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
        self._validated_scales[key] = (value, version)

    def _validate_pack(self, act, weights):
        if act.routing_input_mode not in self.supported_routing_modes:
            raise NotImplementedError(
                "cuDNN Frost NVFP4 MoE requires precomputed routing"
            )
        x, xsf = act.hidden_states_q, act.hidden_states_scale
        if x.ndim != 2 or x.dtype != torch.uint8:
            raise ValueError("cuDNN Frost NVFP4 MoE requires packed uint8 x[T,H/2]")
        if act.per_token_scale is not None:
            raise ValueError("cuDNN Frost NVFP4 does not consume per-token scales")
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
        view = weights.get_view("cutlass_nvfp4")
        if set(view) != set(_WEIGHT_KEYS):
            raise ValueError(
                "cuDNN Frost NVFP4 requires plain canonical weights without overrides"
            )
        w1, w2, a1, sf1, alpha1, a2, sf2, alpha2 = (view[name] for name in _WEIGHT_KEYS)
        mult = 2 if self.config.activation.is_gated else 1
        expected = (
            (w1, torch.uint8, (e, mult * i, h // 2)),
            (w2, torch.uint8, (e, h, i // 2)),
            (sf1, torch.uint8, (e, mult * i, h // 16)),
            (sf2, torch.uint8, (e, h, i // 16)),
            (alpha1, torch.float32, (e,)),
            (alpha2, torch.float32, (e,)),
        )
        if any(
            v.dtype != dtype or tuple(v.shape) != shape for v, dtype, shape in expected
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
        with torch.cuda.device(self.device):
            self._check_scale(a1, positive=True)
            self._check_scale(a2, positive=True)
            self._check_scale(alpha1, positive=False)
            self._check_scale(alpha2, positive=False)
        return (
            t,
            h,
            i,
            e,
            k,
            w1,
            w2,
            sf1,
            sf2,
            xsf.view(torch.uint8),
            a1.reshape(1),
            alpha1,
            a2.reshape(1),
            alpha2,
        )

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
        return len(first) == len(second) == 2

    def pack_inputs(self, act, weights):
        self._require_built()
        t, h, i, e, k, w1, w2, sf1, sf2, xsf, a1, alpha1, a2, alpha2 = (
            self._validate_pack(act, weights)
        )
        mult = 2 if self.config.activation.is_gated else 1
        version = _tensor_version(sf1)
        cached = self._fc1_scale_views.get(id(sf1))
        if cached is None or cached[0] is not sf1 or cached[1] != version:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Prepare NVFP4 weight scales outside CUDA Graph capture"
                    )
                # Frozen SF descriptors have contiguous expert strides. Data
                # descriptors accept canonical interleaved up/gate weights.
                packed_sf1 = (
                    sf1.reshape(e, mult, i, h // 16).transpose(0, 1).contiguous()
                )
            cached = (sf1, version, packed_sf1)
            self._fc1_scale_views[id(sf1)] = cached
        sf1 = cached[2]
        first, second = _selected_kernels(
            t, h, i, e, k, self.device, self.config.activation
        )
        if len(first) != 2 or len(second) != 2:
            raise ValueError(
                "No measured two-by-two cuDNN Frost NVFP4 shortlist for this problem"
            )
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
                )
        output = torch.empty((t, h), dtype=torch.bfloat16, device=self.device)
        ids, scores = act.topk_ids, act.topk_weights

        def preserve_routing(tensors):
            # Autotuning must keep valid NVFP4 bytes/scales and the real route
            # distribution; independently random scale bytes may encode NaN.
            tensors[1:] = [
                act.hidden_states_q,
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
            ]
            return tensors

        tuning = TuningConfig(
            use_cuda_graph=True,
            cuda_graph_profile_replays=3,
            inputs_pre_hook=preserve_routing,
        )
        return _Inputs(
            [
                output,
                act.hidden_states_q,
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
            self._plans[key],
            tuning,
        )

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        state = self.launch_state_for(inputs)
        if state is not None:
            return list(state.launches)
        tokens, packed_hidden = inputs[1].shape
        hidden = 2 * packed_hidden
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
    runner = CudnnFrostNvfp4MoeRunner(config, device)
    try:
        runner.check_support()
    except (NotImplementedError, ValueError):
        return None
    runner.build()
    return runner
