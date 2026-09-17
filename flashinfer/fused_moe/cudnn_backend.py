# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN graph adapter for BF16 MoE from precomputed routing.

Sorting/gather/finalize are explicit adapter stages. FROST currently accepts
only already grouped tokens; its two shared-input matmuls can fuse gated activations.
"""

from __future__ import annotations

import logging
import struct
from collections import OrderedDict
from itertools import product
from typing import Any

import torch

from ..autotuner import TuningConfig
from ..gemm.gemm_base import _check_cudnn_plan_build_not_capturing
from ..grouped_mm.cudnn.core import (
    _check_cudnn_version,
    _plan_index,
    _plan_indices,
    _runtime_key,
)
from .api import (
    CudnnMoeConfig,
    GeGLU,
    GeGLUTanh,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantFormat,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from .runners import MoERunner, _validate_pack_devices, _validate_prerouted_inputs
from ..tllm_enums import RoutingInputMode

_LOG = logging.getLogger(__name__)


def _gated_activation(g, up, gate, activation, device, scalar_bindings):
    """Lower typed semantics once; scalar device storage belongs to the plan."""
    import cudnn

    scalars = {}

    def scalar(value):
        if value not in scalars:
            desc = g.tensor(
                name=f"activation_scalar_{len(scalars)}",
                dim=[1, 1, 1],
                stride=[1, 1, 1],
                data_type=cudnn.data_type.FLOAT,
            )
            scalar_bindings[desc] = torch.full(
                (1, 1, 1), value, device=device, dtype=torch.float32
            )
            scalars[value] = desc
        return scalars[value]

    def clamp_linear(value, limit):
        return g.min(g.max(value, scalar(-limit)), scalar(limit))

    def divide_by_scale(value, scale_value, scale):
        # Match the FP32 scale stored in the graph. Prepare its reciprocal once
        # instead of dividing every FC1 output element during execution.
        rounded_scale = struct.unpack("f", struct.pack("f", scale_value))[0]
        limits = torch.finfo(torch.float32)
        if limits.tiny <= rounded_scale <= limits.max:
            reciprocal = 1.0 / rounded_scale
            # Preserve division when the reciprocal would overflow or become
            # subnormal. In the fast path, rounding can differ by an FP32 ULP;
            # activation order and the public BF16 boundaries remain unchanged.
            if limits.tiny <= reciprocal <= limits.max:
                return g.mul(value, scalar(reciprocal))
        return g.div(value, scale)

    if isinstance(activation, SwiGLU):
        if activation.limit != SwiGLU().limit:
            up = clamp_linear(up, activation.limit)
            gate = g.min(gate, scalar(activation.limit))
        if activation.beta != 0:
            up = g.add(up, scalar(activation.beta))
        return g.mul(up, g.swish(gate, swish_beta=activation.alpha))
    if isinstance(activation, SiTU):
        if activation.clamp_limit is not None:
            up = clamp_linear(up, activation.clamp_limit)
            gate = g.min(gate, scalar(activation.clamp_limit))
        if activation.linear_scale is not None:
            scale = scalar(activation.linear_scale)
            up = g.mul(
                scale, g.tanh(divide_by_scale(up, activation.linear_scale, scale))
            )
        scale = scalar(activation.gate_scale)
        return g.mul(
            g.mul(
                g.mul(up, scale),
                g.tanh(divide_by_scale(gate, activation.gate_scale, scale)),
            ),
            g.sigmoid(gate),
        )
    if isinstance(activation, GeGLU):
        return g.mul(up, g.gelu(gate))
    if isinstance(activation, GeGLUTanh):
        return g.mul(up, g.gelu_approx_tanh(gate))
    if isinstance(activation, SwiGLUStep):
        return g.mul(
            clamp_linear(up, activation.limit),
            g.min(g.swish(gate), scalar(activation.limit)),
        )
    raise NotImplementedError(f"cuDNN MoE does not support {activation!r}")


class _Stage:
    def __init__(
        self,
        a,
        weights,
        offsets,
        out,
        *,
        fused=False,
        tactic=None,
        activation=None,
        tactics=(),
    ):
        import cudnn

        _check_cudnn_plan_build_not_capturing("MoE")
        self.handle = cudnn.create_handle()
        self.graph = g = cudnn.pygraph(
            io_data_type=cudnn.data_type.BFLOAT16,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        self.a = g.tensor(
            name="tokens",
            dim=[1, *a.shape],
            stride=[a.numel(), *a.stride()],
            data_type=cudnn.data_type.BFLOAT16,
        )
        self.offsets = g.tensor(
            name="first_token_offset",
            dim=[weights[0].shape[0], 1, 1],
            stride=[1, 1, 1],
            data_type=cudnn.data_type.INT32,
        )
        self.weights = []
        gemms = []
        for i, weight in enumerate(weights):
            view = weight.transpose(1, 2)
            desc = g.tensor(
                name=f"weight_{i}",
                dim=list(view.shape),
                stride=list(view.stride()),
                data_type=cudnn.data_type.BFLOAT16,
            )
            self.weights.append(desc)
            gemms.append(
                g.moe_grouped_matmul(
                    self.a,
                    desc,
                    self.offsets,
                    mode=cudnn.moe_grouped_matmul_mode.NONE,
                    name=f"gemm_{i}",
                )
            )
        self.scalar_bindings = {}
        self.out = (
            _gated_activation(
                g,
                gemms[0],
                gemms[1],
                activation or SwiGLU(),
                a.device,
                self.scalar_bindings,
            )
            if fused
            else gemms[0]
        )
        self.out.set_dim([1, *out.shape]).set_stride([out.numel(), *out.stride()])
        self.out.set_output(True).set_data_type(
            cudnn.data_type.FLOAT
            if out.dtype == torch.float32
            else cudnn.data_type.BFLOAT16
        )
        g.validate()
        g.build_operation_graph()
        if tactic is not None and tactics:
            raise ValueError("Specify one tactic or a candidate domain for a stage")
        requested = tactics or ((tactic,) if tactic is not None else ())
        if requested:
            for engine, knobs in requested:
                g.create_execution_plan(
                    int(engine), {cudnn.knob_type(int(k)): int(v) for k, v in knobs}
                )
        else:
            g.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        g.check_support()
        self.tactic_indices = {}
        for record, index in _plan_indices(g).items():
            try:
                g.build_plan_at_index(index)
            except (NotImplementedError, cudnn.cudnnGraphNotSupportedError) as exc:
                if requested:
                    raise
                _LOG.debug("Skipping declined BF16 candidate %d: %s", index, exc)
                continue
            self.tactic_indices[record] = index
        if not self.tactic_indices:
            raise NotImplementedError("No prepared cuDNN plan supports this BF16 stage")
        g.select_plan(next(iter(self.tactic_indices.values())))
        self.tactics = tuple(self.tactic_indices)
        workspace_bytes = max(
            g.get_workspace_size_plan_at_index(index)
            for index in self.tactic_indices.values()
        )
        self.workspace = torch.empty(
            workspace_bytes, device=a.device, dtype=torch.uint8
        )

    def plan_index(self, tactic):
        index = _plan_index(self.graph, tactic)
        if index != -1 and index not in self.tactic_indices.values():
            raise ValueError(
                "cuDNN BF16 stage tactic is unavailable; retune this shape"
            )
        return index

    def run(self, a, weights, offsets, out, tactic=-1):
        import cudnn

        cudnn.set_stream(
            handle=self.handle, stream=torch.cuda.current_stream(a.device).cuda_stream
        )
        pack = {
            self.a: a.unsqueeze(0),
            self.offsets: offsets[:-1].view(-1, 1, 1),
            self.out: out.unsqueeze(0),
        }
        pack.update(
            (desc, weight.transpose(1, 2))
            for desc, weight in zip(self.weights, weights, strict=True)
        )
        pack.update(self.scalar_bindings)
        index = self.plan_index(tactic)
        if index == -1:
            self.graph.execute(pack, self.workspace, handle=self.handle)
        else:
            self.graph.execute_plan_at_index(
                pack, self.workspace, index, handle=self.handle
            )


class _Activation:
    """Keep FC1 accumulation in FP32 when shared-input fusion declines."""

    def __init__(self, fc1, output, activation=None):
        import cudnn

        self.handle = cudnn.create_handle()
        self.graph = g = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )
        m, n = output.shape
        self.gate = g.tensor(
            name="gate",
            dim=[1, m, n],
            stride=[2 * m * n, 2 * n, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        self.up = g.tensor(
            name="up",
            dim=[1, m, n],
            stride=[2 * m * n, 2 * n, 1],
            data_type=cudnn.data_type.FLOAT,
        )
        self.scalar_bindings = {}
        self.out = _gated_activation(
            g,
            self.up,
            self.gate,
            activation or SwiGLU(),
            fc1.device,
            self.scalar_bindings,
        )
        self.out.set_dim([1, m, n]).set_stride([m * n, n, 1])
        self.out.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
        self.workspace = torch.empty(
            g.get_workspace_size(), device=fc1.device, dtype=torch.uint8
        )

    def run(self, fc1, output):
        import cudnn

        gate, up = fc1.chunk(2, dim=-1)
        cudnn.set_stream(
            handle=self.handle, stream=torch.cuda.current_stream(fc1.device).cuda_stream
        )
        self.graph.execute(
            {
                self.gate: gate.unsqueeze(0),
                self.up: up.unsqueeze(0),
                self.out: output.unsqueeze(0),
                **self.scalar_bindings,
            },
            self.workspace,
            handle=self.handle,
        )


class CudnnMoeRunner(MoERunner):
    backend_key = "cudnn"
    _backend_config_type: type = CudnnMoeConfig
    # Joint stage tactics and explicit fusion routes change cache identities.
    _cache_version = "cudnn-bf16-v7-joint-routes"
    _activation_dtype = torch.bfloat16
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
    supported_activation_classes = (SwiGLU, SiTU, GeGLU, GeGLUTanh, SwiGLUStep)
    supports_expert_parallelism = False

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.backend_config = next(
            c for c in config.backend if isinstance(c, self._backend_config_type)
        )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CudnnMoeRunner requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # Exact shapes retain the caller's routing tensors when profiling. No
        # synthetic integer expert IDs or independent block-scale offsets.
        self.tuning_config = TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True)
        self._resources_cache: OrderedDict[tuple, dict[str, Any]] = OrderedDict()

    def _check_support(self):
        super()._check_support()
        major, minor = torch.cuda.get_device_capability(self.device)
        if not self._backend_config_type.supported(major * 10 + minor):
            raise NotImplementedError(
                f"CudnnMoeRunner does not support SM{major}{minor}"
            )
        _check_cudnn_version(92100, "BF16 MoE")
        if not self.config.finalize.do_finalize:
            raise NotImplementedError(
                "CudnnMoeRunner currently requires weighted finalize"
            )
        if self.config.execution.enable_pdl:
            raise NotImplementedError(
                "CudnnMoeRunner currently requires enable_pdl=False"
            )
        if self.config.quant.per_token_scale:
            raise NotImplementedError(
                "CudnnMoeRunner BF16 does not use per-token scales"
            )

    def _build(self):
        # Load the existing FI finalization kernel eagerly. Native
        # grouped GEMM plans depend on the exact token/hidden extents and warm
        # outside capture on the first pack/forward.
        from .cute_dsl.moe_utils import _get_moe_utils_module

        _get_moe_utils_module()

    def _cache_key_extras(self):
        props = torch.cuda.get_device_properties(self.device)
        return (
            *super()._cache_key_extras(),
            self._cache_version,
            repr(self.backend_config),
            _runtime_key(self.device),
            props.name,
            props.major,
            props.minor,
        )

    def get_cache_key_extras(self, inputs):
        return (*self._cache_key_extras(), str(inputs[2].dtype), inputs[8])

    def pack_inputs(self, act: MoEActivationPack, weights: MoEWeightPack):
        self._require_built()
        if act.routing_input_mode not in self.supported_routing_modes:
            raise NotImplementedError(
                "CudnnMoeRunner requires precomputed top-k routing"
            )
        _validate_pack_devices(act, type(self).__name__)
        x = act.hidden_states_q
        if (
            x.device != self.device
            or x.dtype != torch.bfloat16
            or x.ndim != 2
            or not x.is_contiguous()
        ):
            raise ValueError(
                "cuDNN MoE requires contiguous [tokens, hidden] BF16 activations on the runner device"
            )
        if x.shape[0] == 0:
            raise ValueError("cuDNN MoE currently requires at least one token")
        _validate_prerouted_inputs(
            act,
            x.shape[0],
            self.config.routing.top_k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )
        if act.hidden_states_scale is not None or act.per_token_scale is not None:
            raise ValueError("cuDNN BF16 MoE does not consume activation scales")
        v = weights.get_view(self.backend_key)
        e, i, h = (
            self.config.routing.num_experts,
            self.config.experts.intermediate_size,
            x.shape[1],
        )
        tensors = [v[name] for name in ("up", "gate", "down", "gate_up")]
        for t, shape in zip(
            tensors, ((e, i, h), (e, i, h), (e, h, i), (e, 2 * i, h)), strict=True
        ):
            if (
                t.device != self.device
                or t.dtype != torch.bfloat16
                or tuple(t.shape) != shape
                or not t.is_contiguous()
            ):
                raise ValueError(
                    f"Invalid cuDNN MoE weight layout; expected contiguous BF16 {shape} on {self.device}"
                )
        # Packed mode's public contract narrows routing weights to BF16. The
        # cuDNN adapter does not encode IDs into their high bits, but preserves
        # the same numeric boundary. Unpacked mode retains FP32 weights.
        scales = act.topk_weights
        round_scales = act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        inputs = [x, act.topk_ids, scales, *tensors]
        # Expose the reusable output as an ordinary operand so autotuning and
        # the unified fuzzer can initialize/poison it before every execution.
        return [*inputs, self._resources(inputs)["output"], round_scales]

    def _weight_layout_key(self, inputs):
        # BF16 currently accepts only contiguous weights. Quantized runners
        # can specialize this without adding work to the BF16 hot path.
        return ()

    def _resources(self, inputs):
        x, ids, scales, up, gate, down, gate_up = inputs[:7]
        t, h = x.shape
        e, i = self.config.routing.num_experts, self.config.experts.intermediate_size
        r = self.config.routing.top_k
        # MoELayer requires one runner per thread/stream; eager warmup also
        # prepares storage used by the subsequent capture on its side stream.
        # A plan declares operand strides. Canonical split-FC1 views have a
        # different expert pitch from independently contiguous weight packs.
        key = (t, h, e, i, _runtime_key(self.device), self._weight_layout_key(inputs))
        if key in self._resources_cache:
            self._resources_cache.move_to_end(key)
            return self._resources_cache[key]
        _check_cudnn_plan_build_not_capturing("MoE routing workspace and plans")

        def empty(shape, dtype):
            return torch.empty(shape, device=self.device, dtype=dtype)

        s = {
            "expert_range": torch.arange(e + 1, device=self.device, dtype=torch.int32),
            "offsets": empty(e + 1, torch.int32),
            "routed": empty((t * r, h), self._activation_dtype),
            "intermediate": empty((t * r, i), self._activation_dtype),
            "projected": empty((t * r, h), torch.bfloat16),
            "output": empty((t, h), torch.bfloat16),
        }
        if self.backend_config.use_native_routing:
            from .cute_dsl.moe_utils import allocate_moe_sort_buffers

            s["native_sort"] = allocate_moe_sort_buffers(
                t, e, r, tile_tokens_dim=1, device=self.device
            )
            s["expert_counts"] = empty(2 * e, torch.int32)
        else:
            s.update(
                sorted_ids=empty(t * r, torch.int32),
                order=empty(t * r, torch.int64),
                source_rows=empty(t * r, torch.int64),
                inverse64=empty(t * r, torch.int64),
                inverse=empty((t, r), torch.int32),
                arange=torch.arange(t * r, device=self.device, dtype=torch.int64),
            )
        self._prepare_stages(s, inputs, empty)
        self._resources_cache[key] = s
        # Caller owns this runner until all captures using it are retired.
        # Do not evict workspace referenced by an existing CUDA graph.
        return s

    def _prepare_stages(self, s, inputs, empty):
        """Build BF16 stages; quantized runners specialize this preparation."""
        import cudnn

        x, _, _, up, gate, down, gate_up = inputs[:7]
        t, _ = x.shape
        i = up.shape[1]
        r = self.config.routing.top_k
        # FROST's shared-A FC1 fusion is optional. A graph decline is a route
        # choice; record it and build the ordinary backend-compatible FC1.
        s["fused"] = False
        if self.backend_config.fc1_fusion is not False:
            try:
                s["fc1"] = _Stage(
                    s["routed"],
                    [up, gate],
                    s["offsets"],
                    s["intermediate"],
                    fused=True,
                    tactic=self.backend_config.fc1_tactic,
                    activation=self.config.activation,
                    tactics=self.backend_config.fc1_tactics,
                )
                s["fused"] = True
            except (
                NotImplementedError,
                RuntimeError,
                cudnn.cudnnGraphNotSupportedError,
            ) as exc:
                _LOG.info("cuDNN MoE shared-input FC1 fusion declined: %s", exc)
                if self.backend_config.fc1_fusion is True:
                    raise
        if not s["fused"]:
            s["fc1_output"] = empty((t * r, 2 * i), torch.float32)
            s["fc1"] = _Stage(
                s["routed"],
                [gate_up],
                s["offsets"],
                s["fc1_output"],
                tactic=self.backend_config.fc1_tactic,
                tactics=self.backend_config.fc1_tactics,
            )
            s["activation"] = _Activation(
                s["fc1_output"], s["intermediate"], self.config.activation
            )
            s["fused"] = False
        s["fc2"] = _Stage(
            s["intermediate"],
            [down],
            s["offsets"],
            s["projected"],
            tactic=self.backend_config.fc2_tactic,
            tactics=self.backend_config.fc2_tactics,
        )

    def get_valid_tactics(self, inputs, profile):
        s = self._resources(inputs)
        return list(product(s["fc1"].tactics, s["fc2"].tactics))

    @staticmethod
    def _stage_indices(s, tactic):
        # Preserve direct FC1 replay for callers of the initial runner. New
        # tuning records contain stable identities for both stages.
        if type(tactic) is int or (
            isinstance(tactic, tuple) and len(tactic) == 2 and type(tactic[0]) is int
        ):
            return s["fc1"].plan_index(tactic), -1
        if not isinstance(tactic, tuple) or len(tactic) != 2:
            raise ValueError(
                "BF16 MoE tactic must contain FC1 and FC2 engine/knob records"
            )
        for record in tactic:
            if (
                not isinstance(record, tuple)
                or len(record) != 2
                or type(record[0]) is not int
            ):
                raise ValueError(
                    "BF16 MoE joint tactics require stable engine/knob records"
                )
        return tuple(
            s[stage].plan_index(record)
            for stage, record in zip(("fc1", "fc2"), tactic, strict=True)
        )

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        self._require_built()
        from .cute_dsl.moe_utils import moe_unpermute

        x, ids, scales, up, gate, down, gate_up = inputs[:7]
        s = self._resources(inputs)
        # Validate the complete record before routing or either GEMM launches.
        self._stage_indices(s, tactic)
        output = inputs[7]
        if do_preparation:
            return output
        r = self.config.routing.top_k
        if self.backend_config.use_native_routing:
            from .cute_dsl.moe_utils import (
                _get_moe_utils_module,
                _try_moe_route_permute_small,
                moe_permute,
            )

            b = s["native_sort"]
            t, e = x.shape[0], up.shape[0]
            # The bounded Frost path consumes only these three operands.
            # Generic native routing retains ownership of its other metadata.
            compact = (
                0 < t <= 64
                and e == 128
                and r == 8
                and x.shape[1] == 2048
                and _try_moe_route_permute_small(
                    x,
                    ids,
                    s["routed"],
                    b["out_expanded_idx_to_permuted_idx"],
                    s["offsets"],
                    e,
                )
            )
            if not compact:
                # One row per tile means exactly T*top_k tiles, with no padding or
                # unwritten tail. The native precomputed-ID path fills every slot.
                # Calling the native binding also lets us own the large-token
                # expert-count scratch; the general wrapper allocates it per call.
                _get_moe_utils_module()["flashinfer_moe_sort_with_offsets"](
                    ids.data_ptr(),
                    scales.data_ptr(),
                    t,
                    e,
                    r,
                    0,
                    e,
                    1,
                    False,
                    b["out_tile_idx_to_expert_idx"].data_ptr(),
                    b["out_tile_idx_to_mn_limit"].data_ptr(),
                    b["out_expanded_idx_to_permuted_idx"].data_ptr(),
                    b["out_permuted_idx_to_expanded_idx"].data_ptr(),
                    b["out_total_num_padded_tokens"].data_ptr(),
                    b["out_num_non_exiting_tiles"].data_ptr(),
                    s["expert_counts"].data_ptr() if t > 1024 else 0,
                    torch.cuda.current_stream(self.device).cuda_stream,
                    s["offsets"].data_ptr(),
                )
                moe_permute(
                    x,
                    s["routed"],
                    b["out_tile_idx_to_mn_limit"],
                    b["out_permuted_idx_to_expanded_idx"],
                    b["out_num_non_exiting_tiles"],
                    t * r,
                    r,
                    1,
                    enable_pdl=False,
                )
            inverse = b["out_expanded_idx_to_permuted_idx"]
        else:
            # Fixed output extents: no device-to-host reads or dynamic allocations.
            torch.sort(ids.view(-1), out=(s["sorted_ids"], s["order"]))
            torch.div(s["order"], r, rounding_mode="floor", out=s["source_rows"])
            torch.searchsorted(
                s["sorted_ids"], s["expert_range"], out_int32=True, out=s["offsets"]
            )
            s["inverse64"].scatter_(0, s["order"], s["arange"])
            s["inverse"].view(-1).copy_(s["inverse64"])
            torch.index_select(x, 0, s["source_rows"], out=s["routed"])
            inverse = s["inverse"]
        self._run_stages(s, inputs, tactic)
        moe_unpermute(
            s["projected"],
            output,
            inverse,
            scales,
            x.shape[0],
            r,
            enable_pdl=False,
            round_scales_to_bf16=inputs[8],
            # Reuse the NVIDIA TRT-LLM finalizer for the measured small-token
            # H2048/top8 region; preserve the routing mode's scale precision.
            use_native_finalize=(
                0 < x.shape[0] <= 64 and output.shape[1] == 2048 and r == 8
            ),
            use_wide_tiling=(
                (
                    0 < x.shape[0] <= 1024
                    or (0 < x.shape[0] <= 3072 and output.shape[1] == 4096 and r == 8)
                )
                and output.shape[1] in (4096, 8192)
                and r in (2, 4, 8, 16)
                and s["projected"].dtype == output.dtype == torch.bfloat16
                and scales.dtype == torch.float32
            ),
        )
        return output

    def _run_stages(self, s, inputs, tactic):
        fc1_index, fc2_index = self._stage_indices(s, tactic)
        _, _, _, up, gate, down, gate_up = inputs[:7]
        if s["fused"]:
            s["fc1"].run(
                s["routed"], [up, gate], s["offsets"], s["intermediate"], fc1_index
            )
        else:
            s["fc1"].run(
                s["routed"], [gate_up], s["offsets"], s["fc1_output"], fc1_index
            )
            s["activation"].run(s["fc1_output"], s["intermediate"])
        s["fc2"].run(s["intermediate"], [down], s["offsets"], s["projected"], fc2_index)
