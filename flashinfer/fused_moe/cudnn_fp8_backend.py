# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frost grouped FP8 MoE with live per-expert epilogue scales."""

from __future__ import annotations

from itertools import product
import logging

import torch

from ..gemm.gemm_base import _check_cudnn_plan_build_not_capturing
from ..grouped_mm.cudnn.core import _plan_index
from ..tllm_enums import RoutingInputMode
from .api import CudnnFp8PerTensorConfig, QuantFormat, SwiGLU
from .cudnn_backend import CudnnMoeRunner, _gated_activation
from .prepare import _fp8_per_tensor_scale, _quantize_fp8_per_expert
from .runners import _validate_pack_devices, _validate_prerouted_inputs

_LOG = logging.getLogger(__name__)


def prepare_cudnn_fp8_per_tensor_weights(
    w1_bf16,
    w2_bf16,
    *,
    hidden_states_scale_global,
    intermediate_scale_global,
    num_local_experts,
    hidden_size,
    intermediate_size,
    activation=None,
    device=None,
    copy_fc1_weights=True,
    weight_layout=None,
):
    """Quantize canonical BF16 weights; no TRTLLM row shuffling is applied.

    Scale values are quantization multipliers. Both halves of FC1 use the
    same multiplier per expert. Validation and quantization occur at model
    preparation, where host synchronization is allowed, never during forward.
    """
    if type(copy_fc1_weights) is not bool:
        raise ValueError("copy_fc1_weights must be bool")
    if weight_layout not in (None, "blocked_128x128_v1"):
        raise ValueError(f"Unsupported Frost FP8 weight_layout {weight_layout!r}")
    if weight_layout is not None and (hidden_size % 128 or intermediate_size % 128):
        raise ValueError(
            "blocked_128x128_v1 requires hidden/intermediate multiples of 128"
        )
    activation = activation or SwiGLU()
    if not isinstance(activation, CudnnMoeRunner.supported_activation_classes):
        raise NotImplementedError("Frost FP8 MoE requires a gated activation")
    e, h, i = num_local_experts, hidden_size, intermediate_size
    if min(e, h, i) <= 0:
        raise ValueError("Frost FP8 MoE weight dimensions must be positive")
    if w1_bf16.dtype != torch.bfloat16 or w2_bf16.dtype != torch.bfloat16:
        raise ValueError("Frost FP8 weight preparation requires BF16 weights")
    if tuple(w1_bf16.shape) != (e, 2 * i, h) or tuple(w2_bf16.shape) != (e, h, i):
        raise ValueError("Frost FP8 MoE expects [up, gate] FC1 and [E,H,I] FC2 weights")
    device = torch.device(device or w1_bf16.device)
    if device.type == "cuda":
        _check_cudnn_plan_build_not_capturing("FP8 MoE weight preparation")
    input_multiplier = _fp8_per_tensor_scale(
        hidden_states_scale_global, name="hidden_states_scale_global", device=device
    )
    intermediate_multiplier = _fp8_per_tensor_scale(
        intermediate_scale_global, name="intermediate_scale_global", device=device
    )
    w1, w1_multiplier = _quantize_fp8_per_expert(w1_bf16.to(device).contiguous())
    w2, w2_multiplier = _quantize_fp8_per_expert(w2_bf16.to(device).contiguous())
    fc1_scale = 1.0 / (input_multiplier * w1_multiplier)
    fc2_scale = 1.0 / (intermediate_multiplier * w2_multiplier)
    for name, tensor in (("fc1_scale", fc1_scale), ("fc2_scale", fc2_scale)):
        if not bool((torch.isfinite(tensor) & (tensor > 0)).all()):
            raise ValueError(f"{name} must remain finite and positive in FP32")
    up, gate = w1.split(i, dim=1)
    if weight_layout is not None:

        def pack(weight):
            e, n, k = weight.shape
            # Preparation only. Retain the physical rank-5 shape so ordinary
            # rank-3 weight views cannot accidentally satisfy this contract.
            return (
                weight.reshape(e, n // 128, 128, k // 128, 128)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
            )

        if copy_fc1_weights:
            up, gate = pack(up), pack(gate)
        else:
            # Retain up/gate views of one packed FC1 allocation. Each expert's
            # blocks are contiguous; TMA carries the doubled expert pitch.
            up, gate = pack(w1).split(i // 128, dim=1)
        w2 = pack(w2)
    elif copy_fc1_weights:
        up, gate = up.contiguous(), gate.contiguous()
    return {
        "up": up,
        "gate": gate,
        "down": w2.contiguous(),
        "fc1_scale": fc1_scale.reshape(e, 1, 1).contiguous(),
        "intermediate_multiplier": intermediate_multiplier.reshape(1, 1, 1),
        "fc2_scale": fc2_scale.reshape(e, 1, 1).contiguous(),
        "hidden_states_scale_global": input_multiplier,
    }


class _Fp8Stage:
    """Plan-owned scalar storage; every operand comes from the current call."""

    def __init__(
        self,
        x,
        weights,
        offsets,
        scale,
        output,
        *,
        activation=None,
        multiplier=None,
        tactic=None,
        tactics=(),
        weight_layout=None,
    ):
        import cudnn

        self.weight_layout = weight_layout
        self.handle = cudnn.create_handle()
        self.graph = g = cudnn.pygraph(
            io_data_type=cudnn.data_type.FP8_E4M3,
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=self.handle,
        )

        def desc(name, tensor, dtype):
            return g.tensor(
                name=name,
                dim=list(tensor.shape),
                stride=list(tensor.stride()),
                data_type=dtype,
            )

        self.x = desc("x", x.unsqueeze(0), cudnn.data_type.FP8_E4M3)
        self.offsets = desc(
            "offsets", offsets[:-1].view(-1, 1, 1), cudnn.data_type.INT32
        )
        self.scale = desc("expert_scale", scale, cudnn.data_type.FLOAT)
        self.weights = []
        products = []
        for index, weight in enumerate(weights):
            b = desc(
                f"weight{index}",
                weight if weight_layout is not None else weight.transpose(1, 2),
                cudnn.data_type.FP8_E4M3,
            )
            self.weights.append(b)
            product = g.moe_grouped_matmul(
                self.x,
                b,
                self.offsets,
                mode=cudnn.moe_grouped_matmul_mode.NONE,
                **(
                    {"weight_layout": weight_layout}
                    if weight_layout is not None
                    else {}
                ),
            )
            products.append(g.mul(product, self.scale))
        self.scalar_bindings = {}
        self.multiplier = None
        if activation is not None:
            y = _gated_activation(
                g, products[0], products[1], activation, x.device, self.scalar_bindings
            )
            self.multiplier = desc(
                "intermediate_multiplier", multiplier, cudnn.data_type.FLOAT
            )
            bounds = []
            for name, value in (("fp8_min", -448.0), ("fp8_max", 448.0)):
                storage = torch.full(
                    (1, 1, 1), value, device=x.device, dtype=torch.float32
                )
                bound = desc(name, storage, cudnn.data_type.FLOAT)
                self.scalar_bindings[bound] = storage
                bounds.append(bound)
            y = g.min(g.max(g.mul(y, self.multiplier), bounds[0]), bounds[1])
            dtype = cudnn.data_type.FP8_E4M3
        else:
            y = products[0]
            dtype = cudnn.data_type.BFLOAT16
        self.output = y.set_dim([1, *output.shape]).set_stride(
            [output.numel(), *output.stride()]
        )
        y.set_output(True).set_data_type(dtype)
        g.validate()
        g.build_operation_graph()
        if tactic is not None and tactics:
            raise ValueError("Specify one tactic or a candidate domain for a stage")
        requested = tactics or ((tactic,) if tactic is not None else ())
        if requested:
            for engine, knobs in requested:
                if int(engine) != 20400:
                    raise ValueError(
                        "Frost FP8 MoE requires engine20400 for each stage"
                    )
                g.create_execution_plan(
                    int(engine), {cudnn.knob_type(int(k)): int(v) for k, v in knobs}
                )
        else:
            # FE's ordinary strategy supplies concrete knobs. Retain only
            # Frost entries: this runner never builds a closed-source fallback.
            g.create_execution_plans([cudnn.heur_mode.OPENSOURCE])
        g.check_support()
        self.tactic_indices = {}
        for index in range(g.get_execution_plan_count()):
            try:
                engine, knobs = g.get_engine_and_knobs_at_index(index)
            except NotImplementedError as exc:
                if requested:
                    raise
                # The OSS query can also report a C++ heuristic delegate
                # without a replayable identity. It cannot be a Frost tactic.
                _LOG.debug("Skipping non-replayable FP8 candidate %d: %s", index, exc)
                continue
            if int(engine) == 20400:
                record = (
                    int(engine),
                    tuple(sorted((int(k), int(v)) for k, v in knobs.items())),
                )
                self.tactic_indices.setdefault(record, index)
        if not self.tactic_indices:
            raise NotImplementedError("No Frost20400 plan supports this FP8 stage")
        for index in self.tactic_indices.values():
            # Explicit candidate preparation is strict; do not advertise a
            # declined/unbuilt plan and discover it during CUDA graph capture.
            g.build_plan_at_index(index)
        g.select_plan(next(iter(self.tactic_indices.values())))
        self.tactics = tuple(self.tactic_indices)
        # get_workspace_size() describes the selected plan. A different
        # candidate can need more storage, so size over every built plan.
        workspace_bytes = max(
            g.get_workspace_size_plan_at_index(index)
            for index in self.tactic_indices.values()
        )
        self.workspace = torch.empty(
            workspace_bytes, device=x.device, dtype=torch.uint8
        )

    def run(self, x, weights, offsets, scale, output, multiplier=None, tactic=-1):
        import cudnn

        cudnn.set_stream(
            handle=self.handle, stream=torch.cuda.current_stream(x.device).cuda_stream
        )
        pack = {
            self.x: x.unsqueeze(0),
            self.offsets: offsets[:-1].view(-1, 1, 1),
            self.scale: scale,
            self.output: output.unsqueeze(0),
            **self.scalar_bindings,
        }
        pack.update(
            (desc, weight if self.weight_layout is not None else weight.transpose(1, 2))
            for desc, weight in zip(self.weights, weights, strict=True)
        )
        if self.multiplier is not None:
            pack[self.multiplier] = multiplier
        if type(tactic) is int:
            index = _plan_index(self.graph, tactic)
        else:
            try:
                index = self.tactic_indices[tactic]
            except (KeyError, TypeError):
                raise ValueError(
                    "FP8 stage tactic is unavailable; retune this shape"
                ) from None
        if index == -1:
            self.graph.execute(pack, self.workspace, handle=self.handle)
        else:
            self.graph.execute_plan_at_index(
                pack, self.workspace, index, handle=self.handle
            )


class CudnnFp8PerTensorRunner(CudnnMoeRunner):
    """Calibrated FP8 MoE, sharing BF16 routing and workspace lifetime rules."""

    backend_key = "cudnn_fp8_per_tensor"
    _backend_config_type = CudnnFp8PerTensorConfig
    _cache_version = "cudnn-fp8-per-tensor-v4-blocked-weights"
    _activation_dtype = torch.float8_e4m3fn
    supported_quant_variants = ((QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),)

    def _weight_layout_key(self, inputs):
        return (
            self.backend_config.weight_layout,
            tuple(
                (tuple(tensor.shape), tuple(tensor.stride())) for tensor in inputs[3:6]
            ),
        )

    def get_cache_key_extras(self, inputs):
        return (*super().get_cache_key_extras(inputs), self._weight_layout_key(inputs))

    def _check_support(self):
        super()._check_support()
        if not self.backend_config.use_native_routing:
            raise NotImplementedError("Frost FP8 MoE requires use_native_routing=True")

    def pack_inputs(self, act, weights):
        self._require_built()
        if act.routing_input_mode not in self.supported_routing_modes:
            raise NotImplementedError(
                "Frost FP8 MoE requires precomputed top-k routing"
            )
        _validate_pack_devices(act, type(self).__name__)
        x = act.hidden_states_q
        if (
            x.device != self.device
            or x.dtype != self._activation_dtype
            or x.ndim != 2
            or not x.is_contiguous()
        ):
            raise ValueError(
                "Frost FP8 MoE requires contiguous E4M3 [tokens,hidden] on the runner device"
            )
        if x.shape[0] == 0:
            raise ValueError("Frost FP8 MoE requires at least one token")
        _validate_prerouted_inputs(
            act,
            x.shape[0],
            self.config.routing.top_k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )
        if act.hidden_states_scale is not None or act.per_token_scale is not None:
            raise ValueError(
                "Frost FP8 MoE calibration is folded into the weight-view epilogue scales"
            )
        e, h, i = (
            self.config.routing.num_experts,
            x.shape[1],
            self.config.experts.intermediate_size,
        )
        view = weights.get_view(self.backend_key)
        names = (
            "up",
            "gate",
            "down",
            "fc1_scale",
            "intermediate_multiplier",
            "fc2_scale",
        )
        shapes = ((e, i, h), (e, i, h), (e, h, i), (e, 1, 1), (1, 1, 1), (e, 1, 1))
        if self.backend_config.weight_layout is not None:
            if h % 128 or i % 128:
                raise ValueError(
                    "blocked_128x128_v1 requires hidden/intermediate multiples of 128"
                )
            shapes = tuple(
                (s[0], s[1] // 128, s[2] // 128, 128, 128) if j < 3 else s
                for j, s in enumerate(shapes)
            )
        tensors = []
        for index, (name, shape) in enumerate(zip(names, shapes, strict=True)):
            tensor = view[name]
            dtype = self._activation_dtype if index < 3 else torch.float32
            # Also serve up/gate views of canonical [E, 2*I, H] storage.
            # Inner matrices stay contiguous; the graph/TMA descriptor keeps
            # the doubled expert pitch. No execution-time conversion occurs.
            layout_ok = tensor.is_contiguous() or (
                index < 2 and tuple(tensor.stride()) == (2 * i * h, h, 1)
            )
            if self.backend_config.weight_layout is not None and index < 3:
                strides = tuple(tensor.stride())
                expert_elements = shape[1] * shape[2] * 16384
                layout_ok = (
                    len(strides) == 5
                    and strides[1:] == (shape[2] * 16384, 16384, 128, 1)
                    and strides[0] >= expert_elements
                    and strides[0] % 16 == 0
                )
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not layout_ok
            ):
                raise ValueError(
                    f"{name} must be contiguous {dtype} {shape} on {self.device}"
                    " (up/gate may use canonical doubled expert pitch; packed weights may use aligned nonoverlapping expert pitch)"
                )
            tensors.append(tensor)
        up, gate, down, first_scale, multiplier, second_scale = tensors
        inputs = [
            x,
            act.topk_ids,
            act.topk_weights,
            up,
            gate,
            down,
            None,
            None,
            act.routing_input_mode == RoutingInputMode.PackedPrecomputed,
            first_scale,
            multiplier,
            second_scale,
        ]
        inputs[7] = self._resources(inputs)["output"]
        return inputs

    def _prepare_stages(self, s, inputs, empty):
        _, _, _, up, gate, down = inputs[:6]
        first_scale, multiplier, second_scale = inputs[9:12]
        s["fc1"] = _Fp8Stage(
            s["routed"],
            [up, gate],
            s["offsets"],
            first_scale,
            s["intermediate"],
            activation=self.config.activation,
            multiplier=multiplier,
            tactic=self.backend_config.fc1_tactic,
            tactics=self.backend_config.fc1_tactics,
            weight_layout=self.backend_config.weight_layout,
        )
        s["fc2"] = _Fp8Stage(
            s["intermediate"],
            [down],
            s["offsets"],
            second_scale,
            s["projected"],
            tactic=self.backend_config.fc2_tactic,
            tactics=self.backend_config.fc2_tactics,
            weight_layout=self.backend_config.weight_layout,
        )
        s["fused"] = True

    def get_valid_tactics(self, inputs, profile):
        s = self._resources(inputs)
        return list(product(s["fc1"].tactics, s["fc2"].tactics))

    @staticmethod
    def _stage_indices(s, tactic):
        if type(tactic) is int and tactic == -1:
            return -1, -1
        if not isinstance(tactic, tuple) or len(tactic) != 2:
            raise ValueError(
                "FP8 MoE tactic must contain FC1 and FC2 engine/knob records"
            )
        try:
            # Validate both before launching either GEMM. Cache records select
            # only plans prepared for this runner's declared candidate domains.
            return tuple(
                s[stage].tactic_indices[record]
                for stage, record in zip(("fc1", "fc2"), tactic, strict=True)
            )
        except (KeyError, TypeError):
            raise ValueError(
                "FP8 MoE joint tactic is unavailable; retune this shape"
            ) from None

    def _run_stages(self, s, inputs, tactic):
        fc1_index, fc2_index = self._stage_indices(s, tactic)
        _, _, _, up, gate, down = inputs[:6]
        first_scale, multiplier, second_scale = inputs[9:12]
        s["fc1"].run(
            s["routed"],
            [up, gate],
            s["offsets"],
            first_scale,
            s["intermediate"],
            multiplier,
            fc1_index,
        )
        s["fc2"].run(
            s["intermediate"],
            [down],
            s["offsets"],
            second_scale,
            s["projected"],
            tactic=fc2_index,
        )
