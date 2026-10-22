# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light adapter for the unified MoELayer API."""

from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version

from ....autotuner import TuningConfig
from ...api import FrostLowLatencyConfig, QuantFormat, RoutingInputMode, SwiGLU
from ...runners import MoERunner, _validate_prerouted_inputs
from .fma.activation_config import ACTIVATIONS, lower_activation


class FrostLowLatencyRunner(MoERunner):
    backend_key = "frost_low_latency"
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supported_quant_variants = (
        (QuantFormat.MXFP4, QuantFormat.MXFP8),
        (QuantFormat.MXFP8, QuantFormat.MXFP8),
        (QuantFormat.BF16, QuantFormat.BF16),
    )
    supported_activation_classes = (SwiGLU,)
    supported_activation_classes_by_quant = {
        (QuantFormat.MXFP4, QuantFormat.MXFP8): (SwiGLU,),
        (QuantFormat.MXFP8, QuantFormat.MXFP8): (SwiGLU,),
        (QuantFormat.BF16, QuantFormat.BF16): tuple(ACTIVATIONS),
    }
    supports_expert_parallelism = False

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None and self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.options = next(
            c for c in config.backend.candidates if isinstance(c, FrostLowLatencyConfig)
        )
        # Exact shapes, no synthetic token buckets: route IDs remain valid and
        # workspace shapes cannot drift away from the profiled activation.
        self.tuning_config = TuningConfig(use_cuda_graph=True)

    def _check_support(self):
        super()._check_support()
        try:
            installed = Version(version("nvidia-cutlass-dsl"))
        except PackageNotFoundError:
            installed = None
        if installed is None or installed < Version("4.8"):
            raise NotImplementedError(
                "frost_low_latency requires nvidia-cutlass-dsl >= 4.8"
            )
        from ....utils import get_compute_capability

        if get_compute_capability(self.device) not in ((10, 0), (10, 3), (10, 7)):
            raise NotImplementedError(
                "frost_low_latency requires SM100, SM103 or SM107"
            )
        if self.config.quant.weight is QuantFormat.BF16:
            name, _ = lower_activation(self.config.activation)
            if self.config.routing.top_k > 32:
                raise NotImplementedError("Frost BF16 FMA requires top_k <= 32")
            if name in ("geglu", "gelu"):
                from cutlass import cute

                if not callable(getattr(cute.math, "erf", None)):
                    raise NotImplementedError("Frost GELU/GeGLU requires cute.math.erf")
        if self.config.quant.swizzled_scale_factors:
            raise NotImplementedError(
                "frost_low_latency requires linear UE8M0 activation scales"
            )
        if (
            self.config.experts.intermediate_size <= 0
            or self.config.experts.intermediate_size % 32
        ):
            raise NotImplementedError(
                "frost_low_latency requires intermediate_size divisible by 32"
            )

    def _build(self):
        pass

    def _accepts_shape(self, act):
        return (
            act.hidden_states_q.ndim == 2
            and 1 <= act.num_tokens <= self.options.max_num_tokens
            and act.hidden_states_q.shape[1] >= 128
            and act.hidden_states_q.shape[1] % 128 == 0
        )

    def _cache_key_extras(self):
        return super()._cache_key_extras() + (
            # Do not reuse winners selected with the deferred individual-call timer.
            "unified-fma-tc-standard-timing-v5",
            self.options.max_num_tokens,
            self.config.execution.enable_pdl,
        )

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        if self.config.quant.weight is QuantFormat.BF16:
            return [
                ("ffma", first, second)
                for first in (1, 2, 4, 8)
                for second in (1, 2, 4, 8)
            ]
        tactics = [k for k in (128, 256) if inputs[0].shape[1] % k == 0]
        if self.config.routing.top_k <= 32:
            tactics.append(("ffma", 1))
        return tactics

    def pack_inputs(self, act, weights):
        self._require_built()
        self._validate_pack_contract(act)
        if not self._accepts_shape(act):
            raise ValueError(
                "frost_low_latency input exceeds configured token or alignment limits"
            )
        _validate_prerouted_inputs(
            act,
            act.num_tokens,
            self.config.routing.top_k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32, torch.bfloat16),
            require_contiguous=True,
        )
        if self.config.quant.weight is QuantFormat.BF16:
            return self._pack_bf16(act, weights)
        q, sf = act.hidden_states_q, act.hidden_states_scale
        t, h = q.shape
        e, k = self.config.routing.num_experts, self.config.routing.top_k
        i = self.config.experts.intermediate_size
        p = (i + 127) // 128 * 128
        if (
            q.device != self.device
            or not q.is_cuda
            or q.dtype != torch.float8_e4m3fn
            or not q.is_contiguous()
        ):
            raise ValueError(
                "hidden_states_q must be contiguous CUDA E4M3 on the runner device"
            )
        if (
            sf is None
            or sf.shape != (t, h // 32)
            or not sf.is_contiguous()
            or sf.dtype not in (torch.uint8, torch.float8_e4m3fn, torch.float8_e8m0fnu)
        ):
            raise ValueError(
                "hidden_states_scale must be contiguous linear UE8M0 bytes [T,H/32]"
            )
        v = weights.get_view(self.backend_key)
        wd = (
            torch.uint8
            if self.config.quant.weight is QuantFormat.MXFP4
            else torch.float8_e4m3fn
        )
        pack = 2 if wd is torch.uint8 else 1
        specs = (
            ("w1_weight", (e, 2 * p, h // pack), wd),
            ("w1_weight_sf", (e, 2 * p // 128, h // 128, 32, 4, 4), torch.uint8),
            ("w2_weight", (e, h, p // pack), wd),
            ("w2_weight_sf", (e, h // 128, p // 128, 32, 4, 4), torch.uint8),
        )
        for name, shape, dtype in specs:
            value = v[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or value.device != q.device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must have shape {shape}, dtype {dtype}, and contiguous storage on {q.device}"
                )
            if value.data_ptr() % 16:
                raise ValueError(f"{name} must be 16-byte aligned")
        if q.data_ptr() % 16:
            raise ValueError("hidden_states_q must be 16-byte aligned")

        # Workspaces belong to this call. No pointer, stream, or writable storage
        # is shared by the runner/compile cache across concurrent executions.
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=q.device)

        packed_routing = act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        # Finalized output can narrow inside the existing combine kernel.
        # Raw output still exposes BF16 weights under the packed contract.
        narrow_in_combine = packed_routing and self.config.finalize.do_finalize
        routing_weights = (
            act.topk_weights.to(torch.bfloat16)
            if packed_routing and not narrow_in_combine
            else act.topk_weights
        )
        return [
            q,
            sf.view(torch.uint8),
            act.topk_ids,
            routing_weights,
            *(v[name] for name, _, _ in specs),
            empty(t * 128 * (h // 32), torch.uint8),
            empty((t * k, 2 * p), torch.float32),
            empty((t * k, p), torch.float8_e4m3fn),
            empty(t * k * 128 * (p // 32), torch.uint8),
            empty((t * k, h), torch.bfloat16),
            empty((t, h), torch.bfloat16),
            narrow_in_combine,
        ]

    def _pack_bf16(self, act, weights):
        x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
        t, h = x.shape
        e, k = self.config.routing.num_experts, self.config.routing.top_k
        p = (self.config.experts.intermediate_size + 127) // 128 * 128
        branches = 2 if self.config.activation.is_gated else 1
        if act.hidden_states_scale is not None:
            raise ValueError("BF16 Frost does not consume activation scales")
        if (
            not x.is_cuda
            or x.device != self.device
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.data_ptr() % 16
        ):
            raise ValueError(
                "BF16 input must be contiguous, 16-byte aligned and on the runner device"
            )
        view = weights.get_view(self.backend_key)
        for name, shape in (
            ("w1_weight", (e, branches * p, h)),
            ("w2_weight", (e, h, p)),
        ):
            value = view[name]
            if (
                value.shape != shape
                or value.dtype != torch.bfloat16
                or value.device != x.device
                or not value.is_contiguous()
                or value.data_ptr() % 16
            ):
                raise ValueError(
                    f"{name} must be aligned contiguous BF16 {shape} on {x.device}"
                )
        packed = act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        narrow = packed and self.config.finalize.do_finalize
        if packed and not narrow:
            scores = scores.to(torch.bfloat16)

        def empty(shape):
            return torch.empty(shape, device=x.device, dtype=torch.bfloat16)

        return [
            x,
            ids,
            scores,
            view["w1_weight"],
            view["w2_weight"],
            empty((t * k, p)),
            empty((t * k, h)),
            empty((t, h)),
            narrow,
        ]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        self._require_built()
        from .runtime import execute

        if tactic == -1:
            tactic = (
                ("ffma", 1) if self.config.quant.weight is QuantFormat.BF16 else 128
            )
        return execute(inputs, self.config, tactic)
