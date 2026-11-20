from dataclasses import asdict
from functools import partial
from typing import ClassVar
import warnings

from ...api import ActivationConfig

import torch

from ....autotuner import DynamicTensorSpec, TuningConfig
from ...api import (
    ClampedReLU2,
    ColonelsBf16Config,
    ColonelsFp8BlockConfig,
    QuantFormat,
    ReLU2,
    RoutingInputMode,
    SwiGLU,
)
from ...runners import MoERunner, _validate_prerouted_inputs
from ...utils import make_hybrid_bucket_mapper, map_to_hybrid_bucket
from .activations import activation_kind
from .tactics import ColonelsTactic, get_colonels_tactics


class _Inputs(list):
    def __init__(self, tensors, tuning_config):
        super().__init__(tensors)
        self.tuning_config = tuning_config


class ColonelsRunner(MoERunner):
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    )
    supports_fused_shared_experts = True
    _backend_config: ClassVar[type]
    _dtype: ClassVar[torch.dtype]
    _tuning_input_idx: ClassVar[tuple[int, ...]] = (0, 1, 2, 3)
    _tuning_dim_idx: ClassVar[tuple[int, ...]] = (0, 0, 0, 0)

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability(self.device)
        self.arch = major * 10 + minor
        self.kind = activation_kind(config.activation)
        self.local_experts = (
            config.experts.local_num_experts or config.routing.num_experts
        )
        self.shared_experts = config.experts.num_fused_shared_experts
        self.total_experts = self.local_experts + self.shared_experts
        self.top_k = config.routing.top_k + self.shared_experts
        backend = config.backend.candidates[0]
        self.automatic_tactics = not backend.tactics
        self.tactics = backend.tactics or self._get_tactics()
        self._workspaces = {}
        self._scratch = {}
        self._valid = {}
        self._default_tactics = {}

    def _get_tactics(self):
        raise NotImplementedError

    @property
    def _blocked_weights(self):
        return False

    def _pack_scales(self, act, view):
        raise NotImplementedError

    def _launch_scales(self, x_scale, w1_scale, w2_scale):
        raise NotImplementedError

    def _check_support(self):
        super()._check_support()
        if not self._backend_config.supported(self.arch):
            raise NotImplementedError(
                f"{self.backend_key} does not support SM{self.arch}"
            )
        if not self.config.finalize.do_finalize:
            raise NotImplementedError(
                "Colonels kernels always finalize the weighted output"
            )
        if self.config.execution.enable_pdl is True:
            warnings.warn(
                "Colonels ignores enable_pdl=True and uses standard kernel launches.",
                stacklevel=2,
            )
        if self.config.quant.swizzled_scale_factors:
            raise NotImplementedError("Colonels requires linear block scales")
        if (
            min(
                self.config.routing.top_k,
                self.config.routing.num_experts,
                self.config.experts.intermediate_size,
            )
            <= 0
        ):
            raise ValueError("Routing and expert dimensions must be positive")
        if self.config.routing.top_k > self.config.routing.num_experts:
            raise ValueError("top_k must not exceed the number of routed experts")
        if self.top_k > 32 or self.total_experts > 2048:
            raise NotImplementedError(
                "FlashInfer sorting requires top_k <= 32 and experts <= 2048"
            )
        if (
            self.local_experts <= 0
            or self.config.experts.local_expert_offset < 0
            or (
                self.local_experts + self.config.experts.local_expert_offset
                > self.config.routing.num_experts
            )
        ):
            raise ValueError("Invalid local expert shard")
        allowed = set(self._get_tactics())
        if any(t not in allowed for t in self.tactics):
            raise ValueError(
                "Colonels tactic is outside the architecture's kernel catalogue"
            )

    def _build(self):
        from . import runtime

        self.runtime = runtime
        self._empty = torch.empty(0, device=self.device)
        self._gemv_experts = torch.zeros(1, device=self.device, dtype=torch.int32)
        self._gemv_count = torch.zeros(1, device=self.device, dtype=torch.int32)

    def _accepts_shape(self, act):
        return any(
            t.supports_shape(
                act.hidden_states_q.shape[1],
                self.config.experts.intermediate_size,
                self.config.activation.is_gated,
            )
            for t in self.tactics
        )

    def _cache_key_extras(self):
        return super()._cache_key_extras() + (self.arch, tuple(self.tactics))

    def _workspace(self, num_tokens, hidden_size):
        # The sorter uses the current batch size, independently of buffer capacity.
        capacity = self.config.execution.tune_max_num_tokens
        key = (
            capacity,
            hidden_size,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if key not in self._workspaces:
            # GMEM scratch is independent of the token count. Reuse it across
            # shapes on one stream rather than retaining a copy for every batch.
            needs_scratch = any("gmem" in t.variant for t in self.tactics)
            stream = key[2]
            if needs_scratch and stream not in self._scratch:
                self._scratch[stream] = self.runtime.allocate_scratch(
                    self.config.experts.intermediate_size, self.device
                )
            self._workspaces[key] = self.runtime.Workspace(
                capacity,
                hidden_size,
                self.config.experts.intermediate_size,
                self.total_experts,
                self.top_k,
                {t.block_m for t in self.tactics if t.variant != "bf16_gemv"},
                self.device,
                needs_scratch,
                scratch=self._scratch.get(stream),
            )
        return self._workspaces[key]

    def _prepare_tuning_inputs(self, inputs, *, expert_map):
        m = inputs[1].shape[0]
        ids = torch.arange(
            m * self.config.routing.top_k, device=self.device, dtype=torch.int32
        ).reshape(m, self.config.routing.top_k)
        routed_k = self.config.routing.top_k
        inputs[2][:, :routed_k].copy_(ids % self.config.routing.num_experts)
        inputs[3][:, :routed_k].fill_(1.0 / routed_k)
        if inputs[2].shape[1] > routed_k:
            inputs[2][:, routed_k:].copy_(
                torch.arange(
                    self.config.routing.num_experts,
                    self.config.routing.num_experts + self.shared_experts,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            inputs[3][:, routed_k:].fill_(1.0)
        if expert_map is not None:
            inputs[9].copy_(expert_map)
        self._workspace(m, inputs[1].shape[1])
        return inputs

    def pack_inputs(self, act, weights):
        self._require_built()
        self._validate_pack_contract(act)
        x = act.hidden_states_q
        if x.device != self.device or x.ndim != 2 or not x.is_contiguous():
            raise ValueError(
                "Expected contiguous CUDA activations on the runner device"
            )
        if x.dtype != self._dtype:
            raise ValueError("Unexpected Colonels activation dtype")
        m, h = x.shape
        if m > self.config.execution.tune_max_num_tokens:
            raise ValueError("Token count exceeds tune_max_num_tokens")
        if act.topk_ids.ndim != 2:
            raise ValueError("Expected two-dimensional top-k IDs")
        routes = act.topk_ids.shape[1]
        if routes not in (self.config.routing.top_k, self.top_k):
            raise ValueError("Expected routed top_k or top_k + shared expert slots")
        _validate_prerouted_inputs(
            act,
            m,
            routes,
            self.backend_key,
            allowed_weights_dtypes=(
                (torch.float32, torch.bfloat16)
                if act.routing_input_mode is RoutingInputMode.UnpackedPrecomputed
                else (torch.float32,)
            ),
            require_contiguous=True,
        )
        v = weights.get_view(self.backend_key)
        if v.get("activation") != self.config.activation:
            raise ValueError("Prepared weights and layer activation must match")
        i = self.config.experts.intermediate_size
        if v.get("blocked_weights", False) != self._blocked_weights:
            raise ValueError("Prepare Colonels weights on the layer's target device")
        expected = (
            (self.total_experts, i * (2 if self.config.activation.is_gated else 1), h),
            (self.total_experts, h, i),
        )
        for t, shape in zip((v["w1"], v["w2"]), expected, strict=True):
            if (
                tuple(t.shape) != shape
                or t.dtype != x.dtype
                or t.device != self.device
                or not t.is_contiguous()
            ):
                raise ValueError(f"Invalid Colonels prepared weight; expected {shape}")
        expert_map = v.get("expert_map")
        if expert_map is not None:
            if (
                expert_map.shape != (self.config.routing.num_experts,)
                or expert_map.dtype != torch.int32
                or expert_map.device != self.device
                or not expert_map.is_contiguous()
                or v.get("expert_map_max", self.local_experts) >= self.local_experts
            ):
                raise ValueError(
                    "Invalid global-to-local expert_map in prepared weights"
                )
            if self.config.experts.local_expert_offset:
                raise ValueError(
                    "Use local_expert_offset=0 with an explicit expert_map"
                )
        scales = self._pack_scales(act, v)
        self._workspace(m, h)
        bucket = map_to_hybrid_bucket(m, self.config.execution.tune_max_num_tokens)
        specs = [
            DynamicTensorSpec(
                input_idx=self._tuning_input_idx,
                dim_idx=self._tuning_dim_idx,
                gen_tuning_buckets=(bucket,),
                map_to_tuning_buckets=make_hybrid_bucket_mapper(
                    self.config.execution.tune_max_num_tokens
                ),
            )
        ]
        tuning = TuningConfig(
            dynamic_tensor_specs=tuple(specs),
            use_cuda_graph=True,
            inputs_pre_hook=partial(self._prepare_tuning_inputs, expert_map=expert_map),
        )
        return _Inputs(
            [
                torch.empty((m, h), device=x.device, dtype=torch.bfloat16),
                x,
                act.topk_ids,
                act.topk_weights.float(),
                v["w1"],
                v["w2"],
                *scales,
                expert_map if expert_map is not None else self._empty,
            ],
            tuning,
        )

    def _routes(self, ids, weights, expert_map):
        offset = self.config.experts.local_expert_offset
        if not offset and not expert_map.numel() and not self.shared_experts:
            # The sorter and GEMV already discard IDs outside the local expert range.
            return ids, weights
        if expert_map.numel():
            valid = (ids >= 0) & (ids < expert_map.numel())
            local = expert_map[ids.clamp(0, expert_map.numel() - 1).long()]
            local = torch.where(valid, local, -1)
        else:
            local = ids - offset
        local = torch.where((local >= 0) & (local < self.local_experts), local, -1)
        if self.shared_experts and ids.shape[1] == self.config.routing.top_k:
            shared = torch.arange(
                self.local_experts,
                self.total_experts,
                device=ids.device,
                dtype=torch.int32,
            ).expand(ids.shape[0], -1)
            local = torch.cat((local, shared), dim=1)
            weights = torch.cat(
                (weights, torch.ones_like(shared, dtype=torch.float32)), dim=1
            )
        elif self.shared_experts:
            shared_id = ids - self.config.routing.num_experts
            local = torch.where(
                (shared_id >= 0) & (shared_id < self.shared_experts),
                shared_id + self.local_experts,
                local,
            )
        return local, weights

    def _run(self, inputs, tactic):
        with torch.cuda.device(self.device):
            return self._run_on_device(inputs, tactic)

    def _run_on_device(self, inputs, tactic):
        out, x, ids, weights, w1, w2, xs, s1, s2, expert_map = inputs
        if x.shape[0] == 0:
            out.zero_()
            return True
        workspace = self._workspace(*x.shape)
        ids, weights = self._routes(ids, weights, expert_map)
        routing = (
            (ids, self._gemv_experts, self._gemv_count)
            if tactic.variant == "bf16_gemv"
            else workspace.sort(ids, weights, self.total_experts, tactic.block_m)
        )
        xs, s1, s2 = self._launch_scales(xs, s1, s2)
        return self.runtime.launch(
            tactic,
            self.kind,
            x,
            w1,
            w2,
            routing,
            weights,
            out,
            scratch=workspace.scratch if "gmem" in tactic.variant else None,
            x_scale=xs,
            w1_scale=s1,
            w2_scale=s2,
            clamp=self.config.activation.limit
            if isinstance(self.config.activation, ClampedReLU2)
            else 0.5,
        )

    def _candidate_tactics(self, num_tokens, hidden_size):
        return (
            t
            for t in self.tactics
            if t.supports_shape(
                hidden_size,
                self.config.experts.intermediate_size,
                self.config.activation.is_gated,
            )
            and (
                t.variant != "bf16_gemv"
                or not self.automatic_tactics
                or num_tokens <= 16
            )
        )

    def get_valid_tactics(self, inputs, profile):
        key = tuple(inputs[1].shape)
        if key not in self._valid:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm Colonels before CUDA graph capture")
            candidates = list(self._candidate_tactics(*key))
            if key[0]:
                self.runtime.precompile_kernels(candidates, self.kind, self.arch)
            self._valid[key] = [
                tuple(asdict(t).values()) for t in candidates if self._run(inputs, t)
            ]
        return self._valid[key]

    def precompile_tactics(self, inputs, tactics, profile, **kwargs):
        # get_valid_tactics also checks runtime resource limits outside capture.
        # therefore we need to have it compiled there and at this point we just
        # return that compilatinon is done
        return True

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        self._require_built()
        if tactic == -1:
            num_tokens, hidden_size = inputs[1].shape
            # Empty batches do not test kernel resource limits; keep them separate.
            bucket = (
                map_to_hybrid_bucket(
                    num_tokens, self.config.execution.tune_max_num_tokens
                )
                if num_tokens
                else 0
            )
            key = (bucket, hidden_size)
            cached = self._default_tactics.get(key)
            if cached is not None:
                if self._run(inputs, cached):
                    return inputs[0]
                del self._default_tactics[key]
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm Colonels before CUDA graph capture")
            for candidate in self._candidate_tactics(num_tokens, hidden_size):
                if candidate != cached and self._run(inputs, candidate):
                    self._default_tactics[key] = candidate
                    return inputs[0]
            raise NotImplementedError(
                "No Colonels tactic supports this shape and device"
            )
        tactic = ColonelsTactic(*tactic)
        if not self._run(inputs, tactic):
            raise RuntimeError(f"Selected Colonels tactic cannot launch: {tactic}")
        return inputs[0]


class ColonelsRunnerBF16(ColonelsRunner):
    backend_key = "colonels_bf16"
    supported_quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
    supported_activation_classes: ClassVar[tuple[type[ActivationConfig], ...]] = (
        SwiGLU,
        ReLU2,
        ClampedReLU2,
    )
    _backend_config = ColonelsBf16Config
    _dtype = torch.bfloat16

    def _get_tactics(self):
        return get_colonels_tactics(self.arch)

    @property
    def _blocked_weights(self):
        return self.arch == 103

    def _pack_scales(self, act, view):
        if act.hidden_states_scale is not None:
            raise ValueError("BF16 activations must not have a scale tensor")
        return self._empty, self._empty, self._empty

    def _launch_scales(self, x_scale, w1_scale, w2_scale):
        return None, None, None


class ColonelsRunnerFP8(ColonelsRunner):
    backend_key = "colonels_fp8_block"
    supported_quant_variants = ((QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),)
    supported_activation_classes = (SwiGLU,)
    _backend_config = ColonelsFp8BlockConfig
    _dtype = torch.float8_e4m3fn
    _tuning_input_idx = (0, 1, 2, 3, 6)
    _tuning_dim_idx = (0, 0, 0, 0, 1)

    def _get_tactics(self):
        return get_colonels_tactics(self.arch, fp8=True)

    def _pack_scales(self, act, view):
        m, h = act.hidden_states_q.shape
        scale = act.hidden_states_scale
        if (
            scale is None
            or tuple(scale.shape) != (h // 128, m)
            or scale.dtype != torch.float32
            or scale.device != self.device
        ):
            raise ValueError(
                "Expected canonical FP8 scales float32 [hidden/128, tokens]"
            )
        return scale, view["w1_scale"], view["w2_scale"]

    def _launch_scales(self, x_scale, w1_scale, w2_scale):
        return x_scale.T.contiguous(), w1_scale, w2_scale
