# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light adapter for the unified MoELayer API."""

from importlib.metadata import PackageNotFoundError, version

import torch
from packaging.version import Version

from ....autotuner import TuningConfig
from ...api import (
    FrostLowLatencyConfig,
    GeGLU,
    GeGLUTanh,
    SiTU,
    SwiGLUStep,
    QuantFormat,
    ReLU2,
    RoutingInputMode,
    SiLU,
    SwiGLU,
)
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
        (QuantFormat.FP16, QuantFormat.FP16),
        (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),
        (QuantFormat.FP8PerTensor, QuantFormat.BF16),
        (QuantFormat.FP8PerTensor, QuantFormat.FP16),
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
        (QuantFormat.NVFP4, QuantFormat.NVFP4),
        (QuantFormat.NVFP4, QuantFormat.BF16),
        (QuantFormat.NVFP4, QuantFormat.FP16),
        (QuantFormat.MXFP4, QuantFormat.BF16),
        (QuantFormat.MXFP4, QuantFormat.FP16),
        (QuantFormat.MXINT4, QuantFormat.BF16),
        (QuantFormat.MXINT4, QuantFormat.FP16),
    )
    supported_output_formats = (QuantFormat.BF16, QuantFormat.FP16)
    supported_activation_classes = tuple(ACTIVATIONS)
    supported_activation_classes_by_quant = {
        (QuantFormat.MXFP4, QuantFormat.MXFP8): tuple(ACTIVATIONS),
        (QuantFormat.MXFP8, QuantFormat.MXFP8): tuple(ACTIVATIONS),
        (QuantFormat.BF16, QuantFormat.BF16): tuple(ACTIVATIONS),
        (QuantFormat.FP16, QuantFormat.FP16): tuple(ACTIVATIONS),
        (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor): tuple(ACTIVATIONS),
        (QuantFormat.FP8PerTensor, QuantFormat.BF16): tuple(ACTIVATIONS),
        (QuantFormat.FP8PerTensor, QuantFormat.FP16): tuple(ACTIVATIONS),
        (QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8): tuple(ACTIVATIONS),
        (QuantFormat.NVFP4, QuantFormat.NVFP4): tuple(ACTIVATIONS),
        (QuantFormat.NVFP4, QuantFormat.BF16): tuple(ACTIVATIONS),
        (QuantFormat.NVFP4, QuantFormat.FP16): tuple(ACTIVATIONS),
        (QuantFormat.MXFP4, QuantFormat.BF16): tuple(ACTIVATIONS),
        (QuantFormat.MXFP4, QuantFormat.FP16): tuple(ACTIVATIONS),
        (QuantFormat.MXINT4, QuantFormat.BF16): tuple(ACTIVATIONS),
        (QuantFormat.MXINT4, QuantFormat.FP16): tuple(ACTIVATIONS),
    }
    supports_expert_parallelism = False
    # This path honors an explicit disabled recipe; 4over6 is rejected below.
    supports_nvfp4_4over6 = True

    @classmethod
    def supports_quant(cls, quant):
        return super().supports_quant(quant) and (
            quant.output is QuantFormat.BF16
            or quant.weight in (QuantFormat.BF16, QuantFormat.FP16)
            or (
                quant.weight
                in (
                    QuantFormat.NVFP4,
                    QuantFormat.MXFP4,
                    QuantFormat.MXINT4,
                    QuantFormat.FP8PerTensor,
                )
                and quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
            )
        )

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self._fp4_weight_only = config.quant.activation in (
            QuantFormat.BF16,
            QuantFormat.FP16,
        ) and config.quant.weight in (
            QuantFormat.NVFP4,
            QuantFormat.MXFP4,
            QuantFormat.MXINT4,
        )
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
        if not self.supports_quant(self.config.quant):
            raise NotImplementedError(
                "Frost quantized-activation paths require BF16 output"
            )
        try:
            installed = Version(version("nvidia-cutlass-dsl"))
        except PackageNotFoundError:
            installed = None
        if installed is None or installed < Version("4.8"):
            raise NotImplementedError(
                "frost_low_latency requires nvidia-cutlass-dsl >= 4.8"
            )
        from ....utils import get_compute_capability

        arch = get_compute_capability(self.device)
        self._supports_tc = arch in ((10, 0), (10, 3), (10, 7))
        if not self._supports_tc and arch not in ((8, 0), (9, 0), (12, 0)):
            raise NotImplementedError(
                "frost_low_latency requires SM80, SM90, SM100, SM103, SM107 or SM120"
            )
        if arch == (8, 0) and self.config.quant.weight not in (
            QuantFormat.BF16,
            QuantFormat.FP16,
        ):
            raise NotImplementedError(
                "Frost SM80 currently supports BF16/FP16 inputs and weights only"
            )
        if not self._supports_tc and self.config.routing.top_k > 32:
            raise NotImplementedError("Frost FMA requires top_k <= 32")
        if (
            self.config.quant.activation is QuantFormat.MXFP8
            and self.config.routing.top_k > self.config.routing.num_experts
            and self.config.routing.top_k > 32
        ):
            raise NotImplementedError(
                "Frost MX with top_k > num_experts requires top_k <= 32"
            )
        if (
            arch == (9, 0)
            and not self._fp4_weight_only
            and self.config.quant.weight
            not in (
                QuantFormat.BF16,
                QuantFormat.FP16,
                QuantFormat.FP8PerTensor,
                QuantFormat.DeepSeekFp8,
            )
        ):
            raise NotImplementedError(
                "Frost SM90 supports BF16, FP16, per-tensor FP8, DeepSeek FP8 and packed-four-bit-weight/16-bit-activation FMA only"
            )
        name, _ = lower_activation(self.config.activation)
        if self.config.quant.weight in (
            QuantFormat.BF16,
            QuantFormat.FP16,
            QuantFormat.FP8PerTensor,
            QuantFormat.DeepSeekFp8,
        ):
            if self.config.routing.top_k > 32:
                raise NotImplementedError(
                    "Frost dense/per-tensor FMA requires top_k <= 32"
                )
        if (
            self.config.quant.weight is QuantFormat.DeepSeekFp8
            and self.config.experts.intermediate_size % 128
        ):
            raise NotImplementedError("DeepSeek Frost requires I divisible by 128")
        if self.config.quant.weight is QuantFormat.NVFP4 or self._fp4_weight_only:
            if self.config.routing.top_k > 32:
                raise NotImplementedError("Frost NVFP4 requires top_k <= 32")
            if not self._fp4_weight_only:
                self._check_nvfp4_recipe()
            if (
                not self._fp4_weight_only
                and self.config.experts.intermediate_size % 128
            ):
                raise NotImplementedError(
                    "Frost NVFP4 requires top_k <= 32 and I divisible by 128"
                )
        if name in ("geglu", "gelu"):
            from cutlass import cute

            if not callable(getattr(cute.math, "erf", None)):
                raise NotImplementedError("Frost GELU/GeGLU requires cute.math.erf")
        if self.config.quant.swizzled_scale_factors:
            raise NotImplementedError(
                "frost_low_latency requires linear activation scales"
            )
        if (
            self.config.experts.intermediate_size <= 0
            or self.config.experts.intermediate_size % 32
        ):
            raise NotImplementedError(
                "frost_low_latency requires intermediate_size divisible by 32"
            )

    @property
    def _supports_mx_tc(self):
        # MX Tensor Core Geometry requires top_k <= E; repeated routes remain
        # valid through FMA (up to 32 slots), including when top_k exceeds E.
        return (
            self._supports_tc
            and self.config.routing.top_k <= self.config.routing.num_experts
        )

    def _build(self):
        pass

    def _check_nvfp4_recipe(self):
        from ....quantization.nvfp4_quantization_utils import resolve_nvfp4_4over6

        if resolve_nvfp4_4over6(self.config.quant.nvfp4_4over6) is not None:
            raise NotImplementedError(
                "Frost NVFP4 does not implement the 4over6 intermediate quantizer"
            )

    def _accepts_shape(self, act):
        if self.config.quant.weight is QuantFormat.NVFP4 and not self._fp4_weight_only:
            return (
                act.hidden_states_q.ndim == 2
                and 1 <= act.num_tokens <= self.options.max_num_tokens
                and act.hidden_states_q.shape[1] >= 256
                and act.hidden_states_q.shape[1] % 256 == 0
            )
        return (
            act.hidden_states_q.ndim == 2
            and 1 <= act.num_tokens <= self.options.max_num_tokens
            and act.hidden_states_q.shape[1] >= 128
            and act.hidden_states_q.shape[1] % 128 == 0
        )

    def _cache_key_extras(self):
        return super()._cache_key_extras() + (
            # Do not reuse winners selected with the deferred individual-call timer.
            "fp8-grouped-tc-w8-weight-first-v6"
            if self.config.quant.weight is QuantFormat.FP8PerTensor
            and self.config.quant.activation is QuantFormat.FP8PerTensor
            else "block-fp8-grouped-tc-w8-batch-v3"
            if self.config.quant.weight is QuantFormat.DeepSeekFp8
            else "mxint4-load-width-v1"
            if self.config.quant.weight is QuantFormat.MXINT4
            else (
                "fp4-route-reuse-load-width-lut-fp16-v8"
                if self._fp4_weight_only
                else (
                    "w8a16-grouped-tc-hopper-k256-v22"
                    if self.config.quant.weight is QuantFormat.FP8PerTensor
                    and self.config.quant.activation
                    in (QuantFormat.BF16, QuantFormat.FP16)
                    else (
                        "bf16-grouped-tc-w8-weight-first-v16"
                        if self.config.quant.weight is QuantFormat.BF16
                        else (
                            "fp16-grouped-tc-w8-weight-first-v14"
                            if self.config.quant.weight is QuantFormat.FP16
                            else (
                                "mx-tc-contracts-v6"
                                if self.config.quant.activation is QuantFormat.MXFP8
                                else "unified-fma-tc-standard-timing-v5"
                            )
                        )
                    )
                )
            ),
            self.options.max_num_tokens,
            self.config.execution.enable_pdl,
        )

    def get_valid_tactics(self, inputs, profile):
        self._require_built()
        if self.config.quant.weight is QuantFormat.NVFP4 and not self._fp4_weight_only:
            return [("nvfp4_ffma", width) for width in (2, 4)]
        from ....utils import get_compute_capability

        capability = get_compute_capability(self.device)
        hopper_fp4 = (
            self._fp4_weight_only
            and self.config.quant.weight in (QuantFormat.NVFP4, QuantFormat.MXFP4)
            and capability == (9, 0)
        )
        if (
            self._fp4_weight_only
            and (
                self.config.quant.activation is QuantFormat.BF16
                or self.config.quant.weight in (QuantFormat.NVFP4, QuantFormat.MXFP4)
            )
            and (
                capability in ((10, 3), (12, 0))
                or hopper_fp4
                or (
                    capability == (10, 0)
                    and self.config.quant.activation is QuantFormat.FP16
                    and self.config.quant.weight
                    in (QuantFormat.NVFP4, QuantFormat.MXFP4)
                )
            )
        ):
            columns = (1, 2, 4, 8)
            tactics = [("ffma", a, b) for a in columns for b in columns] + [
                ("ffma", a, b, v1, v2)
                for a in columns
                for b in columns
                for v1, v2 in ((4, 8), (8, 4), (8, 8))
            ]
            if hopper_fp4 and inputs[0].shape[0] >= 2:
                tactics += [
                    ("ffma", a, b, group)
                    for group in (2, 4)
                    for a in (1, 2, 4)
                    for b in columns
                ]
            return tactics
        if (
            self.config.quant.weight is QuantFormat.FP8PerTensor
            and self.config.quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
            and self.config.quant.output is self.config.quant.activation
            and capability == (9, 0)
            and (
                inputs[0].shape == (8, 2048)
                or (
                    inputs[0].shape == (4, 2048)
                    and self.config.quant.activation is QuantFormat.BF16
                    and self.config.activation == GeGLU()
                )
            )
            and self.config.experts.intermediate_size == 512
            and self.config.routing.top_k == 10
            and (
                self.config.activation in (SwiGLU(), GeGLU())
                or (
                    self.config.quant.activation is QuantFormat.BF16
                    and self.config.activation in (ReLU2(), SiLU())
                )
            )
            and self.config.finalize.do_finalize
        ):
            # Reused experts amortize typed fragment conversion. Keep FMA for
            # sparse overlap, and both group sizes for warm/cold variation.
            tactics = [("ffma", a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)]
            if (
                self.config.quant.activation is QuantFormat.BF16
                and self.config.activation == GeGLU()
            ):
                # Wider K tiles amortize producer synchronization for reused
                # experts. Keep the existing T8 K128 choice for tuning.
                tokens = inputs[0].shape[0]
                tactics.append(("warp_mma", tokens, 2, 256))
                if tokens == 4:
                    return tactics
            return tactics + [
                ("warp_mma", group, 2, 128)
                for group in (
                    (4, 8) if self.config.quant.activation is QuantFormat.FP16 else (8,)
                )
            ]
        if (
            self.config.quant.weight is QuantFormat.FP8PerTensor
            and self.config.quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
            and self.config.quant.output is self.config.quant.activation
            and capability == (10, 3)
            and (
                (
                    inputs[0].shape in ((4, 2048), (8, 2048))
                    and self.config.experts.intermediate_size == 512
                    and self.config.routing.top_k == 10
                    and (
                        self.config.activation in (SwiGLU(), GeGLU())
                        or (
                            self.config.quant.activation is QuantFormat.FP16
                            and inputs[0].shape[0] == 8
                            and self.config.activation in (ReLU2(), SiLU())
                        )
                    )
                )
                or (
                    self.config.quant.activation is QuantFormat.BF16
                    and self.config.activation == SwiGLU()
                    and (
                        *inputs[0].shape,
                        self.config.experts.intermediate_size,
                        self.config.routing.top_k,
                    )
                    == (8, 7168, 2048, 8)
                )
            )
            and self.config.finalize.do_finalize
        ):
            # High expert reuse amortizes compact-weight fragment conversion;
            # partial/disjoint routing still needs all original FMA choices.
            tactics = [("ffma", a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)]
            if inputs[0].shape[0] == 4:
                stages = (
                    (2, 4) if self.config.quant.activation is QuantFormat.BF16 else (2,)
                )
                return tactics + [("warp_mma", 4, stage, 128) for stage in stages]
            return tactics + [("warp_mma", 8, 2, 128)]
        if (
            self.config.quant.weight is QuantFormat.FP8PerTensor
            and self.config.quant.activation in (QuantFormat.BF16, QuantFormat.FP16)
            and capability == (12, 0)
            and inputs[0].shape[0] >= 4
        ):
            # T=2 warm reuse gains did not offset its cold-cache regression.
            tactics = [("ffma", a, b) for a in (1, 2, 4, 8) for b in (1, 2, 4, 8)] + [
                ("ffma", a, b, group)
                for a in (1, 2, 4)
                for b in (1, 2, 4, 8)
                for group in (2, 4)
            ]
            if (
                self.config.quant.output is self.config.quant.activation
                and (
                    (
                        inputs[0].shape == (8, 2048)
                        and self.config.experts.intermediate_size == 512
                        and self.config.routing.top_k == 10
                        and (
                            self.config.activation in (SwiGLU(), GeGLU())
                            or (
                                self.config.quant.activation is QuantFormat.FP16
                                and self.config.activation in (ReLU2(), SiLU())
                            )
                        )
                    )
                    or (
                        self.config.activation == SwiGLU()
                        and (
                            *inputs[0].shape,
                            self.config.experts.intermediate_size,
                            self.config.routing.top_k,
                        )
                        == (8, 7168, 2048, 8)
                    )
                )
                and self.config.finalize.do_finalize
            ):
                # Shared T8 routes amortize fragment conversion. Preserve all
                # FMA choices for colder or less-overlapping expert routing.
                tactics.append(("warp_mma", 8, 2, 128))
            return tactics
        if self._fp4_weight_only or self.config.quant.weight in (
            QuantFormat.BF16,
            QuantFormat.FP16,
            QuantFormat.FP8PerTensor,
            QuantFormat.DeepSeekFp8,
        ):
            tactics = [
                ("ffma", first, second)
                for first in (1, 2, 4, 8)
                for second in (1, 2, 4, 8)
            ]
            if (
                capability in ((9, 0), (12, 0))
                and self.config.quant.weight is QuantFormat.FP8PerTensor
                and self.config.quant.activation is QuantFormat.FP8PerTensor
                and self.config.quant.output is QuantFormat.BF16
                and inputs[0].shape[0] in (4, 8)
                and inputs[0].shape[1] == 2048
                and self.config.experts.intermediate_size == 512
                and self.config.routing.top_k == 10
                and (
                    self.config.activation in (SwiGLU(), SiTU())
                    or (
                        self.config.activation == ReLU2()
                        and (capability == (12, 0) or inputs[0].shape[0] == 8)
                    )
                )
                and self.config.finalize.do_finalize
            ):
                # Shared routes amortize TC setup. Preserve every FMA choice:
                # cold/disjoint routes can still favor the scalar path.
                if capability == (9, 0):
                    tactics.append(("wgmma", 2, 4, 128))
                else:
                    tactics += [("warp_mma", group, 2, 128) for group in (2, 4)]
                    if self.config.activation in (SwiGLU(), SiTU()) or (
                        inputs[0].shape[0] == 8 and self.config.activation == ReLU2()
                    ):
                        # Retain the original schedule for cache/reuse controls;
                        # serial weight-first is an additional tuning choice.
                        tactics += [
                            ("warp_mma", group, 2, 128, True) for group in (2, 4)
                        ]
            if (
                capability in ((9, 0), (12, 0))
                and self.config.quant.weight is QuantFormat.DeepSeekFp8
                and self.config.quant.activation is QuantFormat.DeepSeekFp8
                and self.config.quant.output is QuantFormat.BF16
                and inputs[0].shape[0] in (4, 8)
                and inputs[0].shape[1] == 2048
                and self.config.experts.intermediate_size == 512
                and self.config.routing.top_k == 10
                and (
                    self.config.activation == SwiGLU()
                    or (
                        inputs[0].shape[0] == 8
                        and self.config.activation in (ReLU2(), SiLU())
                    )
                )
                and self.config.finalize.do_finalize
            ):
                # Weight reuse can amortize block-scale promotion and TC setup.
                # Keep all FMA choices for cold and disjoint-route workloads.
                if capability == (9, 0):
                    groups = (2, 4) if self.config.activation == SwiGLU() else (2,)
                    tactics += [("wgmma", group, 4, 128) for group in groups]
                else:
                    tactics += [("warp_mma", group, 2, 128) for group in (2, 4)]
                    if inputs[0].shape[0] == 4:
                        tactics.append(("warp_mma", 4, 4, 128))
            if (
                capability in ((8, 0), (9, 0), (12, 0))
                and self.config.quant.weight in (QuantFormat.BF16, QuantFormat.FP16)
                and self.config.quant.activation is self.config.quant.weight
                and self.config.quant.output is self.config.quant.weight
                and inputs[0].shape[0] in (4, 8)
                and (
                    (
                        capability == (8, 0)
                        and (
                            (
                                (
                                    inputs[0].shape[1],
                                    self.config.experts.intermediate_size,
                                    self.config.routing.top_k,
                                )
                                == (2048, 512, 10)
                                and self.config.activation in (SwiGLU(), ReLU2())
                            )
                            or (
                                (
                                    inputs[0].shape[1],
                                    self.config.experts.intermediate_size,
                                    self.config.routing.top_k,
                                )
                                == (7168, 2048, 8)
                                and self.config.quant.weight is QuantFormat.BF16
                                and self.config.activation == SwiGLU()
                            )
                        )
                    )
                    or (
                        capability in ((9, 0), (12, 0))
                        and (
                            inputs[0].shape[1],
                            self.config.experts.intermediate_size,
                            self.config.routing.top_k,
                        )
                        == (2048, 512, 10)
                        and (
                            self.config.activation in (SwiGLU(), GeGLU())
                            or (
                                capability == (12, 0)
                                and self.config.activation
                                in (GeGLUTanh(), SiTU(), SwiGLUStep())
                            )
                            or (
                                self.config.activation in (ReLU2(), SiLU())
                                and (capability == (12, 0) or inputs[0].shape[0] == 8)
                            )
                        )
                    )
                )
                and self.config.finalize.do_finalize
            ):
                # Qualified reuse cases benefit, while disjoint routes favor FMA.
                # Offer both to tuning; keep the untuned default unchanged.
                if capability == (8, 0):
                    choice = (
                        (8, 2, 64)
                        if inputs[0].shape[1] == 7168
                        else ((4, 4, 128) if inputs[0].shape[0] == 4 else (4, 2, 64))
                    )
                    tactics.append(("ampere_mma", *choice))
                elif capability == (9, 0):
                    if self.config.activation == GeGLU():
                        tile_k = 128 if inputs[0].shape[0] == 4 else 64
                        tactics.append(("wgmma", 2, 4, tile_k))
                    elif (
                        self.config.activation in (ReLU2(), SiLU())
                        or self.config.quant.weight is QuantFormat.FP16
                    ):
                        tiles = (
                            (128,)
                            if self.config.activation in (ReLU2(), SiLU())
                            else ((64, 128) if inputs[0].shape[0] == 4 else (64,))
                        )
                        tactics += [("wgmma", 2, 4, tile_k) for tile_k in tiles]
                    else:
                        tile_k = 128 if inputs[0].shape[0] == 4 else 64
                        tactics += [
                            ("wgmma", 2, stages, tile_k) for stages in (2, 4, 8)
                        ]
                else:
                    choices = (
                        ((4, 4, 64), (2, 2, 128))
                        if inputs[0].shape[0] == 4
                        else ((8, 2, 128), (4, 2, 64), (2, 2, 64), (2, 2, 128))
                    )
                    tactics += [("warp_mma", *choice) for choice in choices]
                    if inputs[0].shape[0] == 8 and self.config.activation == SwiGLU():
                        # Keep both weight/activation issue orders available to
                        # tuning; cold and cache-resident weights prefer different
                        # schedules. Untuned selection remains the original FMA.
                        tactics += [("warp_mma", *choice, True) for choice in choices]
            return tactics
        tactics = (
            [k for k in (128, 256) if inputs[0].shape[1] % k == 0]
            if self._supports_mx_tc
            else []
        )
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
        if self.config.quant.weight is QuantFormat.FP8PerTensor:
            from .fp8_per_tensor import pack_inputs

            return pack_inputs(act, weights, self.config, self.device)
        if self._fp4_weight_only:
            from .fp4_weight_only import pack_inputs

            return pack_inputs(act, weights, self.config, self.device)
        if self.config.quant.weight is QuantFormat.DeepSeekFp8:
            from .deepseek_fp8 import pack_inputs

            return pack_inputs(act, weights, self.config, self.device)
        if self.config.quant.weight in (QuantFormat.BF16, QuantFormat.FP16):
            return self._pack_float16(act, weights)
        if self.config.quant.weight is QuantFormat.NVFP4:
            self._check_nvfp4_recipe()
            from .nvfp4 import pack_inputs

            return pack_inputs(
                act, weights.get_view(self.backend_key), self.config, self.device
            )
        q, sf = act.hidden_states_q, act.hidden_states_scale
        t, h = q.shape
        e, k = self.config.routing.num_experts, self.config.routing.top_k
        i = self.config.experts.intermediate_size
        p = (i + 127) // 128 * 128
        branches = 2 if self.config.activation.is_gated else 1
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
            ("w1_weight", (e, branches * p, h // pack), wd),
            ("w1_weight_sf", (e, branches * p // 128, h // 128, 32, 4, 4), torch.uint8),
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
            # Packed E2M1 TMA copies require a 32-byte base. MXFP8 values and
            # the byte-encoded scale tensors retain their 16-byte contract.
            alignment = 32 if wd is torch.uint8 and name.endswith("_weight") else 16
            if value.data_ptr() % alignment:
                raise ValueError(f"{name} must be {alignment}-byte aligned")
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
            empty((t * k, branches * p), torch.float32),
            empty((t * k, p), torch.float8_e4m3fn),
            empty(t * k * 128 * (p // 32), torch.uint8),
            empty((t * k, h), torch.bfloat16),
            empty((t, h), torch.bfloat16),
            narrow_in_combine,
        ]

    def _pack_float16(self, act, weights):
        x, ids, scores = act.hidden_states_q, act.topk_ids, act.topk_weights
        dtype_name = self.config.quant.weight.name
        dtype = (
            torch.bfloat16
            if self.config.quant.weight is QuantFormat.BF16
            else torch.float16
        )
        output_dtype = (
            torch.bfloat16
            if self.config.quant.output is QuantFormat.BF16
            else torch.float16
        )
        t, h = x.shape
        e, k = self.config.routing.num_experts, self.config.routing.top_k
        p = (self.config.experts.intermediate_size + 127) // 128 * 128
        branches = 2 if self.config.activation.is_gated else 1
        if act.hidden_states_scale is not None:
            raise ValueError("BF16/FP16 Frost does not consume activation scales")
        if (
            not x.is_cuda
            or x.device != self.device
            or x.dtype != dtype
            or not x.is_contiguous()
            or x.data_ptr() % 16
        ):
            raise ValueError(
                f"{dtype_name} input must be contiguous, 16-byte aligned and on the runner device"
            )
        view = weights.get_view(self.backend_key)
        for name, shape in (
            ("w1_weight", (e, branches * p, h)),
            ("w2_weight", (e, h, p)),
        ):
            value = view[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or value.device != x.device
                or not value.is_contiguous()
                or value.data_ptr() % 16
            ):
                raise ValueError(
                    f"{name} must be aligned contiguous {dtype_name} {shape} on {x.device}"
                )
        packed = act.routing_input_mode == RoutingInputMode.PackedPrecomputed
        narrow = packed and self.config.finalize.do_finalize
        if packed and not narrow:
            scores = scores.to(torch.bfloat16)

        def empty(shape, element_type):
            return torch.empty(shape, device=x.device, dtype=element_type)

        return [
            x,
            ids,
            scores,
            view["w1_weight"],
            view["w2_weight"],
            empty((t * k, p), dtype),
            empty((t * k, h), output_dtype),
            empty((t, h), output_dtype),
            narrow,
        ]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        self._require_built()
        from .runtime import execute

        if tactic == -1:
            if self._fp4_weight_only:
                tactic = ("ffma", 1, 4)
            elif self.config.quant.weight is QuantFormat.NVFP4:
                tactic = ("nvfp4_ffma", 2)
            else:
                tactic = (
                    ("ffma", 1)
                    if self.config.quant.weight
                    in (
                        QuantFormat.BF16,
                        QuantFormat.FP16,
                        QuantFormat.FP8PerTensor,
                        QuantFormat.DeepSeekFp8,
                    )
                    or not self._supports_mx_tc
                    else 128
                )
        if not self._supports_mx_tc and not isinstance(tactic, tuple):
            raise ValueError(
                "Frost MX Tensor Core tactics require SM100, SM103 or SM107 "
                "and top_k <= num_experts"
            )
        return execute(inputs, self.config, tactic)
