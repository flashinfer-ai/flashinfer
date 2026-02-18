"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
from abc import ABC, abstractmethod
from typing import Dict
import torch
from cuda.bindings import runtime
from torch.nn import functional as F

from flashinfer import (
    ActivationType,
    RoutingMethodType,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp8_dequantize_host,
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.autotuner import autotune
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_bf16_moe,
    trtllm_mxint4_block_scale_moe,
)
from flashinfer.fused_moe.core import (
    get_w2_permute_indices_with_cache,
    _maybe_get_cached_w3_w1_permute_indices,
    Fp8QuantizationType,
)
from .utils import is_gated_activation, skip_checks, QuantMode


# Max num tokens to tune for trtllm-gen fused moe
TUNE_MAX_NUM_TOKENS = 4096


def check_cuda(err):
    """Unified CUDA error checking function used throughout the file."""
    if err != runtime.cudaError_t.cudaSuccess:
        error_name = runtime.cudaGetErrorName(err)
        error_string = runtime.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error: {error_name[1]}: {error_string[1]}")


class CUDAGraphMoE:
    """
    Simple CUDA Graph wrapper for MoE operations.

    The graph captures tensor references and automatically updates them during execution.

    Three core methods: capture(), launch(), cleanup()

    Usage:
        cuda_graph = CUDAGraphMoE(moe_impl, static_data, **config)
        cuda_graph.capture(hidden_states_sample, expert_logits=logits, routing_bias=bias)
        output = cuda_graph.launch(new_hidden_states)  # Repeat as needed
        cuda_graph.cleanup()
    """

    def __init__(self, moe_impl, static_data, **config):
        self.moe_impl = moe_impl
        self.static_data = static_data
        self.config = config
        self.enable_autotune = config.get("enable_autotune", True)
        self.graph = None
        self.graph_exec = None
        self.stream = None
        self.input_tensor = None
        self.output_tensor = None
        self.is_captured = False

    def capture(self, hidden_states_sample, **runtime_args):
        """Capture CUDA graph with the given sample input."""
        if self.is_captured:
            raise RuntimeError(
                "Graph already captured. Call cleanup() first to re-capture."
            )
        if not isinstance(self.moe_impl, FP4Moe):
            raise NotImplementedError(
                f"CUDA graph capture not yet implemented for {type(self.moe_impl)}"
            )

        # Create stream
        err, self.stream = runtime.cudaStreamCreate()
        check_cuda(err)

        # Get the raw stream pointer for PyTorch
        stream_ptr = int(self.stream)
        torch_stream = torch.cuda.ExternalStream(stream_ptr)

        # Store input tensor reference (will be updated in place during launch)
        self.input_tensor = hidden_states_sample.clone()

        # Warmup
        with torch.cuda.stream(torch_stream), autotune(self.enable_autotune):
            for _ in range(1):
                self._run_moe_computation(runtime_args)

        # Synchronize our stream after warmup
        err = runtime.cudaStreamSynchronize(self.stream)[0]
        check_cuda(err)

        # Begin capture
        err, self.graph = runtime.cudaGraphCreate(0)
        check_cuda(err)
        err = runtime.cudaStreamBeginCapture(
            self.stream, runtime.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
        )[0]
        check_cuda(err)

        try:
            # Capture computation on our stream
            with torch.cuda.stream(torch_stream):
                self.output_tensor = self._run_moe_computation(runtime_args)
            err, self.graph = runtime.cudaStreamEndCapture(self.stream)
            check_cuda(err)
            err, self.graph_exec = runtime.cudaGraphInstantiate(self.graph, 0)
            check_cuda(err)
            self.is_captured = True
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"CUDA graph capture failed: {e}") from e

    def launch(self, hidden_states_new):
        """Launch captured CUDA graph with new input."""
        if not self.is_captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        # Update input tensor in place
        self.input_tensor.copy_(hidden_states_new)

        # Launch graph
        err = runtime.cudaGraphLaunch(self.graph_exec, self.stream)[0]
        check_cuda(err)
        err = runtime.cudaStreamSynchronize(self.stream)[0]
        check_cuda(err)

        # Return output tensor (automatically updated by graph execution)
        return self.output_tensor

    def cleanup(self):
        """Clean up all CUDA graph resources."""
        if self.graph_exec is not None:
            err = runtime.cudaGraphExecDestroy(self.graph_exec)[0]
            check_cuda(err)
            self.graph_exec = None
        if self.graph is not None:
            err = runtime.cudaGraphDestroy(self.graph)[0]
            check_cuda(err)
            self.graph = None
        if self.stream is not None:
            err = runtime.cudaStreamDestroy(self.stream)[0]
            check_cuda(err)
            self.stream = None
        self.input_tensor = None
        self.output_tensor = None
        self.is_captured = False

    def _run_moe_computation(self, runtime_args):
        """Run the MoE computation."""
        input_quantized = self.moe_impl.quantize_inputs(
            self.input_tensor,
            self.config["hidden_states_scale_global"],
            is_swizzling=False,
        )

        output = trtllm_fp4_block_scale_moe(
            routing_logits=runtime_args["expert_logits"],
            routing_bias=runtime_args["routing_bias"],
            hidden_states=input_quantized["hidden_states"],
            hidden_states_scale=input_quantized["hidden_states_scale"],
            gemm1_weights=self.static_data["gemm1_weights_fp4_shuffled"],
            gemm1_weights_scale=self.static_data["gemm1_scales_fp4_shuffled"],
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self.static_data["gemm2_weights_fp4_shuffled"],
            gemm2_weights_scale=self.static_data["gemm2_scales_fp4_shuffled"],
            gemm2_bias=None,
            output1_scale_scalar=self.static_data["scale_c_fc1"],
            output1_scale_gate_scalar=self.static_data["scale_gate_fc1"],
            output2_scale_scalar=self.static_data["scale_c_fc2"],
            num_experts=self.config["num_experts"],
            top_k=self.config["top_k"],
            n_group=self.config["n_groups"],
            topk_group=self.config["top_k_groups"],
            intermediate_size=self.config["intermediate_size"],
            local_expert_offset=0,
            local_num_experts=self.config["num_experts"],
            routed_scaling_factor=self.config["routed_scaling"],
            routing_method_type=self.config["routing_method_type"],
            activation_type=self.config["activation_type"],
            do_finalize=True,
            tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
        )
        return output  # Extract tensor from tuple


# ====================================================================================
# Abstract Base Class for MoE Implementations
# ====================================================================================


class Moe(ABC):
    """Abstract base class for MoE implementations."""

    def __init__(self):
        self.name = self.__class__.__name__

    @property
    @abstractmethod
    def quant_mode(self) -> QuantMode:
        """Get the quantization mode of this MoE implementation."""
        pass

    @abstractmethod
    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize static weights and compute global scale factors (done offline)."""
        pass

    @abstractmethod
    def quantize_inputs(self, hidden_states, hidden_states_scale_global):
        """Quantize dynamic inputs/hidden states using pre-computed global scale (done at runtime)."""
        pass

    @abstractmethod
    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """
        Prepare quantized weights for kernel (done offline with weights).

        Args:
            args_dequant: Contains c_global_sf and other dequantization parameters
            args: Contains already quantized weights (gemm1_weights, gemm2_weights) and scales
            gemm1_weights_orig: Original unquantized FC1 weights (used by FP4 for re-quantization)
            gemm2_weights_orig: Original unquantized FC2 weights (used by FP4 for re-quantization)

        Note:
            - FP4 implementations use both original weights (for linear layout quantization)
              and args.gemm*_weights (for swizzled layout)
            - FP8 implementations typically only use args.gemm*_weights (already quantized)
        """
        pass

    @abstractmethod
    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        pass

    @abstractmethod
    def compute_reference(self, args):
        """Compute reference output using dequantized operations."""
        pass

    def compute_production(self, args_dequant, args, **kwargs):
        """Unified actual computation that delegates to implementation-specific methods."""
        return _compute_moe_actual_unified(self, args_dequant, args, **kwargs)

    @abstractmethod
    def get_tolerances(self):
        """Get accuracy tolerances for this quantization mode."""
        pass

    def __str__(self):
        return self.name


# ====================================================================================
# FP4 Quantization Implementation
# ====================================================================================


class FP4Moe(Moe):
    """
    FP4 NvFP4 / MxFP4 MoE implementation with block scaling.
    Args:
        is_mxfp4: Whether to use MxFP4 or NvFP4 weight quantization
            If True, the activation is quantized to MxFP8, else the activation is quantized to NvFP4
    """

    def __init__(self, quant_mode: QuantMode):
        super().__init__()
        self._quant_mode = quant_mode
        self.is_mxfp4 = (
            quant_mode == QuantMode.FP4_MXFP4_MXFP8
            or quant_mode == QuantMode.FP4_MXFP4_Bf16
        )
        self.sf_vec_size = 32 if self.is_mxfp4 else 16

    @property
    def quant_mode(self) -> QuantMode:
        return self._quant_mode

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP4 format and compute global scale factors."""
        num_experts = gemm1_weights.shape[0]
        # Compute global scale factor for hidden states (offline calibration)
        if self.quant_mode == QuantMode.FP4_NVFP4_NVFP4:
            # nvfp4 hidden states
            hidden_states_scale_global = calculate_fp4_global_scale_factor(
                hidden_states_sample,
                False,
            )
        else:
            # mxfp8 / bf16 hidden states
            hidden_states_scale_global = 1.0

        # Quantize the weights for FC1
        gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = (
            quant_fp4_batches(gemm1_weights, num_experts, self.is_mxfp4, True)
        )

        # Quantize the weights for FC2
        gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = (
            quant_fp4_batches(gemm2_weights, num_experts, self.is_mxfp4, True)
        )

        return {
            "hidden_states_scale_global": hidden_states_scale_global,
            "gemm1_weights": gemm1_weights_fp4_bytes,
            "gemm1_scales": gemm1_scales_fp4_bytes,
            "gemm1_scales_global": gemm1_scales_global,
            "gemm2_weights": gemm2_weights_fp4_bytes,
            "gemm2_scales": gemm2_scales_fp4_bytes,
            "gemm2_scales_global": gemm2_scales_global,
        }

    def quantize_inputs(
        self, hidden_states, hidden_states_scale_global, is_swizzling=True
    ):
        if self.quant_mode == QuantMode.FP4_MXFP4_MXFP8:
            """Quantize hidden states to MxFP8 format."""
            hidden_states_quant, hidden_states_scale = mxfp8_quantize(
                hidden_states, is_swizzling
            )
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            )
            return {
                "hidden_states": hidden_states_quant,
                "hidden_states_scale": hidden_states_scale,
            }
        elif self.quant_mode == QuantMode.FP4_NVFP4_NVFP4:
            """Quantize hidden states to NvFP4 format using pre-computed global scale."""
            (
                hidden_states_fp4_bytes,
                hidden_states_scale_fp4_bytes,
                _,
            ) = quant_fp4(
                hidden_states, hidden_states_scale_global, False, is_swizzling
            )
            hidden_states_scale_fp4_bytes = hidden_states_scale_fp4_bytes.view(
                torch.float8_e4m3fn
            ).reshape(*hidden_states.shape[:-1], -1)

            return {
                "hidden_states": hidden_states_fp4_bytes,
                "hidden_states_scale": hidden_states_scale_fp4_bytes,
            }
        else:  # bf16
            return {
                "hidden_states": hidden_states.to(torch.bfloat16),
                "hidden_states_scale": None,
            }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""
        use_ue8m0 = self.is_mxfp4
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Quantize weights with linear layout for kernels
        _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm1_weights_orig, num_experts, use_ue8m0, False
        )
        _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
            gemm2_weights_orig, num_experts, use_ue8m0, False
        )

        # Convert quantized weights to proper formats
        intermediate_size_factor = 2 if is_gated_activation(args.activation_type) else 1
        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, intermediate_size_factor * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts,
            intermediate_size_factor * intermediate_size,
            hidden_size // self.sf_vec_size,
        )  # fp8 scaling factors

        gemm2_weights_fp4 = args.gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // self.sf_vec_size
        )  # fp8 scaling factors

        # Using cached permute index calculation can speed up weights preprocessing
        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                is_gated_act_gemm=is_gated_activation(args.activation_type),
            )
            gemm1_weights_fp4_shuffled.append(
                gemm1_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
                is_gated_act_gemm=is_gated_activation(args.activation_type),
            )
            gemm1_scales_fp4_shuffled.append(
                block_scale_interleave(
                    gemm1_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_fp4_shuffled.append(
                block_scale_interleave(
                    gemm2_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(
                num_experts,
                intermediate_size_factor * intermediate_size,
                hidden_size // self.sf_vec_size,
            )
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // self.sf_vec_size)
        )

        # Calculate scaling factors that depend on weights
        if is_gated_activation(args.activation_type):
            scale_c_fc1 = (
                args_dequant.c_global_sf
                * (1.0 / args.gemm1_scales_global)
                * (1.0 / args.hidden_states_scale_global)
            )
        else:
            scale_c_fc1 = torch.full_like(
                args.gemm1_scales_global, args_dequant.c_global_sf
            )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
            1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (
            1.0 / args.gemm2_scales_global
        )

        return {
            "gemm1_weights_fp4_shuffled": gemm1_weights_fp4_shuffled,
            "gemm1_scales_fp4_shuffled": gemm1_scales_fp4_shuffled,
            "gemm2_weights_fp4_shuffled": gemm2_weights_fp4_shuffled,
            "gemm2_scales_fp4_shuffled": gemm2_scales_fp4_shuffled,
            "scale_c_fc1": scale_c_fc1,
            "scale_gate_fc1": scale_gate_fc1,
            "scale_c_fc2": scale_c_fc2,
        }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE using CUDA graph for maximum performance (create, capture, launch)."""
        # Extract runtime arguments
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        activation_type = kwargs["activation_type"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)

        # Create CUDA graph configuration
        config = {
            "hidden_states_scale_global": hidden_states_scale_global,
            "num_experts": num_experts,
            "top_k": top_k,
            "n_groups": n_groups,
            "top_k_groups": top_k_groups,
            "intermediate_size": intermediate_size,
            "routed_scaling": routed_scaling,
            "activation_type": activation_type,
            "routing_method_type": routing_method_type,
            "enable_autotune": enable_autotune,
        }

        runtime_args = {
            "expert_logits": expert_logits,
            "routing_bias": routing_bias,
        }

        # Create, capture and launch CUDA graph in one shot
        cuda_graph = CUDAGraphMoE(self, static_data, **config)
        try:
            cuda_graph.capture(hidden_states_orig, **runtime_args)
            output = cuda_graph.launch(hidden_states_orig)
            return output[0].to(torch.float)
        finally:
            cuda_graph.cleanup()

    def compute_reference(self, args):
        return run_moe_reference_fp4(args, self.quant_mode)

    def get_tolerances(self):
        """Get FP4-specific accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# MxInt4 Block Scale Quantization Implementation
# ====================================================================================


def mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    x_reshaped = x.reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0].to(torch.float32)
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0].to(torch.float32)
    x_max = x_max * 8.0 / 7.0
    amax = torch.where(x_max > -x_min, x_max, -x_min)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    )
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return x_int4.reshape(*x.shape[:-1], x.shape[-1] // 2), scales.reshape(
        -1, sf_vec_size
    )


class MxInt4BlockScaleMoe(Moe):
    """MxInt4 MoE implementation with block scaling (DeepSeek style)."""

    @property
    def quant_mode(self) -> QuantMode:
        return QuantMode.MXINT4_BF16_BF16

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to MxInt4 with block scaling."""
        num_experts = gemm1_weights.shape[0]
        intermediate_size = gemm1_weights.shape[1] // 2
        hidden_size = gemm1_weights.shape[
            2
        ]  # [num_experts, 2*intermediate_size, hidden_size]

        # Quantize weights to MxInt4
        sf_vec_size = 32
        gemm1_weights_int4, gemm1_scales = mxint4_quantize(gemm1_weights, sf_vec_size)
        gemm2_weights_int4, gemm2_scales = mxint4_quantize(gemm2_weights, sf_vec_size)
        gemm1_scales = gemm1_scales.to(torch.bfloat16).reshape(
            num_experts,
            2 * intermediate_size,
            hidden_size // sf_vec_size,
        )
        gemm2_scales = gemm2_scales.to(torch.bfloat16).reshape(
            num_experts, hidden_size, intermediate_size // sf_vec_size
        )
        return {
            "hidden_states_scale_global": None,
            "gemm1_weights": gemm1_weights_int4,
            "gemm2_weights": gemm2_weights_int4,
            "gemm1_scales": gemm1_scales,
            "gemm2_scales": gemm2_scales,
            "gemm1_scales_global": None,
            "gemm2_scales_global": None,
        }

    def quantize_inputs(self, hidden_states, *unused_args):
        """No scaling for hidden states."""
        return {
            "hidden_states": hidden_states.to(torch.bfloat16),
            "hidden_states_scale": None,
        }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""

        epilogue_tile_m = 128
        gemm1_weights_mxint4_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_mxint4_shuffled = []
        gemm2_scales_shuffled = []

        for i in range(num_experts):
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                args.gemm1_weights[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_shuffled = (
                args.gemm1_weights[i]
                .view(torch.uint8)[permute_indices.to(args.gemm1_weights.device)]
                .contiguous()
            )
            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                args.gemm1_scales[i].view(torch.bfloat16),
                epilogue_tile_m,
                num_elts_per_sf=32,
            )
            gemm1_scales_shuffled.append(
                block_scale_interleave(
                    args.gemm1_scales[i]
                    .view(torch.bfloat16)[
                        permute_sf_indices.to(args.gemm1_scales.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                args.gemm2_weights[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_shuffled = (
                args.gemm2_weights[i]
                .view(torch.uint8)[permute_indices.to(args.gemm2_weights.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                args.gemm2_scales[i].view(torch.bfloat16),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_shuffled.append(
                block_scale_interleave(
                    args.gemm2_scales[i]
                    .view(torch.bfloat16)[
                        permute_sf_indices.to(args.gemm2_scales.device)
                    ]
                    .contiguous()
                )
            )

            block_k = 128
            gemm1_weights_shuffled = convert_to_block_layout(
                gemm1_weights_shuffled, block_k
            )
            gemm2_weights_shuffled = convert_to_block_layout(
                gemm2_weights_shuffled.view(torch.uint8), block_k
            )

            gemm1_weights_mxint4_shuffled.append(gemm1_weights_shuffled)
            gemm2_weights_mxint4_shuffled.append(gemm2_weights_shuffled)

        gemm1_weights_mxint4_shuffled = torch.stack(gemm1_weights_mxint4_shuffled)
        gemm2_weights_mxint4_shuffled = torch.stack(gemm2_weights_mxint4_shuffled)
        gemm1_scales_shuffled = torch.stack(gemm1_scales_shuffled).view(torch.bfloat16)
        gemm2_scales_shuffled = torch.stack(gemm2_scales_shuffled).view(torch.bfloat16)

        return {
            "gemm1_weights": gemm1_weights_mxint4_shuffled,
            "gemm1_scales": gemm1_scales_shuffled,
            "gemm2_weights": gemm2_weights_mxint4_shuffled,
            "gemm2_scales": gemm2_scales_shuffled,
        }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)
        routed_scaling = kwargs.get("routed_scaling", 1.0)

        # Use autotuner for optimal kernel selection
        with autotune(enable_autotune):
            output = trtllm_mxint4_block_scale_moe(
                expert_logits,  # float
                routing_bias,
                hidden_states_orig,
                static_data["gemm1_weights"],
                static_data["gemm1_scales"],
                None,
                None,
                None,
                static_data["gemm2_weights"],
                static_data["gemm2_scales"],
                num_experts,
                top_k,
                n_groups,
                top_k_groups,
                intermediate_size,
                0,
                num_experts,
                routed_scaling,
                routing_method_type=routing_method_type,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
            )
        return output.to(torch.float)

    def compute_reference(self, args):
        return run_moe_reference_mxint4(args)

    def get_tolerances(self):
        """Get MXINT4-specific accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# FP8 Block Scale Quantization Implementation
# ====================================================================================


class FP8BlockScaleMoe(Moe):
    """FP8 MoE implementation with block scaling (DeepSeek style or MxFp8 x MxFp8)."""

    def __init__(
        self, fp8_quantization_type: QuantMode = QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
    ):
        super().__init__()
        self.fp8_quantization_type = fp8_quantization_type

    @property
    def quant_mode(self) -> QuantMode:
        return self.fp8_quantization_type

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP8 with block scaling."""
        num_experts = gemm1_weights.shape[0]
        intermediate_size = gemm1_weights.shape[1] // 2
        hidden_size = gemm1_weights.shape[
            2
        ]  # [num_experts, 2*intermediate_size, hidden_size]

        if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_DEEPSEEK:
            # Quantize weights to FP8
            gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
            gemm1_scales = 2 * torch.rand(
                (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
                device="cuda",
            ).to(torch.float)

            gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)
            gemm2_scales = 2 * torch.rand(
                (num_experts, hidden_size // 128, intermediate_size // 128),
                device="cuda",
            ).to(torch.float)
        elif self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
            gemm1_weights_fp8, gemm1_scales = mxfp8_quantize_batches(
                gemm1_weights, False
            )
            gemm2_weights_fp8, gemm2_scales = mxfp8_quantize_batches(
                gemm2_weights, False
            )
        else:
            raise ValueError(
                f"Unsupported FP8 quantization type: {self.fp8_quantization_type}"
            )

        return {
            "hidden_states_scale_global": None,  # Block scales computed at runtime
            "gemm1_weights": gemm1_weights_fp8,
            "gemm1_scales": gemm1_scales,
            "gemm1_scales_global": None,
            "gemm2_weights": gemm2_weights_fp8,
            "gemm2_scales": gemm2_scales,
            "gemm2_scales_global": None,
        }

    def quantize_inputs(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale_global: torch.Tensor = None,
        is_swizzling: bool = False,
    ):
        """For FP8 block scaling, no pre-quantization - everything happens at runtime."""

        def to_float8_blockwise(
            x,
            block_size_m=128,
            block_size_n=128,
            dtype=torch.float8_e4m3fn,
            transpose_scale=True,
            is_blockm=False,
            is_blockn=True,
        ):
            assert x.dtype == torch.bfloat16
            x = x.contiguous()
            assert x.dim() == 2
            m, n = x.shape

            m_tile = block_size_m if is_blockm else 1
            n_tile = block_size_n if is_blockn else 1
            num_blocks_m = m // m_tile
            num_blocks_n = n // n_tile

            # Initialize output tensors
            quantized_x = torch.empty_like(x, dtype=dtype, device=x.device)
            scales = torch.empty(
                (num_blocks_m, num_blocks_n), dtype=torch.float32, device=x.device
            )

            # Quantize tensor in blocks
            finfo = torch.finfo(dtype)
            for i in range(num_blocks_m):
                for j in range(num_blocks_n):
                    # Determine block slices
                    start_m, end_m = i * m_tile, min((i + 1) * m_tile, m)
                    start_n, end_n = j * n_tile, min((j + 1) * n_tile, n)

                    # Extract the block
                    block = x[start_m:end_m, start_n:end_n]

                    # Per-block quantization logic
                    min_val, max_val = block.aminmax()
                    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
                    scale = finfo.max / amax

                    # Quantize the block and store the scale
                    quantized_block = (block * scale).clamp(
                        min=finfo.min, max=finfo.max
                    )
                    quantized_x[start_m:end_m, start_n:end_n] = quantized_block.to(
                        dtype
                    )
                    scales[i, j] = scale.float().reciprocal()

            if transpose_scale:
                scales = scales.t()

            return quantized_x, scales

        # todo(Yingyi):quantize bf16 to fp8
        if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_DEEPSEEK:
            hidden_states_quant, hidden_states_scale = to_float8_blockwise(
                hidden_states
            )
        elif self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
            hidden_states_quant, hidden_states_scale = mxfp8_quantize(
                hidden_states, is_swizzling
            )
            hidden_states_scale = hidden_states_scale.view(torch.uint8).reshape(
                *hidden_states.shape[:-1], -1
            )
        else:
            raise ValueError(
                f"Unsupported FP8 quantization type: {self.fp8_quantization_type}"
            )
        return {
            "hidden_states": hidden_states_quant,
            "hidden_states_scale": hidden_states_scale,
        }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""

        # Use shuffled weights with BlockMajorK layout for better performance
        use_shuffled_weight = weight_processing["use_shuffled_weight"]
        weight_layout = weight_processing["layout"]

        if use_shuffled_weight:
            # FIXME: this depends on the kernel internals
            epilogue_tile_m = (
                64
                if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
                else 128
            )

            intermediate_size_factor = (
                2 if is_gated_activation(args.activation_type) else 1
            )

            gemm1_weights_fp8_interleaved = args.gemm1_weights.clone()
            gemm1_scales_fp8_interleaved = args.gemm1_scales.clone()
            if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
                # Reorder rows of W1 for fused gated activation
                gemm1_weights_fp8_interleaved = []
                gemm1_scales_fp8_interleaved = []
                for i in range(num_experts):
                    gemm1_weights_fp8_interleaved.append(
                        reorder_rows_for_gated_act_gemm(
                            args.gemm1_weights[i]
                            .clone()
                            .reshape(intermediate_size_factor * intermediate_size, -1)
                        )
                    )
                    gemm1_scales_fp8_interleaved.append(
                        reorder_rows_for_gated_act_gemm(
                            args.gemm1_scales[i]
                            .clone()
                            .reshape(intermediate_size_factor * intermediate_size, -1)
                        )
                    )

                # Stack weights and scales for all experts
                gemm1_weights_fp8_interleaved = torch.stack(
                    gemm1_weights_fp8_interleaved
                ).reshape(args.gemm1_weights.shape)
                gemm1_scales_fp8_interleaved = torch.stack(
                    gemm1_scales_fp8_interleaved
                ).reshape(args.gemm1_scales.shape)

            gemm1_weights_fp8_shuffled = []
            gemm2_weights_fp8_shuffled = []
            gemm1_scales_fp8_shuffled = []
            gemm2_scales_fp8_shuffled = []
            for i in range(num_experts):
                tmp_weights1 = shuffle_matrix_a(
                    gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
                )
                tmp_weights2 = shuffle_matrix_a(
                    args.gemm2_weights[i].view(torch.uint8), epilogue_tile_m
                )
                if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
                    tmp_scales1 = shuffle_matrix_sf_a(
                        gemm1_scales_fp8_interleaved[i]
                        .view(torch.uint8)
                        .reshape(2 * intermediate_size, -1),
                        epilogue_tile_m,
                    )
                    tmp_scales2 = shuffle_matrix_sf_a(
                        args.gemm2_scales[i].view(torch.uint8).reshape(hidden_size, -1),
                        epilogue_tile_m,
                    )
                    gemm1_scales_fp8_shuffled.append(tmp_scales1)
                    gemm2_scales_fp8_shuffled.append(tmp_scales2)

                if weight_layout == WeightLayout.BlockMajorK:
                    block_k = 128
                    tmp_weights1 = convert_to_block_layout(tmp_weights1, block_k)
                    tmp_weights2 = convert_to_block_layout(tmp_weights2, block_k)

                gemm1_weights_fp8_shuffled.append(tmp_weights1)
                gemm2_weights_fp8_shuffled.append(tmp_weights2)

            kernel_gemm1_weights = torch.stack(gemm1_weights_fp8_shuffled).view(
                torch.float8_e4m3fn
            )
            kernel_gemm2_weights = torch.stack(gemm2_weights_fp8_shuffled).view(
                torch.float8_e4m3fn
            )
            if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
                kernel_gemm1_scales = torch.stack(gemm1_scales_fp8_shuffled).reshape(
                    args.gemm1_scales.shape
                )
                kernel_gemm2_scales = torch.stack(gemm2_scales_fp8_shuffled).reshape(
                    args.gemm2_scales.shape
                )
            else:
                kernel_gemm1_scales = args.gemm1_scales
                kernel_gemm2_scales = args.gemm2_scales
        else:
            kernel_gemm1_weights = args.gemm1_weights
            kernel_gemm2_weights = args.gemm2_weights
            kernel_gemm1_scales = args.gemm1_scales
            kernel_gemm2_scales = args.gemm2_scales

        return {
            "gemm1_weights": kernel_gemm1_weights,
            "gemm1_scales": kernel_gemm1_scales,
            "gemm2_weights": kernel_gemm2_weights,
            "gemm2_scales": kernel_gemm2_scales,
            "use_shuffled_weight": use_shuffled_weight,
            "weight_layout": weight_layout,
        }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime block scale generation + kernel execution."""
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)
        enable_pdl = kwargs.get("enable_pdl")
        hidden_states_scale = kwargs["hidden_states_scale"]
        hidden_states_quant = kwargs["hidden_states_quant"]

        # Generate block scales and quantize hidden states at runtime
        hidden_states_fp8 = hidden_states_quant.to(torch.float8_e4m3fn)
        assert not torch.isnan(hidden_states_fp8.float()).any(), (
            "NaN detected in hidden_states_fp8"
        )

        if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
            quantization_mode = Fp8QuantizationType.MxFp8
        elif self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_DEEPSEEK:
            quantization_mode = Fp8QuantizationType.DeepSeekFp8
        else:
            raise ValueError(
                f"Unsupported FP8 quantization type: {self.fp8_quantization_type}"
            )

        # Use autotuner for optimal kernel selection
        with autotune(enable_autotune):
            output = trtllm_fp8_block_scale_moe(
                expert_logits,
                routing_bias,
                hidden_states_fp8,
                hidden_states_scale,
                static_data["gemm1_weights"],
                static_data["gemm1_scales"],
                static_data["gemm2_weights"],
                static_data["gemm2_scales"],
                num_experts,
                top_k,
                n_groups,
                top_k_groups,
                intermediate_size,
                0,
                num_experts,
                routed_scaling,
                routing_method_type,
                use_shuffled_weight=static_data["use_shuffled_weight"],
                weight_layout=static_data["weight_layout"],
                enable_pdl=enable_pdl,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
                fp8_quantization_type=quantization_mode,
            )
        return output.to(torch.float)

    def compute_reference(self, args):
        """FP8 block-scale reference implementation."""
        if self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_DEEPSEEK:
            return run_moe_reference_dsfp8(args)
        elif self.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8:
            return run_moe_reference_mxfp8(args)
        else:
            raise ValueError(
                f"Unsupported FP8 quantization type: {self.fp8_quantization_type}"
            )

    def get_tolerances(self):
        """Get FP8 block-scale accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.8}


# ====================================================================================
# FP8 Per-Tensor Quantization Implementation
# ====================================================================================


class FP8PerTensorMoe(Moe):
    """FP8 MoE implementation with per-tensor scaling (Llama4 style)."""

    @property
    def quant_mode(self) -> QuantMode:
        return QuantMode.FP8_PER_TENSOR

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP8 per-tensor and compute global scale factors."""
        # Compute global scale factor for hidden states (offline calibration)
        hidden_states_global_scale = calculate_fp8_global_scale_factor(
            hidden_states_sample
        )

        # Quantize to FP8 per-tensor
        gemm1_weights_quant, gemm1_global_scales = quant_fp8_per_tensor_batches(
            gemm1_weights
        )
        gemm2_weights_quant, gemm2_global_scales = quant_fp8_per_tensor_batches(
            gemm2_weights
        )

        return {
            "hidden_states_scale_global": hidden_states_global_scale,
            "gemm1_weights": gemm1_weights_quant,
            "gemm1_scales": None,
            "gemm1_scales_global": gemm1_global_scales,
            "gemm2_weights": gemm2_weights_quant,
            "gemm2_scales": None,
            "gemm2_scales_global": gemm2_global_scales,
        }

    def quantize_inputs(self, hidden_states, hidden_states_scale_global):
        """Quantize hidden states to FP8 per-tensor using pre-computed global scale."""
        # Quantize to FP8 per-tensor using pre-computed global scale factor
        hidden_states_quant, _ = quant_fp8_per_tensor(
            hidden_states, hidden_states_scale_global
        )

        return {
            "hidden_states": hidden_states_quant,
            "hidden_states_scale": None,
        }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""
        # FIXME: this depends on the kernel internals
        epilogue_tile_m = 128

        # Reorder rows of W1 for fused gated activation
        gemm1_weights_fp8_interleaved = []
        for i in range(num_experts):
            if is_gated_activation(args.activation_type):
                weights = reorder_rows_for_gated_act_gemm(args.gemm1_weights[i].clone())
            else:
                weights = args.gemm1_weights[i].clone()
            gemm1_weights_fp8_interleaved.append(weights)

        # Stack weights and scales for all experts
        gemm1_weights_fp8_interleaved = torch.stack(
            gemm1_weights_fp8_interleaved
        ).reshape(
            num_experts,
            (2 if is_gated_activation(args.activation_type) else 1) * intermediate_size,
            hidden_size,
        )

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_fp8_shuffled = []
        gemm2_weights_fp8_shuffled = []
        for i in range(num_experts):
            gemm1_weights_fp8_shuffled.append(
                shuffle_matrix_a(
                    gemm1_weights_fp8_interleaved[i].view(torch.uint8), epilogue_tile_m
                )
            )

            gemm2_weights_fp8_shuffled.append(
                shuffle_matrix_a(
                    args.gemm2_weights[i].view(torch.uint8), epilogue_tile_m
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp8_shuffled = torch.stack(gemm1_weights_fp8_shuffled).view(
            torch.float8_e4m3fn
        )
        gemm2_weights_fp8_shuffled = torch.stack(gemm2_weights_fp8_shuffled).view(
            torch.float8_e4m3fn
        )

        # Calculate scaling factors that depend on weights
        if is_gated_activation(args.activation_type):
            scale_c_fc1 = (
                args_dequant.c_global_sf
                * (1.0 / args.gemm1_scales_global)
                * (1.0 / args.hidden_states_scale_global)
            )
        else:
            scale_c_fc1 = torch.full_like(
                args.gemm1_scales_global, args_dequant.c_global_sf
            )
        scale_gate_fc1 = (1.0 / args.gemm1_scales_global) * (
            1.0 / args.hidden_states_scale_global
        )
        scale_c_fc2 = (1.0 / args_dequant.c_global_sf) * (
            1.0 / args.gemm2_scales_global
        )

        return {
            "gemm1_weights": gemm1_weights_fp8_shuffled,
            "gemm2_weights": gemm2_weights_fp8_shuffled,
            "scale_c_fc1": scale_c_fc1,
            "scale_gate_fc1": scale_gate_fc1,
            "scale_c_fc2": scale_c_fc2,
        }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)
        activation_type = kwargs["activation_type"]

        # Quantize to FP8 per-tensor using pre-computed global scale factor
        hidden_states_fp8, _ = quant_fp8_per_tensor(
            hidden_states_orig, hidden_states_scale_global
        )

        # Use autotuner for optimal kernel selection
        with autotune(enable_autotune):
            output = trtllm_fp8_per_tensor_scale_moe(
                (
                    expert_logits.to(torch.bfloat16)
                    if routing_method_type == RoutingMethodType.Llama4
                    else expert_logits
                ),
                routing_bias,
                hidden_states_fp8,
                static_data["gemm1_weights"],
                static_data["scale_c_fc1"],
                static_data["scale_gate_fc1"],
                static_data["gemm2_weights"],
                static_data["scale_c_fc2"],
                num_experts,
                top_k,
                n_groups,
                top_k_groups,
                intermediate_size,
                0,
                num_experts,
                routed_scaling,
                routing_method_type
                == RoutingMethodType.Llama4,  # Use_routing_scales_on_input
                routing_method_type,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
                activation_type=activation_type,
            )

        return output.to(torch.float)

    def compute_reference(self, args):
        """FP8 per-tensor reference implementation."""
        return run_moe_reference_per_tensor_scale_fp8(args)

    def get_tolerances(self):
        """Get FP8 per-tensor accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# BF16 Implementation
# ====================================================================================


class BF16Moe(Moe):
    """BF16 MoE implementation."""

    @property
    def quant_mode(self) -> QuantMode:
        return QuantMode.BF16

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """No scaling for weights."""
        return {
            "hidden_states_scale_global": None,
            "gemm1_weights": gemm1_weights.to(torch.bfloat16),
            "gemm1_scales": None,
            "gemm1_scales_global": None,
            "gemm2_weights": gemm2_weights.to(torch.bfloat16),
            "gemm2_scales": None,
            "gemm2_scales_global": None,
        }

    def quantize_inputs(self, hidden_states, *unused_args):
        """No scaling for hidden states."""
        return {
            "hidden_states": hidden_states.to(torch.bfloat16),
            "hidden_states_scale": None,
        }

    def prepare_static_weights_for_kernel(
        self,
        args_dequant,
        args,
        gemm1_weights_orig,
        gemm2_weights_orig,
        hidden_size,
        intermediate_size,
        num_experts,
        weight_processing,
    ):
        """Prepare quantized weights for kernel (done offline with weights)."""

        # Use shuffled weights with BlockMajorK layout for better performance
        use_shuffled_weight = weight_processing["use_shuffled_weight"]
        weight_layout = weight_processing["layout"]

        if use_shuffled_weight:
            # FIXME: this depends on the kernel internals
            epilogue_tile_m = 128

            # Reorder rows of W1 for fused gated activation and shuffle for both W1 and W2
            # Using cached permute index calculation can speed up weights preprocessing
            gemm1_weights_bf16_shuffled = []
            gemm2_weights_bf16_shuffled = []
            for i in range(num_experts):
                permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                    self._cache_permute_indices,
                    args.gemm1_weights[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights1 = (
                    args.gemm1_weights[i]
                    .view(torch.uint8)[permute_indices.to(args.gemm1_weights.device)]
                    .contiguous()
                )

                permute_indices = get_w2_permute_indices_with_cache(
                    self._cache_permute_indices,
                    args.gemm2_weights[i].view(torch.uint8),
                    epilogue_tile_m,
                )
                tmp_weights2 = (
                    args.gemm2_weights[i]
                    .view(torch.uint8)[permute_indices.to(args.gemm2_weights.device)]
                    .contiguous()
                )

                if weight_layout == WeightLayout.BlockMajorK:
                    block_k = 128
                    tmp_weights1 = convert_to_block_layout(
                        tmp_weights1.view(torch.uint8), block_k
                    )
                    tmp_weights2 = convert_to_block_layout(
                        tmp_weights2.view(torch.uint8), block_k
                    )

                gemm1_weights_bf16_shuffled.append(tmp_weights1.view(torch.bfloat16))
                gemm2_weights_bf16_shuffled.append(tmp_weights2.view(torch.bfloat16))

            # Stack weights for all experts
            gemm1_weights_bf16_shuffled = (
                torch.stack(gemm1_weights_bf16_shuffled)
                .view(torch.bfloat16)
                .contiguous()
            )
            gemm2_weights_bf16_shuffled = (
                torch.stack(gemm2_weights_bf16_shuffled)
                .view(torch.bfloat16)
                .contiguous()
            )

            return {
                "gemm1_weights": gemm1_weights_bf16_shuffled,
                "gemm2_weights": gemm2_weights_bf16_shuffled,
                "use_shuffled_weight": use_shuffled_weight,
                "weight_layout": weight_layout,
            }

    def call_moe(
        self, static_data, hidden_states_orig, hidden_states_scale_global, **kwargs
    ):
        """Call MoE with runtime input quantization + kernel execution (done at runtime)."""
        expert_logits = kwargs["expert_logits"]
        routing_bias = kwargs["routing_bias"]
        num_experts = kwargs["num_experts"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        routing_method_type = kwargs["routing_method_type"]
        enable_autotune = kwargs.get("enable_autotune", True)

        # Use autotuner for optimal kernel selection
        with autotune(enable_autotune):
            output = trtllm_bf16_moe(
                expert_logits,  # float
                routing_bias,
                hidden_states_orig,
                static_data["gemm1_weights"],
                static_data["gemm2_weights"],
                num_experts,
                top_k,
                n_groups,
                top_k_groups,
                intermediate_size,
                0,
                num_experts,
                routed_scaling,
                use_shuffled_weight=static_data["use_shuffled_weight"],
                weight_layout=static_data["weight_layout"],
                routing_method_type=routing_method_type,
                tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
            )
        return output.to(torch.float)

    def compute_reference(self, args):
        """BF16 reference implementation."""
        return run_moe_reference_bf16(args)

    def get_tolerances(self):
        """Get BF16 accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# Quantizer Factory
# ====================================================================================
def get_moe_impl(quant_mode: QuantMode):
    """Factory function to get the appropriate MoE implementation."""
    if quant_mode == QuantMode.FP8_BLOCK_SCALE_DEEPSEEK:
        return FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
        )
    elif quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8:
        return FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8)
    elif quant_mode == QuantMode.FP8_PER_TENSOR:
        return FP8PerTensorMoe()
    else:
        return FP4Moe(quant_mode)


class moe_args:
    """Arguments container for MoE operations."""

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        hidden_states_scale,
        hidden_states_scale_global,
        expert_logits,
        gemm1_weights,
        gemm1_scales,
        gemm1_scales_global,
        gemm2_weights,
        gemm2_scales,
        gemm2_scales_global,
        permute_info,
        use_routing_scales_on_input,
        activation_type,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.hidden_states_scale = hidden_states_scale
        self.hidden_states_scale_global = hidden_states_scale_global
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm1_scales = gemm1_scales
        self.gemm1_scales_global = gemm1_scales_global
        self.gemm2_weights = gemm2_weights
        self.gemm2_scales = gemm2_scales
        self.gemm2_scales_global = gemm2_scales_global
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input
        self.activation_type = activation_type


class moe_args_dequant:
    """Arguments container for dequantized MoE operations."""

    def __init__(
        self,
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        hidden_states,
        expert_logits,
        gemm1_weights,
        gemm2_weights,
        permute_info,
        use_routing_scales_on_input,
        activation_type,
        hidden_states_scale=None,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.padding = padding
        self.hidden_states = hidden_states
        self.expert_logits = expert_logits
        self.gemm1_weights = gemm1_weights
        self.gemm2_weights = gemm2_weights
        self.permute_info = permute_info
        self.use_routing_scales_on_input = use_routing_scales_on_input
        self.activation_type = activation_type
        self.hidden_states_scale = hidden_states_scale


def routing_reference(expertLogits, topK, padding):
    """Reference routing implementation for permutation calculation."""
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK, dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1, dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[ii + 1] = paddedTokensPerExpertPrefixSum[
            ii
        ] + divUpMul(numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK, dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum": paddedTokensPerExpertPrefixSum.to(
            originalDevice
        ),
        "permutedBufferSize": permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx": expandedTokenIdxToPermutedIdx.to(
            originalDevice
        ),
        "permutedIdxToExpandedIdx": permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert": numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert": expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits": topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx": permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices": topKIndices.to(originalDevice),
    }


def noaux_tc_ref(logits, bias, n_group, topk_group, top_k, routed_scaling_factor):
    """DeepSeek-style no-aux routing reference implementation."""
    scores = F.sigmoid(logits)
    scores_with_bias = scores + bias
    if n_group > 1:
        scores_shape = list(scores_with_bias.shape)
        group_scores = torch.sum(
            torch.topk(
                scores_with_bias.view(
                    scores_shape[:-1] + [n_group, scores_shape[-1] // n_group]
                ),
                k=2,
                dim=-1,
                largest=True,
                sorted=True,
            )[0],
            dim=-1,
        )
        _, group_idx = torch.topk(
            group_scores, k=topk_group, dim=-1, largest=True, sorted=True
        )
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(-1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(scores_shape[:-1] + [n_group, scores_shape[-1] // n_group])
            .reshape(scores_shape)
        )
        scores_with_bias = scores_with_bias * score_mask

    _, topk_idx = torch.topk(
        scores_with_bias, k=top_k, dim=-1, largest=True, sorted=True
    )
    new_mask = torch.zeros_like(scores)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = scores * new_mask
    score_sum = torch.sum(scores, dim=-1, keepdim=True) + 1e-20
    scores = scores / score_sum * routed_scaling_factor
    return scores


def routing_reference_no_aux(
    expert_logits,
    routing_bias,
    top_k,
    n_groups,
    top_k_groups,
    routed_scaling,
    padding,
    use_routing_scales_on_input=False,
):
    """Tiered TopK routing used by DeepSeek."""
    routing_logits = expert_logits.to(dtype=torch.float, device="cuda")
    if use_routing_scales_on_input:
        # if using routing scales on input, topK == 1 and the score is a plain sigmoid
        scores = F.sigmoid(routing_logits)
    else:
        scores = noaux_tc_ref(
            routing_logits, routing_bias, n_groups, top_k_groups, top_k, routed_scaling
        )
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_renormalize(expert_logits, top_k, num_experts, padding):
    """TopK -> Softmax routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_renormalize_naive(expert_logits, top_k, num_experts, padding):
    """Softmax->TopK -> Normalize routing reference."""
    norm_topk_prob = True
    scores = torch.nn.functional.softmax(expert_logits.float(), dim=-1)
    topk_values, topk_idx = torch.topk(scores, k=top_k, dim=-1)

    if norm_topk_prob:  # only diff with mixtral sparse moe block!
        topk_values /= topk_values.sum(dim=-1, keepdim=True)
    topk_values = topk_values.to(expert_logits.dtype)
    scores = scores.to(expert_logits.dtype)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def routing_reference_topk(expert_logits, top_k, num_experts, padding):
    """TopK only (no softmax) routing reference."""
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def check_accuracy(a, b, atol, rtol, percent):
    """Unified accuracy checking function with detailed error reporting."""
    if not torch.isfinite(a).all():
        raise Exception("Non-finite values in reference output")
    if not torch.isfinite(b).all():
        raise Exception("Non-finite values in actual output")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    close = torch.isclose(a, b, atol=atol, rtol=rtol)
    match_ratio = close.float().mean()
    if match_ratio >= percent:
        return

    mismatch_percent = 1.0 - match_ratio.item()
    if mismatch_percent > 1 - percent:
        raise Exception(
            f"Mismatch percentage is {mismatch_percent:.4f} for rtol {rtol} "
            f"(threshold: {1 - percent:.4f})"
        )


# ====================================================================================
# FP4 Quantization Functions
# ====================================================================================


def calculate_fp4_global_scale_factor(tensor, use_ue8m0=False):
    """
    Calculate FP4 global scale factor for a tensor.

    NOTE: In production, global scale factors are typically obtained offline during:
    - Post-Training Quantization (PTQ) calibration process
    - Quantization-Aware Training (QAT) process

    This function is used here for testing/reference purposes.
    Formula: (448 * 6) represents max representable value in FP4 format.
    """
    if use_ue8m0:
        return torch.tensor(1.0, dtype=torch.float32)
    else:
        return (448 * 6) / tensor.float().abs().nan_to_num().max()


def e2m1_and_ufp8_scale_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
    global_scale_tensor: torch.Tensor,
    sf_vec_size: int,
    ufp8_type: int = 1,
):
    """Batch FP4 dequantization helper."""
    num_batches = mat_fp4.size(0)
    scale_tensor = scale_tensor.view(num_batches, -1)

    tensors = [
        e2m1_and_ufp8sf_scale_to_float(
            mat_fp4[b, :, :].cpu(),
            scale_tensor[b, :].cpu().reshape(-1),
            global_scale_tensor[b].cpu(),
            sf_vec_size,
            ufp8_type,
            True,  # is_sf_swizzled_layout
        )
        for b in range(num_batches)
    ]

    result = torch.stack(tensors)
    return result


def quant_fp4(a, a_global_sf, use_ue8m0=False, is_sf_swizzled_layout=True):
    """
    Quantize FP4 with pre-computed global scale factor.

    This function expects global scale factors that have been pre-computed offline
    during PTQ/QAT calibration process. The global scale factor should NOT be
    computed at runtime to avoid performance overhead.

    Pure function - same inputs always produce same outputs.
    """
    sf_vec_size = 32 if use_ue8m0 else 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a, num_experts, use_ue8m0=False, is_sf_swizzled_layout=True):
    """FP4 batch quantization function with centralized global scale factor calculation."""
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        # Use centralized global scale factor calculation
        a_global_sf = calculate_fp4_global_scale_factor(a[i], use_ue8m0)
        a_fp4, a_sf, _ = quant_fp4(a[i], a_global_sf, use_ue8m0, is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)

    return result_quant_a, result_sfs, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    """FP4 quantize-dequantize roundtrip function with centralized global scale factor calculation."""
    # Use centralized global scale factor calculation
    a_global_sf = calculate_fp4_global_scale_factor(a, use_ue8m0)
    sf_vec_size = 32 if use_ue8m0 else 16

    a_fp4, a_sf = fp4_quantize(
        a.cuda(), a_global_sf.cuda(), sf_vec_size, use_ue8m0, is_sf_swizzled_layout
    )

    a_pt = e2m1_and_ufp8sf_scale_to_float(
        a_fp4.cpu(),
        a_sf.cpu().reshape(-1),
        (1 / a_global_sf).cpu(),
        sf_vec_size,
        1 if not use_ue8m0 else 0,  # ufp8_type
        is_sf_swizzled_layout,
    )

    return a_pt.cuda(), a_global_sf


# ====================================================================================
# FP8 Quantization Functions
# ====================================================================================


def calculate_fp8_global_scale_factor(tensor):
    """
    Calculate FP8 global scale factor for a tensor.

    NOTE: In production, global scale factors are typically obtained offline during:
    - Post-Training Quantization (PTQ) calibration process
    - Quantization-Aware Training (QAT) process

    This function is used here for testing/reference purposes.
    Formula: 448 represents max representable value in FP8 E4M3 format.
    """
    return 448 / tensor.float().abs().nan_to_num().max()


def quant_fp8_per_tensor(a, a_global_sf):
    """
    Quantize FP8 per-tensor with pre-computed global scale factor.

    This function expects global scale factors that have been pre-computed offline
    during PTQ/QAT calibration process. The global scale factor should NOT be
    computed at runtime to avoid performance overhead.

    Pure function - same inputs always produce same outputs.
    """
    a_fp8 = (a * a_global_sf).to(torch.float8_e4m3fn)
    return a_fp8, a_global_sf


def quant_fp8_per_tensor_batches(a):
    """FP8 per-tensor batch quantization function with centralized global scale factor calculation."""
    num_batches = a.size(0)
    a_quant = []
    a_scales = []

    for i in range(num_batches):
        # Use centralized global scale factor calculation
        a_global_sf = calculate_fp8_global_scale_factor(a[i])
        a_fp8, _ = quant_fp8_per_tensor(a[i], a_global_sf)
        a_quant.append(a_fp8)
        a_scales.append(a_global_sf)

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def quant_dequant_per_tensor_fp8(a):
    """FP8 per-tensor quantize-dequantize roundtrip function with centralized global scale factor calculation."""
    # Use centralized global scale factor calculation
    a_global_sf = calculate_fp8_global_scale_factor(a)
    a_fp8, _ = quant_fp8_per_tensor(a, a_global_sf)
    a_pt = a_fp8.to(torch.float) / a_global_sf
    return a_pt.cuda(), a_global_sf


def dequant_reference_dsfp8(input, scale, transpose_scale, block_m, block_n):
    """Reference FP8 block-scale dequantization."""
    input = input.to(torch.float)
    scale = scale.to(torch.float)
    if transpose_scale:
        scale = scale.t()

    m, n = input.shape
    m_tile = 128 if block_m else 1
    n_tile = 128 if block_n else 1

    assert m % m_tile == 0
    assert n % n_tile == 0
    assert scale.shape == (m // m_tile, n // n_tile)

    # Expand scale to match input dimensions using tensor operations
    if m_tile > 1:
        scale = torch.repeat_interleave(scale, m_tile, dim=0)
    if n_tile > 1:
        scale = torch.repeat_interleave(scale, n_tile, dim=1)

    # Element-wise multiplication (equivalent to the nested loop logic)
    output = input * scale
    return output


def mxfp8_quantize_batches(a, is_swizzling=True):
    """MxFp8 batch quantization function with centralized global scale factor calculation."""
    num_batches = a.size(0)
    a_quant = []
    a_scales = []
    for i in range(num_batches):
        mx_fp8_quant, mx_fp8_scale = mxfp8_quantize(a[i], is_swizzling)
        a_quant.append(mx_fp8_quant)
        a_scales.append(mx_fp8_scale.view(torch.uint8))

    result_a_quant = torch.stack(a_quant)
    result_a_scales = torch.stack(a_scales)

    return result_a_quant, result_a_scales


def mxfp8_dequantize_batches(a, a_scales, is_swizzling=True):
    """MxFp8 batch dequantization function."""
    num_batches = a.size(0)
    a_dequant = []
    for i in range(num_batches):
        mx_fp8_dequant = mxfp8_dequantize_host(
            a[i].cpu().view(torch.uint8),
            a_scales[i].cpu().view(torch.uint8).reshape(-1),
            is_swizzling,
        )
        a_dequant.append(mx_fp8_dequant.cuda())

    result_a_dequant = torch.stack(a_dequant)

    return result_a_dequant


# ====================================================================================
# Common MoE Reference Implementation
# ====================================================================================


def run_moe_dequant(args, quant_mode: QuantMode):
    """Common dequantized MoE reference implementation."""
    # Permute
    total_num_padded_tokens = args.permute_info["permutedBufferSize"]
    expanded_idx_to_permuted_idx = args.permute_info[
        "expandedTokenIdxToPermutedIdx"
    ].cpu()
    num_tokens_per_expert = args.permute_info["numTokensPerExpert"].cpu()
    permute_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        for j in range(args.top_k):
            permuted_idx = expanded_idx_to_permuted_idx[i * args.top_k + j]
            permute_output[permuted_idx] = args.hidden_states[i]

    # Gemm1
    gemm1_output = torch.full(
        (
            total_num_padded_tokens,
            (2 if is_gated_activation(args.activation_type) else 1)
            * args.intermediate_size,
        ),
        float("nan"),
        device="cuda",
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = permute_output[i : i + my_num_tokens]
        my_b = args.gemm1_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm1_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if args.use_routing_scales_on_input:
        assert args.top_k == 1
        # For each token and its top_k experts
        for token_idx in range(args.num_tokens):
            for k in range(args.top_k):
                # Get the permuted index for this token's k-th expert
                expanded_idx = token_idx * args.top_k + k
                permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
                expert_weight = args.permute_info["topKLogits"].to(torch.float)
                # Get the expert weight for this token and expert
                weight = expert_weight[token_idx, k]
                # Scale the corresponding row in gemm1_output
                gemm1_output[permuted_idx] *= weight

    # Activation
    activation_output = torch.full(
        (total_num_padded_tokens, args.intermediate_size), float("nan"), device="cuda"
    ).to(torch.float)

    activation_type = args.activation_type
    activation_type_to_func = {
        ActivationType.Swiglu: F.silu,
        ActivationType.Geglu: F.gelu,
        ActivationType.Relu2: lambda x: F.relu(x) ** 2,
    }
    activation_func = activation_type_to_func[activation_type]

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        if is_gated_activation(args.activation_type):
            my_x1 = my_a[:, : args.intermediate_size]
            my_x2 = my_a[:, args.intermediate_size :]
            activation_output[i : i + my_num_tokens] = activation_func(my_x2) * my_x1
        else:
            my_x1 = my_a[:, : args.intermediate_size]
            activation_output[i : i + my_num_tokens] = activation_func(my_x1)
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    if quant_mode == QuantMode.FP4_NVFP4_NVFP4:
        # Use centralized function for activation quantization
        activation_output, c_global_sf = quant_dequant_fp4(
            activation_output.to(torch.bfloat16), False, True
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf
    elif quant_mode == QuantMode.FP8_PER_TENSOR:
        activation_output, c_global_sf = quant_dequant_per_tensor_fp8(
            activation_output.to(torch.bfloat16)
        )
        activation_output = activation_output.to(torch.float)
        args.c_global_sf = c_global_sf
    elif (
        quant_mode == QuantMode.FP4_MXFP4_MXFP8
        or quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8
    ):
        activation_output, scale_bytes = mxfp8_quantize(
            activation_output.to(torch.bfloat16), True
        )
        scale_bytes = scale_bytes.view(torch.uint8).reshape(-1).cpu()
        activation_output = (
            mxfp8_dequantize_host(
                activation_output.cpu().view(torch.uint8), scale_bytes
            )
            .cuda()
            .to(torch.float)
        )
        args.c_global_sf = 1.0
    else:  # Bf16, MxFp4xBf16, MxInt4xBf16
        activation_output = activation_output.to(torch.bfloat16).to(torch.float)
        args.c_global_sf = 1.0

    # Gemm2
    gemm2_output = torch.full(
        (total_num_padded_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = activation_output[i : i + my_num_tokens]
        my_b = args.gemm2_weights[expert_idx]
        my_c = my_a @ my_b.t()
        gemm2_output[i : i + my_num_tokens] = my_c
        i += my_num_tokens
        i = (i + args.padding - 1) // args.padding * args.padding

    # Finalize
    expert_weight = args.permute_info["topKLogits"].to(torch.float)
    finalize_output = torch.full(
        (args.num_tokens, args.hidden_size), float("nan"), device="cuda"
    ).to(torch.float)
    for i in range(args.num_tokens):
        acc = torch.zeros(args.hidden_size, dtype=torch.float, device="cuda")
        for top_k_idx in range(args.top_k):
            expanded_idx = i * args.top_k + top_k_idx
            permuted_idx = expanded_idx_to_permuted_idx[expanded_idx]
            original_vector = gemm2_output[permuted_idx]
            weight = (
                expert_weight[i, top_k_idx]
                if not args.use_routing_scales_on_input
                else 1.0
            )
            acc += original_vector * weight
        finalize_output[i] = acc
    return finalize_output


# ====================================================================================
# Quantization-Specific Reference Implementations
# ====================================================================================


def run_moe_reference_fp4(args, quant_mode: QuantMode):
    sf_vec_size = 16 if quant_mode == QuantMode.FP4_NVFP4_NVFP4 else 32
    ufp8_type_weights = 1 if quant_mode == QuantMode.FP4_NVFP4_NVFP4 else 0

    if quant_mode == QuantMode.FP4_NVFP4_NVFP4:
        hidden_states_dequant = e2m1_and_ufp8sf_scale_to_float(
            args.hidden_states.cpu(),
            args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
            (1 / args.hidden_states_scale_global).cpu(),
            sf_vec_size,
            ufp8_type_weights,
            True,  # is_sf_swizzled_layout
        ).cuda()
    elif quant_mode == QuantMode.FP4_MXFP4_MXFP8:
        hidden_states_dequant = mxfp8_dequantize_host(
            args.hidden_states.cpu().view(torch.uint8),
            args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
            True,  # is_sf_swizzled_layout
        ).cuda()
    else:
        hidden_states_dequant = args.hidden_states.to(torch.bfloat16).to(torch.float)

    gemm1_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm1_weights,
        args.gemm1_scales,
        1 / args.gemm1_scales_global,
        sf_vec_size,
        ufp8_type_weights,
    ).cuda()

    gemm2_weights_dequant = e2m1_and_ufp8_scale_batches(
        args.gemm2_weights,
        args.gemm2_scales,
        1 / args.gemm2_scales_global,
        sf_vec_size,
        ufp8_type_weights,
    ).cuda()

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.activation_type,
    )

    return run_moe_dequant(args_dequant, quant_mode), args_dequant


def run_moe_reference_mxfp8(args):
    hidden_states_dequant = mxfp8_dequantize_host(
        args.hidden_states.cpu().view(torch.uint8),
        args.hidden_states_scale.cpu().view(torch.uint8).reshape(-1),
        False,  # is_sf_swizzled_layout
    ).cuda()

    gemm1_weights_dequant = mxfp8_dequantize_batches(
        args.gemm1_weights,
        args.gemm1_scales,
        False,
    ).cuda()

    gemm2_weights_dequant = mxfp8_dequantize_batches(
        args.gemm2_weights,
        args.gemm2_scales,
        False,
    ).cuda()

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.activation_type,
    )

    return run_moe_dequant(args_dequant, QuantMode.FP8_BLOCK_SCALE_MXFP8), args_dequant


def run_moe_reference_dsfp8(args):
    """FP8 block-scale reference implementation (DeepSeek style)."""
    # Generate block scales at runtime for FP8 block scaling

    def dequant_reference_dsfp8(input, scale, transpose_scale, block_m, block_n):
        """Reference FP8 block-scale dequantization."""
        input = input.to(torch.float)
        scale = scale.to(torch.float)
        if transpose_scale:
            scale = scale.t()

        m, n = input.shape
        m_tile = 128 if block_m else 1
        n_tile = 128 if block_n else 1

        assert m % m_tile == 0
        assert n % n_tile == 0
        assert scale.shape == (m // m_tile, n // n_tile)

        # Expand scale to match input dimensions using tensor operations
        if m_tile > 1:
            scale = torch.repeat_interleave(scale, m_tile, dim=0)
        if n_tile > 1:
            scale = torch.repeat_interleave(scale, n_tile, dim=1)

        # Element-wise multiplication (equivalent to the nested loop logic)
        output = input * scale
        return output

    # todo(Yingyi): use original hidden_states??
    hidden_states_dequant = dequant_reference_dsfp8(
        args.hidden_states, args.hidden_states_scale, True, False, True
    )

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm1_weights[i], args.gemm1_scales[i], False, True, True
        )

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = dequant_reference_dsfp8(
            args.gemm2_weights[i], args.gemm2_scales[i], False, True, True
        )

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.activation_type,
    )

    return run_moe_dequant(
        args_dequant, QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
    ), args_dequant


def run_moe_reference_per_tensor_scale_fp8(args):
    """FP8 per-tensor reference implementation."""
    hidden_states_dequant = (
        args.hidden_states.to(torch.float) / args.hidden_states_scale_global
    )

    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = (
            args.gemm1_weights[i].to(torch.float) / args.gemm1_scales_global[i]
        )

    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = (
            args.gemm2_weights[i].to(torch.float) / args.gemm2_scales_global[i]
        )

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.activation_type,
    )

    return run_moe_dequant(args_dequant, QuantMode.FP8_PER_TENSOR), args_dequant


def run_moe_reference_bf16(args):
    """BF16 reference implementation."""

    # no scaling for hidden states and weights
    hidden_states_dequant = args.hidden_states.to(torch.float)
    gemm1_weights_dequant = {}
    for i in range(args.num_experts):
        gemm1_weights_dequant[i] = args.gemm1_weights[i].to(torch.float)
    gemm2_weights_dequant = {}
    for i in range(args.num_experts):
        gemm2_weights_dequant[i] = args.gemm2_weights[i].to(torch.float)

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.activation_type,
    )

    return run_moe_dequant(args_dequant, QuantMode.BF16), args_dequant


def run_moe_reference_mxint4(args):
    sf_vec_size = 32

    hidden_states_dequant = args.hidden_states.to(torch.bfloat16).to(torch.float)

    num_experts = args.gemm1_weights.shape[0]

    def dequantize(weights, scales):
        k = weights.shape[-1] * 2
        n = weights.shape[-2]
        # Unpack two 4-bit values (stored in two's-complement) from each byte
        weights_int8 = (
            torch.stack([weights & 0x0F, (weights >> 4) & 0x0F], dim=-1)
            .reshape(num_experts, n, k)
            .to(torch.int8)
        )

        # Interpret nibbles as signed 4-bit two's-complement values in [-8, 7]
        weights_int8 = torch.where(weights_int8 < 8, weights_int8, weights_int8 - 16)

        weights_float = weights_int8.to(torch.float)
        scales_expanded = (
            scales.to(torch.bfloat16)
            .to(torch.float)
            .repeat_interleave(sf_vec_size, dim=-1)
            .reshape(weights_float.shape)
        )
        return weights_float * scales_expanded

    gemm1_weights_dequant = dequantize(args.gemm1_weights, args.gemm1_scales)
    gemm2_weights_dequant = dequantize(args.gemm2_weights, args.gemm2_scales)

    args_dequant = moe_args_dequant(
        args.num_tokens,
        args.num_experts,
        args.hidden_size,
        args.intermediate_size,
        args.top_k,
        args.padding,
        hidden_states_dequant,
        args.expert_logits,
        gemm1_weights_dequant,
        gemm2_weights_dequant,
        args.permute_info,
        args.use_routing_scales_on_input,
        args.activation_type,
    )

    return run_moe_dequant(args_dequant, QuantMode.MXINT4_BF16_BF16), args_dequant


def _compute_moe_actual_unified(moe_impl, args_dequant, args, **kwargs):
    """Unified actual computation that delegates to implementation-specific methods."""
    # 1. Prepare static weights for the kernel (offline processing)
    static_data = moe_impl.prepare_static_weights_for_kernel(
        args_dequant,
        args,
        kwargs["gemm1_weights_orig"],
        kwargs["gemm2_weights_orig"],
        args.hidden_size,
        args.intermediate_size,
        args.num_experts,
        kwargs["weight_processing"],
    )

    # 2. Call MoE with runtime input quantization + kernel execution
    kernel_kwargs = {
        "expert_logits": kwargs["expert_logits"],
        "routing_bias": kwargs["routing_bias"],
        "num_experts": args.num_experts,
        "num_tokens": args.num_tokens,
        "hidden_size": args.hidden_size,
        "top_k": args.top_k,
        "n_groups": kwargs["n_groups"],
        "top_k_groups": kwargs["top_k_groups"],
        "intermediate_size": args.intermediate_size,
        "routed_scaling": kwargs["routed_scaling"],
        "routing_method_type": kwargs["routing_method_type"],
        "do_finalize": True,
        "activation_type": args.activation_type,
        "hidden_states_scale": args.hidden_states_scale,
        "hidden_states_quant": kwargs["hidden_states_quant"],
        "enable_autotune": kwargs.get("enable_autotune", True),
    }

    return moe_impl.call_moe(
        static_data,
        kwargs["hidden_states_orig"],
        args.hidden_states_scale_global,
        **kernel_kwargs,
    )


@pytest.fixture(scope="module")
def cache_permute_indices():
    # The cache key is now a tuple of (weight_type, shape)
    _cache_permute_indices: Dict[tuple, torch.Tensor] = {}
    return _cache_permute_indices


def run_moe_test(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
    zero_hidden_states=False,
):
    """Common test logic for all routing methods."""
    skip_checks(
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        num_tokens,
        hidden_size,
        intermediate_size,
        zero_hidden_states=zero_hidden_states,
    )

    torch.cuda.synchronize()

    moe_impl._cache_permute_indices = cache_permute_indices

    seed = 0
    torch.random.manual_seed(seed)

    # Extract routing configuration
    top_k = routing_config["top_k"]
    padding = routing_config["padding"]
    n_groups = routing_config["n_groups"]
    top_k_groups = routing_config["top_k_groups"]
    routed_scaling = routing_config["routed_scaling"]
    num_experts = routing_config["num_experts"]
    routing_method_type = routing_config["routing_method_type"]

    # Validation checks
    assert top_k <= num_experts
    assert top_k <= 22
    if (top_k_groups is not None) and (n_groups is not None) and (n_groups > 0):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    # Create test data based on routing method
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    else:
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    if routing_config["has_routing_bias"]:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states_fn = torch.zeros if zero_hidden_states else torch.randn
    hidden_states = 2 * hidden_states_fn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (
            num_experts,
            (2 if is_gated_activation(activation_type) else 1) * intermediate_size,
            hidden_size,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )

    # Generate routing info
    use_routing_scales_on_input = routing_method_type == RoutingMethodType.Llama4

    if routing_method_type == RoutingMethodType.DeepSeekV3:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            use_routing_scales_on_input,
        )
    elif routing_method_type == RoutingMethodType.Renormalize:
        permute_info, scores = routing_reference_renormalize(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, scores = routing_reference_renormalize_naive(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.TopK:
        permute_info, scores = routing_reference_topk(
            expert_logits, top_k, num_experts, padding
        )
    elif routing_method_type == RoutingMethodType.Llama4:
        permute_info, scores = routing_reference_no_aux(
            expert_logits,
            routing_bias,
            top_k,
            n_groups,
            top_k_groups,
            routed_scaling,
            padding,
            use_routing_scales_on_input=True,
        )
    else:
        raise NotImplementedError(
            f"Routing method {routing_method_type} not implemented"
        )

    # 1. Quantize weights offline
    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )

    # 2. Quantize inputs at runtime
    inputs_data = moe_impl.quantize_inputs(
        hidden_states, weights_data["hidden_states_scale_global"]
    )

    # 3. Combine quantized data
    quant_data = {**weights_data, **inputs_data}

    # Create arguments for reference computation
    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        quant_data["hidden_states"],
        quant_data["hidden_states_scale"],
        quant_data["hidden_states_scale_global"],
        scores,
        quant_data["gemm1_weights"],
        quant_data["gemm1_scales"],
        quant_data["gemm1_scales_global"],
        quant_data["gemm2_weights"],
        quant_data["gemm2_scales"],
        quant_data["gemm2_scales_global"],
        permute_info,
        use_routing_scales_on_input,
        activation_type,
    )

    # Compute reference output
    output_dequant_reference, args_dequant = moe_impl.compute_reference(args)

    if output_dequant_reference is None:
        pytest.fail("Reference computation failed to produce output")

    # Compute actual output
    enable_autotune = routing_config.get("enable_autotune", True)

    output_dequant_actual = moe_impl.compute_production(
        args_dequant,
        args,
        expert_logits=expert_logits,
        routing_bias=routing_bias,
        hidden_states_orig=hidden_states,
        gemm1_weights_orig=gemm1_weights,
        gemm2_weights_orig=gemm2_weights,
        n_groups=n_groups,
        top_k_groups=top_k_groups,
        routed_scaling=routed_scaling,
        routing_method_type=routing_method_type,
        weight_processing=weight_processing,
        enable_pdl=True,
        hidden_states_quant=inputs_data["hidden_states"],
        enable_autotune=enable_autotune,
    )

    # Compare outputs
    tolerances = moe_impl.get_tolerances()
    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )


# Test: Renormalize routing
@pytest.mark.parametrize(
    "zero_hidden_states",
    [
        pytest.param(True, id="ZeroHiddenStates"),
        pytest.param(False, id="RandomHiddenStates"),
    ],
)
@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 768, 512, 384])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(BF16Moe(), id="BF16xBF16"),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8),
            id="FP8_Block_MxFp8",
        ),
        pytest.param(FP8PerTensorMoe(), id="FP8_Tensor"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_Bf16), id="MxFP4xBf16"),
        pytest.param(MxInt4BlockScaleMoe(), id="MxInt4xBf16"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": True,
            },
            id="Qwen3_MOE",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 1024],
                "enable_autotune": False,
            },
            id="Renorm",
        ),
        pytest.param(
            {
                "num_experts": 512,
                "top_k": 10,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [512],
                "enable_autotune": True,
            },
            id="Qwen3_next",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu.value, id="Swiglu"),
        pytest.param(ActivationType.Geglu.value, id="Geglu"),
    ],
)
def test_renormalize_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
    zero_hidden_states,
):
    """Test Renormalize routing configurations."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
        zero_hidden_states=zero_hidden_states,
    )


# Test: DeepSeekV3 routing
@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [2944, 2048, 1024, 768, 512, 384])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP8PerTensorMoe(), id="FP8_PerTensor"),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8),
            id="FP8_Block_MxFp8",
        ),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_Bf16), id="MxFP4xBf16"),
        pytest.param(MxInt4BlockScaleMoe(), id="MxInt4xBf16"),
        pytest.param(BF16Moe(), id="Bf16xBf16"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 512,
                "top_k": 22,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP8PerTensorMoe, FP4Moe],
                "compatible_intermediate_size": [2944],
                "compatible_activation_types": [ActivationType.Relu2],
                "enable_autotune": True,
            },
            id="nemotron_3_dummy",
        ),
        pytest.param(
            {
                "num_experts": 384,
                "top_k": 8,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP4Moe, FP8BlockScaleMoe],
                "compatible_intermediate_size": [1024, 2048],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": True,
            },
            id="kimi_k2",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [
                    FP4Moe,
                    FP8BlockScaleMoe,
                    MxInt4BlockScaleMoe,
                    BF16Moe,
                ],
                "compatible_intermediate_size": [512, 1024, 2048],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": True,
            },
            id="DSv3",
        ),
        pytest.param(
            {
                "num_experts": 72,
                "top_k": 6,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP4Moe, FP8BlockScaleMoe],
                "compatible_intermediate_size": [384, 768],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": False,
            },
            id="DSLite",
        ),
        pytest.param(
            {
                "num_experts": 160,
                "top_k": 8,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP4Moe, FP8BlockScaleMoe, BF16Moe],
                "compatible_intermediate_size": [512, 1024, 1536],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": False,
            },
            id="GLM4_MoE",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    MxInt4BlockScaleMoe,
                    BF16Moe,
                ],
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu.value, id="Swiglu"),
        pytest.param(ActivationType.Geglu.value, id="Geglu"),
        pytest.param(ActivationType.Relu2.value, id="Relu2"),
    ],
)
def test_deepseekv3_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
):
    """Test DeepSeekV3 routing configurations."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
    )


# Test: TopK routing
@pytest.mark.parametrize("num_tokens", [8, 128])  # Limited for GeGlu
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [384, 512, 768, 1024])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 16,
                "top_k": 2,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.TopK,
                "compatible_moe_impls": [FP4Moe],
                "compatible_intermediate_size": [512, 768, 1024],
                "enable_autotune": True,
            },
            id="TopK",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu.value, id="Swiglu"),
        pytest.param(ActivationType.Geglu.value, id="Geglu"),
    ],
)
def test_topk_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
):
    """Test TopK routing configuration."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
    )


# Test: Llama4 routing
@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP8PerTensorMoe(), id="FP8_Tensor"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 1,
                "padding": 8,
                "n_groups": 0,
                "top_k_groups": 0,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.Llama4,
                "compatible_moe_impls": [FP8PerTensorMoe],
                "compatible_intermediate_size": [1024, 2048],
                "enable_autotune": True,
            },
            id="Llama4",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu.value, id="Swiglu"),
    ],
)
def test_llama4_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
):
    """Test Llama4 routing configuration with FP8 per-tensor."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
    )
