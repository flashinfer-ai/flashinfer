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

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict

import pytest
import torch
from cuda.bindings import runtime
from torch.nn import functional as F

from flashinfer import (
    RoutingMethodType,
    GatedActType,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp8_dequantize_host,
    mxfp8_quantize,
    next_positive_power_of_2,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
)
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    WeightLayout,
    convert_to_block_layout,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_per_tensor_scale_moe,
)
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w2_permute_indices,
    _maybe_get_cached_w3_w1_permute_indices,
)


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
        with torch.cuda.stream(torch_stream):
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
            tile_tokens_dim=self.config["tile_tokens_dim"],
            routing_method_type=self.config["routing_method_type"],
            gated_act_type=self.config["gated_act_type"],
            do_finalize=True,
        )
        return output  # Extract tensor from tuple


class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1
    FP4_MXFP4_MXFP8 = 2
    FP4_MXFP4_Bf16 = 3
    FP8_BLOCK_SCALE = 4
    FP8_PER_TENSOR = 5


# ====================================================================================
# Abstract Base Class for MoE Implementations
# ====================================================================================


class Moe(ABC):
    """Abstract base class for MoE implementations."""

    def __init__(self):
        self.name = self.__class__.__name__

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
        self.quant_mode = quant_mode
        self.is_mxfp4 = (
            quant_mode == QuantMode.FP4_MXFP4_MXFP8
            or quant_mode == QuantMode.FP4_MXFP4_Bf16
        )
        self.sf_vec_size = 32 if self.is_mxfp4 else 16

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
        gemm1_weights_fp4 = args.gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
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

            permute_indices = _maybe_get_cached_w2_permute_indices(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = _maybe_get_cached_w2_permute_indices(
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
                num_experts, 2 * intermediate_size, hidden_size // self.sf_vec_size
            )
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // self.sf_vec_size)
        )

        # Calculate scaling factors that depend on weights
        scale_c_fc1 = (
            args_dequant.c_global_sf
            * (1.0 / args.gemm1_scales_global)
            * (1.0 / args.hidden_states_scale_global)
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
        gated_act_type = kwargs["gated_act_type"]
        routing_method_type = kwargs["routing_method_type"]
        tile_tokens_dim = kwargs["tile_tokens_dim"]

        # Create CUDA graph configuration
        config = {
            "hidden_states_scale_global": hidden_states_scale_global,
            "num_experts": num_experts,
            "top_k": top_k,
            "n_groups": n_groups,
            "top_k_groups": top_k_groups,
            "intermediate_size": intermediate_size,
            "routed_scaling": routed_scaling,
            "tile_tokens_dim": tile_tokens_dim,
            "gated_act_type": gated_act_type,
            "routing_method_type": routing_method_type,
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
# FP8 Block Scale Quantization Implementation
# ====================================================================================


class FP8BlockScaleMoe(Moe):
    """FP8 MoE implementation with block scaling (DeepSeek style)."""

    def quantize_weights(self, gemm1_weights, gemm2_weights, hidden_states_sample):
        """Quantize weights to FP8 with block scaling."""
        num_experts = gemm1_weights.shape[0]
        intermediate_size = gemm1_weights.shape[1] // 2
        hidden_size = gemm1_weights.shape[
            2
        ]  # [num_experts, 2*intermediate_size, hidden_size]

        # Quantize weights to FP8
        gemm1_weights_fp8 = gemm1_weights.to(torch.float8_e4m3fn)
        gemm1_scales = 2 * torch.rand(
            (num_experts, 2 * intermediate_size // 128, hidden_size // 128),
            device="cuda",
        ).to(torch.float)

        gemm2_weights_fp8 = gemm2_weights.to(torch.float8_e4m3fn)
        gemm2_scales = 2 * torch.rand(
            (num_experts, hidden_size // 128, intermediate_size // 128), device="cuda"
        ).to(torch.float)

        return {
            "hidden_states_scale_global": None,  # Block scales computed at runtime
            "gemm1_weights": gemm1_weights_fp8,
            "gemm1_scales": gemm1_scales,
            "gemm1_scales_global": None,
            "gemm2_weights": gemm2_weights_fp8,
            "gemm2_scales": gemm2_scales,
            "gemm2_scales_global": None,
        }

    def quantize_inputs(self, hidden_states, hidden_states_scale_global):
        """For FP8 block scaling, no pre-quantization - everything happens at runtime."""
        return {
            "hidden_states": hidden_states,  # Keep original
            "hidden_states_scale": None,  # No pre-computed scales
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
            epilogue_tile_m = 64

            gemm1_weights_fp8_shuffled = []
            gemm2_weights_fp8_shuffled = []
            for i in range(num_experts):
                tmp_weights1 = shuffle_matrix_a(
                    args.gemm1_weights[i].view(torch.uint8), epilogue_tile_m
                )
                tmp_weights2 = shuffle_matrix_a(
                    args.gemm2_weights[i].view(torch.uint8), epilogue_tile_m
                )

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
        else:
            kernel_gemm1_weights = args.gemm1_weights
            kernel_gemm2_weights = args.gemm2_weights

        return {
            "gemm1_weights": kernel_gemm1_weights,
            "gemm1_scales": args.gemm1_scales,
            "gemm2_weights": kernel_gemm2_weights,
            "gemm2_scales": args.gemm2_scales,
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
        num_tokens = kwargs["num_tokens"]
        hidden_size = kwargs["hidden_size"]
        top_k = kwargs["top_k"]
        n_groups = kwargs["n_groups"]
        top_k_groups = kwargs["top_k_groups"]
        intermediate_size = kwargs["intermediate_size"]
        routed_scaling = kwargs["routed_scaling"]
        routing_method_type = kwargs["routing_method_type"]
        tile_tokens_dim = kwargs["tile_tokens_dim"]
        enable_pdl = kwargs.get("enable_pdl")

        # Generate block scales and quantize hidden states at runtime
        hidden_states_fp8 = hidden_states_orig.to(torch.float8_e4m3fn)
        # Use deterministic scales for testing consistency
        hidden_states_scale = 2.0 * torch.ones(
            (hidden_size // 128, num_tokens), device="cuda", dtype=torch.float
        )

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
            tile_tokens_dim,
            routing_method_type,
            use_shuffled_weight=static_data["use_shuffled_weight"],
            weight_layout=static_data["weight_layout"],
            enable_pdl=enable_pdl,
        )

        return output.to(torch.float)

    def compute_reference(self, args):
        """FP8 block-scale reference implementation."""
        return run_moe_reference_dsfp8(args)

    def get_tolerances(self):
        """Get FP8 block-scale accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# FP8 Per-Tensor Quantization Implementation
# ====================================================================================


class FP8PerTensorMoe(Moe):
    """FP8 MoE implementation with per-tensor scaling (Llama4 style)."""

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
            gemm1_weights_fp8_interleaved.append(
                reorder_rows_for_gated_act_gemm(args.gemm1_weights[i].clone())
            )

        # Stack weights and scales for all experts
        gemm1_weights_fp8_interleaved = torch.stack(
            gemm1_weights_fp8_interleaved
        ).reshape(num_experts, 2 * intermediate_size, hidden_size)

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
        scale_c_fc1 = (
            args_dequant.c_global_sf
            * (1.0 / args.gemm1_scales_global)
            * (1.0 / args.hidden_states_scale_global)
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
        tile_tokens_dim = kwargs["tile_tokens_dim"]

        # Quantize to FP8 per-tensor using pre-computed global scale factor
        hidden_states_fp8, _ = quant_fp8_per_tensor(
            hidden_states_orig, hidden_states_scale_global
        )

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
            tile_tokens_dim,
            routing_method_type,
        )

        return output.to(torch.float)

    def compute_reference(self, args):
        """FP8 per-tensor reference implementation."""
        return run_moe_reference_per_tensor_scale_fp8(args)

    def get_tolerances(self):
        """Get FP8 per-tensor accuracy tolerances."""
        return {"atol": 0.1, "rtol": 0.85, "percent": 0.925}


# ====================================================================================
# Quantizer Factory
# ====================================================================================


def get_moe_impl(quant_mode: QuantMode):
    """Factory function to get the appropriate MoE implementation."""
    if quant_mode == QuantMode.FP8_BLOCK_SCALE:
        return FP8BlockScaleMoe()
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
        gated_act_type,
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
        self.gated_act_type = gated_act_type


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
        gated_act_type,
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
        self.gated_act_type = gated_act_type


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
    if torch.any(torch.isnan(a)):
        raise Exception("NaN in reference output")
    if torch.any(torch.isnan(b)):
        raise Exception("NaN in actual output")
    if torch.any(torch.isinf(a)):
        raise Exception("Inf in reference output")
    if torch.any(torch.isinf(b)):
        raise Exception("Inf in actual output")
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
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
        (total_num_padded_tokens, 2 * args.intermediate_size),
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

    gated_act_type = args.gated_act_type
    gated_act_type_to_func = {
        0: F.silu,
        1: F.gelu,
    }
    gated_act_func = gated_act_type_to_func[gated_act_type]

    i = 0
    for expert_idx in range(args.num_experts):
        my_num_tokens = num_tokens_per_expert[expert_idx]
        if my_num_tokens == 0:
            continue
        my_a = gemm1_output[i : i + my_num_tokens]
        my_x1 = my_a[:, : args.intermediate_size]
        my_x2 = my_a[:, args.intermediate_size :]
        activation_output[i : i + my_num_tokens] = gated_act_func(my_x2) * my_x1
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
    elif quant_mode == QuantMode.FP4_MXFP4_MXFP8:
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
    else:  # mxfp4Bf16
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
        args.gated_act_type,
    )

    return run_moe_dequant(args_dequant, quant_mode), args_dequant


def run_moe_reference_dsfp8(args):
    """FP8 block-scale reference implementation."""
    # Generate block scales at runtime for FP8 block scaling
    # Use deterministic scales for testing consistency
    hidden_states_scale = 2.0 * torch.ones(
        (args.hidden_size // 128, args.num_tokens), device="cuda", dtype=torch.float
    )

    hidden_states_dequant = dequant_reference_dsfp8(
        args.hidden_states, hidden_states_scale, True, False, True
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
        GatedActType.SwiGlu.value,  # gated_act_type
    )

    return run_moe_dequant(args_dequant, QuantMode.FP8_BLOCK_SCALE), args_dequant


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
        GatedActType.SwiGlu.value,  # gated_act_type
    )

    return run_moe_dequant(args_dequant, QuantMode.FP8_PER_TENSOR), args_dequant


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
        "tile_tokens_dim": kwargs["tile_tokens_dim"],
        "do_finalize": True,
        "gated_act_type": args.gated_act_type,
    }

    return moe_impl.call_moe(
        static_data,
        kwargs["hidden_states_orig"],
        args.hidden_states_scale_global,
        **kernel_kwargs,
    )


def calculate_tile_tokens_dim(num_tokens: int, num_experts: int, top_k: int) -> int:
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = num_tokens * top_k // num_experts

    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

    return tile_tokens_dim


@pytest.fixture(scope="module")
def cache_permute_indices():
    _cache_permute_indices: Dict[torch.Size, torch.Tensor] = {}
    return _cache_permute_indices


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [1024, 8192])
@pytest.mark.parametrize("intermediate_size", [2048, 1024, 768, 384])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_Bf16), id="MxFP4xBf16"),
        pytest.param(FP8BlockScaleMoe(), id="FP8_Block"),
        pytest.param(FP8PerTensorMoe(), id="FP8_Tensor"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
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
                ],
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
                "compatible_moe_impls": [
                    FP4Moe,
                    FP8BlockScaleMoe,
                ],
            },
            id="DSLite",
        ),
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
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Renorm",
            marks=pytest.mark.skip(
                reason="Disabled for testing speed - similar to RenormalizeNaive"
            ),
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.RenormalizeNaive,
                "compatible_moe_impls": [FP4Moe],
            },
            id="RenormNaive",
        ),
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
            },
            id="TopK",
        ),
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
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "gated_act_type",
    [
        pytest.param(GatedActType.SwiGlu, id="SwiGlu"),
        pytest.param(GatedActType.GeGlu, id="GeGlu"),
    ],
)
def test_moe_quantization_classes(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    gated_act_type,
    cache_permute_indices,
):
    """
    Test MoE implementations using separated quantization workflow.

    This test demonstrates the clean separation between:
    - Static weight quantization (done offline)
    - Dynamic input quantization (done at runtime)

    Each quantization class clearly shows which precision is being used.
    """
    # Skip incompatible combinations
    if gated_act_type == GatedActType.GeGlu and (
        type(moe_impl) is not FP4Moe
        or moe_impl.quant_mode != QuantMode.FP4_NVFP4_NVFP4
        or routing_config["routing_method_type"] != RoutingMethodType.TopK
        or num_tokens > 128
    ):
        # GeGlu is only supported for FP4Moe FP4_NVFP4_NVFP4 and TopK routing
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {gated_act_type} + {routing_config['routing_method_type']} + {num_tokens}"
        )
    elif gated_act_type == GatedActType.SwiGlu and (
        hidden_size > 1024 or intermediate_size > 1024
    ):
        # Skip some tests for SwiGlu for testing speed
        pytest.skip(
            f"Skip for testing speed: {gated_act_type} + {hidden_size} + {intermediate_size}"
        )

    if type(moe_impl) not in routing_config["compatible_moe_impls"]:
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {routing_config['routing_method_type'].name}"
        )
    if type(moe_impl) not in weight_processing["compatible_moe_impls"]:
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {weight_processing['use_shuffled_weight']} + {weight_processing['layout']}"
        )

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
    tile_tokens_dim = calculate_tile_tokens_dim(num_tokens, num_experts, top_k)

    # Validation checks
    assert top_k <= num_experts
    assert top_k <= 8
    if (top_k_groups is not None) and (n_groups is not None) and (n_groups > 0):
        assert top_k_groups <= 4
        assert num_experts > n_groups
        assert num_experts % n_groups == 0
        assert num_experts % 4 == 0
        assert top_k < (top_k_groups * num_experts / n_groups)

    # Create test data based on routing method and quantization mode
    # Different kernels have different dtype requirements for routing logits
    if routing_method_type == RoutingMethodType.DeepSeekV3:
        # DeepSeekV3 uses float for routing logits
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.float
        )
    else:
        # Other routing methods (Renormalize, RenormalizeNaive, Llama4) use bfloat16
        expert_logits = torch.randn((num_tokens, num_experts), device="cuda").to(
            torch.bfloat16
        )

    if routing_config["has_routing_bias"]:
        routing_bias = torch.randn(num_experts, device="cuda", dtype=torch.bfloat16)
    else:
        routing_bias = None

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
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

    # 1. Quantize weights offline (static, done once) + compute global scale factors
    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )

    # 2. Quantize inputs at runtime (dynamic, done per inference) using pre-computed scales
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
        gated_act_type,
    )

    # Compute reference output using the moe_impl
    output_dequant_reference, args_dequant = moe_impl.compute_reference(args)

    # Validate that reference computation succeeded
    if output_dequant_reference is None:
        pytest.fail("Reference computation failed to produce output")

    # Compute actual output using the moe_impl
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
        tile_tokens_dim=tile_tokens_dim,
        weight_processing=weight_processing,
        enable_pdl=True,
    )

    # Compare outputs using moe_impl-specific tolerances
    tolerances = moe_impl.get_tolerances()
    check_accuracy(
        output_dequant_reference,
        output_dequant_actual,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
