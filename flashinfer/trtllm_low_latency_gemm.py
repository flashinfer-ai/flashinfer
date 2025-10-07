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

from types import SimpleNamespace
from typing import Dict, List

import functools

from flashinfer.fused_moe.core import (
    convert_to_block_layout,
    get_w2_permute_indices_with_cache,
)
from flashinfer.jit.gemm.core import gen_trtllm_low_latency_gemm_module
import torch

from flashinfer.autotuner import (
    AutoTuner,
    TuningConfig,
    DynamicTensorSpec,
    ConstraintSpec,
    TunableRunner,
    OptimizationProfile,
)
from flashinfer.fused_moe.utils import (
    get_last_power_of_2_num_tokens_buckets,
    last_positive_power_of_2,
)
from flashinfer.jit import setup_cubin_loader
from flashinfer.utils import _get_cache_buf


@functools.cache
def get_trtllm_low_latency_gemm_module():
    mod = gen_trtllm_low_latency_gemm_module()
    op = mod.build_and_load()
    setup_cubin_loader(str(mod.get_library_path()))

    class TrtllmLowLatencyGemmRunner(TunableRunner):
        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            a_tensor_index = 0
            b_tensor_index = 1

            # NOTE : expects  A=MxK, B=(K//B)xNxB, out=MxN
            a = profile.get_opt_shapes()[a_tensor_index]
            b = profile.get_opt_shapes()[b_tensor_index]
            m = a[0]
            n = b[1]
            k = a[1]
            (
                a,
                b,
                global_scale,
                out,
            ) = inputs
            type_e4m3 = 1
            type_bf16 = 2
            valid_tactics = list(
                op.trtllm_low_latency_gemm_tactics(m, n, k, type_e4m3, type_bf16)
            )
            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            (
                a,
                b,
                global_scale,
                out,
            ) = inputs
            if tactic < 0:
                return out
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]
            workspace_size = op.get_workspace_size_in_bytes(m, n, k, tactic)
            workspace_buffer = _get_cache_buf(
                "trllm_low_latency_gemm", workspace_size, a.device
            )
            op.trtllm_low_latency_gemm(
                workspace_buffer,
                a,
                b,
                global_scale,
                out,
                tactic,
            )
            return out

    def gemm_runner():
        return TrtllmLowLatencyGemmRunner()

    # Register the module
    return SimpleNamespace(
        gemm_runner=gemm_runner,
    )


def trtllm_low_latency_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    global_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    r"""GEMM optimized for low M dimension. B needs to be shuffled and its layout needs to be adjusted.
    Only supported on Blackwell GPUs.

    Parameters
    ----------
    A: torch.Tensor
        Input tensor, shape (m, k), fp8 e4m3.

    B: torch.Tensor
        Mat2 tensor, shape (k // block_size, n, block_size), fp8 e4m3. block_size is 128 for e4m3.

    global_scale: torch.Tensor
        Scale tensor for the output, float.

    out: torch.Tensor
        Out tensor, shape (m, n), bf16.

    Examples
    --------
    >>> import torch
    >>> from flashinfer import mm_fp8, prepare_low_latency_gemm_weights
    >>> m = 16
    >>> n = 2560
    >>> k = 32768
    >>> a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    >>> a_fp8, a_inv_s = to_float8(a, dtype=torch.float8_e4m3fn)
    >>> b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    >>> b_fp8, b_inv_s = to_float8(b, dtype=torch.float8_e4m3fn)
    >>> prepared_b = prepare_low_latency_gemm_weights(b_fp8, _cache_permute_indices)
    >>> prepared_b.shape
    torch.Size([256, 16, 128])
    >>> global_scale = a_inv_s * b_inv_s
    >>> out = torch.zeros([m, n], device="cuda", dtype=torch.bfloat16)
    >>> mm_fp8(a_fp8, prepared_b, global_scale, out)
    >>> out.shape
    torch.Size([16, 2560])
    """

    tuner = AutoTuner.get()
    a_tensor_index = 0
    out_tensor_index = 3
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (a_tensor_index,),
                (-2,),
                get_last_power_of_2_num_tokens_buckets,
                last_positive_power_of_2,
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                out_tensor_index, -2, lambda shapes: shapes[a_tensor_index][-2]
            ),
        ),
    )
    inputs = [A, B, global_scale, out]
    runners: List[TunableRunner] = []
    runners.append(get_trtllm_low_latency_gemm_module().gemm_runner())
    runner, tactic = tuner.choose_one(
        "trtllm_low_latency_gemm",
        runners,
        tuning_config,
        inputs,
    )

    runner(inputs=inputs, tactic=tactic)
    return out


def prepare_low_latency_gemm_weights(
    w: torch.Tensor, permutation_indices_cache: Dict[torch.Size, torch.Tensor]
) -> torch.Tensor:
    r"""Helper method to prepare the input weight tensor for low-latency TRTLLM GEMM. It includes shuffling and converting to block layout.

    Parameters
    ----------
    w: torch.Tensor
        The weight tensor to shuffle, shape (n, k), fp8 e4m3.

    permutation_indices_cache: dict
        Some location to cache permutation indices. Calculating them is expensive.

    Returns
    -------
    block_layout_shuffled_weights: torch.Tensor
        The shuffled and block-layout weight tensor, shape (k // 128, n, 128), fp8 e4m3.
    """

    epilogue_tile_m = 128  # NOTE: should be aligned with kernel configuration.

    permute_indices = get_w2_permute_indices_with_cache(
        permutation_indices_cache, w, epilogue_tile_m
    )
    shuffled_weights = w[permute_indices.to(device=w.device)].contiguous()

    block_k = 128
    block_layout_weights = convert_to_block_layout(shuffled_weights, block_k)
    return block_layout_weights
