"""
Copyright (c) 2024 by FlashInfer team.

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

import functools
from types import SimpleNamespace
from typing import Optional, Union, Tuple
import torch

from .api_logging import flashinfer_api
from .jit.gdn import gen_gdn_prefill_module
from .utils import (
    register_custom_op,
    register_fake_op,
    check_shape_dtype_device,
)


@functools.cache
def get_gdn_prefill_module():
    module = gen_gdn_prefill_module().build_and_load()

    @register_custom_op("flashinfer::gdn_prefill", mutates_args=())
    def gdn_prefill(
        output: torch.Tensor,
        output_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
    ) -> None:
        module.gdn_prefill(
            output,
            output_state,
            q,
            k,
            v,
            cu_seqlens,
            initial_state,
            g,
            beta,
            scale,
        )

    @register_fake_op("flashinfer::gdn_prefill")
    def _fake_gdn_prefill(
        output: torch.Tensor,
        output_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
    ) -> None:
        pass

    return SimpleNamespace(gdn_prefill=gdn_prefill)


@flashinfer_api
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]`
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`
        beta (torch.Tensor):
            betas of shape `[B, T, H]`
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        use_qk_l2norm_in_kernel (bool):
            Whether to use QK L2 norm in the kernel. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """

    get_gdn_prefill_module().gdn_prefill(
        output,
        output_state,
        q,
        k,
        v,
        cu_seqlens.to(torch.int64),  # C++ kernel expects int64
        initial_state,
        g,
        beta,
        scale if scale is not None else 0.0,
    )

    if output_final_state:
        return output, output_state
    else:
        return output
