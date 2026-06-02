# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SWA paged slot-id + per-token window-length compute. Builds the per-token
`indices` tensor consumed by the sparse-MLA paged-attention API on the
sliding-window layers of DSv4-Flash / GLM-5.1."""

from __future__ import annotations

import functools

import torch

from .jit.swa_indices import gen_swa_indices_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_swa_indices_module():
    """Build and cache the SWA-indices module + bound custom op."""
    module = gen_swa_indices_module().build_and_load()

    @register_custom_op(
        "flashinfer::compute_swa_indices_and_lens",
        mutates_args=("swa_indices", "swa_lens"),
    )
    def _compute_swa_indices_and_lens(
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        window_size: int,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        is_valid_token: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        token_offset: int,
        num_tokens: int,
    ) -> None:
        module.compute_swa_indices_and_lens(
            swa_indices,
            swa_lens,
            window_size,
            query_start_loc,
            seq_lens,
            token_to_req_indices,
            is_valid_token,
            block_table,
            block_size,
            token_offset,
            num_tokens,
        )

    @register_fake_op("flashinfer::compute_swa_indices_and_lens")
    def _fake_compute_swa_indices_and_lens(*_args, **_kwargs) -> None:
        return None

    return _compute_swa_indices_and_lens


def compute_swa_indices_and_lens(
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    window_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    is_valid_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    token_offset: int,
    num_tokens: int,
) -> None:
    r"""Compute SWA paged slot IDs + per-token window lengths.

    Used by the sliding-window layers of DSv4-Flash / GLM-5.1 to build the
    ``indices`` tensor fed into the sparse-MLA paged-attention API.

    Parameters
    ----------
    swa_indices : torch.Tensor
        Output, ``[num_tokens, window_size]`` or ``[num_tokens, 1, window_size]``,
        int32. Innermost stride must be 1. Invalid-token rows untouched.
    swa_lens : torch.Tensor
        Output, ``[num_tokens]``, int32. Invalid rows get 0.
    window_size : int
        SWA window size (positive).
    query_start_loc : torch.Tensor
        Cumulative query starts, ``[num_reqs + 1]``, int32.
    seq_lens : torch.Tensor
        Total per-request sequence length (prefix + new), ``[num_reqs]``, int32.
    token_to_req_indices : torch.Tensor
        Global-token → request map, ``[>= token_offset + num_tokens]``, int32.
    is_valid_token : torch.Tensor
        Per-token validity, same length as ``token_to_req_indices``, bool.
    block_table : torch.Tensor
        Paged block table, ``[num_reqs, max_blocks]``, int32. Innermost stride 1.
    block_size : int
        Page block size in tokens (positive).
    token_offset : int
        Start of this call's window in the global queue (``>= 0``).
    num_tokens : int
        Output rows to write (``>= 0``).
    """
    get_swa_indices_module()(
        swa_indices,
        swa_lens,
        window_size,
        query_start_loc,
        seq_lens,
        token_to_req_indices,
        is_valid_token,
        block_table,
        block_size,
        token_offset,
        num_tokens,
    )
