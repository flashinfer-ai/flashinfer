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
import os
from enum import IntEnum
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api
from .jit.topk import gen_topk_module
from .trace.templates.sampling import (
    top_k_page_table_transform_trace_dispatch,
    top_k_ragged_transform_trace,
)
from .utils import (
    _get_cache_buf,
    check_shape_dtype_device,
    get_compute_capability,
    get_shared_bytes_per_block_optin,
    register_custom_op,
    register_fake_op,
)


class TopKTieBreak(IntEnum):
    """Top-k tie-break mode.

    This mirrors an enum-class style API while keeping int-compatible values
    for FFI dispatch:
      - NONE  = 0 (legacy behavior)
      - SMALL = 1 (prefer smaller indices)
      - LARGE = 2 (prefer larger indices)
    """

    NONE = 0
    SMALL = 1
    LARGE = 2

    def __str__(self) -> str:
        return self.name.lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    def __format__(self, format_spec: str) -> str:
        return format(str(self), format_spec)


@functools.cache
def get_topk_module():
    module = gen_topk_module().build_and_load()

    @register_custom_op(
        "flashinfer::radix_topk", mutates_args=("row_states_buffer", "output_values")
    )
    def radix_topk(
        input: torch.Tensor,
        top_k: int,
        sorted_output: bool,
        deterministic: bool,
        tie_break: int,
        row_states_buffer: Optional[torch.Tensor],
        output_values: Optional[torch.Tensor],
        dsa_graph_safe: bool = False,
    ) -> torch.Tensor:
        device = input.device
        # Supports float32, float16, bfloat16
        assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {input.dtype}, expected float32, float16, or bfloat16"
        )
        batch_size = input.size(0)
        output_indices = torch.empty(
            batch_size, top_k, dtype=torch.int32, device=device
        )
        module.radix_topk(
            input,
            output_indices,
            output_values,
            row_states_buffer,
            top_k,
            sorted_output,
            deterministic,
            tie_break,
            dsa_graph_safe,
        )
        return output_indices

    @register_fake_op("flashinfer::radix_topk")
    def _fake_radix_topk(
        input: torch.Tensor,
        top_k: int,
        sorted_output: bool,
        deterministic: bool,
        tie_break: int,
        row_states_buffer: Optional[torch.Tensor],
        output_values: Optional[torch.Tensor],
        dsa_graph_safe: bool = False,
    ) -> torch.Tensor:
        batch_size = input.size(0)
        return torch.empty(batch_size, top_k, dtype=torch.int32, device=input.device)

    @register_custom_op(
        "flashinfer::fast_topk_clusters_exact",
        mutates_args=("indices", "output_values", "cached_overflow"),
    )
    def _fast_topk_clusters_exact(
        logits: torch.Tensor,
        indices: torch.Tensor,
        output_values: Optional[torch.Tensor],
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_exact(
            logits,
            indices,
            output_values,
            histogram,
            cached_overflow,
            top_k,
            num_cached,
            num_clusters,
            pdl_enabled,
        )

    @register_fake_op("flashinfer::fast_topk_clusters_exact")
    def _fake_fast_topk_clusters_exact(
        logits: torch.Tensor,
        indices: torch.Tensor,
        output_values: Optional[torch.Tensor],
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::fast_topk_clusters_exact_page_table_transform",
        mutates_args=("indices", "cached_overflow"),
    )
    def _fast_topk_clusters_exact_page_table_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_exact_page_table_transform(
            logits,
            indices,
            seq_lens,
            page_table,
            histogram,
            cached_overflow,
            top_k,
            num_cached,
            num_clusters,
            pdl_enabled,
        )

    @register_fake_op("flashinfer::fast_topk_clusters_exact_page_table_transform")
    def _fake_fast_topk_clusters_exact_page_table_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::fast_topk_clusters_exact_ragged_transform",
        mutates_args=("indices", "cached_overflow"),
    )
    def _fast_topk_clusters_exact_ragged_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        module.fast_topk_clusters_exact_ragged_transform(
            logits,
            indices,
            seq_lens,
            offsets,
            histogram,
            cached_overflow,
            top_k,
            num_cached,
            num_clusters,
            pdl_enabled,
        )

    @register_fake_op("flashinfer::fast_topk_clusters_exact_ragged_transform")
    def _fake_fast_topk_clusters_exact_ragged_transform(
        logits: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        offsets: torch.Tensor,
        histogram: Optional[torch.Tensor],
        cached_overflow: torch.Tensor,
        top_k: int,
        num_cached: int,
        num_clusters: int,
        pdl_enabled: bool,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::radix_topk_page_table_transform",
        mutates_args=(
            "row_states_buffer",
            "output_page_table",
            "output_raw_indices",
        ),
    )
    def radix_topk_page_table_transform(
        input: torch.Tensor,
        output_page_table: torch.Tensor,
        src_page_table: torch.Tensor,
        row_to_batch: Optional[torch.Tensor],
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic: bool,
        tie_break: int,
        page_size: int = 1,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
        page_table_row_starts: Optional[torch.Tensor] = None,
        output_raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {input.dtype}, expected float32, float16, or bfloat16"
        )
        module.radix_topk_page_table_transform(
            input,
            output_page_table,
            src_page_table,
            row_to_batch,
            lengths,
            row_states_buffer,
            top_k,
            deterministic,
            tie_break,
            page_size,
            dsa_graph_safe,
            row_starts,
            page_table_row_starts,
            output_raw_indices,
        )

    @register_fake_op("flashinfer::radix_topk_page_table_transform")
    def _fake_radix_topk_page_table_transform(
        input: torch.Tensor,
        output_page_table: torch.Tensor,
        src_page_table: torch.Tensor,
        row_to_batch: Optional[torch.Tensor],
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic: bool,
        tie_break: int,
        page_size: int = 1,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
        page_table_row_starts: Optional[torch.Tensor] = None,
        output_raw_indices: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::radix_topk_ragged_transform",
        mutates_args=("row_states_buffer", "output_indices"),
    )
    def radix_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic: bool,
        tie_break: int,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
    ) -> None:
        assert input.dtype in [torch.float32, torch.float16, torch.bfloat16], (
            f"Unsupported dtype {input.dtype}, expected float32, float16, or bfloat16"
        )
        module.radix_topk_ragged_transform(
            input,
            output_indices,
            offsets,
            lengths,
            row_states_buffer,
            top_k,
            deterministic,
            tie_break,
            dsa_graph_safe,
            row_starts,
        )

    @register_fake_op("flashinfer::radix_topk_ragged_transform")
    def _fake_radix_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        row_states_buffer: Optional[torch.Tensor],
        top_k: int,
        deterministic: bool,
        tie_break: int,
        dsa_graph_safe: bool = False,
        row_starts: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::cub_topk_page_table_transform",
        mutates_args=("workspace_buffer", "out", "out_raw_indices"),
    )
    def cub_topk_page_table_transform(
        input: torch.Tensor,
        top_k: int,
        tie_break: int,
        page_size: int,
        lengths: torch.Tensor,
        src_page_table: torch.Tensor,
        out: torch.Tensor,
        out_raw_indices: Optional[torch.Tensor],
        workspace_buffer: Optional[torch.Tensor],
        row_to_batch: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        page_table_row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The binding leaves short rows' output tails untouched; pre-fill with -1
        # so the padding contract matches the native fused transforms. The fill
        # kernels are issued here so they are captured with the call under CUDA
        # graphs.
        out.fill_(-1)
        if out_raw_indices is not None:
            out_raw_indices.fill_(-1)
        module.cub_topk_page_table_transform(
            input,
            out,
            src_page_table,
            lengths,
            out_raw_indices,
            workspace_buffer,
            top_k,
            tie_break,
            page_size,
            row_to_batch,
            row_starts,
            page_table_row_starts,
        )
        return out

    @register_fake_op("flashinfer::cub_topk_page_table_transform")
    def _fake_cub_topk_page_table_transform(
        input: torch.Tensor,
        top_k: int,
        tie_break: int,
        page_size: int,
        lengths: torch.Tensor,
        src_page_table: torch.Tensor,
        out: torch.Tensor,
        out_raw_indices: Optional[torch.Tensor],
        workspace_buffer: Optional[torch.Tensor],
        row_to_batch: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        page_table_row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return out

    @register_custom_op(
        "flashinfer::cub_topk_ragged_transform",
        mutates_args=("workspace_buffer", "output_indices"),
    )
    def cub_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        workspace_buffer: Optional[torch.Tensor],
        top_k: int,
        tie_break: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> None:
        # The binding leaves short rows' output tails untouched; pre-fill with -1
        # so the padding contract matches the native fused transforms. The fill
        # kernel is issued here so it is captured with the call under CUDA graphs.
        output_indices.fill_(-1)
        module.cub_topk_ragged_transform(
            input,
            output_indices,
            offsets,
            lengths,
            workspace_buffer,
            top_k,
            tie_break,
            row_starts,
        )

    @register_fake_op("flashinfer::cub_topk_ragged_transform")
    def _fake_cub_topk_ragged_transform(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
        workspace_buffer: Optional[torch.Tensor],
        top_k: int,
        tie_break: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> None:
        pass

    @register_custom_op(
        "flashinfer::cub_topk",
        mutates_args=("workspace_buffer", "output_indices", "output_values"),
    )
    def cub_topk(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        output_values: torch.Tensor,
        workspace_buffer: Optional[torch.Tensor],
        top_k: int,
        tie_break: int,
    ) -> None:
        # No -1 prefill: plain top_k has no lengths window, every output slot is
        # written by the kernel.
        module.cub_topk(
            input,
            output_indices,
            output_values,
            workspace_buffer,
            top_k,
            tie_break,
        )

    @register_fake_op("flashinfer::cub_topk")
    def _fake_cub_topk(
        input: torch.Tensor,
        output_indices: torch.Tensor,
        output_values: torch.Tensor,
        workspace_buffer: Optional[torch.Tensor],
        top_k: int,
        tie_break: int,
    ) -> None:
        pass

    return SimpleNamespace(
        radix_topk=radix_topk,
        radix_topk_page_table_transform=radix_topk_page_table_transform,
        radix_topk_ragged_transform=radix_topk_ragged_transform,
        cub_topk_page_table_transform=cub_topk_page_table_transform,
        cub_topk_page_table_transform_workspace_size=module.cub_topk_page_table_transform_workspace_size,
        cub_topk_ragged_transform=cub_topk_ragged_transform,
        cub_topk_ragged_transform_workspace_size=module.cub_topk_ragged_transform_workspace_size,
        cub_topk=cub_topk,
        cub_topk_workspace_size=module.cub_topk_workspace_size,
        can_implement_filtered_topk=module.can_implement_filtered_topk,
        fast_topk_clusters_exact=_fast_topk_clusters_exact,
        fast_topk_clusters_exact_page_table_transform=_fast_topk_clusters_exact_page_table_transform,
        fast_topk_clusters_exact_ragged_transform=_fast_topk_clusters_exact_ragged_transform,
    )


def can_implement_filtered_topk() -> bool:
    r"""Check if the GPU supports enough shared memory for FilteredTopK algorithm.

    FilteredTopK requires 128KB dynamic shared memory. This function checks if the
    current GPU's max shared memory per SM is sufficient.

    Returns
    -------
    bool
        True if GPU supports FilteredTopK, False otherwise.
    """
    return get_topk_module().can_implement_filtered_topk()


def roundup_kbyte(x):
    return (x + 1023) // 1024 * 1024


@functools.cache
def get_num_cached_for_topk(device, k):
    regs_per_thread = 32
    threads_per_block = 1024
    blocks_per_sm = 65536 // (threads_per_block * regs_per_thread)

    shared_per_block = (
        get_shared_bytes_per_block_optin(device) // blocks_per_sm
    )  # SMEM_CARVEOUT // blocks_per_sm

    buffers_used = (k + 5 + 3 * 256 + 8) * 4  # other shared memory for buffers
    # num_bytes = 2 * 2 * sizeof(int) * num_cached, double buffer on indices and values cache
    return (shared_per_block - buffers_used - 1024) // 16


def get_fast_topk_clusters(batch_size: int) -> int:
    # low batch size, allocate more clusters to get more parallelism
    # high batch size, more parallelism available per row
    if batch_size <= 32:
        return 8
    elif batch_size < 128:
        return 4
    elif batch_size < 256:
        return 2
    else:
        return 1


def topk_clusters_exact(
    logits, top_k, output_values=False, out_dtype=torch.int32, pdl=False
):
    assert out_dtype in (torch.int32, torch.int64), (
        "out_dtype must be torch.int32 or torch.int64"
    )
    batch_size, max_model_len = logits.shape
    indices = torch.empty(batch_size, top_k, dtype=out_dtype, device=logits.device)
    num_clusters = get_fast_topk_clusters(batch_size)
    if max_model_len < 8192:
        num_clusters = 1
    topk_global_overflow = max_model_len // num_clusters
    overflow_buf = torch.empty(
        batch_size,
        4 * topk_global_overflow * num_clusters,
        device=logits.device,
        dtype=torch.int32,
    )
    output_vals = None
    if output_values:
        output_vals = torch.empty(
            batch_size, top_k, dtype=logits.dtype, device=logits.device
        )

    num_cached = get_num_cached_for_topk(logits.device, top_k)
    get_topk_module().fast_topk_clusters_exact(
        logits,
        indices,
        output_vals,
        None,  # histogram
        overflow_buf,
        top_k,
        num_cached,  # num_cached
        num_clusters,
        pdl,
    )
    return indices, output_vals


def topk_clusters_page_table_transform(
    logits, seq_lens, src_page_table, top_k, pdl=False
):
    batch_size, max_model_len = logits.shape
    indices = torch.empty(batch_size, top_k, dtype=torch.int32, device=logits.device)
    num_clusters = get_fast_topk_clusters(batch_size)
    if max_model_len < 8192:
        num_clusters = 1
    topk_global_overflow = max_model_len // num_clusters
    overflow_buf = torch.empty(
        batch_size,
        4 * topk_global_overflow * num_clusters,
        device=logits.device,
        dtype=torch.int32,
    )
    num_cached = get_num_cached_for_topk(logits.device, top_k)
    get_topk_module().fast_topk_clusters_exact_page_table_transform(
        logits,
        indices,
        seq_lens,
        src_page_table,
        None,  # histogram
        overflow_buf,
        top_k,
        num_cached,  # num_cached
        num_clusters,
        pdl,
    )
    return indices


def topk_clusters_ragged_transform(logits, seq_lens, offsets, top_k, pdl=False):
    batch_size, max_model_len = logits.shape
    indices = torch.empty(batch_size, top_k, dtype=torch.int32, device=logits.device)
    num_clusters = get_fast_topk_clusters(batch_size)
    if max_model_len < 8192:
        num_clusters = 1
    topk_global_overflow = max_model_len // num_clusters
    overflow_buf = torch.empty(
        batch_size,
        4 * topk_global_overflow * num_clusters,
        device=logits.device,
        dtype=torch.int32,
    )
    num_cached = get_num_cached_for_topk(logits.device, top_k)
    get_topk_module().fast_topk_clusters_exact_ragged_transform(
        logits,
        indices,
        seq_lens,
        offsets,
        None,  # histogram
        overflow_buf,
        top_k,
        num_cached,  # num_cached
        num_clusters,
        pdl,
    )
    return indices


def can_use_clusters_topk(algo, device, deterministic, tie_break, dsa_graph_safe):
    if dsa_graph_safe or tie_break != TopKTieBreak.NONE:
        return False
    cap = get_compute_capability(device)
    return (algo is None or algo == "clusters") and not deterministic and cap[0] == 10


def can_use_cub_topk(algo, input, tie_break, deterministic, sorted_output=False):
    """Whether the CUB (DeviceBatchedTopK) backend can serve this call."""

    # CUB returns unsorted results and has no reproducible-ordering mode, so
    # sorted / deterministic calls fall through to the native backends.
    if sorted_output or deterministic:
        return False
    if algo is not None and algo != "cub":
        return False  # user forced another backend
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    d = input.size(1)
    if d > (1 << 21):
        return False  # DeviceBatchedTopK per-segment limit
    # Pre-SM90 devices only have the single-block backend (d <= 8192) and no
    # tie-break support (the tie-break requirement configs need SM90+).
    if (d > 8192 or tie_break != TopKTieBreak.NONE) and (
        get_compute_capability(input.device)[0] < 9
    ):
        return False
    return True


def is_cub_topk_beneficial(
    algo, num_rows, d, k, dtype, tie_break, dsa_graph_safe, clusters_eligible
):
    """Whether the CUB backend is expected to outperform the native backends for a
    plain top_k call. Based on benchmarks collected from `bench_topk.py` on a
    B200. See https://gist.github.com/NaderAlAwar/4038e4c44365b93737a55add0e6ec1b5
    for the benchmark results.

    ``clusters_eligible`` states whether the clusters backend could serve this
    call (``can_use_clusters_topk``); it selects between the clusters-calibrated
    and radix-calibrated thresholds. Returning False falls through to the
    clusters / radix dispatch below the CUB branch.
    """
    if algo == "cub":
        return True

    # If replaying in a CUDA graph, CUB wins for larger values of `d`. This is
    # checked before clusters eligibility: an unflagged caller capturing a graph
    # should get CUB's graph-calibrated (and capture-validated) path, not fall
    # through to the never-captured clusters backend.
    if dsa_graph_safe or torch.cuda.is_current_stream_capturing():
        if num_rows < 128 or d <= 8192:
            return True
        return d >= (32768 if dtype == torch.float32 else 65536)

    # When the clusters backend is available (SM100, eager, non-deterministic,
    # no tie-break), it beats CUB nearly everywhere; CUB only keeps the fp32
    # d ~ 8192 band and single-row long-d calls (up to 1.68x fp32, 1.46x 16-bit).
    if clusters_eligible:
        if dtype == torch.float32:
            return (4096 < d <= 8192) or (num_rows == 1 and d > 4096)
        return num_rows == 1 and d >= 262144

    # Eager with no clusters backend available: CUB vs the radix backend.
    # CUB always wins under these conditions
    if num_rows < 128 or d <= 8192:
        return True

    # CUB wins for 16 bit dtypes and larger values of `d` only under certain
    # conditions
    if dtype != torch.float32 and d >= 131072:
        if tie_break == TopKTieBreak.NONE or num_rows < 256:
            return True
        # The native tie-break path is ~30% slower for bf16 specifically (fp16
        # tie is free; CUB is dtype-invariant here), so bf16 tie-break keeps
        # winning through batch 256.
        return dtype == torch.bfloat16 and num_rows <= 256

    # The native no-tie path slows sharply at k >= 4096 while CUB does not
    # (1.3-1.6x on fp32 long rows at any batch); the native tie path rejects
    # k >= 4096 outright
    if dtype == torch.float32 and k >= 4096 and d >= 65536:
        return True

    return False


def is_cub_page_table_transform_beneficial(
    algo, num_rows, d, dtype, tie_break, dsa_graph_safe, clusters_eligible
):
    """Whether the CUB backend is expected to outperform the native backends for a
    fused page-table transform call. Based on benchmarks collected from
    `bench_topk.py` on a B200. See
    https://gist.github.com/NaderAlAwar/4038e4c44365b93737a55add0e6ec1b5
    for benchmark results.
    """
    if algo == "cub":
        return True

    fp32 = dtype == torch.float32
    tie = tie_break != TopKTieBreak.NONE

    if dsa_graph_safe or torch.cuda.is_current_stream_capturing():
        if tie:
            # Tie-break wins at both ends of the d range with a losing pocket in
            # the middle; larger batches enter the pocket earlier and leave later
            if fp32:
                return num_rows < 128 or d <= 2048 or d >= 262144
            if num_rows < 128:
                return d < 1024 or d >= 16384
            return d < 512 or d >= 262144

        if fp32:
            return d >= (32768 if num_rows < 128 else 262144)
        # For 16 bit dtypes, the winning `d` threshold moves up with batch size
        return (
            (num_rows < 64 and d >= 32768)
            or (num_rows < 128 and d >= 65536)
            or d >= 524288
        )

    # When the clusters backend is available (SM100, eager, non-deterministic,
    # no tie-break, clusters-compatible arguments), it beats CUB on every
    # measured transform cell. Check after the graphed branch because the clusters
    # backend doesn't support graph capture.
    if clusters_eligible:
        return False

    if tie:
        if fp32:
            return (num_rows <= 16 and d >= 262144) or (num_rows <= 32 and d >= 524288)
        return (num_rows <= 32 and d >= 262144) or (num_rows <= 64 and d >= 524288)

    return False


def is_cub_ragged_transform_beneficial(
    algo, num_rows, d, dtype, tie_break, dsa_graph_safe, clusters_eligible
):
    """Whether the CUB backend is expected to outperform the native backends for a
    fused ragged transform call. Based on benchmarks collected from
    `bench_topk.py` on a B200. See
    https://gist.github.com/NaderAlAwar/4038e4c44365b93737a55add0e6ec1b5
    for the benchmark results.
    """
    if algo == "cub":
        return True

    fp32 = dtype == torch.float32
    tie = tie_break != TopKTieBreak.NONE

    if dsa_graph_safe or torch.cuda.is_current_stream_capturing():
        if tie:
            if fp32:
                # Small batches win at every d; large batches only lose in the
                # 16384-32768 pocket
                return num_rows < 128 or d < 16384 or d > 32768
            if num_rows < 128:
                return d >= 8192
            return 4096 <= d <= 8192 or d >= 65536

        if fp32:
            if num_rows < 128:
                return d >= 8192
            return 4096 <= d <= 8192 or d >= 65536
        # For 16 bit dtypes, the winning `d` threshold moves up with batch size
        # (and fp16 lags bf16 at large batch)
        if num_rows < 128:
            return d >= 32768
        return d >= (65536 if dtype == torch.bfloat16 else 131072)

    # When the clusters backend is available (SM100, eager, non-deterministic,
    # no tie-break, clusters-compatible arguments), it beats CUB on every
    # measured transform cell. Check after the graphed branch because the clusters
    # backend doesn't support graph capture.
    if clusters_eligible:
        return False

    if tie:
        if fp32:
            return (num_rows <= 16 and d >= 262144) or (num_rows <= 32 and d >= 524288)

        if dtype == torch.bfloat16:
            return (num_rows <= 64 and d >= 262144) or d >= 524288
        return num_rows <= 64 and d >= 262144

    return False


@flashinfer_api
def top_k(
    input: torch.Tensor,
    k: int,
    sorted: bool = False,
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Top-K selection.

    This function selects the top-k largest elements from each row of the input
    tensor. It automatically dispatches between several backends — the native
    radix-based kernels, the clusters kernel (SM100), and CUB's
    ``DeviceBatchedTopK`` — based on shape, dtype, and the requested modes.
    Set ``FLASHINFER_TOPK_ALGO`` (``"default"``, ``"clusters"``, ``"cub"``) to
    force a specific backend.

    This is designed as a drop-in replacement for ``torch.topk`` with better
    performance for large tensors (vocab_size > 10000).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape ``(batch_size, d)`` containing the values to select from.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    k : int
        Number of top elements to select from each row.
    sorted : bool, optional
        If True, the returned top-k elements will be sorted in descending order.
        Default is False (unsorted, which is faster).
    deterministic : bool, optional
        If True, uses deterministic mode.
        Default is False (non-deterministic, which is faster).

        Deterministic mode guarantees repeatable FlashInfer output ordering for
        the selected top-k set on a fixed input and system.
    tie_break : int, optional
        Tie-breaking mode for equal values at the selection boundary.
        Supported modes are (or use ``TopKTieBreak`` enum values):

        - ``0``: no explicit index tie-break
        - ``1``: prefer smaller indices
        - ``2``: prefer larger indices

        Default is ``0``.
        Tie-breaking controls which boundary elements are selected; it does not
        imply deterministic output ordering. Set ``deterministic=True`` when
        repeatable output ordering is also required.
    dsa_graph_safe : bool, optional
        If True, require a CUDA-graph-safe execution path. The native radix
        backend satisfies this by forcing FilteredTopK with graph-safe
        vectorization (VEC_SIZE=1); the CUB backend is graph-safe by
        construction, so it may still serve such calls. Default is False.

    Returns
    -------
    values : torch.Tensor
        Tensor of shape ``(batch_size, k)`` containing the top-k values.
        Same dtype as input.
    indices : torch.Tensor
        Tensor of shape ``(batch_size, k)`` with int64 dtype containing the
        indices of the top-k elements.

    Note
    ----
    - Unlike ``torch.topk``, the default behavior returns unsorted results for
      better performance. Set ``sorted=True`` if you need sorted output.
    - The radix-based algorithm is O(n) in vocabulary size, compared to O(n log k)
      for heap-based methods, making it faster for large vocabularies.
    - For small vocabularies (< 1000), ``torch.topk`` may be faster.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> torch.manual_seed(42)
    >>> batch_size = 4
    >>> vocab_size = 32000
    >>> k = 256
    >>> logits = torch.randn(batch_size, vocab_size, device="cuda")
    >>> values, indices = flashinfer.top_k(logits, k)
    >>> values.shape, indices.shape
    (torch.Size([4, 256]), torch.Size([4, 256]))

    With sorting enabled (for compatibility with torch.topk):

    >>> values_sorted, indices_sorted = flashinfer.top_k(logits, k, sorted=True)
    >>> # Values are now in descending order within each row

    Deterministic mode (bitwise-reproducible output):

    >>> values, indices = flashinfer.top_k(logits, k, deterministic=True)

    See Also
    --------
    torch.topk : PyTorch's built-in top-k function
    sampling.top_k_mask_logits : Top-k masking for logits (sets non-top-k to -inf)
    sampling.top_k_renorm_probs : Top-k filtering and renormalization for probabilities
    """
    batch_size = input.size(0)
    device = input.device

    algo = os.environ.get("FLASHINFER_TOPK_ALGO")
    clusters_eligible = can_use_clusters_topk(
        algo, input.device, deterministic, tie_break, dsa_graph_safe
    )

    if can_use_cub_topk(
        algo, input, tie_break, deterministic, sorted
    ) and is_cub_topk_beneficial(
        algo,
        batch_size,
        input.size(1),
        k,
        input.dtype,
        tie_break,
        dsa_graph_safe,
        clusters_eligible,
    ):
        topk_module = get_topk_module()
        # Host-side size query (launches nothing); the workspace is cached per device
        # so repeated calls (including under CUDA graph capture) reuse a stable
        # allocation.
        workspace_bytes = topk_module.cub_topk_workspace_size(input, k, int(tie_break))
        workspace_buffer: torch.Tensor = _get_cache_buf(
            f"cub_topk_workspace_{device}", workspace_bytes, device
        )

        output_values = torch.empty(batch_size, k, dtype=input.dtype, device=device)
        indices = torch.empty(batch_size, k, dtype=torch.int64, device=device)
        topk_module.cub_topk(
            input,
            indices,
            output_values,
            workspace_buffer,
            k,
            int(tie_break),
        )

        return output_values, indices

    if clusters_eligible:
        indices, output_values = topk_clusters_exact(
            input, k, output_values=True, out_dtype=torch.int64
        )
        if sorted:
            sorted_values, sort_indices = torch.sort(
                output_values, dim=-1, descending=True
            )
            sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
            return sorted_values, sorted_indices
        return output_values, indices

    # Allocate row_states buffer for multi-CTA path
    # 1MB is enough for any reasonable GPU (covers up to ~200 groups for deterministic
    # mode and ~300 groups for non-deterministic mode)
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{input.device}",
        1024 * 1024,  # 1MB
        input.device,
        zero_init=True,
    )

    # Allocate output_values for kernel to write directly
    output_values = torch.empty(batch_size, k, dtype=input.dtype, device=device)

    # For deterministic + sorted + k <= 2048: CUDA handles the stable value sort on device.
    sorted_cuda = sorted and deterministic and k <= 2048
    indices_int32 = get_topk_module().radix_topk(
        input,
        k,
        sorted_cuda,
        deterministic,
        tie_break,
        row_states_buffer,
        output_values,
        dsa_graph_safe,
    )

    # Convert to int64 for compatibility
    indices = indices_int32.long()

    if sorted and not sorted_cuda:
        # Sort within each row by value (descending)
        sorted_values, sort_indices = torch.sort(
            output_values, dim=-1, descending=True, stable=deterministic
        )
        sorted_indices = torch.gather(indices, dim=-1, index=sort_indices)
        return sorted_values, sorted_indices

    return output_values, indices


# Alias for compatibility
topk = top_k


@flashinfer_api(trace=top_k_page_table_transform_trace_dispatch)
def top_k_page_table_transform(
    input: torch.Tensor,
    src_page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_to_batch: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
    row_starts: Optional[torch.Tensor] = None,
    page_table_row_starts: Optional[torch.Tensor] = None,
    *,
    page_size: int = 1,
    out: Optional[torch.Tensor] = None,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fused Top-K selection + Page Table Transform for sparse attention.

    This function performs top-k selection on input scores and translates the
    selected indices through a page table in a single fused kernel. It
    automatically dispatches between several backends — the native radix-based
    kernels, the clusters kernel (SM100), and CUB's ``DeviceBatchedTopK`` — based on shape, dtype,
    and the requested modes. Set ``FLASHINFER_TOPK_ALGO`` (``"default"``,
    ``"clusters"``, ``"cub"``) to force a specific backend. Each
    page-table entry represents ``page_size`` consecutive score positions. For
    each selected local index ``idx`` in row ``i``::

        physical_page = src_page_table[
            batch_idx, page_table_row_start[i] + idx // page_size
        ]
        output[i, j] = physical_page * page_size + idx % page_size

    where ``batch_idx`` is determined by ``row_to_batch[i]`` if provided,
    otherwise ``i``. ``topk_indices`` are relative to ``row_starts[i]``.

    Parameters
    ----------
    input : torch.Tensor
        Input scores tensor of shape ``(num_rows, max_len)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    src_page_table : torch.Tensor
        Source page table of shape ``(batch_size, max_page_table_length)`` with
        dtype ``int32``. Entries used by selected indices must be nonnegative,
        and each resulting ``physical_page * page_size + offset`` must fit in
        signed ``int32``. These value constraints are not checked at runtime.
    lengths : torch.Tensor
        Actual KV lengths per row of shape ``(num_rows,)`` with dtype ``int32``.
    k : int
        Number of top elements to select from each row.
    row_to_batch : Optional[torch.Tensor], optional
        Mapping from row index to batch index of shape ``(num_rows,)`` with
        dtype ``int32``. If None, uses 1:1 mapping (row_idx == batch_idx).
        Default is None.
    deterministic : bool, optional
        If True, uses deterministic mode.
        Default is False (non-deterministic, which is faster).
    tie_break : int, optional
        Tie-breaking mode for equal values at the selection boundary.
        Supported modes are (or use ``TopKTieBreak`` enum values):

        - ``0``: no explicit index tie-break
        - ``1``: prefer smaller indices
        - ``2``: prefer larger indices

        Default is ``0``.
        Tie-breaking controls which boundary elements are selected; it does not
        imply deterministic output ordering. Set ``deterministic=True`` when
        repeatable output ordering is also required.
    dsa_graph_safe : bool, optional
        If True, require a CUDA-graph-safe execution path. The native radix
        backend satisfies this by forcing FilteredTopK with graph-safe
        vectorization (VEC_SIZE=1); the CUB backend is graph-safe by
        construction, so it may still serve such calls. Default is False.
    row_starts : Optional[torch.Tensor], optional
        Per-row start indices of shape ``(num_rows,)`` with dtype ``int32``.
        Top-k is computed over ``[row_starts[i], row_starts[i] + lengths[i])`` for row ``i``.
        Default is None (equivalent to all zeros).
    page_table_row_starts : Optional[torch.Tensor], optional
        Per-row page-table start indices of shape ``(num_rows,)`` with dtype
        ``int32``, measured in page-table entries. If None, defaults to
        ``row_starts``, so score and page-table windows share the same start.
        When ``page_size > 1`` and
        ``row_starts`` is provided, this argument must also be provided because
        the two starts use different units.
    page_size : int, optional
        Number of score positions represented by each page-table entry. Must
        be a positive power of two no greater than ``2**30``. Setting this to
        1 preserves the one-entry-per-score behavior. Default is 1.
    out : Optional[torch.Tensor], optional
        Optional contiguous ``int32`` output buffer of shape ``(num_rows, k)``.
        Supplying this buffer avoids an allocation and is CUDA-graph friendly.
    out_raw_indices : Optional[torch.Tensor], optional
        Optional contiguous ``int32`` output buffer of shape ``(num_rows, k)``.
        Receives selected indices relative to each score window before
        page-table translation. Padding positions are set to -1 and remain
        positionally aligned with ``out``. Must not overlap ``out``.

    Returns
    -------
    output : torch.Tensor
        Physical indices of shape ``(num_rows, k)`` with dtype ``int32``. This
        is the same tensor as ``out`` when one is supplied. Positions beyond
        actual length are set to -1.

    Note
    ----
    - This is specifically designed for sparse attention's second stage.
    - ``input`` may have padding between rows, but its last dimension must be
      contiguous.
    - If ``lengths[i] <= k``, raw indices are ``0..lengths[i]-1`` and remaining
      positions are set to -1.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_rows = 8
    >>> max_len = 4096
    >>> k = 256
    >>> scores = torch.randn(num_rows, max_len, device="cuda", dtype=torch.float16)
    >>> src_page_table = torch.randint(0, 1000, (num_rows, max_len), device="cuda", dtype=torch.int32)
    >>> lengths = torch.full((num_rows,), max_len, device="cuda", dtype=torch.int32)
    >>> output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)
    >>> output.shape
    torch.Size([8, 256])
    """
    device = input.device
    num_rows = input.size(0)

    if page_size != 1:
        if page_size <= 0 or page_size > 1 << 30 or page_size & (page_size - 1):
            raise ValueError(
                "page_size must be a positive power of two no greater than 2**30, "
                f"got {page_size}"
            )
        if row_starts is not None and page_table_row_starts is None:
            raise ValueError(
                "page_table_row_starts is required with page_size > 1 and row_starts"
            )
    if out is not None:
        check_shape_dtype_device(out, (num_rows, k), torch.int32, device, "out")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    if out_raw_indices is not None:
        check_shape_dtype_device(
            out_raw_indices,
            (num_rows, k),
            torch.int32,
            device,
            "out_raw_indices",
        )
        if not out_raw_indices.is_contiguous():
            raise ValueError("out_raw_indices must be contiguous")

    algo = os.environ.get("FLASHINFER_TOPK_ALGO")
    clusters_eligible = (
        can_use_clusters_topk(
            algo, input.device, deterministic, tie_break, dsa_graph_safe
        )
        and row_to_batch is None
        and row_starts is None
        and page_table_row_starts is None
        and page_size == 1
        and out is None
        and out_raw_indices is None
        and input.is_contiguous()
    )

    if can_use_cub_topk(
        algo, input, tie_break, deterministic
    ) and is_cub_page_table_transform_beneficial(
        algo,
        input.size(0),
        input.size(1),
        input.dtype,
        tie_break,
        dsa_graph_safe,
        clusters_eligible,
    ):
        topk_module = get_topk_module()
        # Host-side size query (launches nothing); the workspace is cached per device
        # so repeated calls (including under CUDA graph capture) reuse a stable
        # allocation.
        workspace_bytes = topk_module.cub_topk_page_table_transform_workspace_size(
            input, lengths, k, int(tie_break), out_raw_indices is not None
        )
        workspace_buffer: torch.Tensor = _get_cache_buf(
            f"cub_topk_workspace_{device}", workspace_bytes, device
        )
        if out is None:
            out = torch.empty(num_rows, k, dtype=torch.int32, device=device)
        return topk_module.cub_topk_page_table_transform(
            input,
            k,
            int(tie_break),
            page_size,
            lengths,
            src_page_table,
            out,
            out_raw_indices,
            workspace_buffer,
            row_to_batch,
            row_starts,
            page_table_row_starts,
        )

    if clusters_eligible:
        return topk_clusters_page_table_transform(input, lengths, src_page_table, k)

    # Allocate row_states buffer for multi-CTA path
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{device}",
        1024 * 1024,  # 1MB
        device,
        zero_init=True,
    )

    if out is None:
        out = torch.empty(num_rows, k, dtype=torch.int32, device=device)

    get_topk_module().radix_topk_page_table_transform(
        input,
        out,
        src_page_table,
        row_to_batch,
        lengths,
        row_states_buffer,
        k,
        deterministic,
        tie_break,
        page_size,
        dsa_graph_safe,
        row_starts=row_starts,
        page_table_row_starts=page_table_row_starts,
        output_raw_indices=out_raw_indices,
    )

    return out


@flashinfer_api(trace=top_k_ragged_transform_trace)
def top_k_ragged_transform(
    input: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    deterministic: bool = False,
    tie_break: int = TopKTieBreak.NONE,
    dsa_graph_safe: bool = False,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Fused Top-K selection + Ragged Index Transform for sparse attention.

    This function performs top-k selection on input scores and transforms the
    selected indices by adding an offset in a single fused kernel.
    Used in sparse attention's second stage with ragged/variable-length KV cache.
    It automatically dispatches between several backends — the native
    radix-based kernels, the clusters kernel (SM100), and CUB's
    ``DeviceBatchedTopK`` — based on shape, dtype, and the requested modes. Set
    ``FLASHINFER_TOPK_ALGO`` (``"default"``, ``"clusters"``, ``"cub"``) to
    force a specific backend.

    For each row i:
        output_indices[i, j] = topk_indices[j] + offsets[i]

    Parameters
    ----------
    input : torch.Tensor
        Input scores tensor of shape ``(num_rows, max_len)``.
        Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
    offsets : torch.Tensor
        Offset to add per row of shape ``(num_rows,)`` with dtype ``int32``.
    lengths : torch.Tensor
        Actual KV lengths per row of shape ``(num_rows,)`` with dtype ``int32``.
    k : int
        Number of top elements to select from each row.
    deterministic : bool, optional
        If True, uses deterministic mode.
        Default is False (non-deterministic, which is faster).
    tie_break : int, optional
        Tie-breaking mode for equal values at the selection boundary.
        Supported modes are (or use ``TopKTieBreak`` enum values):

        - ``0``: no explicit index tie-break
        - ``1``: prefer smaller indices
        - ``2``: prefer larger indices

        Default is ``0``.
        Tie-breaking controls which boundary elements are selected; it does not
        imply deterministic output ordering. Set ``deterministic=True`` when
        repeatable output ordering is also required.
    dsa_graph_safe : bool, optional
        If True, require a CUDA-graph-safe execution path. The native radix
        backend satisfies this by forcing FilteredTopK with graph-safe
        vectorization (VEC_SIZE=1); the CUB backend is graph-safe by
        construction, so it may still serve such calls. Default is False.
    row_starts : Optional[torch.Tensor], optional
        Per-row start indices of shape ``(num_rows,)`` with dtype ``int32``.
        Top-k is computed over ``[row_starts[i], row_starts[i] + lengths[i])`` for row ``i``.
        Output indices remain ``local_topk + offsets[i]`` where ``local_topk`` is relative to
        ``row_starts[i]``. Default is None (equivalent to all zeros).


    Returns
    -------
    output_indices : torch.Tensor
        Output indices of shape ``(num_rows, k)`` with dtype ``int32``.
        Contains the top-k indices plus offsets.
        Positions beyond actual length are set to -1.

    Note
    ----
    - This is specifically designed for sparse attention's second stage with
      ragged KV cache layout.
    - If lengths[i] <= k, the output contains [offsets[i], offsets[i]+1, ..., offsets[i]+lengths[i]-1]
      with remaining positions set to -1.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> num_rows = 8
    >>> max_len = 4096
    >>> k = 256
    >>> scores = torch.randn(num_rows, max_len, device="cuda", dtype=torch.float16)
    >>> offsets = torch.arange(0, num_rows * max_len, max_len, device="cuda", dtype=torch.int32)
    >>> lengths = torch.full((num_rows,), max_len, device="cuda", dtype=torch.int32)
    >>> output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)
    >>> output.shape
    torch.Size([8, 256])
    """
    device = input.device
    num_rows = input.size(0)

    algo = os.environ.get("FLASHINFER_TOPK_ALGO")
    clusters_eligible = (
        can_use_clusters_topk(
            algo, input.device, deterministic, tie_break, dsa_graph_safe
        )
        and row_starts is None
    )

    if can_use_cub_topk(
        algo, input, tie_break, deterministic
    ) and is_cub_ragged_transform_beneficial(
        algo,
        input.size(0),
        input.size(1),
        input.dtype,
        tie_break,
        dsa_graph_safe,
        clusters_eligible,
    ):
        topk_module = get_topk_module()
        # Host-side size query (launches nothing); the workspace is cached per device
        # so repeated calls (including under CUDA graph capture) reuse a stable
        # allocation.
        workspace_bytes = topk_module.cub_topk_ragged_transform_workspace_size(
            input, lengths, k, int(tie_break)
        )
        workspace_buffer: torch.Tensor = _get_cache_buf(
            f"cub_topk_workspace_{device}", workspace_bytes, device
        )
        output_indices = torch.empty(num_rows, k, dtype=torch.int32, device=device)
        topk_module.cub_topk_ragged_transform(
            input,
            output_indices,
            offsets,
            lengths,
            workspace_buffer,
            k,
            int(tie_break),
            row_starts,
        )
        return output_indices

    if clusters_eligible:
        return topk_clusters_ragged_transform(input, lengths, offsets, k)

    # Allocate row_states buffer for multi-CTA path
    row_states_buffer: Optional[torch.Tensor] = _get_cache_buf(
        f"radix_topk_row_states_{device}",
        1024 * 1024,  # 1MB
        device,
        zero_init=True,
    )

    # Allocate output
    output_indices = torch.empty(num_rows, k, dtype=torch.int32, device=device)

    get_topk_module().radix_topk_ragged_transform(
        input,
        output_indices,
        offsets,
        lengths,
        row_states_buffer,
        k,
        deterministic,
        tie_break,
        dsa_graph_safe,
        row_starts=row_starts,
    )

    return output_indices
