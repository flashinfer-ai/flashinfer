# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native D512 sparse MLA over one or two pools using task-scheduled kernels."""

from typing import NamedTuple

import torch

from flashinfer.api_logging import flashinfer_experimental_api
from flashinfer.trace.templates.sparse_mla import (
    sparse_mla_wrapper_trace_dispatch,
    sparse_mla_one_shot_trace_dispatch,
)


class SparseMLAPreparedMetadata(NamedTuple):
    """Caller-owned storage-row indices and prepared attention metadata.

    Indices are int32 [rows, capacity], with -1 invalid; lengths are int32
    [rows]. Primary and extra indices address their own flattened native pool.
    No block-table lookup or page-stride conversion occurs in run.

    Combined-route schedules also consume routes [passes, rows, route_capacity],
    execution_lengths/valid_counts [passes, rows], and FP32 scale_params
    [passes, 2 + heads + rows]. Route bit 31 selects the extra pool; bits 0:31
    encode the storage row, with 0x7fffffff invalid. Each source's execution
    span is rounded to 128. Empty rows execute one masked slot. Scale rows
    contain [QK scale, PV scale, sinks[heads], valid_counts[rows]]. Two passes
    are required when the sources use independent KV descales.

    Preparation must refresh all dependent fields when routes, lengths,
    scales or sinks change. Buffers must remain alive and unaliased with
    writable attention buffers through graph replay. Applications may supply
    these buffers from any preparer satisfying this contract.
    """

    indices: torch.Tensor
    lengths: torch.Tensor
    extra_indices: torch.Tensor | None = None
    extra_lengths: torch.Tensor | None = None
    routes: torch.Tensor | None = None
    execution_lengths: torch.Tensor | None = None
    valid_counts: torch.Tensor | None = None
    scale_params: torch.Tensor | None = None


class BatchSparseMLADecodePagedTSWrapper:
    """Plan native BF16/E4M3 attention over one or two pools, then bind metadata.

    Q is [B,Sq,H,512] or packed [total_q,H,512]. Caller-prepared metadata
    contains int32 [rows, capacity] storage-row indices and live lengths.
    Indices already include physical page strides; -1 is masked. Both source
    lists participate in one attention distribution. Packed KV is unsupported.

    Planning with ``assume_valid_prefix=True`` promises that every index before
    each live source length is valid (no -1 holes). Direct FP8 2CTA kernels
    then derive masks from lengths without reading indices or issuing ballots.
    Other kernels retain their generic mask path; no gain was established.
    ``run(validate=True)`` checks the promise; graph replay with validation
    disabled must preserve it. Tail entries beyond the lengths are ignored.

    A wrapper/workspace permits one in-flight run. Graphs require a completed
    warmup, stable addresses, preallocated outputs, and validate=False.
    Input/output/workspace storage must not overlap; this is an unchecked
    caller precondition. Separate wrappers/workspaces are needed per stream.
    """

    @flashinfer_experimental_api
    def __init__(self, workspace_buffer=None):
        from flashinfer.experimental.prims_ts_sparse_mla.runtime import (
            SparseMLADecodePlan,
        )

        self._impl = SparseMLADecodePlan(workspace_buffer)

    @property
    def workspace_size_bytes(self):
        """Number of bytes required by the current attention plan."""
        return self._impl.workspace_size_bytes

    def plan(
        self,
        device,
        batch_size,
        num_heads,
        *,
        max_topk,
        max_extra_topk=0,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=torch.bfloat16,
        kv_layout="NHD",
        has_sinks=False,
        return_lse=False,
        assume_valid_prefix=False,
    ):
        """Plan D512 attention over one primary and an optional extra pool.

        ``max_topk`` describes arbitrary selected primary entries, not an SWA
        window. Prepared indices already include the physical page stride, so
        the core uses page size one regardless of the pools' external pages.
        Planning compiles no metadata-preparation kernel.
        """
        return self._impl.plan(
            device,
            batch_size,
            num_heads,
            max_topk=max_topk,
            max_extra_topk=max_extra_topk,
            max_seq_len_q=max_seq_len_q,
            packed_query=packed_query,
            q_data_type=q_data_type,
            kv_layout=kv_layout,
            has_sinks=has_sinks,
            return_lse=return_lse,
            assume_valid_prefix=assume_valid_prefix,
        )

    @flashinfer_experimental_api(trace=sparse_mla_wrapper_trace_dispatch)
    def run(
        self,
        query,
        kv_cache,
        metadata: SparseMLAPreparedMetadata,
        extra_kv_cache=None,
        *,
        qo_indptr=None,
        softmax_scale=512**-0.5,
        q_scale=1.0,
        kv_scale=1.0,
        extra_kv_scale=None,
        output_scale=1.0,
        sinks=None,
        out=None,
        lse=None,
        validate=True,
    ):
        """Launch with caller-prepared metadata, without index conversion.

        Caller-owned metadata and plan workspace must outlive graph replay. Q and
        native KV are used directly; flattening padded pages only creates a
        tensor view. Required attention reductions/finishing remain included.
        Metadata is an unchecked consistency contract: its packed routes,
        counts and scales must agree with the supplied indices and scalars.
        """
        return self._impl.run(
            query,
            kv_cache,
            metadata,
            extra_kv_cache,
            qo_indptr=qo_indptr,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            extra_kv_scale=extra_kv_scale,
            output_scale=output_scale,
            sinks=sinks,
            out=out,
            lse=lse,
            validate=validate,
        )


def get_prims_ts_sparse_mla_decode_workspace_size(*plan_args, **plan_kwargs):
    """Return workspace bytes using the same arguments as wrapper.plan().

    Resolves the default kernel geometry without allocating GPU scratch or
    compiling a kernel. Bind the resulting byte buffer to the wrapper constructor
    and call plan() before graph capture; run() is the prepared standalone launch.
    """
    from flashinfer.experimental.prims_ts_sparse_mla.runtime import (
        get_prims_ts_sparse_mla_decode_workspace_size as impl,
    )

    return impl(*plan_args, **plan_kwargs)


@flashinfer_experimental_api(trace=sparse_mla_one_shot_trace_dispatch)
def batch_sparse_mla_decode_with_paged_kv_cache(
    query,
    kv_cache,
    metadata: SparseMLAPreparedMetadata,
    extra_kv_cache=None,
    *,
    qo_indptr=None,
    max_seq_len_q=None,
    kv_layout="NHD",
    softmax_scale=512**-0.5,
    q_scale=1.0,
    kv_scale=1.0,
    extra_kv_scale=None,
    output_scale=1.0,
    sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    workspace_buffer=None,
    assume_valid_prefix=False,
):
    """Eager plan-and-run helper using caller-prepared metadata.

    Use a planned wrapper for CUDA Graph replay. Preparation remains external.
    """
    from flashinfer.experimental.prims_ts_sparse_mla.runtime import (
        batch_sparse_mla_decode_with_paged_kv_cache as impl,
    )

    return impl(
        query,
        kv_cache,
        metadata,
        extra_kv_cache,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        kv_layout=kv_layout,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        kv_scale=kv_scale,
        extra_kv_scale=extra_kv_scale,
        output_scale=output_scale,
        sinks=sinks,
        out=out,
        lse=lse,
        return_lse=return_lse,
        workspace_buffer=workspace_buffer,
        assume_valid_prefix=assume_valid_prefix,
    )
