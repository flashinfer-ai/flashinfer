# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#
# Vendored from the NVIDIA dynamic-kernel-generator (DKG) repository,
# cutlass_ir/compiler/python/examples/CuTeDSL/cute/blackwell/kernel/top_k/filtered_top_k_decode_varlen.py
# at DKG master b45e50a7336 (merge request !25590, "[CuTeDSL] Adapt radix top-k to
# Rubin arch").
#
# Changes from upstream are limited to: this header, removal of DKG-internal
# release markers, and import rewrites to flashinfer-relative paths. The kernel
# algorithm is unmodified so upstream fixes can be re-applied by re-vendoring.

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass import testing
from cutlass.torch import dtype as torch_dtype

from .filtered_topk_util import (
    _RUBIN_TOPK_ARCHITECTURES,
    FilteredTopKKernelVarlen,
    auto_tma_load,
    auto_unroll_factor,
    compare_top_k_results,
    create_random_logits,
    get_topk_architecture_config,
    run_reference_top_k,
    tma_tuned_default,
)


def _get_num_sms(device: torch.device | None = None) -> int:
    """Return the number of SMs on ``device`` (default: current), cached.

    Cached PER DEVICE: a single process-wide scalar would pin the first
    caller's SM count and silently mis-size the persistent grid and the
    large-occupancy mode decision for every later call on a different GPU of a
    heterogeneous host. Call sites that have a tensor in scope pass its
    device. DIVERGENCE FROM UPSTREAM (single ambient-device scalar), after
    review (flashinfer PR #4621).
    """
    if device is not None and device.index is not None:
        idx = device.index
    else:
        idx = torch.cuda.current_device()
    cache = getattr(_get_num_sms, "_values", None)
    if cache is None:
        cache = _get_num_sms._values = {}
    if idx not in cache:
        cache[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return cache[idx]


def _compile_cc(device: torch.device) -> str:
    """Compute-capability tag for the compile cache keys (e.g. ``"sm100"``).

    Both the in-process dict and the on-disk artifact are architecture
    specific (SMEM sizing, occupancy config, cluster limits), so identical
    shapes compiled on one GPU must not be reused on a different architecture
    in the same process. DIVERGENCE FROM UPSTREAM, after review (flashinfer
    PR #4621).
    """
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm{major}{minor}"


def _dsl_gpu_arch(device: torch.device) -> str:
    """Explicit ``cute.compile`` target for *device* (e.g. ``"sm_100a"``).

    Without an explicit ``--gpu-arch``, the DSL resolves its target from the
    process-ambient device, which can disagree with the INPUT tensor's device
    on a multi-GPU host (e.g. cutlass imported while cuda:0 is current, input
    on cuda:1): the kernel then compiles for the wrong architecture even
    under ``torch.cuda.device(input.device)``, failing in ptxas at best and
    persisting a wrong-arch artifact under the right-arch module directory at
    worst. DIVERGENCE FROM UPSTREAM, after review (flashinfer PR #4621).
    """
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _persistent_compile(kernel_name: str, compile_fn):
    """Route a ``cute.compile`` through FlashInfer's persistent JIT cache.

    ``build_and_load_cute_dsl_kernel`` adds what a bare ``cute.compile``
    lacks: an on-disk artifact keyed by architecture and DSL version,
    cross-process locking, source-hash invalidation over the three files this
    kernel is built from, and ``FLASHINFER_DISABLE_JIT`` support -- without it
    every new process (e.g. each SGLang worker) recompiles on first use.
    DIVERGENCE FROM UPSTREAM, after review (flashinfer PR #4621).
    """
    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel  # noqa: PLC0415

    from . import block_scan as _block_scan  # noqa: PLC0415
    from . import filtered_topk_util as _util  # noqa: PLC0415

    return build_and_load_cute_dsl_kernel(
        "radix_filter_topk",
        kernel_name,
        compile_fn,
        extra_key_files=(__file__, _util.__file__, _block_scan.__file__),
    )


def _kernel_name_from_key(key: tuple) -> str:
    """Filename-safe specialization name derived from a compile-cache key."""
    raw = "_".join(str(k) for k in key)
    return "".join(ch if (ch.isalnum() or ch in "._") else "-" for ch in raw)


"""
A high-performance topk kernel example based on radix-based filter algorithm for
the NVIDIA Blackwell SM100 architecture based on CuTe DSL.

The radix-based filter top-k algorithm mainly includes two phases: coarse filter and multi-round fine-grained filter.
For each phase:
1. histogram: Build a histogram of the input values using vectorized loads.
2. prefix sum: Find the threshold bin using prefix sum.
3. find target bin id: Find the target bin id using multiple rounds.
Finally, write the top-k values and indices to the output tensor.

Supported data types:
- Float32
- Float16
- BFloat16

To run this example:
.. code-block:: bash
    python examples/blackwell/sort/filter_top_k_decode_varlen.py  \
      --dtype Float32 --batch_size 1 --max_num_cols 4096 --next_n 3 \
      --top_k 2048 --wrapper-mode single-cta \
      --do_ref_check --return_val --benchmark

Constraints for this example:
* The supported top_k range is [1, 16384].
* The input tensor has data contiguous on the n dimension (row-major).
* The supported input data types are Float32, Float16, or BFloat16.
"""


class FilteredTopKKernelVarlenDecode(FilteredTopKKernelVarlen):
    def __init__(
        self,
        dtype: cutlass.Numeric,
        max_num_cols: int,
        top_k: int,
        next_n: int = 1,
        num_copy_bits: int = 256,
        return_val: bool = True,
        large_occupancy: bool = False,
        # for multi-cta version.
        enable_multi_cta: bool = False,
        chunk_size_per_cta: int = 16384,
        num_ctas_per_row: int = 1,
        merge_blocks: bool = False,
        enable_dynamic_multi_cta: bool = False,
        varlen_merge_input: bool = False,
        overflow_policy: str = "REREAD",
        cache_smem_values: bool = False,
        single_pass_multi_cta: bool = False,
        target_blocks_per_sm: int = 4,
        architecture: str = "sm_100",
        unroll_factor: int = 4,
        enable_tma_load: bool = False,
        enable_tma_load_p3: bool = False,
        tma_num_stages: int = 4,
        tma_num_stages_p3: int = 2,
    ):
        self._large_occupancy = large_occupancy
        self._target_blocks_per_sm = target_blocks_per_sm
        # async-TMA load (p1 + p3), hard-restricted to the large-occupancy single-CTA
        # path (_tma_ok) -- cluster / small-occupancy always keep LDG even if forced on.
        # The tuned auto-default lives in _prepare_one_pass_topk, which passes a concrete
        # flag; None here defaults OFF (constructing this class directly needs an opt-in).
        _tma_ok = large_occupancy and not single_pass_multi_cta
        _enable_tma_load = _tma_ok and enable_tma_load
        _enable_tma_load_p3 = _tma_ok and enable_tma_load_p3
        _tma_num_stages = tma_num_stages
        _tma_num_stages_p3 = tma_num_stages_p3
        super().__init__(
            dtype,
            max_num_cols,
            top_k,
            num_copy_bits,
            return_val,
            enable_multi_cta,
            chunk_size_per_cta,
            num_ctas_per_row,
            merge_blocks,
            overflow_policy=overflow_policy,
            # large_occupancy always uses 512 threads; pass it early so that
            # _compute_smem_input_size_for_occupancy() sees the correct num_warps.
            num_threads_override=512 if large_occupancy else 0,
            cache_smem_values=cache_smem_values,
            single_pass_multi_cta=single_pass_multi_cta,
            architecture=architecture,
            unroll_factor=unroll_factor,
            enable_tma_load=_enable_tma_load,
            tma_num_stages=_tma_num_stages,
            enable_tma_load_p3=_enable_tma_load_p3,
            tma_num_stages_p3=_tma_num_stages_p3,
        )
        self.next_n = next_n
        self.enable_multi_cta = enable_multi_cta
        self.chunk_size_per_cta = chunk_size_per_cta
        self.merge_blocks = merge_blocks
        self.num_ctas_per_row = num_ctas_per_row
        self.enable_dynamic_multi_cta = enable_dynamic_multi_cta
        self.varlen_merge_input = varlen_merge_input

        if cutlass.const_expr(self.merge_blocks):
            # Cap vec_size so tile_width (num_threads_per_cta * vec_size) <= max_num_cols,
            # preventing OOB s_indices from _fill_oob padding.
            _vec_cap = max(
                1,
                2
                ** int(
                    math.log2(max(self.max_num_cols // self.num_threads_per_cta, 1))
                ),
            )
            self.num_copy_bits = min(self.num_copy_bits, _vec_cap * self.dtype.width)
            self.vec_size = self.num_copy_bits // self.dtype.width

        # Resolve the "auto" (0) unroll sentinel now that threads/vec_size and
        # the per-CTA scan extent are known.
        if self.unroll_factor == 0:
            blocks_per_sm = self._target_blocks_per_sm if self._large_occupancy else 1
            per_cta_cols = (
                self.chunk_size_per_cta
                if (self.enable_multi_cta or self.single_pass_multi_cta)
                else self.max_num_cols
            )
            self.unroll_factor = auto_unroll_factor(
                per_cta_cols,
                self.num_threads_per_cta,
                blocks_per_sm,
                self.vec_size,
                self.dtype.width // 8,
            )

    def _compute_smem_input_size(self) -> int:
        if cutlass.const_expr(self._large_occupancy):
            return self._compute_smem_input_size_for_occupancy(
                target_blocks_per_sm=self._target_blocks_per_sm
            )
        return self._compute_smem_input_size_for_occupancy(target_blocks_per_sm=1)

    @cute.kernel
    def filtered_topk_kernel(
        self,
        input: cute.Tensor,
        indices: cute.Tensor,
        extra_buffer: cute.Tensor,
        seqlen: cute.Tensor,
        output_indices: cute.Tensor,
        output_values: cute.Tensor,
        tma_atom=None,
        tma_tensor=None,
    ):
        """CuTe DSL implementation of TopK kernel based on radix-based filter algorithm."""
        cute.arch.griddepcontrol_wait()

        smem = cutlass.memory.SmemAllocator()
        # Keep control fields separate because each has a distinct indexing scheme.
        s_histogram_buf_layout = cute.make_ordered_layout((self.radix + 1), order=(0))
        s_histogram = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=s_histogram_buf_layout,
            byte_alignment=128,
        )
        s_counter = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        s_threshold_bin_id = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1), order=(0)),
            byte_alignment=128,
        )
        s_num_input = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((2,), order=(0)),
            byte_alignment=128,
        )
        if cutlass.const_expr(self.enable_gmem_store or self.enable_bounded_spill):
            g_num_input = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2), order=(0)),
                byte_alignment=128,
            )
        else:
            g_num_input = None
        s_indices = smem.allocate_tensor(
            element_type=self.index_type,
            layout=cute.make_ordered_layout((self.top_k,), order=(0)),
            byte_alignment=128,
        )
        # Candidate buffer + async-TMA staging. Each TMA ring (p1 histogram, p3 filter)
        # needs SMEM; instead of a dedicated buffer (which would shrink the candidate
        # capacity S), it ALIASES a candidate region that is idle while that phase runs
        # -> zero extra SMEM. The ring must fit its aliased region
        # (n_stages * tile_bytes <= region bytes); that is guaranteed upstream (__init__
        # asserts capacity, adaptively caps n_stages, and reserves the mbarriers), so no
        # runtime check is needed here. Three cases:
        #   p1                : whole candidate buffer is unused during the histogram
        #                       (no candidate written yet) -> aliased below.
        #   p3 fp32 (nbuf==2) : buffer 1 is idle during the filter (candidates go to
        #                       buffer 0) -> alias buffer 1 (this branch).
        #   p3 fp16/bf16 (1)  : single buffer, nothing idle -> dedicated carve (else).
        if cutlass.const_expr(
            self.enable_tma_load_p3 and self.num_buffer_smem_input_idx == 2
        ):
            # Buffer-major layout: one contiguous Int32 pool split into `n_buffers`
            # equal, 128B-padded slots (so buffer 1 is one contiguous region the p3
            # staging can alias). idx/val are (n_buffers, capacity) strided views into
            # the slots, so kernel access is unchanged.
            n_buffers = self.num_buffer_smem_input_idx
            capacity = self.filtered_topk_smem_input_size  # candidate slots / buffer
            idx_bytes = self.index_type.width // 8
            val_bytes = self.ordered_type.width // 8 if self.cache_smem_values else 0
            slot_bytes = ((capacity * (idx_bytes + val_bytes) + 127) // 128) * 128
            int32_bytes = 4  # pool element size; byte offsets below are / int32_bytes

            pool = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout(
                    ((n_buffers * slot_bytes) // int32_bytes,), order=(0,)
                ),
                byte_alignment=128,
            )
            base = pool.iterator
            # idx double-buffer: buffer b at byte offset b * slot_bytes.
            s_input_idx = cute.make_tensor(
                cute.recast_ptr(base, dtype=self.index_type),
                cute.make_layout(
                    (n_buffers, capacity), stride=(slot_bytes // idx_bytes, 1)
                ),
            )
            if cutlass.const_expr(self.cache_smem_values):
                # val double-buffer, placed right after idx inside each slot.
                s_input_val = cute.make_tensor(
                    cute.recast_ptr(
                        base + (capacity * idx_bytes) // int32_bytes,
                        dtype=self.ordered_type,
                    ),
                    cute.make_layout(
                        (n_buffers, capacity), stride=(slot_bytes // val_bytes, 1)
                    ),
                )
            else:
                s_input_val = None
            # p3 staging ring aliases buffer 1 (byte offset slot_bytes).
            tile_cols = self.num_threads_per_cta * self.vec_size
            s_tma_stage_p3 = cute.make_tensor(
                cute.recast_ptr(base + slot_bytes // int32_bytes, dtype=self.dtype),
                cute.make_ordered_layout(
                    (self.tma_num_stages_p3, 1, tile_cols), order=(2, 1, 0)
                ),
            )
            s_tma_mbar_p3 = smem.allocate_array(
                cutlass.Int64, 2 * self.tma_num_stages_p3
            )
        else:
            # fp16/bf16 p3: dedicated carve (no idle buffer to alias); or p3 off.
            # Either way the candidate buffer keeps its plain flat (nbuf,S) layout.
            if cutlass.const_expr(self.enable_tma_load_p3):
                tile_cols = self.num_threads_per_cta * self.vec_size
                s_tma_stage_p3 = smem.allocate_tensor(
                    element_type=self.dtype,
                    layout=cute.make_ordered_layout(
                        (self.tma_num_stages_p3, 1, tile_cols), order=(2, 1, 0)
                    ),
                    byte_alignment=128,
                )
                s_tma_mbar_p3 = smem.allocate_array(
                    cutlass.Int64, 2 * self.tma_num_stages_p3
                )
            else:
                s_tma_stage_p3 = None
                s_tma_mbar_p3 = None
            if cutlass.const_expr(not self.enable_reread_always):
                s_input_idx = smem.allocate_tensor(
                    element_type=self.index_type,
                    layout=cute.make_ordered_layout(
                        (
                            self.num_buffer_smem_input_idx,
                            self.filtered_topk_smem_input_size,
                        ),
                        order=(1, 0),
                    ),
                    byte_alignment=128,
                )
            else:
                s_input_idx = None
            if cutlass.const_expr(
                self.cache_smem_values and not self.enable_reread_always
            ):
                s_input_val = smem.allocate_tensor(
                    element_type=self.ordered_type,
                    layout=cute.make_ordered_layout(
                        (
                            self.num_buffer_smem_input_idx,
                            self.filtered_topk_smem_input_size,
                        ),
                        order=(1, 0),
                    ),
                    byte_alignment=128,
                )
            else:
                s_input_val = None
        # p1 staging ring aliases the (idle) candidate buffer s_input_idx -- see the
        # alias note above. Zero extra SMEM; S unchanged.
        if cutlass.const_expr(self.enable_tma_load):
            tile_cols = self.num_threads_per_cta * self.vec_size
            s_tma_stage = cute.make_tensor(
                cute.recast_ptr(s_input_idx.iterator, dtype=self.dtype),
                cute.make_ordered_layout(
                    (self.tma_num_stages, 1, tile_cols), order=(2, 1, 0)
                ),
            )
            s_tma_mbar = smem.allocate_array(cutlass.Int64, 2 * self.tma_num_stages)
        else:
            s_tma_stage = None
            s_tma_mbar = None
        if cutlass.const_expr(self.enable_reread or self.enable_bounded_spill):
            s_overflow_flag = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
        else:
            s_overflow_flag = None
        s_last_remain = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((1,), order=(0,)),
            byte_alignment=128,
        )
        num_warps = cutlass.const_expr(
            min(self.radix, self.num_threads_per_cta) // cutlass.Int32(32)
        )
        s_warp_sums = smem.allocate_tensor(
            element_type=cute.Int32,
            layout=cute.make_ordered_layout((num_warps,), order=(0,)),
            byte_alignment=128,
        )
        # SP multi-CTA (radix-filter cluster): separate DSMEM merge target so the
        # local s_histogram is never written in-place while peers read it.
        # (The collection prefix-scan scratch reuses s_histogram, not this buffer.)
        if cutlass.const_expr(self.single_pass_multi_cta):
            s_hist_merged = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.radix + 1), order=(0)),
                byte_alignment=128,
            )
        else:
            s_hist_merged = None

        # Thread and block indexing
        bidx, bidy, _ = cute.arch.block_idx()

        # ---- SP multi-CTA (radix-filter cluster) block indexing + dispatch ----
        # 1D grid = (num_rows * ctas_per_group,), cluster = (ctas_per_group,1,1).
        # row_id / cta_in_group derived from the global block index; needed_ctas
        # from seqlen decides solo (needed_ctas==1) vs cluster (>=2) at runtime.
        need_cluster_sync = False
        cta_in_group = 0
        # Cluster mode derives its row and CTA coordinates before the common path.
        if cutlass.const_expr(self.single_pass_multi_cta):
            row_id = bidx // self.num_ctas_per_row
            cta_in_group = bidx % self.num_ctas_per_row
            _batch = row_id // self.next_n
            _off = row_id % self.next_n
            _eff = seqlen[_batch] - self.next_n + _off + 1
            chunk_start = self.chunk_size_per_cta * cta_in_group
            row_start = chunk_start
            row_end = min(_eff, chunk_start + self.chunk_size_per_cta)
            length = row_end - row_start
            _needed = (_eff + self.chunk_size_per_cta - 1) // self.chunk_size_per_cta
            if _needed < 1:
                _needed = 1
            need_cluster_sync = _needed >= 2
            # The cluster radix search has no threshold when the whole row fits
            # in top_k. Let CTA 0 take the full row through the trivial path.
            if _eff <= self.top_k:
                need_cluster_sync = False
                row_start = 0
                length = _eff
            # score/dst index the row (not the global block).
            bidx = row_id

        if cutlass.const_expr(not self.single_pass_multi_cta):
            if cutlass.const_expr(self.enable_dynamic_multi_cta):
                # 2D grid with early exit: bidx = row_id, bidy = chunk_id.
                # Each CTA computes how many chunks its row actually needs
                # from seqlen and exits early if bidy >= num_needed_ctas.
                # This avoids prefix sum + binary search overhead entirely.
                num_rows_val = seqlen.shape[0] * self.next_n

            row_start = 0
            row_end = 0
            length = 0
            seq_len = 0

            if not cutlass.const_expr(self.merge_blocks):
                seq_len = seqlen[bidx // self.next_n]
                row_end = seq_len - self.next_n + (bidx % self.next_n) + 1
                length = row_end - row_start

        if cutlass.const_expr(self.enable_multi_cta):
            # update row_start and row_end.
            row_start = self.chunk_size_per_cta * bidy
            row_end = min(row_end, row_start + self.chunk_size_per_cta)
            length = row_end - row_start
            output_indices = cute.flat_divide(output_indices, (1, self.top_k))[
                0, None, bidx, bidy
            ]
            output_values = cute.flat_divide(output_values, (1, self.top_k))[
                0, None, bidx, bidy
            ]

        if cutlass.const_expr(self.merge_blocks):
            if cutlass.const_expr(self.varlen_merge_input):
                # Varlen merge: compute per-row valid length from seqlen.
                _batch = bidx // self.next_n
                _off = bidx % self.next_n
                _eff = seqlen[_batch] - self.next_n + _off + 1
                _num_ctas = (
                    _eff + self.chunk_size_per_cta - 1
                ) // self.chunk_size_per_cta
                if _num_ctas < 1:
                    _num_ctas = 1
                merge_width = _num_ctas * self.top_k
                row_end = merge_width
                length = merge_width
            else:
                # Existing fixed-length path
                # Note, after 1st kernel, the output is fix-lenght.
                # Note, for merge_block kernels, need to ensure max_num_cols is the same as bucketed_num_cols.
                row_end = self.max_num_cols
                length = self.max_num_cols

        # Skip CTAs that exceed this row's actual chunk count.
        _should_run = True
        if cutlass.const_expr(self.enable_dynamic_multi_cta):
            _batch_check = bidx // self.next_n
            _off_check = bidx % self.next_n
            _eff_check = seqlen[_batch_check] - self.next_n + _off_check + 1
            _needed_ctas = (
                _eff_check + self.chunk_size_per_cta - 1
            ) // self.chunk_size_per_cta
            if _needed_ctas < 1:
                _needed_ctas = 1
            _should_run = (bidx < num_rows_val) and (bidy < _needed_ctas)
        if cutlass.const_expr(self.single_pass_multi_cta):
            # Solo fast path (needed_ctas == 1): only cta_in_group 0 has data;
            # the rest exit silently. Because the branch is cluster-uniform
            # (all CTAs of a cluster compute the same need_cluster_sync) no CTA
            # waits on a cluster barrier, so this cannot deadlock. In cluster
            # mode (need_cluster_sync) every CTA must run (no early exit).
            if (not need_cluster_sync) and cta_in_group != 0:
                _should_run = False

        if _should_run:
            self.filtered_topk_kernel_per_row(
                input,
                indices,
                extra_buffer,
                output_indices,
                output_values,
                row_start,
                length,
                bidx,
                s_histogram,
                s_counter,
                s_threshold_bin_id,
                s_num_input,
                g_num_input,
                s_indices,
                s_input_idx,
                s_input_val,
                s_last_remain,
                num_warps,
                s_warp_sums,
                s_overflow_flag,
                need_cluster_sync,
                s_hist_merged,
                cta_in_group,
                tma_atom=tma_atom,
                tma_tensor=tma_tensor,
                s_tma_stage=s_tma_stage,
                s_tma_mbar=s_tma_mbar,
                s_tma_stage_p3=s_tma_stage_p3,
                s_tma_mbar_p3=s_tma_mbar_p3,
            )

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        input_values,
        indices,
        extra_buffer,
        seqlen,
        output_indices,
        output_values,
        stream: cuda.CUstream,
        min_blocks_per_mp: cutlass.Constexpr[int] = 1,
    ):
        """Host function for the filtered topk kernel"""
        num_rows = input_values.shape[0]
        if cutlass.const_expr(self.single_pass_multi_cta):
            # 1D grid = num_rows * ctas_per_group; each cluster owns one row.
            blocks = (num_rows * self.num_ctas_per_row, 1, 1)
            cluster = (self.num_ctas_per_row, 1, 1)
        else:
            blocks = (num_rows, self.num_ctas_per_row, 1)
            cluster = None

        # p1/p3 async-TMA: build the cp.async.bulk (UTMALDG) atom on the 2D input; the
        # per-CTA tile is one coarse-scan iteration's columns (num_threads * vec_size),
        # loaded at runtime coord (row=bidx, col_tile). Off by default.
        if cutlass.const_expr(self.enable_tma_load or self.enable_tma_load_p3):
            tile_cols = self.num_threads_per_cta * self.vec_size
            tma_smem_layout = cute.make_layout((1, tile_cols), stride=(tile_cols, 1))
            tma_cta_tiler = cute.product_each(tma_smem_layout.shape)
            tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                input_values,
                tma_smem_layout,
                tma_cta_tiler,
            )
        else:
            tma_atom, tma_tensor = None, None

        self.filtered_topk_kernel(
            input_values,
            indices,
            extra_buffer,
            seqlen,
            output_indices,
            output_values,
            tma_atom,
            tma_tensor,
        ).launch(
            grid=blocks,
            block=(self.num_threads_per_cta, 1, 1),
            cluster=cluster,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=min_blocks_per_mp,
            # Outside large-occupancy the grid gives at most 1 block/SM, so ask
            # for the smallest carveout (0) and let the driver pick a tier that
            # fits one block instead of starving L1 for memory it cannot use.
            preferred_smem_carveout=None if self._large_occupancy else 0,
        )
        return


def _next_positive_power_of_2(x: int) -> int:
    """Round up to the next power of 2 (returns x if already a power of 2)."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _bucket_num_cols(num_cols: int) -> int:
    """Bucket num_cols to the next power of 2 for compilation caching.

    This reduces recompilations when num_cols changes slightly (e.g.,
    KV cache length growing each decode step). Safe because num_cols
    only affects compile-time config; actual data access is bounded
    by seq_lens.
    """
    return _next_positive_power_of_2(num_cols)


# This function is used for integration of framework, e.g. trtllm.
compiled_filter_topk_dict = {}

_TOPK_DECODE_WRAPPER_MODES = (
    "single-cta",
    "single-pass-multi-cta",
    "multi-pass-multi-cta",
)
_SM100_MAX_CLUSTER_SIZE = 16


def auto_cluster_size(
    num_tokens: int,
    num_rows: int,
    is_fp32: bool,
    num_sms: int | None = None,
) -> int:
    """Choose a one-pass cluster size."""
    num_sms = num_sms or _get_num_sms()
    peak = (
        1
        if num_tokens <= 8192 or (is_fp32 and num_tokens <= 16384)
        else 4
        if num_tokens <= 32768
        else 8
        if num_tokens <= 131072
        else 16
    )
    occupancy = (
        16
        if num_rows <= 4
        else 8
        if num_rows <= 8
        else 4
        if num_rows <= 32
        else 2
        if num_rows <= 64
        else 1
        if num_rows <= num_sms
        else 2
        if num_tokens >= 262144
        else 1
    )
    return min(peak, occupancy, _SM100_MAX_CLUSTER_SIZE)


class _TopKDecodeLaunch:
    """A compiled top-k launch and its preallocated runtime workspace."""

    def __init__(
        self,
        launcher,
        arguments: testing.JitArguments,
        output_indices: torch.Tensor,
        output_values: torch.Tensor | None,
        workspace_tensors: tuple[torch.Tensor | None, ...],
    ):
        self.launcher = launcher
        self.arguments = arguments
        self.output_indices = output_indices
        self.output_values = output_values
        self.workspace_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in workspace_tensors
            if tensor is not None
        )

    def run(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.launcher(*self.arguments.args, **self.arguments.kwargs)
        return self.output_indices, self.output_values


def _prepare_one_pass_topk(
    input_values,
    seq_lens,
    top_k,
    next_n,
    cluster_size: int | None = None,
    return_val=True,
    num_copy_bits=256,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool | None = None,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
    out_indices=None,
    out_values=None,
):
    torch_dtype = input_values.dtype
    dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
    num_rows, num_cols = input_values.shape
    bucketed_num_cols = _bucket_num_cols(num_cols)

    if cluster_size is None:
        cluster_size = auto_cluster_size(
            num_cols,
            num_rows,
            dtype == cutlass.Float32,
            num_sms=_get_num_sms(input_values.device),
        )
    if cluster_size <= 0 or cluster_size > _SM100_MAX_CLUSTER_SIZE:
        raise ValueError(
            f"cluster_size must be in [1, {_SM100_MAX_CLUSTER_SIZE}], "
            f"got {cluster_size}"
        )

    single_pass_multi_cta = cluster_size > 1
    large_occupancy = not single_pass_multi_cta and num_rows > _get_num_sms(
        input_values.device
    )
    architecture, large_occupancy_min_blocks_per_mp = get_topk_architecture_config()
    min_blocks_per_mp = large_occupancy_min_blocks_per_mp if large_occupancy else 1
    chunk_size_per_cta = (
        math.ceil(bucketed_num_cols / cluster_size) if single_pass_multi_cta else None
    )

    # Arch-gated tuned defaults (None == "use the tuned default for this arch").
    # Rubin: SMEM value caching on + bytes-in-flight auto unroll. Blackwell:
    # baseline (caching off, no unroll), keeping SM100 non-regressing.
    is_rubin = architecture in _RUBIN_TOPK_ARCHITECTURES
    if cache_smem_values is None:
        cache_smem_values = is_rubin
    if unroll_factor is None:
        unroll_factor = 0 if is_rubin else 1

    # async-TMA auto-enable (large-occupancy single-CTA path only): the tuned default
    # (fp32 on Rubin at large N) fires only when the actual num_cols meets the TMA
    # row-stride alignment -- divisible by 16 // dtype_bytes (fp32 -> 4, fp16/bf16 -> 8)
    # -- so non-conforming shapes fall back to LDG instead of the descriptor ValueError.
    # Env vars still force TMA regardless. Resolved flags go into the compile key so the
    # same bucketed shape with different divisibility does not collide.
    _tma_div = 16 // (dtype.width // 8)
    _tma_ok = large_occupancy and not single_pass_multi_cta
    _tuned_tma = (
        tma_tuned_default(dtype, architecture, bucketed_num_cols)
        and (num_cols % _tma_div == 0)
        # With the symbolic leading stride, num_cols divisibility alone no
        # longer proves the TMA row stride is 16-byte aligned: a padded view
        # (stride(0) > num_cols) can carry a misaligned byte stride that
        # cuTensorMapEncodeTiled rejects. The auto default therefore also
        # requires an aligned leading stride and falls back to LDG otherwise.
        # DIVERGENCE FROM UPSTREAM (compact-only fake), after review
        # (flashinfer PR #4621).
        and (input_values.stride(0) % _tma_div == 0)
        # TMA aliases the candidate buffer; REREAD_ALWAYS allocates none, so the
        # auto default must skip it (env-forced TMA still hits the explicit guard).
        and overflow_policy != "REREAD_ALWAYS"
    )
    _enable_tma_load = _tma_ok and auto_tma_load(enable_tma_load, _tuned_tma)
    _enable_tma_load_p3 = _tma_ok and auto_tma_load(enable_tma_load_p3, _tuned_tma)
    if (_enable_tma_load or _enable_tma_load_p3) and (
        input_values.stride(0) % _tma_div != 0
    ):
        # Only reachable when TMA is forced explicitly (argument or env var):
        # fail fast with an actionable message instead of the opaque
        # cuTensorMapEncodeTiled error from inside the launch.
        raise ValueError(
            f"TMA load requires the input leading stride to be 16-byte "
            f"aligned (divisible by {_tma_div} elements for this dtype); got "
            f"stride(0)={input_values.stride(0)}. Disable enable_tma_load or "
            f"pass a compact/aligned input."
        )

    # Kernel-writable destinations. Caller-provided buffers are used
    # directly (the kernel writes every slot, including -1 / -inf padding,
    # so no stale data survives); otherwise allocate on the INPUT device --
    # not the ambient one -- so multi-GPU callers get outputs next to their
    # data. DIVERGENCE FROM UPSTREAM (internal-only allocation), after
    # review (flashinfer PR #4621): threading the buffers through avoids a
    # full num_rows x top_k allocate+copy per public call and gives
    # CUDA-graph users stable destinations.
    def _as_out(buf, dtype_, what):
        if buf is None:
            return None
        if not (buf.is_cuda and buf.dtype == dtype_):
            raise ValueError(f"{what} must be a CUDA {dtype_} tensor")
        if buf.device != input_values.device:
            raise ValueError(
                f"{what} must be on the input device {input_values.device}, "
                f"got {buf.device}; the kernel launches on the input's device "
                f"and cannot write a foreign-device buffer"
            )
        if buf.numel() != num_rows * top_k or not buf.is_contiguous():
            raise ValueError(
                f"{what} must be contiguous with numel == num_rows * top_k "
                f"({num_rows * top_k}), got numel={buf.numel()}"
            )
        return buf.view(num_rows, top_k)

    output_indices_torch = _as_out(out_indices, torch.int32, "out_indices")
    if output_indices_torch is None:
        output_indices_torch = torch.empty(
            num_rows, top_k, dtype=torch.int32, device=input_values.device
        )
    if return_val:
        output_values_torch = _as_out(out_values, torch_dtype, "out_values")
        if output_values_torch is None:
            output_values_torch = torch.empty(
                num_rows, top_k, dtype=torch_dtype, device=input_values.device
            )
    else:
        output_values_torch = None

    key = (
        _compile_cc(input_values.device),
        "single-pass-multi-cta" if single_pass_multi_cta else "single-cta",
        dtype,
        bucketed_num_cols,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        cluster_size if single_pass_multi_cta else None,
        chunk_size_per_cta,
        large_occupancy,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
        _enable_tma_load,
        _enable_tma_load_p3,
    )
    if key not in compiled_filter_topk_dict:
        n_rows = cute.sym_int()
        _tma_on = _enable_tma_load or _enable_tma_load_p3
        n_cols = cute.sym_int(divisibility=_tma_div) if _tma_on else cute.sym_int()
        n_batch = cute.sym_int()
        # Symbolic leading stride: a padded row view (stride0 > n_cols, e.g.
        # a framework's score buffer sliced to the vocab width) is part of
        # the declared ABI, zero-copy. Inner stride stays 1 and the base
        # stays 32-byte aligned; the public wrapper materializes inputs
        # violating those. When TMA is enabled the leading stride also
        # declares the 16-byte divisibility the descriptor needs, so a
        # misaligned runtime stride is rejected at argument check rather
        # than inside cuTensorMapEncodeTiled. DIVERGENCE FROM UPSTREAM
        # (compact-only fake), after review (flashinfer PR #4621).
        input_fake = cute.runtime.make_fake_tensor(
            dtype,
            (n_rows, n_cols),
            stride=(
                cute.sym_int64(divisibility=_tma_div) if _tma_on else cute.sym_int64(),
                1,
            ),
            assumed_align=32,
        )
        if overflow_policy in ("GMEM_SPILL", "BOUNDED_SPILL"):
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (cute.sym_int(), cute.sym_int(), cute.sym_int()),
                stride_order=(2, 1, 0),
                assumed_align=32,
            )
        else:
            buffer_fake = None
        seqlen_fake = cute.runtime.make_fake_compact_tensor(
            cute.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows, top_k),
            stride_order=(1, 0),
        )
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
        else:
            output_values_fake = None
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        if single_pass_multi_cta:
            filtered_topk_func = FilteredTopKKernelVarlenDecode(
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                chunk_size_per_cta=chunk_size_per_cta,
                num_ctas_per_row=cluster_size,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
                single_pass_multi_cta=True,
                architecture=architecture,
                unroll_factor=unroll_factor,
            )
        else:
            filtered_topk_func = FilteredTopKKernelVarlenDecode(
                dtype,
                bucketed_num_cols,
                top_k,
                next_n,
                num_copy_bits=num_copy_bits,
                return_val=return_val,
                large_occupancy=large_occupancy,
                target_blocks_per_sm=min_blocks_per_mp,
                overflow_policy=overflow_policy,
                cache_smem_values=cache_smem_values,
                architecture=architecture,
                unroll_factor=unroll_factor,
                # num_cols-divisibility-gated resolution from above (explicit so the
                # kernel does not re-resolve the tuned default from bucketed_num_cols).
                enable_tma_load=_enable_tma_load,
                enable_tma_load_p3=_enable_tma_load_p3,
            )

        compiled_kernel = _persistent_compile(
            _kernel_name_from_key(key),
            lambda: cute.compile(
                filtered_topk_func,
                input_fake,
                None,  # indices_fake: unused in this path
                buffer_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=min_blocks_per_mp,
                options=f"--enable-tvm-ffi --gpu-arch={_dsl_gpu_arch(input_values.device)}",
            ),
        )
        compiled_filter_topk_dict[key] = compiled_kernel
    else:
        compiled_kernel = compiled_filter_topk_dict[key]

    if overflow_policy == "GMEM_SPILL":
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        buffer_rows = num_rows * cluster_size if single_pass_multi_cta else num_rows
        buffer_cols = chunk_size_per_cta if single_pass_multi_cta else num_cols
        buffer_torch = torch.empty(
            buffer_rows,
            buffer_numbers,
            buffer_cols,
            dtype=torch.int32,
            device="cuda",
        )
    elif overflow_policy == "BOUNDED_SPILL":
        # Size-capped spill buffer: last dim = G, host-chosen via spill_capacity
        # and/or a total-byte budget (stricter wins); kernel reads buffer.shape[1].
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        buffer_rows = num_rows * cluster_size if single_pass_multi_cta else num_rows
        _bs_cap = chunk_size_per_cta if single_pass_multi_cta else num_cols
        if spill_capacity is not None:
            # Fail fast on a degenerate cap instead of silently clamping (0/negative
            # is meaningless -- use overflow_policy=REREAD for no GMEM tier).
            assert spill_capacity >= 1, (
                f"spill_capacity must be >= 1, got {spill_capacity}; use "
                "overflow_policy=REREAD for no GMEM spill tier."
            )
            _bs_cap = min(_bs_cap, int(spill_capacity))
        if spill_budget_bytes is not None:
            _bytes_per_g = buffer_rows * buffer_numbers * 4
            assert spill_budget_bytes >= _bytes_per_g, (
                f"spill_budget_bytes ({spill_budget_bytes}) too small to fit even "
                f"one candidate per row; need >= {_bytes_per_g} bytes."
            )
            _bs_cap = min(_bs_cap, int(spill_budget_bytes) // _bytes_per_g)
        buffer_torch = torch.empty(
            buffer_rows,
            buffer_numbers,
            _bs_cap,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        buffer_torch = None

    return _TopKDecodeLaunch(
        compiled_kernel,
        testing.JitArguments(
            input_values,
            None,  # indices, used for merge blocks kernel of the multi-cta.
            buffer_torch,
            seq_lens,
            output_indices_torch,
            output_values_torch,
        ),
        output_indices_torch,
        output_values_torch,
        (
            input_values,
            buffer_torch,
            seq_lens,
            output_indices_torch,
            output_values_torch,
        ),
    )


def cute_dsl_radix_filter_topk_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    return_val=True,
    num_copy_bits=256,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool | None = None,
    cluster_size: int | None = None,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
    out_indices=None,
    out_values=None,
):
    """Compile and launch the one-pass decode wrapper.

    ``cluster_size=None`` selects the cluster size automatically. A resolved size
    of one uses the ordinary single-CTA setup.

    ``cache_smem_values`` / ``unroll_factor`` / ``enable_tma_load`` /
    ``enable_tma_load_p3`` default to ``None`` = the tuned best config for the
    detected arch (Rubin: SMEM caching + auto unroll; async-TMA for fp32 large-occ
    at N >= 131072). Pass explicit values to override for tuning.
    """
    return _prepare_one_pass_topk(
        input_values,
        seq_lens,
        top_k,
        next_n,
        cluster_size=cluster_size,
        return_val=return_val,
        num_copy_bits=num_copy_bits,
        overflow_policy=overflow_policy,
        cache_smem_values=cache_smem_values,
        unroll_factor=unroll_factor,
        enable_tma_load=enable_tma_load,
        enable_tma_load_p3=enable_tma_load_p3,
        spill_capacity=spill_capacity,
        spill_budget_bytes=spill_budget_bytes,
        out_indices=out_indices,
        out_values=out_values,
    ).run()


# KNOWN LIMITATION (flashinfer PR #4621 review): the merge stage scans the
# fixed num_ctas_per_row * top_k candidate width, which includes stage-one
# padding. Padded (-1, -inf) entries and genuinely valid -inf elements have
# identical radix keys, so when a row contains valid -inf values among its
# top-k, the merge can emit -1 for slots that had valid candidates. A correct
# fix needs either a pad encoding that sorts strictly below -inf (which would
# change the shared trivial-branch padding and the -inf output contract) or
# index-aware masking in the shared hot scan loops -- both cross-stage
# redesigns belonging upstream. This path is NOT reachable from
# flashinfer's public top_k_varlen (the wrapper uses the one-pass path
# exclusively); do not route public traffic here until fixed.
def _prepare_multi_pass_multi_cta_topk(
    input_values,
    seq_lens,
    top_k,
    next_n,
    return_val=True,
    num_copy_bits=256,
    chunk_size_per_cta=16384,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool | None = None,
    unroll_factor: int | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
):
    torch_dtype = input_values.dtype
    dtype = _TORCH_TO_CUTLASS_DTYPE[torch_dtype]
    num_rows, num_cols = input_values.shape
    bucketed_num_cols = _bucket_num_cols(num_cols)

    large_occupancy = num_rows > _get_num_sms(input_values.device)
    architecture, large_occupancy_min_blocks_per_mp = get_topk_architecture_config()
    min_blocks_per_mp = large_occupancy_min_blocks_per_mp if large_occupancy else 1

    # Arch-gated tuned defaults (None == "use the tuned default for this arch").
    is_rubin = architecture in _RUBIN_TOPK_ARCHITECTURES
    if cache_smem_values is None:
        cache_smem_values = is_rubin
    if unroll_factor is None:
        unroll_factor = 0 if is_rubin else 1

    # Note: don't forget num_cols, which means the maximum columns.
    enable_multi_cta = True
    num_ctas_per_row = math.ceil(num_cols / chunk_size_per_cta)
    key = (
        _compile_cc(input_values.device),
        "multi-pass-multi-cta",
        dtype,
        bucketed_num_cols,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        large_occupancy,
        enable_multi_cta,
        chunk_size_per_cta,
        num_ctas_per_row,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
    )
    if key not in compiled_filter_topk_dict:
        # Create fake tensors for compilation
        n_rows = cute.sym_int()
        n_cols = cute.sym_int()
        n_batch = cute.sym_int()
        # Symbolic leading stride: a padded row view (stride0 > n_cols, e.g.
        # a framework's score buffer sliced to the vocab width) is part of
        # the declared ABI, zero-copy. Inner stride stays 1 and the base
        # stays 32-byte aligned; the public wrapper materializes inputs
        # violating those. DIVERGENCE FROM UPSTREAM (compact-only fake),
        # after review (flashinfer PR #4621).
        input_fake = cute.runtime.make_fake_tensor(
            dtype,
            (n_rows, n_cols),
            stride=(cute.sym_int64(), 1),
            assumed_align=32,
        )
        # Shared buffer for both kernels: only needed when policy spills to GMEM
        if overflow_policy in ("GMEM_SPILL", "BOUNDED_SPILL"):
            buffer_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32,
                (cute.sym_int(), cute.sym_int(), cute.sym_int()),
                stride_order=(2, 1, 0),
                assumed_align=32,
            )
        else:
            buffer_fake = None
        seqlen_fake = cute.runtime.make_fake_compact_tensor(
            cute.Int32,
            (n_batch,),
            stride_order=(0,),
        )
        # The first stage writes a dynamic-width candidate list for the merge stage.
        n_first_output_cols = cute.sym_int()
        first_kernel_output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows, n_first_output_cols),
            stride_order=(1, 0),
        )
        first_kernel_output_values_fake = cute.runtime.make_fake_compact_tensor(
            dtype,
            (n_rows, n_first_output_cols),
            stride_order=(1, 0),
            assumed_align=32,
        )
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        filtered_topk_func_first = FilteredTopKKernelVarlenDecode(
            dtype,
            chunk_size_per_cta,  # num_cols
            top_k,
            next_n,
            num_copy_bits=num_copy_bits,
            # for the first kernel, it must return values.
            return_val=True,
            large_occupancy=large_occupancy,
            enable_multi_cta=True,
            chunk_size_per_cta=chunk_size_per_cta,
            num_ctas_per_row=num_ctas_per_row,
            merge_blocks=False,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
            target_blocks_per_sm=min_blocks_per_mp,
            architecture=architecture,
            unroll_factor=unroll_factor,
        )
        # Compile the kernel (persistently cached; see _persistent_compile)
        compiled_kernel_first = _persistent_compile(
            _kernel_name_from_key(key) + "_p1",
            lambda: cute.compile(
                filtered_topk_func_first,
                input_fake,
                None,  # indices_fake: unused in this path
                buffer_fake,
                seqlen_fake,
                first_kernel_output_indices_fake,
                first_kernel_output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=min_blocks_per_mp,
                options=f"--enable-tvm-ffi --gpu-arch={_dsl_gpu_arch(input_values.device)}",
            ),
        )

        # The merge stage consumes the first stage's candidate indices.
        indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows, n_first_output_cols),
            stride_order=(1, 0),
        )
        output_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (n_rows, top_k),
            stride_order=(1, 0),
        )
        if return_val:
            output_values_fake = cute.runtime.make_fake_compact_tensor(
                dtype,
                (n_rows, top_k),
                stride_order=(1, 0),
            )
        else:
            output_values_fake = None
        filtered_topk_func_second = FilteredTopKKernelVarlenDecode(
            dtype,
            num_ctas_per_row * top_k,  # num_cols
            top_k,
            next_n,
            num_copy_bits=num_copy_bits,
            return_val=return_val,
            large_occupancy=large_occupancy,
            enable_multi_cta=False,
            # chunk_size_per_cta=chunk_size_per_cta, # no use
            # num_ctas_per_row=1, # no use
            merge_blocks=True,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
            target_blocks_per_sm=min_blocks_per_mp,
            architecture=architecture,
            unroll_factor=unroll_factor,
        )
        # Compile the kernel (persistently cached; see _persistent_compile)
        compiled_kernel_second = _persistent_compile(
            _kernel_name_from_key(key) + "_p2",
            lambda: cute.compile(
                filtered_topk_func_second,
                input_fake,
                indices_fake,
                buffer_fake,
                seqlen_fake,
                output_indices_fake,
                output_values_fake,
                stream=fake_stream,
                min_blocks_per_mp=min_blocks_per_mp,
                options=f"--enable-tvm-ffi --gpu-arch={_dsl_gpu_arch(input_values.device)}",
            ),
        )

        compiled_filter_topk_dict[key] = (compiled_kernel_first, compiled_kernel_second)
    else:
        compiled_kernel_first, compiled_kernel_second = compiled_filter_topk_dict[key]

    first_kernel_output_indices_torch = torch.empty(
        num_rows, num_ctas_per_row * top_k, dtype=torch.int32, device="cuda"
    )
    first_kernel_output_values_torch = torch.empty(
        num_rows, num_ctas_per_row * top_k, dtype=torch_dtype, device="cuda"
    )
    output_indices_torch = torch.empty(
        num_rows, top_k, dtype=torch.int32, device="cuda"
    )
    if return_val:
        output_values_torch = torch.empty(
            num_rows, top_k, dtype=torch_dtype, device="cuda"
        )
    else:
        output_values_torch = None

    if overflow_policy == "GMEM_SPILL":
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        buffer_torch = torch.empty(
            num_rows * num_ctas_per_row,
            buffer_numbers,
            max(chunk_size_per_cta, num_ctas_per_row * top_k),
            dtype=torch.int32,
            device="cuda",
        )
    elif overflow_policy == "BOUNDED_SPILL":
        # Size-capped spill buffer (last dim = G); host caps G via spill_capacity
        # and/or spill_budget_bytes (stricter wins). Kernel reads buffer.shape[1].
        buffer_numbers = 2 if dtype == cutlass.Float32 else 1
        _bs_rows = num_rows * num_ctas_per_row
        _bs_cap = max(chunk_size_per_cta, num_ctas_per_row * top_k)
        if spill_capacity is not None:
            assert spill_capacity >= 1, (
                f"spill_capacity must be >= 1, got {spill_capacity}; use "
                "overflow_policy=REREAD for no GMEM spill tier."
            )
            _bs_cap = min(_bs_cap, int(spill_capacity))
        if spill_budget_bytes is not None:
            _bytes_per_g = _bs_rows * buffer_numbers * 4
            assert spill_budget_bytes >= _bytes_per_g, (
                f"spill_budget_bytes ({spill_budget_bytes}) too small to fit even "
                f"one candidate per row; need >= {_bytes_per_g} bytes."
            )
            _bs_cap = min(_bs_cap, int(spill_budget_bytes) // _bytes_per_g)
        buffer_torch = torch.empty(
            _bs_rows,
            buffer_numbers,
            _bs_cap,
            dtype=torch.int32,
            device="cuda",
        )
    else:
        buffer_torch = None

    def launch_multi_pass(
        first_kernel_args: testing.JitArguments,
        second_kernel_args: testing.JitArguments,
    ):
        compiled_kernel_first(
            *first_kernel_args.args,
            **first_kernel_args.kwargs,
        )
        compiled_kernel_second(
            *second_kernel_args.args,
            **second_kernel_args.kwargs,
        )

    first_kernel_args = testing.JitArguments(
        input_values,
        None,  # indices, used for merge blocks kernel of the multi-cta.
        buffer_torch,
        seq_lens,
        first_kernel_output_indices_torch,
        first_kernel_output_values_torch,
    )
    second_kernel_args = testing.JitArguments(
        first_kernel_output_values_torch,
        first_kernel_output_indices_torch,
        buffer_torch,
        seq_lens,
        output_indices_torch,
        output_values_torch,
    )
    return _TopKDecodeLaunch(
        launch_multi_pass,
        testing.JitArguments(first_kernel_args, second_kernel_args),
        output_indices_torch,
        output_values_torch,
        (
            input_values,
            buffer_torch,
            seq_lens,
            first_kernel_output_indices_torch,
            first_kernel_output_values_torch,
            output_indices_torch,
            output_values_torch,
        ),
    )


def cute_dsl_radix_filter_topk_multi_cta_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    return_val=True,
    num_copy_bits=256,
    chunk_size_per_cta=None,
    overflow_policy: str = "REREAD",
    cache_smem_values: bool | None = None,
    unroll_factor: int | None = None,
):
    """Compile and launch the two-pass multi-CTA decode wrapper.

    ``cache_smem_values`` / ``unroll_factor`` default to ``None``, which selects
    the tuned best configuration for the detected architecture (Rubin: SMEM value
    caching + bytes-in-flight auto unroll; Blackwell: baseline).
    """
    if chunk_size_per_cta is None:
        chunk_size_per_cta = _multi_pass_chunk_size(top_k)

    return _prepare_multi_pass_multi_cta_topk(
        input_values,
        seq_lens,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        chunk_size_per_cta,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
    ).run()


def _multi_pass_chunk_size(top_k: int) -> int:
    """Choose a first-stage chunk that can emit top_k candidates."""
    return max(8192, _next_positive_power_of_2(top_k))


def _run_topk_decode_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    wrapper_mode,
    cluster_size,
    return_val,
    num_copy_bits,
    overflow_policy,
    cache_smem_values,
    unroll_factor,
    enable_tma_load=None,
    enable_tma_load_p3=None,
    spill_capacity=None,
    spill_budget_bytes=None,
):
    if wrapper_mode != "multi-pass-multi-cta":
        return cute_dsl_radix_filter_topk_wrapper(
            input_values,
            seq_lens,
            top_k,
            next_n,
            return_val=return_val,
            num_copy_bits=num_copy_bits,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
            cluster_size=1 if wrapper_mode == "single-cta" else cluster_size,
            unroll_factor=unroll_factor,
            enable_tma_load=enable_tma_load,
            enable_tma_load_p3=enable_tma_load_p3,
            spill_capacity=spill_capacity,
            spill_budget_bytes=spill_budget_bytes,
        )
    return cute_dsl_radix_filter_topk_multi_cta_wrapper(
        input_values,
        seq_lens,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        _multi_pass_chunk_size(top_k),
        overflow_policy,
        cache_smem_values,
        unroll_factor,
    )


def _prepare_topk_decode_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n,
    wrapper_mode,
    cluster_size,
    return_val,
    num_copy_bits,
    overflow_policy,
    cache_smem_values,
    unroll_factor,
    enable_tma_load=None,
    enable_tma_load_p3=None,
    spill_capacity=None,
    spill_budget_bytes=None,
):
    if wrapper_mode != "multi-pass-multi-cta":
        return _prepare_one_pass_topk(
            input_values,
            seq_lens,
            top_k,
            next_n,
            cluster_size=1 if wrapper_mode == "single-cta" else cluster_size,
            return_val=return_val,
            num_copy_bits=num_copy_bits,
            overflow_policy=overflow_policy,
            cache_smem_values=cache_smem_values,
            unroll_factor=unroll_factor,
            enable_tma_load=enable_tma_load,
            enable_tma_load_p3=enable_tma_load_p3,
            spill_capacity=spill_capacity,
            spill_budget_bytes=spill_budget_bytes,
        )
    return _prepare_multi_pass_multi_cta_topk(
        input_values,
        seq_lens,
        top_k,
        next_n,
        return_val,
        num_copy_bits,
        _multi_pass_chunk_size(top_k),
        overflow_policy,
        cache_smem_values,
        unroll_factor,
    )


def generate_seq_lens(batch_size, num_tokens):
    """Draw decode sequence lengths uniformly from [1, num_tokens].

    A sizeable fraction of rows therefore ends up shorter than top_k, which is
    the degenerate extent the kernel must still handle correctly.
    """
    return torch.randint(
        1, num_tokens + 1, (batch_size,), dtype=torch.int32, device="cuda"
    )


def run_filtered_topk_decode(
    dtype: type[cutlass.Numeric],
    batch_size: int,
    max_num_cols: int,
    top_k: int,
    next_n: int,
    num_copy_bits: int = 256,
    return_val: bool = True,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    print_verbose: bool = True,
    overflow_policy: str = "GMEM_SPILL",
    cache_smem_values: bool = False,
    variable_lengths: bool = True,
    wrapper_mode: str = "single-cta",
    cluster_size: int | None = None,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
) -> None:
    """Run correctness checks and benchmarks through one decode wrapper."""
    if wrapper_mode not in _TOPK_DECODE_WRAPPER_MODES:
        raise ValueError(
            f"wrapper_mode must be one of {_TOPK_DECODE_WRAPPER_MODES}, "
            f"got {wrapper_mode!r}"
        )
    if print_verbose:
        print("=" * 60)
        print("Launching Blackwell Filtered TopK Test")
        print("-" * 60)
        print(f"Data Types & Precision: {dtype}")
        print(f"    Input matrix: {dtype}")
        print(f"    Output indices: {cutlass.Int32}")
        print(f"    Output values: {dtype}")
        print(
            f"Input dimensions (batch_size, max_num_cols, top_k): {batch_size, max_num_cols, top_k}"
        )
        print(f"    batch_size: {batch_size}")
        print(f"    next_n: {next_n}")
        print(f"    max_num_cols: {max_num_cols}")
        print(f"    top_k: {top_k}")
        print(f"    num_copy_bits: {num_copy_bits}")
        print(f"    return_val: {return_val}")
        print(f"    variable_lengths: {variable_lengths}")
        print(f"    wrapper_mode: {wrapper_mode}")
        if wrapper_mode == "single-pass-multi-cta":
            print(
                f"    cluster_size: {'auto' if cluster_size is None else cluster_size}"
            )
        elif wrapper_mode == "multi-pass-multi-cta":
            print(f"    chunk_size_per_cta: {_multi_pass_chunk_size(top_k)}")
        print(f"Do reference checking: {do_ref_check}")
        print(f"Do benchmark: {do_benchmark}")
        print(f"Warmup iterations: {warmup_iterations}")
        print(f"Iterations: {iterations}")
        print(f"Use cold L2: {use_cold_l2}")
        print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    seed = 1111
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set input data
    # num_gen_tokens is the number of rows in the input tensor
    torch.cuda.synchronize()
    num_gen_tokens = batch_size * next_n  # Use the same variable name as dsa.py
    row_starts = torch.zeros(num_gen_tokens, dtype=torch.int32, device="cuda")
    row_indices = torch.arange(num_gen_tokens, device="cuda") // next_n
    next_n_offset = torch.arange(num_gen_tokens, device="cuda") % next_n

    # max_num_cols is the maximum col length in the input tensor
    if variable_lengths:
        # clamp(min=next_n) keeps the row_ends arithmetic below well formed and
        # is a no-op for next_n == 1.
        seq_lens = generate_seq_lens(batch_size, max_num_cols).clamp(min=next_n)
    else:
        seq_lens = torch.full(
            (batch_size,), max_num_cols, dtype=torch.int32, device="cuda"
        )
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    row_ends = row_ends.to(torch.int32)

    def create_input_tensor() -> torch.Tensor:
        input_tensor = create_random_logits(
            row_starts,
            row_ends,
            torch_dtype(dtype),
            seed,
        )
        if input_tensor.shape[1] == max_num_cols:
            return input_tensor

        padded_input = torch.full(
            (num_gen_tokens, max_num_cols),
            float("-inf"),
            dtype=torch_dtype(dtype),
            device="cuda",
        )
        padded_input[:, : input_tensor.shape[1]] = input_tensor
        return padded_input

    input_torch = create_input_tensor()
    output_indices_torch, output_values_torch = _run_topk_decode_wrapper(
        input_torch,
        seq_lens,
        top_k,
        next_n,
        wrapper_mode,
        cluster_size,
        return_val,
        num_copy_bits,
        overflow_policy,
        cache_smem_values,
        unroll_factor,
        enable_tma_load,
        enable_tma_load_p3,
        spill_capacity,
        spill_budget_bytes,
    )

    if do_ref_check:
        torch.cuda.synchronize()
        torch_indices = run_reference_top_k(input_torch, row_starts, row_ends, top_k)
        assert compare_top_k_results(
            input_torch,
            output_indices_torch,
            torch_indices,
            row_starts,
            row_ends,
            top_k,
        ), f"{wrapper_mode} results don't match torch.topk"

        if return_val:
            valid_mask = output_indices_torch != -1
            output_rows = torch.arange(num_gen_tokens, device="cuda")[:, None]
            output_rows = output_rows.expand_as(output_indices_torch)
            absolute_indices = output_indices_torch + row_starts[:, None]
            expected_values = input_torch[
                output_rows[valid_mask],
                absolute_indices[valid_mask],
            ]
            assert torch.allclose(
                output_values_torch[valid_mask],
                expected_values,
                rtol=1e-5,
                atol=1e-5,
            ), f"{wrapper_mode} returned values that do not match its indices"
            assert torch.isneginf(output_values_torch[~valid_mask]).all(), (
                f"{wrapper_mode} did not write -inf for invalid outputs"
            )

        if print_verbose:
            print(f"{wrapper_mode}: PASSED")

    if do_benchmark:
        torch.cuda.synchronize()
        benchmark_launch = _prepare_topk_decode_wrapper(
            input_torch,
            seq_lens,
            top_k,
            next_n,
            wrapper_mode,
            cluster_size,
            return_val,
            num_copy_bits,
            overflow_policy,
            cache_smem_values,
            unroll_factor,
            enable_tma_load,
            enable_tma_load_p3,
            spill_capacity,
            spill_budget_bytes,
        )
        use_prepared_workspace = True

        def generate_inputs() -> testing.JitArguments:
            nonlocal use_prepared_workspace
            if use_prepared_workspace:
                use_prepared_workspace = False
                return benchmark_launch.arguments

            input_tensor = create_input_tensor()
            return _prepare_topk_decode_wrapper(
                input_tensor,
                seq_lens.clone(),
                top_k,
                next_n,
                wrapper_mode,
                cluster_size,
                return_val,
                num_copy_bits,
                overflow_policy,
                cache_smem_values,
                unroll_factor,
                enable_tma_load,
                enable_tma_load_p3,
                spill_capacity,
                spill_budget_bytes,
            ).arguments

        workspace_count = (
            testing.get_workspace_count(
                benchmark_launch.workspace_bytes,
                warmup_iterations,
                iterations,
            )
            if use_cold_l2
            else 1
        )
        if print_verbose:
            print(f"Workspace count: {workspace_count}")
        torch_stream = torch.cuda.Stream()
        benchmark_stream = cuda.CUstream(torch_stream.cuda_stream)
        with torch.cuda.stream(torch_stream):
            time = testing.benchmark(
                benchmark_launch.launcher,
                workspace_generator=generate_inputs,
                workspace_count=workspace_count,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                use_cuda_graphs=True,
                stream=benchmark_stream,
            )
        if print_verbose:
            print(f"Time: {time} us")
        print(f"{wrapper_mode}-{dtype}-{batch_size}-{max_num_cols}-{top_k} {time}")
    torch.cuda.synchronize()


def run_topk_decode(
    dtype: type[cutlass.Numeric],
    batch_size: int,
    max_num_cols: int,
    top_k: int,
    next_n: int,
    num_copy_bits: int = 256,
    return_val: bool = True,
    do_ref_check: bool = True,
    do_benchmark: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    overflow_policy: str = "GMEM_SPILL",
    cache_smem_values: bool = False,
    variable_lengths: bool = True,
    wrapper_mode: str = "single-cta",
    cluster_size: int | None = None,
    unroll_factor: int | None = None,
    enable_tma_load: bool | None = None,
    enable_tma_load_p3: bool | None = None,
    spill_capacity: int | None = None,
    spill_budget_bytes: int | None = None,
) -> None:
    run_filtered_topk_decode(
        dtype=dtype,
        batch_size=batch_size,
        max_num_cols=max_num_cols,
        top_k=top_k,
        next_n=next_n,
        num_copy_bits=num_copy_bits,
        return_val=return_val,
        do_ref_check=do_ref_check,
        do_benchmark=do_benchmark,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cold_l2=use_cold_l2,
        overflow_policy=overflow_policy,
        cache_smem_values=cache_smem_values,
        variable_lengths=variable_lengths,
        wrapper_mode=wrapper_mode,
        cluster_size=cluster_size,
        unroll_factor=unroll_factor,
        enable_tma_load=enable_tma_load,
        enable_tma_load_p3=enable_tma_load_p3,
        spill_capacity=spill_capacity,
        spill_budget_bytes=spill_budget_bytes,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Blackwell CuTE DSL filtered top-k decode benchmark."
    )
    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.Float32,
        choices=[cutlass.Float32, cutlass.Float16, cutlass.BFloat16],
        help="Data type of the input matrix",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch_size",
    )
    parser.add_argument("--max_num_cols", type=int, default=4096, help="max_num_cols")
    parser.add_argument("--next_n", type=int, default=3, help="next_n")
    parser.add_argument("--top_k", type=int, default=2048, help="top_k")
    parser.add_argument(
        "--num_copy_bits",
        type=int,
        default=256,
        help="num_copy_bits, used for vectorization",
    )
    parser.add_argument(
        "--return_val",
        action="store_true",
        default=False,
        help="Return values",
    )
    parser.add_argument(
        "--wrapper-mode",
        "--wrapper_mode",
        choices=_TOPK_DECODE_WRAPPER_MODES,
        default="single-cta",
        help="Decode wrapper implementation to run",
    )
    parser.add_argument(
        "--cluster-size",
        "--cluster_size",
        type=int,
        default=None,
        help=(
            "CTA cluster size for single-pass-multi-cta; automatically selected "
            f"when omitted (maximum {_SM100_MAX_CLUSTER_SIZE})"
        ),
    )
    parser.add_argument(
        "--do_ref_check",
        action="store_true",
        default=False,
        help="Do reference checking",
    )
    parser.add_argument(
        "--benchmark",
        "--do_benchmark",
        dest="do_benchmark",
        action="store_true",
        default=False,
        help="Run the benchmark",
    )
    parser.add_argument(
        "--warmup",
        "--warmup_iterations",
        dest="warmup_iterations",
        type=int,
        default=10,
        help="Warmup iterations",
    )
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument(
        "--no-cold-l2",
        dest="use_cold_l2",
        action="store_false",
        help="Disable cold L2 cache simulation",
    )
    parser.add_argument(
        "--use_cold_l2",
        dest="use_cold_l2",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(use_cold_l2=True)
    parser.add_argument(
        "--overflow_policy",
        type=str,
        default="GMEM_SPILL",
        choices=["GMEM_SPILL", "TRUNCATE", "REREAD", "REREAD_ALWAYS", "BOUNDED_SPILL"],
        help="Overflow policy when candidates exceed SMEM capacity",
    )
    parser.add_argument(
        "--spill_capacity",
        type=int,
        default=None,
        help="BOUNDED_SPILL: per-row GMEM candidate capacity G (upper bound)",
    )
    parser.add_argument(
        "--spill_budget_bytes",
        type=int,
        default=None,
        help="BOUNDED_SPILL: total GMEM scratch byte budget; caps G to fit the budget",
    )
    parser.add_argument(
        "--cache_smem_values",
        action="store_true",
        default=False,
        help="Cache ordered values alongside indices in SMEM to avoid re-reading from GMEM in refinement rounds",
    )
    parser.add_argument(
        "--length_mode",
        "--length-mode",
        dest="length_mode",
        type=str,
        default="fixlen",
        choices=["fixlen", "varlen"],
        help=(
            "Sequence length distribution: fixlen gives every sequence "
            "max_num_cols; varlen draws seq_lens uniformly from "
            "[1, max_num_cols], so many rows end up shorter than top_k"
        ),
    )

    def _parse_unroll_factor(value):
        # "auto" -> None: pick the arch-tuned default (Rubin: bytes-in-flight
        # auto; Blackwell: baseline uf=1). Otherwise an explicit 1/2/4/8.
        if value == "auto":
            return None
        try:
            iv = int(value)
        except ValueError:
            iv = None
        if iv not in (1, 2, 4, 8):
            raise argparse.ArgumentTypeError(
                f"--unroll_factor must be 'auto' or one of 1,2,4,8; got {value!r}"
            )
        return iv

    parser.add_argument(
        "--unroll_factor",
        "--unroll-factor",
        dest="unroll_factor",
        type=_parse_unroll_factor,
        default=None,
        metavar="{auto,1,2,4,8}",
        help=(
            "load-instruction unroll factor for the coarse/filter input scans. Default 'auto' "
            "picks the arch-tuned factor (bytes-in-flight on Rubin, baseline uf=1 "
            "on Blackwell). 1 = original baseline (1-in-flight while); 2/4/8 "
            "overlap that many loads for memory-level parallelism"
        ),
    )

    def _parse_tma(value):
        # auto -> None (tuned default: async-TMA for fp32 large-occ at N>=131072).
        return {"auto": None, "on": True, "off": False}[value]

    parser.add_argument(
        "--enable_tma_load",
        "--enable-tma-load",
        dest="enable_tma_load",
        type=_parse_tma,
        default=None,
        metavar="{auto,on,off}",
        help="p1 async-TMA load (large-occupancy path only). auto = tuned default.",
    )
    parser.add_argument(
        "--enable_tma_load_p3",
        "--enable-tma-load-p3",
        dest="enable_tma_load_p3",
        type=_parse_tma,
        default=None,
        metavar="{auto,on,off}",
        help="p3 async-TMA load (large-occupancy path only). auto = tuned default.",
    )

    args = parser.parse_args()
    run_topk_decode(
        variable_lengths=(args.length_mode == "varlen"),
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_num_cols=args.max_num_cols,
        top_k=args.top_k,
        next_n=args.next_n,
        num_copy_bits=args.num_copy_bits,
        return_val=args.return_val,
        do_ref_check=args.do_ref_check,
        do_benchmark=args.do_benchmark,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        use_cold_l2=args.use_cold_l2,
        overflow_policy=args.overflow_policy,
        spill_capacity=args.spill_capacity,
        spill_budget_bytes=args.spill_budget_bytes,
        cache_smem_values=args.cache_smem_values,
        wrapper_mode=args.wrapper_mode,
        cluster_size=args.cluster_size,
        unroll_factor=args.unroll_factor,
        enable_tma_load=args.enable_tma_load,
        enable_tma_load_p3=args.enable_tma_load_p3,
    )
