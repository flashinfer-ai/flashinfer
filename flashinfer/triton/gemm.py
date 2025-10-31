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

import triton
import triton.language as tl


@triton.jit
def compute_sm80_group_gemm_args(
    all_problems_ptr,
    x_ptr,
    w_ptr,
    y_ptr,
    x_ld_ptr,
    w_ld_ptr,
    y_ld_ptr,
    x,
    w,
    y,
    xy_indptr,
    w_indices,
    d_in,
    d_out,
    w_column_major,
):
    pid = tl.program_id(0)

    m = tl.load(xy_indptr + pid + 1) - tl.load(xy_indptr + pid)
    k, n = d_in, d_out

    tl.store(all_problems_ptr + pid * 3, m)
    tl.store(all_problems_ptr + pid * 3 + 1, n)
    tl.store(all_problems_ptr + pid * 3 + 2, k)

    w_i = tl.load(w_indices + pid) if w_indices else tl.cast(pid, tl.int64)
    w_curr_ptr = w + w_i * k * n
    tl.store(w_ptr + pid, w_curr_ptr)

    x_curr_ptr = x + tl.load(xy_indptr + pid) * k
    tl.store(x_ptr + pid, x_curr_ptr)

    y_curr_ptr = y + tl.load(xy_indptr + pid) * n
    tl.store(y_ptr + pid, y_curr_ptr)

    tl.store(x_ld_ptr + pid, k)
    tl.store(w_ld_ptr + pid, k if w_column_major else n)
    tl.store(y_ld_ptr + pid, n)


@triton.jit
def compute_sm90_group_gemm_args(
    all_problems_ptr,
    x_ptr,
    w_ptr,
    y_ptr,
    x_stride_ptr,
    w_stride_ptr,
    y_stride_ptr,
    x,
    w,
    y,
    xy_indptr,
    w_indices,
    d_in,
    d_out,
    w_column_major,
):
    pid = tl.program_id(0)

    m = tl.load(xy_indptr + pid + 1) - tl.load(xy_indptr + pid)
    k, n = d_in, d_out

    tl.store(all_problems_ptr + pid * 3, m)
    tl.store(all_problems_ptr + pid * 3 + 1, n)
    tl.store(all_problems_ptr + pid * 3 + 2, k)

    w_i = tl.load(w_indices + pid) if w_indices else tl.cast(pid, tl.int64)
    w_curr_ptr = w + w_i * k * n
    tl.store(w_ptr + pid, w_curr_ptr)

    x_curr_ptr = x + tl.load(xy_indptr + pid) * k
    tl.store(x_ptr + pid, x_curr_ptr)

    y_curr_ptr = y + tl.load(xy_indptr + pid) * n
    tl.store(y_ptr + pid, y_curr_ptr)

    tl.store(x_stride_ptr + pid, k)
    tl.store(w_stride_ptr + pid, k if w_column_major else n)
    tl.store(y_stride_ptr + pid, n)


@triton.jit
def compute_padding_mapping(
    m_indptr,
    padded_m_indptr,
    m_rank,
    padded_m_rank,
):
    pid = tl.program_id(0)
    m_start = tl.load(m_indptr + pid)
    m_end = tl.load(m_indptr + pid + 1)
    padded_m_start = tl.load(padded_m_indptr + pid)
    for i in range(m_end - m_start):
        tl.store(m_rank + m_start + i, m_start + i)
        tl.store(padded_m_rank + m_start + i, padded_m_start + i)
