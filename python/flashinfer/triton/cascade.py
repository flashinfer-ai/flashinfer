from typing import Optional

import torch

from .kernels.cascade import (
    merge_state_in_place_kernel,
    merge_state_kernel,
    merge_states_kernel,
    variable_length_merge_states_kernel,
)
from .utils import check_device, check_dim, check_input, check_shape


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
):
    check_input(v_a)
    check_input(s_a)
    check_input(v_b)
    check_input(s_b)
    check_device([v_a, s_a, v_b, s_b])
    check_dim(3, v_a)
    check_dim(2, s_a)
    check_dim(3, v_b)
    check_dim(2, s_b)
    check_shape(v_a, v_b)
    check_shape(s_a, s_b)
    assert v_a.size(0) == s_a.size(0)
    assert v_a.size(1) == s_b.size(1)
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    seq_len = v_a.size(0)
    num_heads = v_a.size(1)
    head_dim = v_a.size(2)
    v_merged = torch.empty_like(v_a).to(s_a.device)
    s_merged = torch.empty((seq_len, num_heads)).to(s_a.device)
    bdx = head_dim
    bdy = num_heads

    merge_state_kernel[lambda meta: (seq_len,)](
        v_a, s_a, v_b, s_b, v_merged, s_merged, num_heads, head_dim, bdx=bdx, bdy=bdy
    )

    return v_merged, s_merged


def merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    check_input(v)
    check_input(s)
    check_input(v_other)
    check_input(s_other)
    check_device([v, s, v_other, s_other])
    check_dim(3, v)
    check_dim(2, s)
    check_dim(3, v_other)
    check_dim(2, s_other)
    check_shape(v, v_other)
    check_shape(s, s_other)
    assert v.size(0) == s.size(0)
    assert v.size(1) == s.size(1)
    assert s.dtype == torch.float32
    assert s_other.dtype == torch.float32
    if mask is not None:
        check_dim(1, mask)
        assert v.size(0) == mask.size(0)
        assert mask.device == device
    seq_len = v.size(0)
    num_heads = v.size(1)
    head_dim = v.size(2)

    bdx = head_dim
    bdy = num_heads
    merge_state_in_place_kernel[(seq_len,)](
        v, s, v_other, s_other, num_heads, head_dim, mask, bdx=bdx, bdy=bdy
    )


def merge_states(v: torch.Tensor, s: torch.Tensor):
    check_input(v)
    check_input(s)
    check_device([v, s])
    check_dim(4, v)
    check_dim(3, s)
    assert v.size(0) == s.size(0)
    assert v.size(1) == s.size(1)
    assert v.size(2) == s.size(2)
    seq_len = v.size(0)
    num_index_sets = v.size(1)
    num_heads = v.size(2)
    head_dim = v.size(3)
    s = s.to(torch.float32)
    v_merged = torch.empty(
        (seq_len, num_heads, head_dim), dtype=v.dtype, device=v.device
    )
    s_merged = torch.empty((seq_len, num_heads), dtype=s.dtype, device=s.device)

    bdx = head_dim
    bdy = num_heads
    merge_states_kernel[(seq_len,)](
        v,
        s,
        v_merged,
        s_merged,
        num_index_sets,
        num_heads,
        head_dim,
        bdx=bdx,
        bdy=bdy,
    )
    return v_merged, s_merged


def variable_length_merge_states(
    v: torch.Tensor, s: torch.Tensor, indptr: torch.Tensor
):
    check_input(v)
    check_input(s)
    check_device([v, s])
    check_dim(3, v)
    check_dim(2, s)
    assert v.size(0) == s.size(0)
    assert v.size(1) == s.size(1)
    seq_len = indptr.size(0) - 1
    num_heads = v.size(1)
    head_dim = v.size(2)
    s = s.to(torch.float32)
    indptr = indptr.to(torch.int32)
    v_merged = torch.empty(
        (seq_len, num_heads, head_dim), dtype=v.dtype, device=v.device
    )
    s_merged = torch.empty((seq_len, num_heads), dtype=s.dtype, device=s.device)

    bdx = head_dim
    bdy = num_heads
    variable_length_merge_states_kernel[(seq_len,)](
        v,
        s,
        indptr,
        v_merged,
        s_merged,
        num_heads,
        head_dim,
        bdx=bdx,
        bdy=bdy,
    )
    return v_merged, s_merged
