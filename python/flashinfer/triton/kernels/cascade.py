import triton
import triton.language as tl


@triton.jit
def state_merge(o, m, d, other_o, other_m, other_d):
    m_max = tl.maximum(m, other_m)
    d = d * tl.exp2(m - m_max) + other_d * tl.exp2(other_m - m_max)
    o = o * tl.exp2(m - m_max) + other_o * tl.exp2(other_m - m_max)
    return o, m_max, d


@triton.jit
def state_normalize(o, m, d):
    o = o / d
    return o, m, d


@triton.jit
def state_get_lse(o, m, d):
    return m + tl.log2(d)


@triton.jit
def merge_state_kernel(
    v_a_ptr,
    s_a_ptr,
    v_b_ptr,
    s_b_ptr,
    v_merged_ptr,
    s_merged_ptr,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            s_a_val = tl.load(s_a_ptr + pos * num_heads + head_idx)
            s_b_val = tl.load(s_b_ptr + pos * num_heads + head_idx)

            offsets = (pos * num_heads + head_idx) * head_dim + tx
            v_a = tl.load(v_a_ptr + offsets)
            v_b = tl.load(v_b_ptr + offsets)

            v_merged, s_max, d = state_merge(
                o=v_a, m=s_a_val, d=1, other_o=v_b, other_m=s_b_val, other_d=1
            )
            v_merged, s_max, d = state_normalize(v_merged, s_max, d)
            v_merged_offset = (pos * num_heads + head_idx) * head_dim + tx
            tl.store(v_merged_ptr + v_merged_offset, v_merged)

            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx,
                    tl.log2(d) + s_max,
                )


@triton.jit
def merge_state_in_place_kernel(
    v_ptr,
    s_ptr,
    v_other_ptr,
    s_other_ptr,
    num_heads,
    head_dim,
    mask_ptr,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    if mask_ptr:
        if tl.load(mask_ptr + pos) == 0:
            return

    for head_idx in tl.range(bdy):
        s_val = tl.load(s_ptr + pos * num_heads + head_idx)
        s_other_val = tl.load(s_other_ptr + pos * num_heads + head_idx)
        s_max = tl.maximum(s_val, s_other_val)
        s_val = tl.exp2(s_val - s_max)
        s_other_val = tl.exp2(s_other_val - s_max)
        scale = s_val / (s_val + s_other_val)
        other_scale = s_other_val / (s_val + s_other_val)
        for tx in tl.range(bdx):
            offset = (pos * num_heads + head_idx) * head_dim + tx
            v_vec = tl.load(v_ptr + offset)
            v_other_vec = tl.load(v_other_ptr + offset)
            v_vec = scale * v_vec + other_scale * v_other_vec
            tl.store(v_ptr + offset, v_vec)
        if s_ptr:
            tl.store(
                s_ptr + pos * num_heads + head_idx,
                tl.log2(s_val + s_other_val) + s_max,
            )


@triton.jit
def merge_states_kernel(
    v_ptr,
    s_ptr,
    v_merged_ptr,
    s_merged_ptr,
    num_index_sets,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)

    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            o, m, d = 0.0, -5e4, 1.0
            for iter in tl.range(num_index_sets):
                s = tl.load(
                    s_ptr + (pos * num_index_sets + iter) * num_heads + head_idx
                )
                v = tl.load(
                    v_ptr
                    + ((pos * num_index_sets + iter) * num_heads + head_idx) * head_dim
                    + tx
                )
                o, m, d = state_merge(o, m, d, v, s, 1)
            o, m, d = state_normalize(o, m, d)
            tl.store(v_merged_ptr + (pos * num_heads + head_idx) * head_dim + tx, o)
            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx, state_get_lse(o, m, d)
                )


@triton.jit
def variable_length_merge_states_kernel(
    v_ptr,
    s_ptr,
    indptr,
    v_merged_ptr,
    s_merged_ptr,
    num_heads,
    head_dim,
    bdx: tl.constexpr,
    bdy: tl.constexpr,
):
    pos = tl.program_id(axis=0)
    for tx in tl.range(bdx):
        for head_idx in tl.range(bdy):
            o, m, d = 0.0, -5e4, 1.0
            for iter in tl.range(tl.load(indptr + pos), tl.load(indptr + pos + 1)):
                s = tl.load(s_ptr + iter * num_heads + head_idx)
                v = tl.load(v_ptr + (iter * num_heads + head_idx) * head_dim + tx)
                o, m, d = state_merge(o, m, d, v, s, 1)
            o, m, d = state_normalize(o, m, d)
            tl.store(v_merged_ptr + (pos * num_heads + head_idx) * head_dim + tx, o)
            if s_merged_ptr:
                tl.store(
                    s_merged_ptr + pos * num_heads + head_idx, state_get_lse(o, m, d)
                )
