"""Tests for parallel attention (Ulysses + Ring).

Launch with:
    torchrun --nproc_per_node=4 -m pytest tests/attention/test_parallel_attention.py -v
"""

import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from flashinfer.utils import (
    get_compute_capability,
    is_sm90a_supported,
    is_sm100a_supported,
)
from flashinfer.parallel_attention import (
    UnevenCPConfig,
    VarlenCPConfig,
    ParallelAttention,
    split_varlen_input,
    get_parallel_groups,
    uneven_cp_config,
    ulysses_varlen_config,
    ring_varlen_config,
)

# Skip all tests when not launched via torchrun / torch.distributed.launch
pytestmark = pytest.mark.skipif(
    "RANK" not in os.environ,
    reason="Must be launched with torchrun (RANK env var not set)",
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def dist_setup():
    """Initialize and tear down the distributed process group once per session."""
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    yield
    dist.destroy_process_group()


@pytest.fixture
def world_size():
    return dist.get_world_size()


@pytest.fixture
def rank():
    return dist.get_rank()


@pytest.fixture
def device(rank):
    return torch.device(f"cuda:{rank}")


@pytest.fixture(autouse=True)
def skip_if_unsupported(request):
    """Skip test if the attention backend requires unsupported hardware."""
    attn_type = request.node.callspec.params.get("attn_type", None)
    if attn_type == "flash-attn3" and not is_sm90a_supported(torch.device("cuda")):
        cc = get_compute_capability(torch.device("cuda"))
        pytest.skip(f"flash-attn3 requires SM90a+, got {cc}")

    if attn_type == "cutlass" and not is_sm100a_supported(torch.device("cuda")):
        cc = get_compute_capability(torch.device("cuda"))
        pytest.skip(f"cutlass requires SM100a+, got {cc}")


# ── Helpers ───────────────────────────────────────────────────────────────


def _sample_tensors(num_heads, seq_len, head_dim, world_size):
    """Create sample tensors for attention testing."""
    shape = (num_heads, seq_len, head_dim)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = q.chunk(world_size, dim=1)[rank]
    local_k = k.chunk(world_size, dim=1)[rank]
    local_v = v.chunk(world_size, dim=1)[rank]
    return q, k, v, local_q, local_k, local_v


def _sample_ring_varlen_tensors(num_heads, head_dim, world_size, seq_len_list):
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    total_seq_len = sum(seq_len_list)
    shape = (num_heads, total_seq_len, head_dim)

    q = torch.randn(shape, device=device, dtype=torch.bfloat16)
    k = torch.randn(shape, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, device=device, dtype=torch.bfloat16)

    dist.broadcast(q, src=0)
    dist.broadcast(k, src=0)
    dist.broadcast(v, src=0)

    local_q = split_varlen_input(q, seq_len_list, world_size, rank)
    local_k = split_varlen_input(k, seq_len_list, world_size, rank)
    local_v = split_varlen_input(v, seq_len_list, world_size, rank)

    return q, k, v, local_q, local_k, local_v


def _assert_cos_similarity(output, ref_output, threshold=0.99):
    cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    similarity = cos_sim(
        output.reshape(-1).to(torch.float32),
        ref_output.reshape(-1).to(torch.float32),
    )
    assert similarity >= threshold, f"Cosine similarity {similarity:.6f} < {threshold}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("num_heads", [24])
@pytest.mark.parametrize("seq_len", [6 * 8 * 1024])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("ulysses_size,ring_size", [(4, 1), (1, 4), (2, 2)])
@pytest.mark.parametrize("attn_type", ["flash-attn3", "cutlass"])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_attn_parallel(
    num_heads,
    seq_len,
    head_dim,
    ulysses_size,
    ring_size,
    attn_type,
    tensor_layout,
    world_size,
):
    query, key, value, local_query, local_key, local_value = _sample_tensors(
        num_heads, seq_len, head_dim, world_size
    )

    if tensor_layout == "NHD":
        local_query = local_query.permute(1, 0, 2).contiguous()
        local_key = local_key.permute(1, 0, 2).contiguous()
        local_value = local_value.permute(1, 0, 2).contiguous()

    ring_group, ulysses_group = get_parallel_groups(
        ulysses_size=ulysses_size, ring_size=ring_size
    )
    attn = ParallelAttention(
        attn_type=attn_type,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
    )

    local_output = attn.run(local_query, local_key, local_value, tensor_layout)

    if tensor_layout == "NHD":
        local_output = local_output.permute(1, 0, 2)

    ref_output = F.scaled_dot_product_attention(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), is_causal=False
    )
    local_ref_output = ref_output.chunk(world_size, dim=2)[dist.get_rank()]

    _assert_cos_similarity(local_output, local_ref_output)


@pytest.mark.parametrize("num_heads", [24])
@pytest.mark.parametrize("seq_len_padded", [6 * 8 * 1024])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("ulysses_size,ring_size", [(2, 2)])
@pytest.mark.parametrize("attn_type", ["flash-attn3", "cutlass"])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_uneven_attn_parallel(
    num_heads,
    seq_len_padded,
    head_dim,
    ulysses_size,
    ring_size,
    attn_type,
    tensor_layout,
    world_size,
    rank,
    device,
):
    # _sample_tensors returns HND layout
    query, key, value, local_query, local_key, local_value = _sample_tensors(
        num_heads, seq_len_padded, head_dim, world_size
    )

    uneven_number = world_size - 1
    seq_len_cur_rank = local_query.shape[1]
    if rank == world_size - 1:
        seq_len_cur_rank = seq_len_cur_rank - uneven_number

    if tensor_layout == "NHD":
        local_query = local_query.permute(1, 0, 2).contiguous()
        local_key = local_key.permute(1, 0, 2).contiguous()
        local_value = local_value.permute(1, 0, 2).contiguous()

    ring_group, ulysses_group = get_parallel_groups(
        ulysses_size=ulysses_size, ring_size=ring_size
    )
    seq_len_cur_ring_group = uneven_cp_config(
        seq_len=seq_len_padded - uneven_number,
        seq_len_padded=seq_len_padded,
        seq_len_cur_rank=seq_len_cur_rank,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
    )
    ucp_config = UnevenCPConfig(
        seq_len=seq_len_padded - uneven_number,
        seq_len_padded=seq_len_padded,
        seq_len_cur_ring_group=seq_len_cur_ring_group,
    )
    attn = ParallelAttention(
        attn_type=attn_type,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        uneven_cp_config=ucp_config,
    )

    local_output = attn.run(
        local_query, local_key, local_value, tensor_layout=tensor_layout
    )

    if tensor_layout == "NHD":
        local_output = local_output.permute(1, 0, 2)

    query = query[:, :-uneven_number, :]
    key = key[:, :-uneven_number, :]
    value = value[:, :-uneven_number, :]

    ref_output = F.scaled_dot_product_attention(
        query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), is_causal=False
    )
    local_ref_output = ref_output.chunk(world_size, dim=2)[rank]

    if rank == world_size - 1:
        local_output = local_output[:, :-uneven_number, :]

    _assert_cos_similarity(local_output, local_ref_output)


@pytest.mark.parametrize("num_heads", [24])
@pytest.mark.parametrize("seq_len_list", [[1 * 8 * 1024 - 1, 3 * 8 * 1024]])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("attn_type", ["flash-attn3", "cutlass"])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_ulysses_varlen_attn_parallel(
    num_heads, seq_len_list, head_dim, attn_type, tensor_layout, world_size, rank
):
    ulysses_size = world_size
    ring_size = 1

    total_seq_len = sum(seq_len_list)
    seq_len_padded = math.ceil(total_seq_len / world_size) * world_size
    uneven_number = seq_len_padded - total_seq_len

    # _sample_tensors returns HND layout
    query, key, value, local_query, local_key, local_value = _sample_tensors(
        num_heads, seq_len_padded, head_dim, world_size
    )

    if tensor_layout == "NHD":
        local_query = local_query.permute(1, 0, 2).contiguous()
        local_key = local_key.permute(1, 0, 2).contiguous()
        local_value = local_value.permute(1, 0, 2).contiguous()

    ring_group, ulysses_group = get_parallel_groups(
        ulysses_size=ulysses_size, ring_size=ring_size
    )
    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = ulysses_varlen_config(
        seq_len_list, seq_len_list
    )
    vcp_config = VarlenCPConfig(
        cu_seqlens_q_cur_ulysses_group=cu_seqlens_q,
        cu_seqlens_kv_cur_ulysses_group=cu_seqlens_kv,
        max_seq_len_q_cur_ulysses_group=max_seqlen_q,
        max_seq_len_kv_cur_ulysses_group=max_seqlen_kv,
    )
    attn = ParallelAttention(
        attn_type=attn_type,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        varlen_cp_config=vcp_config,
    )

    local_output = attn.run(
        local_query, local_key, local_value, tensor_layout=tensor_layout
    )

    if tensor_layout == "NHD":
        local_output = local_output.permute(1, 0, 2)

    cu_seqlens_q = cu_seqlens_q.cpu()
    cu_seqlens_kv = cu_seqlens_kv.cpu()
    local_ref_output_list = []
    for i in range(len(seq_len_list)):
        q_tmp = query[:, cu_seqlens_q[i] : cu_seqlens_q[i + 1], :]
        k_tmp = key[:, cu_seqlens_kv[i] : cu_seqlens_kv[i + 1], :]
        v_tmp = value[:, cu_seqlens_kv[i] : cu_seqlens_kv[i + 1], :]
        tmp_output = F.scaled_dot_product_attention(
            q_tmp.unsqueeze(0), k_tmp.unsqueeze(0), v_tmp.unsqueeze(0), is_causal=False
        )
        local_ref_output_list.append(tmp_output)

    ref_output = torch.cat(local_ref_output_list, dim=2)
    local_ref_output = ref_output.chunk(world_size, dim=2)[rank]

    if rank == world_size - 1 and seq_len_padded > total_seq_len:
        local_output = local_output[:, :-uneven_number, :]

    _assert_cos_similarity(local_output, local_ref_output)


@pytest.mark.parametrize("num_heads", [24])
@pytest.mark.parametrize(
    "seq_len_list",
    [torch.tensor([1021, 1024, 1027, 750, 826], dtype=torch.int32)],
)
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("attn_type", ["flash-attn3", "cutlass"])
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_ring_varlen_attn_parallel(
    num_heads, seq_len_list, head_dim, attn_type, tensor_layout, world_size, rank
):
    ring_size = world_size

    full_cu_seqlens = [0]
    for seq_len in seq_len_list:
        full_cu_seqlens.append(full_cu_seqlens[-1] + seq_len)

    # _sample_ring_varlen_tensors returns HND layout
    query, key, value, local_query, local_key, local_value = (
        _sample_ring_varlen_tensors(num_heads, head_dim, world_size, seq_len_list)
    )

    if tensor_layout == "NHD":
        local_query = local_query.permute(1, 0, 2).contiguous()
        local_key = local_key.permute(1, 0, 2).contiguous()
        local_value = local_value.permute(1, 0, 2).contiguous()

    ring_group, ulysses_group = get_parallel_groups(ulysses_size=1, ring_size=ring_size)
    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = ring_varlen_config(
        seq_len_list, seq_len_list, ring_group
    )
    vcp_config = VarlenCPConfig(
        cu_seqlens_q_cur_ring_group=cu_seqlens_q,
        cu_seqlens_kv_cur_ring_group=cu_seqlens_kv,
        max_seq_len_q_cur_ring_group=max_seqlen_q,
        max_seq_len_kv_cur_ring_group=max_seqlen_kv,
    )
    attn = ParallelAttention(
        attn_type=attn_type,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        varlen_cp_config=vcp_config,
    )
    local_output = attn.run(
        local_query, local_key, local_value, tensor_layout=tensor_layout
    )

    if tensor_layout == "NHD":
        local_output = local_output.permute(1, 0, 2)

    local_ref_output_list = []
    for i in range(len(seq_len_list)):
        q_tmp = query[:, full_cu_seqlens[i] : full_cu_seqlens[i + 1], :]
        k_tmp = key[:, full_cu_seqlens[i] : full_cu_seqlens[i + 1], :]
        v_tmp = value[:, full_cu_seqlens[i] : full_cu_seqlens[i + 1], :]
        tmp_output = F.scaled_dot_product_attention(
            q_tmp.unsqueeze(0), k_tmp.unsqueeze(0), v_tmp.unsqueeze(0), is_causal=False
        )
        local_ref_output_list.append(tmp_output)

    ref_output = torch.cat(local_ref_output_list, dim=2).squeeze(0)
    local_ref_output = split_varlen_input(ref_output, seq_len_list, world_size, rank)

    _assert_cos_similarity(local_output, local_ref_output)
