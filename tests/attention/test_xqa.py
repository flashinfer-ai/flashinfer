import math

import numpy as np
import pytest
import torch

from flashinfer import xqa
from flashinfer.utils import get_compute_capability


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def round_up(a, b):
    return math.ceil(a / b) * b


def div_up(a, b):
    return math.ceil(a / b)


props = torch.cuda.get_device_properties(0)
sm_count = props.multi_processor_count

beam_width = 1
q_scale = 1.0


class CacheSeq:
    def __init__(
        self,
        pool: torch.Tensor,
        page_indices: torch.Tensor,
        nb_heads: int,
        idx_head: int,
        tokens_per_page: int = 32,
    ):
        self.pool = pool
        self.page_indices = page_indices
        self.nb_heads = nb_heads
        self.idx_head = idx_head
        self.tokens_per_page = tokens_per_page

    def __getitem__(self, i: int) -> torch.Tensor:
        page_idx = self.page_indices[i // self.tokens_per_page].to(torch.int32)
        idx_head = (
            self.tokens_per_page * self.nb_heads * page_idx
            + self.tokens_per_page * self.idx_head
            + i % self.tokens_per_page
        )
        return self.pool[idx_head]


def ref_attention(
    q,
    k_cache_seq,
    v_cache_seq,
    seq_len,
    q_scale,
    kv_scale,
    x_scale,
    attention_sinks,
    sliding_win_size,
    valid_elems_per_head,
):
    head_grp_size = q.shape[0]
    rcp_x_scale = 1.0 / x_scale
    qk_scale = q_scale * kv_scale / math.sqrt(valid_elems_per_head)

    q_f32 = q.to(torch.float32)  # [head_grp_size, valid_elems_per_head]

    k_cache_f32 = torch.zeros(
        seq_len, valid_elems_per_head, dtype=torch.float32, device="cuda"
    )
    v_cache_f32 = torch.zeros(
        seq_len, valid_elems_per_head, dtype=torch.float32, device="cuda"
    )

    for j in range(seq_len):
        k_cache_f32[j] = k_cache_seq[j].to(torch.float32)
        v_cache_f32[j] = v_cache_seq[j].to(torch.float32)

    # q_f32: [head_grp_size, valid_elems_per_head]
    # k_cache_f32: [seq_len, valid_elems_per_head]
    # gemm0_acc: [head_grp_size, seq_len]
    gemm0_acc = torch.zeros(
        head_grp_size, seq_len, dtype=torch.float32, device=q_f32.device
    )

    # Calculate sliding window start position
    if sliding_win_size == 0 or seq_len < sliding_win_size:
        seq_beg = 0
    else:
        seq_beg = seq_len - sliding_win_size

    # Set positions before sliding window to negative infinity (masking)
    if seq_beg > 0:
        gemm0_acc[:, :seq_beg] = float("-inf")

    # q_f32: [head_grp_size, valid_elems_per_head]
    # k_cache_f32[seq_beg:seq_len]: [valid_seq_len, valid_elems_per_head]
    if seq_beg < seq_len:
        valid_k_cache = k_cache_f32[
            seq_beg:seq_len
        ]  # [valid_seq_len, valid_elems_per_head]
        valid_scores = (
            torch.matmul(q_f32, valid_k_cache.t()) * qk_scale
        )  # [head_grp_size, valid_seq_len]
        gemm0_acc[:, seq_beg:seq_len] = valid_scores

    row_max = torch.max(gemm0_acc, dim=1, keepdim=True)[0]  # [head_grp_size, 1]
    x = torch.exp(gemm0_acc - row_max)  # [head_grp_size, seq_len]

    row_sum = torch.sum(x, dim=1, keepdim=True)  # [head_grp_size, 1]

    x = x * rcp_x_scale

    if seq_beg < seq_len:
        valid_x = x[:, seq_beg:seq_len]  # [head_grp_size, valid_seq_len]
        valid_v_cache = v_cache_f32[
            seq_beg:seq_len
        ]  # [valid_seq_len, valid_elems_per_head]
        out = torch.matmul(
            valid_x, valid_v_cache
        )  # [head_grp_size, valid_elems_per_head]
    else:
        out = torch.zeros(
            head_grp_size,
            valid_elems_per_head,
            dtype=torch.float32,
            device=q_f32.device,
        )

    if attention_sinks is not None:
        sink_weights = torch.exp(
            attention_sinks - row_max.squeeze(-1)
        )  # [head_grp_size]
        row_sum.squeeze(-1)[:] += sink_weights

    out = out * (x_scale * kv_scale) / row_sum

    return out


@pytest.mark.skipif(
    get_compute_capability(torch.device(device="cuda"))[0] != 9,
    reason="XQA is only supported on SM90 GPUs",
)
@pytest.mark.parametrize("use_sliding_window", [True, False])
@pytest.mark.parametrize("use_fp16", [True, False])
@pytest.mark.parametrize("use_attention_sinks", [True, False])
@pytest.mark.parametrize("seq_len", [2, 15, 256, 514])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("nb_k_heads", [1, 4, 8])
@pytest.mark.parametrize("tokens_per_page", [16, 64])
@pytest.mark.parametrize("valid_elems_per_head", [32, 128])
@pytest.mark.parametrize("head_grp_size", [8, 16])
def test_xqa(
    batch_size,
    nb_k_heads,
    seq_len,
    tokens_per_page,
    use_fp16,
    valid_elems_per_head,
    head_grp_size,
    use_attention_sinks,
    use_sliding_window,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 9:
        pytest.skip("XQA only supports on Hopper at this moment")
    set_random_seed(42)

    nb_v_heads = nb_k_heads
    nb_q_heads = nb_k_heads * head_grp_size

    output = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head,
        dtype=torch.bfloat16 if not use_fp16 else torch.float16,
        device="cuda",
    )
    output.fill_(float("nan"))
    q_heads = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head,
        dtype=torch.bfloat16 if not use_fp16 else torch.float16,
        device="cuda",
    )
    q_heads.normal_(0, 1)
    if use_attention_sinks:
        attention_sinks = torch.zeros(
            nb_k_heads, head_grp_size, dtype=torch.float32, device="cuda"
        )
        for i in range(nb_k_heads):
            for j in range(head_grp_size):
                attention_sinks[i, j] = 2.0 + float(j % 4)
    else:
        attention_sinks = None
    if use_sliding_window:
        sliding_win_size = 256
    else:
        sliding_win_size = 0

    max_seq_len = round_up(seq_len, tokens_per_page)
    total_nb_cache_heads = (
        (nb_k_heads + nb_v_heads) * max_seq_len * beam_width * batch_size
    )
    cache_heads = torch.zeros(
        total_nb_cache_heads,
        valid_elems_per_head,
        dtype=torch.bfloat16 if not use_fp16 else torch.float16,
        device="cuda",
    )
    cache_heads.normal_(0, 1)

    nb_pages_per_seq = div_up(max_seq_len, tokens_per_page)
    total_nb_pages = nb_pages_per_seq * 2 * beam_width * batch_size
    page_list_arg = torch.zeros(
        batch_size, beam_width, 2, nb_pages_per_seq, dtype=torch.uint32, device="cuda"
    )
    page_list_arg.view(-1)[:total_nb_pages] = torch.arange(
        total_nb_pages, dtype=torch.int32, device="cuda"
    ).to(torch.uint32)
    flattened = page_list_arg.flatten()
    indices = torch.randperm(flattened.numel())
    shuffled_flat = flattened.to(torch.int32)[indices].to(torch.uint32)
    page_list_arg = shuffled_flat.view(page_list_arg.shape)

    def cache_head_at(
        batch,
        is_k,
        idx_kv_head,
        pos,
        cache_heads,
        page_list,
        beam_width,
        nb_k_heads,
        tokens_per_page,
    ):
        beam = 0
        kv = 0 if is_k else 1

        page_idx = page_list_arg[batch][beam][kv][pos // tokens_per_page].to(
            torch.int32
        )

        idx_head = (
            tokens_per_page * (nb_k_heads * page_idx + idx_kv_head)
            + pos % tokens_per_page
        )

        return cache_heads[idx_head]

    for batch in range(batch_size):
        for kv in range(2):
            for idx_kv_head in range(nb_k_heads):
                for pos in range(seq_len, max_seq_len):
                    cache_head = cache_head_at(
                        batch,
                        kv == 0,
                        idx_kv_head,
                        pos,
                        cache_heads,
                        page_list_arg,
                        beam_width,
                        nb_k_heads,
                        tokens_per_page,
                    )
                    cache_head.fill_(0.0)

    seq_len_list = torch.zeros(
        batch_size, beam_width, dtype=torch.uint32, device="cuda"
    )
    seq_len_list.fill_(seq_len)

    kv_cache_scale = torch.ones(1, dtype=torch.float32, device="cuda")

    nb_seq = nb_k_heads * batch_size
    nb_semaphores = round_up(nb_seq, 2) + 2 + nb_seq + 2

    semaphores = torch.zeros(nb_semaphores, dtype=torch.uint32, device="cuda")

    scratch_size = 256 << 20
    scratch_buf = torch.zeros(scratch_size, dtype=torch.uint8, device="cuda")

    xqa(
        use_fp16,
        tokens_per_page,
        valid_elems_per_head,
        head_grp_size,
        use_sliding_window,
        sliding_win_size,
        sm_count,
        nb_k_heads,
        q_scale,
        output,
        q_heads,
        attention_sinks,
        cache_heads,
        page_list_arg,
        max_seq_len,
        seq_len_list,
        batch_size,
        kv_cache_scale,
        semaphores,
        scratch_buf,
    )

    for req in range(batch_size):
        for b in range(beam_width):
            for idx_k_head in range(nb_k_heads):
                k_cache_seq = CacheSeq(
                    pool=cache_heads,
                    page_indices=page_list_arg[req][b][0],
                    nb_heads=nb_k_heads,
                    idx_head=idx_k_head,
                    tokens_per_page=tokens_per_page,
                )
                v_cache_seq = CacheSeq(
                    pool=cache_heads,
                    page_indices=page_list_arg[req][b][1],
                    nb_heads=nb_k_heads,
                    idx_head=idx_k_head,
                    tokens_per_page=tokens_per_page,
                )

        ref_output = ref_attention(
            q=q_heads[req][b][
                idx_k_head * head_grp_size : (idx_k_head + 1) * head_grp_size
            ],
            k_cache_seq=k_cache_seq,
            v_cache_seq=v_cache_seq,
            seq_len=seq_len,
            q_scale=q_scale,
            kv_scale=kv_cache_scale[0],
            x_scale=1.0,
            attention_sinks=attention_sinks[idx_k_head, :]
            if use_attention_sinks
            else None,
            sliding_win_size=sliding_win_size if use_sliding_window else 0,
            valid_elems_per_head=valid_elems_per_head,
        )
        kernel_output = output[req][b][
            idx_k_head * head_grp_size : (idx_k_head + 1) * head_grp_size
        ].to(torch.float32)
        assert torch.allclose(ref_output, kernel_output, atol=0.01, rtol=0.01)
