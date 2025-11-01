import math

import numpy as np
import pytest
import torch

from flashinfer import xqa, xqa_mla
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
        kv_layout: str = "NHD",
    ):
        self.pool = pool
        self.page_indices = page_indices
        self.nb_heads = nb_heads
        self.idx_head = idx_head
        self.tokens_per_page = tokens_per_page
        self.kv_layout = kv_layout

    def __getitem__(self, i: int) -> torch.Tensor:
        page_idx = self.page_indices[i // self.tokens_per_page].to(torch.int32)
        token_in_page = i % self.tokens_per_page
        if self.kv_layout == "NHD":
            # NHD layout: [page_idx, token_in_page, idx_head, :]
            return self.pool[page_idx, token_in_page, self.idx_head, :]
        else:  # HND
            # HND layout: [page_idx, idx_head, token_in_page, :]
            return self.pool[page_idx, self.idx_head, token_in_page, :]


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
    valid_elems_per_v_head=None,  # Optional: for MLA where V dim != K dim
):
    """
    For MLA:
    - Q/K dimension: 576 (valid_elems_per_head)
    - V dimension: 512 (valid_elems_per_v_head)
    - Output dimension: matches valid_elems_per_head (576) but only first
      valid_elems_per_v_head (512) elements are valid
    """
    head_grp_size = q.shape[0]
    rcp_x_scale = 1.0 / x_scale
    qk_scale = q_scale * kv_scale / math.sqrt(valid_elems_per_head)

    # For MLA: V dimension may differ from K dimension
    if valid_elems_per_v_head is None:
        valid_elems_per_v_head = valid_elems_per_head

    q_f32 = q.to(torch.float32)  # [head_grp_size, valid_elems_per_head]

    k_cache_f32 = torch.zeros(
        seq_len, valid_elems_per_head, dtype=torch.float32, device="cuda"
    )
    # V cache: load only valid_elems_per_v_head dimensions
    v_cache_f32 = torch.zeros(
        seq_len, valid_elems_per_v_head, dtype=torch.float32, device="cuda"
    )

    for j in range(seq_len):
        k_cache_f32[j] = k_cache_seq[j].to(torch.float32)
        # For MLA: V cache storage is 576 but only first 512 elements are valid
        v_cache_f32[j] = v_cache_seq[j][:valid_elems_per_v_head].to(torch.float32)

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
        ]  # [valid_seq_len, valid_elems_per_v_head]
        out = torch.matmul(
            valid_x, valid_v_cache
        )  # [head_grp_size, valid_elems_per_v_head]
    else:
        out = torch.zeros(
            head_grp_size,
            valid_elems_per_v_head,
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
    get_compute_capability(torch.device(device="cuda"))[0] not in [9, 10, 12],
    reason="XQA is only supported on SM90, SM100, SM120 GPUs",
)
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("use_sliding_window", [True, False])
@pytest.mark.parametrize("input_type", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_kv_cache", [True, False])
@pytest.mark.parametrize("use_attention_sinks", [True, False])
@pytest.mark.parametrize("seq_len", [2, 15, 256, 514])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("nb_k_heads", [2, 4])
@pytest.mark.parametrize("tokens_per_page", [16, 64])
@pytest.mark.parametrize("valid_elems_per_head", [32, 128])
@pytest.mark.parametrize("head_grp_size", [8, 16])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_xqa(
    batch_size,
    nb_k_heads,
    seq_len,
    tokens_per_page,
    input_type,
    fp8_kv_cache,
    valid_elems_per_head,
    head_grp_size,
    use_attention_sinks,
    use_sliding_window,
    enable_pdl,
    kv_layout,
):
    set_random_seed(42)

    nb_q_heads = nb_k_heads * head_grp_size

    output = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head,
        dtype=input_type,
        device="cuda",
    )
    output.fill_(float("nan"))
    q_heads = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head,
        dtype=input_type,
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
    nb_pages_per_seq = div_up(max_seq_len, tokens_per_page)
    # Total number of pages needed for all sequences
    total_num_pages = nb_pages_per_seq * batch_size

    # Create cache with specified layout
    if kv_layout == "NHD":
        # NHD layout: [num_pages, page_size, num_kv_heads, head_dim]
        cache_k_heads = torch.zeros(
            total_num_pages,
            tokens_per_page,
            nb_k_heads,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )
        cache_v_heads = torch.zeros(
            total_num_pages,
            tokens_per_page,
            nb_k_heads,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )
    else:  # HND layout
        # HND layout: [num_pages, num_kv_heads, page_size, head_dim]
        cache_k_heads = torch.zeros(
            total_num_pages,
            nb_k_heads,
            tokens_per_page,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )
        cache_v_heads = torch.zeros(
            total_num_pages,
            nb_k_heads,
            tokens_per_page,
            valid_elems_per_head,
            dtype=input_type,
            device="cuda",
        )

    cache_k_heads.normal_(0, 1)
    cache_v_heads.normal_(0, 1)

    if fp8_kv_cache:
        # Scale down the cache heads to keep values within the representable range of FP8
        # and prevent overflow during computation. The factor 4.0 is chosen empirically.
        cache_k_heads /= 4.0
        cache_v_heads /= 4.0
    page_list_arg = torch.zeros(
        batch_size, nb_pages_per_seq, dtype=torch.int32, device="cuda"
    )

    # Initialize page list sequentially
    page_idx = 0
    for batch in range(batch_size):
        for page in range(nb_pages_per_seq):
            page_list_arg[batch, page] = page_idx
            page_idx += 1

    flattened = page_list_arg.flatten()
    indices = torch.randperm(flattened.numel())
    shuffled_flat = flattened[indices]
    page_list_arg = shuffled_flat.view(page_list_arg.shape)

    def cache_head_at(
        batch,
        is_k,
        idx_kv_head,
        pos,
        cache_k_heads,
        cache_v_heads,
        page_list,
        beam_width,
        nb_k_heads,
        tokens_per_page,
        kv_layout,
    ):
        # K and V share page indices
        page_idx = page_list[batch][pos // tokens_per_page].to(torch.int32)
        token_in_page = pos % tokens_per_page

        cache = cache_k_heads if is_k else cache_v_heads
        if kv_layout == "NHD":
            # NHD layout: [page_idx, token_in_page, idx_kv_head, :]
            return cache[page_idx, token_in_page, idx_kv_head, :]
        else:  # HND
            # HND layout: [page_idx, idx_kv_head, token_in_page, :]
            return cache[page_idx, idx_kv_head, token_in_page, :]

    for batch in range(batch_size):
        for kv in range(2):
            for idx_kv_head in range(nb_k_heads):
                for pos in range(seq_len, max_seq_len):
                    cache_head = cache_head_at(
                        batch,
                        kv == 0,
                        idx_kv_head,
                        pos,
                        cache_k_heads,
                        cache_v_heads,
                        page_list_arg,
                        beam_width,
                        nb_k_heads,
                        tokens_per_page,
                        kv_layout,
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
        q_heads,
        cache_k_heads.to(torch.float8_e4m3fn) if fp8_kv_cache else cache_k_heads,
        cache_v_heads.to(torch.float8_e4m3fn) if fp8_kv_cache else cache_v_heads,
        page_list_arg,
        seq_len_list,
        output,
        scratch_buf,
        semaphores,
        nb_k_heads,
        tokens_per_page,
        sinks=attention_sinks,
        q_scale=q_scale,
        kv_scale=kv_cache_scale,
        sliding_win_size=sliding_win_size,
        kv_layout=kv_layout,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
    )

    for req in range(batch_size):
        for b in range(beam_width):
            for idx_k_head in range(nb_k_heads):
                # K and V use separate pools but share page indices
                k_cache_seq = CacheSeq(
                    pool=cache_k_heads,
                    page_indices=page_list_arg[req],
                    nb_heads=nb_k_heads,
                    idx_head=idx_k_head,
                    tokens_per_page=tokens_per_page,
                    kv_layout=kv_layout,
                )
                v_cache_seq = CacheSeq(
                    pool=cache_v_heads,
                    page_indices=page_list_arg[req],
                    nb_heads=nb_k_heads,
                    idx_head=idx_k_head,
                    tokens_per_page=tokens_per_page,
                    kv_layout=kv_layout,
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
                if fp8_kv_cache:
                    atol = 0.05
                    rtol = 0.05
                else:
                    atol = 0.01
                    rtol = 0.01

                diff_abs = torch.abs(ref_output - kernel_output)
                diff_rel = diff_abs / (torch.abs(ref_output) + 1e-8)

                within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

                pass_ratio = within_tolerance.float().mean().item()

                required_ratio = 0.99
                assert pass_ratio >= required_ratio, (
                    f"req={req}, b={b}, idx_k_head={idx_k_head}: "
                    f"Total {ref_output.numel()} elements, only {pass_ratio:.1%} meet tolerance criteria, "
                    f"require at least {required_ratio:.1%}"
                )


@pytest.mark.skipif(
    get_compute_capability(torch.device(device="cuda"))[0] not in [12],
    reason="XQA mla is only supported on SM120 GPUs",
)
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("seq_len", [2, 15, 256, 514, 2048])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("tokens_per_page", [32, 64])
def test_xqa_mla(
    batch_size,
    seq_len,
    tokens_per_page,
    enable_pdl,
):
    set_random_seed(42)

    # MLA specific constants (fixed, not parameterized)
    nb_k_heads = 1  # MLA only supports 1 K head
    head_grp_size = 128  # Fixed for MLA
    valid_elems_per_head_qk = 576  # Q and K dimension
    valid_elems_per_head_v = 512  # V dimension (output dimension)

    nb_q_heads = nb_k_heads * head_grp_size

    output = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head_v,  # Output dimension is 512 (V dimension)
        dtype=torch.bfloat16,
        device="cuda",
    )
    output.fill_(float("nan"))
    q_heads = torch.zeros(
        batch_size,
        beam_width,
        nb_q_heads,
        valid_elems_per_head_qk,  # Q dimension is 576
        dtype=torch.float32,
        device="cuda",
    )
    q_heads.normal_(0, 1)

    max_seq_len = round_up(seq_len, tokens_per_page)
    nb_pages_per_seq = div_up(max_seq_len, tokens_per_page)
    # Total number of pages needed for all sequences
    total_num_pages = nb_pages_per_seq * batch_size

    # NHD layout: [num_pages, page_size, num_kv_heads, head_dim]
    cache_k_heads = torch.zeros(
        total_num_pages,
        tokens_per_page,
        nb_k_heads,
        valid_elems_per_head_qk,  # K dimension is 576
        dtype=torch.float32,
        device="cuda",
    )
    cache_k_heads.normal_(0, 1)

    cache_v_heads = torch.zeros(
        total_num_pages,
        tokens_per_page,
        nb_k_heads,
        valid_elems_per_head_qk,  # V storage is 576 (but only 512 used)
        dtype=torch.float32,
        device="cuda",
    )
    cache_v_heads.normal_(0, 1)

    cache_k_heads /= 4.0
    cache_v_heads /= 4.0

    page_list_arg = torch.zeros(
        batch_size, nb_pages_per_seq, dtype=torch.int32, device="cuda"
    )

    # Initialize page list sequentially
    page_idx = 0
    for batch in range(batch_size):
        for page in range(nb_pages_per_seq):
            page_list_arg[batch, page] = page_idx
            page_idx += 1

    flattened = page_list_arg.flatten()
    indices = torch.randperm(flattened.numel())
    shuffled_flat = flattened[indices]
    page_list_arg = shuffled_flat.view(page_list_arg.shape)

    def cache_head_at(
        batch,
        is_k,
        idx_kv_head,
        pos,
        cache_k_heads,
        cache_v_heads,
        page_list,
        beam_width,
        nb_k_heads,
        tokens_per_page,
    ):
        # K and V share page indices
        page_idx = page_list[batch][pos // tokens_per_page].to(torch.int32)
        token_in_page = pos % tokens_per_page

        # NHD layout: [page_idx, token_in_page, idx_kv_head, :]
        cache = cache_k_heads if is_k else cache_v_heads
        return cache[page_idx, token_in_page, idx_kv_head, :]

    for batch in range(batch_size):
        for kv in range(2):
            for idx_kv_head in range(nb_k_heads):
                for pos in range(seq_len, max_seq_len):
                    cache_head = cache_head_at(
                        batch,
                        kv == 0,
                        idx_kv_head,
                        pos,
                        cache_k_heads,
                        cache_v_heads,
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

    xqa_mla(
        q_heads.to(torch.float8_e4m3fn),
        cache_k_heads.to(torch.float8_e4m3fn),
        cache_v_heads.to(torch.float8_e4m3fn),
        page_list_arg,
        seq_len_list,
        output,
        scratch_buf,
        semaphores,
        tokens_per_page,
        q_scale=q_scale,
        kv_scale=kv_cache_scale,
        sm_count=sm_count,
        enable_pdl=enable_pdl,
    )

    for req in range(batch_size):
        for b in range(beam_width):
            for idx_k_head in range(nb_k_heads):
                # K and V use separate pools but share page indices
                k_cache_seq = CacheSeq(
                    pool=cache_k_heads,
                    page_indices=page_list_arg[req],
                    nb_heads=nb_k_heads,
                    idx_head=idx_k_head,
                    tokens_per_page=tokens_per_page,
                )
                v_cache_seq = CacheSeq(
                    pool=cache_v_heads,
                    page_indices=page_list_arg[req],
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
                    q_scale=q_scale * math.sqrt(576),
                    kv_scale=kv_cache_scale[0],
                    x_scale=1.0,
                    attention_sinks=None,
                    sliding_win_size=0,
                    valid_elems_per_head=valid_elems_per_head_qk,  # Q/K dimension (576)
                    valid_elems_per_v_head=valid_elems_per_head_v,  # V dimension (512)
                ).to(torch.float32)
                kernel_output = output[req][b][
                    idx_k_head * head_grp_size : (idx_k_head + 1) * head_grp_size
                ].to(torch.float32)
                atol = 0.05
                rtol = 0.05

                diff_abs = torch.abs(ref_output - kernel_output)
                diff_rel = diff_abs / (torch.abs(ref_output) + 1e-8)

                within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

                pass_ratio = within_tolerance.float().mean().item()

                required_ratio = 0.95
                assert pass_ratio >= required_ratio, (
                    f"req={req}, b={b}, idx_k_head={idx_k_head}: "
                    f"Total {ref_output.numel()} elements, only {pass_ratio:.1%} meet tolerance criteria, "
                    f"require at least {required_ratio:.1%}"
                )
