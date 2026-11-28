import pytest
import torch

import flashinfer
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability

global_workspace_buffer = None  # can.be empty initialized
global_xqa_workspace_buffer = None  # must be zero initialized
workspace_size = 128 * 1024 * 1024


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("enable_pdl", [True, False, None])
@pytest.mark.parametrize("num_q_heads", [8, 16, 32, 64, 128])
def test_xqa_mla_batch_decode(
    batch_size: int,
    scale: float,
    page_size: int,
    enable_pdl: bool,
    num_q_heads: int,
):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 12:
        pytest.skip("These tests are only guaranteed to work on SM120/SM121 GPUs.")

    torch.manual_seed(42)
    dtype = torch.float8_e4m3fn
    q_len_per_request = 1
    device = "cuda:0"

    # Fixed max sequence length
    max_seq_len = 1024

    # Deepseek attention config (decode-MLA)
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 512

    # Initialize tensors
    query = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
    ).to(dtype)

    num_blocks_per_seq = (max_seq_len + page_size - 1) // page_size
    num_blocks = num_blocks_per_seq * batch_size

    # Sequence lengths and block tables
    seq_lens = [torch.randint(1, max_seq_len, (1,)).item() for _ in range(batch_size)]
    seq_lens[-1] = max_seq_len
    max_seq_len = max(seq_lens)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)

    blocks_per_seq = (seq_lens_tensor + page_size - 1) // page_size
    max_num_blocks_per_seq = blocks_per_seq.max().item()

    # Generate random but unique block IDs for all sequences
    total_blocks_needed = int(blocks_per_seq.sum().item())
    all_block_ids = torch.randperm(
        total_blocks_needed, device=device
    )  # Random permutation

    # Generate unique block IDs for all sequences
    block_id = 0
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int, device=device
    )

    # Populate block tables and track block assignments
    block_id = 0
    for i in range(batch_size):
        num_blocks_needed = int(blocks_per_seq[i].item())
        block_tables[i, :num_blocks_needed] = all_block_ids[
            block_id : block_id + num_blocks_needed
        ]
        block_id += num_blocks_needed

    # Create interleaved KV cache
    # Allocate more than needed blocks, block_id is just enough, to mimick real-world cases
    kv_cache = torch.randn(
        size=(num_blocks, page_size, kv_lora_rank + qk_rope_head_dim), device=device
    ).to(dtype)
    # (num_blocks, 1, page_size, kv_lora_rank + qk_rope_head_dim)

    global global_workspace_buffer, global_xqa_workspace_buffer
    if global_workspace_buffer is None:
        global_workspace_buffer = torch.empty(
            workspace_size, dtype=torch.int8, device=device
        )
    if global_xqa_workspace_buffer is None:
        global_xqa_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device=device
        )
    workspace_buffer = global_xqa_workspace_buffer
    workspace_buffer_ref = global_workspace_buffer

    # Run decode-MLA
    output = flashinfer.decode.xqa_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens_tensor,
        max_seq_len=max_seq_len,
        bmm1_scale=scale / ((128 + 64) ** 0.5),
        bmm2_scale=1.0,
        enable_pdl=enable_pdl,
    )

    # Run reference attention and align output
    sm_scale = scale / (
        (128 + 64) ** 0.5
    )  # use head dimension before matrix absorption
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer_ref,
        backend="fa2",
    )

    if dtype == torch.float8_e4m3fn:
        # convert query and kv_cache to bfloat16
        query = query.to(torch.bfloat16)
        kv_cache = kv_cache.to(torch.bfloat16)

    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * q_len_per_request
    )
    kv_indptr = torch.zeros_like(q_indptr)
    kv_indptr[1:] = torch.cumsum(blocks_per_seq, dim=0)
    kv_indices = all_block_ids.int()

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        seq_lens_tensor,
        num_q_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        True,
        sm_scale,
        query.dtype,
        kv_cache.dtype,
    )
    q_nope = query[..., :kv_lora_rank].view(
        batch_size * q_len_per_request, num_q_heads, kv_lora_rank
    )
    q_pe = query[..., kv_lora_rank:].view(
        batch_size * q_len_per_request, num_q_heads, qk_rope_head_dim
    )

    # todo: fix kv_cache
    ckv = kv_cache[..., :kv_lora_rank]
    kpe = kv_cache[..., kv_lora_rank:]

    o_ref = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    atol = 0.05
    rtol = 0.05

    diff_abs = torch.abs(
        o_ref.view(batch_size, q_len_per_request, num_q_heads, -1) - output
    )
    diff_rel = diff_abs / (
        torch.abs(o_ref.view(batch_size, q_len_per_request, num_q_heads, -1)) + 1e-8
    )

    within_tolerance = (diff_abs <= atol) | (diff_rel <= rtol)

    pass_ratio = within_tolerance.float().mean().item()

    required_ratio = 0.95
    assert pass_ratio >= required_ratio, (
        f"Total {o_ref.numel()} elements, only {pass_ratio:.1%} meet tolerance criteria, "
        f"require at least {required_ratio:.1%}"
    )


@pytest.fixture
def tensor_scale_case(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 or SM121")
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner(warmup=1, repeat=2))
    torch.manual_seed(482)
    return dict(
        query=torch.randn(2, 1, 128, 576, dtype=torch.bfloat16, device="cuda") * 0.5,
        kv=torch.randn(8, 32, 576, dtype=torch.bfloat16, device="cuda") * 0.5,
        tables=torch.tensor(
            [[3, 0, 2, 1], [6, 4, 7, 5]], dtype=torch.int32, device="cuda"
        ),
        lengths=torch.tensor([47, 91], dtype=torch.int32, device="cuda"),
        workspace=torch.zeros(128 * 1024**2, dtype=torch.uint8, device="cuda"),
    )


def _tensor_scale_reference(tensor_scale_case, q_scale=0.125, kv_scale=0.75):
    outputs = []
    for batch in range(2):
        query = tensor_scale_case["query"][batch, 0].float()
        kv = tensor_scale_case["kv"][tensor_scale_case["tables"][batch].long()].reshape(
            -1, 576
        )
        kv = kv[: tensor_scale_case["lengths"][batch].item()].float()
        # Preserve XQA's existing convention: the KV scale applies to K and V.
        scores = torch.einsum("hd,kd->hk", query, kv) * q_scale * kv_scale
        outputs.append(scores.softmax(-1) @ kv[:, :512] * kv_scale)
    return torch.stack(outputs)


def _tensor_scale_functional(
    tensor_scale_case, *, out=None, kv_4d=False, scales=(0.125, 0.75), **changes
):
    options = dict(
        query=tensor_scale_case["query"],
        kv_cache=(
            tensor_scale_case["kv"].unsqueeze(1) if kv_4d else tensor_scale_case["kv"]
        ),
        workspace_buffer=tensor_scale_case["workspace"],
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=tensor_scale_case["tables"],
        seq_lens=tensor_scale_case["lengths"],
        max_seq_len=128,
        out=out,
        bmm1_scale=scales[0],
        bmm2_scale=scales[1],
        enable_pdl=False,
    )
    options.update(changes)
    return flashinfer.decode.xqa_batch_decode_with_kv_cache_mla(**options)


def test_tensor_scales_tune_and_bind_current_inputs(tensor_scale_case, monkeypatch):
    tensor_scale_case["query"] = tensor_scale_case["query"].to(torch.float8_e4m3fn)
    tensor_scale_case["kv"] = tensor_scale_case["kv"].to(torch.float8_e4m3fn)
    scales = tuple(torch.tensor([v], device="cuda") for v in (0.125, 0.75))
    tuner = AutoTuner.get()
    with flashinfer.autotune(True):
        out = _tensor_scale_functional(tensor_scale_case, scales=scales)
    assert len(tuner.profiling_cache) == 1
    torch.testing.assert_close(
        out[:, 0].float(),
        _tensor_scale_reference(tensor_scale_case),
        rtol=3e-2,
        atol=3e-2,
    )

    def no_profile(*args, **kwargs):
        raise AssertionError("functional XQA inference performed tuning/cache work")

    monkeypatch.setattr(tuner, "_generate_optimization_profiles", no_profile)
    monkeypatch.setattr(tuner, "choose_one", no_profile)
    monkeypatch.setattr(tuner, "search_cache", no_profile)
    tensor_scale_case["query"] = (tensor_scale_case["query"].float() * 0.25).to(
        torch.float8_e4m3fn
    )
    scales = tuple(torch.tensor([v], device="cuda") for v in (0.25, 0.5))
    out = _tensor_scale_functional(tensor_scale_case, scales=scales)
    torch.testing.assert_close(
        out[:, 0].float(),
        _tensor_scale_reference(tensor_scale_case, 0.25, 0.5),
        rtol=3e-2,
        atol=3e-2,
    )


def _skip_unless_sm12x():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM120 or SM121")


def _head_group_case(batch_size, page_size, max_seq_len, num_q_heads, dtype, seed):
    """Paged decode inputs with mixed lengths (including 1 and tile edges).

    Every query head shares a direction ``u`` and each request's last token is a large
    multiple of ``u``, so every head's running max jumps on its last KV tile.
    """
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    lens = torch.randint(1, max_seq_len + 1, (batch_size,), generator=g, device=device)
    for i, edge in enumerate([max_seq_len, 1, 63, 64, 65, page_size + 1][:batch_size]):
        lens[i] = min(edge, max_seq_len)
    lens = lens.int()
    pages_per_seq = (lens + page_size - 1) // page_size
    align = 128 // page_size
    width = (int(pages_per_seq.max()) + align - 1) // align * align
    num_pages = int(pages_per_seq.sum())
    page_ids = torch.randperm(num_pages, generator=g, device=device).int()
    block_tables = torch.zeros(batch_size, width, dtype=torch.int32, device=device)
    offset = 0
    for b in range(batch_size):
        n = int(pages_per_seq[b])
        block_tables[b, :n] = page_ids[offset : offset + n]
        offset += n
    u = torch.randn(576, generator=g, device=device)
    u = u / u.norm()
    query = 0.5 * torch.randn(
        batch_size, 1, num_q_heads, 576, generator=g, device=device
    )
    query += 2 * u
    kv_cache = 0.5 * torch.randn(num_pages, page_size, 576, generator=g, device=device)
    for b in range(batch_size):
        last = int(lens[b]) - 1
        kv_cache[block_tables[b, last // page_size], last % page_size] = 30 * u
    return query.to(dtype), kv_cache.to(dtype), block_tables, lens


def _xqa_decode(query, kv_cache, block_tables, seq_lens, out=None):
    global global_xqa_workspace_buffer
    if global_xqa_workspace_buffer is None:
        global_xqa_workspace_buffer = torch.zeros(
            workspace_size, dtype=torch.int8, device="cuda"
        )
    return flashinfer.decode.xqa_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=global_xqa_workspace_buffer,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=block_tables.shape[1] * kv_cache.shape[1],
        out=out,
        bmm1_scale=1.0 / (128 + 64) ** 0.5,
        bmm2_scale=1.0,
        enable_pdl=False,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_xqa_mla_head_groups_match_128_head_kernel(dtype, page_size):
    """Query heads are independent rows, so 64 heads (64-head kernel) and 8/16/32 heads
    (64-head kernel on zero-padded heads) must reproduce the 128-head kernel bit for bit."""
    _skip_unless_sm12x()
    query, kv_cache, block_tables, seq_lens = _head_group_case(
        6, page_size, 3000, 128, dtype, seed=page_size
    )
    out_128 = _xqa_decode(query, kv_cache, block_tables, seq_lens)
    for num_q_heads in (8, 16, 32, 64):
        for start in (0, 128 - num_q_heads):
            heads = slice(start, start + num_q_heads)
            out = _xqa_decode(
                query[:, :, heads].contiguous(), kv_cache, block_tables, seq_lens
            )
            assert torch.equal(out, out_128[:, :, heads]), (num_q_heads, start)


@pytest.mark.parametrize("num_q_heads", [16, 64])
def test_xqa_mla_head_groups_cuda_graph(num_q_heads):
    """Graph replay of the padded (16) and native (64) head-group paths matches eager."""
    _skip_unless_sm12x()
    query, kv_cache, block_tables, seq_lens = _head_group_case(
        8, 64, 1024, num_q_heads, torch.float8_e4m3fn, seed=11
    )
    out = torch.empty(8, num_q_heads, 512, dtype=torch.bfloat16, device="cuda")
    _xqa_decode(query, kv_cache, block_tables, seq_lens, out=out)  # build + warm up
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _xqa_decode(query, kv_cache, block_tables, seq_lens, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    for scale in (0.5, 2.0):
        query.copy_((query.float() * scale).to(query.dtype))
        graph.replay()
        eager = _xqa_decode(query, kv_cache, block_tables, seq_lens)
        torch.cuda.synchronize()
        assert torch.equal(out, eager.squeeze(1))


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("num_q_heads", [16, 64])
def test_xqa_mla_head_groups_planned_wrapper(dtype, num_q_heads):
    """BatchMLAPagedAttentionWrapper(backend="xqa") plans the new head counts and matches
    the functional call bit for bit."""
    _skip_unless_sm12x()
    batch_size, page_size = 5, 32
    query, kv_cache, block_tables, seq_lens = _head_group_case(
        batch_size, page_size, 777, num_q_heads, dtype, seed=3
    )
    expected = _xqa_decode(query, kv_cache, block_tables, seq_lens)
    metadata = flashinfer.mla.MLAPlanMetadata.dense(
        torch.arange(batch_size + 1, dtype=torch.int32, device="cuda"),
        block_tables,
        seq_lens,
        max_q_len=1,
    )
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(workspace_size, dtype=torch.uint8, device="cuda"), backend="xqa"
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=num_q_heads,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=page_size,
        causal=False,
        sm_scale=1.0 / (128 + 64) ** 0.5,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_dtype=torch.bfloat16,
        enable_pdl=False,
        scale_mode="bmm-scalar",
    )
    out = torch.empty(batch_size, num_q_heads, 512, dtype=torch.bfloat16, device="cuda")
    wrapper.run(
        query=query.squeeze(1),
        kv_cache=kv_cache,
        out=out,
        bmm1_scale=1.0 / (128 + 64) ** 0.5,
        bmm2_scale=1.0,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, expected.squeeze(1))
