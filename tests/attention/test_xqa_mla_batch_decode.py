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
def test_xqa_mla_batch_decode(
    batch_size: int,
    scale: float,
    page_size: int,
    enable_pdl: bool,
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
    num_q_heads = 128
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
