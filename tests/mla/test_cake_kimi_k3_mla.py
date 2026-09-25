"""GPU tests for the CAKE Kimi-K3 MLA FP8 paged-attention backend (SM100 / SM103).

Covers the three call families of the Cake contract against an FP32 reference: low-head paged
decode (q_len = 1), packed variable-Q / MTP (cum_seq_lens_q) and incremental prefill on the
paged FP8 cache (prefix reuse, ragged KV), plus CUDA-Graph replay with a changed page table.
"""

import math

import pytest
import torch

from flashinfer.utils import get_compute_capability

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
PAGE = 64


def _skip_unless_sm100_family():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _minor = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("CAKE Kimi-K3 MLA requires SM100 / SM103")


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _make_case(batch, q_lens, kv_lens, num_heads, *, seed, device, pool_pages=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_seq = [(k + PAGE - 1) // PAGE for k in kv_lens]
    width = max(pages_per_seq)
    total_pages = sum(pages_per_seq)
    pool_pages = pool_pages or total_pages + 8
    kv_cache = _fp8(torch.randn((pool_pages, PAGE, QK_DIM), generator=gen, device=device) * 0.5)
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages]
    block_tables = torch.zeros((batch, width), dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(pages_per_seq):
        block_tables[b, :n] = perm[off : off + n].to(torch.int32)
        off += n
    total_q = sum(q_lens)
    query = _fp8(torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device) * 0.5)
    q_indptr = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return dict(
        query=query, kv_cache=kv_cache, block_tables=block_tables, seq_lens=seq_lens, q_indptr=q_indptr,
        q_lens=list(q_lens), kv_lens=list(kv_lens), num_heads=num_heads,
        bmm1_scale=1.0 / math.sqrt(LATENT + ROPE), bmm2_scale=1.0,
    )


def _reference(case) -> torch.Tensor:
    cache = case["kv_cache"]
    num_heads = case["num_heads"]
    q_rows = case["query"].float()
    out = torch.zeros((q_rows.shape[0], num_heads, LATENT), dtype=torch.float32, device=q_rows.device)
    q_indptr = case["q_indptr"].tolist()
    for b, (q_len, kv_len) in enumerate(zip(case["q_lens"], case["kv_lens"], strict=True)):
        n_pages = (kv_len + PAGE - 1) // PAGE
        pages = case["block_tables"][b, :n_pages].long()
        values = cache[pages].reshape(-1, QK_DIM)[:kv_len].float()
        q = q_rows[q_indptr[b] : q_indptr[b + 1]].reshape(q_len * num_heads, QK_DIM)
        logits = (q @ values.T) * case["bmm1_scale"]
        if q_len > 1:
            positions = torch.arange(kv_len, device=q.device)
            limit = kv_len - q_len + torch.arange(q_len, device=q.device) + 1
            mask = positions[None, :] < limit[:, None]
            logits = logits.reshape(q_len, num_heads, kv_len).masked_fill(~mask[:, None, :], float("-inf"))
            logits = logits.reshape(q_len * num_heads, kv_len)
        probs = torch.softmax(logits, dim=-1)
        out[q_indptr[b] : q_indptr[b + 1]] = (probs @ values[:, :LATENT]).reshape(q_len, num_heads, LATENT)
    return (out * case["bmm2_scale"]).to(torch.bfloat16)


def _run(case, *, fixed_q_len=None, graph=False):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.mla.cake_kimi_k3_mla import workspace_bytes

    device = case["query"].device
    num_heads = case["num_heads"]
    total_q = case["query"].shape[0]
    workspace = torch.zeros(workspace_bytes(total_q * num_heads, 256), dtype=torch.uint8, device=device)
    out = torch.full((total_q, num_heads, LATENT), float("nan"), dtype=torch.bfloat16, device=device)
    kwargs = dict(
        kv_cache=case["kv_cache"], workspace_buffer=workspace, qk_nope_head_dim=128, kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE, block_tables=case["block_tables"], seq_lens=case["seq_lens"],
        max_seq_len=int(case["block_tables"].shape[1]) * PAGE, bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"], backend="cake",
    )
    if fixed_q_len is not None:
        batch = len(case["q_lens"])
        query = case["query"].reshape(batch, fixed_q_len, num_heads, QK_DIM)
        out_view = out.reshape(batch, fixed_q_len, num_heads, LATENT)
        call = lambda: trtllm_batch_decode_with_kv_cache_mla(query, out=out_view, **kwargs)  # noqa: E731
    else:
        call = lambda: trtllm_batch_decode_with_kv_cache_mla(  # noqa: E731
            case["query"], out=out, cum_seq_lens_q=case["q_indptr"], max_q_len=max(case["q_lens"]), **kwargs
        )
    if not graph:
        call()
        torch.cuda.synchronize()
        return out
    call()  # warm the JIT modules outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
        call()
    torch.cuda.synchronize()
    return out, g


def _check(out, ref, *, atol, rtol):
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("num_heads", [12, 96])
@pytest.mark.parametrize("kv_lens", [[1, 200, 64], [4096, 777, 65]])
def test_decode_q1(num_heads, kv_lens):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(len(kv_lens), [1] * len(kv_lens), kv_lens, num_heads, seed=645001, device=device)
    out = _run(case, fixed_q_len=1)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_heads,q_lens,kv_lens", [
    (12, [1, 5, 8, 3], [300, 1500, 64, 129]),
    (12, [5, 2], [1500, 2600]),  # 60 query rows -> the 64-row tile
    (96, [4, 4], [2048, 333]),
])
def test_mtp_variable_q(num_heads, q_lens, kv_lens):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(len(q_lens), q_lens, kv_lens, num_heads, seed=645003, device=device)
    out = _run(case)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


def test_incremental_prefill_prefix_reuse():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    # Incremental prefill: 384 new tokens appended to 4096 cached tokens (prefix reuse), ragged
    # second request with a partially filled last page.
    case = _make_case(2, [384, 97], [4096 + 384, 97 + 1000], 12, seed=645006, device=device)
    out = _run(case)
    _check(out, _reference(case), atol=5e-3, rtol=2e-2)


def test_cuda_graph_replay_changing_page_table():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(3, [1, 1, 1], [500, 2100, 64], 12, seed=645008, device=device, pool_pages=128)
    out, g = _run(case, fixed_q_len=1, graph=True)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)
    # Same shapes, new page assignment and new cache contents: replay must follow the tables.
    gen = torch.Generator(device=device).manual_seed(7)
    case["kv_cache"].copy_(_fp8(torch.randn(case["kv_cache"].shape, generator=gen, device=device) * 0.5))
    perm = torch.randperm(128, generator=gen, device=device).to(torch.int32)
    width = case["block_tables"].shape[1]
    for b in range(3):
        n = (case["kv_lens"][b] + PAGE - 1) // PAGE
        case["block_tables"][b, :n] = perm[b * width : b * width + n]
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


def test_unsupported_options_rejected():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(1, [1], [128], 12, seed=1, device=device)
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    workspace = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="does not support"):
        trtllm_batch_decode_with_kv_cache_mla(
            case["query"].reshape(1, 1, 12, QK_DIM), kv_cache=case["kv_cache"], workspace_buffer=workspace,
            qk_nope_head_dim=128, kv_lora_rank=LATENT, qk_rope_head_dim=ROPE, block_tables=case["block_tables"],
            seq_lens=case["seq_lens"], max_seq_len=128, bmm1_scale=1.0, backend="cake", return_lse=True,
        )
