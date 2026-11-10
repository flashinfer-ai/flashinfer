"""GPU tests for the CAKE Kimi-K3 MLA FP8 paged-attention backend (SM100 / SM103).

Covers the three call families against an FP32 reference: low-head paged decode (q_len = 1),
packed variable-Q / MTP (cum_seq_lens_q) and incremental prefill on the paged FP8 cache (prefix
reuse, ragged KV), plus CUDA-Graph replay with a changed page table, row-strided query / cache
views on a row-tile route and on the wide route, the route selection and the split planners,
and the extended programs (LSE output, non-causal masking, cyclic decode context parallelism)
against an FP64 reference (rows over at most FEW_KEYS visible keys against the default programs
and an E4M3-probability reference).
"""

import dataclasses
import math

import pytest
import torch

from flashinfer.jit.cake_kimi_k3_mla import (
    KERNELS,
    MODULES,
    get_cake_kimi_k3_mla_kernel,
    supported_arches,
)
from flashinfer.utils import get_compute_capability

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
PAGE = 64


def _skip_unless_sm100_family():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, minor = get_compute_capability(torch.device("cuda"))
    arch = f"sm_{major}{minor}a"
    # Only architectures with generated programs (sm_100a, sm_103a); other SM100-family parts
    # such as sm_107a are not registered.
    if arch not in supported_arches():
        pytest.skip(f"CAKE Kimi-K3 MLA has no generated programs for {arch}")


def _fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _make_case(batch, q_lens, kv_lens, num_heads, *, seed, device, pool_pages=None):
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_seq = [(k + PAGE - 1) // PAGE for k in kv_lens]
    width = max(pages_per_seq)
    total_pages = sum(pages_per_seq)
    pool_pages = pool_pages or total_pages + 8
    kv_cache = _fp8(
        torch.randn((pool_pages, PAGE, QK_DIM), generator=gen, device=device) * 0.5
    )
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages]
    block_tables = torch.zeros((batch, width), dtype=torch.int32, device=device)
    off = 0
    for b, n in enumerate(pages_per_seq):
        block_tables[b, :n] = perm[off : off + n].to(torch.int32)
        off += n
    total_q = sum(q_lens)
    query = _fp8(
        torch.randn((total_q, num_heads, QK_DIM), generator=gen, device=device) * 0.5
    )
    q_indptr = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    return dict(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        q_lens=list(q_lens),
        kv_lens=list(kv_lens),
        num_heads=num_heads,
        bmm1_scale=1.0 / math.sqrt(LATENT + ROPE),
        bmm2_scale=1.0,
    )


def _reference(case) -> torch.Tensor:
    cache = case["kv_cache"]
    num_heads = case["num_heads"]
    q_rows = case["query"].float()
    out = torch.zeros(
        (q_rows.shape[0], num_heads, LATENT), dtype=torch.float32, device=q_rows.device
    )
    q_indptr = case["q_indptr"].tolist()
    for b, (q_len, kv_len) in enumerate(
        zip(case["q_lens"], case["kv_lens"], strict=True)
    ):
        n_pages = (kv_len + PAGE - 1) // PAGE
        pages = case["block_tables"][b, :n_pages].long()
        values = cache[pages].reshape(-1, QK_DIM)[:kv_len].float()
        q = q_rows[q_indptr[b] : q_indptr[b + 1]].reshape(q_len * num_heads, QK_DIM)
        logits = (q @ values.T) * case["bmm1_scale"]
        if q_len > 1:
            positions = torch.arange(kv_len, device=q.device)
            limit = kv_len - q_len + torch.arange(q_len, device=q.device) + 1
            mask = positions[None, :] < limit[:, None]
            logits = logits.reshape(q_len, num_heads, kv_len).masked_fill(
                ~mask[:, None, :], float("-inf")
            )
            logits = logits.reshape(q_len * num_heads, kv_len)
        probs = torch.softmax(logits, dim=-1)
        out[q_indptr[b] : q_indptr[b + 1]] = (probs @ values[:, :LATENT]).reshape(
            q_len, num_heads, LATENT
        )
    return (out * case["bmm2_scale"]).to(torch.bfloat16)


def _run(case, *, fixed_q_len=None, graph=False):
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.mla.cake_kimi_k3_mla import workspace_bytes

    device = case["query"].device
    num_heads = case["num_heads"]
    total_q = case["query"].shape[0]
    workspace = torch.zeros(
        workspace_bytes(total_q * num_heads, 256), dtype=torch.uint8, device=device
    )
    out = torch.full(
        (total_q, num_heads, LATENT), float("nan"), dtype=torch.bfloat16, device=device
    )
    kwargs = dict(
        kv_cache=case["kv_cache"],
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=int(case["block_tables"].shape[1]) * PAGE,
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"],
        backend="cake",
    )
    if fixed_q_len is not None:
        batch = len(case["q_lens"])
        query = case["query"].reshape(batch, fixed_q_len, num_heads, QK_DIM)
        out_view = out.reshape(batch, fixed_q_len, num_heads, LATENT)
        call = lambda: trtllm_batch_decode_with_kv_cache_mla(
            query, out=out_view, **kwargs
        )  # noqa: E731
    else:
        call = lambda: trtllm_batch_decode_with_kv_cache_mla(  # noqa: E731
            case["query"],
            out=out,
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=max(case["q_lens"]),
            **kwargs,
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
@pytest.mark.parametrize(
    "kv_lens", [[1, 200, 64], [4096, 777, 65], [20000, 16384, 17001]]
)
def test_decode_q1(num_heads, kv_lens):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(
        len(kv_lens), [1] * len(kv_lens), kv_lens, num_heads, seed=645001, device=device
    )
    out = _run(case, fixed_q_len=1)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "num_heads,q_lens,kv_lens",
    [
        (12, [1, 5, 8, 3], [300, 1500, 64, 129]),
        (12, [5, 2], [1500, 2600]),  # 60 query rows -> the 64-row tile
        (96, [4, 4], [2048, 333]),
        (96, [4, 3], [20000, 16385]),  # two-CTA wide route (rows > 64, KV >= 8192)
    ],
)
def test_mtp_variable_q(num_heads, q_lens, kv_lens):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(
        len(q_lens), q_lens, kv_lens, num_heads, seed=645003, device=device
    )
    out = _run(case)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "num_heads,kv_lens",
    [
        (12, [770]),  # one request: the 16-row tile, 7 KV tiles, three splits
        (
            12,
            [770, 768, 785, 802, 812, 738, 889, 741],
        ),  # the Kimi-K3 layer-91 band (11-14 pages)
        (96, [6000, 5000]),  # 96-row tile, always-exact path
        (96, [9000]),  # two-CTA wide route
    ],
)
def test_decode_sink_key_rows(num_heads, kv_lens):
    """A sink key ~150 octaves above every other key for most heads of a row tile, flat scores for the others.

    The lazy E4M3 reference of the row-tile programs is re-referenced per tile for every column of an exceeding
    warpgroup; before the bounded-drop rule a sink-less column's later move pulled the sink columns ~120 octaves
    down and the FP32 rescale overflowed (Inf / NaN rows in Kimi-K3 serving).  The output must be finite and match
    the FP32 reference.
    """
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(
        len(kv_lens), [1] * len(kv_lens), kv_lens, num_heads, seed=645011, device=device
    )
    sink_heads = [h for h in range(num_heads) if h % 4 != 3]
    for b in range(len(kv_lens)):
        page = int(case["block_tables"][b, 0].item())
        case["kv_cache"][page, 0] = _fp8(torch.full((QK_DIM,), 2.0, device=device))
        case["query"][b, sink_heads] = _fp8(torch.full((QK_DIM,), 1.25, device=device))
    out = _run(case, fixed_q_len=1)
    _check(out, _reference(case), atol=0.1, rtol=0.1)
    rel = (out.float() - _reference(case).float()).norm(dim=-1) / _reference(
        case
    ).float().norm(dim=-1).clamp_min(1e-6)
    assert float(rel.max()) <= 0.05


def test_incremental_prefill_prefix_reuse():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    # Incremental prefill: 384 new tokens appended to 4096 cached tokens (prefix reuse), ragged
    # second request with a partially filled last page.
    case = _make_case(
        2, [384, 97], [4096 + 384, 97 + 1000], 12, seed=645006, device=device
    )
    out = _run(case)
    # Uniform-profile contract tolerance of the Cake route (FP32 reference): atol 0.01 / rtol 0.02.
    _check(out, _reference(case), atol=1e-2, rtol=2e-2)


def test_prefill_wide_route():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    # 512 new tokens x 12 heads = 6144 packed rows over a 32768-token cached prefix: the two-CTA
    # wide route with per-row causal tails (bottom-right aligned) and a planned KV split.
    case = _make_case(1, [512], [32768], 12, seed=645402, device=device)
    out = _run(case)
    _check(out, _reference(case), atol=1e-2, rtol=2e-2)


def test_route_selection():
    _skip_unless_sm100_family()
    from flashinfer.mla.cake_kimi_k3_mla import (
        KimiK3MlaFp8PagedAttention,
        workspace_bytes,
    )

    device = torch.device("cuda")
    workspace = torch.zeros(64 << 20, dtype=torch.uint8, device=device)

    def runner(case):
        total_q = case["query"].shape[0]
        out = torch.empty(
            (total_q, case["num_heads"], LATENT), dtype=torch.bfloat16, device=device
        )
        assert workspace.numel() >= workspace_bytes(total_q * case["num_heads"], 256)
        return KimiK3MlaFp8PagedAttention(
            query=case["query"],
            kv_cache=case["kv_cache"],
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            out=out,
            workspace_buffer=workspace,
            bmm1_scale=case["bmm1_scale"],
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=max(case["q_lens"]),
            max_seq_len=int(case["block_tables"].shape[1]) * PAGE,
        )

    # 96 rows, longest KV 20000 -> wide route: a flat grid of two-CTA clusters, num_split per (tile, request) item
    # (two items, fewer than the SM pairs: nothing runs unsplit).
    wide = runner(_make_case(2, [1, 1], [20000, 300], 96, seed=1, device=device))
    assert wide.route_metadata["route"] == "wide" and wide.rt is None
    assert wide.plan.n_full_items == 0
    assert wide.plan.grid_main == (2 * 2 * wide.num_split, 1, 1)
    # 96 rows but longest KV below 8192 -> row tiles (lazy-E4M3 precision gate of the wide route).
    # max_seq_len is the page-rounded table width, so the longest KV must stay below 8192 pages-wise.
    short = runner(_make_case(2, [1, 1], [8000, 300], 96, seed=2, device=device))
    assert short.route_metadata["route"] == "swapped" and short.rt == 96
    # 12 rows -> the 16-row tile whatever the KV.
    small = runner(_make_case(1, [1], [40000], 12, seed=3, device=device))
    assert small.route_metadata["route"] == "swapped" and small.rt == 16
    assert small.plan.grid_main == (small.num_split, 1, 1)


def test_cuda_graph_replay_changing_page_table():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(
        3, [1, 1, 1], [500, 2100, 64], 12, seed=645008, device=device, pool_pages=128
    )
    out, g = _run(case, fixed_q_len=1, graph=True)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)
    # Same shapes, new page assignment and new cache contents: replay must follow the tables.
    gen = torch.Generator(device=device).manual_seed(7)
    case["kv_cache"].copy_(
        _fp8(torch.randn(case["kv_cache"].shape, generator=gen, device=device) * 0.5)
    )
    perm = torch.randperm(128, generator=gen, device=device).to(torch.int32)
    width = case["block_tables"].shape[1]
    for b in range(3):
        n = (case["kv_lens"][b] + PAGE - 1) // PAGE
        case["block_tables"][b, :n] = perm[b * width : b * width + n]
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


def test_cuda_graph_replay_wide_route():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    # H96 decode over long KV (two-CTA wide route, planned KV split + merge) captured once and replayed
    # with a new page assignment and new cache contents.
    case = _make_case(
        2, [1, 1], [20000, 16384], 96, seed=645205, device=device, pool_pages=1024
    )
    out, g = _run(case, fixed_q_len=1, graph=True)
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)
    gen = torch.Generator(device=device).manual_seed(11)
    case["kv_cache"].copy_(
        _fp8(torch.randn(case["kv_cache"].shape, generator=gen, device=device) * 0.5)
    )
    perm = torch.randperm(1024, generator=gen, device=device).to(torch.int32)
    width = case["block_tables"].shape[1]
    for b in range(2):
        n = (case["kv_lens"][b] + PAGE - 1) // PAGE
        case["block_tables"][b, :n] = perm[b * width : b * width + n]
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    _check(out, _reference(case), atol=1e-2, rtol=1e-2)


def test_row_strided_query_and_kv_views_match_contiguous():
    _skip_unless_sm100_family()
    from flashinfer.mla.cake_kimi_k3_mla import (
        KimiK3MlaFp8PagedAttention,
        workspace_bytes,
    )

    device = torch.device("cuda")
    case = _make_case(2, [3, 1], [5000, 300], 12, seed=645301, device=device)
    total_q, num_heads = case["query"].shape[:2]
    workspace = torch.zeros(
        workspace_bytes(total_q * num_heads, 256), dtype=torch.uint8, device=device
    )

    def run(query, kv_cache):
        out = torch.full(
            (total_q, num_heads, LATENT),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        attention = KimiK3MlaFp8PagedAttention(
            query=query,
            kv_cache=kv_cache,
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            out=out,
            workspace_buffer=workspace,
            bmm1_scale=case["bmm1_scale"],
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=max(case["q_lens"]),
            max_seq_len=int(case["block_tables"].shape[1]) * PAGE,
        )
        # This case runs a row tile (36 rows, longest KV 5000 -> main_rt48); the wide route is
        # covered by test_wide_route_row_strided_views_match_contiguous.
        assert attention.route_metadata["route"] == "swapped"
        assert attention.rt == 48
        attention.launch()
        torch.cuda.synchronize()
        return out

    dense = run(case["query"], case["kv_cache"])
    # Query rows living inside a wider per-row buffer (640-byte rows) and a cache whose token rows
    # carry 64 bytes of padding: both are addressed through the row stride, no copy.
    wide_q = torch.zeros(
        (total_q, num_heads, QK_DIM + 64), dtype=torch.float8_e4m3fn, device=device
    )
    wide_q[..., :QK_DIM] = case["query"]
    pages = case["kv_cache"].shape[0]
    wide_kv = torch.zeros(
        (pages, PAGE, QK_DIM + 64), dtype=torch.float8_e4m3fn, device=device
    )
    wide_kv[..., :QK_DIM] = case["kv_cache"]
    assert wide_q[..., :QK_DIM].stride(-2) == QK_DIM + 64
    assert wide_kv[..., :QK_DIM].stride(-2) == QK_DIM + 64
    strided = run(wide_q[..., :QK_DIM], wide_kv[..., :QK_DIM])
    assert torch.equal(strided, dense)
    _check(dense, _reference(case), atol=1e-2, rtol=2e-2)
    # The output rows are written with a dense 512-element stride; a wider view is rejected.
    wide_out = torch.empty(
        (total_q, num_heads, LATENT + 64), dtype=torch.bfloat16, device=device
    )
    with pytest.raises(ValueError, match="out rows must be dense"):
        KimiK3MlaFp8PagedAttention(
            query=case["query"],
            kv_cache=case["kv_cache"],
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            out=wide_out[..., :LATENT],
            workspace_buffer=workspace,
            bmm1_scale=case["bmm1_scale"],
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=max(case["q_lens"]),
        )


def test_wide_route_row_strided_views_match_contiguous():
    _skip_unless_sm100_family()
    from flashinfer.mla.cake_kimi_k3_mla import (
        KimiK3MlaFp8PagedAttention,
        use_wide_route,
        workspace_bytes,
    )

    device = torch.device("cuda")
    # 96 heads, longest KV 8192 -> the two-CTA wide route; its TMA descriptors carry the row
    # stride like the row-tile programs, so padded query / cache rows are addressed in place.
    case = _make_case(1, [1], [8192], 96, seed=645302, device=device)
    assert use_wide_route(96, 8192)
    total_q, num_heads = case["query"].shape[:2]
    workspace = torch.zeros(
        workspace_bytes(total_q * num_heads, 256), dtype=torch.uint8, device=device
    )

    def run(query, kv_cache):
        out = torch.full(
            (total_q, num_heads, LATENT),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        attention = KimiK3MlaFp8PagedAttention(
            query=query,
            kv_cache=kv_cache,
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            out=out,
            workspace_buffer=workspace,
            bmm1_scale=case["bmm1_scale"],
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=max(case["q_lens"]),
            max_seq_len=int(case["block_tables"].shape[1]) * PAGE,
        )
        assert attention.route_metadata["route"] == "wide"
        attention.launch()
        torch.cuda.synchronize()
        return out

    dense = run(case["query"], case["kv_cache"])
    wide_q = torch.zeros(
        (total_q, num_heads, QK_DIM + 64), dtype=torch.float8_e4m3fn, device=device
    )
    wide_q[..., :QK_DIM] = case["query"]
    pages = case["kv_cache"].shape[0]
    wide_kv = torch.zeros(
        (pages, PAGE, QK_DIM + 64), dtype=torch.float8_e4m3fn, device=device
    )
    wide_kv[..., :QK_DIM] = case["kv_cache"]
    assert wide_q[..., :QK_DIM].stride(-2) == QK_DIM + 64
    assert wide_kv[..., :QK_DIM].stride(-2) == QK_DIM + 64
    for query, kv_cache in (
        (wide_q[..., :QK_DIM], case["kv_cache"]),
        (case["query"], wide_kv[..., :QK_DIM]),
        (wide_q[..., :QK_DIM], wide_kv[..., :QK_DIM]),
    ):
        assert torch.equal(run(query, kv_cache), dense)
    _check(dense, _reference(case), atol=1e-2, rtol=1e-2)


def test_plan_is_shared_across_calls_of_one_shape():
    from flashinfer.mla.cake_kimi_k3_mla import plan_attention

    first = plan_attention(
        batch=8, max_q_len=1, num_heads=96, max_seq_len=342305, sm_count=148
    )
    second = plan_attention(
        batch=8, max_q_len=1, num_heads=96, max_seq_len=342305, sm_count=148
    )
    assert first is second
    assert first.wide and first.main_kind == "main_wide" and first.num_split == 9
    narrow = plan_attention(
        batch=1, max_q_len=1, num_heads=12, max_seq_len=40000, sm_count=148
    )
    assert not narrow.wide and narrow.main_kind == "main_rt16_late"


def test_late_trigger_route_follows_the_main_grid():
    from flashinfer.mla.cake_kimi_k3_mla import (
        LATE_TRIGGER_MIN_MAIN_CTAS,
        plan_attention,
    )

    # Main CTAs on 152 SMs: 16, 32, and 128 with one split (no reducer).
    for batch, max_seq_len, kind in (
        (1, 4096, "main_rt16"),
        (8, 1024, "main_rt16_late"),
        (128, 1024, "main_rt16"),
    ):
        plan = plan_attention(
            batch=batch,
            max_q_len=1,
            num_heads=12,
            max_seq_len=max_seq_len,
            sm_count=152,
        )
        ctas = plan.grid_main[0] * plan.grid_main[1] * plan.grid_main[2]
        assert plan.main_kind == kind
        assert (kind == "main_rt16_late") == (
            plan.num_split > 1 and ctas >= LATE_TRIGGER_MIN_MAIN_CTAS
        )


def test_every_kernel_key_resolves_on_the_running_architecture():
    _skip_unless_sm100_family()
    major, minor = get_compute_capability(torch.device("cuda"))
    arch = f"sm_{major}{minor}a"
    for key, entry in KERNELS.items():
        record = get_cake_kimi_k3_mla_kernel(key, arch=arch)
        assert arch in record["arches"]
        assert record["name"] == (entry if isinstance(entry, str) else entry[arch])
        assert set(MODULES[record["name"]]) >= {
            "sources",
            "compile_flags",
            "ffi_entry",
            "arg_plan",
            "arches",
        }
    # A per-architecture key names, for each supported architecture, a program generated for it.
    for _key, entry in KERNELS.items():
        if isinstance(entry, dict):
            assert set(entry) == supported_arches()
            assert all(
                name_arch in MODULES[name]["arches"]
                for name_arch, name in entry.items()
            )


def test_unsupported_options_rejected():
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    case = _make_case(1, [1], [128], 12, seed=1, device=device)
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    workspace = torch.zeros(1 << 20, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="does not support"):
        trtllm_batch_decode_with_kv_cache_mla(
            case["query"].reshape(1, 1, 12, QK_DIM),
            kv_cache=case["kv_cache"],
            workspace_buffer=workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            max_seq_len=128,
            bmm1_scale=1.0,
            backend="cake",
            sinks=torch.zeros(12, dtype=torch.float32, device=device),
        )


@pytest.mark.parametrize(
    ("clusters", "max_seq_len", "sm_count", "expected"),
    [
        # B200 (148 SMs = 74 pairs): the measured-best plan of every long decode row is the largest
        # split count that still fits one wave; the per-wave cost term keeps the planner there.
        (6, 467751, 148, 12),
        (7, 299613, 148, 10),
        (8, 342305, 148, 9),
        (48, 342305, 148, 3),
        # B300 (160 SMs = 80 pairs).
        (6, 467751, 160, 13),
        (8, 342305, 160, 10),
        # GB300 NVL72 (152 SMs = 76 pairs): 19 splits = two full waves ran 286 us against 9 splits =
        # one wave at 278 us on the 8-request H96 decode row; the 16-cluster MTP row 14 -> 9 splits
        # (528 -> 516 us) and the 48-cluster MTP row 11 -> 3 splits (1552 -> 1486 us).
        (8, 342305, 152, 9),
        (16, 342305, 152, 9),
        (48, 342305, 152, 3),
    ],
)
def test_wide_split_plan_fills_one_wave(clusters, max_seq_len, sm_count, expected):
    from flashinfer.mla.cake_kimi_k3_mla import (
        WIDE_CLUSTER,
        WIDE_WAVE_COST_TILES,
        plan_num_split_wide,
    )

    assert WIDE_WAVE_COST_TILES == 16
    splits = plan_num_split_wide(clusters, max_seq_len, sm_count)
    assert splits == expected
    assert -(-(clusters * splits) // (sm_count // WIDE_CLUSTER)) <= 2
    # Without the per-wave term the pure wave model prefers three waves of 99 tiles on this row.
    assert plan_num_split_wide(6, 467751, 148, wave_cost_tiles=0) == 37
    # The former 8-tile term let two full waves of 141 tiles beat one wave of 298 on 152 SMs.
    assert plan_num_split_wide(8, 342305, 152, wave_cost_tiles=8) == 19


@pytest.mark.parametrize(
    ("clusters", "max_seq_len", "sm_count", "expected"),
    [
        # prefill_h12_b1 on 148 SMs: two full waves of 74 unsplit items, the 44 tail items split five ways
        # (the r50 forced-split sweep measured uniform 3 splits flat and 5 splits 3.5 % slower than unsplit).
        (192, 131072, 148, (148, 5)),
        # prefill_h96_b2: ten full waves, the 28 tail items split in two (one wave of 56).
        (768, 32768, 148, (740, 2)),
        # prefill_h96_b1: the last wave holds 56 of 74 pairs; the predicted gain is under the 2 % margin.
        (1536, 131072, 148, (0, 1)),
        # Decode rows (fewer items than pairs) keep the uniform one-wave plan.
        (8, 342305, 148, (0, 9)),
        (48, 342305, 148, (0, 3)),
    ],
)
def test_wide_tail_plan(clusters, max_seq_len, sm_count, expected):
    from flashinfer.mla.cake_kimi_k3_mla import plan_wide_work

    assert plan_wide_work(clusters, max_seq_len, sm_count) == expected
    assert plan_wide_work(clusters, max_seq_len, sm_count, forced_split=3) == (0, 3)


# ---------------------------------------------------------------------------
# Persistent wide route (main_wide_persistent + reduce_w<N>_tile_flags)
# ---------------------------------------------------------------------------

# Devices as the route keys them: ((major, minor) compute capability, SM count).
B200 = ((10, 0), 148)
B300 = ((10, 3), 148)
GB300 = ((10, 3), 152)

# (batch, q_len, num_heads, kv): wide tail plans on both sides of the item-length bound and of
# the reducer class, and a uniform plan.  On B200: (13, 64, 12, 8192), (13, 128, 12, 10240)
# and (16, 6, 96, 8192) merge on the two-warp reducer and take the persistent route (the last
# also keeps the extended persistent program covered); (1, 1024, 12, 8192) and
# (7, 256, 12, 12288) merge on the single-warp reducer, (5, 256, 24, 24576) streams items longer
# than PERSISTENT_MAX_ITEM_KV and (1, 1024, 12, 16384) is a uniform plan (no unsplit item), so
# these keep the shipped kernels.
PERSISTENT_CASES = [
    (13, 64, 12, 8192),
    (13, 128, 12, 10240),
    (1, 1024, 12, 8192),
    (7, 256, 12, 12288),
    (16, 6, 96, 8192),
    (5, 256, 24, 24576),
    (1, 1024, 12, 16384),
]


def _sm_count() -> int:
    return int(
        torch.cuda.get_device_properties(torch.device("cuda")).multi_processor_count
    )


def _device():
    """``(compute capability, SM count)`` of the current CUDA device."""
    return tuple(get_compute_capability(torch.device("cuda"))), _sm_count()


def _persistent_route(batch, q_len, num_heads, kv, device):
    from flashinfer.mla.cake_kimi_k3_mla import plan_attention, use_persistent_route

    capability, sm_count = device
    plan = plan_attention(
        batch=batch,
        max_q_len=q_len,
        num_heads=num_heads,
        max_seq_len=kv,
        sm_count=sm_count,
    )
    return plan, use_persistent_route(plan, capability, sm_count, kv)


def test_persistent_route_predicate():
    from flashinfer.mla.cake_kimi_k3_mla import (
        PERSISTENT_MAX_ITEM_KV,
        PERSISTENT_SHIPPED_REDUCE_WARPS,
        use_persistent_route,
    )

    assert PERSISTENT_MAX_ITEM_KV == {B200: 16384}
    assert PERSISTENT_SHIPPED_REDUCE_WARPS == {B200: (1,)}
    # Tail plans (unsplit full waves, split last wave) with a second round and idle pairs whose
    # merge runs on the two-warp reducer.
    for shape in ((13, 64, 12, 8192), (13, 128, 12, 8192), (13, 128, 12, 10240)):
        plan, routed = _persistent_route(*shape, B200)
        assert plan.wide and plan.n_full_items > 0 and 2 <= plan.num_split <= 32
        assert plan.reduce_kind == "reduce_w2", (shape, plan.reduce_kind)
        assert routed, shape
    # The same tail-plan conditions with a single-warp reducer (more than 1024 merge rows) keep
    # the shipped kernels on B200; the reducer class alone decides.
    for shape in (
        (4, 1024, 12, 8192),
        (1, 1024, 12, 8192),
        (13, 256, 12, 16384),
        (15, 64, 24, 16384),
    ):
        plan, routed = _persistent_route(*shape, B200)
        assert plan.wide and plan.n_full_items > 0 and 2 <= plan.num_split <= 32
        assert plan.reduce_kind == "reduce_w1", (shape, plan.reduce_kind)
        assert not routed, shape
        for warps in (2, 4):
            other = dataclasses.replace(plan, reduce_warps=warps)
            assert use_persistent_route(other, *B200, shape[3]), (shape, warps)
    # Items longer than the bound, uniform plans, row tiles and devices without a bound keep the
    # shipped kernels.
    for shape, device in (
        ((5, 256, 24, 24576), B200),
        ((1, 1024, 12, 16384), B200),
        ((8, 2, 96, 342305), B200),
        ((1, 1, 12, 40000), B200),
        ((4, 1024, 12, 8192), ((10, 0), 160)),
        ((13, 64, 12, 8192), GB300),
    ):
        assert not _persistent_route(*shape, device)[1], (shape, device)


def test_persistent_route_keys_on_compute_capability_and_sm_count():
    from flashinfer.mla.cake_kimi_k3_mla import use_persistent_route

    # Two-warp reducer tail plans that take the persistent route on B200 (compute capability 10.0,
    # 148 SMs).  B300 (10.3) has the same SM count, hence the same plan, and keeps the shipped
    # kernels whatever the plan's reducer class, as do GB300 (10.3, 152 SMs) and the B200
    # compute capability on another SM count.
    for shape in ((13, 64, 12, 8192), (13, 128, 12, 8192), (13, 128, 12, 10240)):
        plan, routed = _persistent_route(*shape, B200)
        assert routed, shape
        b300_plan, b300_routed = _persistent_route(*shape, B300)
        assert b300_plan == plan and not b300_routed, shape
        for warps in (1, 2, 4):
            other = dataclasses.replace(plan, reduce_warps=warps)
            assert not use_persistent_route(other, *B300, shape[3]), (shape, warps)
        for device in (GB300, ((10, 0), 152)):
            assert not _persistent_route(*shape, device)[1], (shape, device)


def test_persistent_cases_cover_both_paths_on_this_device():
    _skip_unless_sm100_family()
    from flashinfer.mla.cake_kimi_k3_mla import PERSISTENT_MAX_ITEM_KV

    device = _device()
    if device not in PERSISTENT_MAX_ITEM_KV:
        pytest.skip(f"no persistent wide route on {device}")
    routes = [_persistent_route(*case, device)[1] for case in PERSISTENT_CASES]
    assert any(routes) and not all(routes), routes


@pytest.mark.parametrize("batch,q_len,num_heads,kv", PERSISTENT_CASES)
def test_persistent_route_is_bitwise_the_shipped_kernels(
    batch, q_len, num_heads, kv, monkeypatch
):
    """The persistent route keeps the split plan, per-item code and fold order: its output must
    equal the shipped kernels' bit for bit, on eager launches and on CUDA-Graph replays (the
    launcher's flag words carry the launch epoch from one launch to the next)."""
    _skip_unless_sm100_family()
    import flashinfer.mla.cake_kimi_k3_mla as host

    device = torch.device("cuda")
    plan, expected = _persistent_route(batch, q_len, num_heads, kv, _device())
    case = _make_case(
        batch, [q_len] * batch, [kv] * batch, num_heads, seed=645501, device=device
    )
    total_q = case["query"].shape[0]

    def launcher():
        out = torch.full(
            (total_q, num_heads, LATENT),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
        workspace = torch.zeros(
            host.workspace_bytes(plan.rows_max, plan.num_split),
            dtype=torch.uint8,
            device=device,
        )
        attention = host.KimiK3MlaFp8PagedAttention(
            query=case["query"],
            kv_cache=case["kv_cache"],
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            out=out,
            workspace_buffer=workspace,
            bmm1_scale=case["bmm1_scale"],
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=q_len,
            max_seq_len=kv,
        )
        return attention, out

    attention, out = launcher()
    arch = attention.arch
    assert attention.plan == plan
    assert attention.route_metadata["persistent"] == expected
    if expected:
        assert (
            attention.route_metadata["main_module"]
            == get_cake_kimi_k3_mla_kernel("main_wide_persistent", arch=arch)["name"]
        )
        assert attention.route_metadata["reducer"] == f"{plan.reduce_kind}_tile_flags"
    else:
        assert (
            attention.route_metadata["main_module"]
            == get_cake_kimi_k3_mla_kernel(plan.main_kind, arch=arch)["name"]
        )
        assert attention.route_metadata["reducer"] == plan.reduce_kind
    attention.launch()
    torch.cuda.synchronize()
    eager = out.clone()
    _check(eager, _reference(case), atol=1e-2, rtol=2e-2)
    # A second eager launch on the same launcher (the flag epoch advances).
    out.fill_(float("nan"))
    attention.launch()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)
    # CUDA-Graph capture and three replays.
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
        attention.launch()
    torch.cuda.synchronize()
    for _ in range(3):
        out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    # The shipped kernels of the same plan (no persistent route on any device).
    monkeypatch.setattr(host, "PERSISTENT_MAX_ITEM_KV", {})
    shipped, shipped_out = launcher()
    assert not shipped.route_metadata["persistent"]
    assert (
        shipped.route_metadata["main_module"]
        == get_cake_kimi_k3_mla_kernel(plan.main_kind, arch=arch)["name"]
    )
    shipped.launch()
    torch.cuda.synchronize()
    assert torch.equal(shipped_out, eager)


# ---------------------------------------------------------------------------
# LSE output, non-causal masking and cyclic decode context parallelism (DCP)
# ---------------------------------------------------------------------------

DCP_WORLD = 8
# Absolute LSE tolerance (natural log), fixed before any GPU run from an FP64 emulation of these
# kernels' arithmetic (FP32 scores and exp sums, the ex2 / lg2 approximations, E4M3 probabilities,
# BF16 split partials and the reducer merge; the MMA accumulator also rounded to 14 mantissa bits
# per 32-element step) over this file's route cells and their DCP8 shards at the amplitude of its
# data (0.5, where the output tolerance below holds): the smallest of 1e-4, 2e-4, 5e-4 and 1e-3
# at least four times the emulation's largest LSE error (2.0e-4).
LSE_ATOL = 1e-3


def _visible_limits(q_len, global_len, causal=True):
    """Exclusive bound of the global key positions each query token of a request sees."""
    if not causal:
        return [global_len] * q_len
    return [max(global_len - q_len + t + 1, 0) for t in range(q_len)]


def _reference64(case, *, causal=True, global_rows=None):
    """FP64 reference over the dequantized FP8 operands: BF16 output and the natural-log LSE of
    every (token, head) row.  ``global_rows[b]`` (``[G, 576]``) replaces the request's paged rows,
    so one reference serves a sharded run.  A row that sees no key gives O = 0 and LSE = -inf."""
    cache = case["kv_cache"]
    num_heads = case["num_heads"]
    q_rows = case["query"].double()
    device = q_rows.device
    out = torch.zeros(
        (q_rows.shape[0], num_heads, LATENT), dtype=torch.float64, device=device
    )
    lse = torch.full(
        (q_rows.shape[0], num_heads), float("-inf"), dtype=torch.float64, device=device
    )
    q_indptr = case["q_indptr"].tolist()
    for b, q_len in enumerate(case["q_lens"]):
        if global_rows is not None:
            values = global_rows[b].double()
        else:
            kv_len = case["kv_lens"][b]
            n_pages = (kv_len + PAGE - 1) // PAGE
            pages = case["block_tables"][b, :n_pages].long()
            values = cache[pages].reshape(-1, QK_DIM)[:kv_len].double()
        g = values.shape[0]
        q = q_rows[q_indptr[b] : q_indptr[b + 1]].reshape(q_len, num_heads, QK_DIM)
        logits = torch.einsum("thd,kd->thk", q, values) * case["bmm1_scale"]
        limit = torch.tensor(_visible_limits(q_len, g, causal), device=device)
        mask = torch.arange(g, device=device)[None, :] < limit[:, None]
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        row_lse = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(logits - row_lse[..., None]).nan_to_num(0.0)
        out[q_indptr[b] : q_indptr[b + 1]] = probs @ values[:, :LATENT]
        lse[q_indptr[b] : q_indptr[b + 1]] = row_lse
    return (out * case["bmm2_scale"]).to(torch.bfloat16), lse


def _paged_from_rows(rows, *, seed, device):
    """A paged FP8 cache and block table holding ``rows[b]`` (``[n_b, 576]`` FP8) per request,
    with shuffled pages; an empty request keeps one page so its table row stays valid."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pages_per_seq = [max((r.shape[0] + PAGE - 1) // PAGE, 1) for r in rows]
    total = sum(pages_per_seq)
    perm = torch.randperm(total + 4, generator=gen, device=device)[:total]
    cache = torch.zeros((total + 4, PAGE, QK_DIM), dtype=torch.uint8, device=device)
    table = torch.zeros(
        (len(rows), max(pages_per_seq)), dtype=torch.int32, device=device
    )
    off = 0
    for b, (r, n) in enumerate(zip(rows, pages_per_seq, strict=True)):
        table[b, :n] = perm[off : off + n].to(torch.int32)
        # Byte views: index_put is not defined for FP8 tensors.
        flat = torch.zeros((n * PAGE, QK_DIM), dtype=torch.uint8, device=device)
        flat[: r.shape[0]] = r.view(torch.uint8)
        cache[table[b, :n].long()] = flat.reshape(n, PAGE, QK_DIM)
        off += n
    return cache.view(torch.float8_e4m3fn), table


def _global_rows(case):
    rows = []
    for b, kv_len in enumerate(case["kv_lens"]):
        n_pages = (kv_len + PAGE - 1) // PAGE
        pages = case["block_tables"][b, :n_pages].long()
        rows.append(case["kv_cache"][pages].reshape(-1, QK_DIM)[:kv_len])
    return rows


def _launch(
    case, *, kv_cache=None, block_tables=None, seq_lens=None, max_seq_len=None, **ext
):
    """One ``KimiK3MlaFp8PagedAttention`` over ``case`` (packed rows) with the extended options;
    returns ``(launcher, out, lse)`` after an eager launch.  ``out`` starts NaN-filled."""
    from flashinfer.mla.cake_kimi_k3_mla import (
        KimiK3MlaFp8PagedAttention,
        plan_attention,
        workspace_bytes,
    )

    device = case["query"].device
    num_heads = case["num_heads"]
    total_q = case["query"].shape[0]
    block_tables = case["block_tables"] if block_tables is None else block_tables
    max_seq_len = max_seq_len or int(block_tables.shape[1]) * PAGE
    plan = plan_attention(
        batch=len(case["q_lens"]),
        max_q_len=max(case["q_lens"]),
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        sm_count=_sm_count(),
    )
    workspace = torch.zeros(
        workspace_bytes(plan.rows_max, plan.num_split), dtype=torch.uint8, device=device
    )
    out = torch.full(
        (total_q, num_heads, LATENT), float("nan"), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_heads), float("nan"), dtype=torch.float32, device=device
    )
    attention = KimiK3MlaFp8PagedAttention(
        query=case["query"],
        kv_cache=case["kv_cache"] if kv_cache is None else kv_cache,
        block_tables=block_tables,
        seq_lens=case["seq_lens"] if seq_lens is None else seq_lens,
        out=out,
        workspace_buffer=workspace,
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"],
        cum_seq_lens_q=case["q_indptr"],
        max_q_len=max(case["q_lens"]),
        max_seq_len=max_seq_len,
        lse=lse,
        **ext,
    )
    assert attention.plan == plan
    attention.launch()
    torch.cuda.synchronize()
    return attention, out, lse


def _shard(case, rank, world=DCP_WORLD, *, seed):
    """Rank ``rank``'s cyclic shard of ``case``: global position ``p`` lives on rank ``p % world``
    at local index ``p // world``."""
    rows = [r[rank::world] for r in _global_rows(case)]
    cache, table = _paged_from_rows(rows, seed=seed + rank, device=case["query"].device)
    seq_lens = torch.tensor(
        [r.shape[0] for r in rows], dtype=torch.int32, device=cache.device
    )
    return cache, table, seq_lens


def _merge(states, *, base="e"):
    """Merge of per-rank ``(O, lse)`` states with the arithmetic of vLLM's DCP combine: FP32,
    weights ``exp(lse - max)`` in the states' LSE base, normalized by their sum.  Rows no rank sees
    stay O = 0, LSE = -inf."""
    exp, log = (torch.exp, torch.log) if base == "e" else (torch.exp2, torch.log2)
    lses = torch.stack([s[1].float() for s in states])
    outs = torch.stack([s[0].float() for s in states])
    m = lses.max(dim=0).values
    m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    w = exp(lses - m[None])
    total = w.sum(dim=0)
    scale = w / total.clamp_min(torch.finfo(torch.float32).tiny)[None]
    o = (outs * scale[..., None]).sum(dim=0)
    lse = torch.where(total > 0, log(total) + m, float("-inf"))
    return o, lse


def _check_dcp(merged, single, ref):
    """The merged DCP output (cast to BF16) against the FP64 reference: within the output
    tolerance (atol = rtol = 1e-2), or no farther than 1.1 times the single-rank launch's own
    largest error."""
    merged = merged.to(torch.bfloat16).float()
    assert torch.isfinite(merged).all()
    ref = ref.float()
    diff = (merged - ref).abs()
    inside = bool((diff <= 1e-2 + 1e-2 * ref.abs()).all())
    single_err = (single.float() - ref).abs().max().item()
    assert inside or diff.max().item() <= 1.1 * single_err, (
        diff.max().item(),
        single_err,
    )


def _check_lse(lse, ref, *, atol):
    """Finite rows match the FP64 reference; rows without a visible key are exactly -inf."""
    lse = lse.double()
    empty = torch.isneginf(ref)
    assert not torch.isnan(lse).any()
    assert torch.isneginf(lse[empty]).all()
    torch.testing.assert_close(lse[~empty], ref[~empty], atol=atol, rtol=0)


# Visible-key bound of the FP64 output check of the extended programs.  Every program rounds the
# probabilities p = 2^(x - max x + 7.807355) to E4M3 before P.V and divides by the FP32 sum of the
# unrounded p, so a row over few keys keeps the E4M3 rounding of its weights: the shipped default
# programs put rows with 2 to 128 visible keys outside atol = rtol = 1e-2 of the FP64 reference
# (bit for bit the same outputs as the extended programs), and none above 128 (largest error
# 4.4e-3).  The bound is the smallest multiple of the 128-key tile that covers every visible-key
# count at which a default row fails, with every default row above it within half the tolerance.
# Rows at or below it are checked against the default programs bit for bit and against the
# E4M3-probability reference within its rounding bound (_fp8p_reference).
FEW_KEYS = 128
# log2 of the largest stored probability (224 = 1.75 * 2^7 in E4M3)
P_SHIFT = 7.807355


def _row_visibility(
    case, *, seq_lens=None, causal=True, causal_seqs=None, cp_world=1, cp_rank=0
):
    """Visible local keys of every packed query row: local key ``j`` is global position
    ``j * cp_world + cp_rank``, seen by token ``t`` of a ``q_len``-token request of global length
    ``G`` iff it is below ``G - q_len + t + 1`` (causal) or ``G``; always a prefix of the local
    keys."""
    local = (case["seq_lens"] if seq_lens is None else seq_lens).tolist()
    glob = local if causal_seqs is None else causal_seqs.tolist()
    visible = []
    for b, q_len in enumerate(case["q_lens"]):
        for t in range(q_len):
            bound = glob[b] - q_len + t + 1 if causal else glob[b]
            n = 0 if bound <= cp_rank else (bound - cp_rank + cp_world - 1) // cp_world
            visible.append(min(max(n, 0), local[b]))
    return visible


def _local_rows(kv_cache, block_tables, seq_lens):
    rows = []
    for b, n in enumerate(seq_lens.tolist()):
        pages = block_tables[b, : max((n + PAGE - 1) // PAGE, 1)].long()
        rows.append(kv_cache[pages].reshape(-1, QK_DIM)[:n])
    return rows


def _fp8p_reference(query, keys, scale):
    """Output of one query row (``[H, 576]``) over its visible keys (``[v, 576]``) with the
    kernels' probability rounding, in FP64, and the per-element bound of its distance to the
    kernels' BF16 output.  ``O = sum_k E4M3(p_k) V_k / sum_k p_k``.  The bound lets every
    probability whose E4M3 rounding the score error can change round either way: the score
    error is that of an MMA accumulator rounded toward zero to 14 mantissa bits after each
    32-element K step (the pessimistic model of the LSE tolerance study), plus the FP32 scaling
    and the ex2 approximation; it adds the matching relative change of the sum, the P.V
    accumulation under the same model and two BF16 roundings of the output."""
    q, k = query.double(), keys.double()
    parts = torch.einsum("hcd,kcd->hkc", q.reshape(-1, 18, 32), k.reshape(-1, 18, 32))
    log2_scale = scale / math.log(2.0)
    x = parts.sum(-1) * log2_scale
    dx = (
        parts.cumsum(-1).abs().sum(-1) * 2.0**-13 * log2_scale
        + (x.abs() + 2 * P_SHIFT) * 2.0**-22
        + 2.0**-20
    )
    shift = x.max(dim=1, keepdim=True).values
    d = dx + dx.max(dim=1, keepdim=True).values
    p = torch.exp2(x - shift + P_SHIFT)

    def e4m3(t):
        return t.to(torch.float8_e4m3fn).double()

    rounded = e4m3(p)
    spread = e4m3(p * torch.exp2(d)) - e4m3(p * torch.exp2(-d))
    values = k[:, :LATENT]
    total = p.sum(dim=1, keepdim=True)
    out = rounded @ values / total
    magnitude = rounded @ values.abs() / total
    steps = (keys.shape[0] + 31) // 32
    bound = (
        spread @ values.abs() / (total * torch.exp2(-d.max()))
        + out.abs() * (torch.exp2(d.max()) - 1 + 2.0**-8)
        + magnitude * steps * 2.0**-13
    )
    return out, bound


def _default_equivalent(case, out, visible, *, kv_cache, block_tables, seq_lens):
    """Rows of an extended launch with 1 <= v <= FEW_KEYS visible keys against the default
    programs on the same keys, bit for bit.  For token ``t`` one default launch keeps the batch,
    its token counts, heads and table width (hence the plan) and gives each request its first
    ``v`` local keys followed by ``q_len - 1 - t`` copies of its first key, so the default
    causal bound of token ``t`` is exactly those ``v`` keys.  Returns the rows compared."""
    rows = [r.view(torch.uint8) for r in _local_rows(kv_cache, block_tables, seq_lens)]
    width = block_tables.shape[1]
    starts = case["q_indptr"].tolist()
    device = out.device
    compared = 0
    for t in range(max(case["q_lens"])):
        new_rows, pick = [], []
        for b, q_len in enumerate(case["q_lens"]):
            r = starts[b] + t
            v = visible[r] if t < q_len else 0
            pad = q_len - 1 - t
            if 1 <= v <= FEW_KEYS and v + pad <= width * PAGE:
                keys = torch.cat([rows[b][:v], rows[b][:1].expand(pad, -1)])
                pick.append(r)
            else:  # a filler request whose every token sees a key; its rows are not compared
                keys = rows[b] if rows[b].shape[0] >= q_len else None
                if keys is None:
                    keys = torch.zeros(
                        (q_len, QK_DIM), dtype=torch.uint8, device=device
                    )
            new_rows.append(keys.view(torch.float8_e4m3fn))
        if not pick:
            continue
        cache, table = _paged_from_rows(new_rows, seed=7900 + t, device=device)
        table = torch.nn.functional.pad(table, (0, width - table.shape[1]))
        lens = torch.tensor(
            [k.shape[0] for k in new_rows], dtype=torch.int32, device=device
        )
        default = _run(dict(case, kv_cache=cache, block_tables=table, seq_lens=lens))
        index = torch.tensor(pick, device=device)
        assert torch.equal(
            out[index].view(torch.int16), default[index].view(torch.int16)
        ), pick
        compared += len(pick)
    return compared


def _check_ext(
    out,
    ref,
    case,
    *,
    kv_cache=None,
    block_tables=None,
    seq_lens=None,
    causal=True,
    causal_seqs=None,
    cp_world=1,
    cp_rank=0,
    atol=1e-2,
    rtol=1e-2,
):
    """The output of an extended launch, by each row's visible-key count ``v``: above FEW_KEYS
    the FP64 reference within ``atol`` / ``rtol``; from 1 to FEW_KEYS the default programs on the
    same keys bit for bit (_default_equivalent) and the E4M3-probability reference within its
    bound (_fp8p_reference); 0 exactly zero."""
    kv_cache = case["kv_cache"] if kv_cache is None else kv_cache
    block_tables = case["block_tables"] if block_tables is None else block_tables
    seq_lens = case["seq_lens"] if seq_lens is None else seq_lens
    visible = _row_visibility(
        case,
        seq_lens=seq_lens,
        causal=causal,
        causal_seqs=causal_seqs,
        cp_world=cp_world,
        cp_rank=cp_rank,
    )
    assert torch.isfinite(out.float()).all()
    v = torch.tensor(visible, device=out.device)
    many, none = v > FEW_KEYS, v == 0
    torch.testing.assert_close(
        out[many].float(), ref[many].float(), atol=atol, rtol=rtol
    )
    assert torch.equal(out[none].float(), torch.zeros_like(out[none].float()))
    if not bool(((v >= 1) & (v <= FEW_KEYS)).any()):
        return
    rows = _local_rows(kv_cache, block_tables, seq_lens)
    starts = case["q_indptr"].tolist()
    for b, q_len in enumerate(case["q_lens"]):
        for t in range(q_len):
            r = starts[b] + t
            if 1 <= visible[r] <= FEW_KEYS:
                ref8, bound = _fp8p_reference(
                    case["query"][r], rows[b][: visible[r]], case["bmm1_scale"]
                )
                err = (out[r].double() - ref8 * case["bmm2_scale"]).abs()
                excess = err - bound * case["bmm2_scale"]
                assert bool((excess <= 0).all()), (r, visible[r], excess.max().item())
    _default_equivalent(
        case,
        out,
        visible,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )


# The (num_heads, q_len) cells of the route census: H12 q1 -> rt16, H12 q2 / q4 / q5 -> rt32 /
# rt48 / rt64 (the multi-token verify of 12 heads; eight 2-token requests take the late-trigger
# tile), H12 q6 -> rt96, H16 q1 / H48 q1 at 65536 keys -> the 16- / 48-row late-trigger tiles with
# the CTA reducers (prefetching / plain), H96 q1 -> rt96 / wide, H96 q6 -> rt96 (6 m-tiles) /
# wide; KV lengths straddle the wide threshold (8192) and include split plans.
EXT_CELLS = [
    (12, 1, [500, 2100, 64]),
    (12, 2, [500, 2100]),
    (12, 2, [1800] * 8),
    (12, 4, [700, 3000]),
    (12, 5, [900, 1500]),
    (12, 6, [700, 9000]),
    (16, 1, [65536]),
    (48, 1, [65536]),
    (96, 1, [300, 9000]),
    (96, 6, [1500, 12000]),
]


@pytest.mark.parametrize("num_heads,q_len,kv_lens", EXT_CELLS)
@pytest.mark.parametrize("lse_base", ["e", "2"])
def test_lse_matches_fp64_reference(num_heads, q_len, kv_lens, lse_base):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(
        batch, [q_len] * batch, kv_lens, num_heads, seed=7809, device=device
    )
    attention, out, lse = _launch(case, lse_base=lse_base)
    assert attention.route_metadata["ext"]
    ref_out, ref_lse = _reference64(case)
    _check_ext(out, ref_out, case)
    scale = 1.0 if lse_base == "e" else 1.0 / math.log(2.0)
    _check_lse(lse, ref_lse * scale, atol=LSE_ATOL * scale)


@pytest.mark.parametrize("num_heads,q_len,kv_lens", EXT_CELLS)
def test_lse_does_not_change_the_output(num_heads, q_len, kv_lens):
    """The LSE sibling keeps the default program's output bit for bit."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(
        batch, [q_len] * batch, kv_lens, num_heads, seed=7810, device=device
    )
    _, out, _ = _launch(case)
    default = _run(case)
    assert torch.equal(out, default)


@pytest.mark.parametrize("num_heads,q_len,kv_lens", EXT_CELLS)
def test_non_causal(num_heads, q_len, kv_lens):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(
        batch, [q_len] * batch, kv_lens, num_heads, seed=7811, device=device
    )
    _, out, lse = _launch(case, causal=False)
    ref_out, ref_lse = _reference64(case, causal=False)
    _check_ext(out, ref_out, case, causal=False)
    _check_lse(lse, ref_lse, atol=LSE_ATOL)


@pytest.mark.parametrize(
    "num_heads,q_len,kv_lens", [cell for cell in EXT_CELLS if cell[1] == 1]
)
def test_q1_non_causal_is_causal(num_heads, q_len, kv_lens):
    """A one-token request sees every key either way: the non-causal launch equals the causal
    one bit for bit."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(
        batch, [q_len] * batch, kv_lens, num_heads, seed=7818, device=device
    )
    _, out, lse = _launch(case)
    _, out_nc, lse_nc = _launch(case, causal=False)
    assert torch.equal(out.view(torch.int16), out_nc.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), lse_nc.view(torch.int32))


# Global lengths of every residue modulo the DCP world, mid-page, on the row tiles and on the
# wide route, and lengths shorter than the world (ranks without a key).
DCP_RESIDUE_LENS = [
    [1000 + r for r in range(DCP_WORLD)],
    [9000 + r for r in range(DCP_WORLD)],
    [3, 5, 7],
]


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("q_len", [1, 6])
@pytest.mark.parametrize("kv_lens", DCP_RESIDUE_LENS)
def test_dcp8_every_residue(kv_lens, q_len, causal):
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(batch, [q_len] * batch, kv_lens, 12, seed=7819, device=device)
    global_lens = case["seq_lens"].clone()
    _, single_out, single_lse = _launch(case, causal=causal)
    states = []
    for rank in range(DCP_WORLD):
        cache, table, local_lens = _shard(case, rank, seed=7819)
        _, out, lse = _launch(
            case,
            kv_cache=cache,
            block_tables=table,
            seq_lens=local_lens,
            causal=causal,
            causal_seqs=global_lens,
            cp_world=DCP_WORLD,
            cp_rank=rank,
        )
        assert torch.isfinite(out.float()).all()
        assert not torch.isnan(lse).any()
        states.append((out, lse))
    merged_out, merged_lse = _merge(states)
    ref_out, ref_lse = _reference64(case, causal=causal)
    _check_ext(single_out, ref_out, case, causal=causal)
    _check_lse(single_lse, ref_lse, atol=LSE_ATOL)
    _check_dcp(merged_out, single_out, ref_out)
    _check_lse(merged_lse, ref_lse, atol=LSE_ATOL)


@pytest.mark.parametrize("batch,q_len,num_heads,kv", PERSISTENT_CASES)
def test_persistent_route_ext(batch, q_len, num_heads, kv):
    """The extended programs keep the default route decision (the persistent wide decode and
    its flag-waiting reducer where the default takes them), keep its output bit for bit and
    store the LSE."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    plan, expected = _persistent_route(batch, q_len, num_heads, kv, _device())
    case = _make_case(
        batch, [q_len] * batch, [kv] * batch, num_heads, seed=645502, device=device
    )
    attention, out, lse = _launch(case)
    arch = attention.arch
    assert attention.route_metadata["ext"]
    assert attention.route_metadata["persistent"] == expected
    main_kind = "main_wide_persistent" if expected else plan.main_kind
    assert (
        attention.route_metadata["main_module"]
        == get_cake_kimi_k3_mla_kernel(f"{main_kind}_ext", arch=arch)["name"]
    )
    assert torch.equal(out.view(torch.int16), _run(case).view(torch.int16))
    ref_out, ref_lse = _reference64(case)
    _check_ext(out, ref_out, case, rtol=2e-2)
    _check_lse(lse, ref_lse, atol=LSE_ATOL)


@pytest.mark.parametrize("num_heads,q_len,kv_lens", EXT_CELLS)
def test_dcp8_merge_matches_single_rank(num_heads, q_len, kv_lens):
    """Eight cyclic shards, each run with its rank and the global lengths, merge to the
    single-rank result and to the FP64 reference."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(
        batch, [q_len] * batch, kv_lens, num_heads, seed=7812, device=device
    )
    global_lens = case["seq_lens"].clone()
    _, single_out, single_lse = _launch(case)
    states = []
    for rank in range(DCP_WORLD):
        cache, table, local_lens = _shard(case, rank, seed=7812)
        _, out, lse = _launch(
            case,
            kv_cache=cache,
            block_tables=table,
            seq_lens=local_lens,
            causal_seqs=global_lens,
            cp_world=DCP_WORLD,
            cp_rank=rank,
        )
        assert torch.isfinite(out.float()).all()
        assert not torch.isnan(lse).any()
        states.append((out, lse))
    merged_out, merged_lse = _merge(states)
    ref_out, ref_lse = _reference64(case)
    _check_ext(single_out, ref_out, case)
    _check_dcp(merged_out, single_out, ref_out)
    _check_lse(merged_lse, ref_lse, atol=LSE_ATOL)
    # Both are within LSE_ATOL of the reference.
    _check_lse(merged_lse, single_lse.double(), atol=2 * LSE_ATOL)


def test_rows_without_visible_keys_are_finite():
    """A prefill row whose causal bound is 0 (more query tokens than keys), and a DCP shard whose
    keys all lie beyond a row's bound, give O = 0 (never NaN) and LSE = -inf."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    for num_heads in (12, 96):
        # Row tiles (longest KV 64) and the wide route (longest KV 9000).
        for kv_lens in ([3, 64], [3, 9000]):
            case = _make_case(2, [6, 6], kv_lens, num_heads, seed=7813, device=device)
            attention, out, lse = _launch(case)
            assert attention.plan.wide == (kv_lens[1] > 8192)
            ref_out, ref_lse = _reference64(case)
            assert torch.isneginf(ref_lse[:3]).all()
            _check_ext(out, ref_out, case)
            assert torch.equal(out[:3].float(), torch.zeros_like(out[:3].float()))
            _check_lse(lse, ref_lse, atol=LSE_ATOL)
        # Global length 5 on 8 ranks: ranks 5..7 hold no key, ranks 0..4 one key each.
        case = _make_case(1, [1], [5], num_heads, seed=7814, device=device)
        for rank in range(DCP_WORLD):
            cache, table, local_lens = _shard(case, rank, seed=7814)
            _, out, lse = _launch(
                case,
                kv_cache=cache,
                block_tables=table,
                seq_lens=local_lens,
                causal_seqs=case["seq_lens"],
                cp_world=DCP_WORLD,
                cp_rank=rank,
            )
            assert torch.isfinite(out.float()).all()
            if rank >= 5:
                assert torch.equal(out.float(), torch.zeros_like(out.float()))
                assert torch.isneginf(lse).all()


@pytest.mark.parametrize("num_heads,q_len,kv_lens", [EXT_CELLS[0], EXT_CELLS[-1]])
def test_cuda_graph_replay_with_changed_lengths(num_heads, q_len, kv_lens):
    """A captured DCP launch replays with new ``seq_lens`` / ``causal_seqs`` contents (same
    table width) and, for multi-token requests, new ``cum_seq_lens_q`` contents (same token
    count and longest request): the replay follows them and equals an eager launch on the same
    contents bit for bit."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    rank = 3
    q_lens = [q_len - 2 * (b % 2) if q_len > 2 else q_len for b in range(batch)]
    case = _make_case(batch, q_lens, kv_lens, num_heads, seed=7815, device=device)
    global_lens = case["seq_lens"].clone()
    cache, table, local_lens = _shard(case, rank, seed=7815)
    attention, out, lse = _launch(
        case,
        kv_cache=cache,
        block_tables=table,
        seq_lens=local_lens,
        causal_seqs=global_lens,
        cp_world=DCP_WORLD,
        cp_rank=rank,
    )
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(g, stream=stream):
        attention.launch()
    torch.cuda.synchronize()
    # Shorter global lengths: the shard keeps its rows, the new bounds hide the tail.
    new_global = [max(k // 2 - 7, 1) for k in kv_lens]
    global_lens.copy_(torch.tensor(new_global, dtype=torch.int32, device=device))
    local_lens.copy_(
        torch.tensor(
            [len(range(rank, k, DCP_WORLD)) for k in new_global],
            dtype=torch.int32,
            device=device,
        )
    )
    # The same tokens in requests of swapped lengths.
    new_q_lens = q_lens[::-1]
    case["q_indptr"].copy_(
        torch.tensor(
            [0] + list(torch.tensor(new_q_lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
    )
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    replayed = (out.clone(), lse.clone())
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    attention.launch()
    torch.cuda.synchronize()
    assert torch.equal(replayed[0].view(torch.int16), out.view(torch.int16))
    assert torch.equal(replayed[1].view(torch.int32), lse.view(torch.int32))
    rows = [
        r[rank::DCP_WORLD][: int(n)]
        for r, n in zip(_global_rows(case), local_lens.tolist(), strict=True)
    ]
    # The reference of the shard: its local keys under the global bound of each token.
    shard_case = dict(
        case, q_lens=new_q_lens, kv_lens=[int(n) for n in local_lens.tolist()]
    )
    ref_out, ref_lse = _reference64_shard(shard_case, rows, new_global, rank)
    _check_ext(
        out,
        ref_out,
        shard_case,
        kv_cache=cache,
        block_tables=table,
        seq_lens=local_lens,
        causal_seqs=global_lens,
        cp_world=DCP_WORLD,
        cp_rank=rank,
    )
    _check_lse(lse, ref_lse, atol=LSE_ATOL)


def _reference64_shard(case, rows, global_lens, rank, world=DCP_WORLD):
    """FP64 reference of one rank's state: local key ``j`` is global position ``j * world + rank``
    and is visible to token ``t`` iff that position is below ``G - q_len + t + 1``."""
    num_heads = case["num_heads"]
    q_rows = case["query"].double()
    device = q_rows.device
    out = torch.zeros(
        (q_rows.shape[0], num_heads, LATENT), dtype=torch.float64, device=device
    )
    lse = torch.full(
        (q_rows.shape[0], num_heads), float("-inf"), dtype=torch.float64, device=device
    )
    q_indptr = case["q_indptr"].tolist()
    for b, q_len in enumerate(case["q_lens"]):
        values = rows[b].double()
        n = values.shape[0]
        q = q_rows[q_indptr[b] : q_indptr[b + 1]].reshape(q_len, num_heads, QK_DIM)
        logits = torch.einsum("thd,kd->thk", q, values) * case["bmm1_scale"]
        pos = torch.arange(n, device=device) * world + rank
        limit = torch.tensor(_visible_limits(q_len, global_lens[b]), device=device)
        mask = pos[None, :] < limit[:, None]
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        row_lse = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(logits - row_lse[..., None]).nan_to_num(0.0)
        out[q_indptr[b] : q_indptr[b + 1]] = probs @ values[:, :LATENT]
        lse[q_indptr[b] : q_indptr[b + 1]] = row_lse
    return (out * case["bmm2_scale"]).to(torch.bfloat16), lse


@pytest.mark.parametrize("num_heads,q_len,kv_lens", EXT_CELLS)
def test_workspace_bytes_unchanged_by_the_options(num_heads, q_len, kv_lens):
    """LSE, non-causal and DCP keep the plan, so the default ``workspace_bytes`` suffices."""
    _skip_unless_sm100_family()
    device = torch.device("cuda")
    batch = len(kv_lens)
    case = _make_case(
        batch, [q_len] * batch, kv_lens, num_heads, seed=7816, device=device
    )
    for ext in (
        {},
        {"causal": False},
        {"causal_seqs": case["seq_lens"].clone(), "cp_world": DCP_WORLD, "cp_rank": 1},
    ):
        attention, _, _ = _launch(
            case, **ext
        )  # asserts the default plan and fits its workspace
        assert attention.route_metadata["ext"]


def test_trtllm_entry_returns_lse_and_runs_dcp():
    """``trtllm_batch_decode_with_kv_cache_mla(backend="cake")`` plumbs ``return_lse``,
    ``return_lse_base`` and ``enable_dcp`` to the extended programs."""
    _skip_unless_sm100_family()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.mla.cake_kimi_k3_mla import workspace_bytes

    device = torch.device("cuda")
    case = _make_case(2, [1, 1], [700, 2100], 12, seed=7817, device=device)
    workspace = torch.zeros(workspace_bytes(24, 256), dtype=torch.uint8, device=device)
    common = dict(
        workspace_buffer=workspace,
        qk_nope_head_dim=128,
        kv_lora_rank=LATENT,
        qk_rope_head_dim=ROPE,
        bmm1_scale=case["bmm1_scale"],
        bmm2_scale=case["bmm2_scale"],
        backend="cake",
        return_lse=True,
    )
    query = case["query"].reshape(2, 1, 12, QK_DIM)
    out, lse = trtllm_batch_decode_with_kv_cache_mla(
        query,
        kv_cache=case["kv_cache"],
        block_tables=case["block_tables"],
        seq_lens=case["seq_lens"],
        max_seq_len=int(case["block_tables"].shape[1]) * PAGE,
        return_lse_base="base2",
        **common,
    )
    ref_out, ref_lse = _reference64(case)
    _check(out.reshape(-1, 12, LATENT), ref_out, atol=1e-2, rtol=1e-2)
    _check_lse(
        lse.reshape(-1, 12), ref_lse / math.log(2.0), atol=LSE_ATOL / math.log(2.0)
    )
    states = []
    for rank in range(DCP_WORLD):
        cache, table, local_lens = _shard(case, rank, seed=7817)
        states.append(
            tuple(
                t.reshape(-1, 12, *t.shape[3:])
                for t in trtllm_batch_decode_with_kv_cache_mla(
                    query,
                    kv_cache=cache,
                    block_tables=table,
                    seq_lens=local_lens,
                    max_seq_len=int(table.shape[1]) * PAGE,
                    enable_dcp=True,
                    cp_world=DCP_WORLD,
                    cp_rank=rank,
                    causal_seqlens_kv_global=case["seq_lens"],
                    **common,
                )
            )
        )
    merged_out, merged_lse = _merge(states)
    _check_dcp(merged_out, out.reshape(-1, 12, LATENT), ref_out)
    _check_lse(merged_lse, ref_lse, atol=LSE_ATOL)


def test_ext_option_validation():
    """Construction-time refusals of the extended options (CPU only)."""
    from flashinfer.mla.cake_kimi_k3_mla import _ext_values

    cpu = torch.device("cpu")
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    kw = dict(
        lse=None,
        lse_base="e",
        causal=True,
        causal_seqs=None,
        cp_world=1,
        cp_rank=0,
        cp_interleave=1,
        rows=24,
        batch=2,
        seq_lens=seq_lens,
        device=cpu,
    )
    assert _ext_values(**kw) is None
    values = _ext_values(**dict(kw, causal=False))
    assert values["causal"] == 0 and values["causal_seqs"] is seq_lens
    assert values["lse"].numel() >= 24
    g = torch.tensor([80, 160], dtype=torch.int32)
    values = _ext_values(**dict(kw, causal_seqs=g, cp_world=8, cp_rank=5, lse_base="2"))
    assert (
        values["cp_shift"],
        values["cp_ishift"],
        values["cp_rank_off"],
        values["cp_span"],
    ) == (3, 0, 5, 1)
    assert values["causal_seqs"] is g and values["lse_scale"] == 1.0
    lse = torch.empty((2, 12), dtype=torch.float32)
    values = _ext_values(**dict(kw, lse=lse))
    assert values["lse"].data_ptr() == lse.data_ptr()
    assert values["lse_scale"] == pytest.approx(math.log(2.0))
    with pytest.raises(NotImplementedError):
        _ext_values(**dict(kw, causal_seqs=g, cp_world=8, cp_interleave=2))
    for bad, exc in (
        (dict(lse_base="10"), ValueError),
        (dict(causal=1), TypeError),
        (dict(cp_world=3, causal_seqs=g), ValueError),
        (dict(cp_world=8, cp_rank=8, causal_seqs=g), ValueError),
        (dict(cp_world=8), ValueError),
        (dict(causal_seqs=g), ValueError),
        (dict(cp_world=8, causal_seqs=g.to(torch.int64)), ValueError),
        (dict(cp_world=8, causal_seqs=g[:1]), ValueError),
        (dict(cp_interleave=3), ValueError),
        (dict(lse=torch.empty(24, dtype=torch.float16)), ValueError),
        (dict(lse=torch.empty(23, dtype=torch.float32)), ValueError),
        (dict(lse=torch.empty((12, 4), dtype=torch.float32)[:, :2]), ValueError),
    ):
        with pytest.raises(exc):
            _ext_values(**dict(kw, **bad))


def test_persistent_route_reads_the_device_compute_capability(monkeypatch):
    """The launcher keys the route on the device's compute capability as well as its SM count: on
    a device with the persistent route, the same plan under the B300 compute capability (same SM
    count, same programs) keeps the shipped kernels."""
    _skip_unless_sm100_family()
    import flashinfer.mla.cake_kimi_k3_mla as host

    device = torch.device("cuda")
    capability, sm_count = _device()
    if (capability, sm_count) not in host.PERSISTENT_MAX_ITEM_KV:
        pytest.skip(f"no persistent wide route on {(capability, sm_count)}")
    batch, q_len, num_heads, kv = PERSISTENT_CASES[0]
    plan, expected = _persistent_route(
        batch, q_len, num_heads, kv, (capability, sm_count)
    )
    assert expected
    case = _make_case(
        batch, [q_len] * batch, [kv] * batch, num_heads, seed=645502, device=device
    )
    total_q = case["query"].shape[0]
    arch = host._device_facts(host._device_index(device))[0]

    def route_metadata():
        attention = host.KimiK3MlaFp8PagedAttention(
            query=case["query"],
            kv_cache=case["kv_cache"],
            block_tables=case["block_tables"],
            seq_lens=case["seq_lens"],
            out=torch.empty(
                (total_q, num_heads, LATENT), dtype=torch.bfloat16, device=device
            ),
            workspace_buffer=torch.zeros(
                host.workspace_bytes(plan.rows_max, plan.num_split),
                dtype=torch.uint8,
                device=device,
            ),
            bmm1_scale=case["bmm1_scale"],
            cum_seq_lens_q=case["q_indptr"],
            max_q_len=q_len,
            max_seq_len=kv,
        )
        assert attention.plan == plan
        return attention.route_metadata

    assert route_metadata()["persistent"]
    monkeypatch.setattr(host, "_device_facts", lambda index: (arch, B300[0], sm_count))
    shipped = route_metadata()
    assert not shipped["persistent"]
    assert (
        shipped["main_module"]
        == get_cake_kimi_k3_mla_kernel(plan.main_kind, arch=arch)["name"]
    )
    assert shipped["reducer"] == plan.reduce_kind
