import dataclasses
import random
from typing import Literal

import pytest
import torch

from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
from flashinfer.utils import get_compute_capability


WORKSPACE_SIZE = 128 * 1024 * 1024
DSV4_HEAD_DIM = 512
DSV4_SWA_TOPK = 128
TEST_SEED_BASE = 2026

DECODE_BATCH_SIZE = 3
DECODE_Q_LEN = 5
DECODE_SEQ_LEN_CASES = (
    (512, 1024, 128),
    (1024, 2048, 256),
)

PREFILL_BATCH_SIZE = 2
PREFILL_Q_LEN = 257
PREFILL_SEQ_LENS = (4096, 16 * 1024)

SWA_PAGE_SIZE = 256
C4_PAGE_SIZE = 64
C128_PAGE_SIZE = 2

QUERY_RANDOM_SCALE = 0.05
KV_RANDOM_SCALE = 0.05
SINK_RANDOM_SCALE = 0.05
SWA_KV_OFFSET = -0.20
COMPRESSED_KV_OFFSET = 0.25
TOPK_BATCH_VARIATION_DIVISOR = 16
TOPK_PER_QUERY_DECAY = 1

KVLayout = Literal["HND", "NHD"]
SparseCase = Literal["swa128", "swa128+topk4x", "swa128+topk128x"]
TopkLengthMode = Literal["none", "profile", "seqlen"]


@dataclasses.dataclass
class ExtraTestParamForDecode:
    b: int
    is_varlen: bool
    have_zero_seqlen_k: bool
    extra_s_k: int | None = None
    extra_topk: int | None = None
    block_size: int = SWA_PAGE_SIZE
    extra_block_size: int | None = None
    have_extra_topk_length: bool = False


@dataclasses.dataclass
class TestParam:
    __test__ = False

    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = DSV4_HEAD_DIM
    d_v: int = DSV4_HEAD_DIM
    seed: int = -1
    check_correctness: bool = True
    is_all_indices_invalid: bool = False
    num_runs: int = 0
    have_attn_sink: bool = True
    have_topk_length: bool = False
    decode: ExtraTestParamForDecode | None = None
    dtype: torch.dtype = torch.bfloat16
    kv_layout: KVLayout = "HND"
    sparse_case: SparseCase = "swa128"

    @property
    def id(self) -> str:
        assert self.decode is not None
        dtype_name = "bf16" if self.dtype == torch.bfloat16 else "fp8"
        q_mode = "varq" if self.decode.is_varlen else "denseq"
        return (
            f"{self.sparse_case}-{q_mode}-b{self.decode.b}-h{self.h_q}-q{self.s_q}-"
            f"kv{self.s_kv}-{dtype_name}-{self.kv_layout}"
        )


@dataclasses.dataclass
class RawTestParamForDecode:
    """FlashMLA-style flattened decode testcase parameters.

    The dtype and kv_layout fields are FlashInfer-only extensions. DSv4
    FlashInfer sparse MLA supports BF16 and per-tensor FP8 E4M3, so these tests
    intentionally do not use FlashMLA's block-wise FP8 cache recipe.
    """

    b: int
    h_q: int
    s_q: int
    h_kv: int
    s_kv: int
    is_varlen: bool
    topk: int
    is_all_indices_invalid: bool = False
    have_zero_seqlen_k: bool = False
    have_topk_length: bool = False
    enable_attn_sink: bool = True
    extra_s_k: int | None = None
    extra_topk: int | None = None
    block_size: int = SWA_PAGE_SIZE
    extra_block_size: int | None = None
    have_extra_topk_length: bool = False
    d_qk: int = DSV4_HEAD_DIM
    d_v: int = DSV4_HEAD_DIM
    check_correctness: bool = True
    num_runs: int = 0
    seed: int = -1
    dtype: torch.dtype = torch.bfloat16
    kv_layout: KVLayout = "HND"
    sparse_case: SparseCase = "swa128"

    def to_test_param(self) -> TestParam:
        return TestParam(
            self.s_q,
            self.s_kv,
            self.topk,
            self.h_q,
            self.h_kv,
            self.d_qk,
            self.d_v,
            self.seed,
            self.check_correctness,
            self.is_all_indices_invalid,
            self.num_runs,
            self.enable_attn_sink,
            self.have_topk_length,
            decode=ExtraTestParamForDecode(
                self.b,
                self.is_varlen,
                self.have_zero_seqlen_k,
                self.extra_s_k,
                self.extra_topk,
                self.block_size,
                self.extra_block_size,
                self.have_extra_topk_length,
            ),
            dtype=self.dtype,
            kv_layout=self.kv_layout,
            sparse_case=self.sparse_case,
        )


@dataclasses.dataclass
class KVScope:
    t: TestParam
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor
    blocked_k: torch.Tensor
    abs_indices: torch.Tensor
    indices_in_kvcache: torch.Tensor
    topk_length: torch.Tensor | None

    def get_kvcache_for_flashinfer(self, kv_layout: KVLayout) -> torch.Tensor:
        if kv_layout == "HND":
            return self.blocked_k.transpose(1, 2).contiguous()
        return self.blocked_k.contiguous()


@dataclasses.dataclass
class TestcaseForDecode:
    __test__ = False

    p: TestParam
    q: torch.Tensor
    attn_sink: torch.Tensor | None
    sm_scale: float
    kv_scope: KVScope
    extra_kv_scope: KVScope | None
    q_lens: torch.Tensor
    valid_q: torch.Tensor
    workspace_buffer: torch.Tensor


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _c4_topk(h_q: int) -> int:
    return 512 if h_q == 64 else 1024


def gen_testcase() -> tuple[RawTestParamForDecode, ...]:
    cases: list[RawTestParamForDecode] = []
    seed = TEST_SEED_BASE

    def add_case(**kwargs) -> None:
        nonlocal seed
        cases.append(RawTestParamForDecode(seed=seed, **kwargs))
        seed += 1

    for h_q in (64, 128):
        c4_topk = _c4_topk(h_q)
        for swa_seq_len, c4_seq_len, c128_seq_len in DECODE_SEQ_LEN_CASES:
            c128_topk = _round_up(
                c128_seq_len + (DECODE_BATCH_SIZE - 1) * C128_PAGE_SIZE, 4
            )
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for kv_layout in ("HND", "NHD"):
                    add_case(
                        b=DECODE_BATCH_SIZE,
                        h_q=h_q,
                        s_q=DECODE_Q_LEN,
                        h_kv=1,
                        s_kv=swa_seq_len,
                        is_varlen=True,
                        topk=DSV4_SWA_TOPK,
                        block_size=SWA_PAGE_SIZE,
                        dtype=dtype,
                        kv_layout=kv_layout,
                        sparse_case="swa128",
                    )
                    add_case(
                        b=DECODE_BATCH_SIZE,
                        h_q=h_q,
                        s_q=DECODE_Q_LEN,
                        h_kv=1,
                        s_kv=swa_seq_len,
                        is_varlen=True,
                        topk=DSV4_SWA_TOPK,
                        extra_s_k=c4_seq_len,
                        extra_topk=c4_topk,
                        block_size=SWA_PAGE_SIZE,
                        extra_block_size=C4_PAGE_SIZE,
                        have_extra_topk_length=True,
                        dtype=dtype,
                        kv_layout=kv_layout,
                        sparse_case="swa128+topk4x",
                    )
                    add_case(
                        b=DECODE_BATCH_SIZE,
                        h_q=h_q,
                        s_q=DECODE_Q_LEN,
                        h_kv=1,
                        s_kv=swa_seq_len,
                        is_varlen=True,
                        topk=DSV4_SWA_TOPK,
                        extra_s_k=c128_seq_len,
                        extra_topk=c128_topk,
                        block_size=SWA_PAGE_SIZE,
                        extra_block_size=C128_PAGE_SIZE,
                        have_extra_topk_length=True,
                        dtype=dtype,
                        kv_layout=kv_layout,
                        sparse_case="swa128+topk128x",
                    )

    # Smaller head counts use the sparse MLA small-head kernel selection path.
    swa_seq_len, c4_seq_len, c128_seq_len = DECODE_SEQ_LEN_CASES[0]
    c128_topk = _round_up(c128_seq_len + (DECODE_BATCH_SIZE - 1) * C128_PAGE_SIZE, 4)
    for h_q in (8, 16, 32):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for kv_layout in ("HND", "NHD"):
                add_case(
                    b=DECODE_BATCH_SIZE,
                    h_q=h_q,
                    s_q=DECODE_Q_LEN,
                    h_kv=1,
                    s_kv=swa_seq_len,
                    is_varlen=True,
                    topk=DSV4_SWA_TOPK,
                    block_size=SWA_PAGE_SIZE,
                    dtype=dtype,
                    kv_layout=kv_layout,
                    sparse_case="swa128",
                )
                add_case(
                    b=DECODE_BATCH_SIZE,
                    h_q=h_q,
                    s_q=DECODE_Q_LEN,
                    h_kv=1,
                    s_kv=swa_seq_len,
                    is_varlen=True,
                    topk=DSV4_SWA_TOPK,
                    extra_s_k=c4_seq_len,
                    extra_topk=h_q * 8,
                    block_size=SWA_PAGE_SIZE,
                    extra_block_size=C4_PAGE_SIZE,
                    have_extra_topk_length=True,
                    dtype=dtype,
                    kv_layout=kv_layout,
                    sparse_case="swa128+topk4x",
                )
                add_case(
                    b=DECODE_BATCH_SIZE,
                    h_q=h_q,
                    s_q=DECODE_Q_LEN,
                    h_kv=1,
                    s_kv=swa_seq_len,
                    is_varlen=True,
                    topk=DSV4_SWA_TOPK,
                    extra_s_k=c128_seq_len,
                    extra_topk=c128_topk,
                    block_size=SWA_PAGE_SIZE,
                    extra_block_size=C128_PAGE_SIZE,
                    have_extra_topk_length=True,
                    dtype=dtype,
                    kv_layout=kv_layout,
                    sparse_case="swa128+topk128x",
                )

    # Guard seq_lens-driven SWA masking. With q_len == kv_len == 128, early
    # query tokens have fewer than 128 valid SWA entries, so the kernel must use
    # real seq_lens instead of treating every SWA tile as full.
    add_case(
        b=1,
        h_q=64,
        s_q=DSV4_SWA_TOPK,
        h_kv=1,
        s_kv=DSV4_SWA_TOPK,
        is_varlen=True,
        topk=DSV4_SWA_TOPK,
        block_size=SWA_PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_layout="HND",
        sparse_case="swa128",
    )
    add_case(
        b=2,
        h_q=64,
        s_q=DECODE_Q_LEN,
        h_kv=1,
        s_kv=DECODE_SEQ_LEN_CASES[0][0],
        is_varlen=False,
        topk=DSV4_SWA_TOPK,
        extra_s_k=DECODE_SEQ_LEN_CASES[0][1],
        extra_topk=_c4_topk(64),
        block_size=SWA_PAGE_SIZE,
        extra_block_size=C4_PAGE_SIZE,
        have_extra_topk_length=True,
        dtype=torch.bfloat16,
        kv_layout="HND",
        sparse_case="swa128+topk4x",
    )

    # The DSv4 decode kernels are also used for prefill-style varlen Q. These
    # mirror the FlashMLA MODEL1 production rows, with larger Q and 16K KV.
    for prefill_seq_len in PREFILL_SEQ_LENS:
        for h_q in (64, 128):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                add_case(
                    b=PREFILL_BATCH_SIZE,
                    h_q=h_q,
                    s_q=PREFILL_Q_LEN,
                    h_kv=1,
                    s_kv=prefill_seq_len,
                    is_varlen=True,
                    topk=DSV4_SWA_TOPK,
                    extra_s_k=prefill_seq_len,
                    extra_topk=_c4_topk(h_q),
                    block_size=SWA_PAGE_SIZE,
                    extra_block_size=C4_PAGE_SIZE,
                    have_extra_topk_length=True,
                    dtype=dtype,
                    kv_layout="HND",
                    sparse_case="swa128+topk4x",
                )

    return tuple(cases)


TESTCASES = tuple(t.to_test_param() for t in gen_testcase())


def _make_cum_seq_lens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=lengths.device),
            lengths.cumsum(0, dtype=torch.int32),
        )
    )


def _make_q_lens(t: TestParam) -> torch.Tensor:
    assert t.decode is not None
    if not t.decode.is_varlen:
        return torch.full((t.decode.b,), t.s_q, dtype=torch.int32, device="cuda:0")
    min_len = max(1, (t.s_q + 1) // 2)
    return (
        torch.linspace(min_len, t.s_q, t.decode.b, dtype=torch.float32, device="cuda:0")
        .round()
        .to(torch.int32)
    )


def _num_pages(seq_lens: torch.Tensor, page_size: int) -> torch.Tensor:
    return torch.div(seq_lens + page_size - 1, page_size, rounding_mode="floor")


def _make_block_table(cache_seqlens: torch.Tensor, block_size: int) -> torch.Tensor:
    batch_size = cache_seqlens.numel()
    max_pages = int(_num_pages(cache_seqlens, block_size).max().item())
    block_table = torch.arange(
        batch_size * max_pages, dtype=torch.int32, device="cuda:0"
    ).view(batch_size, max_pages)
    return block_table.view(-1)[
        torch.randperm(block_table.numel(), device="cuda:0")
    ].view(batch_size, max_pages)


def _abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    indices = abs_indices.clone()
    valid = indices >= 0
    safe = indices.clamp_min(0)
    block_idx = torch.div(safe, block_size, rounding_mode="floor")
    block_offset = safe % block_size
    batch_ids = torch.arange(indices.shape[0], device=indices.device).view(-1, 1, 1)
    physical_blocks = block_table[batch_ids, block_idx]
    indices = physical_blocks * block_size + block_offset
    indices[~valid] = -1
    return indices.to(torch.int32)


def _make_kv_cache(
    num_blocks: int,
    block_size: int,
    h_kv: int,
    dtype: torch.dtype,
    value_offset: float,
) -> torch.Tensor:
    cache = torch.randn(
        (num_blocks, block_size, h_kv, DSV4_HEAD_DIM),
        dtype=torch.float32,
        device="cuda:0",
    )
    return cache.mul_(KV_RANDOM_SCALE).add_(value_offset).clamp_(-1.0, 1.0).to(dtype)


def _make_causal_abs_indices(
    cache_seqlens: torch.Tensor,
    q_lens: torch.Tensor,
    topk: int,
    max_q_len: int,
) -> torch.Tensor:
    indices = torch.empty(
        (cache_seqlens.numel(), max_q_len, topk),
        dtype=torch.int32,
        device="cuda:0",
    )
    for batch_idx in range(cache_seqlens.numel()):
        seq_len = int(cache_seqlens[batch_idx].item())
        q_len = int(q_lens[batch_idx].item())
        for q_idx in range(max_q_len):
            token_idx = seq_len - q_len + min(q_idx, q_len - 1)
            num_valid = min(topk, token_idx + 1)
            start = token_idx - num_valid + 1
            logical = torch.arange(
                start, start + num_valid, dtype=torch.int32, device="cuda:0"
            )
            indices[batch_idx, q_idx].fill_(-1)
            indices[batch_idx, q_idx, :num_valid] = logical
    return indices


def _randperm_batch(
    perm_range: torch.Tensor,
    perm_size: int,
    padding: int,
) -> torch.Tensor:
    batch_size = perm_range.numel()
    perm_range_max = max(int(perm_range.max().item()), perm_size)
    rand = torch.rand(
        batch_size,
        perm_range_max,
        dtype=torch.float32,
        device=perm_range.device,
    )
    rand[
        torch.arange(perm_range_max, device=perm_range.device).view(1, -1)
        >= perm_range.view(batch_size, 1)
    ] = float("-inf")
    res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
    res[res >= perm_range.view(batch_size, 1)] = padding
    return res


def _make_random_abs_indices(
    cache_seqlens: torch.Tensor,
    topk: int,
    max_q_len: int,
) -> torch.Tensor:
    perm_range = cache_seqlens.repeat_interleave(max_q_len)
    return _randperm_batch(perm_range, topk, -1).view(
        cache_seqlens.numel(), max_q_len, topk
    )


def _make_cache_seqlens(
    t: TestParam,
    s_k: int,
    block_size: int,
    q_lens: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    assert t.decode is not None
    base_len = max(s_k, topk, int(q_lens.max().item()))
    cache_seqlens = base_len + block_size * torch.arange(
        t.decode.b, dtype=torch.int32, device="cuda:0"
    )
    if t.decode.have_zero_seqlen_k:
        cache_seqlens[::2] = 0
    return cache_seqlens.contiguous()


def _make_topk_length(
    cache_seqlens: torch.Tensor,
    topk: int,
    max_q_len: int,
    mode: TopkLengthMode,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "seqlen":
        return cache_seqlens.view(-1, 1).expand(-1, max_q_len).clamp(max=topk)

    batch_delta = max(1, topk // TOPK_BATCH_VARIATION_DIVISOR)
    base_topk = topk - batch_delta * torch.arange(
        cache_seqlens.numel(), dtype=torch.int32, device="cuda:0"
    )
    q_positions = torch.arange(max_q_len, dtype=torch.int32, device="cuda:0")
    lens = base_topk.view(-1, 1) - TOPK_PER_QUERY_DECAY * q_positions.view(1, -1)
    return lens.clamp(min=1, max=topk).contiguous()


def _topk_length_for_flashinfer(
    topk_length: torch.Tensor,
    valid_q: torch.Tensor,
) -> torch.Tensor:
    if topk_length.ndim == 1:
        topk_length = topk_length.view(-1, 1).expand_as(valid_q)
    return topk_length[valid_q].contiguous()


def _topk_length_for_ref(
    topk_length: torch.Tensor,
    batch_size: int,
    max_q_len: int,
) -> torch.Tensor:
    if topk_length.ndim == 1:
        return topk_length.view(batch_size, 1, 1)
    return topk_length.view(batch_size, max_q_len, 1)


def generate_one_k_scope(
    t: TestParam,
    q_lens: torch.Tensor,
    s_k: int,
    block_size: int,
    topk: int,
    value_offset: float,
    topk_length_mode: TopkLengthMode,
    random_indices: bool,
) -> KVScope:
    assert t.decode is not None
    cache_seqlens = _make_cache_seqlens(t, s_k, block_size, q_lens, topk)
    block_table = _make_block_table(cache_seqlens, block_size)
    blocked_k = _make_kv_cache(
        block_table.numel(), block_size, t.h_kv, t.dtype, value_offset
    )
    if random_indices:
        abs_indices = _make_random_abs_indices(cache_seqlens, topk, t.s_q)
    else:
        abs_indices = _make_causal_abs_indices(cache_seqlens, q_lens, topk, t.s_q)
    if t.is_all_indices_invalid:
        abs_indices.fill_(-1)
    indices_in_kvcache = _abs_indices2indices_in_kvcache(
        abs_indices, block_table, block_size
    )
    topk_length = _make_topk_length(cache_seqlens, topk, t.s_q, topk_length_mode)
    return KVScope(
        t,
        cache_seqlens,
        block_table,
        blocked_k,
        abs_indices,
        indices_in_kvcache,
        topk_length,
    )


def generate_testcase_for_decode(t: TestParam) -> TestcaseForDecode:
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    torch.cuda.manual_seed_all(t.seed)

    assert t.h_q % t.h_kv == 0
    assert t.h_kv == 1
    assert t.decode is not None
    assert t.d_qk == DSV4_HEAD_DIM and t.d_v == DSV4_HEAD_DIM
    assert t.topk == DSV4_SWA_TOPK
    assert t.dtype in (torch.bfloat16, torch.float8_e4m3fn)

    q = torch.randn(
        (t.decode.b, t.s_q, t.h_q, t.d_qk),
        dtype=torch.float32,
        device="cuda:0",
    )
    q = q.mul_(QUERY_RANDOM_SCALE).clamp_(min=-1.0, max=1.0).to(t.dtype)

    q_lens = _make_q_lens(t)
    valid_q = torch.arange(t.s_q, device="cuda:0").view(1, t.s_q) < q_lens.view(
        t.decode.b, 1
    )

    attn_sink = None
    if t.have_attn_sink:
        attn_sink = (
            torch.randn((t.h_q,), dtype=torch.float32, device="cuda:0")
            * SINK_RANDOM_SCALE
        )

    kv_scope0 = generate_one_k_scope(
        t,
        q_lens,
        t.s_kv,
        t.decode.block_size,
        t.topk,
        SWA_KV_OFFSET,
        "none",
        False,
    )

    kv_scope1 = None
    if t.decode.extra_topk is not None:
        assert t.decode.extra_s_k is not None
        assert t.decode.extra_block_size is not None
        topk_length_mode: TopkLengthMode = (
            "seqlen" if t.decode.extra_block_size == C128_PAGE_SIZE else "profile"
        )
        kv_scope1 = generate_one_k_scope(
            t,
            q_lens,
            t.decode.extra_s_k,
            t.decode.extra_block_size,
            t.decode.extra_topk,
            COMPRESSED_KV_OFFSET,
            topk_length_mode,
            True,
        )

    return TestcaseForDecode(
        t,
        q,
        attn_sink,
        t.d_qk**-0.55,
        kv_scope0,
        kv_scope1,
        q_lens,
        valid_q,
        torch.zeros(WORKSPACE_SIZE, dtype=torch.int8, device="cuda:0"),
    )


def _scale_for_flashinfer(t: TestParam, value: float) -> float | torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return torch.tensor([value], dtype=torch.float32, device="cuda:0")
    return value


def run_flashinfer_decode(t: TestParam, testcase: TestcaseForDecode) -> torch.Tensor:
    assert t.decode is not None
    swa_indices = testcase.kv_scope.indices_in_kvcache[testcase.valid_q].contiguous()
    if testcase.extra_kv_scope is not None:
        assert testcase.extra_kv_scope.topk_length is not None
        compressed_kv_cache = testcase.extra_kv_scope.get_kvcache_for_flashinfer(
            t.kv_layout
        )
        compressed_indices = testcase.extra_kv_scope.indices_in_kvcache[
            testcase.valid_q
        ].contiguous()
        sparse_topk_lens = (
            _topk_length_for_flashinfer(
                testcase.extra_kv_scope.topk_length, testcase.valid_q
            )
            + DSV4_SWA_TOPK
        ).contiguous()
    else:
        compressed_kv_cache = testcase.kv_scope.get_kvcache_for_flashinfer(t.kv_layout)
        compressed_indices = swa_indices.new_empty((swa_indices.size(0), 0))
        sparse_topk_lens = swa_indices.new_full((swa_indices.size(0),), DSV4_SWA_TOPK)
    sparse_indices = torch.cat((swa_indices, compressed_indices), dim=-1).contiguous()
    if t.decode.is_varlen:
        query = testcase.q[testcase.valid_q].contiguous()
        cum_seq_lens_q = _make_cum_seq_lens(testcase.q_lens)
        max_q_len = t.s_q
    else:
        query = testcase.q.contiguous()
        cum_seq_lens_q = None
        max_q_len = None

    try:
        return trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            swa_kv_cache=testcase.kv_scope.get_kvcache_for_flashinfer(t.kv_layout),
            workspace_buffer=testcase.workspace_buffer,
            sparse_indices=sparse_indices,
            compressed_kv_cache=compressed_kv_cache,
            sparse_topk_lens=sparse_topk_lens,
            seq_lens=testcase.kv_scope.cache_seqlens,
            bmm1_scale=_scale_for_flashinfer(t, testcase.sm_scale),
            bmm2_scale=_scale_for_flashinfer(t, 1.0),
            sinks=testcase.attn_sink,
            kv_layout=t.kv_layout,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
            enable_pdl=False,
        )
    except RuntimeError as err:
        err_msg = str(err)
        if (
            "trtllm_paged_attention_decode_sparse_mla_dsv4 is not available" in err_msg
            or "Missing TRTLLM-GEN kernel (decode)" in err_msg
        ):
            pytest.skip("DeepSeek V4 sparse MLA TRTLLM-GEN cubins are not available")
        if (
            "Ninja build failed" in err_msg
            and "trtllm_fmha_kernel_launcher.cu" in err_msg
            and "missing and no known rule" in err_msg
        ):
            pytest.skip("TRTLLM-GEN JIT launcher sources are not available")
        raise


def ref_sparse_attn_decode(
    p: TestParam,
    t: TestcaseForDecode,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert p.h_kv == 1
    assert p.decode is not None
    batch_size = p.decode.b

    def process_kv_scope(kv_scope: KVScope) -> tuple[torch.Tensor, torch.Tensor]:
        topk = kv_scope.indices_in_kvcache.size(-1)
        indices_fixed = kv_scope.indices_in_kvcache.clamp_min(0)
        flat_kv = kv_scope.blocked_k.view(-1, p.d_qk).float()
        gathered_kv = flat_kv.index_select(0, indices_fixed.view(-1).long()).view(
            batch_size, p.s_q, topk, p.d_qk
        )
        invalid_mask = kv_scope.indices_in_kvcache == -1
        if kv_scope.topk_length is not None:
            lens = _topk_length_for_ref(kv_scope.topk_length, batch_size, p.s_q)
            invalid_mask |= (
                torch.arange(0, topk, device=t.q.device).view(1, 1, topk) >= lens
            )
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(t.kv_scope)
    if t.extra_kv_scope is not None:
        gathered_kv1, invalid_mask1 = process_kv_scope(t.extra_kv_scope)
        gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)
        invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)

    gathered_kv = gathered_kv.view(batch_size * p.s_q, -1, p.d_qk).float()
    q = t.q.float().view(batch_size * p.s_q, p.h_q, p.d_qk)
    attn_weight = q @ gathered_kv.transpose(-1, -2)
    attn_weight *= t.sm_scale
    attn_weight[
        invalid_mask.view(batch_size * p.s_q, 1, -1).broadcast_to(
            batch_size * p.s_q, p.h_q, invalid_mask.size(-1)
        )
    ] = float("-inf")
    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., : p.d_v]
    output = output.view(batch_size, p.s_q, p.h_q, p.d_v)
    lse = lse.view(batch_size, p.s_q, p.h_q)

    if t.attn_sink is not None:
        sink_scale = 1.0 / (1.0 + torch.exp(t.attn_sink.view(1, 1, p.h_q) - lse))
        sink_scale = torch.where(
            torch.isfinite(sink_scale), sink_scale, torch.zeros_like(sink_scale)
        )
        output *= sink_scale.unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(output.shape)] = 0.0
    lse[lonely_q_mask] = float("+inf")
    return output.to(torch.bfloat16), lse.transpose(1, 2)


def _assert_close(out: torch.Tensor, ref: torch.Tensor, dtype: torch.dtype) -> None:
    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    if dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-1, atol=1e-1)
    else:
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=8e-4)


def _skip_unless_sm100_or_sm103() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRTLLM-GEN sparse MLA tests")
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability not in ((10, 0), (10, 3)):
        pytest.skip(
            "TRTLLM-GEN DeepSeek V4 sparse MLA requires SM100/SM103, "
            f"got SM{compute_capability[0]}{compute_capability[1]}"
        )


@pytest.mark.parametrize("p", TESTCASES, ids=lambda p: p.id)
@torch.inference_mode()
def test_trtllm_gen_sparse_mla_dsv4(p: TestParam) -> None:
    _skip_unless_sm100_or_sm103()
    testcase = generate_testcase_for_decode(p)
    out_ans = run_flashinfer_decode(p, testcase)
    out_ref, _ = ref_sparse_attn_decode(p, testcase)
    if p.decode is not None and p.decode.is_varlen:
        out_ref = out_ref[testcase.valid_q]
    _assert_close(out_ans, out_ref, p.dtype)
