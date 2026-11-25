"""Tests of Cake's Nemotron 3.5 Super NVFP4 routes, ``decode_super_nvfp4`` and
``context_super_nvfp4`` of ``backend="cake"`` (sm_100).

A call a kernel serves is compared row by row, where a row is one head of one
query token, with ``backend="trtllm-gen"`` and with an FP32 reference on the
same quantized inputs. A call the routes run on trtllm-gen must return exactly
what ``backend="trtllm-gen"`` returns. The route counters tell the two apart.

The routing rules are tested without a GPU in
``tests/cpu/attention/test_cake_fmha_super_nvfp4_routing.py``.
"""

import collections
import math
import os
from typing import NamedTuple, Tuple

import pytest
import torch

import flashinfer
import flashinfer.cake_fmha
import flashinfer.cake_fmha_super_nvfp4 as super_nvfp4
from flashinfer.fp4_quantization import nvfp4_quantize_paged_kv_cache
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.utils import get_device_sm_count

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="Cake's Nemotron 3.5 Super NVFP4 kernels run on sm_100 only",
)

MTP, BASE = super_nvfp4.DECODE_MTP, super_nvfp4.DECODE_BASE
CAKE, TRTLLM_GEN = "cake", "trtllm-gen"
NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 32, 2, 128, 128
MIB = 1 << 20
O_SCALE = 0.05

# Relative L2 error of one output row. The kernels, trtllm-gen and the FP32
# reference differ by FP8 rounding only: over uniform and peaked attention, 1 to
# 1M KV tokens and 1 to 256 requests, the largest row error measured between
# any two of them is 0.089 and the largest mean is 0.049. A row that misses one
# of 70 pages is off by 0.12, and a row from the wrong tokens by more than 1.
MAX_ROW_ERROR = 0.13
MAX_MEAN_ROW_ERROR = 0.06


@pytest.fixture(scope="module", autouse=True)
def super_nvfp4_kernels():
    """Build the kernels. Skips only below the CUDA 13.0 toolkit they need: a
    build that fails otherwise is an error."""
    from flashinfer.jit.cake_fmha_super_nvfp4 import (
        gen_cake_fmha_super_nvfp4_module,
        load_cake_fmha_super_nvfp4_module,
    )

    if not (
        gen_cake_fmha_super_nvfp4_module().is_compiled
        or "FLASHINFER_NVCC" in os.environ
        or is_cuda_version_at_least("13.0")
    ):
        pytest.skip("building the kernels needs the CUDA 13.0 toolkit or newer")
    load_cake_fmha_super_nvfp4_module()
    assert super_nvfp4._module() is not None


_TAKEN: collections.Counter = collections.Counter()


@pytest.fixture(autouse=True)
def _record_routes(monkeypatch):
    """Record the kernel of every super_nvfp4 route the calls select."""

    def recorder(stage, select):
        def recording(**kwargs):
            kernel = select(**kwargs)
            if kernel is not None:
                _TAKEN[(stage, kernel)] += 1
            return kernel

        return recording

    for stage, name in (
        ("decode", "select_decode_route"),
        ("prefill", "select_prefill_route"),
    ):
        monkeypatch.setattr(
            super_nvfp4, name, recorder(stage, getattr(super_nvfp4, name))
        )


def _reset_routes():
    _TAKEN.clear()


def _routes_taken():
    return dict(_TAKEN)


_BUILD_FUSED_FP4_MUL = super_nvfp4._uses_fused_fp4_mul


@pytest.fixture(autouse=True)
def fused_routing(monkeypatch):
    """Route decode as a build with the fused NVFP4 multiply does, so that every
    build runs the same kernels on the same shapes."""
    monkeypatch.setattr(super_nvfp4, "_uses_fused_fp4_mul", lambda: True)


def test_dequantization_path_follows_the_toolkit():
    # The fused NVFP4 multiply needs the CUDA 13.4 assembler; older toolkits
    # build the FP16 path, which every other test here then runs.
    from flashinfer.jit.cake_fmha_super_nvfp4 import gen_cake_fmha_super_nvfp4_module

    if gen_cake_fmha_super_nvfp4_module().is_aot or "FLASHINFER_NVCC" in os.environ:
        pytest.skip("the module was not built by this toolkit's compiler")
    fused = super_nvfp4._module().super_nvfp4_fused_fp4_mul()
    assert fused == is_cuda_version_at_least(super_nvfp4.FUSED_DEQUANT_CUDA_TOOLKIT)


class Inputs(NamedTuple):
    query: torch.Tensor  # [rows, 32, 128] E4M3
    kv_cache: Tuple[torch.Tensor, torch.Tensor]  # NVFP4 pages, HND
    kv_cache_sf: Tuple[torch.Tensor, torch.Tensor]
    block_tables: torch.Tensor
    seq_lens: torch.Tensor  # on the device
    bmm1_scale: float
    bmm2_scale: float


def _to_fp8(x: torch.Tensor):
    scale = 448.0 * 0.1 / x.abs().max().clamp(min=1e-12)
    return (x * scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), 1.0 / scale.item()


def _make_inputs(
    seq_lens, num_query_rows: int, sharpness: float = 1.0, needles=None
) -> Inputs:
    """NVFP4 pages for ``seq_lens`` (one row of unique pages per request) and an
    FP8 query of ``num_query_rows`` rows. The attention logits have a standard
    deviation of ``sharpness``: 1 spreads a row over thousands of tokens, 4
    concentrates it on a few, wherever they are in the KV.

    ``needles`` lists KV positions per request, which may lie past its length.
    Their keys match the query, which is then the same in every row, so a row
    returns the mean value of the needles in its causal range: one token more
    or less in that range changes it."""
    device = "cuda"
    seq_lens = torch.as_tensor(seq_lens, dtype=torch.int32)
    pages_per_seq = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
    max_pages = int(pages_per_seq.max())
    num_pages = int(pages_per_seq.sum())
    shape = (num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    page_ids = torch.randperm(num_pages, dtype=torch.int32, device=device)
    block_tables = torch.zeros(
        (seq_lens.numel(), max_pages), dtype=torch.int32, device=device
    )
    start = 0
    for i, pages in enumerate(pages_per_seq.tolist()):
        block_tables[i, :pages] = page_ids[start : start + pages]
        start += pages
    query = torch.randn((num_query_rows, NUM_QO_HEADS, HEAD_DIM), device=device)
    if needles is not None:
        query = query[:1, :1].expand_as(query)
        for i, positions in enumerate(needles):
            for position in positions:
                page = block_tables[i, position // PAGE_SIZE]
                k[page, :, position % PAGE_SIZE] = 2.0 * query[0, 0]
    kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(
        k, v, kv_layout="HND"
    )
    query, q_scale = _to_fp8(query * sharpness)
    return Inputs(
        query,
        kv_cache,
        kv_cache_sf,
        block_tables,
        seq_lens.to(device),
        q_scale * float(k_scale) / math.sqrt(HEAD_DIM),
        float(v_scale) / O_SCALE,
    )


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize(inputs: Inputs):
    """K and V as the kernels use them: NVFP4 times its block scale, rounded to
    E4M3 with saturation. ``[pages, 2, 128, 128]`` FP32."""
    table = torch.tensor(_E2M1 + tuple(-x for x in _E2M1), device="cuda")
    result = []
    for packed, scales, interleaved in zip(
        inputs.kv_cache, inputs.kv_cache_sf, (False, True), strict=True
    ):
        nibbles = packed.long()
        values = torch.stack((table[nibbles & 15], table[nibbles >> 4]), -1).flatten(-2)
        if interleaved:  # the V scales of four tokens are stored interleaved
            pages = scales.shape[0]
            scales = scales.reshape(pages, NUM_KV_HEADS, PAGE_SIZE // 4, 4, 2, 4)
            scales = scales.permute(0, 1, 2, 5, 3, 4)
            scales = scales.reshape(pages, NUM_KV_HEADS, PAGE_SIZE, 8)
        values = values * scales.float().repeat_interleave(16, dim=-1)
        result.append(values.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float())
    return result


def _reference(inputs: Inputs, q_lens) -> torch.Tensor:
    """FP32 causal attention on the dequantized inputs. Query token ``t`` of a
    request with ``q`` query tokens and ``kv`` KV tokens attends to the first
    ``kv - q + t + 1`` of them."""
    k_all, v_all = _dequantize(inputs)
    out = torch.empty(inputs.query.shape, dtype=torch.float32, device="cuda")
    heads_per_kv = NUM_QO_HEADS // NUM_KV_HEADS
    row = 0
    for i, (kv_len, q_len) in enumerate(
        zip(inputs.seq_lens.tolist(), q_lens, strict=True)
    ):
        pages = inputs.block_tables[i, : (kv_len + PAGE_SIZE - 1) // PAGE_SIZE].long()
        k = k_all[pages].transpose(0, 1).reshape(NUM_KV_HEADS, -1, HEAD_DIM)[:, :kv_len]
        v = v_all[pages].transpose(0, 1).reshape(NUM_KV_HEADS, -1, HEAD_DIM)[:, :kv_len]
        position = torch.arange(kv_len, device="cuda")
        for begin in range(0, q_len, 256):
            end = min(q_len, begin + 256)
            q = inputs.query[row + begin : row + end].float()
            q = q.reshape(end - begin, NUM_KV_HEADS, heads_per_kv, HEAD_DIM)
            logits = torch.einsum("qhgd,hnd->qhgn", q, k) * inputs.bmm1_scale
            visible = kv_len - q_len + torch.arange(begin, end, device="cuda") + 1
            hidden = position[None, None, None, :] >= visible[:, None, None, None]
            p = torch.softmax(logits.masked_fill_(hidden, -math.inf), dim=-1)
            o = torch.einsum("qhgn,hnd->qhgd", p, v) * inputs.bmm2_scale
            out[row + begin : row + end] = o.reshape(
                end - begin, NUM_QO_HEADS, HEAD_DIM
            )
        row += q_len
    return out


def _assert_rows_close(out: torch.Tensor, expected: torch.Tensor, name: str):
    out, expected = out.float(), expected.float()
    assert torch.isfinite(out).all()
    error = (out - expected).norm(dim=-1) / expected.norm(dim=-1).clamp(min=1e-12)
    worst = error.argmax()
    token, head = divmod(int(worst), NUM_QO_HEADS)
    assert error.max() < MAX_ROW_ERROR, (
        f"query token {token}, head {head}: relative error {error.max():.3f} to {name}"
    )
    assert error.mean() < MAX_MEAN_ROW_ERROR, (
        f"mean relative row error {error.mean():.3f} to {name}"
    )


def _assert_kernel_output(out: torch.Tensor, expected: torch.Tensor, inputs, q_lens):
    _assert_rows_close(out, expected, "trtllm-gen")
    _assert_rows_close(out, _reference(inputs, q_lens), "the FP32 reference")


def _assert_same_bytes(out: torch.Tensor, expected: torch.Tensor):
    same = torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
    assert same, "the two outputs differ"


def _cumulative(lengths) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    result[1:] = torch.cumsum(torch.as_tensor(lengths), 0)
    return result.cuda()


def _strided(tensor: torch.Tensor) -> torch.Tensor:
    """The same values in every second slot of the first dimension."""
    spaced = tensor.new_zeros((2 * tensor.shape[0],) + tensor.shape[1:])
    spaced[::2] = tensor
    return spaced[::2]


def _offset(tensor: torch.Tensor, offset: int) -> torch.Tensor:
    """The same values, ``offset`` bytes past an aligned allocation."""
    size = tensor.numel() * tensor.element_size()
    buffer = torch.zeros(size + 256, dtype=torch.uint8, device=tensor.device)
    view = buffer[offset : offset + size].view(tensor.dtype).view(tensor.shape)
    view.copy_(tensor)
    return view


def _workspace(num_bytes: int = 128 * MIB) -> torch.Tensor:
    return torch.zeros(num_bytes, dtype=torch.uint8, device="cuda")


def _decode_workspace_bytes(batch_size: int, q_len: int, fused: int) -> int:
    """Workspace a decode kernel needs when it fuses ``fused`` query tokens: the
    layout of ``fillDecodePlan`` in ``cake_fmha_super_nvfp4/host.h``."""
    ctas = get_device_sm_count(torch.device("cuda")) & ~1
    units = batch_size * (q_len // fused)
    slots = (units + ctas // 2) * 2
    stats = slots * 16 * fused * HEAD_DIM
    ranges = stats + slots * 16 * fused * 2
    segments = (ranges + units * 4 + 3) // 4 * 4
    return (segments + ctas * (units + 1) * 4) * 4


def _routes(stage: str, kernel: str):
    return {(stage, kernel): 1}


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _run_decode(backend, inputs: Inputs, q_len, cum_seq_lens_q=None, **overrides):
    """Decode ``inputs`` with ``q_len`` query tokens per request, or with the
    packed offsets ``cum_seq_lens_q`` and at most ``q_len`` tokens."""
    args = dict(
        query=inputs.query,
        kv_cache=inputs.kv_cache,
        workspace_buffer=None,
        block_tables=inputs.block_tables,
        seq_lens=inputs.seq_lens,
        max_seq_len=None,
        bmm1_scale=inputs.bmm1_scale,
        bmm2_scale=inputs.bmm2_scale,
        out=None,
        kv_layout="HND",
        enable_pdl=True,
        backend=backend,
        q_len_per_req=None if cum_seq_lens_q is not None else q_len,
        max_q_len=q_len if cum_seq_lens_q is not None else None,
        cum_seq_lens_q=cum_seq_lens_q,
        kv_cache_sf=inputs.kv_cache_sf,
    )
    args.update(overrides)
    if args["workspace_buffer"] is None:
        args["workspace_buffer"] = _workspace()
    if args["max_seq_len"] is None:
        args["max_seq_len"] = int(inputs.seq_lens.max())
    if args["out"] is None:
        args["out"] = torch.zeros_like(inputs.query)
    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(**args)


def _check_decode(kernel, inputs, q_len, q_lens=None, **overrides):
    """Decode on both backends; ``kernel`` must serve the cake call, and its
    output must match trtllm-gen and the reference."""
    cum_seq_lens_q = None if q_lens is None else _cumulative(q_lens)
    _reset_routes()
    out = _run_decode(CAKE, inputs, q_len, cum_seq_lens_q, **overrides)
    assert _routes_taken() == _routes("decode", kernel)
    # A workspace sized for the kernel can be too small for trtllm-gen.
    overrides.pop("workspace_buffer", None)
    expected = _run_decode(TRTLLM_GEN, inputs, q_len, cum_seq_lens_q, **overrides)
    if q_lens is None:
        q_lens = [q_len] * inputs.seq_lens.numel()
    _assert_kernel_output(out, expected, inputs, q_lens)
    return out


def _check_decode_fallback(inputs, q_len, cum_seq_lens_q=None, **overrides):
    """The cake call must run on trtllm-gen and return exactly its output.
    ``overrides`` maps an argument to a function that builds it per call."""
    outs = []
    for backend in (CAKE, TRTLLM_GEN):
        made = {name: make() for name, make in overrides.items()}
        _reset_routes()
        outs.append(_run_decode(backend, inputs, q_len, cum_seq_lens_q, **made))
        if backend == CAKE:
            assert _routes_taken() == _routes("decode", TRTLLM_GEN)
    _assert_same_bytes(*outs)


@pytest.mark.parametrize("sharpness", [1.0, 4.0])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize(
    "q_len,kernel", [(1, BASE), (2, MTP), (3, BASE), (4, MTP), (5, BASE), (8, MTP)]
)
@pytest.mark.parametrize("batch_size", [1, 4, 33])
def test_decode_kernels(batch_size, q_len, kernel, enable_pdl, sharpness):
    torch.manual_seed(batch_size * 100 + q_len)
    seq_lens = torch.randint(8300, 10000, (batch_size,))
    inputs = _make_inputs(seq_lens, batch_size * q_len, sharpness)
    _check_decode(kernel, inputs, q_len, enable_pdl=enable_pdl)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_decode_six_tokens_few_requests_run_mtp(batch_size):
    torch.manual_seed(batch_size)
    inputs = _make_inputs(torch.randint(8300, 10000, (batch_size,)), batch_size * 6)
    _check_decode(MTP, inputs, 6)


@pytest.mark.parametrize(
    "batch_size,q_len,seq_lens",
    [
        (4, 6, None),  # 6 query tokens and more than 2 requests
        (4, 7, None),
        (2, 1, [8192, 900]),  # no KV longer than 8192 tokens
        (2, 3, [8192, 900]),
        (2, 8, [8192, 900]),
    ],
)
def test_decode_shapes_routed_to_trtllm_gen(batch_size, q_len, seq_lens):
    torch.manual_seed(q_len)
    if seq_lens is None:
        seq_lens = torch.randint(8300, 10000, (batch_size,))
    inputs = _make_inputs(seq_lens, batch_size * q_len)
    _check_decode_fallback(inputs, q_len)


def test_decode_routing_follows_the_build(monkeypatch):
    # 4 requests of 1 query token: "base" with the fused multiply, trtllm-gen
    # without it.
    monkeypatch.setattr(super_nvfp4, "_uses_fused_fp4_mul", _BUILD_FUSED_FP4_MUL)
    torch.manual_seed(10)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4)
    if super_nvfp4._module().super_nvfp4_fused_fp4_mul():
        _check_decode(BASE, inputs, 1)
    else:
        _check_decode_fallback(inputs, 1)


def test_decode_mtp_six_tokens_with_more_requests_than_routed(monkeypatch):
    # "mtp" runs 6 query tokens as 3 groups of 2; the rules send it 2 requests.
    monkeypatch.setattr(super_nvfp4, "select_decode_kernel", lambda *args: MTP)
    torch.manual_seed(6)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * 6, sharpness=4.0)
    _check_decode(MTP, inputs, 6)


@pytest.mark.parametrize("kernel,q_len", [(BASE, 2), (BASE, 4), (BASE, 8), (MTP, 3)])
def test_decode_kernel_declines_query_lengths_it_does_not_serve(
    monkeypatch, kernel, q_len
):
    # "base" fuses 1, 3 or 5 query tokens and "mtp" 2, 4 or 8.
    monkeypatch.setattr(super_nvfp4, "select_decode_kernel", lambda *args: kernel)
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len)
    _check_decode_fallback(inputs, q_len)


@pytest.mark.parametrize("kernel,q_len", [(BASE, 2), (MTP, 3)])
def test_decode_kernel_declining_its_route_raises(monkeypatch, kernel, q_len):
    # The kernels keep their own checks: a route the selection should not have
    # chosen fails loudly instead of running on the wrong kernel.
    monkeypatch.setattr(super_nvfp4, "select_decode_route", lambda **_: kernel)
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len)
    with pytest.raises(RuntimeError, match="declined the call its route selected"):
        _run_decode(CAKE, inputs, q_len)


@pytest.mark.parametrize("q_lens", [[3, 1, 2, 3], [5, 1, 4, 2], [3, 3, 1, 3]])
def test_decode_ragged_queries_run_base(q_lens):
    torch.manual_seed(7)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), sum(q_lens), 4.0)
    _check_decode(BASE, inputs, max(q_lens), q_lens)


def test_decode_ragged_queries_on_base_up_to_six_tokens(monkeypatch):
    # The longest ragged request "base" accepts, which the rules do not send it.
    monkeypatch.setattr(super_nvfp4, "select_decode_kernel", lambda *args: BASE)
    torch.manual_seed(8)
    q_lens = [6, 1, 4, 2]
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), sum(q_lens), 4.0)
    _check_decode(BASE, inputs, 6, q_lens)


@pytest.mark.parametrize("q_lens", [[2, 1, 2, 2], [4, 1, 3, 4], [6, 5], [8, 3, 8, 8]])
def test_decode_ragged_queries_routed_to_trtllm_gen(q_lens):
    # "mtp" serves no ragged call, so these are not offered to it.
    torch.manual_seed(9)
    seq_lens = torch.randint(8300, 10000, (len(q_lens),))
    inputs = _make_inputs(seq_lens, sum(q_lens))
    _check_decode_fallback(inputs, max(q_lens), _cumulative(q_lens))


@pytest.mark.parametrize(
    "batch_size,q_len,kernel",
    [(4, 1, BASE), (4, 2, None), (4, 3, BASE), (2, 6, None), (4, 8, None)],
)
def test_decode_uniform_queries_with_offsets(batch_size, q_len, kernel):
    # A call with cum_seq_lens_q is routed as a ragged one even when it gives
    # every request q_len tokens: the offsets can change between graph replays.
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (batch_size,)), batch_size * q_len)
    q_lens = [q_len] * batch_size
    if kernel is None:
        _check_decode_fallback(inputs, q_len, _cumulative(q_lens))
    else:
        _check_decode(kernel, inputs, q_len, q_lens)


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (5, BASE), (2, MTP), (8, MTP)])
def test_decode_256_requests(q_len, kernel):
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 9000, (256,)), 256 * q_len)
    # "mtp" with 8 query tokens is the largest workspace a decode call needs.
    workspace = _workspace(_decode_workspace_bytes(256, q_len, fused=q_len))
    _check_decode(kernel, inputs, q_len, workspace_buffer=workspace)


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (5, BASE), (2, MTP), (8, MTP)])
def test_decode_long_kv(q_len, kernel):
    torch.manual_seed(q_len)
    inputs = _make_inputs([131072, 140000], 2 * q_len, sharpness=4.0)
    _check_decode(kernel, inputs, q_len)


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (3, BASE), (2, MTP), (8, MTP)])
def test_decode_short_and_long_requests(q_len, kernel):
    # Requests below one page, at page boundaries and far above, in one call.
    torch.manual_seed(q_len)
    seq_lens = [9, 20000, 127, 128, 129, 8, 4000, 12345, 64, 256, 9000]
    inputs = _make_inputs(seq_lens, len(seq_lens) * q_len, sharpness=4.0)
    _check_decode(kernel, inputs, q_len)


@pytest.mark.parametrize(
    "q_len,kernel", [(1, BASE), (3, BASE), (5, BASE), (2, MTP), (4, MTP), (8, MTP)]
)
def test_decode_causal_range(q_len, kernel):
    # Needles on the last q_len + 1 tokens of every request and on the token
    # after it: query token t must see exactly t + 2 of them. The requests end
    # at the last token of a page, on the first of the next, and inside one.
    torch.manual_seed(q_len)
    seq_lens = [8448, 8449, 9000, 9999]
    needles = [
        [p for p in range(n - q_len - 1, n + 1) if p < n or n % PAGE_SIZE]
        for n in seq_lens
    ]
    inputs = _make_inputs(seq_lens, 4 * q_len, needles=needles)
    _check_decode(kernel, inputs, q_len)


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (2, MTP)])
@pytest.mark.parametrize("scales", ["bmm1", "bmm2", "both", "bmm1_log2"])
def test_decode_device_scales(q_len, kernel, scales):
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len, sharpness=4.0)
    bmm1 = torch.tensor([inputs.bmm1_scale], dtype=torch.float32, device="cuda")
    bmm2 = torch.tensor([inputs.bmm2_scale], dtype=torch.float32, device="cuda")
    bmm1_log2 = torch.tensor(
        [inputs.bmm1_scale * math.log2(math.e)], dtype=torch.float32, device="cuda"
    )
    overrides = {
        "bmm1": dict(bmm1_scale=bmm1),
        "bmm2": dict(bmm2_scale=bmm2),
        "both": dict(bmm1_scale=bmm1, bmm2_scale=bmm2),
        "bmm1_log2": dict(bmm1_scale_log2=bmm1_log2),
    }[scales]
    out = _check_decode(kernel, inputs, q_len, **overrides)
    # A bmm1_scale tensor is converted to log2 in FP32, a float in FP64, so only
    # the other two forms carry exactly the scales of the float call.
    if scales in ("bmm2", "bmm1_log2"):
        _assert_same_bytes(out, _run_decode(CAKE, inputs, q_len))


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (2, MTP)])
def test_decode_nhd_layout(q_len, kernel):
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len)
    nhd = dict(
        kv_cache=tuple(x.transpose(1, 2).contiguous() for x in inputs.kv_cache),
        kv_cache_sf=tuple(x.transpose(1, 2).contiguous() for x in inputs.kv_cache_sf),
        kv_layout="NHD",
    )
    out = _check_decode(kernel, inputs, q_len, **nhd)
    _assert_same_bytes(out, _run_decode(CAKE, inputs, q_len))


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (2, MTP)])
def test_decode_stacked_kv_cache(q_len, kernel):
    # K and V as the two halves of one tensor: pages are 2 pages apart.
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len)
    stacked = torch.stack(inputs.kv_cache, dim=1)
    out = _check_decode(kernel, inputs, q_len, kv_cache=stacked)
    _assert_same_bytes(out, _run_decode(CAKE, inputs, q_len))


@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize(
    "declined",
    [
        "workspace too small",
        "workspace not 16-byte aligned",
        "max_seq_len above the page table",
        "block_tables not contiguous",
        "seq_lens not contiguous",
        "query not packed",
        "out not packed",
        "bmm1_scale zero",
        "bmm1_scale zero in float",
        "bmm1_scale negative",
    ],
)
def test_decode_declined_calls_run_trtllm_gen(q_len, declined):
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len)
    required = _decode_workspace_bytes(4, q_len, fused=q_len)
    table_tokens = inputs.block_tables.size(1) * PAGE_SIZE
    overrides = {
        "workspace too small": dict(workspace_buffer=lambda: _workspace(required - 16)),
        "workspace not 16-byte aligned": dict(
            workspace_buffer=lambda: _workspace(128 * MIB + 16)[8:-8]
        ),
        "max_seq_len above the page table": dict(max_seq_len=lambda: table_tokens + 1),
        "block_tables not contiguous": dict(
            block_tables=lambda: _strided(inputs.block_tables.T).T
        ),
        "seq_lens not contiguous": dict(seq_lens=lambda: _strided(inputs.seq_lens)),
        "query not packed": dict(query=lambda: _strided(inputs.query)),
        "out not packed": dict(out=lambda: _strided(torch.zeros_like(inputs.query))),
        # Such scales would leave masked positions visible to the kernels.
        "bmm1_scale zero": dict(bmm1_scale=lambda: 0.0),
        "bmm1_scale zero in float": dict(bmm1_scale=lambda: 1e-300),
        "bmm1_scale negative": dict(bmm1_scale=lambda: -inputs.bmm1_scale),
    }[declined]
    if declined in (
        "block_tables not contiguous",
        "seq_lens not contiguous",
        "out not packed",
    ):
        # The direct backend needs packed tensors. Compare to the original
        # packed inputs so identical misaddressing cannot pass this check.
        _reset_routes()
        out = _run_decode(
            CAKE, inputs, q_len, **{name: make() for name, make in overrides.items()}
        )
        assert _routes_taken() == _routes("decode", TRTLLM_GEN)
        _assert_same_bytes(out, _run_decode(TRTLLM_GEN, inputs, q_len))
    else:
        _check_decode_fallback(inputs, q_len, **overrides)
    if declined == "workspace too small":
        # The kernel serves the call with exactly the workspace it asks for.
        kernel = super_nvfp4.select_decode_kernel(4, q_len, int(inputs.seq_lens.max()))
        _check_decode(kernel, inputs, q_len, workspace_buffer=_workspace(required))


@pytest.mark.parametrize("unaligned", ["query", "k_cache"])
def test_decode_unaligned_tensors_fail_as_on_trtllm_gen(unaligned):
    # The kernels decline pointers that are not 16-byte aligned. trtllm-gen
    # rejects them, so the call fails the way it does on backend="trtllm-gen".
    torch.manual_seed(0)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4)
    if unaligned == "query":
        overrides = dict(query=_offset(inputs.query, 8))
    else:
        k_cache, v_cache = inputs.kv_cache
        overrides = dict(kv_cache=(_offset(k_cache, 8), v_cache))
    for backend in (CAKE, TRTLLM_GEN):
        _reset_routes()
        with pytest.raises(RuntimeError):
            _run_decode(backend, inputs, 1, **overrides)
        if backend == CAKE:
            assert _routes_taken() == _routes("decode", TRTLLM_GEN)


@pytest.mark.parametrize("declined", ["v_cache 16 bytes past 32", "empty bmm2_scale"])
def test_decode_route_declines_before_launch(declined):
    # Decode reads V through a tensor map that needs 32-byte alignment, and a
    # scale tensor must hold one value: route selection hands such calls to
    # trtllm-gen instead of launching a kernel that would fail.
    torch.manual_seed(1)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4)
    k_cache, v_cache = inputs.kv_cache
    k_sf, v_sf = inputs.kv_cache_sf
    bmm2_scale = inputs.bmm2_scale
    if declined == "v_cache 16 bytes past 32":
        v_cache = _offset(v_cache, 16)
    else:
        bmm2_scale = torch.empty(0, dtype=torch.float32, device="cuda")
    out = torch.zeros_like(inputs.query)
    kernel = super_nvfp4.select_decode_route(
        out=out,
        query=inputs.query,
        k_cache=k_cache,
        v_cache=v_cache,
        k_block_scales=k_sf,
        v_block_scales=v_sf,
        workspace_buffer=_workspace(),
        block_tables=inputs.block_tables,
        seq_lens=inputs.seq_lens,
        cum_seq_lens_q=None,
        batch_size=4,
        max_q_len=1,
        max_kv_len=int(inputs.seq_lens.max()),
        bmm1_scale=inputs.bmm1_scale,
        bmm2_scale=bmm2_scale,
        enable_pdl=True,
        dense_causal=True,
    )
    assert kernel == TRTLLM_GEN


def test_decode_features_run_trtllm_gen():
    torch.manual_seed(0)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4)
    _check_decode_fallback(inputs, 1, window_left=lambda: 4096)


def test_decode_other_geometry_is_not_routed():
    # 64-token pages are outside the geometry: the call keeps Cake's other
    # routes, and the Nemotron routes do not count it.
    torch.manual_seed(11)
    shape = (40, NUM_KV_HEADS, 64, HEAD_DIM)
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    kv_cache, kv_cache_sf, k_scale, v_scale = nvfp4_quantize_paged_kv_cache(k, v)
    query, q_scale = _to_fp8(torch.randn((2, NUM_QO_HEADS, HEAD_DIM), device="cuda"))
    block_tables = torch.arange(40, dtype=torch.int32, device="cuda").reshape(2, 20)
    seq_lens = torch.tensor([1280, 1000], dtype=torch.int32, device="cuda")
    route = flashinfer.cake_fmha.select_cake_fmha_decode_route(
        query.device,
        query=query,
        key_cache=kv_cache[0],
        value_cache=kv_cache[1],
        out=torch.empty_like(query),
        workspace_buffer=_workspace(),
        block_tables=block_tables,
        seq_lens=seq_lens,
        batch_size=2,
        q_len=1,
        max_seq_len=1280,
        window_left=-1,
        bmm1_scale=q_scale * float(k_scale) / math.sqrt(HEAD_DIM),
        bmm2_scale=float(v_scale) / O_SCALE,
        o_scale=None,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=kv_cache_sf[0],
        value_block_scales=kv_cache_sf[1],
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
    )
    assert route is None or route.component != "decode_super_nvfp4"


@pytest.mark.parametrize("q_len,kernel", [(1, BASE), (3, BASE), (2, MTP), (8, MTP)])
def test_decode_cuda_graph(q_len, kernel):
    torch.manual_seed(q_len)
    inputs = _make_inputs(torch.randint(8300, 10000, (4,)), 4 * q_len)
    # Nothing may be copied to or from the device during capture.
    fixed = dict(workspace_buffer=_workspace(), max_seq_len=int(inputs.seq_lens.max()))
    eager = _run_decode(CAKE, inputs, q_len, **fixed).clone()

    out = torch.empty_like(inputs.query)
    graph = torch.cuda.CUDAGraph()
    _reset_routes()
    with torch.cuda.graph(graph):
        _run_decode(CAKE, inputs, q_len, out=out, **fixed)
    assert _routes_taken() == _routes("decode", kernel)
    for _ in range(2):
        out.zero_()
        graph.replay()
        _assert_same_bytes(out, eager)


@pytest.mark.parametrize(
    "q_len,kernel",
    [(2, TRTLLM_GEN), (3, BASE), (4, TRTLLM_GEN), (5, BASE), (8, TRTLLM_GEN)],
)
@pytest.mark.parametrize("kv", ["short", "long"])
def test_decode_cuda_graph_replays_new_query_offsets(kv, q_len, kernel):
    # A graph captured with uniform query lengths and cum_seq_lens_q replays
    # whatever lengths the offsets hold then, within the captured rows and
    # max_q_len: speculative decoding that trims drafts on the device does this.
    torch.manual_seed(q_len)
    batch_size = 4
    if kv == "short":
        capacity = [10000, 9800, 9500, 9900]
    else:
        capacity = [140000, 131072, 9000, 135000]
    inputs = _make_inputs(capacity, batch_size * q_len, sharpness=4.0)
    cum_seq_lens_q = _cumulative([q_len] * batch_size)
    fixed = dict(workspace_buffer=_workspace(), max_seq_len=max(capacity))

    out = torch.empty_like(inputs.query)
    graph = torch.cuda.CUDAGraph()
    _reset_routes()
    with torch.cuda.graph(graph):
        _run_decode(CAKE, inputs, q_len, cum_seq_lens_q, out=out, **fixed)
    routes = _routes_taken()

    layouts = [
        [q_len] * batch_size,
        [1] * batch_size,
        [q_len, 1, q_len - 1, 2],
        [1, q_len, 2, q_len - 1],
    ]
    for q_lens in layouts:
        num_rows = sum(q_lens)
        rows = torch.randperm(inputs.query.size(0), device="cuda")
        inputs.query.copy_(inputs.query[rows])
        seq_lens = [int(torch.randint(n // 2, n + 1, ())) for n in capacity]
        inputs.seq_lens.copy_(torch.tensor(seq_lens, dtype=torch.int32))
        cum_seq_lens_q.copy_(_cumulative(q_lens))
        out.zero_()
        graph.replay()

        replayed = inputs._replace(query=inputs.query[:num_rows])
        expected = _run_decode(TRTLLM_GEN, replayed, q_len, cum_seq_lens_q, **fixed)
        _assert_kernel_output(out[:num_rows], expected, replayed, q_lens)
    assert routes == _routes("decode", kernel)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _run_prefill(backend, inputs: Inputs, q_lens, workspace_bytes=None, **overrides):
    seq_lens = inputs.seq_lens
    args = dict(
        query=inputs.query,
        kv_cache=inputs.kv_cache,
        workspace_buffer=None,
        block_tables=inputs.block_tables,
        seq_lens=seq_lens,
        max_q_len=max(q_lens),
        max_kv_len=None,
        bmm1_scale=inputs.bmm1_scale,
        bmm2_scale=inputs.bmm2_scale,
        batch_size=seq_lens.numel(),
        cum_seq_lens_q=None,
        cum_seq_lens_kv=None,
        out=None,
        kv_layout="HND",
        kv_cache_sf=inputs.kv_cache_sf,
        backend=backend,
    )
    args.update(overrides)
    if args["max_kv_len"] is None:
        args["max_kv_len"] = int(seq_lens.max())
    if args["workspace_buffer"] is None:
        if workspace_bytes is None:
            workspace_bytes = super_nvfp4.prefill_workspace_bytes(int(seq_lens.max()))
        args["workspace_buffer"] = _workspace(workspace_bytes)
    if args["cum_seq_lens_q"] is None:
        args["cum_seq_lens_q"] = _cumulative(q_lens)
    if args["cum_seq_lens_kv"] is None:
        pages_per_seq = (seq_lens.cpu() + PAGE_SIZE - 1) // PAGE_SIZE
        args["cum_seq_lens_kv"] = _cumulative(pages_per_seq.tolist())
    if args["out"] is None:
        args["out"] = torch.zeros_like(inputs.query)
    return flashinfer.prefill.trtllm_batch_context_with_kv_cache(**args)


def _check_prefill(inputs, q_len, **overrides):
    """Prefill one request on both backends; the kernel must serve the cake call."""
    _reset_routes()
    out = _run_prefill(CAKE, inputs, [q_len], **overrides)
    assert _routes_taken() == _routes("prefill", "prefill")
    expected = _run_prefill(TRTLLM_GEN, inputs, [q_len], **overrides)
    _assert_kernel_output(out, expected, inputs, [q_len])
    return out


def _check_prefill_fallback(inputs, q_lens, workspace_bytes=None, **overrides):
    outs = []
    for backend in (CAKE, TRTLLM_GEN):
        made = {name: make() for name, make in overrides.items()}
        _reset_routes()
        outs.append(_run_prefill(backend, inputs, q_lens, workspace_bytes, **made))
        if backend == CAKE:
            assert _routes_taken() == _routes("prefill", TRTLLM_GEN)
    _assert_same_bytes(*outs)


def _tail_only_q_len() -> int:
    """A query length whose tiles (256 rows of one head) fill 6 to 10 waves of
    one CTA per SM and leave a tail of at most half the SMs: the schedule that
    runs the complete waves on the plain kernel and splits only the tail."""
    sm_count = get_device_sm_count(torch.device("cuda"))
    for q_tiles in range(1, 129):
        waves, tail = divmod(q_tiles * NUM_QO_HEADS, sm_count)
        if 6 <= waves <= 10 and 0 < 2 * tail <= sm_count:
            return q_tiles * 256 - 100
    pytest.skip(f"no tail-only prefill schedule with {sm_count} SMs")


@pytest.mark.parametrize(
    "prefix_len,q_len,sharpness",
    [
        # The schedules named are those of 152 SMs.
        (8192, 1, 1.0),  # less than one wave of tiles: all split
        (8000, 300, 1.0),
        (9000, 1024, 4.0),
        (9000, 2048, 1.0),  # one complete wave and a split tail
        (4000, 4864, 4.0),  # complete waves only
        (0, 8193, 1.0),  # no cached prefix; a tail above half the SMs is not split
        (16384, 8192, 4.0),
        (131072, 512, 4.0),  # long KV
        (3000, 32768, 1.0),  # the longest query
    ],
)
def test_prefill_kernel(prefix_len, q_len, sharpness):
    torch.manual_seed(prefix_len + q_len)
    inputs = _make_inputs([prefix_len + q_len], q_len, sharpness)
    _check_prefill(inputs, q_len)


def test_prefill_causal_range():
    # Needles around the first query token, around the 256-row tile borders and
    # on the token after the request: every row must see exactly those up to
    # its own position.
    torch.manual_seed(1)
    prefix_len, q_len = 9001, 600
    rows = (-1, 0, 1, 2, 254, 255, 256, 257, 511, 512, 598, 599, 600)
    needles = [[prefix_len + row for row in rows]]
    inputs = _make_inputs([prefix_len + q_len], q_len, needles=needles)
    _check_prefill(inputs, q_len)


def test_prefill_tail_only_schedule():
    q_len = _tail_only_q_len()
    torch.manual_seed(q_len)
    # The prefill kernel serves only KV lengths above 8192 (fewer SMs mean a shorter q_len).
    prefix_len = max(2000, super_nvfp4.TRTLLM_GEN_MAX_KV_LEN + 1 - q_len)
    inputs = _make_inputs([prefix_len + q_len], q_len, sharpness=4.0)
    _check_prefill(inputs, q_len)


@pytest.mark.parametrize(
    "seq_lens,q_lens",
    [
        ([9000, 8500], [512, 300]),  # two requests
        ([8192], [512]),  # no KV longer than 8192 tokens
    ],
)
def test_prefill_shapes_routed_to_trtllm_gen(seq_lens, q_lens):
    torch.manual_seed(3)
    inputs = _make_inputs(seq_lens, sum(q_lens))
    _check_prefill_fallback(inputs, q_lens, 256 * MIB)


def test_prefill_workspace_bytes_is_what_the_kernel_needs():
    torch.manual_seed(4)
    inputs = _make_inputs([9000], 512)
    required = super_nvfp4.prefill_workspace_bytes(9000)
    _check_prefill(inputs, 512, workspace_buffer=_workspace(required))
    for too_small in (required - 1, 64 * MIB, MIB):
        _check_prefill_fallback(inputs, [512], too_small)


@pytest.mark.parametrize(
    "declined",
    [
        "workspace not 128-byte aligned",
        "max_kv_len above the page table",
        "K and V strides differ",
        "query not packed",
        "bmm1_scale zero",
    ],
)
def test_prefill_declined_calls_run_trtllm_gen(declined):
    torch.manual_seed(5)
    inputs = _make_inputs([9000], 512)
    required = super_nvfp4.prefill_workspace_bytes(9000 + PAGE_SIZE)
    k_cache, v_cache = inputs.kv_cache
    overrides = {
        "workspace not 128-byte aligned": dict(
            workspace_buffer=lambda: _workspace(required + 128)[16:-112]
        ),
        "max_kv_len above the page table": dict(
            max_kv_len=lambda: inputs.block_tables.size(1) * PAGE_SIZE + 1
        ),
        "K and V strides differ": dict(kv_cache=lambda: (_strided(k_cache), v_cache)),
        "query not packed": dict(query=lambda: _strided(inputs.query)),
        "bmm1_scale zero": dict(bmm1_scale=lambda: 0.0),
    }[declined]
    if declined == "K and V strides differ":
        _reset_routes()
        out = _run_prefill(
            CAKE,
            inputs,
            [512],
            required,
            kv_cache=overrides["kv_cache"](),
        )
        assert _routes_taken() == _routes("prefill", TRTLLM_GEN)
        _assert_same_bytes(out, _run_prefill(TRTLLM_GEN, inputs, [512], required))
    else:
        _check_prefill_fallback(inputs, [512], required, **overrides)


def test_prefill_features_run_trtllm_gen():
    torch.manual_seed(6)
    inputs = _make_inputs([9000], 512)
    _check_prefill_fallback(inputs, [512], window_left=lambda: 4096)


@pytest.mark.parametrize("variant", ["device scales", "NHD", "stacked kv_cache"])
def test_prefill_input_variants(variant):
    torch.manual_seed(7)
    inputs = _make_inputs([9000], 512, sharpness=4.0)
    if variant == "device scales":
        overrides = dict(
            bmm1_scale=torch.tensor([inputs.bmm1_scale], device="cuda"),
            bmm2_scale=torch.tensor([inputs.bmm2_scale], device="cuda"),
        )
    elif variant == "NHD":
        overrides = dict(
            kv_cache=tuple(x.transpose(1, 2).contiguous() for x in inputs.kv_cache),
            kv_cache_sf=tuple(
                x.transpose(1, 2).contiguous() for x in inputs.kv_cache_sf
            ),
            kv_layout="NHD",
        )
    else:
        overrides = dict(kv_cache=torch.stack(inputs.kv_cache, dim=1))
    out = _check_prefill(inputs, 512, **overrides)
    if variant != "device scales":  # those round bmm1_scale differently
        _assert_same_bytes(out, _run_prefill(CAKE, inputs, [512]))


@pytest.mark.parametrize("device_scales", [False, True])
@pytest.mark.parametrize("q_len", [300, 2048])
def test_prefill_cuda_graph(q_len, device_scales):
    torch.manual_seed(q_len)
    inputs = _make_inputs([9000 + q_len], q_len)
    # Nothing may be copied to or from the device during capture: with device
    # scales, the route must read them on the device (reads_device_scales).
    fixed = dict(
        workspace_buffer=_workspace(super_nvfp4.prefill_workspace_bytes(9000 + q_len)),
        max_kv_len=9000 + q_len,
        cum_seq_lens_q=_cumulative([q_len]),
        cum_seq_lens_kv=_cumulative([inputs.block_tables.size(1)]),
    )
    if device_scales:
        fixed["bmm1_scale"] = torch.tensor([inputs.bmm1_scale], device="cuda")
        fixed["bmm2_scale"] = torch.tensor([inputs.bmm2_scale], device="cuda")
    eager = _run_prefill(CAKE, inputs, [q_len], **fixed).clone()

    out = torch.empty_like(inputs.query)
    graph = torch.cuda.CUDAGraph()
    _reset_routes()
    with torch.cuda.graph(graph):
        _run_prefill(CAKE, inputs, [q_len], out=out, **fixed)
    assert _routes_taken() == _routes("prefill", "prefill")
    for _ in range(2):
        out.zero_()
        graph.replay()
        _assert_same_bytes(out, eager)
