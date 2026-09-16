"""NVFP4 paged-KV MSA decode on compute capability 10.0/10.3 -- all three bodies.

This route is a new capability, not a specialization of an existing one: NVFP4
K/V raised ``NotImplementedError`` on this architecture before it existed. So
the guard is a capability guard, and shape generality is a correctness
requirement rather than a nicety -- the device tests below deliberately sweep
batch sizes, KV lengths and query lengths well outside anything the kernel was
tuned on, including partial final blocks, empty requests, prime batch sizes and
every cluster-width boundary.

The guard orders its predicates semantics -> layout -> device, so the whole
semantic and layout surface is exercisable on a host with no GPU.

THREE BODIES, AND EVERY DEVICE TEST NAMES THE ONE IT EXERCISES
--------------------------------------------------------------
One call into ``msa_sparse_decode_attention`` can be served by any of three
kernel bodies, chosen inside the route:

    plan()                    -> CuTe-DSL           or   the C++ decode kernel
      +- selects_pinned_path  ->                         pinned or parametric

* ``CUTE`` -- ``flashinfer.msa_ops.cute_dsl.sparse_decode_nvfp4_sm100``. At the
  deployment geometry (tp 1: 64 query heads over 4 KV heads, top-k 16, page
  128, one decode token, causal) ``plan()`` declines 0 of 512 batch sizes, so
  under the production route this is what serves essentially every decode step
  of a tp-1 rank.
* ``PARAMETRIC`` -- the ``general::`` family of
  ``csrc/msa_decode_nvfp4_specialized.cu``. It is the tp 2 (32/2) and tp 4
  (16/1) SERVING path: at those ranks ``plan()`` serves 0 of 512, so this body
  runs on every decode step. It also takes every tp-1 call the CuTe-DSL body
  declines and the pinned envelope refuses.
* ``PINNED`` -- the ``geom::pinned`` family of the same translation unit. Rare:
  under the production route it needs the CuTe-DSL body to decline AND the
  pinned envelope to admit, which across this suite happens only at
  ``causal = 0`` coordinates, plus a forced decline in a test and the
  warm-failure path.

THIS FILE HAS NO AMBIENT ROUTE DEFAULT, AND THAT IS THE POINT
-------------------------------------------------------------
The route has no switch: it picks a body from the call's shape, and nothing
outside it can change that. A module-level default that forced one body would
make every sweep in this file test whichever kernel that default selects,
while reading as though it tested the route. So the backend is a PARAMETER,
not an ambient default:

* the ambient state of every test is the production route; a test that needs
  the C++ translation unit forces the CuTe-DSL body to decline through a
  monkeypatch of one module function, inside ``expect_backend`` only;
* every device test that dispatches names its body -- ``_call_on(CUTE, ...)``,
  ``_call_on(PINNED, ...)``, ``_call_on(PARAMETRIC, ...)``, or a
  ``@pytest.mark.parametrize("backend", ...)`` that puts the name in the test
  id;
* and ``_call_on`` READS THE DISPATCH COUNTERS ACROSS THE CALL and fails if a
  different body served it. A test that silently runs on the wrong backend
  fails here rather than passing, which is the property the pin destroyed and
  the whole reason these files are one file.
"""

import contextlib
import itertools
import math
import sys

import pytest
import torch

from flashinfer.msa_ops import _nvfp4_decode_sm100 as nvfp4
from flashinfer.msa_ops import msa_sparse_decode_attention
from flashinfer.msa_ops._blackwell_sm100 import MSASparseAttentionWorkspace

HEAD_DIM = 128
PAGE_SIZE = 128
NUM_QO_HEADS = 64
NUM_KV_HEADS = 4
TOPK = 16
MAX_BLOCKS = 128  # only the default row width used by these fixtures
SCALE_VEC = 16
DATA_DIM = HEAD_DIM // 2
SCALE_DIM = HEAD_DIM // SCALE_VEC
PAGE_BYTES = 2 * NUM_KV_HEADS * PAGE_SIZE * (DATA_DIM + SCALE_DIM)

# MiniMax-M3's per-rank head geometry under tensor parallelism, as vLLM shards
# it: num_heads = num_attention_heads // tp, num_kv_heads = max(1,
# num_key_value_heads // tp). tp 8 is absent on purpose -- it gives GQA group 8,
# which the PREFILL kernel's tile shape cannot serve, and half a capability with
# no fallback is worse than none. See the manifest's head_geometry section.
TP_GEOMETRIES = ((1, 64, 4), (2, 32, 2), (4, 16, 1))

# The single geometry the CuTe-DSL body is specialised for, in the argument
# shape its own plan() takes. Every row of TP_GEOMETRIES but the first is
# outside it, which is why tp 2 and tp 4 are served by the C++ parametric
# family on every decode step.
_GEOMETRY = dict(
    num_qo_heads=64,
    num_kv_heads=4,
    grp=16,
    topk=16,
    page_size=128,
    seqlen_q=1,
    causal=1,
)

K_GLOBAL_SCALE = 0.75
V_GLOBAL_SCALE = 0.85

_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _supported_device() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) in ((10, 0), (10, 3))


sm100_only = pytest.mark.skipif(
    not _supported_device(),
    reason="requires an MSA-capable compute capability 10.0/10.3 device",
)


# ---------------------------------------------------------------------------
# input construction
# ---------------------------------------------------------------------------
def _unit_rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)


def _swizzled_scale_position(t: torch.Tensor, s: torch.Tensor):
    """Where the cache writer puts the block scale of logical ``(t, s)``."""
    groups = SCALE_DIM // 4
    return (t // 4) * 4 + s // groups, (s % groups) * 4 + t % 4


def _quantize_nvfp4(x: torch.Tensor, global_scale: float):
    """(..., 128) -> (packed uint8 (..., 64), e4m3 scale bytes (..., 8)).

    ``sf = e4m3(amax16 / (6 * global_scale))`` and ``q = e2m1(x / (sf *
    global_scale))``, i.e. the dequant is ``e2m1 * float(sf) * global_scale``.
    """
    sf_scale = 1.0 / float(global_scale)
    grouped = x.float().reshape(*x.shape[:-1], HEAD_DIM // SCALE_VEC, SCALE_VEC)
    amax = grouped.abs().amax(dim=-1)
    sf = (sf_scale * amax / 6.0).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    out_scale = torch.where(
        sf_f > 0, sf_scale / sf_f.clamp(min=1e-30), torch.zeros_like(sf_f)
    )
    y = grouped * out_scale.unsqueeze(-1)
    bounds = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=x.device
    )
    magnitude = torch.bucketize(y.abs().contiguous(), bounds).to(torch.uint8)
    codes = (magnitude | ((y < 0).to(torch.uint8) << 3)).reshape(
        *x.shape[:-1], HEAD_DIM
    )
    packed = (codes[..., 0::2] & 0x0F) | ((codes[..., 1::2] & 0x0F) << 4)
    return packed.to(torch.uint8), sf.view(torch.uint8)


def _page_bytes(num_kv_heads: int = NUM_KV_HEADS) -> int:
    """The planar page for ``num_kv_heads`` heads. It SHRINKS with the rank."""
    return 2 * num_kv_heads * PAGE_SIZE * (DATA_DIM + SCALE_DIM)


def _page_views(pool: torch.Tensor, num_pages: int, num_kv_heads: int = NUM_KV_HEADS):
    page_bytes = _page_bytes(num_kv_heads)
    data_shape = (num_pages, num_kv_heads, PAGE_SIZE, DATA_DIM)
    scale_shape = (num_pages, num_kv_heads, PAGE_SIZE, SCALE_DIM)
    data_stride = (page_bytes, PAGE_SIZE * DATA_DIM, DATA_DIM, 1)
    scale_stride = (page_bytes, PAGE_SIZE * SCALE_DIM, SCALE_DIM, 1)
    k_scale_offset = num_kv_heads * PAGE_SIZE * DATA_DIM
    v_data_offset = k_scale_offset + num_kv_heads * PAGE_SIZE * SCALE_DIM
    v_scale_offset = v_data_offset + k_scale_offset
    return (
        torch.as_strided(pool, data_shape, data_stride, 0),
        torch.as_strided(pool, scale_shape, scale_stride, k_scale_offset),
        torch.as_strided(pool, data_shape, data_stride, v_data_offset),
        torch.as_strided(pool, scale_shape, scale_stride, v_scale_offset),
    )


def _build_inputs(
    batch,
    seq_lengths,
    device,
    seed=0,
    num_pages=None,
    topk=TOPK,
    num_kv_heads=NUM_KV_HEADS,
    num_qo_heads=NUM_QO_HEADS,
):
    """One decode step against a planar NVFP4 page pool, vLLM's layout."""
    generator = torch.Generator(device=device).manual_seed(seed)
    seqused_k = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    assert int(seqused_k.numel()) == batch
    blocks = (seqused_k.long() + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages = num_pages or _page_count(seq_lengths)

    page_table = torch.full((batch, MAX_BLOCKS), -1, dtype=torch.int32, device=device)
    permutation = torch.randperm(num_pages, generator=generator, device=device).to(
        torch.int32
    )
    cursor = 0
    for request, count in enumerate(blocks.tolist()):
        page_table[request, :count] = permutation[cursor : cursor + count]
        cursor += count

    pool = torch.zeros(
        num_pages * _page_bytes(num_kv_heads), dtype=torch.uint8, device=device
    )
    k_data, k_scale, v_data, v_scale = _page_views(pool, num_pages, num_kv_heads)

    keys = _unit_rms(
        torch.randn(
            num_pages,
            num_kv_heads,
            PAGE_SIZE,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    )
    values = torch.randn(
        num_pages,
        num_kv_heads,
        PAGE_SIZE,
        HEAD_DIM,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    packed_k, sf_k = _quantize_nvfp4(keys.to(torch.bfloat16), K_GLOBAL_SCALE)
    packed_v, sf_v = _quantize_nvfp4(values.to(torch.bfloat16), V_GLOBAL_SCALE)
    k_data.copy_(packed_k)
    v_data.copy_(packed_v)
    k_scale.copy_(sf_k)  # linear

    tokens = torch.arange(PAGE_SIZE, device=device).unsqueeze(1)
    groups = torch.arange(SCALE_DIM, device=device).unsqueeze(0)
    swizzled_t, swizzled_s = _swizzled_scale_position(tokens, groups)
    swizzled = torch.zeros_like(sf_v)
    swizzled[:, :, swizzled_t.reshape(-1), swizzled_s.reshape(-1)] = sf_v[
        :,
        :,
        tokens.expand(-1, SCALE_DIM).reshape(-1),
        groups.expand(PAGE_SIZE, -1).reshape(-1),
    ]
    v_scale.copy_(swizzled)

    q = _unit_rms(
        torch.randn(
            batch,
            num_qo_heads,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)

    # msa_topk_select on this architecture takes a single batch-wide valid-page
    # bound, so every request is offered the same block range; short requests
    # therefore receive ids past their own extent, and the kernel is what has
    # to reject them.
    bound = int(blocks.max())
    indices = torch.full(
        (num_kv_heads, batch, topk), -1, dtype=torch.int32, device=device
    )
    keep = min(topk, bound)
    for request in range(batch):
        for head in range(num_kv_heads):
            selection = torch.randperm(bound, generator=generator, device=device)[:keep]
            indices[head, request, :keep] = selection.sort().values.to(torch.int32)

    return dict(
        q=q,
        k=k_data,
        v=v_data,
        k_scale=k_scale,
        v_scale=v_scale,
        q2k_indices=indices,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=1,
        causal=True,
        softmax_scale=HEAD_DIM**-0.5,
        k_global_scale=K_GLOBAL_SCALE,
        v_global_scale=V_GLOBAL_SCALE,
    )


def _surface_kwargs(inputs):
    return dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        q2k_indices=inputs["q2k_indices"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        cu_seqlens_k=None,
        seqlen_q=inputs["seqlen_q"],
        causal=inputs["causal"],
        return_softmax_lse=False,
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        q_offset=None,
        force_fused=None,
    )


def _call(inputs, workspace=None, out=None):
    return msa_sparse_decode_attention(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["q2k_indices"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        seqlen_q=inputs["seqlen_q"],
        causal=inputs["causal"],
        softmax_scale=inputs["softmax_scale"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        workspace=workspace,
        out=out,
    )


def _reference(inputs):
    out = torch.empty_like(inputs["q"])
    return nvfp4.reference(
        q=inputs["q"],
        k_data=inputs["k"],
        v_data=inputs["v"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        q2k_indices=inputs["q2k_indices"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
        out=out,
        seqlen_q=inputs["seqlen_q"],
        causal=inputs["causal"],
    )


def _assert_peer(actual, expected, *, min_cosine=0.99, max_rel_fro=0.06):
    """The tolerance family the FP8/FP4 policy sets for this op.

    Cosine alone is not enough: with a BF16 output quantizing FP32 accumulator
    error, the cosine bands of a legal and an illegal operand chain overlap, so
    a relative Frobenius bound is carried alongside it.
    """
    assert torch.isfinite(actual).all()
    a = actual.float().reshape(-1)
    b = expected.float().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    rel = ((a - b).norm() / b.norm().clamp(min=1e-30)).item()
    assert cosine >= min_cosine, f"cosine {cosine}"
    assert rel <= max_rel_fro, f"rel_fro {rel}"


def _dispatch_count():
    return nvfp4.msa_decode_nvfp4_specialized_stats()["dispatch_count"]


# ---------------------------------------------------------------------------
# THE BACKEND CONTRACT
#
# One route, three kernel bodies. A test that does not say which one it is
# exercising is not testing a kernel, it is testing whichever kernel the route
# happens to pick -- and this file learned that the expensive way (see the
# module docstring). So the body is a NAMED PARAMETER of every dispatching
# test, and `_call_on` proves from the route's own dispatch counters that the
# call went where the test asked.
# ---------------------------------------------------------------------------
CUTE = "cute"
PINNED = "pinned"
PARAMETRIC = "parametric"
BACKENDS = (CUTE, PINNED, PARAMETRIC)

# Which counter each body moves, as the PUBLIC stats dict reports it -- the
# same numbers a consumer's capability probe and the kernel-level A/B harness
# read, rather than a private global this file could drift away from.
_BACKEND_COUNTER = {
    CUTE: ("specialised_route", "dispatch_count"),
    PINNED: ("pinned_dispatch_count",),
    PARAMETRIC: ("general_dispatch_count",),
}

# What a test does to REACH each body. It does not make the body SELECTED --
# the call's own geometry does that, which is exactly why every call below is
# checked against the counters afterwards. `auto` is the production route
# untouched, which is what the CuTe-DSL body needs; `pingpong` forces the C++
# translation unit by making the CuTe-DSL body decline the call, and the
# pinned predicate then picks the family. This is a test seam -- a monkeypatch
# of one module function -- not a setting the route exposes; the route has no
# override.
_BACKEND_ROUTE = {CUTE: "auto", PINNED: "pingpong", PARAMETRIC: "pingpong"}
_FORCED_DECLINE = "the test forced this call to the C++ translation unit"


def _nested(payload, path):
    for key in path:
        payload = payload[key]
    return payload


def _counters():
    stats = nvfp4.msa_decode_nvfp4_specialized_stats()
    return {name: _nested(stats, path) for name, path in _BACKEND_COUNTER.items()}


@contextlib.contextmanager
def expect_backend(backend, monkeypatch, *, route=None, calls=1, device=None):
    """Run the block on ``backend`` -- and FAIL if another body served it.

    This is the whole point of merging the route's tests and the ping-pong
    kernel's tests into one file. Both suites used to carry an ambient default
    (one pinned to ``pingpong``, one to ``auto``), so a test could be
    re-pointed at the other kernel by a change somewhere else entirely and
    would keep passing while measuring nothing it claimed to. Naming the body
    is not enough on its own -- the name has to be CHECKED -- so the dispatch
    counters are read across the block and a mismatch is an error, not a
    detail.

    ``route`` defaults to what makes ``backend`` reachable (see
    ``_BACKEND_ROUTE``); a test that wants to prove the PRODUCTION route
    reaches a body passes ``route="auto"`` explicitly and still gets the
    assertion.

    ``device`` is warmed first when given. ``warm()`` launches BOTH C++
    families for itself and the decode hook calls it on every eager dispatch,
    so the first dispatch on a cold device moves two family counters and the
    block would be unreadable. Warming before the snapshot removes that from
    the measurement instead of tolerating it.
    """

    assert backend in BACKENDS, f"{backend!r} is not one of {list(BACKENDS)}"
    choice = _BACKEND_ROUTE[backend] if route is None else route
    assert choice in ("auto", "pingpong"), choice
    if choice == "pingpong":
        monkeypatch.setattr(
            nvfp4, "specialised_route_reason", lambda **_kwargs: _FORCED_DECLINE
        )
    if device is not None:
        nvfp4.warm(device)
    before = _counters()
    yield
    after = _counters()
    moved = {name for name in BACKENDS if after[name] > before[name]}
    assert moved == {backend}, (
        f"this test asked for the {backend!r} body and "
        + (
            f"the {sorted(moved)} body/bodies served the call(s)"
            if moved
            else "nothing dispatched at all"
        )
        + f" (route={choice!r}). A test that silently runs on a "
        f"kernel it did not ask for is vacuous however green it looks, so "
        f"this is a failure."
    )
    assert after[backend] - before[backend] == calls, (
        f"expected {calls} dispatch(es) on {backend!r}, saw "
        f"{after[backend] - before[backend]}"
    )


def _call_on(backend, inputs, monkeypatch, *, route=None, **kwargs):
    """One call, on ``backend``, asserted. See :func:`expect_backend`."""

    with expect_backend(backend, monkeypatch, route=route, device=inputs["q"].device):
        result = _call(inputs, **kwargs)
    return result


def _page_count(seq_lengths):
    """Pages ``_build_inputs`` allocates for ``seq_lengths``. ONE copy.

    The pinned predicate reads the page count, so the expected C++ family of a
    sweep row is a function of it; deriving it here and using the same function
    in ``_build_inputs`` is what keeps the expectation and the fixture from
    drifting apart.
    """
    return max(1, sum((int(s) + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lengths))


def cpp_family(
    seq_lengths,
    *,
    batch=None,
    seqlen_q=1,
    topk=TOPK,
    max_blocks=MAX_BLOCKS,
    num_qo_heads=NUM_QO_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    num_pages=None,
):
    """Which C++ family the route's OWN predicate selects for this shape.

    Derived, not tabulated: a table here would be a second copy of the pinned
    envelope and would go stale the first time the envelope moved. The device
    test then asserts the counters agree with this, so a drift between the
    prediction and the dispatch is a failing test rather than a silent
    re-pointing.
    """
    batch = len(seq_lengths) if batch is None else batch
    reason = nvfp4.pinned_path_reason(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        topk=topk,
        max_blocks=max_blocks,
        seqlen_q=seqlen_q,
        total_q=batch * seqlen_q,
        num_pages=_page_count(seq_lengths) if num_pages is None else num_pages,
    )
    return PARAMETRIC if reason is not None else PINNED


def bodies_for(
    seq_lengths,
    *,
    batch=None,
    seqlen_q=1,
    topk=TOPK,
    causal=True,
    max_blocks=MAX_BLOCKS,
    num_qo_heads=NUM_QO_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    num_pages=None,
):
    """``(backend, route)`` for every body that can serve this coordinate.

    The CuTe-DSL implementation is specialised for ONE geometry -- 64 query
    heads over 4 KV heads, top-k 16, page 128, one decode token, causal -- so
    it serves a coordinate only there, and everything else already reaches the
    C++ translation unit under the PRODUCTION route. Where both can serve, the
    coordinate is swept on both: that is the coverage the old module-level pin
    to ``pingpong`` silently removed from every sweep in this file.

    A PREDICTION, and ``expect_backend`` checks it against the counters, so a
    wrong entry here fails a test rather than quietly re-pointing one.
    ``test_the_body_prediction_agrees_with_the_implementations_own_predicate``
    checks the CuTe half against ``specialised_reason`` itself.
    """

    cpp = cpp_family(
        seq_lengths,
        batch=batch,
        seqlen_q=seqlen_q,
        topk=topk,
        max_blocks=max_blocks,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
    )
    cute_serves = (
        (num_qo_heads, num_kv_heads) == (NUM_QO_HEADS, NUM_KV_HEADS)
        and topk == _GEOMETRY["topk"]
        and seqlen_q == _GEOMETRY["seqlen_q"]
        and bool(causal) == bool(_GEOMETRY["causal"])
    )
    if cute_serves:
        return [(CUTE, "auto"), (cpp, "pingpong")]
    # Not the specialised geometry, so the production route reaches the C++
    # translation unit on its own -- no override, and that is the serving path
    # at tensor-parallel ranks 2 and 4.
    return [(cpp, "auto")]


def sweep_params(rows, *, tag, **kwargs):
    """``(batch, seq_lengths, backend, route)`` for a shape sweep x its bodies.

    The backend is in the TEST ID, so ``-k`` selects a body and a failure names
    one without anybody reading the file.
    """

    params = []
    for index, (batch, seq_lengths) in enumerate(rows):
        for backend, route in bodies_for(seq_lengths, batch=batch, **kwargs):
            params.append(
                pytest.param(
                    batch,
                    seq_lengths,
                    backend,
                    route,
                    id=f"{tag}{index:02d}-b{batch}-{backend}",
                )
            )
    return params


# ---------------------------------------------------------------------------
# host-only guard tests
# ---------------------------------------------------------------------------
@pytest.fixture
def cpu_inputs():
    return _build_inputs(4, [1024, 300, 9000, 8192], torch.device("cpu"))


@pytest.fixture(scope="module")
def impl():
    """The CuTe-DSL implementation module, or a skip if it is unavailable."""

    module = nvfp4._specialised_module()
    if module is None:
        pytest.skip(
            f"the specialised implementation is unavailable: "
            f"{nvfp4._specialised_import_error}"
        )
    return module


def test_allowlist_and_stats_agree_with_the_module_constants():
    stats = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert stats["allowlist_fields"] == list(nvfp4._WORKLOAD_FIELDS)
    # Model geometry only, and top-k is no longer part of it. Batch size, KV
    # length, block-table width, query length, causality, top-k and the
    # selection tensor's outer strides are parametric and must NOT appear here:
    # narrowing them would make the capability unreachable, not route it
    # somewhere else.
    # stats() reports sorted(_load_allowlist()), so the rows come back
    # ascending by query-head count -- tp 4, tp 2, tp 1.
    assert stats["allowlist"] == [
        [16, 1, 128, 128],
        [32, 2, 128, 128],
        [64, 4, 128, 128],
    ]
    assert stats["allowlist_rows"] == 3
    assert set(stats["parametric_axes"]) == {
        "batch_size",
        "seqlen_q",
        "seqused_k",
        "max_blocks",
        "causal",
        "topk",
        "q2k_indices_outer_strides",
    }
    assert stats["topk_range"] == [1, nvfp4._MAX_TOPK]
    # One translation unit, thirty precompiled instantiations across the two
    # families, and a compile cache keyed only by the architecture target: no
    # call shape can trigger a build, which is what makes CUDA graph capture
    # safe by construction.
    assert stats["distinct_kernels_for_allowlist"] == 30
    assert len(stats["kernel_instantiations"]) == 30
    assert sum(n.startswith("pinned_") for n in stats["kernel_instantiations"]) == 6
    assert stats["compile_cache_key"] == "(compute capability target,)"
    assert stats["precompiled"] is True
    assert stats["supported_compute_capability"] == [(10, 0), (10, 3)]


def test_the_cuda_graph_contract_is_stated_where_a_consumer_can_read_it():
    """A serving engine must not have to infer this from a raised exception."""
    stats = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert stats["cuda_graph"]["requires_workspace"] is False
    assert nvfp4.capture_requires_workspace() is False
    assert stats["cuda_graph"]["requires_eager_warm"] is True
    assert stats["cuda_graph"]["warm_entry_point"].endswith(
        "msa_decode_nvfp4_specialized_warmup"
    )


def test_the_warm_dummy_is_a_legal_call_for_this_route():
    """warm() must exercise the shipped guard's surface, not a neighbour of it."""
    inputs = nvfp4._warm_inputs(torch.device("cpu"))
    kwargs = dict(inputs)
    for key in ("out", "softmax_scale"):
        kwargs.pop(key)
    kwargs.update(
        cu_seqlens_k=None,
        return_softmax_lse=False,
        q_offset=None,
        force_fused=None,
    )
    # "q must be a CUDA tensor" is the guard's LAST predicate: reaching it
    # means the whole semantic and layout surface admitted the call.
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"


def test_the_layout_of_the_test_inputs_matches_the_production_page_map(cpu_inputs):
    k, k_scale = cpu_inputs["k"], cpu_inputs["k_scale"]
    v, v_scale = cpu_inputs["v"], cpu_inputs["v_scale"]
    assert k.stride() == (PAGE_BYTES, PAGE_SIZE * DATA_DIM, DATA_DIM, 1)
    assert k_scale.stride() == (PAGE_BYTES, PAGE_SIZE * SCALE_DIM, SCALE_DIM, 1)
    assert not k.is_contiguous() and not k_scale.is_contiguous()
    base = k.data_ptr()
    assert k_scale.data_ptr() - base == 32768
    assert v.data_ptr() - base == 36864
    assert v_scale.data_ptr() - base == 69632


def test_a_well_formed_call_reaches_the_device_check(cpu_inputs):
    """Everything except device residency must pass on a host-only tensor."""
    reason = nvfp4.check_surface(**_surface_kwargs(cpu_inputs))
    assert reason == "q must be a CUDA tensor"


@pytest.mark.parametrize(
    "override, fragment",
    [
        ({"seqlen_q": 3}, "q rows"),
        ({"seqlen_q": 0}, "seqlen_q must be positive"),
        ({"return_softmax_lse": True}, "softmax LSE"),
        ({"q_offset": 3}, "q_offset"),
        ({"k_global_scale": None}, "k_global_scale"),
        ({"k_scale": None}, "k_scale"),
        ({"cu_seqlens_k": True}, "cu_seqlens_k"),
    ],
)
def test_unsupported_semantics_are_rejected_before_any_device_query(
    cpu_inputs, override, fragment
):
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs.update(override)
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None and fragment in reason


@pytest.mark.parametrize(
    "override",
    [
        {"causal": False},
        {"force_fused": True},
        {"force_fused": False},
    ],
)
def test_parametric_options_are_admitted(cpu_inputs, override):
    """Only the device predicate may reject a call on a parametric axis."""
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs.update(override)
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"


def test_a_narrower_block_table_is_admitted(cpu_inputs):
    """``max_blocks`` follows max_model_len; it is a runtime argument."""
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs["page_table"] = cpu_inputs["page_table"][:, :64].contiguous()
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"


def test_a_non_bf16_query_is_rejected(cpu_inputs):
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs["q"] = cpu_inputs["q"].float()
    assert "bfloat16" in nvfp4.check_surface(**kwargs)


def test_a_dense_repack_of_the_page_is_rejected(cpu_inputs):
    """A ``.contiguous()`` copy has the right shape and dtype and wrong strides.

    The kernel derives every byte address from the packed page stride, so a
    view that was silently densified must not reach it.
    """
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs["k"] = cpu_inputs["k"].contiguous()
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None and "stride" in reason


def test_scales_from_a_different_allocation_are_rejected(cpu_inputs):
    """Shape, dtype and stride cannot see the (4, 4) V-scale swizzle.

    Only the byte offset between the four base pointers ties them to one page.
    """
    kwargs = _surface_kwargs(cpu_inputs)
    num_pages = int(cpu_inputs["k"].shape[0])
    other = torch.zeros(num_pages * PAGE_BYTES, dtype=torch.uint8)
    kwargs["v_scale"] = torch.as_strided(
        other, cpu_inputs["v_scale"].shape, cpu_inputs["v_scale"].stride(), 0
    )
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None and "region of the same packed page" in reason


def test_a_block_table_with_the_wrong_row_count_is_rejected(cpu_inputs):
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs["page_table"] = cpu_inputs["page_table"][:2].contiguous()
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None and "page_table" in reason


@pytest.mark.parametrize("topk", [1, 2, 4, 8, 12, 16, 24, 31, 32])
def test_every_top_k_up_to_the_ballot_width_is_admitted(cpu_inputs, topk):
    """top-k stopped being a compile-time constant; the guard has to follow.

    This test used to assert the opposite -- that anything but 16 was rejected.
    It is inverted deliberately, and the boundary below is what replaces it.
    """
    kwargs = _surface_kwargs(cpu_inputs)
    total_q = int(cpu_inputs["q"].shape[0])
    kwargs["q2k_indices"] = torch.zeros(
        (NUM_KV_HEADS, total_q, topk), dtype=torch.int32
    )
    # Everything else about this fixture is a CPU tensor, so the architecture
    # conjunct is what stops it -- i.e. the selection surface admitted it.
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"


def test_the_top_k_ceiling_is_executed_at_its_boundary(cpu_inputs):
    """Admits at 32, refuses at 33, BY NAME. The bound is the ballot width."""
    kwargs = _surface_kwargs(cpu_inputs)
    total_q = int(cpu_inputs["q"].shape[0])

    kwargs["q2k_indices"] = torch.zeros((NUM_KV_HEADS, total_q, 32), dtype=torch.int32)
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"

    kwargs["q2k_indices"] = torch.zeros((NUM_KV_HEADS, total_q, 33), dtype=torch.int32)
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None
    assert "top-k must be in [1, 32]" in reason and "got 33" in reason

    kwargs["q2k_indices"] = torch.zeros((NUM_KV_HEADS, total_q, 0), dtype=torch.int32)
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None and "top-k must be in [1, 32]" in reason


def test_a_strided_selection_view_is_admitted_and_a_gappy_one_is_not(cpu_inputs):
    """The whole point of the stride change, on the host side.

    A token-major buffer transposed to head-major is exactly what the MSA
    indexer produces; it is non-contiguous and must be ADMITTED. A view whose
    innermost dimension is not dense is the one thing the kernels cannot read,
    and it must be refused by name.
    """
    kwargs = _surface_kwargs(cpu_inputs)
    total_q = int(cpu_inputs["q"].shape[0])
    token_major = torch.zeros((total_q + 5, NUM_KV_HEADS, TOPK), dtype=torch.int32)

    strided = token_major[:total_q].transpose(0, 1)
    assert not strided.is_contiguous() and strided.stride(2) == 1
    kwargs["q2k_indices"] = strided
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"

    # ...and the one layout that is genuinely unreadable.
    gappy = torch.zeros((NUM_KV_HEADS, total_q, 2 * TOPK), dtype=torch.int32)[:, :, ::2]
    assert gappy.stride(2) == 2
    kwargs["q2k_indices"] = gappy
    reason = nvfp4.check_surface(**kwargs)
    assert reason is not None and "innermost" in reason


def test_a_declined_call_says_WHICH_axis_declined_it(cpu_inputs):
    """The blanket message was false, and falseness is the whole defect.

    `NVFP4 K/V is not supported by MSA on compute capability 10.0/10.3` is what
    this architecture raised BEFORE the route existed, and the route is an
    exception carved in front of it -- so a decline lands back on the original
    blanket refusal and the operator is told the capability is missing when in
    fact one axis of one call is out of range. There is no other implementation
    of this operation over an NVFP4 cache, so that message is also the last
    thing they get.
    """
    from flashinfer.msa_ops import _blackwell_sm100

    inputs = dict(cpu_inputs)
    inputs["q2k_indices"] = torch.zeros(
        (NUM_KV_HEADS, int(inputs["q"].shape[0]), 33), dtype=torch.int32
    )
    with pytest.raises(NotImplementedError) as excinfo:
        _blackwell_sm100.blackwell_msa_sparse_decode_attention(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["q2k_indices"],
            page_table=inputs["page_table"],
            seqused_k=inputs["seqused_k"],
            seqlen_q=1,
            causal=True,
            softmax_scale=inputs["softmax_scale"],
            k_scale=inputs["k_scale"],
            v_scale=inputs["v_scale"],
            k_global_scale=inputs["k_global_scale"],
            v_global_scale=inputs["v_global_scale"],
        )
    text = str(excinfo.value)
    # the axis, the bound and the offending VALUE, all three
    assert "top-k" in text and "[1, 32]" in text and "33" in text, text
    # ...and the correction of the false claim
    assert "IS supported on this architecture" in text, text
    assert "no other implementation" in text, text


def test_guard_rejections_are_observable(cpu_inputs):
    before = sum(
        nvfp4.msa_decode_nvfp4_specialized_stats()["guard_rejections"].values()
    )
    kwargs = _surface_kwargs(cpu_inputs)
    kwargs["seqlen_q"] = 4
    nvfp4.check_surface(**kwargs)
    after = sum(nvfp4.msa_decode_nvfp4_specialized_stats()["guard_rejections"].values())
    assert after == before + 1


# ---------------------------------------------------------------------------
# device tests
# ---------------------------------------------------------------------------
# The coordinates this kernel was tuned on. Kept so that a regression here is
# attributable, but they are not the interesting rows.
# ---------------------------------------------------------------------------
# host-only tests for the pinned instantiation family
#
# The kernel carries a geometry-pinned family beside the parametric one. It is
# a SPEED decision: both compute the same function and the pinned envelope is
# the deployment's, so the interesting failure is not "wrong answer" but "the
# pin quietly stopped matching the deployment and nothing failed".
# ---------------------------------------------------------------------------
_DEPLOYMENT = dict(
    num_qo_heads=NUM_QO_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    page_size=PAGE_SIZE,
    topk=TOPK,
    max_blocks=MAX_BLOCKS,
    seqlen_q=1,
    total_q=16,
    num_pages=2048,
)


def test_the_pinned_envelope_is_the_deployment():
    assert nvfp4.pinned_path_reason(**_DEPLOYMENT) is None
    assert nvfp4.selects_pinned_path(**_DEPLOYMENT) is True
    envelope = nvfp4.msa_decode_nvfp4_specialized_stats()["pinned_path_envelope"]
    for axis, value in envelope.items():
        assert _DEPLOYMENT[axis] == value, axis


@pytest.mark.parametrize(
    "axis,value",
    [
        ("num_qo_heads", 32),
        ("num_kv_heads", 8),
        ("head_dim", 64),
        ("page_size", 64),
        ("topk", 32),
        ("max_blocks", 129),
        ("seqlen_q", 2),
    ],
)
def test_each_axis_of_the_pin_is_load_bearing(axis, value):
    """A rule nothing can violate is not a rule: each axis is moved alone."""
    coordinate = dict(_DEPLOYMENT, **{axis: value})
    reason = nvfp4.pinned_path_reason(**coordinate)
    assert reason is not None and axis.split("_")[-1] in reason.replace(
        "block-table width", "blocks"
    )


def test_a_batch_of_32_over_a_toy_page_pool_is_outside_the_pinned_envelope():
    """The eval-shaped small-pool case, which serving never produces.

    Batch 32 is now outside the envelope for a SECOND and stronger reason --
    it takes the clustered multi-chunk instantiation, which is not
    deterministic (see test_the_pin_never_admits_a_clustered_multichunk_shape)
    -- so the small-pool rule is asserted at a batch the new rule does not
    already cover.
    """
    assert nvfp4.pinned_path_reason(**dict(_DEPLOYMENT, total_q=32, num_pages=64))
    assert (
        nvfp4.pinned_path_reason(**dict(_DEPLOYMENT, total_q=16, num_pages=2048))
        is None
    )


def test_the_pin_never_admits_a_clustered_multichunk_shape():
    """The correctness rule the pinned family now carries, over every batch.

    `geom::pinned`'s chunk loop runs `kPages / kChunk` times with
    `kPages = topk / split`, and the `c != 0` iterations read-modify-write the
    running FP32 numerator in Tensor Memory. Inside a cluster that store is
    observed only partially: GB300, batch 32 / seq 8192, 3,000 trials per
    cell, twice --

        cluster 2, two chunks    3.37% / 3.40%      the same code, no rescale
        cluster 2, four chunks  19.67% / 18.73%     0.07% / 0.00%
        cluster 4, two chunks    7.87%
        NO cluster, four chunks  0.00%   (0 of 3,000, twice)
        NO cluster, eight chunks 0.00%   (0 of 3,000, twice)

    -- so no shape may take a clustered instantiation that runs more than one
    chunk. This walks every batch the dispatch can reach, in both page-pool
    regimes, against the module's own copy of the launcher's arithmetic.
    """
    admitted_clustered_multichunk = []
    rejected = []
    for total_q in range(1, 513):
        for num_pages in (64 * total_q, 8 * total_q):
            coordinate = dict(_DEPLOYMENT, total_q=total_q, num_pages=num_pages)
            reason = nvfp4.pinned_path_reason(**coordinate)
            multichunk = nvfp4.pinned_is_clustered_multichunk(total_q, num_pages)
            if reason is None and multichunk:
                admitted_clustered_multichunk.append((total_q, num_pages))
            if multichunk:
                rejected.append(total_q)
    assert not admitted_clustered_multichunk, admitted_clustered_multichunk
    # ...and the rule bites exactly where the defect was measured: the split-2
    # band, total_q 17..32, and nowhere else. A rule that rejected everything
    # would also satisfy the assertion above.
    assert sorted(set(rejected)) == list(range(17, 33))


def test_missing_the_pin_is_a_speed_statement_not_a_refusal(cpu_inputs):
    """A shape outside the pinned envelope must still be ADMITTED.

    Narrowing the guard to the pinned envelope would convert a slower
    instantiation into a NotImplementedError mid-serve.
    """
    kwargs = _surface_kwargs(cpu_inputs)
    wider = torch.zeros(
        cpu_inputs["page_table"].shape[0], MAX_BLOCKS + 1, dtype=torch.int32
    )
    wider[:, :MAX_BLOCKS] = cpu_inputs["page_table"]
    kwargs["page_table"] = wider
    assert nvfp4.check_surface(**kwargs) == "q must be a CUDA tensor"
    assert nvfp4.pinned_path_reason(**dict(_DEPLOYMENT, max_blocks=MAX_BLOCKS + 1))


def test_the_dispatch_hands_its_decision_to_the_binding(cpu_inputs):
    """The binding cross-checks it, so it must actually be sent."""
    sent = []

    class _StubModule:
        @staticmethod
        def msa_decode_nvfp4_specialized(*args):
            sent.append(args[-1])

    inputs = dict(cpu_inputs)
    inputs["out"] = torch.empty_like(inputs["q"])
    inputs.pop("k_global_scale_tensor", None)
    call = dict(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        k_scale=inputs["k_scale"],
        v_scale=inputs["v_scale"],
        q2k_indices=inputs["q2k_indices"],
        page_table=inputs["page_table"],
        seqused_k=inputs["seqused_k"],
        out=inputs["out"],
        seqlen_q=inputs["seqlen_q"],
        causal=inputs["causal"],
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
        v_global_scale=inputs["v_global_scale"],
    )
    nvfp4._dispatch(_StubModule, **call)
    assert sent == [1], "the deployment shape must take the pinned family"

    wider = torch.zeros(call["page_table"].shape[0], MAX_BLOCKS + 1, dtype=torch.int32)
    wider[:, :MAX_BLOCKS] = call["page_table"]
    nvfp4._dispatch(_StubModule, **dict(call, page_table=wider))
    assert sent == [1, 0], "a wider block table must take the parametric family"


_TUNED = [
    (8, [8192] * 8),
    (16, [1024] * 16),
    (16, [8192] * 16),
    (32, [1024] * 32),
    (32, [8192] * 32),
    (64, [8192] * 64),
    (128, [8192] * 128),
]

# Coordinates it was never tuned on. Every cluster-width boundary (batch * 4
# tiles crossing 32, 64 and 128), partial final blocks, empty and single-token
# requests, prime batch sizes, and lengths between the two tuned points.
_UNTUNED = [
    (1, [8192]),
    (1, [1]),
    (1, [127]),
    (2, [9000, 130]),
    (3, [129, 255, 1023]),
    (5, [0, 4096, 0, 17, 8192]),
    (7, [1024 * (i + 1) for i in range(7)]),
    (8, [8192 + 97 * i for i in range(8)]),
    (9, [8192] * 9),
    (11, [2049, 2048, 2047, 1, 16383, 512, 640, 768, 896, 1024, 1152]),
    (16, [8192] * 15 + [3]),
    (17, [4096 + 11 * i for i in range(17)]),
    (33, [1024 + 7 * i for i in range(33)]),
    (37, [6000] * 37),
    (129, [2000 + 3 * i for i in range(129)]),
]


@sm100_only
@pytest.mark.parametrize(
    "batch, seq_lengths, backend, route", sweep_params(_TUNED, tag="tuned")
)
def test_matches_the_reference_on_the_tuned_coordinates(
    batch, seq_lengths, backend, route, monkeypatch
):
    """Every tuned coordinate, on EVERY body that can serve it.

    It used to be on whichever body the module-level pin selected, which was
    the C++ one -- so the tuned coordinates were never once checked against
    the implementation that actually serves them in production.
    """
    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=batch)
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


@sm100_only
@pytest.mark.parametrize(
    "batch, seq_lengths, backend, route", sweep_params(_UNTUNED, tag="untuned")
)
def test_matches_the_reference_outside_the_tuned_coordinates(
    batch, seq_lengths, backend, route, monkeypatch
):
    """Shape generality is a correctness requirement on this route.

    There is no other NVFP4 MSA decode implementation on this architecture, so
    a shape the kernel gets wrong has nowhere to fall back to; it has to be
    right everywhere the guard admits it -- and on every body that can be the
    one admitting it.
    """
    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=batch + 1)
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


# ---------------------------------------------------------------------------
# tensor parallelism: the per-rank head geometry
# ---------------------------------------------------------------------------
_TP_ROWS = [(8, [8192] * 8), (32, [8192] * 32)]
_TP_RAGGED = [9000, 130, 8500, 0, 9000, 1, 8192, 127, 4096, 255, 3]


def tp_params(rows, *, tag):
    """``(tp, qo, kv, batch, seq_lengths, backend, route)`` over the ranks.

    Only the tp 1 rank presents the CuTe-DSL body's geometry; tp 2 and tp 4
    reach the C++ parametric family under the production route, which is the
    serving path at those ranks rather than a fallback.
    """

    params = []
    for tp, qo, kv in TP_GEOMETRIES:
        for index, (batch, seq_lengths) in enumerate(rows):
            bodies = bodies_for(
                seq_lengths, batch=batch, num_qo_heads=qo, num_kv_heads=kv
            )
            for backend, route in bodies:
                params.append(
                    pytest.param(
                        tp,
                        qo,
                        kv,
                        batch,
                        seq_lengths,
                        backend,
                        route,
                        id=f"{tag}-tp{tp}-{index:02d}-b{batch}-{backend}",
                    )
                )
    return params


@sm100_only
@pytest.mark.parametrize(
    "tp, num_qo_heads, num_kv_heads, batch, seq_lengths, backend, route",
    tp_params(_TP_ROWS, tag="tp"),
)
def test_matches_the_reference_at_every_tensor_parallel_geometry(
    tp, num_qo_heads, num_kv_heads, batch, seq_lengths, backend, route, monkeypatch
):
    """A relaxed assertion nobody executed at the relaxed value is not support.

    Each rank presents FEWER query heads over FEWER KV heads, over a page that
    has shrunk with it -- 73,728 B at tp 1, 36,864 B at tp 2, 18,432 B at tp 4.
    The reference is a peer here, not a restatement: it reads the head count and
    the group off the tensors for the same reason the kernel does.
    """
    inputs = _build_inputs(
        batch,
        seq_lengths,
        torch.device("cuda"),
        seed=1000 + tp,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
    )
    assert inputs["q"].shape[1] == num_qo_heads
    assert inputs["k"].shape[1] == num_kv_heads
    assert int(inputs["k"].stride(0)) == _page_bytes(num_kv_heads)
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


@sm100_only
@pytest.mark.parametrize(
    "tp, num_qo_heads, num_kv_heads, batch, seq_lengths, backend, route",
    tp_params([(11, _TP_RAGGED)], tag="ragged"),
)
def test_a_tensor_parallel_rank_survives_the_ragged_shapes_too(
    tp, num_qo_heads, num_kv_heads, batch, seq_lengths, backend, route, monkeypatch
):
    """The awkward coordinates, at the awkward head counts.

    Partial final blocks, an empty request among full ones, a one-token request
    and a prime batch. These are where the shared-memory zeroing, the causal
    limit and the compaction interact, and none of that is head-count-free: the
    split heuristic divides the GQA group by the cluster width, and the group is
    what changes across ranks -- at tp 4 one KV head means the grid's x extent
    is the split count alone.
    """
    inputs = _build_inputs(
        batch,
        seq_lengths,
        torch.device("cuda"),
        seed=2000 + tp,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
    )
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


@sm100_only
@pytest.mark.parametrize(
    "tp, num_qo_heads, num_kv_heads, backend, route",
    [
        pytest.param(tp, qo, kv, backend, route, id=f"tp{tp}-{backend}")
        for tp, qo, kv in TP_GEOMETRIES
        for backend, route in bodies_for(
            [8192] * 8, batch=8, num_qo_heads=qo, num_kv_heads=kv
        )
    ],
)
def test_a_tensor_parallel_rank_takes_the_parametric_family(
    tp, num_qo_heads, num_kv_heads, backend, route, monkeypatch
):
    """Which body the PRODUCTION route gives each rank, asserted not assumed.

    No override anywhere in this test: ``route="auto"`` is what a serving
    engine runs. tp 1 presents the CuTe-DSL body's geometry and is served by
    it; tp 2 and tp 4 do not, and the C++ parametric family serves them on
    every decode step -- which is a LATENCY statement, true only if the
    dispatch behaves that way, so it is read off the counters.

    It used to be weaker in two ways, both because of the module-level pin:
    the tp 1 arm could only say "cute OR pinned" (the pin made the answer a
    property of the process), and the whole test ran with the C++ body forced.
    """
    inputs = _build_inputs(
        8,
        [8192] * 8,
        torch.device("cuda"),
        seed=3000 + tp,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
    )
    assert (backend == CUTE) == (tp == 1 and route == "auto"), (tp, backend, route)
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )
    if tp != 1:
        reason = nvfp4.pinned_path_reason(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=HEAD_DIM,
            page_size=PAGE_SIZE,
            topk=TOPK,
            max_blocks=MAX_BLOCKS,
            seqlen_q=1,
            total_q=8,
            num_pages=int(inputs["k"].shape[0]),
        )
        assert reason is not None and "num_qo_heads" in reason


def test_the_gqa_group_ceiling_is_refused_by_name_not_by_the_allowlist():
    """Group > 16 is a STRUCTURAL bound; group 8 is merely not allowlisted.

    Two different refusals, and conflating them would hide which one is a kernel
    limit. The parametric body runs one warp per query head in a 512-thread CTA,
    so 17 siblings have nowhere to go; 8 siblings fit perfectly and are refused
    only because the prefill kernel cannot serve them, which the manifest says.
    """
    device = torch.device("cpu")
    over = _build_inputs(2, [1024, 1024], device, num_kv_heads=1, num_qo_heads=64)
    reason = nvfp4.check_surface(**_surface_kwargs(over))
    assert reason is not None and "head capacity" in reason

    under = _build_inputs(2, [1024, 1024], device, num_kv_heads=1, num_qo_heads=8)
    reason = nvfp4.check_surface(**_surface_kwargs(under))
    assert reason is not None and "capability allowlist" in reason

    payload = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert sorted(payload["allowlist"]) == [
        [16, 1, 128, 128],
        [32, 2, 128, 128],
        [64, 4, 128, 128],
    ]
    assert 8 not in {row[0] for row in payload["allowlist"]}


def test_the_page_map_is_a_function_of_the_kv_head_count():
    """Every offset scales with the rank, and the fixture agrees with the guard.

    A page map stated twice is a page map that can disagree with itself, and the
    disagreement is invisible to shape, dtype and stride -- which is exactly why
    the guard proves it by differencing base pointers. So the two statements are
    compared here at every count that has a body.
    """
    for _tp, _qo, kv in TP_GEOMETRIES:
        layout = nvfp4.page_layout(kv)
        assert layout["page_bytes"] == _page_bytes(kv)
        assert layout["k_scale_byte_offset"] == kv * PAGE_SIZE * DATA_DIM
        assert layout["v_data_byte_offset"] == (
            layout["k_scale_byte_offset"] + kv * PAGE_SIZE * SCALE_DIM
        )
        assert layout["v_scale_byte_offset"] == (
            layout["v_data_byte_offset"] + layout["k_scale_byte_offset"]
        )
        pool = torch.zeros(2 * layout["page_bytes"], dtype=torch.uint8)
        k_data, k_scale, v_data, v_scale = _page_views(pool, 2, kv)
        base = k_data.data_ptr()
        assert k_scale.data_ptr() - base == layout["k_scale_byte_offset"]
        assert v_data.data_ptr() - base == layout["v_data_byte_offset"]
        assert v_scale.data_ptr() - base == layout["v_scale_byte_offset"]


_WIDTH_ROWS = [1024, 2000, 33, 512, 1999, 128]


@sm100_only
@pytest.mark.parametrize(
    "max_blocks, backend, route",
    [
        pytest.param(width, backend, route, id=f"mb{width}-{backend}")
        for width in (16, 64, 128, 256)
        for backend, route in bodies_for(_WIDTH_ROWS, max_blocks=width)
    ],
)
def test_the_block_table_width_is_a_runtime_argument(
    max_blocks, backend, route, monkeypatch
):
    """The width is a runtime argument of BOTH C++ families and of the
    CuTe-DSL body, so all three see every width they can be given: 128 is the
    pinned envelope's value and the other three leave it."""
    inputs = _build_inputs(6, _WIDTH_ROWS, torch.device("cuda"), seed=max_blocks)
    table = torch.full((6, max_blocks), -1, dtype=torch.int32, device="cuda")
    width = min(max_blocks, inputs["page_table"].shape[1])
    table[:, :width] = inputs["page_table"][:, :width]
    inputs["page_table"] = table
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


_SPEC_ROWS = [4096, 8192, 1200, 300, 9000]


@sm100_only
@pytest.mark.parametrize(
    "seqlen_q, causal, backend, route",
    [
        pytest.param(
            seqlen_q,
            causal,
            backend,
            route,
            id=f"sq{seqlen_q}-causal{int(causal)}-{backend}",
        )
        for seqlen_q in (2, 4, 8)
        for causal in (True, False)
        for backend, route in bodies_for(
            _SPEC_ROWS, batch=5, seqlen_q=seqlen_q, causal=causal
        )
    ],
)
def test_multi_token_decode(seqlen_q, causal, backend, route, monkeypatch):
    """Speculative decoding: seqlen_q > 1 with a right-aligned causal limit.

    ``seqlen_q > 1`` is outside the CuTe-DSL body's geometry, so the C++
    parametric family serves these under the PRODUCTION route -- no override,
    which is what the old module-level pin obscured.
    """
    batch = 5
    inputs = _build_inputs(batch, _SPEC_ROWS, torch.device("cuda"), seed=seqlen_q)
    inputs["q"] = inputs["q"].repeat_interleave(seqlen_q, dim=0).contiguous()
    inputs["q2k_indices"] = (
        inputs["q2k_indices"].repeat_interleave(seqlen_q, dim=1).contiguous()
    )
    inputs["seqlen_q"] = seqlen_q
    inputs["causal"] = causal
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


@sm100_only
def test_a_selected_block_entirely_past_the_causal_limit(monkeypatch):
    """Regression: a wholly masked tile must contribute nothing.

    With ``seqlen_q > 1`` a selected block can start inside the request's KV
    length yet wholly past an early token's right-aligned causal limit. Here
    ``seq = 130`` and ``seqlen_q = 8`` put block 1 (columns 128-129) past the
    limit of query rows 0-5, and the cluster width at this size hands block 1
    to one rank on its own, so that rank's very first tile is entirely masked.
    Subtracting a sentinel row maximum from sentinel scores would give every
    column a weight of one.
    """
    inputs = _build_inputs(1, [130], torch.device("cuda"), seed=29)
    inputs["q"] = inputs["q"].repeat_interleave(8, dim=0).contiguous()
    inputs["q2k_indices"] = (
        inputs["q2k_indices"].repeat_interleave(8, dim=1).contiguous()
    )
    inputs["seqlen_q"] = 8
    # seqlen_q 8 is outside the CuTe-DSL geometry, so the parametric family is
    # what the production route reaches here.
    _assert_peer(
        _call_on(PARAMETRIC, inputs, monkeypatch, route="auto"), _reference(inputs)
    )


_NON_CAUSAL_ROWS = [8192, 1024, 129, 4096, 7, 2048]


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for(_NON_CAUSAL_ROWS, causal=False)],
)
def test_a_non_causal_decode_step(backend, route, monkeypatch):
    """``causal = 0`` is one of the few coordinates that reaches the PINNED
    family without an override: the CuTe-DSL body declines the geometry and
    the pinned envelope admits the shape, so the production route lands there.
    """
    inputs = _build_inputs(6, _NON_CAUSAL_ROWS, torch.device("cuda"), seed=23)
    inputs["causal"] = False
    assert backend == PINNED, backend
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


_MIXED_ROWS = [9000, 130, 8500, 256, 9000, 300, 8192, 1]


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for(_MIXED_ROWS)],
)
def test_short_and_long_requests_in_one_batch(backend, route, monkeypatch):
    """Regression: the valid prefix must be measured, not inferred.

    ``msa_topk_select`` on this architecture bounds the selection batch-wide,
    so the short requests below are handed block ids belonging to the long one.
    A kernel that assumed "KV length >= topk * page_size implies all top-k
    entries are valid" would dereference a ``-1`` page id here.
    """
    inputs = _build_inputs(8, _MIXED_ROWS, torch.device("cuda"), seed=7)
    got = _call_on(backend, inputs, monkeypatch, route=route)
    _assert_peer(got, _reference(inputs))


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([8192] * 4)],
)
def test_matches_when_the_query_aligns_with_a_selected_key(backend, route, monkeypatch):
    """The regime a synthetic generator never produces.

    Independent random q and k put ``cos(q, k)`` within a few sigma of zero, so
    a softmax that exponentiates the raw logit survives the usual correctness
    gate. Real attention selects the best-aligned keys, so the query is set
    equal to a selected key here and the logit is driven to its maximum.
    """
    device = torch.device("cuda")
    inputs = _build_inputs(4, [8192] * 4, device, seed=11)
    aligned = torch.empty_like(inputs["q"])
    for request in range(4):
        for head in range(NUM_KV_HEADS):
            block = int(inputs["q2k_indices"][head, request, 0])
            page = int(inputs["page_table"][request, block])
            packed = inputs["k"][page, head, 0]
            scales = inputs["k_scale"][page, head, 0].view(torch.float8_e4m3fn).float()
            lut = torch.tensor(_E2M1, dtype=torch.float32, device=device)
            key = torch.stack(
                (lut[(packed & 0x0F).long()], lut[(packed >> 4).long()]), dim=-1
            ).reshape(HEAD_DIM)
            key = key * scales.repeat_interleave(SCALE_VEC) * K_GLOBAL_SCALE
            aligned[request, head * 16 : (head + 1) * 16] = key.to(torch.bfloat16)
    inputs["q"] = aligned.contiguous()
    # Sanity: the maximum logit really is in the saturating regime that a
    # fixed-exponent softmax would clip.
    peak = (aligned[0, 0].float() * aligned[0, 0].float()).sum().item() * inputs[
        "softmax_scale"
    ]
    assert math.isfinite(peak)
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )


_TOPK_ROWS = [8192, 300, 9000, 1, 4096, 130]


@sm100_only
@pytest.mark.parametrize(
    "topk, backend, route",
    [
        pytest.param(topk, backend, route, id=f"k{topk}-{backend}")
        for topk in (1, 4, 8, 12, 16, 24, 31, 32)
        for backend, route in bodies_for(_TOPK_ROWS, topk=topk)
    ],
)
def test_every_admitted_top_k_matches_the_fp32_reference(
    topk, backend, route, monkeypatch
):
    """Correctness AT THE NEW VALUES, which is the whole point of widening.

    A runtime parameter exercised only at its old compile-time value is not a
    widened axis. 12, 24 and 31 are here because they are not powers of two and
    do not divide the cluster widths the launcher picks between -- the split
    heuristic has to step down to a count that divides the selection, and a
    kernel that silently kept an eight-way split would read past the row.
    """
    inputs = _build_inputs(6, _TOPK_ROWS, torch.device("cuda"), seed=23, topk=topk)
    before = _dispatch_count()
    got = _call_on(backend, inputs, monkeypatch, route=route)
    assert _dispatch_count() == before + 1
    _assert_peer(got, _reference(inputs))


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([8192] * 4, topk=32)],
)
def test_top_k_33_is_refused_at_the_boundary_on_the_device(backend, route, monkeypatch):
    """The ceiling, executed rather than read: 32 runs, 33 is refused by name.

    The guard and the C++ binding are separate copies of the same bound, so
    both are driven: the guard by the public entry point, the binding by
    calling the compiled op directly with a selection the guard would stop.
    """
    device = torch.device("cuda")
    inputs = _build_inputs(4, [8192] * 4, device, seed=5, topk=32)
    # top-k 32 is outside the CuTe-DSL geometry, so the production route lands
    # on the C++ translation unit and its parametric family reads the ballot.
    assert backend == PARAMETRIC, backend
    _assert_peer(
        _call_on(backend, inputs, monkeypatch, route=route), _reference(inputs)
    )

    over = _build_inputs(4, [8192] * 4, device, seed=5, topk=33)
    before = _dispatch_count()
    with pytest.raises(NotImplementedError):
        _call(over)
    assert _dispatch_count() == before

    module = nvfp4.load_msa_decode_nvfp4_specialized_module(nvfp4._target_for(device))
    with pytest.raises(Exception) as excinfo:
        module.msa_decode_nvfp4_specialized(
            over["q"],
            over["k"],
            over["v"],
            over["k_scale"],
            over["v_scale"],
            over["q2k_indices"],
            over["page_table"],
            over["seqused_k"],
            torch.empty_like(over["q"]),
            1,
            1,
            float(over["softmax_scale"]),
            float(over["k_global_scale"]),
            float(over["v_global_scale"]),
            -1,
        )
    assert "top-k must be in [1, 32]" in str(excinfo.value), str(excinfo.value)[:400]


@sm100_only
@pytest.mark.parametrize(
    "topk, backend, route",
    [
        pytest.param(topk, backend, route, id=f"k{topk}-{backend}")
        for topk in (8, 16)
        for backend, route in bodies_for(_TOPK_ROWS, topk=topk)
    ],
)
def test_a_strided_selection_is_bit_identical_to_its_contiguous_copy(
    topk, backend, route, monkeypatch
):
    """The copy removal, proved where it matters: same bits, no copy.

    The consumer's selection buffer is TOKEN-major and padded to
    max_num_batched_tokens, so the head-major view of a slice of it is
    non-contiguous with a token stride that is not `topk`. EVERY body that can
    serve the shape is driven -- named in the test id, and asserted from the
    counters -- and each must return exactly what the contiguous copy returns:
    not "within tolerance", because the two calls read the same integers in the
    same order, so any difference is an addressing bug, not arithmetic.
    """
    device = torch.device("cuda")
    inputs = _build_inputs(6, _TOPK_ROWS, device, seed=29, topk=topk)
    contiguous = inputs["q2k_indices"]
    total_q = int(inputs["q"].shape[0])

    # Rebuild the consumer's layout exactly: (padded_tokens, kv_heads, topk),
    # sliced to the live tokens and transposed.
    padded = total_q + 11
    token_major = torch.zeros(
        (padded, NUM_KV_HEADS, topk), dtype=torch.int32, device=device
    )
    token_major[:total_q] = contiguous.permute(1, 0, 2)
    strided = token_major[:total_q].transpose(0, 1)
    assert not strided.is_contiguous()
    assert strided.stride() == (topk, NUM_KV_HEADS * topk, 1)
    assert strided.stride(1) != topk  # the layout the old kernel assumed
    assert torch.equal(strided, contiguous)

    base = _call_on(backend, inputs, monkeypatch, route=route).clone()
    inputs["q2k_indices"] = strided
    got = _call_on(backend, inputs, monkeypatch, route=route).clone()
    inputs["q2k_indices"] = contiguous
    assert torch.equal(got, base), backend
    _assert_peer(got, _reference(inputs))


@sm100_only
def test_a_selection_with_a_gappy_innermost_dimension_is_refused():
    """The one layout the kernels cannot read, refused rather than misread."""
    device = torch.device("cuda")
    inputs = _build_inputs(4, [8192] * 4, device, seed=5)
    doubled = torch.zeros(
        (NUM_KV_HEADS, int(inputs["q"].shape[0]), 2 * TOPK),
        dtype=torch.int32,
        device=device,
    )
    doubled[:, :, ::2] = inputs["q2k_indices"]
    inputs["q2k_indices"] = doubled[:, :, ::2]
    assert inputs["q2k_indices"].stride(2) == 2
    before = _dispatch_count()
    with pytest.raises(NotImplementedError):
        _call(inputs)
    assert _dispatch_count() == before


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([8192] * 8)],
)
def test_cuda_graph_capture_after_an_eager_warmup(backend, route, monkeypatch):
    """The workspace path, warmed and captured on ONE stream.

    ``_bind_workspace`` latches the stream of the first call and refuses a
    second on a different one, and ``torch.cuda.graph(graph)`` always captures
    on a side stream of its own -- so warming eagerly on the default stream and
    then capturing with the bare context manager cannot succeed, whatever the
    kernel does. The first GPU run of this file is what surfaced that; the
    workspace contract is not wrong, the way this test used it was. Warm on an
    explicit stream and hand the same one to the capture.

    Production is not affected either way: ``capture_requires_workspace()`` is
    False and a serving engine passes no workspace -- which is what
    ``test_capture_without_a_workspace_replays_bit_identically`` covers.
    """

    inputs = _build_inputs(8, [8192] * 8, torch.device("cuda"), seed=17)
    workspace = MSASparseAttentionWorkspace(inputs["q"].device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    before = _dispatch_count()
    # BOTH calls -- the eager warm-up and the captured one -- have to land on
    # the body the test names, or "replays bit identically" is a comparison
    # between two different kernels.
    with expect_backend(
        backend, monkeypatch, route=route, calls=2, device=inputs["q"].device
    ):
        with torch.cuda.stream(stream):
            eager = _call(inputs, workspace=workspace).clone()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = _call(inputs, workspace=workspace)
    assert _dispatch_count() == before + 2, "the specialized kernel must be captured"
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([8192] * 4)],
)
def test_capture_without_a_workspace_replays_bit_identically(
    backend, route, monkeypatch
):
    """The shape a serving engine actually captures in.

    vLLM captures one graph per decode shape out of one shared graph memory
    pool and hands the attention op no workspace: it cannot, because the
    workspace admits a single capture and keys warm-vs-capture identity on
    ``data_ptr()`` while its activations come from a different pool than the
    eager warmup's. This route does not need one -- everything before the
    launch is host-side arithmetic over shapes and strides -- so capture must
    succeed and replay must reproduce the eager answer bit for bit.
    """
    inputs = _build_inputs(4, [8192] * 4, torch.device("cuda"), seed=19)
    before = _dispatch_count()
    with expect_backend(
        backend, monkeypatch, route=route, calls=2, device=inputs["q"].device
    ):
        eager = _call(inputs).clone()  # also warms the device
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _call(inputs)
    assert _dispatch_count() == before + 2, "the specialized kernel must be captured"
    captured.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@sm100_only
def test_capture_before_any_warm_is_rejected_and_warm_alone_clears_it(monkeypatch):
    """The remedy the error names must clear the error.

    ``check_specialized`` refuses capture until the device has taken a real
    eager launch, and the RuntimeError tells the caller to run
    ``msa_decode_nvfp4_specialized_warmup``. That is only true if the warmup
    dispatches rather than merely building, which is what this pins.
    """
    device = torch.device("cuda", torch.cuda.current_device())
    inputs = _build_inputs(4, [8192] * 4, device, seed=21)
    # Build the module so the failure is about warming and not about the JIT.
    # Pinned, explicitly: this test then RESETS the warm latch, so it must not
    # be the test that leaves the route's warm state a variable.
    _call_on(PINNED, inputs, monkeypatch)
    torch.cuda.synchronize()

    warmed = set(nvfp4._warmed_devices)
    monkeypatch.setattr(nvfp4, "_warmed_devices", set())
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="before the first eager dispatch"),
        torch.cuda.graph(graph),
    ):
        _call(inputs)
    torch.cuda.synchronize()

    nvfp4.warm(device)
    assert (device.type, device.index) in nvfp4._warmed_devices
    assert nvfp4.check_specialized(device) is None
    monkeypatch.setattr(nvfp4, "_warmed_devices", warmed)


@sm100_only
def test_the_warmup_hook_dispatches_once_and_is_idempotent(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(nvfp4, "_warmed_devices", set())
    before = nvfp4.msa_decode_nvfp4_specialized_stats()
    nvfp4.warm(device)
    after = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert nvfp4._module_is_loaded()
    assert after["compiled_variants"] >= 1
    # Two at the single-rank geometry (pinned and parametric) plus one per other
    # allowlisted head geometry. Derived from the allowlist rather than written
    # down, so adding a rank moves this with it.
    expected = 2 + len(nvfp4._load_allowlist()) - 1
    assert after["warm_dispatch_count"] == before["warm_dispatch_count"] + expected
    # A warm launch is not a caller's call, and the A/B harness differences
    # dispatch_count around one.
    assert after["dispatch_count"] == before["dispatch_count"]
    nvfp4.warm(device)
    again = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert again["warm_dispatch_count"] == after["warm_dispatch_count"]


@sm100_only
def test_warm_accepts_an_indexless_cuda_device():
    """``warm("cuda")`` must record the key a CUDA tensor's device produces."""
    device = torch.device("cuda", torch.cuda.current_device())
    nvfp4.warm(torch.device("cuda"))
    assert (device.type, device.index) in nvfp4._warmed_devices


@sm100_only
def test_the_two_instantiation_families_agree_on_the_same_call(monkeypatch):
    """The pin must not be able to change the answer, only the latency.

    Widening the block table by one unused column is semantically a no-op --
    every selected block id is below the original width -- but it moves the
    call from the pinned family to the parametric one, so the two families are
    compared on inputs that are otherwise byte-identical.
    """
    inputs = _build_inputs(8, [4096] * 8, torch.device("cuda"), seed=11)
    assert nvfp4.selects_pinned_path(
        **nvfp4._pinned_kwargs_for(
            inputs["q"],
            inputs["k"],
            inputs["q2k_indices"],
            inputs["page_table"],
            inputs["seqlen_q"],
        )
    )
    pinned = _call_on(PINNED, inputs, monkeypatch).clone()

    wider = torch.full(
        (inputs["page_table"].shape[0], MAX_BLOCKS + 1),
        -1,
        dtype=torch.int32,
        device=inputs["page_table"].device,
    )
    wider[:, :MAX_BLOCKS] = inputs["page_table"]
    widened = dict(inputs, page_table=wider)
    assert not nvfp4.selects_pinned_path(
        **nvfp4._pinned_kwargs_for(
            widened["q"],
            widened["k"],
            widened["q2k_indices"],
            widened["page_table"],
            widened["seqlen_q"],
        )
    )
    general = _call_on(PARAMETRIC, widened, monkeypatch).clone()

    # Same function, different instantiation: peers of each other and of the
    # FP32 composable reference.
    _assert_peer(pinned, general)
    reference = _reference(inputs)
    _assert_peer(pinned, reference)
    _assert_peer(general, reference)


@sm100_only
def test_the_deployment_shape_actually_takes_the_pinned_family(monkeypatch):
    """The counter is the instrument that catches a pin that stopped matching.

    Inside the C++ translation unit, which is where the pinned family lives --
    so the C++ body is forced deliberately and named. Under the production route
    this shape is served by the CuTe-DSL body instead, which is what
    ``test_a_tensor_parallel_rank_takes_the_parametric_family[tp1]`` asserts;
    the two facts are separate and neither is the other's default.
    """
    inputs = _build_inputs(16, [8192] * 16, torch.device("cuda"), seed=12)
    _call_on(PINNED, inputs, monkeypatch, route="pingpong")


@sm100_only
@pytest.mark.parametrize("seqlen_q", [2, 4])
def test_multi_token_decode_takes_the_parametric_family(seqlen_q, monkeypatch):
    """seqlen_q > 1 is a precondition failure for the pinned family, not an error.

    Under the PRODUCTION route: the CuTe-DSL body declines the geometry and the
    pinned envelope declines the query length, so the parametric family is what
    is left -- no override, which is the statement worth making.
    """
    inputs = _build_inputs(
        4 * seqlen_q, [4096] * (4 * seqlen_q), torch.device("cuda"), seed=13
    )
    inputs["seqlen_q"] = seqlen_q
    inputs["page_table"] = inputs["page_table"][:4].contiguous()
    inputs["seqused_k"] = inputs["seqused_k"][:4].contiguous()
    out = _call_on(PARAMETRIC, inputs, monkeypatch, route="auto")
    _assert_peer(out, _reference(inputs))


# ---------------------------------------------------------------------------
# out=: the caller's destination
#
# Without it a consumer has to write
#     out.copy_(msa_sparse_decode_attention(...))
# and MEASURED on GB300 at a pinned per-rank batch of 32 that `copy_` is one
# 512 KiB device-to-device CUDA-GRAPH NODE per attention layer per decode step:
# 57 nodes, 70.9 us/step of copy-engine time and 30.2 us/step of per-node
# dispatch gap.  None of it appears in a kernel-time breakdown, because a
# memcpy is not a kernel -- which is exactly why it survived this long.
#
# The kernel already took a destination (`run(..., out=...)`), so this is
# plumbing: the tests below pin that it is plumbing and nothing more, by
# requiring the answer to be BIT-IDENTICAL to the allocate-and-return path.
# ---------------------------------------------------------------------------
def test_out_is_a_keyword_argument_of_the_public_decode_entry_point():
    """The contract a consumer probes with inspect.signature before using it.

    vLLM's NVFP4 MSA impl has to stay importable against an older FlashInfer
    that has this route but not `out=`, so it feature-detects rather than
    version-gates.  This is the thing it detects.
    """
    import inspect

    parameter = inspect.signature(msa_sparse_decode_attention).parameters.get("out")
    assert parameter is not None, "out= is the consumer's feature-detection key"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_out_on_a_route_that_cannot_honour_it_raises_instead_of_copying(cpu_inputs):
    """A silent copy would hand back the cost this parameter exists to remove.

    So every route that allocates its own output refuses `out=` rather than
    accepting it and copying.  Reached here on a host tensor, which is not a
    compute-capability 10.0/10.3 device, so no GPU is needed to prove it.
    """
    with pytest.raises(NotImplementedError, match="out="):
        _call(cpu_inputs, out=torch.empty_like(cpu_inputs["q"]))


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([8192] * 32)],
)
def test_out_is_written_in_place_and_is_the_returned_tensor(
    backend, route, monkeypatch
):
    inputs = _build_inputs(32, [8192] * 32, torch.device("cuda"), seed=31)
    expected = _call_on(backend, inputs, monkeypatch, route=route).clone()

    destination = torch.empty_like(inputs["q"])
    returned = _call_on(backend, inputs, monkeypatch, route=route, out=destination)

    assert returned is destination, "out= must return the caller's own tensor"
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)
    _assert_peer(destination, _reference(inputs))


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([4096] * 16)],
)
def test_out_may_be_a_row_slice_of_a_larger_buffer_and_touches_nothing_else(
    backend, route, monkeypatch
):
    """The shape the consumer actually passes.

    vLLM hands this route ``output[:num_tokens].view(-1, H, D)[:num_decode]`` --
    a contiguous prefix of a buffer whose tail belongs to the prefill half of
    the same step.  Writing past it would corrupt the other half silently, so
    the tail is filled with a sentinel and checked.
    """
    batch = 16
    inputs = _build_inputs(batch, [4096] * batch, torch.device("cuda"), seed=32)
    expected = _call_on(backend, inputs, monkeypatch, route=route).clone()

    buffer = torch.full(
        (batch + 9, NUM_QO_HEADS, HEAD_DIM),
        -7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    returned = _call_on(backend, inputs, monkeypatch, route=route, out=buffer[:batch])

    assert returned.data_ptr() == buffer.data_ptr()
    torch.testing.assert_close(buffer[:batch], expected, rtol=0, atol=0)
    assert bool((buffer[batch:] == -7.0).all()), "the kernel wrote past out"


@sm100_only
@pytest.mark.parametrize(
    "mangle, fragment",
    [
        (lambda t: t.float(), "bfloat16"),
        (lambda t: t[:-1], "shape"),
        (lambda t: t.transpose(0, 1), "shape"),
        (lambda t: t.cpu(), "device"),
        (lambda t: t.expand(-1, -1, -1)[:, :, :HEAD_DIM:2], "shape"),
    ],
)
def test_a_malformed_out_raises_rather_than_being_copied_into(
    mangle, fragment, monkeypatch
):
    inputs = _build_inputs(4, [2048] * 4, torch.device("cuda"), seed=33)
    # Warm, so the failure is about `out` and not about the JIT. Named, because
    # even a warm-up call has to be a call to a body somebody chose.
    _call_on(PINNED, inputs, monkeypatch)
    with pytest.raises((ValueError, TypeError), match=fragment):
        _call(inputs, out=mangle(torch.empty_like(inputs["q"])))


@sm100_only
def test_a_non_contiguous_out_raises(monkeypatch):
    inputs = _build_inputs(4, [2048] * 4, torch.device("cuda"), seed=34)
    _call_on(PINNED, inputs, monkeypatch)
    wide = torch.empty(
        (4, NUM_QO_HEADS, 2 * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    strided = wide[:, :, :HEAD_DIM]
    assert tuple(strided.shape) == tuple(inputs["q"].shape)
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        _call(inputs, out=strided)


_BOTH_FAMILY_ROWS = [
    # (batch, lengths, seqlen_q, the family the shape selects inside the C++ TU)
    (32, [8192] * 32, 1, PARAMETRIC),
    (8, [4096] * 8, 2, PARAMETRIC),
    # The pinned row, which the two above stopped covering when the clustered
    # multi-chunk guard moved batch 17..32 onto the parametric family: without
    # it the test named "across both families" ran on one family twice.
    (16, [8192] * 16, 1, PINNED),
]


@sm100_only
@pytest.mark.parametrize(
    "seed, batch, lengths, seqlen_q, backend",
    [
        pytest.param(
            index,
            batch,
            lengths,
            seqlen_q,
            family,
            id=f"b{batch}-sq{seqlen_q}-{family}",
        )
        for index, (batch, lengths, seqlen_q, family) in enumerate(_BOTH_FAMILY_ROWS)
    ],
)
def test_out_is_bit_identical_to_the_copy_it_replaces_across_both_families(
    seed, batch, lengths, seqlen_q, backend, monkeypatch
):
    """Pinned and parametric both, since `out` is plumbed above the split."""
    inputs = _build_inputs(batch, lengths, torch.device("cuda"), seed=40 + seed)
    if seqlen_q > 1:
        inputs["q"] = inputs["q"].repeat_interleave(seqlen_q, dim=0).contiguous()
        inputs["q2k_indices"] = (
            inputs["q2k_indices"].repeat_interleave(seqlen_q, dim=1).contiguous()
        )
        inputs["seqlen_q"] = seqlen_q
    destination = torch.empty_like(inputs["q"])
    with expect_backend(backend, monkeypatch, calls=2, device=inputs["q"].device):
        allocated = _call(inputs).clone()
        _call(inputs, out=destination)
    torch.testing.assert_close(destination, allocated, rtol=0, atol=0)


@sm100_only
@pytest.mark.parametrize(
    "backend, route",
    [pytest.param(b, r, id=b) for b, r in bodies_for([8192] * 4)],
)
def test_capture_with_out_replays_into_the_caller_buffer(backend, route, monkeypatch):
    """The serving shape: no workspace, `out` is the engine's own buffer.

    The buffer's address is stable across replays -- it is what the graph
    captured -- so a replay must refill it, and must agree bit for bit with the
    eager answer.
    """
    inputs = _build_inputs(4, [8192] * 4, torch.device("cuda"), seed=35)
    destination = torch.empty_like(inputs["q"])
    before = _dispatch_count()
    with expect_backend(
        backend, monkeypatch, route=route, calls=2, device=inputs["q"].device
    ):
        eager = _call(inputs).clone()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _call(inputs, out=destination)
    assert _dispatch_count() == before + 2
    assert captured is destination
    destination.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, eager, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# the predicate, enumerated
# ---------------------------------------------------------------------------
def test_no_reachable_call_selects_an_uncompiled_instantiation(impl):
    """The unreachability claim, by enumeration rather than by argument.

    Over every geometry the route can present and every batch up to 1024, a
    plan that reports ``specialised`` must name an instantiation that is
    actually built -- and in particular never index 0, the only one whose
    split-K partials are published through global memory.
    """

    checked = 0
    for qo, kv, topk, page, sq, causal in itertools.product(
        (16, 32, 64, 128),
        (1, 2, 4, 8),
        (1, 8, 16, 32, 128),
        (16, 64, 128, 256),
        (1, 2, 5),
        (0, 1),
    ):
        if qo % kv:
            continue
        for batch in (
            1,
            2,
            3,
            5,
            8,
            9,
            10,
            16,
            21,
            22,
            31,
            32,
            63,
            64,
            65,
            127,
            128,
            255,
            256,
            1024,
        ):
            plan = impl.plan(
                total_q=batch * sq,
                num_qo_heads=qo,
                num_kv_heads=kv,
                grp=qo // kv,
                topk=topk,
                page_size=page,
                seqlen_q=sq,
                causal=causal,
            )
            checked += 1
            if plan["specialised"]:
                assert plan["kernel_idx"] in impl.SPECIALISED_KERNEL_IDS
                assert plan["kernel_idx"] != 0
                assert plan["scored_geom"]
            else:
                assert plan["reason"]
    assert checked > 10_000


def test_every_batch_lands_on_a_split_count_that_has_a_binary(impl):
    """The split count is not the one the CTA target asks for, and must not be.

    ``ceil(256 / (batch * 4))`` capped at the top-k takes every value in its
    range, not only the ones with an instantiation: batches 1..7 ask for 10..16
    and 10..15 ask for 5..7. Asking for a count with no binary used to DECLINE
    the call, which put 1..7 and 10..15 -- three of the consumer's capture rungs
    among them -- on the other kernel. They now step DOWN to the largest count
    that has one, so the covered set is every batch this geometry reaches, and
    the count each batch lands on is an instantiated one.
    """

    covered = {}
    for batch in range(1, 257):
        plan = impl.plan(total_q=batch, **_GEOMETRY)
        assert plan["specialised"], (batch, plan)
        assert plan["kernel_idx"] in impl.SPECIALISED_KERNEL_IDS, (batch, plan)
        covered[batch] = plan["nsplit"]
    assert set(covered) == set(range(1, 257))
    # And the count is one of the instantiated ones, never an interpolation.
    assert set(covered.values()) <= {1} | set(impl._BASE_K)


@pytest.mark.parametrize(
    "override",
    [
        dict(softmax_scale=0.0),
        dict(softmax_scale=-0.1),
        dict(k_global_scale=0.0),
        dict(k_global_scale=-1.0),
    ],
)
def test_a_non_positive_scale_routes_away_instead_of_raising(impl, override):
    """The specialised binaries scale the row maximum AFTER the reduction.

    That is exact only for a positive scale, so they refuse one. The route must
    not propagate the refusal: the public API accepted these before this kernel
    existed and the ping-pong kernel still serves them.
    """

    kwargs = dict(total_q=64, softmax_scale=0.088, k_global_scale=1.0, **_GEOMETRY)
    kwargs.update(override)
    assert impl.specialised_reason(**kwargs) is not None
    assert (
        impl.specialised_reason(**dict(kwargs, softmax_scale=0.088, k_global_scale=1.0))
        is None
    )


def test_the_route_asks_the_implementation_rather_than_deciding(impl, monkeypatch):
    """One copy of the predicate: break the implementation's and the route
    must follow it, not out-vote it."""

    inputs = _build_inputs(64, [8192] * 64, torch.device("cpu"))
    kwargs = dict(
        q=inputs["q"],
        k=inputs["k"],
        q2k_indices=inputs["q2k_indices"],
        seqlen_q=1,
        causal=True,
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
        device_warm=False,
    )
    assert nvfp4.specialised_route_reason(**kwargs) is None
    monkeypatch.setattr(
        impl, "specialised_reason", lambda **_: "the implementation says no"
    )
    assert nvfp4.specialised_route_reason(**kwargs) == "the implementation says no"


def test_an_unwarmed_device_routes_away_rather_than_compiling(impl, monkeypatch):
    """Compiling on the call path would break a CUDA-graph capture."""

    inputs = _build_inputs(64, [8192] * 64, torch.device("cpu"))
    kwargs = dict(
        q=inputs["q"],
        k=inputs["k"],
        q2k_indices=inputs["q2k_indices"],
        seqlen_q=1,
        causal=True,
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
    )
    monkeypatch.setattr(impl, "is_warm", lambda _device: False)
    assert "not warmed" in nvfp4.specialised_route_reason(**kwargs)


def test_stats_report_what_the_route_holds_and_refuses(impl):
    stats = nvfp4.msa_decode_nvfp4_specialized_stats()["specialised_route"]
    assert stats["available"] is True
    assert stats["persistent_device_bytes"] == 0
    assert stats["persistent_device_bytes_if_generalized_were_reachable"] > 60 << 20
    assert 0 in stats["uncompiled_instantiations"]
    assert stats["compiled_instantiations"] == sorted(impl.SPECIALISED_KERNEL_IDS)
    assert stats["batch_spans_at_geometry"] == [[1, 256]]
    assert stats["concurrent_stream_limit"] is None


def test_the_warm_shapes_cover_every_compiled_instantiation(impl):
    """One eager launch per instantiation, derived rather than tabulated."""

    shapes = nvfp4._specialised_warm_shapes()
    reached = {impl.plan(total_q=batch, **_GEOMETRY)["kernel_idx"] for batch in shapes}
    assert reached == set(impl.SPECIALISED_KERNEL_IDS)


# ---------------------------------------------------------------------------
# device: the route, on both sides of it
# ---------------------------------------------------------------------------
# The batch sizes the PRODUCTION route sends to the CuTe-DSL body, including
# the low and mid rows that used to fall out of the middle of the covered set.
# `expected` is the body, and it is asserted from the counters rather than
# taken on faith.
_ROWS = [
    pytest.param(batch, CUTE, id=f"b{batch}-{CUTE}")
    for batch in (1, 5, 10, 15, 8, 16, 22, 24, 31, 32, 64)
]


@sm100_only
@pytest.mark.parametrize("batch, expected", _ROWS)
def test_both_sides_of_the_route_match_the_reference(
    batch, expected, impl, monkeypatch
):
    """No override: this is the route a serving engine runs."""
    inputs = _build_inputs(batch, [8192] * batch, torch.device("cuda"))
    before = nvfp4.msa_decode_nvfp4_specialized_stats()
    out = _call_on(expected, inputs, monkeypatch, route="auto")
    after = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert after["dispatch_count"] == before["dispatch_count"] + 1
    _assert_peer(out, _reference(inputs), min_cosine=0.998)


def _cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a[None], b[None]).item()


@sm100_only
@pytest.mark.parametrize(
    "batch, cpp",
    [
        pytest.param(
            batch,
            bodies_for([8192] * batch, batch=batch)[1][0],
            id=f"b{batch}-cute-vs-{bodies_for([8192] * batch, batch=batch)[1][0]}",
        )
        for batch in (1, 5, 24, 32, 64)
    ],
)
def test_the_two_implementations_agree_with_each_other(batch, cpp, monkeypatch, impl):
    """Same tensors, same process, both kernels.

    The bar the pair is held to is DERIVED, not chosen: two kernels can agree
    with each other no better than the less accurate of them agrees with the
    reference. Measured on the covered batches, the specialised one sits at
    cosine 0.99999+ against the FP32 reference and the ping-pong one at
    0.9988, and the pair lands at 0.9988 -- which is the bound, not a defect.
    A fixed threshold tighter than that would have been a test of which kernel
    ran; a looser one would have tested nothing. The reference comparisons are
    what carry the correctness claim.
    """

    inputs = _build_inputs(batch, [8192] * batch, torch.device("cuda"))
    pingpong = _call_on(cpp, inputs, monkeypatch, route="pingpong").clone()
    routed = _call_on(CUTE, inputs, monkeypatch, route="auto").clone()
    reference = _reference(inputs)

    _assert_peer(routed, reference, min_cosine=0.998)
    _assert_peer(pingpong, reference, min_cosine=0.998)
    worse = min(_cosine(routed, reference), _cosine(pingpong, reference))
    pair = _cosine(routed, pingpong)
    assert pair >= worse - 1e-4, (pair, worse)


@sm100_only
def test_a_shape_off_the_specialised_surface_is_declined_visibly(monkeypatch, impl):
    """The heuristic is silent to the caller and legible to the operator.

    Off its geometry the CuTe-DSL body declines, the C++ translation unit serves
    the call, and the decline is recorded with its reason -- so "which kernel
    ran and why" is answerable from ``msa_decode_nvfp4_specialized_stats``
    without any switch to flip.
    """

    # Every BATCH is now inside the surface, so the shape that proves the
    # guard has to leave the GEOMETRY instead: a top-k of 8 is not the
    # specialised geometry, and narrowing the selection tensor is the one axis
    # these fixtures can move without rebuilding the page pool.
    inputs = _build_inputs(1, [8192], torch.device("cuda"))
    inputs = dict(inputs, q2k_indices=inputs["q2k_indices"][..., :8].contiguous())
    assert impl.plan(total_q=1, **dict(_GEOMETRY, topk=8))["specialised"] is False
    reason = nvfp4.specialised_route_reason(
        q=inputs["q"],
        k=inputs["k"],
        q2k_indices=inputs["q2k_indices"],
        seqlen_q=1,
        causal=True,
        softmax_scale=inputs["softmax_scale"],
        k_global_scale=inputs["k_global_scale"],
    )
    assert reason is not None and "8" in reason, reason
    declines_before = dict(
        nvfp4.msa_decode_nvfp4_specialized_stats()["specialised_route"]["declines"]
    )
    # Not forced anywhere: this is the production route landing where the
    # heuristic sends it, and the counters saying so.
    _call_on(PARAMETRIC, inputs, monkeypatch, route="auto")
    declines_after = nvfp4.msa_decode_nvfp4_specialized_stats()["specialised_route"][
        "declines"
    ]
    assert sum(declines_after.values()) == sum(declines_before.values()) + 1
    assert any(reason == key for key in declines_after), (reason, declines_after)


@sm100_only
@pytest.mark.parametrize(
    "batch, backend, route",
    [
        pytest.param(batch, backend, route, id=f"b{batch}-{backend}")
        for batch in (1, 32)
        for backend, route in bodies_for([8192] * batch, batch=batch)
    ],
)
def test_a_captured_graph_replays_both_sides_of_the_route(
    batch, backend, route, impl, monkeypatch
):
    """vLLM captures a graph per decode rung, and the rungs straddle the
    route, so capture has to work on both sides of it.

    BOTH SIDES, executed: the batch alone no longer selects a side -- the
    CuTe-DSL body covers 1..256 -- so the side is named and asserted, which is
    what makes "both sides" true rather than aspirational.
    """

    inputs = _build_inputs(batch, [8192] * batch, torch.device("cuda"))
    with expect_backend(
        backend, monkeypatch, route=route, calls=3, device=inputs["q"].device
    ):
        eager = _call(inputs).clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _call(inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _call(inputs)
    graph.replay()
    torch.cuda.synchronize()
    _assert_peer(captured, eager, min_cosine=0.9999)


# The shape sweep the pinned file can no longer run against this kernel.
# Every coordinate here is one the route SENDS to the specialised
# implementation, so the sweep exercises the kernel it names: partial final
# blocks, an empty request among full ones, a single very short request among
# long ones, prime batch sizes, and the split-count boundaries at 8/9, 16/21
# and 32/63.
_SPECIALISED_SHAPES = [
    (8, [8192] * 8),
    (8, [8192 + 97 * i for i in range(8)]),
    (9, [8192] * 9),
    (16, [1024] * 16),
    (16, [8192] * 15 + [3]),
    (16, [0] + [8192] * 15),
    (17, [4096 + 11 * i for i in range(17)]),
    (21, [6000] * 21),
    # 22..31 is the three-way split, and it is the one split count that does
    # NOT divide the sixteen output rows, so its last cluster rank owns four
    # rows where the others own six. Every coordinate that could expose an
    # unwritten or twice-written row is here: the two ends of the span, a
    # prime batch inside it, ragged lengths, an empty request, a request
    # shorter than one page, and a length that leaves a partial final block.
    (22, [8192] * 22),
    (23, [8192 + 89 * i for i in range(23)]),
    (24, [8192] * 24),
    (24, [1024] * 24),
    (24, [8192] * 23 + [3]),
    (24, [0] + [8192] * 23),
    (24, [8000 + 13 * i for i in range(24)]),
    (29, [6000] * 29),
    (31, [4096 + 17 * i for i in range(31)]),
    (32, [1024] * 32),
    (33, [1024 + 7 * i for i in range(33)]),
    (37, [6000] * 37),
    (63, [2048] * 63),
    (64, [8192] * 64),
    (129, [2000 + 3 * i for i in range(129)]),
]


@sm100_only
@pytest.mark.parametrize(
    "batch, seq_lengths, backend, route",
    sweep_params(_SPECIALISED_SHAPES, tag="spec"),
)
def test_the_specialised_implementation_over_its_own_shape_sweep(
    batch, seq_lengths, backend, route, impl, monkeypatch
):
    """Shape generality, on every kernel that can serve these shapes.

    These coordinates were added because the sweep in this file's ping-pong
    half was pinned away from the CuTe-DSL body, so that body was covered by
    seven uniform-length rows and nothing else -- and uniform lengths are
    exactly the case where a partial final block, an empty request and a
    block-id past the staged block-table prefix never occur. The pin is gone
    and the rows stay: they are now swept on the C++ translation unit too,
    where 22..31 is the band the clustered multi-chunk guard moved.
    """

    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=batch + 7)
    out = _call_on(backend, inputs, monkeypatch, route=route)
    _assert_peer(out, _reference(inputs))


@sm100_only
@pytest.mark.parametrize(
    "max_blocks, backend, route",
    [
        pytest.param(width, backend, route, id=f"mb{width}-{backend}")
        for width in (64, 128, 256)
        for backend, route in bodies_for(
            [min(width, 160) * 128] * 16, batch=16, max_blocks=width
        )
    ],
)
def test_a_selection_past_the_staged_block_table_prefix(
    max_blocks, backend, route, monkeypatch, impl
):
    """The block-table row is staged into shared memory only up to a fixed
    prefix; a selected block id past it resolves through a per-entry global
    read instead.

    160 blocks per request is what makes that branch live: it is longer than
    the 128-entry prefix, so roughly a fifth of the selections land beyond it.
    A sweep that never exceeds the prefix -- which is every uniform 8192-token
    row -- cannot tell the two paths apart.
    """

    monkeypatch.setattr(sys.modules[__name__], "MAX_BLOCKS", max_blocks)
    blocks = min(max_blocks, 160)
    inputs = _build_inputs(
        16, [blocks * 128] * 16, torch.device("cuda"), seed=max_blocks
    )
    assert int(inputs["page_table"].shape[1]) == max_blocks
    out = _call_on(backend, inputs, monkeypatch, route=route)
    _assert_peer(out, _reference(inputs))


@sm100_only
def test_warming_the_route_holds_no_persistent_device_memory(impl):
    """The 65 MiB arena the standalone form allocated is not allocated here.

    Measured, not asserted from the constant: warm on a clean allocator and
    compare the allocator's own high-water mark.
    """

    device = torch.device("cuda", torch.cuda.current_device())
    nvfp4.warm(device)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    nvfp4._specialised_warm_devices.discard((device.type, device.index))
    nvfp4._specialised_warm(device)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated(device)
    assert after - before < 1 << 20, (before, after)
