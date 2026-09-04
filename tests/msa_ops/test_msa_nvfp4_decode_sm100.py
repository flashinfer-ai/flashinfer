"""NVFP4 paged-KV MSA decode on compute capability 10.0/10.3.

This route is a new capability, not a specialization of an existing one: NVFP4
K/V raised ``NotImplementedError`` on this architecture before it existed. So
the guard is a capability guard, and shape generality is a correctness
requirement rather than a nicety -- the device tests below deliberately sweep
batch sizes, KV lengths and query lengths well outside anything the kernel was
tuned on, including partial final blocks, empty requests, prime batch sizes and
every cluster-width boundary.

The guard orders its predicates semantics -> layout -> device, so the whole
semantic and layout surface is exercisable on a host with no GPU.

EVERY DEVICE TEST HERE PINS THE ROUTE TO THIS KERNEL. The route now dispatches
two implementations internally (see test_msa_nvfp4_decode_hybrid_route.py), and
the specialised one covers batch 8, 16, 32 and 64 at this geometry -- which is
most of the sweep below. Without the pin this file would keep passing while
testing the OTHER kernel, and the pinned-vs-parametric family assertions, which
are about instantiations that exist only here, would fail for a reason that has
nothing to do with what they check. A test that silently stops exercising its
subject is worse than one that fails.
"""

import math

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


def _page_views(pool: torch.Tensor, num_pages: int):
    data_shape = (num_pages, NUM_KV_HEADS, PAGE_SIZE, DATA_DIM)
    scale_shape = (num_pages, NUM_KV_HEADS, PAGE_SIZE, SCALE_DIM)
    data_stride = (PAGE_BYTES, PAGE_SIZE * DATA_DIM, DATA_DIM, 1)
    scale_stride = (PAGE_BYTES, PAGE_SIZE * SCALE_DIM, SCALE_DIM, 1)
    k_scale_offset = NUM_KV_HEADS * PAGE_SIZE * DATA_DIM
    v_data_offset = k_scale_offset + NUM_KV_HEADS * PAGE_SIZE * SCALE_DIM
    v_scale_offset = v_data_offset + k_scale_offset
    return (
        torch.as_strided(pool, data_shape, data_stride, 0),
        torch.as_strided(pool, scale_shape, scale_stride, k_scale_offset),
        torch.as_strided(pool, data_shape, data_stride, v_data_offset),
        torch.as_strided(pool, scale_shape, scale_stride, v_scale_offset),
    )


def _build_inputs(batch, seq_lengths, device, seed=0, num_pages=None, topk=TOPK):
    """One decode step against a planar NVFP4 page pool, vLLM's layout."""
    generator = torch.Generator(device=device).manual_seed(seed)
    seqused_k = torch.tensor(seq_lengths, dtype=torch.int32, device=device)
    assert int(seqused_k.numel()) == batch
    blocks = (seqused_k.long() + PAGE_SIZE - 1) // PAGE_SIZE
    total_blocks = int(blocks.sum())
    num_pages = num_pages or max(1, total_blocks)

    page_table = torch.full((batch, MAX_BLOCKS), -1, dtype=torch.int32, device=device)
    permutation = torch.randperm(num_pages, generator=generator, device=device).to(
        torch.int32
    )
    cursor = 0
    for request, count in enumerate(blocks.tolist()):
        page_table[request, :count] = permutation[cursor : cursor + count]
        cursor += count

    pool = torch.zeros(num_pages * PAGE_BYTES, dtype=torch.uint8, device=device)
    k_data, k_scale, v_data, v_scale = _page_views(pool, num_pages)

    keys = _unit_rms(
        torch.randn(
            num_pages,
            NUM_KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    )
    values = torch.randn(
        num_pages,
        NUM_KV_HEADS,
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
            NUM_QO_HEADS,
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
        (NUM_KV_HEADS, batch, topk), -1, dtype=torch.int32, device=device
    )
    keep = min(topk, bound)
    for request in range(batch):
        for head in range(NUM_KV_HEADS):
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
# host-only guard tests
# ---------------------------------------------------------------------------
@pytest.fixture
def cpu_inputs():
    return _build_inputs(4, [1024, 300, 9000, 8192], torch.device("cpu"))


@pytest.fixture(autouse=True)
def _pin_route_to_this_kernel(monkeypatch):
    """This file is about the ping-pong kernel; keep the route on it."""

    monkeypatch.setenv(nvfp4._ROUTE_ENV, "pingpong")


def test_allowlist_and_stats_agree_with_the_module_constants():
    stats = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert stats["allowlist_fields"] == list(nvfp4._WORKLOAD_FIELDS)
    assert stats["allowlist_rows"] == 1
    # Model geometry only, and top-k is no longer part of it. Batch size, KV
    # length, block-table width, query length, causality, top-k and the
    # selection tensor's outer strides are parametric and must NOT appear here:
    # narrowing them would make the capability unreachable, not route it
    # somewhere else.
    assert stats["allowlist"] == [[64, 4, 128, 128]]
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
    """The eval-shaped small-pool case, which serving never produces."""
    assert nvfp4.pinned_path_reason(**dict(_DEPLOYMENT, total_q=32, num_pages=64))
    assert (
        nvfp4.pinned_path_reason(**dict(_DEPLOYMENT, total_q=32, num_pages=2048))
        is None
    )


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
@pytest.mark.parametrize("batch, seq_lengths", _TUNED)
def test_matches_the_reference_on_the_tuned_coordinates(batch, seq_lengths):
    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=batch)
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
@pytest.mark.parametrize("batch, seq_lengths", _UNTUNED)
def test_matches_the_reference_outside_the_tuned_coordinates(batch, seq_lengths):
    """Shape generality is a correctness requirement on this route.

    There is no other NVFP4 MSA decode implementation on this architecture, so
    a shape the kernel gets wrong has nowhere to fall back to; it has to be
    right everywhere the guard admits it.
    """
    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=batch + 1)
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
@pytest.mark.parametrize("max_blocks", [16, 64, 128, 256])
def test_the_block_table_width_is_a_runtime_argument(max_blocks):
    inputs = _build_inputs(
        6, [1024, 2000, 33, 512, 1999, 128], torch.device("cuda"), seed=max_blocks
    )
    table = torch.full((6, max_blocks), -1, dtype=torch.int32, device="cuda")
    width = min(max_blocks, inputs["page_table"].shape[1])
    table[:, :width] = inputs["page_table"][:, :width]
    inputs["page_table"] = table
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
@pytest.mark.parametrize("seqlen_q", [2, 4, 8])
@pytest.mark.parametrize("causal", [True, False])
def test_multi_token_decode(seqlen_q, causal):
    """Speculative decoding: seqlen_q > 1 with a right-aligned causal limit."""
    batch = 5
    seq_lengths = [4096, 8192, 1200, 300, 9000]
    inputs = _build_inputs(batch, seq_lengths, torch.device("cuda"), seed=seqlen_q)
    inputs["q"] = inputs["q"].repeat_interleave(seqlen_q, dim=0).contiguous()
    inputs["q2k_indices"] = (
        inputs["q2k_indices"].repeat_interleave(seqlen_q, dim=1).contiguous()
    )
    inputs["seqlen_q"] = seqlen_q
    inputs["causal"] = causal
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
def test_a_selected_block_entirely_past_the_causal_limit():
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
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
def test_a_non_causal_decode_step():
    inputs = _build_inputs(
        6, [8192, 1024, 129, 4096, 7, 2048], torch.device("cuda"), seed=23
    )
    inputs["causal"] = False
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
def test_short_and_long_requests_in_one_batch():
    """Regression: the valid prefix must be measured, not inferred.

    ``msa_topk_select`` on this architecture bounds the selection batch-wide,
    so the short requests below are handed block ids belonging to the long one.
    A kernel that assumed "KV length >= topk * page_size implies all top-k
    entries are valid" would dereference a ``-1`` page id here.
    """
    inputs = _build_inputs(
        8, [9000, 130, 8500, 256, 9000, 300, 8192, 1], torch.device("cuda"), seed=7
    )
    got = _call(inputs)
    _assert_peer(got, _reference(inputs))


@sm100_only
def test_matches_when_the_query_aligns_with_a_selected_key():
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
    _assert_peer(_call(inputs), _reference(inputs))


@sm100_only
@pytest.mark.parametrize("topk", [1, 4, 8, 12, 16, 24, 31, 32])
def test_every_admitted_top_k_matches_the_fp32_reference(topk):
    """Correctness AT THE NEW VALUES, which is the whole point of widening.

    A runtime parameter exercised only at its old compile-time value is not a
    widened axis. 12, 24 and 31 are here because they are not powers of two and
    do not divide the cluster widths the launcher picks between -- the split
    heuristic has to step down to a count that divides the selection, and a
    kernel that silently kept an eight-way split would read past the row.
    """
    inputs = _build_inputs(
        6, [8192, 300, 9000, 1, 4096, 130], torch.device("cuda"), seed=23, topk=topk
    )
    before = _dispatch_count()
    got = _call(inputs)
    assert _dispatch_count() == before + 1
    _assert_peer(got, _reference(inputs))


@sm100_only
def test_top_k_33_is_refused_at_the_boundary_on_the_device():
    """The ceiling, executed rather than read: 32 runs, 33 is refused by name.

    The guard and the C++ binding are separate copies of the same bound, so
    both are driven: the guard by the public entry point, the binding by
    calling the compiled op directly with a selection the guard would stop.
    """
    device = torch.device("cuda")
    inputs = _build_inputs(4, [8192] * 4, device, seed=5, topk=32)
    _assert_peer(_call(inputs), _reference(inputs))

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
@pytest.mark.parametrize("topk", [8, 16])
def test_a_strided_selection_is_bit_identical_to_its_contiguous_copy(topk, monkeypatch):
    """The copy removal, proved where it matters: same bits, no copy.

    The consumer's selection buffer is TOKEN-major and padded to
    max_num_batched_tokens, so the head-major view of a slice of it is
    non-contiguous with a token stride that is not `topk`. Both kernel families
    are driven (the route's own override picks between them) and both must
    return exactly what the contiguous copy returns -- not "within tolerance":
    the two calls read the same integers in the same order, so any difference
    is an addressing bug, not arithmetic.
    """
    device = torch.device("cuda")
    inputs = _build_inputs(
        6, [8192, 300, 9000, 1, 4096, 130], device, seed=29, topk=topk
    )
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

    for route in ("pingpong", "auto"):
        monkeypatch.setenv(nvfp4._ROUTE_ENV, route)
        base = _call(inputs).clone()
        inputs["q2k_indices"] = strided
        got = _call(inputs).clone()
        inputs["q2k_indices"] = contiguous
        assert torch.equal(got, base), route
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
def test_cuda_graph_capture_after_an_eager_warmup():
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
    with torch.cuda.stream(stream):
        eager = _call(inputs, workspace=workspace).clone()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    before = _dispatch_count()
    with torch.cuda.graph(graph, stream=stream):
        captured = _call(inputs, workspace=workspace)
    assert _dispatch_count() == before + 1, "the specialized kernel must be captured"
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


@sm100_only
def test_capture_without_a_workspace_replays_bit_identically():
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
    eager = _call(inputs).clone()  # also warms the device
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    before = _dispatch_count()
    with torch.cuda.graph(graph):
        captured = _call(inputs)
    assert _dispatch_count() == before + 1, "the specialized kernel must be captured"
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
    _call(inputs)  # build the module so the failure is about warming, not JIT
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
    assert after["warm_dispatch_count"] == before["warm_dispatch_count"] + 2
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
def test_the_two_instantiation_families_agree_on_the_same_call():
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
    pinned = _call(inputs).clone()

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
    general = _call(widened).clone()

    # Same function, different instantiation: peers of each other and of the
    # FP32 composable reference.
    _assert_peer(pinned, general)
    reference = _reference(inputs)
    _assert_peer(pinned, reference)
    _assert_peer(general, reference)


@sm100_only
def test_the_deployment_shape_actually_takes_the_pinned_family():
    """The counter is the instrument that catches a pin that stopped matching."""
    inputs = _build_inputs(16, [8192] * 16, torch.device("cuda"), seed=12)
    before = nvfp4.msa_decode_nvfp4_specialized_stats()
    _call(inputs)
    after = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert after["pinned_dispatch_count"] == before["pinned_dispatch_count"] + 1
    assert after["general_dispatch_count"] == before["general_dispatch_count"]


@sm100_only
@pytest.mark.parametrize("seqlen_q", [2, 4])
def test_multi_token_decode_takes_the_parametric_family(seqlen_q):
    """seqlen_q > 1 is a precondition failure for the pinned family, not an error."""
    inputs = _build_inputs(
        4 * seqlen_q, [4096] * (4 * seqlen_q), torch.device("cuda"), seed=13
    )
    inputs["seqlen_q"] = seqlen_q
    inputs["page_table"] = inputs["page_table"][:4].contiguous()
    inputs["seqused_k"] = inputs["seqused_k"][:4].contiguous()
    before = nvfp4.msa_decode_nvfp4_specialized_stats()
    out = _call(inputs)
    after = nvfp4.msa_decode_nvfp4_specialized_stats()
    assert after["general_dispatch_count"] == before["general_dispatch_count"] + 1
    assert after["pinned_dispatch_count"] == before["pinned_dispatch_count"]
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
def test_out_is_written_in_place_and_is_the_returned_tensor():
    inputs = _build_inputs(32, [8192] * 32, torch.device("cuda"), seed=31)
    expected = _call(inputs).clone()

    destination = torch.empty_like(inputs["q"])
    returned = _call(inputs, out=destination)

    assert returned is destination, "out= must return the caller's own tensor"
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)
    _assert_peer(destination, _reference(inputs))


@sm100_only
def test_out_may_be_a_row_slice_of_a_larger_buffer_and_touches_nothing_else():
    """The shape the consumer actually passes.

    vLLM hands this route ``output[:num_tokens].view(-1, H, D)[:num_decode]`` --
    a contiguous prefix of a buffer whose tail belongs to the prefill half of
    the same step.  Writing past it would corrupt the other half silently, so
    the tail is filled with a sentinel and checked.
    """
    batch = 16
    inputs = _build_inputs(batch, [4096] * batch, torch.device("cuda"), seed=32)
    expected = _call(inputs).clone()

    buffer = torch.full(
        (batch + 9, NUM_QO_HEADS, HEAD_DIM),
        -7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    returned = _call(inputs, out=buffer[:batch])

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
def test_a_malformed_out_raises_rather_than_being_copied_into(mangle, fragment):
    inputs = _build_inputs(4, [2048] * 4, torch.device("cuda"), seed=33)
    _call(inputs)  # warm, so the failure is about `out` and not about the JIT
    with pytest.raises((ValueError, TypeError), match=fragment):
        _call(inputs, out=mangle(torch.empty_like(inputs["q"])))


@sm100_only
def test_a_non_contiguous_out_raises():
    inputs = _build_inputs(4, [2048] * 4, torch.device("cuda"), seed=34)
    _call(inputs)
    wide = torch.empty(
        (4, NUM_QO_HEADS, 2 * HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    strided = wide[:, :, :HEAD_DIM]
    assert tuple(strided.shape) == tuple(inputs["q"].shape)
    assert not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        _call(inputs, out=strided)


@sm100_only
def test_out_is_bit_identical_to_the_copy_it_replaces_across_both_families():
    """Pinned and parametric both, since `out` is plumbed above the split."""
    for seed, (batch, lengths, seqlen_q) in enumerate(
        [(32, [8192] * 32, 1), (8, [4096] * 8, 2)]
    ):
        inputs = _build_inputs(batch, lengths, torch.device("cuda"), seed=40 + seed)
        if seqlen_q > 1:
            inputs["q"] = inputs["q"].repeat_interleave(seqlen_q, dim=0).contiguous()
            inputs["q2k_indices"] = (
                inputs["q2k_indices"].repeat_interleave(seqlen_q, dim=1).contiguous()
            )
            inputs["seqlen_q"] = seqlen_q
        allocated = _call(inputs).clone()
        destination = torch.empty_like(inputs["q"])
        _call(inputs, out=destination)
        torch.testing.assert_close(destination, allocated, rtol=0, atol=0)


@sm100_only
def test_capture_with_out_replays_into_the_caller_buffer():
    """The serving shape: no workspace, `out` is the engine's own buffer.

    The buffer's address is stable across replays -- it is what the graph
    captured -- so a replay must refill it, and must agree bit for bit with the
    eager answer.
    """
    inputs = _build_inputs(4, [8192] * 4, torch.device("cuda"), seed=35)
    eager = _call(inputs).clone()
    destination = torch.empty_like(inputs["q"])
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    before = _dispatch_count()
    with torch.cuda.graph(graph):
        captured = _call(inputs, out=destination)
    assert _dispatch_count() == before + 1
    assert captured is destination
    destination.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, eager, rtol=0, atol=0)
