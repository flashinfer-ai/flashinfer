"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the Cake fused QK RMSNorm + NeoX RoPE + paged KV append (BF16) kernel.

Independent torch reference (per-head RMSNorm in f32, NeoX rotation from the
cos/sin table, NHD paged append, last-page clearing).  Tolerances follow the
Cake BF16 standard: Q/K ``atol = rtol = 1e-2``, V bit-exact, every cache /
output element outside the appended rows and the cleared range untouched.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_fused_qk_rope_append`` facade is covered by
``test_cake_fused_qk_rope_append_api.py``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_fused_qk_rope_append")
cake_build = cake_package("cake_fused_qk_rope_append", "cake_build")
loader = cake_package("cake_fused_qk_rope_append", "cake_jit")
cake_fused_qk_rmsnorm_rope_append_paged_kv_cache = (
    pkg.cake_fused_qk_rmsnorm_rope_append_paged_kv_cache
)

HEAD_DIM = 128
POISON = 1234.0
PACKAGE_DIR = Path(pkg.__file__).resolve().parent


def _device() -> torch.device:
    return torch.device("cuda:0")


def _arch():
    if not torch.cuda.is_available():
        return None
    return loader.arch_for_capability(torch.cuda.get_device_capability(_device()))


def _supported() -> bool:
    return _arch() is not None


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if _arch() is None:
        pytest.skip(
            "cake_fused_qk_rope_append ships no build for compute capability "
            f"{torch.cuda.get_device_capability(_device())}"
        )


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
    yield


def rotary_cos_sin_table(
    max_positions: int, device, base: float = 10000.0
) -> torch.Tensor:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float64, device=device) / half)
    )
    pos = torch.arange(max_positions, dtype=torch.float64, device=device)
    freqs = torch.outer(pos, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=1).to(torch.float32)


def reference(
    qkv,
    cos_sin,
    seq_lens,
    q_indptr,
    page_indices,
    key_cache,
    value_cache,
    *,
    num_q_heads,
    num_kv_heads,
    qk_norm_policy,
    q_norm_weight,
    k_norm_weight,
    eps=1e-6,
    out_q=None,
    out_k=None,
    out_v=None,
    clear_last_page=True,
):
    T = qkv.shape[0]
    hq, hkv = num_q_heads, num_kv_heads
    x = qkv.float()
    q = x[:, : hq * HEAD_DIM].reshape(T, hq, HEAD_DIM)
    k = x[:, hq * HEAD_DIM : (hq + hkv) * HEAD_DIM].reshape(T, hkv, HEAD_DIM)
    v = qkv[:, (hq + hkv) * HEAD_DIM :].reshape(
        T, hkv, HEAD_DIM
    )  # bf16, copied bit-exactly
    B = seq_lens.shape[0]
    qi = q_indptr.tolist()
    sl = seq_lens.tolist()
    page_size = key_cache.shape[1]
    if out_q is None:
        out_q = torch.empty((T, hq, HEAD_DIM), dtype=torch.bfloat16, device=qkv.device)

    def rms(t, w):
        var = (t * t).mean(dim=-1, keepdim=True)
        return t * torch.rsqrt(var + eps) * w

    def rope(t, cs):
        half = HEAD_DIM // 2
        c = cs[:, None, :half]
        s = cs[:, None, half:]
        x1, x2 = t[..., :half], t[..., half:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    if clear_last_page:
        for b in range(B):
            if sl[b] <= 0:
                continue
            last = sl[b] - 1
            page = int(page_indices[b, last // page_size])
            first_unused = last % page_size + 1
            key_cache[page, first_unused:] = 0
            value_cache[page, first_unused:] = 0
    for b in range(B):
        rows = list(range(qi[b], qi[b + 1]))
        if not rows:
            continue
        positions = torch.tensor(
            [r + sl[b] - qi[b + 1] for r in rows], device=qkv.device
        )
        valid = positions >= 0
        rows_t = torch.tensor(rows, device=qkv.device)[valid]
        positions = positions[valid]
        if rows_t.numel() == 0:
            continue
        cs = cos_sin[positions]
        qb, kb = q[rows_t], k[rows_t]
        if qk_norm_policy == 2:
            qb, kb = rms(qb, q_norm_weight), rms(kb, k_norm_weight)
        qb, kb = rope(qb, cs), rope(kb, cs)
        if qk_norm_policy == 1:
            qb, kb = rms(qb, q_norm_weight), rms(kb, k_norm_weight)
        out_q[rows_t] = qb.to(torch.bfloat16)
        kb16 = kb.to(torch.bfloat16)
        vb = v[rows_t]
        if out_k is not None:
            out_k[rows_t] = kb16
        if out_v is not None:
            out_v[rows_t] = vb
        if out_k is None or out_v is None:
            pages = page_indices[b][positions // page_size].long()
            slots = positions % page_size
            if out_k is None:
                key_cache[pages, slots] = kb16
            if out_v is None:
                value_cache[pages, slots] = vb
    return out_q


def make_case(
    *,
    q_lens,
    ctx_lens,
    num_q_heads,
    num_kv_heads,
    page_size,
    policy,
    seed,
    device="cuda",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = len(q_lens)
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens, strict=True)]
    pages_per_req = [(s + page_size - 1) // page_size for s in seq_lens]
    max_pages = max(max(pages_per_req), 1)
    total_pages = sum(pages_per_req) + 3
    perm = torch.randperm(total_pages, generator=g)
    page_indices = torch.zeros((B, max_pages), dtype=torch.int32)
    cursor = 0
    for b in range(B):
        n = pages_per_req[b]
        page_indices[b, :n] = perm[cursor : cursor + n].to(torch.int32)
        cursor += n
    q_indptr = torch.zeros(B + 1, dtype=torch.int32)
    q_indptr[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    T = int(q_indptr[-1])
    width = (num_q_heads + 2 * num_kv_heads) * HEAD_DIM
    qkv = (
        torch.randn((T, width), generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .to(device)
    )
    cos_sin = rotary_cos_sin_table(max(seq_lens) + 1, device)
    q_w = (
        (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16).float().to(device)
    )
    k_w = (
        (torch.rand(HEAD_DIM, generator=g) + 0.5).to(torch.bfloat16).float().to(device)
    )
    key_cache = torch.full(
        (total_pages, page_size, num_kv_heads, HEAD_DIM),
        POISON,
        dtype=torch.bfloat16,
        device=device,
    )
    value_cache = torch.full_like(key_cache, POISON)
    out_q = torch.full(
        (T, num_q_heads, HEAD_DIM), POISON, dtype=torch.bfloat16, device=device
    )
    return dict(
        qkv=qkv,
        cos_sin=cos_sin,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        q_indptr=q_indptr.to(device),
        page_indices=page_indices.to(device),
        key_cache=key_cache,
        value_cache=value_cache,
        out_q=out_q,
        q_norm_weight=q_w,
        k_norm_weight=k_w,
        policy=policy,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def run_and_compare(case, *, caller_owned_kv=False, clear=True):
    T = case["qkv"].shape[0]
    hkv = case["num_kv_heads"]
    dev = case["qkv"].device
    kw = dict(
        num_q_heads=case["num_q_heads"],
        num_kv_heads=hkv,
        qk_norm_policy=case["policy"],
        q_norm_weight=case["q_norm_weight"] if case["policy"] else None,
        k_norm_weight=case["k_norm_weight"] if case["policy"] else None,
    )
    ok = ov = rok = rov = None
    if caller_owned_kv:
        ok = torch.full((T, hkv, HEAD_DIM), POISON, dtype=torch.bfloat16, device=dev)
        ov = torch.full_like(ok, POISON)
        rok, rov = ok.clone(), ov.clone()
    kc, vc, oq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    ret = cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        kc,
        vc,
        out_q=oq,
        out_k=ok,
        out_v=ov,
        clear_unused_last_page_rows=clear,
        **kw,
    )
    assert ret is oq
    torch.cuda.synchronize()
    rkc, rvc, roq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    reference(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        rkc,
        rvc,
        num_q_heads=case["num_q_heads"],
        num_kv_heads=hkv,
        qk_norm_policy=case["policy"],
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
        out_q=roq,
        out_k=rok,
        out_v=rov,
        clear_last_page=clear,
    )
    torch.testing.assert_close(oq.float(), roq.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(kc.float(), rkc.float(), atol=1e-2, rtol=1e-2)
    assert torch.equal(vc.view(torch.int16), rvc.view(torch.int16)), (
        "V must be copied bit-exactly"
    )
    if caller_owned_kv:
        torch.testing.assert_close(ok.float(), rok.float(), atol=1e-2, rtol=1e-2)
        assert torch.equal(ov.view(torch.int16), rov.view(torch.int16))
    # untouched regions: everything the reference left at POISON must still be POISON
    for mine, ref in ((kc, rkc), (vc, rvc), (oq, roq)):
        mask = ref == POISON
        assert torch.equal(mine[mask], ref[mask]), (
            "kernel wrote outside the appended / cleared range"
        )


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("policy", [0, 1, 2])
@pytest.mark.parametrize("page_size", [16, 64])
def test_decode_batch(heads, policy, page_size):
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 32,
        ctx_lens=[2048 + 7 * i for i in range(32)],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=page_size,
        policy=policy,
        seed=892,
    )
    run_and_compare(case)


@pytest.mark.parametrize("heads", [(8, 1), (64, 8)])
@pytest.mark.parametrize("policy", [1, 2])
def test_prefill_chunks_cross_pages(heads, policy):
    _skip_unless_supported()
    case = make_case(
        q_lens=[16, 64, 128, 3],
        ctx_lens=[2047, 0, 2000, 63],
        num_q_heads=heads[0],
        num_kv_heads=heads[1],
        page_size=64,
        policy=policy,
        seed=893,
    )
    run_and_compare(case)


def test_ragged_with_empty_requests_and_caller_owned_kv():
    _skip_unless_supported()
    case = make_case(
        q_lens=[5, 0, 1, 33, 0, 2],
        ctx_lens=[100, 64, 4095, 0, 1, 127],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=894,
    )
    run_and_compare(case)
    run_and_compare(case, caller_owned_kv=True)


def test_no_clear_leaves_last_page_tail():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 4,
        ctx_lens=[10, 70, 128, 5],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=895,
    )
    run_and_compare(case, clear=False)


def test_large_batch_lookup_fallback():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 300,
        ctx_lens=[64 + (i % 200) for i in range(300)],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=896,
    )
    run_and_compare(case)


def test_cuda_graph_replay_matches_eager():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 8,
        ctx_lens=[2048] * 8,
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=897,
    )
    kw = dict(
        num_q_heads=8,
        num_kv_heads=1,
        qk_norm_policy=2,
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
    )
    kc, vc, oq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
        case["qkv"],
        case["cos_sin"],
        case["seq_lens"],
        case["q_indptr"],
        case["page_indices"],
        kc,
        vc,
        out_q=oq,
        **kw,
    )
    torch.cuda.synchronize()
    gkc, gvc, goq = (
        case["key_cache"].clone(),
        case["value_cache"].clone(),
        case["out_q"].clone(),
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(
            2
        ):  # warm the JIT module / allocator on the side stream before capture
            cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
                case["qkv"],
                case["cos_sin"],
                case["seq_lens"],
                case["q_indptr"],
                case["page_indices"],
                gkc,
                gvc,
                out_q=goq,
                **kw,
            )
        gkc.fill_(POISON)
        gvc.fill_(POISON)
        goq.fill_(POISON)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
                case["qkv"],
                case["cos_sin"],
                case["seq_lens"],
                case["q_indptr"],
                case["page_indices"],
                gkc,
                gvc,
                out_q=goq,
                **kw,
            )
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(goq.view(torch.int16), oq.view(torch.int16))
    assert torch.equal(gkc.view(torch.int16), kc.view(torch.int16))
    assert torch.equal(gvc.view(torch.int16), vc.view(torch.int16))


def test_repeated_launches_and_buffer_reuse_are_bitwise_identical():
    """Three launches of one problem on fresh buffers write the same bits, and a
    launch into the already written caller-owned buffers writes them again."""
    _skip_unless_supported()
    case = make_case(
        q_lens=[1] * 8 + [16, 0, 3],
        ctx_lens=[2048 + 5 * i for i in range(8)] + [2047, 5, 0],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=2,
        seed=898,
    )
    kw = dict(
        num_q_heads=8,
        num_kv_heads=1,
        qk_norm_policy=2,
        q_norm_weight=case["q_norm_weight"],
        k_norm_weight=case["k_norm_weight"],
    )

    def launch(kc, vc, oq):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            kc,
            vc,
            out_q=oq,
            **kw,
        )
        torch.cuda.synchronize()
        return [t.view(torch.int16).clone() for t in (oq, kc, vc)]

    snapshots = []
    for _ in range(3):
        kc, vc, oq = (
            case["key_cache"].clone(),
            case["value_cache"].clone(),
            case["out_q"].clone(),
        )
        snapshots.append(launch(kc, vc, oq))
    for later in snapshots[1:]:
        for got, want in zip(later, snapshots[0], strict=True):
            assert torch.equal(got, want)
    reused = launch(kc, vc, oq)
    for got, want in zip(reused, snapshots[0], strict=True):
        assert torch.equal(got, want)


def test_changing_shapes_in_one_process_does_not_rebuild():
    """Shapes the policy maps onto one stage share one loaded library: no
    second build, no second load."""
    _skip_unless_supported()
    arch = _arch()
    num_sms = loader.sm_count(_device().index)

    def stage_of(requests: int, page_size: int) -> str:
        return loader.stage_for(
            8,
            1,
            arch=arch,
            num_rows=requests,
            num_requests=requests,
            page_size=page_size,
            num_sms=num_sms,
        )

    def decode_case(requests: int, page_size: int, seed: int):
        return make_case(
            q_lens=[1] * requests,
            ctx_lens=[2048 + i for i in range(requests)],
            num_q_heads=8,
            num_kv_heads=1,
            page_size=page_size,
            policy=2,
            seed=seed,
        )

    base = stage_of(32, 64)
    same_stage = [
        (requests, page_size)
        for requests, page_size in ((40, 64), (16, 64), (33, 64), (8, 64), (48, 128))
        if stage_of(requests, page_size) == base
    ]
    assert same_stage, "the policy left no second decode shape on the B32 stage"
    run_and_compare(decode_case(32, 64, 899))
    misses = loader.load.cache_info().misses
    for requests, page_size in same_stage:
        run_and_compare(decode_case(requests, page_size, 900 + requests))
    assert loader.load.cache_info().misses == misses, (
        "a changed shape on the same stage loaded another library"
    )


def test_rejects_bad_arguments():
    _skip_unless_supported()
    case = make_case(
        q_lens=[1],
        ctx_lens=[5],
        num_q_heads=8,
        num_kv_heads=1,
        page_size=64,
        policy=0,
        seed=1,
    )
    with pytest.raises(ValueError):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            case["key_cache"],
            case["value_cache"],
            num_q_heads=4,
            num_kv_heads=1,
        )
    with pytest.raises(ValueError):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            case["key_cache"],
            case["value_cache"],
            num_q_heads=8,
            num_kv_heads=1,
            qk_norm_policy=2,
        )
    with pytest.raises(ValueError):
        cake_fused_qk_rmsnorm_rope_append_paged_kv_cache(
            case["qkv"],
            case["cos_sin"],
            case["seq_lens"],
            case["q_indptr"],
            case["page_indices"],
            case["key_cache"],
            case["value_cache"],
            num_q_heads=8,
            num_kv_heads=1,
            out_q=torch.empty((1, 8, 64), dtype=torch.bfloat16, device="cuda"),
        )


# ----------------------------------------------------------------------------------------------
# Registry and build manifest (CPU): every module record names its ``build.json`` library
# and the host plan only; the binding's argument order is one contract for every stage.

_ARG_NAMES = [
    "qkv",
    "cos_sin",
    "q_indptr",
    "seq_lens",
    "page_table",
    "k_cache",
    "v_cache",
    "out_q",
    "out_k",
    "out_v",
    "q_norm_weight",
    "k_norm_weight",
    "num_rows",
    "num_requests",
    "max_pages",
    "page_size",
    "k_page_stride",
    "v_page_stride",
    "num_row_ctas",
    "clear_units_per_request",
    "qk_norm_policy",
    "eps",
    "use_out_k",
    "use_out_v",
    "clear_last_page",
    "grid_x",
    "grid_y",
    "grid_z",
]


def test_every_stage_binds_the_public_argument_plan():
    """The binding's argument order (twelve buffers, thirteen scalars, the grid) is
    the same for every stage; the host binds it through the record's ``logical``
    names."""
    for name, record in loader.MODULES.items():
        plan = record["arg_plan"]
        assert [key for _kind, key in plan] == _ARG_NAMES, name
        assert [kind for kind, _key in plan] == (
            ["buffer"] * 12 + ["parameter"] * 13 + ["grid"] * 3
        ), name


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only (stage, knobs, launch record,
    grid rule and the kernel-parameter -> logical-argument map).
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    logical_names = set(loader.LOGICAL_TENSORS) | set(loader.LOGICAL_SCALARS)
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["stage"] == name
        assert record["library"] == record["kernel"], name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        assert record["ffi_entry"] == "run", name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        knobs = record["knobs"]
        assert set(knobs) == set(loader.KNOB_KEYS), name
        heads = (int(knobs["num_q_heads"]), int(knobs["num_kv_heads"]))
        assert heads in loader.SUPPORTED_HEADS, name
        assert name.startswith(f"hq{heads[0]}_hkv{heads[1]}"), name
        assert tuple(record["launch"]["block"]) == (loader.THREADS, 1, 1), name
        assert record["grid_rule"]["kind"] == "row_ctas_plus_clear_ctas", name
        assert {key for kind, key in record["arg_plan"] if kind != "grid"} == set(
            record["logical"]
        ), name
        assert {key for kind, key in record["arg_plan"] if kind == "grid"} == {
            "grid_x",
            "grid_y",
            "grid_z",
        }, name
        assert set(record["logical"].values()) <= logical_names, name
        assert record["arches"], name
        assert set(record["arches"]) <= set(loader.ARCHES), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (PACKAGE_DIR / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert not any("use_fast_math" in flag for flag in entry["nvcc_flags"])
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    for arch in loader.ARCHES:
        shipped = {
            name for name, record in loader.MODULES.items() if arch in record["arches"]
        }
        assert set(loader.stages_for(arch)) == shipped
        assert shipped, arch


def test_stage_policy_table_is_well_formed():
    policy = loader.STAGE_POLICY
    assert set(policy) == {"constants", "rule", "version"}
    assert isinstance(policy["rule"], str) and isinstance(policy["version"], str)
    assert {
        "clear_ctas_per_free_sm",
        "direct_min_requests",
        "sm100_decode_8_1_min_requests",
        "sm100_u2_min_clear_ctas",
        "v_sub_base_8_1",
        "w20",
        "w20_v_sub_base",
    } <= set(policy["constants"])


# ----------------------------------------------------------------------------------------------
# Stage policy (CPU): the host mirrors the Cake policy and must only select shipped builds.
# ``_POLICY_TABLE`` pins the policy's choice at representative shapes (rows change when the
# registry's policy changes; the structure does not); the scan covers the shape space.

_POLICY_TABLE: dict[tuple[str, int, int, int, int, int, int], str] = {
    # (arch, Hq, Hkv, num_rows, num_requests, page_size, num_sms) -> stage
    (
        "sm_90a",
        8,
        1,
        32,
        32,
        64,
        132,
    ): "hq8_hkv1_vsub3_d_lookup_rounds2_packed_xs1_pre_rot_ss1",
    (
        "sm_90a",
        8,
        1,
        128,
        1,
        64,
        132,
    ): "hq8_hkv1_vsub3_d_lookup_rounds2_packed_xs1_pre_rot_ss1",
    (
        "sm_90a",
        8,
        1,
        128,
        8,
        64,
        132,
    ): "hq8_hkv1_vsub3_d_lookup_rounds2_packed_xs1_pre_rot_ss1",
    (
        "sm_90a",
        8,
        1,
        256,
        256,
        64,
        132,
    ): "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1_pt_window1_min_blocks5",
    (
        "sm_90a",
        8,
        1,
        512,
        512,
        64,
        132,
    ): "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1_pt_window1_min_blocks5",
    (
        "sm_90a",
        64,
        8,
        32,
        32,
        64,
        132,
    ): "hq64_hkv8_w20_vsub18_d_lookup_rounds0_single_weight1_pt_window1",
    ("sm_90a", 64, 8, 128, 8, 64, 132): "hq64_hkv8_d2_lookup_rounds2",
    ("sm_90a", 64, 8, 128, 128, 64, 132): "hq64_hkv8_d2_lookup_rounds2",
    (
        "sm_90a",
        64,
        8,
        128,
        1,
        64,
        132,
    ): "hq64_hkv8_w20_vsub18_d_lookup_rounds0_packed_xs1",
    ("sm_90a", 64, 8, 512, 512, 64, 132): "hq64_hkv8_d2_lookup_rounds2",
    ("sm_100a", 8, 1, 32, 32, 64, 148): "hq8_hkv1_vsub3_d",
    ("sm_100a", 8, 1, 128, 8, 64, 148): "hq8_hkv1_lookup_rounds2_wide_f32_loads1",
    ("sm_100a", 8, 1, 128, 128, 64, 148): "hq8_hkv1_vsub3_d",
    ("sm_100a", 8, 1, 256, 32, 64, 148): "hq8_hkv1_lookup_rounds2_wide_f32_loads1",
    (
        "sm_100a",
        8,
        1,
        256,
        256,
        64,
        148,
    ): "hq8_hkv1_vsub3_d_lookup_rounds0_single_weight1",
    (
        "sm_100a",
        8,
        1,
        512,
        512,
        64,
        148,
    ): "hq8_hkv1_vsub3_d_u2_lookup_rounds0_single_weight1_pt_window1",
    (
        "sm_100a",
        64,
        8,
        32,
        32,
        64,
        148,
    ): "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1",
    (
        "sm_100a",
        64,
        8,
        128,
        8,
        64,
        148,
    ): "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1",
    (
        "sm_100a",
        64,
        8,
        128,
        128,
        64,
        148,
    ): "hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1",
    (
        "sm_100a",
        64,
        8,
        128,
        1,
        64,
        148,
    ): "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1",
    ("sm_103a", 8, 1, 32, 32, 64, 148): "hq8_hkv1_vsub3_d",
    ("sm_103a", 8, 1, 32, 32, 64, 152): "hq8_hkv1_vsub3_d",
    (
        "sm_103a",
        64,
        8,
        32,
        32,
        64,
        152,
    ): "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1",
    (
        "sm_103a",
        64,
        8,
        128,
        128,
        64,
        148,
    ): "hq64_hkv8_d_u2_lookup_rounds0_pt_window1_packed_xs1",
}


@pytest.mark.parametrize("key,expected", sorted(_POLICY_TABLE.items()))
def test_stage_policy_table(key, expected):
    arch, hq, hkv, rows, reqs, page, sms = key
    stage = loader.stage_for(
        hq,
        hkv,
        arch=arch,
        num_rows=rows,
        num_requests=reqs,
        page_size=page,
        num_sms=sms,
    )
    assert stage == expected
    assert stage in loader.stages_for(arch)


@pytest.mark.parametrize("arch", loader.ARCHES)
def test_stage_policy_only_selects_shipped_builds(arch):
    shipped = set(loader.stages_for(arch))
    for hq, hkv in loader.SUPPORTED_HEADS:
        for page in (16, 32, 64, 128, 256, 512, 1024, 2048):
            for reqs in (1, 2, 4, 8, 16, 32, 64, 128, 256, 257, 512, 1024):
                # rows < reqs: requests without rows in this step
                for rows in sorted({reqs, 1, 32, 128, 148, 256, 512, 4096}):
                    for sms in (132, 148, 152, 160):
                        stage = loader.stage_for(
                            hq,
                            hkv,
                            arch=arch,
                            num_rows=rows,
                            num_requests=reqs,
                            page_size=page,
                            num_sms=sms,
                        )
                        assert stage in shipped, (
                            arch,
                            hq,
                            hkv,
                            rows,
                            reqs,
                            page,
                            sms,
                            stage,
                        )


def test_stage_policy_rejects_unsupported_heads():
    with pytest.raises(ValueError):
        loader.stage_for(
            16, 2, arch="sm_100a", num_rows=1, num_requests=1, page_size=64, num_sms=148
        )


# Points where the w20 condition (the clearing grid fits beside the row CTAs)
# and the eight-warp CLEAR_UNITS_PER_WARP=2 condition (a large clearing grid)
# both hold: few rows, many pages to clear.  w20 wins the eight-warp u2 rule and
# the policy must still select a shipped build.
_W20_U2_TABLE: dict[tuple[str, int, int, int, int, int, int], str] = {
    # (arch, Hq, Hkv, num_rows, num_requests, page_size, num_sms) -> stage
    (
        "sm_100a",
        64,
        8,
        16,
        16,
        512,
        148,
    ): "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1",
    (
        "sm_100a",
        64,
        8,
        8,
        4,
        2048,
        160,
    ): "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1",
    (
        "sm_100a",
        64,
        8,
        1,
        257,
        32,
        132,
    ): "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1",
    (
        "sm_103a",
        64,
        8,
        32,
        32,
        256,
        160,
    ): "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1",
    (
        "sm_103a",
        64,
        8,
        16,
        16,
        512,
        148,
    ): "hq64_hkv8_w20_vsub18_d_u2_lookup_rounds0_pt_window1",
    (
        "sm_103a",
        64,
        8,
        8,
        4,
        2048,
        160,
    ): "hq64_hkv8_w20_vsub18_lookup_rounds0_single_weight1_wide_f32_loads1",
}


@pytest.mark.parametrize("key,expected", sorted(_W20_U2_TABLE.items()))
def test_stage_policy_w20_and_u2_overlap_selects_shipped_build(key, expected):
    arch, hq, hkv, rows, reqs, page, sms = key
    shape = dict(
        arch=arch, num_rows=rows, num_requests=reqs, page_size=page, num_sms=sms
    )
    c = loader.STAGE_POLICY["constants"]
    units = loader.clear_units_per_request(page, hkv, c["clear_unit_chunks"])
    clear_ctas = (reqs * units + 3) // 4
    # both lever conditions hold at this point
    assert clear_ctas >= c["sm100_u2_min_clear_ctas"]
    assert clear_ctas <= (sms - rows) * c["clear_ctas_per_free_sm"]
    knobs = loader.policy_knobs(hq, hkv, **shape)
    assert knobs.get("warps_per_row") == c["w20"]
    assert knobs.get("v_sub_base") == c["w20_v_sub_base"]
    stage = loader.stage_for(hq, hkv, **shape)
    assert stage in loader.stages_for(arch)
    assert stage == expected


# ----------------------------------------------------------------------------------------------
# Package hygiene (CPU): the vendorable package imports itself, torch and tvm_ffi only at
# module level and reads no environment variables.  Function-level lazy imports are the
# documented exceptions: ``torch`` inside the registry's device helpers and FlashInfer's JIT
# builder inside the byte-pinned ``cake_build.py`` JIT path (the shared-loader exception of
# the Cake package form).


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module is None:
                    names.update(f".{alias.name}" for alias in node.names)
                else:
                    names.add(f".{node.module.split('.')[0]}")
            else:
                assert node.module is not None
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "filename,allowed",
    [
        (
            "cake_jit.py",
            {"__future__", "functools", "typing", "tvm_ffi", ".cake_build"},
        ),
        (
            "__init__.py",
            {"__future__", "functools", "typing", "math", "torch", ".cake_jit"},
        ),
    ],
)
def test_package_module_level_imports_are_whitelisted(filename, allowed):
    assert _module_level_imports(PACKAGE_DIR / filename) <= allowed


def test_cake_build_imports_only_the_standard_library():
    names = _module_level_imports(PACKAGE_DIR / "cake_build.py")
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names


@pytest.mark.parametrize("filename", ["cake_jit.py", "__init__.py", "cake_build.py"])
def test_package_reads_no_environment_variables(filename):
    text = (PACKAGE_DIR / filename).read_text(encoding="utf-8")
    assert "os.environ" not in text
    assert "getenv" not in text
