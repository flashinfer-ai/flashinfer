# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Public batch-prefill dispatch and numerical tests for persistent Rubin GQA.

The reference consumes the exact E4M3 tensors, computes in FP32 with TF32 off,
and checks ordinary attention plus base2 LSE independently. Unsupported empty
KV and fully masked causal rows must be rejected before compilation/launch.
"""

import functools
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from flashinfer.attention.cute_dsl import rubin_gqa
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper


@pytest.fixture
def sm107():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Rubin GQA requires SM107")
    pytest.importorskip("cutlass")
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = previous_matmul
    torch.backends.cudnn.allow_tf32 = previous_cudnn


@pytest.fixture
def gqa_compiler(sm107):
    # Import only after the GPU fixture; ordinary adapter import stays lazy.
    from flashinfer.cute_dsl.attention.fmha import compile_rubin_gqa

    return compile_rubin_gqa


def inputs(
    batch=2,
    q_len=65,
    kv_len=193,
    heads=16,
    ratio=2,
    seed=31,
    device="cuda",
    head_dim=192,
    dtype=torch.float8_e4m3fn,
):
    generator = torch.Generator(device=device).manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, generator=generator, device=device).to(dtype)

    return (
        rand((batch, q_len, heads, head_dim)),
        rand((batch, kv_len, heads // ratio, head_dim)),
        rand((batch, kv_len, heads // ratio, 128)),
    )


def outputs(q, with_lse=True):
    out = torch.full(
        (*q.shape[:-1], 128), math.nan, device=q.device, dtype=torch.bfloat16
    )
    lse = (
        torch.full(q.shape[:-1], math.nan, device=q.device, dtype=torch.float32)
        if with_lse
        else None
    )
    return out, lse


def reference(
    q,
    k,
    v,
    *,
    causal=False,
    sm_scale=None,
    scale_q=1.0,
    scale_k=1.0,
    scale_v=1.0,
    scale_o=1.0,
):
    """Explicit grouped-head mapping and bottom-right causal masking."""
    ratio = q.shape[2] // k.shape[2]
    heads = torch.arange(q.shape[2], device=q.device) // ratio
    keys = k.float().index_select(2, heads)
    values = v.float().index_select(2, heads)
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), keys)
    logits *= (
        (1 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale)
        * scale_q
        * scale_k
    )
    if causal:
        query_position = torch.arange(q.shape[1], device=q.device)[:, None]
        key_position = torch.arange(k.shape[1], device=q.device)[None, :]
        logits.masked_fill_(
            key_position > query_position + k.shape[1] - q.shape[1], -math.inf
        )
    lse = torch.logsumexp(logits, dim=-1).transpose(1, 2) / math.log(2)
    out = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(logits, dim=-1), values)
    return out * (scale_v / scale_o), lse


def assert_outputs(out, lse, expected, expected_lse):
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=0.05, rtol=0.05)
    rel_l2 = torch.linalg.vector_norm(
        out.float() - expected
    ) / torch.linalg.vector_norm(expected).clamp_min(1e-12)
    assert rel_l2.item() <= 0.04
    if lse is not None:
        assert torch.isfinite(lse).all()
        torch.testing.assert_close(lse, expected_lse, atol=0.05, rtol=0.005)


def make_batch(q, k, *, causal=False, sm_scale=None, graph=False):
    q_ptr = torch.arange(q.shape[0] + 1, dtype=torch.int32) * q.shape[1]
    kv_ptr = torch.arange(k.shape[0] + 1, dtype=torch.int32) * k.shape[1]
    graph_q_ptr, graph_kv_ptr = q_ptr.to(q.device), kv_ptr.to(q.device)
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=q.device),
        backend="cute-dsl-rubin",
        use_cuda_graph=graph,
        qo_indptr_buf=graph_q_ptr if graph else None,
        kv_indptr_buf=graph_kv_ptr if graph else None,
    )
    attn.plan(
        q_ptr,
        kv_ptr,
        q.shape[2],
        k.shape[2],
        192,
        head_dim_vo=128,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    return attn, graph_q_ptr, graph_kv_ptr


def run_public(
    q,
    k,
    v,
    out,
    *,
    lse=None,
    causal=False,
    sm_scale=None,
    scale_q=1.0,
    scale_k=1.0,
    scale_v=1.0,
    scale_o=1.0,
    attn=None,
    enable_pdl=None,
):
    if attn is None:
        attn, _, _ = make_batch(q, k, causal=causal, sm_scale=sm_scale)
    flat_out = out.flatten(0, 1)
    flat_lse = lse.flatten(0, 1) if lse is not None else None
    result = attn.run(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        out=flat_out,
        lse=flat_lse,
        return_lse=lse is not None,
        q_scale=scale_q,
        k_scale=scale_k,
        v_scale=scale_v,
        o_scale=scale_o,
        enable_pdl=enable_pdl,
    )
    if lse is not None:
        assert result[0] is flat_out and result[1] is flat_lse
    else:
        assert result is flat_out
    return out


def reject_launch(*args):
    pytest.fail("Invalid input reached kernel launch")


def private_call(q, k, v, out, **kwargs):
    # Only invalid-input tests use a sentinel; numerical tests use public dispatch.
    return rubin_gqa._run_rubin_gqa_prefill(
        q, k, v, out, kernel=(reject_launch, None), **kwargs
    )


@pytest.mark.parametrize("ratio", [2, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_lse", [False, True])
def test_reference_grouped_partial_tiles(sm107, ratio, causal, with_lse):
    q, k, v = inputs(ratio=ratio)
    out, lse = outputs(q, with_lse)
    scales = dict(
        scale_q=0.75,
        scale_k=1.25,
        scale_v=0.625,
        scale_o=1.5,
        sm_scale=0.9 / math.sqrt(192),
    )
    assert run_public(q, k, v, out, causal=causal, lse=lse, **scales) is out
    assert_outputs(out, lse, *reference(q, k, v, causal=causal, **scales))


@pytest.mark.parametrize(
    "q_len,kv_len,causal", [(1, 17, False), (128, 512, True), (129, 257, True)]
)
def test_default_scale_and_boundaries(sm107, q_len, kv_len, causal):
    q, k, v = inputs(batch=1, q_len=q_len, kv_len=kv_len)
    out, lse = outputs(q)
    run_public(q, k, v, out, causal=causal, lse=lse)
    assert_outputs(out, lse, *reference(q, k, v, causal=causal))


@pytest.mark.parametrize("kv_len", [1, 17, 127, 128, 129, 255])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_mask_real_and_padded_tile_boundaries(sm107, kv_len, causal):
    """Unused producer iterations and residual tiles must not add softmax mass."""
    q, k, v = inputs(batch=1, q_len=1, kv_len=kv_len, ratio=2)
    out, lse = outputs(q)
    run_public(q, k, v, out, causal=causal, lse=lse)
    assert_outputs(out, lse, *reference(q, k, v, causal=causal))


def test_unindexed_device_is_resolved_before_cache(gqa_compiler, monkeypatch):
    seen = []

    @functools.cache
    def compile_spy(*args):
        seen.append(args[-1])
        return object(), object()

    monkeypatch.setattr(gqa_compiler, "_compile_rubin_gqa_kernel_cached", compile_spy)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 7)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    first = gqa_compiler.compile_rubin_gqa_kernel(
        16, 8, False, True, torch.device("cuda")
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    second = gqa_compiler.compile_rubin_gqa_kernel(
        16, 8, False, True, torch.device("cuda")
    )
    assert first[0] is not second[0]
    assert seen == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_adapter_import_needs_no_cuda_or_cute():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "flashinfer/attention/cute_dsl/rubin_gqa.py"
    )
    code = """
import importlib.util, sys, torch
def forbidden(*args, **kwargs):
    raise AssertionError('adapter import initialized CUDA')
torch.cuda._lazy_init = forbidden
sys.modules['cutlass'] = None
spec = importlib.util.spec_from_file_location('isolated_rubin_gqa', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert callable(module._run_rubin_gqa_prefill)
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-c", code, str(module_path)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_tensors_rejected_before_cuda_queries(monkeypatch):
    q, k, v = inputs(batch=1, q_len=1, kv_len=1, device="cpu")
    out, _ = outputs(q, False)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU input validation touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    with pytest.raises(ValueError):
        private_call(q, k, v, out)


@pytest.mark.parametrize(
    "name,kind",
    [
        ("q", "dtype"),
        ("k", "strides"),
        ("v", "alignment"),
        ("out", "dtype"),
        ("out", "alignment"),
        ("lse", "dtype"),
        ("lse", "strides"),
    ],
)
def test_private_tensor_contracts_rejected_before_launch(sm107, name, kind):
    q, k, v = inputs(batch=1)
    out, lse = outputs(q)
    tensors = dict(q=q, k=k, v=v, out=out, lse=lse)
    tensor = tensors[name]
    if kind == "dtype":
        tensors[name] = tensor.to(torch.float32 if name != "lse" else torch.bfloat16)
    elif kind == "strides":
        tensors[name] = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        storage = torch.empty(
            tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device
        )
        tensors[name] = storage[1:].view(tensor.shape)

    with pytest.raises(ValueError):
        private_call(**tensors)


@pytest.mark.parametrize(
    "kind",
    [
        "rank",
        "batch",
        "kv_length",
        "output_shape",
        "lse_shape",
        "device",
        "alias",
    ],
)
def test_shape_and_storage_guards(sm107, kind):
    q, k, v = inputs()
    out, lse = outputs(q)
    if kind == "rank":
        q = q.flatten(0, 1)
    elif kind == "batch":
        k = k[:1]
    elif kind == "kv_length":
        v = v[:, :-1].contiguous()
    elif kind == "output_shape":
        out = out[..., :64].contiguous()
    elif kind == "lse_shape":
        lse = lse[:, :, :8].contiguous()
    elif kind == "device":
        out = out.cpu()
    elif kind == "alias":
        out = k.view(torch.bfloat16).flatten()[: out.numel()].view(out.shape)

    with pytest.raises(ValueError):
        private_call(q, k, v, out, lse=lse)


@pytest.mark.parametrize(
    "name,value",
    [
        ("sm_scale", 0.0),
        ("scale_q", -1.0),
        ("scale_k", math.nan),
        ("scale_v", math.inf),
        ("scale_o", True),
        ("scale_q", "1"),
        ("scale_k", torch.tensor(1.0)),
    ],
)
def test_invalid_scales(sm107, name, value):
    q, k, v = inputs(batch=1, q_len=1, kv_len=1)
    out, _ = outputs(q, False)

    with pytest.raises(ValueError):
        private_call(q, k, v, out, **{name: value})


@pytest.mark.parametrize("sm_scale,scale_q", [(3e38, 1.0), (1e38, 3.0)])
def test_public_base2_scale_overflow_rejected_without_writes(sm107, sm_scale, scale_q):
    q, k, v = inputs(batch=1, q_len=1, kv_len=128, heads=2)
    q.zero_()
    k.zero_()
    v.fill_(1)
    attn, _, _ = make_batch(q, k, sm_scale=sm_scale)
    out, lse = outputs(q)
    out.fill_(19)
    lse.fill_(19)
    with pytest.raises(ValueError, match="base-2 score scale"):
        run_public(q, k, v, out, lse=lse, attn=attn, scale_q=scale_q)
    torch.testing.assert_close(out, torch.full_like(out, 19.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 19.0))


def test_public_large_finite_base2_scale(sm107):
    q, k, v = inputs(batch=1, q_len=1, kv_len=128, heads=2)
    q.zero_()
    k.zero_()
    v.fill_(1)
    # Stay just below overflow after the kernel's base-2 conversion.
    sm_scale = torch.finfo(torch.float32).max / math.log2(math.e) * (1 - 2**-20)
    out, lse = outputs(q)
    run_public(q, k, v, out, lse=lse, sm_scale=sm_scale)
    assert_outputs(out, lse, torch.ones_like(out).float(), torch.full_like(lse, 7.0))


def test_public_grouped_grid_limit_preserves_plan_and_buffers(sm107):
    # 1023 * 64 is the largest supported batch at this KV head count.
    q, k, v = inputs(batch=1023, q_len=1, kv_len=1, heads=128)
    q.zero_()
    k.zero_()
    v.fill_(1)
    attn, _, _ = make_batch(q, k)
    out, lse = outputs(q)
    run_public(q, k, v, out, lse=lse, attn=attn)
    expected = torch.ones_like(out).float(), torch.zeros_like(lse)
    assert_outputs(out, lse, *expected)

    params = dict(
        qo_indptr=torch.arange(1025, dtype=torch.int32),
        kv_indptr=torch.arange(1025, dtype=torch.int32),
        num_qo_heads=128,
        num_kv_heads=64,
        head_dim_qk=192,
        head_dim_vo=128,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="batch_size.*num_kv_heads.*65535"):
        attn.plan(**params)
    out.fill_(19)
    lse.fill_(19)
    run_public(q, k, v, out, lse=lse, attn=attn)
    assert_outputs(out, lse, *expected)

    # An invalid first graph plan must not overwrite caller-owned metadata.
    q_ptr = torch.full((1025,), 19, dtype=torch.int32, device=q.device)
    kv_ptr = torch.full_like(q_ptr, 23)
    graph_attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=q.device),
        backend="cute-dsl-rubin",
        use_cuda_graph=True,
        qo_indptr_buf=q_ptr,
        kv_indptr_buf=kv_ptr,
    )
    with pytest.raises(ValueError, match="batch_size.*num_kv_heads.*65535"):
        graph_attn.plan(**params)
    torch.testing.assert_close(q_ptr, torch.full_like(q_ptr, 19))
    torch.testing.assert_close(kv_ptr, torch.full_like(kv_ptr, 23))


def test_architecture_guard_applies_even_after_cache_hit(gqa_compiler, monkeypatch):
    compiler = gqa_compiler.compile_rubin_gqa_kernel
    compiler(16, 4, False, False, torch.device("cuda"))
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (10, 0)
    )
    with pytest.raises(ValueError, match="SM107"):
        compiler(16, 4, False, False, torch.device("cuda"))


def test_public_grouped_pdl_hint(sm107):
    q, k, v = inputs(batch=1, q_len=17, kv_len=129)
    out, lse = outputs(q)
    run_public(q, k, v, out, lse=lse, enable_pdl=True)
    assert_outputs(out, lse, *reference(q, k, v))


@pytest.mark.parametrize("lse_base", ["ln", "log2"])
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
def test_public_lse_layout_and_base(sm107, lse_base, lse_layout):
    q, k, v = inputs(q_len=17, kv_len=129, ratio=4)
    attn, _, _ = make_batch(q, k, causal=True)
    out = torch.empty(
        (*q.flatten(0, 1).shape[:2], 128), dtype=torch.bfloat16, device=q.device
    )
    lse_shape = out.shape[:2] if lse_layout == "NH" else tuple(reversed(out.shape[:2]))
    lse = torch.empty(lse_shape, device=q.device)
    result, result_lse = attn.run(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        out=out,
        lse=lse,
        return_lse=True,
        lse_base=lse_base,
        lse_layout=lse_layout,
        q_scale=0.75,
        k_scale=1.25,
        v_scale=0.625,
        o_scale=1.5,
    )
    assert result is out and result_lse is lse
    expected, expected_lse = reference(
        q, k, v, causal=True, scale_q=0.75, scale_k=1.25, scale_v=0.625, scale_o=1.5
    )
    expected_lse = expected_lse.flatten(0, 1)
    if lse_base == "ln":
        expected_lse *= math.log(2)
    if lse_layout == "HN":
        expected_lse = expected_lse.T
    assert_outputs(out, lse, expected.flatten(0, 1), expected_lse)


def test_public_plan_accepts_replacement_tensors_and_reuses_output(sm107):
    q, k, v = inputs(batch=1, ratio=2)
    attn, _, _ = make_batch(q, k)
    result, result_lse = attn.run(
        q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), return_lse=True
    )
    expected, expected_lse = reference(q, k, v)
    assert_outputs(
        result, result_lse, expected.flatten(0, 1), expected_lse.flatten(0, 1)
    )
    out, lse = outputs(q)
    for seed in (41, 42):
        # Fresh allocations use one plan; only CUDA graph replay binds addresses.
        q, k, v = inputs(batch=1, ratio=2, seed=seed)
        run_public(q, k, v, out, lse=lse, attn=attn)
        assert_outputs(out, lse, *reference(q, k, v))
    run_public(q, k, v, out, attn=attn)
    assert_outputs(out, None, *reference(q, k, v))


@pytest.mark.parametrize("ratio", [1, 2])
def test_review_graph_plan_retry_after_preparation_failure(sm107, ratio):
    q, k, v = inputs(batch=2, q_len=17, kv_len=129, ratio=ratio)
    q_ptr = torch.tensor([0, 17, 34], dtype=torch.int32)
    kv_ptr = torch.tensor([0, 129, 258], dtype=torch.int32)
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=q.device),
        backend="cute-dsl-rubin",
        use_cuda_graph=True,
        qo_indptr_buf=q_ptr.to(q.device),
        kv_indptr_buf=kv_ptr.to(q.device),
    )
    params = dict(
        qo_indptr=q_ptr,
        kv_indptr=kv_ptr,
        num_qo_heads=16,
        num_kv_heads=16 // ratio,
        head_dim_qk=192,
        head_dim_vo=128,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )

    wrong_batch_params = dict(
        params,
        # A smaller invalid first batch must not install a too-small Q capacity.
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 129], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="(?i)(batch size|batch_size)"):
        attn.plan(**wrong_batch_params)

    # Retry must prepare a usable plan after the first attempt failed to finish.
    attn.plan(**params)
    out, lse = outputs(q)
    call = functools.partial(run_public, q, k, v, out, lse=lse, attn=attn)
    expected = reference(q, k, v, causal=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    assert_outputs(out, lse, *expected)
    out.fill_(math.nan)
    lse.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize()
    assert_outputs(out, lse, *expected)


@pytest.mark.parametrize(
    "indptr_device,initial_scale,replan_scale",
    [
        pytest.param("cpu", None, None, id="cpu-default"),
        pytest.param("cuda", None, 1 / math.sqrt(192), id="cuda-default-to-explicit"),
        pytest.param("cpu", 1 / math.sqrt(192), None, id="cpu-explicit-to-default"),
    ],
)
def test_grouped_graph_equivalent_replan(
    sm107, indptr_device, initial_scale, replan_scale
):
    q, k, v = inputs(batch=2, q_len=17, kv_len=129, ratio=2)
    attn, q_ptr, kv_ptr = make_batch(
        q, k, causal=True, sm_scale=initial_scale, graph=True
    )
    original_q, original_kv = q_ptr.clone(), kv_ptr.clone()

    def replan():
        # New metadata allocations may use either host or device storage.
        attn.plan(
            torch.tensor([0, 17, 34], dtype=torch.int32, device=indptr_device),
            torch.tensor([0, 129, 258], dtype=torch.int32, device=indptr_device),
            16,
            8,
            192,
            head_dim_vo=128,
            causal=True,
            sm_scale=replan_scale,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
        torch.testing.assert_close(q_ptr, original_q)
        torch.testing.assert_close(kv_ptr, original_kv)

    replan()
    out, lse = outputs(q)
    call = functools.partial(run_public, q, k, v, out, lse=lse, attn=attn)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    assert_outputs(out, lse, *reference(q, k, v, causal=True))

    for seed in (101, 102):
        replan()
        new_inputs = inputs(batch=2, q_len=17, kv_len=129, ratio=2, seed=seed)
        for tensor, replacement in zip((q, k, v), new_inputs, strict=True):
            tensor.copy_(replacement)
        expected = reference(q, k, v, causal=True)
        # Graph replay must read current contents from the captured addresses.
        out.fill_(math.nan)
        lse.fill_(math.nan)
        graph.replay()
        torch.cuda.synchronize()
        assert_outputs(out, lse, *expected)
        out.fill_(math.nan)
        lse.fill_(math.nan)
        call()
        assert_outputs(out, lse, *expected)


@pytest.mark.parametrize(
    "change",
    [
        "q-length",
        "kv-length",
        "batch-size",
        "q-heads",
        "kv-heads",
        "causal",
        "sm-scale",
        "equal-heads",
        "window-left",
        "q-indptr-origin",
    ],
)
def test_grouped_graph_replan_rejected_without_mutating_buffers(sm107, change):
    q, k, v = inputs(batch=2, q_len=17, kv_len=129, ratio=2)
    attn, q_ptr, kv_ptr = make_batch(q, k, causal=True, graph=True)
    original_q, original_kv = q_ptr.clone(), kv_ptr.clone()
    params = dict(
        qo_indptr=torch.tensor([0, 17, 34], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 129, 258], dtype=torch.int32),
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim_qk=192,
        head_dim_vo=128,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    if change == "q-length":
        params["qo_indptr"] = torch.tensor([0, 8, 16], dtype=torch.int32)
    elif change == "kv-length":
        params["kv_indptr"] = torch.tensor([0, 64, 128], dtype=torch.int32)
    elif change == "batch-size":
        params["qo_indptr"] = torch.tensor([0, 17], dtype=torch.int32)
        params["kv_indptr"] = torch.tensor([0, 129], dtype=torch.int32)
    elif change == "q-heads":
        params["num_qo_heads"] = 32
    elif change == "kv-heads":
        params["num_kv_heads"] = 4
    elif change == "causal":
        params["causal"] = False
    elif change == "sm-scale":
        params["sm_scale"] = 0.5 / math.sqrt(192)
    elif change == "equal-heads":
        params["num_kv_heads"] = 16
    elif change == "window-left":
        params["window_left"] = 8
    elif change == "q-indptr-origin":
        params["qo_indptr"] = torch.tensor([-1, 16, 33], dtype=torch.int32)
    unsupported = change == "window-left"
    expected_error = NotImplementedError if unsupported else ValueError
    error_pattern = "does not support" if unsupported else "(?i)(re.?plan|fixed)"
    if change == "q-indptr-origin":
        error_pattern = "start at zero"
    out, lse = outputs(q)
    call = functools.partial(run_public, q, k, v, out, lse=lse, attn=attn)
    expected = reference(q, k, v, causal=True)

    def reject_replan():
        with pytest.raises(expected_error, match=error_pattern):
            attn.plan(**params)
        torch.testing.assert_close(q_ptr, original_q)
        torch.testing.assert_close(kv_ptr, original_kv)
        # Eager run checks retained host-side state after every refusal.
        out.fill_(math.nan)
        lse.fill_(math.nan)
        call()
        assert_outputs(out, lse, *expected)

    reject_replan()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    reject_replan()
    # Replay checks that a refused plan did not disturb captured execution.
    out.fill_(math.nan)
    lse.fill_(math.nan)
    graph.replay()
    torch.cuda.synchronize()
    assert_outputs(out, lse, *expected)


def test_public_explicit_rubin_replan_between_equal_and_grouped_heads(sm107):
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
    )
    for step, (ratio, batch, q_len, kv_len, causal) in enumerate(
        [
            (2, 2, 17, 129, True),
            (1, 1, 33, 193, False),
            (8, 2, 65, 257, True),
            (4, 2, 29, 131, False),
        ]
    ):
        q, k, v = inputs(
            batch=batch,
            q_len=q_len,
            kv_len=kv_len,
            ratio=ratio,
            seed=91 + step,
        )
        sm_scale = (0.8 + 0.03 * step) / math.sqrt(192)
        attn.plan(
            torch.arange(batch + 1, dtype=torch.int32) * q_len,
            torch.arange(batch + 1, dtype=torch.int32) * kv_len,
            q.shape[2],
            k.shape[2],
            192,
            head_dim_vo=128,
            causal=causal,
            sm_scale=sm_scale,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
        out, lse = outputs(q)
        run_public(q, k, v, out, lse=lse, attn=attn)
        assert_outputs(out, lse, *reference(q, k, v, causal=causal, sm_scale=sm_scale))


def test_public_explicit_rubin_requires_sm107_during_plan(sm107, monkeypatch):
    from flashinfer import prefill
    from flashinfer.attention.cute_dsl import rubin

    monkeypatch.setattr(prefill, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(rubin, "get_compute_capability", lambda device: (10, 0))
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
    )
    with pytest.raises(ValueError, match="SM107"):
        attn.plan(
            torch.tensor([0, 17], dtype=torch.int32),
            torch.tensor([0, 129], dtype=torch.int32),
            16,
            8,
            192,
            head_dim_vo=128,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(128, 128), (192, 64)])
def test_public_explicit_rubin_requires_d192_v128_during_plan(
    sm107, head_dim_qk, head_dim_vo
):
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
    )
    with pytest.raises(ValueError, match="head_dim_qk=192 and head_dim_vo=128"):
        attn.plan(
            torch.tensor([0, 17], dtype=torch.int32),
            torch.tensor([0, 129], dtype=torch.int32),
            16,
            8,
            head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )


def test_public_cute_dsl_d192_does_not_select_rubin(sm107, monkeypatch):
    from flashinfer.attention.cute_dsl.rubin import RubinPrefillPlan

    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl",
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Generic CuTe plan reached Rubin validation")

    monkeypatch.setattr(RubinPrefillPlan, "validate", forbidden)
    with pytest.raises(NotImplementedError, match="head_dim_vo == head_dim_qk"):
        attn.plan(
            torch.tensor([0, 17], dtype=torch.int32),
            torch.tensor([0, 129], dtype=torch.int32),
            16,
            8,
            192,
            head_dim_vo=128,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )


@pytest.mark.parametrize(
    "custom_jit",
    [
        pytest.param({"jit_args": ["custom_attention"]}, id="args"),
        pytest.param({"jit_kwargs": {"extra_cuda_cflags": ["-lineinfo"]}}, id="kwargs"),
    ],
)
def test_public_custom_jit_keeps_selected_backend(sm107, custom_jit):
    q, k, v = inputs(batch=1, q_len=17, kv_len=129)
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
        **custom_jit,
    )
    attn.plan(
        torch.tensor([0, 17], dtype=torch.int32),
        torch.tensor([0, 129], dtype=torch.int32),
        16,
        8,
        192,
        head_dim_vo=128,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    out, lse = outputs(q)
    run_public(q, k, v, out, lse=lse, attn=attn)
    expected, expected_lse = reference(q, k, v, causal=True)
    assert_outputs(out, lse, expected, expected_lse)


@pytest.mark.parametrize(
    "q_lens,kv_lens,causal",
    [
        ([1, 2], [17, 17], False),
        ([1, 1], [17, 18], False),
        ([0, 0], [17, 17], False),
        ([1, 1], [0, 0], False),
        ([17, 17], [1, 1], True),
    ],
)
def test_public_grouped_lengths_rejected(sm107, q_lens, kv_lens, causal):
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
    )
    q_ptr = torch.tensor([0, *q_lens], dtype=torch.int32).cumsum(0).int()
    kv_ptr = torch.tensor([0, *kv_lens], dtype=torch.int32).cumsum(0).int()
    with pytest.raises(
        (ValueError, NotImplementedError),
        match="(?i)(Rubin|GQA|uniform|positive|causal)",
    ):
        attn.plan(
            q_ptr,
            kv_ptr,
            16,
            8,
            192,
            head_dim_vo=128,
            causal=causal,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )


@pytest.mark.parametrize("ratio", [1, 2], ids=["equal", "grouped"])
@pytest.mark.parametrize(
    "o_data_type",
    [None, torch.bfloat16, torch.uint8, torch.float8_e4m3fn],
    ids=["omitted", "bf16", "uint8", "fp8"],
)
def test_public_output_dtype(sm107, ratio, o_data_type):
    q, k, v = inputs(batch=1, q_len=17, kv_len=129, ratio=ratio)
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=q.device),
        backend="cute-dsl-rubin",
    )
    params = dict(
        qo_indptr=torch.tensor([0, 17], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 129], dtype=torch.int32),
        num_qo_heads=q.shape[2],
        num_kv_heads=k.shape[2],
        head_dim_qk=192,
        head_dim_vo=128,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
    )
    if o_data_type is not None:
        params["o_data_type"] = o_data_type
    if o_data_type in (torch.uint8, torch.float8_e4m3fn):
        with pytest.raises(ValueError, match="(?i)(BF16|bfloat16)"):
            attn.plan(**params)
        return
    attn.plan(**params)
    out, lse = attn.run(
        q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), return_lse=True
    )
    assert out.dtype == torch.bfloat16
    assert_outputs(
        out.view(*q.shape[:-1], 128),
        lse.view(q.shape[:-1]),
        *reference(q, k, v),
    )


@pytest.mark.parametrize(
    "option,value",
    [
        ("window_left", 8),
        ("pos_encoding_mode", "ALIBI"),
        ("logits_soft_cap", 10.0),
        ("use_fp16_qk_reduction", True),
        ("fixed_split_size", 16),
        ("q_data_type", torch.bfloat16),
        ("kv_data_type", torch.bfloat16),
        ("o_data_type", torch.float16),
        ("num_kv_heads", 1),
        ("num_qo_heads", 15),
        ("custom_mask", torch.ones(17 * 129, dtype=torch.bool)),
        ("qo_indptr", torch.tensor([0.0, 17.0])),
    ],
)
def test_public_d192_unsupported_options(sm107, option, value):
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
    )
    params = dict(
        qo_indptr=torch.tensor([0, 17], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 129], dtype=torch.int32),
        num_qo_heads=16,
        num_kv_heads=8,
        head_dim_qk=192,
        head_dim_vo=128,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    params[option] = value
    with pytest.raises(
        (ValueError, NotImplementedError), match="(?i)(Rubin|192|cute-dsl|integer)"
    ):
        attn.plan(**params)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("method", ["forward", "forward_return_lse"])
@pytest.mark.parametrize("change", ["causal", "sm_scale"])
def test_review_legacy_overrides_rejected_without_state_mutation(
    sm107, ratio, method, change
):
    q, k, v = inputs(batch=1, q_len=17, kv_len=129, ratio=ratio)
    attn, _, _ = make_batch(q, k, causal=True)
    before = attn._causal, attn._sm_scale
    controls = dict(causal=True, sm_scale=None)
    controls[change] = False if change == "causal" else 0.5 / math.sqrt(192)
    with pytest.raises(ValueError, match="(?i)(causal|scale|plan)"):
        getattr(attn, method)(
            q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), **controls
        )
    assert (attn._causal, attn._sm_scale) == before
    out, lse = outputs(q)
    run_public(q, k, v, out, lse=lse, attn=attn)
    assert_outputs(out, lse, *reference(q, k, v, causal=True))


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("method", ["forward", "forward_return_lse"])
def test_review_legacy_default_scale_equivalence(sm107, ratio, method):
    q, k, v = inputs(batch=1, q_len=17, kv_len=129, ratio=ratio)
    # Test both explicit-to-default and default-to-explicit normalization.
    for planned, override in [(None, 1 / math.sqrt(192)), (1 / math.sqrt(192), None)]:
        attn, _, _ = make_batch(q, k, causal=True, sm_scale=planned)
        result = getattr(attn, method)(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            causal=True,
            sm_scale=override,
        )
        out, lse = result if method == "forward_return_lse" else (result, None)
        assert_outputs(
            out.view(*q.shape[:-1], 128),
            lse.view(q.shape[:-1]) if lse is not None else None,
            *reference(q, k, v, causal=True),
        )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("destination", ["q", "kv"])
def test_review_graph_indptr_destination_dtype_rejected_before_copy(
    sm107, ratio, destination
):
    source_q = torch.tensor([0, 129, 258], dtype=torch.int32)
    source_kv = torch.tensor([0, 257, 514], dtype=torch.int32)
    q_ptr = torch.full(
        (3,), 7, device="cuda", dtype=torch.uint8 if destination == "q" else torch.int32
    )
    kv_ptr = torch.full(
        (3,),
        11,
        device="cuda",
        dtype=torch.uint8 if destination == "kv" else torch.int32,
    )
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
        use_cuda_graph=True,
        qo_indptr_buf=q_ptr,
        kv_indptr_buf=kv_ptr,
    )
    with pytest.raises(ValueError, match="(?i)(int32|int64|dtype|integer)"):
        attn.plan(
            source_q,
            source_kv,
            16,
            16 // ratio,
            192,
            head_dim_vo=128,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
    # The valid source offsets exceed uint8; silent copies would truncate them.
    torch.testing.assert_close(q_ptr, torch.full_like(q_ptr, 7))
    torch.testing.assert_close(kv_ptr, torch.full_like(kv_ptr, 11))


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_review_graph_indptr_destination_integer_types(sm107, ratio, dtype):
    q, k, v = inputs(batch=2, q_len=129, kv_len=257, ratio=ratio)
    source_q = torch.tensor([0, 129, 258], dtype=torch.int32)
    source_kv = torch.tensor([0, 257, 514], dtype=torch.int32)
    q_ptr = torch.empty(3, device="cuda", dtype=dtype)
    kv_ptr = torch.empty(3, device="cuda", dtype=dtype)
    attn = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-rubin",
        use_cuda_graph=True,
        qo_indptr_buf=q_ptr,
        kv_indptr_buf=kv_ptr,
    )
    attn.plan(
        source_q,
        source_kv,
        16,
        16 // ratio,
        192,
        head_dim_vo=128,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
    )
    torch.testing.assert_close(q_ptr, source_q.to(device="cuda", dtype=dtype))
    torch.testing.assert_close(kv_ptr, source_kv.to(device="cuda", dtype=dtype))
    out, lse = outputs(q)
    run_public(q, k, v, out, lse=lse, attn=attn)
    assert_outputs(out, lse, *reference(q, k, v))
