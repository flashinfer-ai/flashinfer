"""Stride contracts for persistent BatchAttention.

NVFP4 LSE uses an absolute tolerance established against the independent
reference before specializing data offsets. Only the known SM120/121
cooperative-launch and head-dimension-256 shared-memory limits are expected.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability, is_sm100a_supported
from tests.test_helpers.paged_kv import make_paged_kv_cache_pair
from tests.test_helpers.utils_fp4 import create_nvfp4_kv, nvfp4_to_float


def _metadata():
    # One decode and one prefill request, both with partial final pages.
    # Reordered pages prevent an accidentally contiguous-request reference.
    cpu = ([0, 1, 130], [0, 3, 13], [0, *range(12, 0, -1)], [35, 147])
    gpu = tuple(torch.tensor(x, dtype=torch.int32, device="cuda") for x in cpu)
    return cpu, gpu


def _gather_reference(layout, q, k, v, meta, causal=True):
    """Independent CPU FP32 attention; no FlashInfer attention kernel is used."""
    qptr, kptr, indices, lengths = meta
    dim = q.shape[-1]
    q = q.detach().float().cpu()
    k, v = k.detach().float().cpu(), v.detach().float().cpu()
    if layout == "HND":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    output, lse = [], []
    for request, length in enumerate(lengths):
        pages = indices[kptr[request] : kptr[request + 1]]
        kr = k[pages].reshape(-1, 2, dim)[:length].repeat_interleave(4, dim=1)
        vr = v[pages].reshape(-1, 2, dim)[:length].repeat_interleave(4, dim=1)
        qr = q[qptr[request] : qptr[request + 1]]
        scores = torch.einsum("qhd,khd->hqk", qr, kr) / math.sqrt(dim)
        if causal:
            allowed = torch.arange(length)[None, :] <= (
                torch.arange(qr.shape[0])[:, None] + length - qr.shape[0]
            )
            scores.masked_fill_(~allowed[None], -torch.inf)
        output.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vr))
        lse.append((torch.logsumexp(scores, -1) / math.log(2.0)).transpose(0, 1))
    return torch.cat(output).to("cuda"), torch.cat(lse).to("cuda")


def _inputs(layout="NHD", dtype=torch.bfloat16, dim=64):
    torch.manual_seed(20260917)
    cpu, gpu = _metadata()
    q = torch.randn(130, 8, dim, dtype=dtype, device="cuda")
    shape = (13, 16, 2, dim) if layout == "NHD" else (13, 2, 16, dim)
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")
    expected = _gather_reference(layout, q, k, v, cpu)
    return q, k, v, cpu, gpu, expected


def _plan(
    gpu, layout="NHD", dtype=torch.bfloat16, dim=64, *, kv_dtype=None, causal=True
):
    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        *gpu,
        8,
        2,
        dim,
        dim,
        16,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=kv_dtype or dtype,
    )
    return wrapper


def _fresh_lazy_wrapper(gpu, layout="NHD"):
    from flashinfer.attention import _core

    _core.get_holistic_attention_independent_module.cache_clear()
    wrapper = _plan(gpu, layout)
    assert not wrapper._independent_module.is_loaded
    return wrapper


def _buffers(q):
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device),
    )


def _check(actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            result.float(), reference.float(), rtol=1e-2, atol=1e-2
        )


def _run(wrapper, q, kv, **kwargs):
    # These fixtures exceed the SM120/121 cooperative-grid or shared-memory limit.
    # Avoid launching them: CUDA retains the error and poisons the next test.
    if q.shape[-1] in (128, 256) and get_compute_capability(q.device)[0] == 12:
        pytest.xfail("SM120/121 persistent BatchAttention resource limit")
    return wrapper.run(q, kv, **kwargs)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "mode,dtype,dim",
    [
        ("padded", torch.bfloat16, 64),
        ("page", torch.float16, 128),
        ("token", torch.bfloat16, 256),
        ("head", torch.float16, 64),
    ],
)
def test_data_strides(layout, mode, dtype, dim):
    q, k, v, _, gpu, expected = _inputs(layout, dtype, dim)
    kv = make_paged_kv_cache_pair(k, v, layout, mode, 8)
    wrapper = _plan(gpu, layout, dtype, dim)
    out, lse = _buffers(q)
    _check(_run(wrapper, q, kv, out=out, lse=lse), expected)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_wrapper_reuse(layout):
    q, k, v, _, gpu, expected = _inputs(layout)
    equal = make_paged_kv_cache_pair(k, v, layout, "padded", 8)
    unequal = make_paged_kv_cache_pair(
        k, v, layout, "token" if layout == "NHD" else "head", 8
    )
    wrapper = _fresh_lazy_wrapper(gpu, layout)
    out, lse = _buffers(q)
    for kv, loaded in ((equal, False), (unequal, True), (equal, True)):
        out.fill_(torch.nan)
        lse.fill_(torch.nan)
        _check(_run(wrapper, q, kv, out=out, lse=lse), expected)
        assert wrapper._independent_module.is_loaded is loaded


def test_cuda_graph_strides():
    q, k, v, cpu, gpu, expected = _inputs()
    equal = make_paged_kv_cache_pair(k, v, "NHD", "padded", 8)
    unequal = make_paged_kv_cache_pair(k, v, "NHD", "token", 8)
    wrapper = _fresh_lazy_wrapper(gpu)
    equal_out, equal_lse = _buffers(q)
    out, lse = _buffers(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    rejected_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(rejected_graph, stream=stream):
        with pytest.raises(RuntimeError, match="prewarm_paged_kv_stride_variant"):
            wrapper.run(q, unequal, out=out, lse=lse)
        # Keep the graph valid after catching the Python-only cold-load guard.
        out.zero_()
    assert not wrapper._independent_module.is_loaded
    wrapper.prewarm_paged_kv_stride_variant("independent")
    assert wrapper._independent_module.is_loaded
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(wrapper, q, equal, out=equal_out, lse=equal_lse)
        _run(wrapper, q, unequal, out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _run(wrapper, q, equal, out=equal_out, lse=equal_lse)
        _run(wrapper, q, unequal, out=out, lse=lse)
    graph.replay()
    _check((equal_out, equal_lse), expected)
    _check((out, lse), expected)
    # Update input contents at stable addresses to catch stale outputs/replay.
    q.mul_(0.5)
    expected = _gather_reference("NHD", q, k, v, cpu)
    for result in (equal_out, equal_lse, out, lse):
        result.fill_(torch.nan)
    graph.replay()
    _check((equal_out, equal_lse), expected)
    _check((out, lse), expected)


@pytest.mark.parametrize(
    "data_equal,scale_equal",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_nvfp4_strides(data_equal, scale_equal):
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("NVFP4 stride contracts require SM100a")
    torch.manual_seed(20260917)
    cpu, gpu = _metadata()
    q = torch.randn(cpu[0][-1], 8, 128, dtype=torch.float16, device="cuda") * 0.2
    shape = (13, 16, 2, 64)
    kd, ks, _ = create_nvfp4_kv(shape, "cuda")
    vd, vs, _ = create_nvfp4_kv(shape, "cuda")
    # Explicit distinct non-unit scales: helper defaults (1, 1) cannot test this.
    kg = torch.tensor(0.75, device="cuda")
    vg = torch.tensor(1.25, device="cuda")
    assert kg.item() != vg.item()
    # Helpers flatten storage: dequantize dense tensors before adding padding.
    k_ref = nvfp4_to_float(kd, ks, kg).to(q.dtype)
    v_ref = nvfp4_to_float(vd, vs, vg).to(q.dtype)
    expected = _gather_reference("NHD", q, k_ref, v_ref, cpu, causal=False)
    kv = make_paged_kv_cache_pair(kd, vd, "NHD", "padded" if data_equal else "head", 8)
    sf = make_paged_kv_cache_pair(
        ks, vs, "NHD", "padded" if scale_equal else "token", 4
    )
    wrapper = _plan(
        gpu, dtype=torch.float16, dim=128, kv_dtype=torch.uint8, causal=False
    )
    out, lse = _buffers(q)
    wrapper.run(
        q, kv, out=out, lse=lse, k_scale=kg.item(), v_scale=vg.item(), kv_cache_sf=sf
    )
    torch.testing.assert_close(out.float(), expected[0], rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(lse, expected[1], rtol=0, atol=5e-4)


@pytest.mark.parametrize("stride_mode", ["page", "token", "head"])
def test_primary_ffi_rejects_unequal_strides(stride_mode):
    from flashinfer.utils import TensorLayout

    q, k, v, _, gpu, _ = _inputs()
    k, v = make_paged_kv_cache_pair(k, v, "NHD", stride_mode, 8)
    wrapper = _plan(gpu)
    out, lse = _buffers(q)
    # Deliberately bypass Python routing to test the exported primary ABI guard.
    with pytest.raises(Exception, match="equal-stride BatchAttention primary"):
        # TVM FFI maps ICHECK to a version-specific Python exception class.
        wrapper.module.run(
            wrapper.float_workspace_buffer,
            wrapper.int_workspace_buffer,
            wrapper._plan_info,
            q,
            k,
            v,
            wrapper._kv_indices,
            out,
            lse,
            wrapper._mask_mode,
            TensorLayout.NHD.value,
            wrapper._num_qo_heads,
            wrapper._num_kv_heads,
            wrapper._page_size,
            1.0,
            1.0 / math.sqrt(q.shape[-1]),
            0.0,
            None,
            None,
        )


def test_lazy_no_jit_recognized_cache_provider(monkeypatch, tmp_path):
    from flashinfer.attention import _core
    from flashinfer.jit import env as jit_env

    q, k, v, _, gpu, expected = _inputs()
    kv = make_paged_kv_cache_pair(k, v, "NHD", "token", 8)
    wrapper = _fresh_lazy_wrapper(gpu)
    wrapper.prewarm_paged_kv_stride_variant("independent")
    spec = wrapper._independent_module._spec
    artifact = spec.get_library_path().resolve(strict=True)
    provider_root = tmp_path / "aot"
    target = provider_root / spec.name / (spec.name + ".so")
    target.parent.mkdir(parents=True)
    target.symlink_to(artifact)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", provider_root)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", ())
    assert spec.is_aot
    assert spec.get_library_path() == target
    wrapper._independent_module = _core._LazyBatchAttentionIndependentModule(spec)
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    out, lse = _buffers(q)
    _check(_run(wrapper, q, kv, out=out, lse=lse), expected)
    assert wrapper._independent_module.is_loaded


def test_lazy_no_jit_missing_provider_is_actionable(monkeypatch, tmp_path):
    from flashinfer.jit import MissingJITCacheError
    from flashinfer.jit import env as jit_env

    q, k, v, _, gpu, expected = _inputs()
    kv = make_paged_kv_cache_pair(k, v, "NHD", "token", 8)
    wrapper = _fresh_lazy_wrapper(gpu)
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_DIR", tmp_path / "absent-provider")
    monkeypatch.setattr(jit_env, "FLASHINFER_AOT_PROVIDERS", ())
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    out, lse = _buffers(q)
    with pytest.raises(MissingJITCacheError, match="prewarm_paged_kv_stride_variant"):
        wrapper.run(q, kv, out=out, lse=lse)
    assert not wrapper._independent_module.is_loaded
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT")
    _check(_run(wrapper, q, kv, out=out, lse=lse), expected)
    assert wrapper._independent_module.is_loaded
