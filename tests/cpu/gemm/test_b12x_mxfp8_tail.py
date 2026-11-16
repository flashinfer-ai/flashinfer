"""Host-side eligibility and specialization regressions for b12x MXFP8 tails."""

from packaging.version import Version
import pytest
import torch

from flashinfer.gemm import gemm_mm_mxfp8_cute_dsl as mxfp8


pytestmark = pytest.mark.cpu_only


def _inputs(k, out_dtype=torch.bfloat16):
    a = torch.empty((6, k), dtype=torch.float8_e4m3fn)
    b = torch.empty((128, k), dtype=torch.float8_e4m3fn).T
    scale = torch.empty(128 * ((k + 127) // 128) * 4, dtype=torch.uint8)
    out = torch.empty((6, 128), dtype=out_dtype)
    return [a, b, scale, scale, out_dtype, out, None]


@pytest.mark.parametrize("k", [128, 160, 192, 224, 544, 576, 608, 640])
@pytest.mark.parametrize("backend", ["auto", "b12x"])
def test_requirement_accepts_whole_mxfp8_groups(monkeypatch, k, backend):
    monkeypatch.setattr(mxfp8, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setattr(mxfp8, "_b12x_mxfp8_dsl_supported", lambda: True)
    a, b, a_sf, b_sf, dtype, out, _ = _inputs(k)
    assert mxfp8._b12x_gemm_mxfp8_requirement(
        a,
        b,
        a_sf,
        b_sf,
        out=out,
        out_dtype=dtype,
        use_8x4_sf_layout=False,
        backend=backend,
        batch_invariant=False,
    )


@pytest.mark.parametrize("k", [144, 559, 577])
def test_requirement_rejects_incomplete_mxfp8_groups(monkeypatch, k):
    monkeypatch.setattr(mxfp8, "get_cuda_version", lambda: Version("13.0"))
    monkeypatch.setattr(mxfp8, "_b12x_mxfp8_dsl_supported", lambda: True)
    a, b, a_sf, b_sf, *_ = _inputs(k)
    kwargs = dict(use_8x4_sf_layout=False)
    assert not mxfp8._b12x_gemm_mxfp8_requirement(a, b, a_sf, b_sf, **kwargs)
    with pytest.raises(ValueError, match="multiple of 32"):
        mxfp8._b12x_gemm_mxfp8_requirement(a, b, a_sf, b_sf, backend="b12x", **kwargs)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_prefetch", [False, True])
def test_runner_cache_specializes_tail_and_reuses_dynamic_k(
    monkeypatch, out_dtype, use_prefetch
):
    pytest.importorskip("cutlass")
    from flashinfer.cute_dsl import utils
    from flashinfer.gemm import gemm_mm_fp4_cute_dsl as helpers

    kernels, launches = [], []

    def fake_compile_fn(kernel, **kwargs):
        kernels.append(kernel)

        def launch(*args):
            launches.append((kernel, args))

        return lambda: launch

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(utils, "get_max_active_clusters", lambda _: 1)
    monkeypatch.setattr(helpers, "_make_blockscaled_gemm_compile_fn", fake_compile_fn)
    monkeypatch.setattr(mxfp8, "_prepare_alpha_for_launch", lambda *args: None)
    monkeypatch.setattr(mxfp8, "_B12X_MM_MXFP8_KERNEL_CACHE", {})

    runner = mxfp8._b12x_gemm_mxfp8_runner(12, 0, False, out_dtype)
    tactic = ((16, 128), (1, 1), False, use_prefetch, "sm120", None)
    for k in (128, 160, 192, 224, 544, 576, 608, 640):
        inputs = _inputs(k, out_dtype)
        assert runner.forward(inputs, tactic=tactic) is inputs[5]
        kernel, args = launches[-1]
        assert kernel.mxfp8_k_tail == k % 128
        assert args[5] == (k + 127) // 128

    assert [kernel.mxfp8_k_tail for kernel in kernels] == [0, 32, 64, 96]
    assert launches[0][0] is launches[7][0]
    assert all(launches[i][0] is launches[i + 3][0] for i in (1, 2, 3))
