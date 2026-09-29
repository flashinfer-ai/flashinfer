"""Standalone grouped FP8 API, trace export and reference correctness."""

import inspect
import json

import pytest
import torch

from flashinfer.gemm import group_deepgemm_fp8_nt_groupwise
from flashinfer.gemm import group_gemm_fp8_nt_groupwise_contiguous as api
from flashinfer.trace.templates.gemm import (
    group_gemm_fp8_nt_groupwise_contiguous_trace as template,
)


def test_grouped_cute_dsl_public_contract():
    import flashinfer.gemm as gemm

    assert api.__name__ == "group_gemm_fp8_nt_groupwise_contiguous"
    assert api.__name__ in gemm.__all__
    assert not api.has_backend_choices()
    assert api.is_compute_capability_supported(100)
    assert api.is_compute_capability_supported(103)
    assert not api.is_compute_capability_supported(107)
    assert group_deepgemm_fp8_nt_groupwise.is_compute_capability_supported(107)
    assert list(inspect.signature(group_deepgemm_fp8_nt_groupwise).parameters) == [
        "a",
        "b",
        "a_scale",
        "b_scale",
        "m_indices",
        "scale_granularity_mnk",
        "out",
        "out_dtype",
    ]
    assert "backend" not in inspect.signature(api).parameters


def test_grouped_cute_dsl_trace_metadata(tmp_path):
    inputs = template.init(M=129, num_groups=2, N=128, K=128, device="cpu")
    definition = api.fi_trace(save_dir=tmp_path, **inputs)
    assert definition["name"] == "group_gemm_fp8_nt_groupwise_contiguous_g2_n128_k128"
    assert definition["axes"]["M"]["type"] == "var"
    assert definition["axes"]["num_groups"]["value"] == 2
    assert list(definition["inputs"]) == ["a", "b", "a_scale", "b_scale", "m_indices"]
    assert definition["outputs"]["out"]["dtype"] == "bfloat16"
    assert (
        "fi_api:flashinfer.gemm.gemm_base.group_gemm_fp8_nt_groupwise_contiguous"
        in definition["tags"]
    )
    assert (
        json.loads((tmp_path / (definition["name"] + ".json")).read_text())
        == definition
    )
    namespace = {}
    exec(definition["init"], namespace)
    generated = namespace[template.init.__name__](
        M=129, num_groups=2, N=128, K=128, device="cpu"
    )
    assert generated["m_indices"].tolist() == [0] * 128 + [1]


@pytest.mark.parametrize(
    "m,n,k", [(0, 128, 128), (1, 128, 128), (129, 256, 384), (257, 128, 4096)]
)
def test_grouped_cute_dsl_trace_reference(m, n, k):
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if not api.is_compute_capability_supported(major * 10 + minor):
        pytest.skip("Requires SM100/SM103")
    if not is_cute_dsl_available():
        pytest.skip("Requires nvidia-cutlass-dsl")
    inputs = template.init(M=m, num_groups=2, N=n, K=k)
    definition = api.fi_trace(**inputs)
    namespace = {}
    exec(definition["reference"], namespace)
    ref = namespace[template.reference.__name__](**inputs)
    out = api(**inputs, validate_indices=True)
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
