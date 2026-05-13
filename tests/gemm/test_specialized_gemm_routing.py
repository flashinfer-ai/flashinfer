import importlib.util
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version

from flashinfer import SfLayout, bmm_fp8, mm_fp4, nvfp4_quantize
from flashinfer.env import (
    FLASHINFER_SPECIALIZED_KERNEL_DISABLE,
    reset_specialized_kernel_env_cache,
)
from flashinfer.gemm.specialized_kernels import (
    is_bmm_fp8_sm121_specialized_problem,
    is_mm_fp4_sm121_specialized_problem,
)
from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


SPECIALIZED_KERNEL_DIR = (
    Path(__file__).resolve().parents[2] / "flashinfer" / "gemm" / "specialized_kernels"
)
MM_FP4_WORKLOADS = SPECIALIZED_KERNEL_DIR / "mm_fp4_sm121" / "workloads.json"
BMM_FP8_WORKLOADS = SPECIALIZED_KERNEL_DIR / "bmm_fp8_sm121" / "workloads.json"
SUPPORTED_IMPLS = {
    "mm_fp4": {"cute_dsl"},
    "bmm_fp8": {"cuda", "cute_dsl", "cutile"},
}


def _sm121_unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "specialized GEMM routing tests require CUDA"
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        cuda_version = get_cuda_version()
    except Exception as exc:
        return f"specialized GEMM routing tests could not determine CUDA version: {exc}"
    if cuda_version < Version("13.0"):
        return f"specialized GEMM routing tests require CUDA 13.0+, got {cuda_version}"
    cc = get_compute_capability(torch.device("cuda"))
    if cc != (12, 1):
        return f"specialized GEMM routing tests require SM121, got SM{cc[0]}{cc[1]}"
    return None


_SM121_SKIP_REASON = _sm121_unavailable_reason()
pytestmark = pytest.mark.skipif(
    _SM121_SKIP_REASON is not None,
    reason=_SM121_SKIP_REASON or "specialized GEMM routing tests require SM121",
)


def _load_cases(function: str, path: Path):
    cases = json.loads(path.read_text())
    unsupported_impls = sorted(
        {str(item["impl"]) for item in cases} - SUPPORTED_IMPLS[function]
    )
    if unsupported_impls:
        raise ValueError(
            f"{function} specialized workloads reference removed implementations: "
            f"{unsupported_impls}"
        )
    return list(enumerate(cases))


def _case_id(item):
    prefix = f"{item['impl']}-m{item['m']}-n{item['n']}-k{item['k']}"
    if "b" in item:
        return f"{item['impl']}-b{item['b']}-m{item['m']}-n{item['n']}-k{item['k']}"
    return prefix


def _params(function: str, path: Path):
    return [
        pytest.param(idx, item, id=_case_id(item))
        for idx, item in _load_cases(function, path)
    ]


def _has_module(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _skip_if_backend_missing(function: str, impl: str):
    if impl == "cuda":
        return
    if impl == "cute_dsl":
        if not _has_module("cutlass"):
            pytest.skip(f"{function} CUTE DSL specialized backend requires cutlass")
        if not _has_module("cuda.bindings.driver"):
            pytest.skip(
                f"{function} CUTE DSL specialized backend requires cuda.bindings"
            )
    elif impl == "cutile":
        if not _has_module("cuda.tile"):
            pytest.skip(f"{function} cuTile specialized backend requires cuda.tile")
    else:
        pytest.fail(f"Unknown specialized backend: {impl}")


@contextmanager
def _specialized_kernel_disabled(disabled: bool):
    old_values = {
        FLASHINFER_SPECIALIZED_KERNEL_DISABLE: os.environ.get(
            FLASHINFER_SPECIALIZED_KERNEL_DISABLE
        ),
    }
    value = "True" if disabled else "False"
    os.environ[FLASHINFER_SPECIALIZED_KERNEL_DISABLE] = value
    reset_specialized_kernel_env_cache()
    try:
        yield
    finally:
        for name, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
        reset_specialized_kernel_env_cache()


def _assert_cosine(reference: torch.Tensor, result: torch.Tensor, threshold: float):
    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), result.reshape(-1).float(), dim=0
    )
    assert cos_sim > threshold, f"Cosine similarity {cos_sim:.6f} <= {threshold}"


@pytest.mark.parametrize("case_idx,item", _params("mm_fp4", MM_FP4_WORKLOADS))
def test_mm_fp4_sm121_specialized_routing(case_idx, item):
    del case_idx
    _skip_if_backend_missing("mm_fp4", item["impl"])

    m, k, n = int(item["m"]), int(item["k"]), int(item["n"])
    torch.manual_seed(m * 1000003 + k * 101 + n)
    a_ref = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_ref = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a_ref, b_ref.T)

    a_global_sf = (448 * 6) / a_ref.float().abs().nan_to_num().max()
    b_global_sf = (448 * 6) / b_ref.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(
        a_ref, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b_ref, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (a_global_sf * b_global_sf)

    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    assert is_mm_fp4_sm121_specialized_problem(
        a_fp4,
        b_fp4.T,
        a_sf,
        b_sf.T,
        alpha,
        torch.bfloat16,
        out,
        block_size=16,
        use_8x4_sf_layout=False,
        backend="b12x",
        use_nvfp4=True,
    )

    with _specialized_kernel_disabled(True):
        baseline = torch.empty_like(out)
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            baseline,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
        )

    with _specialized_kernel_disabled(False):
        result = torch.empty_like(out)
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha,
            torch.bfloat16,
            result,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
        )

    _assert_cosine(reference, baseline, 0.97)
    _assert_cosine(reference, result, 0.97)
    _assert_cosine(baseline, result, 0.97)


@pytest.mark.parametrize("case_idx,item", _params("bmm_fp8", BMM_FP8_WORKLOADS))
def test_bmm_fp8_sm121_specialized_routing(case_idx, item):
    del case_idx
    _skip_if_backend_missing("bmm_fp8", item["impl"])

    batch = int(item["b"])
    m, k, n = int(item["m"]), int(item["k"]), int(item["n"])
    torch.manual_seed(batch * 10000019 + m * 1000003 + k * 101 + n)
    input = torch.randn((batch, m, k), device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)
    mat2 = torch.randn((batch, n, k), device="cuda", dtype=torch.bfloat16).transpose(
        -2, -1
    )
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=torch.float8_e4m3fn)
    reference = torch.bmm(input, mat2)
    out = torch.empty((batch, m, n), device="cuda", dtype=torch.bfloat16)

    assert is_bmm_fp8_sm121_specialized_problem(
        input_fp8,
        mat2_fp8,
        input_inv_s,
        mat2_inv_s,
        torch.bfloat16,
        out,
        backend="cublas",
    )

    with _specialized_kernel_disabled(True):
        baseline = torch.empty_like(out)
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            torch.bfloat16,
            baseline,
            backend="cublas",
        )

    with _specialized_kernel_disabled(False):
        result = torch.empty_like(out)
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            torch.bfloat16,
            result,
            backend="cublas",
        )

    _assert_cosine(reference, baseline, 0.99)
    _assert_cosine(reference, result, 0.99)
    _assert_cosine(baseline, result, 0.99)
