from ..api_logging import flashinfer_api
from ..trace.templates.gemm import (
    mm_M1_16_K6144_N256_trace,
    mm_M1_16_K7168_N256_trace,
    tinygemm_bf16_trace,
)
from flashinfer.jit.cpp_ext import is_cuda_version_at_least
from flashinfer.jit import (
    gen_dsv3_router_gemm_module,
    gen_tinygemm2_module,
    gen_tinygemm2_sm100_module,
)
from flashinfer.jit.cake_router_gemm import (
    gen_cake_router_gemm_module,
    supported_capability as _cake_router_gemm_capability,
)
import functools
import os
from types import SimpleNamespace
from typing import Literal, NamedTuple, Optional, Tuple
import torch
import tvm_ffi
from flashinfer.utils import (
    get_compute_capability,
    register_custom_op,
    supported_compute_capability,
    backend_requirement,
    version_at_least,
)


def _router_gemm_shape_checks(
    mat_a,
    mat_b,
    out,
    launch_with_pdl,
    expected_hidden_dim,
    expected_num_experts,
    expected_out_dtype,
):
    # Dimension checks
    if mat_a.dim() != 2:
        raise ValueError("mat_a must be a 2D tensor")
    if mat_b.dim() != 2:
        raise ValueError("mat_b must be a 2D tensor")
    if out.dim() != 2:
        raise ValueError("out must be a 2D tensor")

    # Stride checks (check these before dimension checks to give better error messages).
    # The kernels address mat_a[m * K + k], mat_b[e * K + k] and out[m * N + e], so every
    # operand must be dense: a padded view would silently produce wrong results.
    if mat_a.stride(1) != 1 or (
        mat_a.shape[0] > 1 and mat_a.stride(0) != mat_a.shape[1]
    ):
        raise ValueError("mat_a must be a dense row-major tensor")
    if out.stride(1) != 1 or (out.shape[0] > 1 and out.stride(0) != out.shape[1]):
        raise ValueError("out must be a dense row-major tensor")
    if mat_b.stride(0) != 1 or (
        mat_b.shape[1] > 1 and mat_b.stride(1) != mat_b.shape[0]
    ):
        raise ValueError("mat_b must be a dense column-major tensor")

    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("mat_a.shape[1] must be equal to mat_b.shape[0]")
    if out.shape[0] != mat_a.shape[0]:
        raise ValueError("out.shape[0] must be equal to mat_a.shape[0]")
    if out.shape[1] != mat_b.shape[1]:
        raise ValueError("out.shape[1] must be equal to mat_b.shape[1]")

    # Problem size checks
    min_tokens = 1
    max_tokens = 16
    if mat_a.shape[0] < min_tokens or mat_a.shape[0] > max_tokens:
        raise ValueError(
            f"mat_a.shape[0] (num_tokens) must be between {min_tokens} and {max_tokens}"
        )
    if mat_a.shape[1] != expected_hidden_dim:
        raise ValueError(
            f"mat_a.shape[1] (hidden_dim) must be equal to {expected_hidden_dim}"
        )
    if mat_b.shape[1] != expected_num_experts:
        raise ValueError(
            f"mat_b.shape[1] (num_experts) must be equal to {expected_num_experts}"
        )

    # Data type checks
    if mat_a.dtype != torch.bfloat16:
        raise ValueError("mat_a must be a bfloat16 tensor")
    if mat_b.dtype != torch.bfloat16:
        raise ValueError("mat_b must be a bfloat16 tensor")
    if out.dtype != expected_out_dtype:
        raise ValueError(f"out must be a {expected_out_dtype} tensor")

    return True


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N256_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=256,
        expected_out_dtype=torch.float32,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N128_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=128,
        expected_out_dtype=torch.bfloat16,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K6144_N256_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=6144,
        expected_num_experts=256,
        expected_out_dtype=torch.float32,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N256_bf16_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=256,
        expected_out_dtype=torch.bfloat16,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N384_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=384,
        expected_out_dtype=torch.float32,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N384_bf16_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=384,
        expected_out_dtype=torch.bfloat16,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N896_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=896,
        expected_out_dtype=torch.float32,
    )


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability([90, 100, 103, 107])
def _mm_M1_16_K7168_N896_bf16_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(
        mat_a,
        mat_b,
        out,
        launch_with_pdl,
        expected_hidden_dim=7168,
        expected_num_experts=896,
        expected_out_dtype=torch.bfloat16,
    )


@functools.cache
def get_dsv3_router_gemm_module():
    module = gen_dsv3_router_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::ml3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N128(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.ml3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::dsv3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N256(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.dsv3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::glm_dsa_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K6144_N256(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.glm_dsa_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::dsv3_bf16_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N256_bf16(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.dsv3_bf16_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::kimi_k2_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N384(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.kimi_k2_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::kimi_k2_bf16_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N384_bf16(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.kimi_k2_bf16_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::kimi_k3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N896(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.kimi_k3_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::kimi_k3_bf16_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N896_bf16(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.kimi_k3_bf16_router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    return SimpleNamespace(
        mm_M1_16_K7168_N128=mm_M1_16_K7168_N128,
        mm_M1_16_K7168_N256=mm_M1_16_K7168_N256,
        mm_M1_16_K6144_N256=mm_M1_16_K6144_N256,
        mm_M1_16_K7168_N256_bf16=mm_M1_16_K7168_N256_bf16,
        mm_M1_16_K7168_N384=mm_M1_16_K7168_N384,
        mm_M1_16_K7168_N384_bf16=mm_M1_16_K7168_N384_bf16,
        mm_M1_16_K7168_N896=mm_M1_16_K7168_N896,
        mm_M1_16_K7168_N896_bf16=mm_M1_16_K7168_N896_bf16,
    )


@functools.cache
def get_cake_router_gemm_module():
    module = gen_cake_router_gemm_module().build_and_load()

    @register_custom_op(
        "flashinfer::cake_ml3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N128(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.run(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::cake_dsv3_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K7168_N256(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.run(mat_a, mat_b, out, launch_with_pdl)

    @register_custom_op(
        "flashinfer::cake_glm_dsa_router_gemm_op",
        mutates_args=["out"],
    )
    def mm_M1_16_K6144_N256(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.run(mat_a, mat_b, out, launch_with_pdl)

    return SimpleNamespace(
        mm_M1_16_K7168_N128=mm_M1_16_K7168_N128,
        mm_M1_16_K7168_N256=mm_M1_16_K7168_N256,
        mm_M1_16_K6144_N256=mm_M1_16_K6144_N256,
    )


@functools.cache
def _router_gemm_module_for_device(device_index: int):
    # The capability query runs once per device; the Cake kernel serves the validated
    # SM100/SM103 build targets and every other device keeps the reference kernel.
    capability = get_compute_capability(torch.device("cuda", device_index))
    if _cake_router_gemm_capability(capability) is not None:
        return get_cake_router_gemm_module()
    return get_dsv3_router_gemm_module()


def get_router_gemm_module(device: torch.device, *, backend: str):
    if backend != "cake":
        raise ValueError(f"unsupported Router GEMM backend: {backend!r}")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _router_gemm_module_for_device(index)


@backend_requirement({}, common_check=_mm_M1_16_K7168_N128_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N128(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in Mistral Large 3.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in Mistral Large 3's Mixture-of-Experts
    (MoE) architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a``
    is a small batch of token embeddings (1-16 rows) and ``mat_b`` is the
    expert routing weight matrix.  Specialized for the dimensions used in
    Mistral Large 3 MoE (``K = 7168``, ``N = 128``).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (128).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be bfloat16, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires Blackwell SM100/SM103 architecture.  The specialized problem-size
    optimization makes this significantly faster than general-purpose GEMM
    implementations for the router op.  Raises ``ValueError`` if tensor
    dimensions, strides, or dtypes do not match the expected Mistral Large 3
    configuration.
    """
    get_router_gemm_module(mat_a.device, backend="cake").mm_M1_16_K7168_N128(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N256_shape_checks)
@flashinfer_api(trace=mm_M1_16_K7168_N256_trace)
def mm_M1_16_K7168_N256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in DeepSeek-V3.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in DeepSeek-V3's Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    DeepSeek-V3 MoE (``K = 7168``, ``N = 256``).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (256).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be float32, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires Blackwell SM100/SM103 architecture.  The specialized
    problem-size optimization makes this significantly faster than
    general-purpose GEMM implementations for the router op.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected DeepSeek-V3 router configuration.
    """
    get_router_gemm_module(mat_a.device, backend="cake").mm_M1_16_K7168_N256(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K6144_N256_shape_checks)
@flashinfer_api(trace=mm_M1_16_K6144_N256_trace)
def mm_M1_16_K6144_N256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in GLM-MoE-DSA.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in GLM-MoE-DSA's Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    GLM-MoE-DSA (``K = 6144``, ``N = 256``).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (6144).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (256).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be float32, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires Blackwell SM100/SM103 architecture.  The specialized
    problem-size optimization makes this significantly faster than
    general-purpose GEMM implementations for the router op.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected GLM-MoE-DSA configuration.
    """
    get_router_gemm_module(mat_a.device, backend="cake").mm_M1_16_K6144_N256(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N256_bf16_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N256_bf16(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in DeepSeek-V3.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in DeepSeek-V3's Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    DeepSeek-V3 MoE (``K = 7168``, ``N = 256``).

    Produces bfloat16 router logits (as opposed to the float32 produced by
    :func:`mm_M1_16_K7168_N256`).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (256).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be bfloat16, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires SM90 (Hopper) or newer.  The specialized problem-size
    optimization makes this significantly faster than general-purpose GEMM
    implementations for the router op, for small token counts; past roughly
    4-8 tokens (shape dependent) a general-purpose GEMM is faster.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected DeepSeek-V3 router configuration.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N256_bf16(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N384_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N384(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in Kimi-K2.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in the Kimi-K2 family's (K2/K2.5/K2.6) Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    Kimi-K2 MoE (``K = 7168``, ``N = 384``).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (384).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be float32, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires SM90 (Hopper) or newer.  The specialized problem-size
    optimization makes this significantly faster than general-purpose GEMM
    implementations for the router op, for small token counts; past roughly
    4-8 tokens (shape dependent) a general-purpose GEMM is faster.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected Kimi-K2 router configuration.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N384(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N384_bf16_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N384_bf16(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in Kimi-K2.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in the Kimi-K2 family's (K2/K2.5/K2.6) Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    Kimi-K2 MoE (``K = 7168``, ``N = 384``).

    Produces bfloat16 router logits (as opposed to the float32 produced by
    :func:`mm_M1_16_K7168_N384`).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (384).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be bfloat16, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires SM90 (Hopper) or newer.  The specialized problem-size
    optimization makes this significantly faster than general-purpose GEMM
    implementations for the router op, for small token counts; past roughly
    4-8 tokens (shape dependent) a general-purpose GEMM is faster.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected Kimi-K2 router configuration.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N384_bf16(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N896_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N896(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in Kimi-K3.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in Kimi-K3's Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    Kimi-K3 MoE (``K = 7168``, ``N = 896``).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (896).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be float32, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires SM90 (Hopper) or newer.  The specialized problem-size
    optimization makes this significantly faster than general-purpose GEMM
    implementations for the router op, for small token counts; past roughly
    4-8 tokens (shape dependent) a general-purpose GEMM is faster.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected Kimi-K3 router configuration.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N896(
        mat_a, mat_b, out, launch_with_pdl
    )


@backend_requirement({}, common_check=_mm_M1_16_K7168_N896_bf16_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N896_bf16(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in Kimi-K3.

    Performs a highly optimized matrix multiplication specifically tailored
    for the expert routing GEMM in Kimi-K3's Mixture-of-Experts (MoE)
    architecture.  Computes ``out = mat_a @ mat_b`` where ``mat_a`` is a
    small batch of token embeddings (1-16 rows) and ``mat_b`` is the expert
    routing weight matrix.  Specialized for the dimensions used in
    Kimi-K3 MoE (``K = 7168``, ``N = 896``).

    Produces bfloat16 router logits (as opposed to the float32 produced by
    :func:`mm_M1_16_K7168_N896`).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)`` where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (7168).  Must be bfloat16,
        row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)`` where ``N`` is the number of
        experts (896).  Must be bfloat16, column-major (transposed layout).
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` containing the routing
        scores.  Must be bfloat16, row-major (contiguous).  Mutated in place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires SM90 (Hopper) or newer.  The specialized problem-size
    optimization makes this significantly faster than general-purpose GEMM
    implementations for the router op, for small token counts; past roughly
    4-8 tokens (shape dependent) a general-purpose GEMM is faster.  Raises
    ``ValueError`` if tensor dimensions, strides, or dtypes do not match the
    expected Kimi-K3 router configuration.
    """
    get_dsv3_router_gemm_module().mm_M1_16_K7168_N896_bf16(
        mat_a, mat_b, out, launch_with_pdl
    )


# ============================================================================
# tinygemm2: SM90+ BF16 small GEMM with bias (from TensorRT-LLM)
# Computes: output = input @ weight.T + bias  (equivalent to F.linear)
# ============================================================================


@supported_compute_capability([90, 100, 103, 107, 110, 120, 121])
def _tinygemm_bf16_shape_checks(input, weight, out, bias, use_pdl):
    if input.dim() != 2:
        raise ValueError("input must be a 2D tensor")
    if weight.dim() != 2:
        raise ValueError("weight must be a 2D tensor")
    if out.dim() != 2:
        raise ValueError("out must be a 2D tensor")

    if not input.is_contiguous():
        raise ValueError("input must be contiguous (row-major)")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous (row-major)")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous (row-major)")

    if input.shape[1] != weight.shape[1]:
        raise ValueError(
            f"input.shape[1] ({input.shape[1]}) must equal weight.shape[1] ({weight.shape[1]})"
        )
    if out.shape[0] != input.shape[0]:
        raise ValueError(
            f"out.shape[0] ({out.shape[0]}) must equal input.shape[0] ({input.shape[0]})"
        )
    if out.shape[1] != weight.shape[0]:
        raise ValueError(
            f"out.shape[1] ({out.shape[1]}) must equal weight.shape[0] ({weight.shape[0]})"
        )
    output_features = weight.shape[0]

    if output_features % 16 != 0:
        raise ValueError(
            f"output_features ({output_features}) must be a multiple of 16 (tile alignment)"
        )

    if input.dtype != torch.bfloat16:
        raise ValueError("input must be bfloat16")
    if weight.dtype != torch.bfloat16:
        raise ValueError("weight must be bfloat16")
    if out.dtype != torch.bfloat16:
        raise ValueError("out must be bfloat16")

    if bias is not None:
        if bias.dim() != 1:
            raise ValueError("bias must be a 1D tensor")
        if bias.shape[0] != weight.shape[0]:
            raise ValueError(
                f"bias.shape[0] ({bias.shape[0]}) must equal weight.shape[0] ({weight.shape[0]})"
            )
        if bias.dtype != torch.bfloat16:
            raise ValueError("bias must be bfloat16")
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")

    return True


@functools.cache
def get_tinygemm2_module():
    module = gen_tinygemm2_module().build_and_load()

    @register_custom_op(
        "flashinfer::tinygemm2_op",
        mutates_args=["out"],
    )
    def tinygemm2_op_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm2_op(input, weight, bias, out, use_pdl)

    @register_custom_op(
        "flashinfer::tinygemm2_nobias_op",
        mutates_args=["out"],
    )
    def tinygemm2_nobias_op_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm2_nobias_op(input, weight, out, use_pdl)

    return SimpleNamespace(
        tinygemm2_op=tinygemm2_op_impl,
        tinygemm2_nobias_op=tinygemm2_nobias_op_impl,
    )


# tinygemm2_sm100: generated SM100/SM103 port of the same kernel with
# bit-identical outputs (csrc/tinygemm2_sm100.cu); selected automatically for
# the bias path on B200/B300-class devices. Ring depth (stage 4/8/16) is
# selected inside the binding, mirroring the reference launcher convention.


@functools.cache
def get_tinygemm2_sm100_module():
    module = gen_tinygemm2_sm100_module().build_and_load()

    @register_custom_op(
        "flashinfer::tinygemm2_sm100_op",
        mutates_args=["out"],
    )
    def tinygemm2_sm100_op_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        use_pdl: bool = False,
    ) -> None:
        module.tinygemm2_sm100_op(input, weight, bias, out, use_pdl)

    return SimpleNamespace(tinygemm2_sm100_op=tinygemm2_sm100_op_impl)


# The generated kernels are validated on SM100 (B200), SM103 (B300/GB300) and
# SM107 (Rubin) exactly; other 10.x devices pass is_sm100a_supported's
# major==10 predicate but must keep using the reference kernel.
_TINYGEMM2_SM100_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))


@functools.cache
def _tinygemm2_sm100_enabled(
    disabled: str, compute_capability: Tuple[int, int]
) -> bool:
    # Evaluated once per (escape-hatch value, compute capability): the CUDA
    # version parsing below is off the per-call path.
    if disabled == "1":
        return False
    if compute_capability not in _TINYGEMM2_SM100_SUPPORTED_COMPUTE_CAPABILITIES:
        return False
    if compute_capability == (10, 7) and not is_cuda_version_at_least("13.4"):
        return False
    return version_at_least(torch.version.cuda, "12.8")


def _use_tinygemm2_sm100(device: torch.device) -> bool:
    # Keyed by the current FLASHINFER_DISABLE_TINYGEMM2_SM100 value so a
    # process that toggles the escape hatch at runtime sees the change.
    return _tinygemm2_sm100_enabled(
        os.environ.get("FLASHINFER_DISABLE_TINYGEMM2_SM100", "0"),
        get_compute_capability(device),
    )


# tinygemm2_tcgen05: generated tcgen05 programs (flashinfer.jit.cake_tinygemm2_tcgen05)
# with the reference's output bits.  They keep the reference's four K chains, the
# K16 order within a chain and its +0 start, and sum the chains and the bias in the
# reference's order, so a routed call returns exactly what the reference returns.
# Nine tiles, each with and without the bias and with and without programmatic
# dependent launch, as (output features, batch rows) per CTA.  The three narrow
# tiles differ only in their loops: PACKED_NARROW unrolls them by two rounds, faster
# on sm_100a; PACKED_NARROW_U4 unrolls its MMA loop by four, faster on sm_107a;
# PACKED_NARROW_FULL_STAGES also issues a full stage's MMAs without a per-step K
# test, faster on sm_103a.  PACKED_M64_N16 and PACKED_M64_N4 put sixteen output
# features in each chain's block of an M=64 MMA, on 16 or 4 batch rows (faster on
# sm_107a).  PACKED_BOX8 is PACKED on 8 batch rows: its activation box is 8 rows
# tall, so a batch of 8 loads no rows past its end (faster on sm_100a, sm_103a and
# sm_107a).  PACKED_M64_N8 puts sixteen output features in each chain's block of an
# M=64 MMA on 8 batch rows with an 8-row activation box (faster on sm_100a, sm_103a
# and sm_107a).
_TINYGEMM2_TCGEN05_TILES = {
    "PACKED_NARROW": (8, 2),
    "PACKED_NARROW_U4": (8, 2),
    "PACKED_NARROW_FULL_STAGES": (8, 2),
    "PACKED_M64_N16": (16, 16),
    "PACKED_M64_N4": (16, 4),
    "PACKED": (32, 16),
    "PACKED_BOX8": (32, 8),
    "PACKED_M64_N8": (16, 8),
    "ACCUMULATORS": (32, 64),
}
_NARROW_TILES = ("PACKED_NARROW", "PACKED_NARROW_U4", "PACKED_NARROW_FULL_STAGES")


class _Tcgen05Window(NamedTuple):
    """Shapes one tile serves: every bound inclusive, ``None`` unbounded."""

    tile: str
    min_batch: int
    max_batch: int
    bias: Optional[bool] = None  # None: with and without a bias
    min_output_features: Optional[int] = None
    max_output_features: Optional[int] = None
    min_input_features: Optional[int] = None
    max_input_features: Optional[int] = None
    min_weight_elements: Optional[int] = None  # output x input features
    one_wave: bool = False  # only while PACKED's grid fits one CTA per SM
    input_features_multiple: int = 1  # only inputs that are a multiple of this

    def admits(
        self, batch: int, out_features: int, in_features: int, bias: bool, sms: int
    ) -> bool:
        def within(value: int, low: Optional[int], high: Optional[int]) -> bool:
            return (low is None or value >= low) and (high is None or value <= high)

        features, rows = _TINYGEMM2_TCGEN05_TILES["PACKED"]
        packed_ctas = -(-out_features // features) * -(-batch // rows)
        return (
            within(batch, self.min_batch, self.max_batch)
            and (self.bias is None or self.bias == bias)
            and within(out_features, self.min_output_features, self.max_output_features)
            and within(in_features, self.min_input_features, self.max_input_features)
            and within(out_features * in_features, self.min_weight_elements, None)
            and (not self.one_wave or packed_ctas <= sms)
            and in_features % self.input_features_multiple == 0
            and (self.tile not in _NARROW_TILES or in_features % 64 == 0)
        )


# The windows where a tile measured faster than the kernel the call runs otherwise
# (the reference without a bias, tinygemm2_sm100 with one) on every measured shape,
# with and without programmatic dependent launch; the first window admitting a shape
# selects its tile, and every other shape keeps that kernel.  The bounds are the
# measured ranges.  Small weights stay on that kernel: the tensor-memory allocation
# adds a fixed half microsecond to a launch that takes four.  The narrow tiles serve
# up to 8 rows and up to 512 output features over long inputs; their windows differ
# per architecture, and they serve only inputs that are a multiple of 64 (128-byte
# rows): off that, some measured slower.  PACKED serves eight rows or fewer only
# without a bias and from 2880 output features (90 CTAs).  With a bias,
# tinygemm2_sm100 stays faster up to 27 rows except for PACKED on 9 to 16 rows from
# 2880 output features and an input of 2112, and up to 32 rows below 1536 output
# features.  From 9 rows without a bias (28 with one) PACKED serves while its grid
# fits one wave; past that ACCUMULATORS (64 rows per CTA) does.  On sm_107a PACKED
# serves only 8 rows without a bias and 9 to 16 rows with one, from 3584 output
# features at inputs of 2112 to 6144; with a bias 28 to 32 rows start at 2048 output
# features, and without one 9 to 16 rows start at 1728 (PACKED measured slower
# below).  PACKED_M64_N4 on 4 rows and PACKED_M64_N16 on 9 to 16 rows serve sm_107a
# alone, ahead of PACKED and ACCUMULATORS at 2048 to 3072 output features, where
# they measured faster.  On sm_107a, 17 to 23 rows without a bias
# whose activations take less than 128 KiB (batch x input features x 2 bytes) load
# slowly on ACCUMULATORS, whose 64-row boxes reach past the batch: PACKED serves 17 to
# 21 of these rows from an input of 2048, and 22 or 23 rows from about 120 KiB of
# activations (PACKED measured faster on every such shape, ACCUMULATORS slower than
# the reference on some; below 120 KiB, 22 and 23 rows stay on ACCUMULATORS, which
# measured faster than the reference there and close to PACKED).  PACKED_BOX8
# serves the 8 rows of PACKED's sm_107a window from an input of 3072 (measured
# faster than PACKED there).  On sm_107a, 8 rows without a bias go first to
# PACKED_M64_N8: from 3584 output features at an input of 2048, and at inputs that
# are a multiple of 64 (off that, some measured slower) from 7168 output features
# and an input of 2048, from 6144 and an input of 3072, and at 3584 to 4608 from
# an input of 4096, all up to 8192; then to PACKED_BOX8 at 4096 to 5120 output
# features and inputs of 2112 to 2560.  After PACKED's 8-row window, the other
# 8-row shapes without a bias at 3584 to 6784 output features go to PACKED_M64_N8
# at inputs of 2561 to 8192 that are a multiple of 64, from 4097 output features
# (up to an input of 3071 from 6144 output features), and to PACKED_BOX8 at the
# other inputs of 2056 to 8184 while PACKED's grid fits one wave (6784 output
# features take 212 CTAs).  Without a bias, PACKED_M64_N8 also serves 4 and 5 rows
# at 3584 to 4608 output features and an input of 2048; PACKED_M64_N4 serves 4
# rows at 4097 to 6144 output features and inputs of 3072 to 6144 that are a multiple
# of 64.  After those, PACKED_M64_N8 serves more shapes without a bias: 8 rows above
# 6784 and up to 7167 output features at inputs of 2112 to 2304 that are a multiple of
# 64, 6 and 7 rows at 4096 output features and an input of 2048, and 8 rows above 6784
# and up to 7056 output features at the inputs of 7169 to 8184 that are not a multiple
# of 64.  Then PACKED serves 9 to 16 rows without a bias at 4097 to 16384 output
# features and inputs of 1024 to 16384, also where its grid takes more than one wave,
# except above 6784 output features at inputs above 12288, where PACKED_M64_N16 serves
# them.  On sm_100a, after the windows above, PACKED_BOX8 and PACKED_M64_N8 first
# serve rows without a bias in windows from 3584 output features and an input of 2048.
# At 8192 output features and inputs of 2048 to 8192, PACKED_M64_N8 serves 1 to 8 rows
# when the input is a multiple of 64 and PACKED_BOX8 serves the other inputs.
# PACKED_BOX8 serves 1 to 7 rows at 4480 to 4608 output features and inputs of 2048 to
# 8192, and at 3584 to 4608 output features and inputs of 6152 to 8192.  After those,
# PACKED_M64_N8 serves 1 to 7 rows at 4480 to 8192 output features and inputs of 2048 to
# 6152, and at 7000 to 8192 and inputs up to 8192 that are a multiple of 64; it serves 8
# rows at 12288 output features and inputs up to 16384 that are a multiple of 64.  The
# rows without a bias that no window above admits then go to PACKED_M64_N8 on 8 rows at
# inputs that are a multiple of 64: at 4480 to 7167 output features and inputs of 2048
# to 4352 or 5632 to 6144, at 7168 to 16384 output features and inputs of 2048 to 6144,
# and at 7168 to 9216 output features and inputs above 6152 up to 8192.  PACKED_BOX8
# serves the other 8-row inputs of 2048 to 8192 at 4480 to 4608 output features, and 8
# rows at 3584 to 4479 output features and inputs of 6152 to 8192.  PACKED serves 9
# to 16 rows at 4097 to 16384 output features and inputs of 2048 to 12288, also where
# its grid takes more than one wave.  PACKED_BOX8 serves 8 rows at 4112 to 4464 output
# features and inputs of 2048 to 6144, and 1 to 7 rows there when the input is a
# multiple of 64.  Last, on inputs that are a multiple of 64, PACKED_M64_N8 serves 1 to
# 7 rows at 8193 to 16384 output features and inputs of 2048 to 4096, and at 8192 output
# features and inputs of 10240 to 12288; 8 rows at 16400 to 24576 output features and
# inputs of 2048 to 6144, at 6144 to 16384 output features and inputs of 1024 to 1984,
# and at 7168 to 7808 output features and inputs of 8256 to 12288; and 1 to 8 rows at
# 2880 to 3568 output features and inputs of 7168 to 16384.  On the other 8-row inputs,
# PACKED_BOX8 serves 7168 to 7808 output features and inputs of 8200 to 9992, and
# PACKED_M64_N8 10240 to 16384 output features and inputs of 2056 to 6136.  On
# sm_103a, after the windows above, PACKED_BOX8 and PACKED_M64_N8 first serve rows
# without a bias in windows from 3584 output features and an input of 2048.  PACKED_BOX8
# serves 8 rows at 4480 to 4608 output features and inputs of 2048 to 12288.  At 8192
# output features and inputs of 2048 to 8192, PACKED_M64_N8 serves 1 to 8 rows when the
# input is a multiple of 64 and PACKED_BOX8 serves the other inputs.  After those,
# PACKED_M64_N8 serves 8 rows at 4480 to 16384 output features and inputs up to 6152
# that are a multiple of 64, and 1 to 7 rows above 4608 and up to 8192 output features
# at inputs of 2048 to 6152.  Then, on inputs that are a multiple of 64, it serves 8
# rows at 8192 and 12288 output features and inputs of 2048 to 12288, and at 3584 output
# features and inputs above 12288 up to 16384.  The rows without a bias that no window
# above admits then go to PACKED_M64_N8 on 8 rows at 7424 to 9216 output features and
# inputs above 6152 up to 8192, and on 1 to 7 rows at 7424 to 8192 output features and
# those inputs when they are a multiple of 64; on 8 rows at 5008 to 7423 output features
# and inputs of 2048 to 5631, and at 7424 to 16384 output features and inputs of 2048 to
# 6144.  PACKED serves 9 to 16 rows at 4097 to 16384 output features and inputs of 2048
# to 12288, also where its grid takes more than one wave, and PACKED_BOX8 8 rows at 4097
# to 4479 output features and inputs of 2048 to 6144.  Last, PACKED_M64_N8 serves 1 to 7
# rows above 8192 output features: up to 16384 at inputs of 2048 to 4096, and up to
# 12288 at inputs above 4096 up to 8192 that are a multiple of 64; 8 rows at 16385 to
# 24576 output features and inputs of 2048 to 6144, and at 3584 to 16384 output features
# and inputs of 1024 to 2047; and 1 to 8 rows at 2880 to 3583 output features and inputs
# above 6144 up to 12288 that are a multiple of 64.
_LARGE_WEIGHT = 6_000_000


def _many_rows_nobias(low: int) -> Tuple[_Tcgen05Window, ...]:
    """Without a bias: 9 to 16 rows from ``low`` output features, 17 to 32 from 1536
    and 33 to 64 from 1024."""
    return (
        _Tcgen05Window(
            "PACKED", 9, 16, False, low, 4096, None, 6144, _LARGE_WEIGHT, one_wave=True
        ),
        _Tcgen05Window(
            "PACKED",
            17,
            32,
            False,
            1536,
            4096,
            None,
            6144,
            _LARGE_WEIGHT,
            one_wave=True,
        ),
        _Tcgen05Window(
            "ACCUMULATORS", 17, 32, False, 1536, 4096, None, 6144, _LARGE_WEIGHT
        ),
        _Tcgen05Window(
            "PACKED",
            33,
            64,
            False,
            1024,
            4096,
            None,
            6144,
            _LARGE_WEIGHT,
            one_wave=True,
        ),
        _Tcgen05Window(
            "ACCUMULATORS", 33, 64, False, 1024, 4096, None, 6144, _LARGE_WEIGHT
        ),
    )


def _many_rows_bias(low: int) -> Tuple[_Tcgen05Window, ...]:
    """With a bias: 28 to 32 rows from ``low`` output features, 33 to 64 from 1024."""
    return (
        _Tcgen05Window(
            "PACKED", 28, 32, True, low, 4096, None, 6144, _LARGE_WEIGHT, one_wave=True
        ),
        _Tcgen05Window(
            "ACCUMULATORS", 28, 32, True, low, 4096, None, 6144, _LARGE_WEIGHT
        ),
        _Tcgen05Window(
            "PACKED", 33, 64, True, 1024, 4096, None, 6144, _LARGE_WEIGHT, one_wave=True
        ),
        _Tcgen05Window(
            "ACCUMULATORS", 33, 64, True, 1024, 4096, None, 6144, _LARGE_WEIGHT
        ),
    )


# sm_107a, 17 to 23 rows without a bias: PACKED from the input in the table up to
# the largest input (a multiple of 8) that keeps the activations under 128 KiB.
_SM107_SMALL_ACTIVATIONS = tuple(
    _Tcgen05Window(
        "PACKED",
        batch,
        batch,
        False,
        1536,
        4096,
        low,
        65535 // batch // 8 * 8,
        _LARGE_WEIGHT,
    )
    for batch, low in (
        (17, 2048),
        (18, 2048),
        (19, 2048),
        (20, 2048),
        (21, 2048),
        (22, 2816),
        (23, 2688),
    )
)
_PACKED_FEW_ROWS = (
    _Tcgen05Window("PACKED", 1, 8, False, 2880, 4096, None, 6144, _LARGE_WEIGHT),
    _Tcgen05Window("PACKED", 9, 16, True, 2880, 4096, 2112, 6144, _LARGE_WEIGHT),
)
_SM100 = (
    _Tcgen05Window("PACKED_NARROW", 1, 4, True, 32, 512, 10240, 20480),
    _Tcgen05Window("PACKED_NARROW", 1, 8, True, 32, 128, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW", 3, 8, True, 32, 256, 3584, 4096),
    _Tcgen05Window("PACKED_NARROW", 3, 6, True, 32, 256, 7168, 20480),
    _Tcgen05Window("PACKED_NARROW", 4, 8, True, 32, 64, 3584, 20480),
    _Tcgen05Window("PACKED_NARROW", 1, 4, True, 32, 320, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW", 1, 6, True, 32, 384, 11264, 20480),
    _Tcgen05Window("PACKED_NARROW", 2, 6, True, 256, 320, 3584, 4096),
    _Tcgen05Window("PACKED_NARROW", 6, 9, True, 32, 192, 6144, 9216),
    _Tcgen05Window("PACKED_NARROW", 1, 4, False, 32, 512, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW", 2, 8, False, 32, 256, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW", 1, 6, False, 32, 384, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW", 6, 8, False, 64, 128, 3584, 20480),
    _Tcgen05Window("PACKED_NARROW", 3, 8, False, 32, 128, 3584, 3584),
    _Tcgen05Window("PACKED_NARROW", 3, 6, False, 32, 320, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW", 3, 8, False, 32, 64, 5120, 20480),
    _Tcgen05Window("PACKED_NARROW", 1, 8, False, 32, 256, 7168, 20480),
    *_PACKED_FEW_ROWS,
    *_many_rows_nobias(1536),
    *_many_rows_bias(1536),
    _Tcgen05Window("PACKED_BOX8", 1, 7, False, 4480, 4608, 2048, 8192),
    _Tcgen05Window("PACKED_BOX8", 1, 7, False, 3584, 4608, 6152, 8192),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 8192, 8192, 2048, 8192, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 1, 7, False, 8192, 8192, 2048, 8192, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 8192, 8192, 2048, 8192),
    _Tcgen05Window("PACKED_BOX8", 1, 7, False, 8192, 8192, 2048, 8192),
    _Tcgen05Window("PACKED_M64_N8", 1, 7, False, 4480, 8192, 2048, 6152),
    _Tcgen05Window(
        "PACKED_M64_N8", 1, 7, False, 7000, 8192, 2048, 8192, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        12288,
        12288,
        2048,
        16384,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 4480, 7167, 2048, 4352, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 4480, 7167, 5632, 6144, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        7168,
        16384,
        2048,
        6144,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 7168, 7679, 6153, 8192, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 7680, 9216, 6153, 8192, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED", 9, 16, False, 4224, 16384, 2048, 12288),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 4112, 4464, 2048, 6144),
    _Tcgen05Window(
        "PACKED_BOX8", 1, 7, False, 4112, 4464, 2048, 6144, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        1,
        7,
        False,
        9216,
        16384,
        2048,
        4096,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        1,
        7,
        False,
        8192,
        8192,
        10240,
        12288,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        16400,
        24576,
        2048,
        6144,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        6144,
        16384,
        1024,
        1984,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        1,
        8,
        False,
        2880,
        3568,
        7168,
        16384,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        7168,
        7808,
        8256,
        12288,
        input_features_multiple=64,
    ),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 7168, 7808, 8200, 9992),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 10240, 16384, 2056, 6136),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 4480, 4608, 2048, 8192),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 3584, 4479, 6152, 8192),
    _Tcgen05Window(
        "PACKED_M64_N8", 1, 7, False, 8193, 9215, 2048, 4096, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED", 9, 16, False, 4097, 4223, 2048, 12288),
)
_SM103 = (
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 6, True, 32, 384, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 3, 8, True, 32, 256, 7168, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 3, 8, True, 32, 256, 3072, 4096),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 2, True, 32, 512, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 6, True, 32, 384, 3584, 3584),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 4, 4, True, 32, 512, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 3, 6, True, 256, 320, 2560, 4096),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 3, 8, True, 128, 256, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 6, False, 32, 320, 7168, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 6, False, 32, 384, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 4, False, 32, 512, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 2, 8, False, 32, 256, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 3, 8, False, 32, 256, 3584, 3584),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 4, 8, False, 64, 192, 3584, 5120),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 4, 8, False, 64, 256, 5120, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 4, 6, False, 32, 64, 3584, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 6, False, 320, 320, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW_FULL_STAGES", 1, 2, False, 32, 512, 7168, 20480),
    *_PACKED_FEW_ROWS,
    *_many_rows_nobias(1536),
    *_many_rows_bias(1536),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 4480, 4608, 2048, 12288),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 8192, 8192, 2048, 8192, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 1, 7, False, 8192, 8192, 2048, 8192, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 8192, 8192, 2048, 8192),
    _Tcgen05Window("PACKED_BOX8", 1, 7, False, 8192, 8192, 2048, 8192),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        4480,
        16384,
        2048,
        6152,
        input_features_multiple=64,
    ),
    _Tcgen05Window("PACKED_M64_N8", 1, 7, False, 4609, 8192, 2048, 6152),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        8192,
        8192,
        2048,
        12288,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        12288,
        12288,
        2048,
        12288,
        input_features_multiple=64,
    ),
    _Tcgen05Window(
        "PACKED_M64_N8",
        8,
        8,
        False,
        3584,
        3584,
        12289,
        16384,
        input_features_multiple=64,
    ),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 7424, 8191, 6153, 8192),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 8193, 9216, 6153, 8192),
    _Tcgen05Window(
        "PACKED_M64_N8", 1, 7, False, 7424, 8192, 6153, 8192, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 5008, 7423, 2048, 5631),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 7424, 16384, 2048, 6144),
    _Tcgen05Window("PACKED", 9, 16, False, 4097, 16384, 2048, 6144),
    _Tcgen05Window("PACKED", 9, 16, False, 4097, 16384, 6145, 12288),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 4097, 4479, 2048, 6144),
    _Tcgen05Window("PACKED_M64_N8", 1, 7, False, 8193, 16384, 2048, 4096),
    _Tcgen05Window(
        "PACKED_M64_N8",
        1,
        7,
        False,
        8193,
        12288,
        4097,
        8192,
        input_features_multiple=64,
    ),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 16385, 24576, 2048, 6144),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 3584, 16384, 1024, 2047),
    _Tcgen05Window(
        "PACKED_M64_N8",
        1,
        8,
        False,
        2880,
        3583,
        6145,
        12288,
        input_features_multiple=64,
    ),
)
_SM107 = (
    _Tcgen05Window("PACKED_NARROW_U4", 1, 8, True, 64, 384, 10240, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 1, 6, True, 32, 512, 13312, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 3, 6, True, 64, 448, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 3, 8, True, 64, 256, 3072, 4096),
    _Tcgen05Window("PACKED_NARROW_U4", 6, 8, True, 128, 320, 7168, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 1, 6, True, 64, 512, 4096, 4096),
    _Tcgen05Window("PACKED_NARROW_U4", 1, 6, False, 32, 512, 8192, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 1, 8, False, 64, 384, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 3, 8, False, 32, 384, 7168, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 1, 4, False, 64, 512, 6144, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 3, 8, False, 64, 128, 5120, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 1, 6, False, 64, 512, 7168, 20480),
    _Tcgen05Window("PACKED_NARROW_U4", 4, 8, False, 64, 320, 4096, 8192),
    _Tcgen05Window("PACKED_M64_N4", 4, 4, True, 2048, 3072, 2880, 8192),
    _Tcgen05Window("PACKED_M64_N4", 4, 4, False, 2048, 3072, 2880, 10240),
    _Tcgen05Window("PACKED_M64_N16", 16, 16, True, 2048, 3072, 2880, 10240),
    _Tcgen05Window("PACKED_M64_N16", 11, 16, True, 2048, 3072, 6144, 10240),
    _Tcgen05Window("PACKED_M64_N16", 9, 16, True, 2048, 3072, 8192, 10240),
    _Tcgen05Window("PACKED_M64_N16", 16, 16, False, 2048, 3072, 2048, 10240),
    _Tcgen05Window("PACKED_M64_N16", 11, 16, False, 2048, 3072, 6144, 10240),
    _Tcgen05Window("PACKED_M64_N16", 9, 16, False, 2048, 3072, 8192, 10240),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 3584, 8192, 2048, 2048),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 7168, 8192, 2048, 8192, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 6144, 8192, 3072, 8192, input_features_multiple=64
    ),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 3584, 4608, 4096, 8192, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 4096, 5120, 2112, 2560),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 3584, 4096, 3072, 6144),
    _Tcgen05Window("PACKED", 8, 8, False, 3584, 4096, 2112, 6144),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 4097, 6143, 2561, 8192, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 3584, 6784, 2056, 2560, one_wave=True),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 6144, 6784, 2561, 3071, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_BOX8", 8, 8, False, 3584, 6784, 2561, 8184, one_wave=True),
    _Tcgen05Window(
        "PACKED_M64_N4", 4, 4, False, 4097, 6144, 3072, 6144, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_M64_N8", 4, 5, False, 3584, 4608, 2048, 2048),
    _Tcgen05Window(
        "PACKED_M64_N8", 8, 8, False, 6785, 7167, 2112, 2304, input_features_multiple=64
    ),
    _Tcgen05Window("PACKED_M64_N8", 6, 7, False, 4096, 4096, 2048, 2048),
    _Tcgen05Window("PACKED_M64_N8", 8, 8, False, 6785, 7056, 7169, 8184),
    _Tcgen05Window("PACKED", 9, 16, False, 4097, 6784, 1024, 16384),
    _Tcgen05Window("PACKED", 9, 16, False, 6785, 16384, 1024, 12288),
    _Tcgen05Window("PACKED_M64_N16", 9, 16, False, 6785, 16384, 12289, 16384),
    _Tcgen05Window("PACKED", 9, 16, True, 3584, 4096, 2112, 6144),
    *_SM107_SMALL_ACTIVATIONS,
    *_many_rows_nobias(1728),
    *_many_rows_bias(2048),
)
_TINYGEMM2_TCGEN05_WINDOWS = {
    (10, 0): _SM100,
    (10, 3): _SM103,
    (10, 7): _SM107,
}


@functools.cache
def _multiprocessor_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.cache
def _tinygemm2_tcgen05_enabled(
    disabled: str, compute_capability: Tuple[int, int]
) -> bool:
    if disabled == "1" or compute_capability not in _TINYGEMM2_TCGEN05_WINDOWS:
        return False
    # The sm_107a programs need CUDA 13.4, as tinygemm2_sm100 does on SM107.
    if compute_capability == (10, 7) and not is_cuda_version_at_least("13.4"):
        return False
    return version_at_least(torch.version.cuda, "12.8")


def _tinygemm2_tcgen05_program(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    has_bias: bool,
    use_pdl: bool,
) -> Optional[str]:
    """The identifier of the generated program serving the call, or ``None``."""
    capability = get_compute_capability(input.device)
    if not _tinygemm2_tcgen05_enabled(
        os.environ.get("FLASHINFER_DISABLE_TINYGEMM2_TCGEN05", "0"), capability
    ):
        return None
    batch, in_features = input.shape
    out_features = weight.shape[0]
    # The programs read the weight and the activations through tensor maps: 16-byte
    # aligned bases and rows.
    if (
        batch == 0
        or in_features % 8 != 0
        or input.data_ptr() % 16 != 0
        or weight.data_ptr() % 16 != 0
    ):
        return None
    tile = _tinygemm2_tcgen05_tile(
        capability,
        batch,
        out_features,
        in_features,
        has_bias,
        _multiprocessor_count(input.device.index),
    )
    if tile is None:
        return None
    # ACCUMULATORS writes a lane's 16 output features as one 32-byte store: a
    # 32-byte aligned ``out`` (its rows then start aligned, the output features
    # being a multiple of 16).
    if tile == "ACCUMULATORS" and out.data_ptr() % 32 != 0:
        return None
    return tile + ("" if has_bias else "_NOBIAS") + ("_PDL" if use_pdl else "")


@functools.cache
def _tinygemm2_tcgen05_tile(
    capability: Tuple[int, int],
    batch: int,
    out_features: int,
    in_features: int,
    has_bias: bool,
    sms: int,
) -> Optional[str]:
    """The tile of the first window admitting the shape, or ``None``: evaluated
    once per shape, so a call does not scan the windows."""
    for window in _TINYGEMM2_TCGEN05_WINDOWS[capability]:
        if window.admits(batch, out_features, in_features, has_bias, sms):
            return window.tile
    return None


@functools.cache
def get_tinygemm2_tcgen05_op():
    from ..jit import cake_tinygemm2_tcgen05 as programs

    @register_custom_op(
        "flashinfer::tinygemm2_tcgen05_op",
        mutates_args=["out"],
    )
    def tinygemm2_tcgen05_op(
        program: str,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        out: torch.Tensor,
    ) -> None:
        record = getattr(programs, program)
        tile = program.removesuffix("_PDL").removesuffix("_NOBIAS")
        # A misaligned 32-byte store faults the CUDA context: refuse the call instead.
        if tile == "ACCUMULATORS" and out.data_ptr() % 32 != 0:
            raise ValueError(
                f"{program} needs a 32-byte aligned out, got {out.data_ptr():#x}"
            )
        features, rows = _TINYGEMM2_TCGEN05_TILES[tile]
        batch, in_features = input.shape
        out_features = weight.shape[0]
        target = record.target_for(get_compute_capability(input.device))
        # The generated binding's arguments by name (``CakeProgram.args`` orders
        # them).  A program without the bias never reads ``bias``; ``out`` stands
        # in for it.
        values = {
            "W": weight,
            "X": input,
            "out": out,
            "bias": out if bias is None else bias,
            "M": out_features,
            "N": batch,
            "K": in_features,
            "grid_x": (out_features + features - 1) // features,
            "grid_y": (batch + rows - 1) // rows,
            "grid_z": 1,
        }
        if set(record.args) != set(values):
            raise RuntimeError(
                f"the generated {record.name} binding takes {list(record.args)}; "
                f"the host spells {sorted(values)}"
            )
        entry = record.entry(target)
        with torch.cuda.device(input.device), tvm_ffi.use_torch_stream():
            entry(*[values[name] for name in record.args])

    return tinygemm2_tcgen05_op


# tinygemm2_wgmma: generated sm_90a wgmma programs (flashinfer.jit.cake_tinygemm2_wgmma)
# with the reference's output bits.  They keep the reference's four K chains (the
# 256-wide input chunk j on chain j % 4), the K16 order within a chain and its +0
# start, and sum the chains and the bias in the reference's order with the same
# flushing adds, so a routed call returns exactly what the reference returns.
# Seven tiles, as (batch rows, output features) per CTA:
#
# * PACKED_U4 and PACKED_S8 pack the four chains of 16 output features into one
#   M64 wgmma; PACKED_U4 issues four input slices per ring stage (four stages),
#   PACKED_S8 one (eight stages).
# * ACCUMULATORS_CN16 and ACCUMULATORS_CN32 keep one accumulator per chain for
#   64 output features and 16 or 32 batch rows.
# * CLUSTER_S2U4, CLUSTER_S8U8 and CLUSTER_R2 run each chain that has a chunk
#   inside the input on its own CTA of a cluster of min(4, ceil(K / 256)) CTAs and
#   sum the chains in rank 0; CLUSTER_S8U8 serves inputs over 768 (a cluster of
#   four) and CLUSTER_R2 (32 output features) inputs up to 256 (one CTA).
#
# The packed tiles and ACCUMULATORS_CN32 load the four chains' boxes of an input
# slice with one tensor-map copy when the input is a multiple of 1024 (the _BOX3D
# programs); the cluster tiles read the input through 64-wide blocks when it is a
# multiple of 64, and CLUSTER_S2U4 through its _2D programs otherwise.
_TINYGEMM2_WGMMA_TILES = {
    "PACKED_U4": (8, 16),
    "PACKED_S8": (8, 16),
    "ACCUMULATORS_CN16": (16, 64),
    "ACCUMULATORS_CN32": (32, 64),
    "CLUSTER_S2U4": (8, 16),
    "CLUSTER_S8U8": (8, 16),
    "CLUSTER_R2": (16, 32),
}
_WGMMA_BOX3D_TILES = ("PACKED_U4", "PACKED_S8", "ACCUMULATORS_CN32")
_WGMMA_CLUSTER_TILES = ("CLUSTER_S2U4", "CLUSTER_S8U8", "CLUSTER_R2")


class _WgmmaWindow(NamedTuple):
    """Shapes one sm_90a tile serves: every bound inclusive."""

    tile: str
    min_batch: int
    max_batch: int
    bias: bool
    min_output_features: int
    max_output_features: int
    min_input_features: int
    max_input_features: int

    def admits(
        self, batch: int, out_features: int, in_features: int, bias: bool
    ) -> bool:
        return (
            self.min_batch <= batch <= self.max_batch
            and self.bias == bias
            and self.min_output_features <= out_features <= self.max_output_features
            and self.min_input_features <= in_features <= self.max_input_features
        )


# The windows where a tile measured at least 3% faster than the reference kernel
# on H100 at every measured shape (9 batches, 7 output and 8 input feature counts,
# with and without a bias, in each input view the host may select), in every one
# of up to four measurements; the first window admitting a shape selects its
# tile, and every other shape keeps the reference.  The bounds are the measured
# values: a shape between two measured values is served only when a window spans
# both.  The packed tiles serve inputs that are multiples of 64 only: off that
# step they measured at parity.
_TINYGEMM2_WGMMA_WINDOWS: Tuple[_WgmmaWindow, ...] = (
    _WgmmaWindow("CLUSTER_S2U4", 1, 16, False, 16, 512, 256, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 64, False, 16, 128, 256, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 64, False, 16, 512, 256, 256),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, False, 16, 4096, 256, 256),
    _WgmmaWindow("CLUSTER_S2U4", 1, 16, False, 16, 2048, 256, 256),
    _WgmmaWindow("CLUSTER_S2U4", 1, 32, False, 16, 1024, 256, 256),
    _WgmmaWindow("CLUSTER_R2", 64, 64, False, 1024, 4096, 256, 256),
    _WgmmaWindow("CLUSTER_R2", 32, 64, False, 2048, 4096, 256, 256),
    _WgmmaWindow("CLUSTER_R2", 13, 64, False, 2880, 4096, 256, 256),
    _WgmmaWindow("ACCUMULATORS_CN16", 64, 64, False, 512, 2048, 720, 720),
    _WgmmaWindow("CLUSTER_S2U4", 2, 4, False, 16, 4096, 256, 720),
    _WgmmaWindow("ACCUMULATORS_CN16", 32, 64, False, 1024, 2048, 720, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, False, 2048, 4096, 256, 720),
    _WgmmaWindow("ACCUMULATORS_CN16", 13, 32, False, 2048, 4096, 720, 720),
    _WgmmaWindow("ACCUMULATORS_CN32", 64, 64, False, 2048, 4096, 720, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, False, 16, 16, 256, 2048),
    _WgmmaWindow("PACKED_U4", 16, 16, False, 16, 512, 1024, 1024),
    _WgmmaWindow("PACKED_U4", 1, 4, False, 16, 512, 1024, 1024),
    _WgmmaWindow("PACKED_U4", 1, 8, False, 16, 128, 1024, 1024),
    _WgmmaWindow("PACKED_U4", 8, 32, False, 128, 512, 1024, 1024),
    _WgmmaWindow("PACKED_U4", 32, 64, False, 128, 128, 1024, 4096),
    _WgmmaWindow("PACKED_S8", 64, 64, False, 512, 512, 1024, 14336),
    _WgmmaWindow("PACKED_S8", 32, 32, False, 1024, 1024, 1024, 14336),
    _WgmmaWindow("ACCUMULATORS_CN16", 64, 64, False, 1024, 2048, 720, 14336),
    _WgmmaWindow("PACKED_S8", 13, 16, False, 2048, 2048, 1024, 14336),
    _WgmmaWindow("ACCUMULATORS_CN16", 16, 64, False, 2048, 2048, 720, 1024),
    _WgmmaWindow("PACKED_S8", 1, 7, False, 2880, 4096, 1024, 1024),
    _WgmmaWindow("PACKED_S8", 2, 8, False, 2880, 2880, 1024, 4096),
    _WgmmaWindow("ACCUMULATORS_CN16", 13, 32, False, 2880, 4096, 720, 14336),
    _WgmmaWindow("ACCUMULATORS_CN32", 64, 64, False, 2880, 4096, 720, 14336),
    _WgmmaWindow("ACCUMULATORS_CN16", 1, 32, False, 4096, 4096, 1024, 2048),
    _WgmmaWindow("CLUSTER_S2U4", 1, 16, False, 16, 128, 2048, 2048),
    _WgmmaWindow("CLUSTER_S2U4", 1, 64, False, 16, 16, 2048, 2048),
    _WgmmaWindow("PACKED_U4", 1, 4, False, 512, 512, 1024, 2048),
    _WgmmaWindow("PACKED_U4", 2, 32, False, 512, 512, 2048, 2880),
    _WgmmaWindow("PACKED_U4", 8, 16, False, 512, 1024, 2048, 2048),
    _WgmmaWindow("PACKED_S8", 1, 1, False, 2048, 2048, 2048, 2048),
    _WgmmaWindow("PACKED_S8", 7, 8, False, 2048, 4096, 2048, 2048),
    _WgmmaWindow("ACCUMULATORS_CN16", 32, 64, False, 2048, 2048, 720, 14336),
    _WgmmaWindow("CLUSTER_S2U4", 1, 4, False, 2880, 2880, 256, 2048),
    _WgmmaWindow("CLUSTER_S8U8", 1, 16, False, 16, 128, 2880, 14336),
    _WgmmaWindow("CLUSTER_S8U8", 1, 64, False, 16, 16, 2880, 14336),
    _WgmmaWindow("PACKED_U4", 13, 16, False, 512, 1024, 2048, 4096),
    _WgmmaWindow("PACKED_S8", 1, 8, False, 2880, 4096, 2880, 4096),
    _WgmmaWindow("PACKED_U4", 7, 32, False, 512, 512, 2048, 4096),
    _WgmmaWindow("PACKED_U4", 32, 32, False, 512, 512, 1024, 14336),
    _WgmmaWindow("PACKED_U4", 13, 16, False, 1024, 1024, 2048, 14336),
    _WgmmaWindow("CLUSTER_S2U4", 1, 4, False, 2880, 2880, 4096, 7168),
    _WgmmaWindow("PACKED_S8", 4, 8, False, 2880, 2880, 1024, 7168),
    _WgmmaWindow("CLUSTER_S2U4", 1, 16, True, 16, 2048, 256, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 64, True, 16, 128, 256, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 64, True, 16, 512, 256, 256),
    _WgmmaWindow("CLUSTER_S2U4", 1, 32, True, 16, 1024, 256, 256),
    _WgmmaWindow("CLUSTER_R2", 64, 64, True, 1024, 4096, 256, 256),
    _WgmmaWindow("CLUSTER_R2", 32, 64, True, 2048, 4096, 256, 256),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, True, 16, 4096, 256, 720),
    _WgmmaWindow("CLUSTER_R2", 13, 64, True, 2880, 4096, 256, 256),
    _WgmmaWindow("ACCUMULATORS_CN16", 64, 64, True, 512, 2048, 720, 1024),
    _WgmmaWindow("ACCUMULATORS_CN16", 32, 64, True, 1024, 2048, 720, 1024),
    _WgmmaWindow("ACCUMULATORS_CN16", 13, 32, True, 2880, 4096, 720, 14336),
    _WgmmaWindow("ACCUMULATORS_CN32", 64, 64, True, 2048, 4096, 720, 720),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, True, 16, 16, 256, 2048),
    _WgmmaWindow("CLUSTER_S2U4", 16, 16, True, 16, 128, 256, 2048),
    _WgmmaWindow("CLUSTER_S2U4", 1, 1, True, 16, 128, 256, 2048),
    _WgmmaWindow("ACCUMULATORS_CN16", 13, 32, True, 2048, 4096, 1024, 1024),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, True, 2880, 2880, 256, 2048),
    _WgmmaWindow("ACCUMULATORS_CN32", 64, 64, True, 2880, 4096, 720, 14336),
    _WgmmaWindow("ACCUMULATORS_CN16", 1, 32, True, 4096, 4096, 1024, 2048),
    _WgmmaWindow("CLUSTER_S2U4", 1, 16, True, 16, 128, 2048, 2048),
    _WgmmaWindow("CLUSTER_S2U4", 1, 64, True, 16, 16, 2048, 2048),
    _WgmmaWindow("PACKED_S8", 64, 64, True, 512, 512, 2048, 14336),
    _WgmmaWindow("PACKED_U4", 16, 16, True, 1024, 1024, 2048, 14336),
    _WgmmaWindow("PACKED_S8", 32, 32, True, 1024, 1024, 2048, 14336),
    _WgmmaWindow("ACCUMULATORS_CN16", 64, 64, True, 1024, 2048, 720, 14336),
    _WgmmaWindow("PACKED_S8", 13, 16, True, 2048, 2048, 2048, 14336),
    _WgmmaWindow("ACCUMULATORS_CN16", 32, 64, True, 2048, 2048, 720, 14336),
    _WgmmaWindow("CLUSTER_S8U8", 1, 16, True, 16, 128, 2880, 14336),
    _WgmmaWindow("CLUSTER_S8U8", 1, 64, True, 16, 16, 2880, 14336),
    _WgmmaWindow("PACKED_U4", 13, 16, True, 1024, 1024, 2880, 14336),
    _WgmmaWindow("PACKED_S8", 1, 8, True, 2880, 4096, 2880, 2880),
    _WgmmaWindow("PACKED_U4", 32, 32, True, 128, 512, 4096, 4096),
    _WgmmaWindow("CLUSTER_S2U4", 1, 8, True, 2880, 2880, 4096, 4096),
    _WgmmaWindow("PACKED_S8", 1, 8, True, 4096, 4096, 2880, 4096),
    _WgmmaWindow("CLUSTER_S2U4", 1, 1, True, 2880, 2880, 256, 7168),
    _WgmmaWindow("PACKED_U4", 32, 32, True, 512, 512, 14336, 14336),
)


@functools.cache
def _tinygemm2_wgmma_enabled(
    disabled: str, compute_capability: Tuple[int, int]
) -> bool:
    if disabled == "1" or compute_capability != (9, 0):
        return False
    return version_at_least(torch.version.cuda, "12.8")


@functools.cache
def _tinygemm2_wgmma_identifier(
    batch: int, out_features: int, in_features: int, has_bias: bool, use_pdl: bool
) -> Optional[str]:
    """The program of the first window admitting the shape, or ``None``:
    evaluated once per shape, so a call does not scan the windows."""
    for window in _TINYGEMM2_WGMMA_WINDOWS:
        if window.admits(batch, out_features, in_features, has_bias):
            tile = window.tile
            break
    else:
        return None
    if tile.startswith("PACKED") and in_features % 64 != 0:
        return None
    program = tile
    if tile in _WGMMA_BOX3D_TILES and in_features % 1024 == 0:
        program += "_BOX3D"
    if tile in _WGMMA_CLUSTER_TILES:
        if in_features % 64 != 0:
            if tile != "CLUSTER_S2U4":
                return None
            program += "_2D"
        cluster = min(4, -(-in_features // 256))
        if (tile == "CLUSTER_S8U8" and cluster != 4) or (
            tile == "CLUSTER_R2" and cluster != 1
        ):
            return None
        program += f"_C{cluster}"
    return program + ("" if has_bias else "_NOBIAS") + ("_PDL" if use_pdl else "")


def _tinygemm2_wgmma_program(
    input: torch.Tensor,
    weight: torch.Tensor,
    has_bias: bool,
    use_pdl: bool,
) -> Optional[str]:
    """The identifier of the generated sm_90a program serving the call, or ``None``."""
    if not _tinygemm2_wgmma_enabled(
        os.environ.get("FLASHINFER_DISABLE_TINYGEMM2_WGMMA", "0"),
        get_compute_capability(input.device),
    ):
        return None
    batch, in_features = input.shape
    # The programs read the weight and the activations through tensor maps: 16-byte
    # aligned bases and rows.
    if (
        batch == 0
        or in_features % 8 != 0
        or input.data_ptr() % 16 != 0
        or weight.data_ptr() % 16 != 0
    ):
        return None
    return _tinygemm2_wgmma_identifier(
        batch, weight.shape[0], in_features, has_bias, use_pdl
    )


def _tinygemm2_wgmma_tile(program: str) -> str:
    """The tile of a program identifier."""
    for tile in sorted(_TINYGEMM2_WGMMA_TILES, key=len, reverse=True):
        if program.startswith(tile):
            return tile
    raise ValueError(f"no tinygemm2_wgmma tile for {program}")


@functools.cache
def get_tinygemm2_wgmma_op():
    from ..jit import cake_tinygemm2_wgmma as programs

    @register_custom_op(
        "flashinfer::tinygemm2_wgmma_op",
        mutates_args=["out"],
    )
    def tinygemm2_wgmma_op(
        program: str,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        out: torch.Tensor,
    ) -> None:
        record = getattr(programs, program)
        tile = _tinygemm2_wgmma_tile(program)
        rows, features = _TINYGEMM2_WGMMA_TILES[tile]
        batch, in_features = input.shape
        out_features = weight.shape[0]
        # A cluster program launches one CTA per chain with a chunk inside the input.
        cluster = min(4, -(-in_features // 256)) if tile in _WGMMA_CLUSTER_TILES else 1
        target = record.target_for(get_compute_capability(input.device))
        # The generated binding's arguments by name (``CakeProgram.args`` orders
        # them).  A program without the bias never reads ``bias``; ``out`` stands
        # in for it.
        values = {
            "W": weight,
            "X": input,
            "out": out,
            "bias": out if bias is None else bias,
            "M": out_features,
            "N": batch,
            "K": in_features,
            "grid_x": cluster * ((batch + rows - 1) // rows),
            "grid_y": (out_features + features - 1) // features,
            "grid_z": 1,
        }
        if set(record.args) != set(values):
            raise RuntimeError(
                f"the generated {record.name} binding takes {list(record.args)}; "
                f"the host spells {sorted(values)}"
            )
        entry = record.entry(target)
        with torch.cuda.device(input.device), tvm_ffi.use_torch_stream():
            entry(*[values[name] for name in record.args])

    return tinygemm2_wgmma_op


def _tinygemm_bf16_cake(
    program: str,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    backend: Literal["cake"],
) -> None:
    """Launches the generated tcgen05 or wgmma program ``program`` (the Cake backend)."""
    if get_compute_capability(input.device) == (9, 0):
        get_tinygemm2_wgmma_op()(program, input, weight, bias, out)
    else:
        get_tinygemm2_tcgen05_op()(program, input, weight, bias, out)


@backend_requirement({}, common_check=_tinygemm_bf16_shape_checks)
@flashinfer_api(trace=tinygemm_bf16_trace)
def tinygemm_bf16(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> None:
    r"""SM90+ optimized small GEMM: ``out = input @ weight.T + bias`` (equivalent to F.linear).

    A latency-optimized, warp-specialized GEMM designed for tiny batch sizes
    (ideally 1-8 rows, where a single ``TILE_N=8`` tile covers the entire batch
    dimension) using Ampere-style HMMA instructions.  Uses TMA for async bulk
    data loads and ``mma.sync.aligned.m16n8k16`` tensor-core instructions with
    BF16 input/weight/bias/output and FP32 internal accumulation.  The
    warp-specialized design (384 threads: 4 compute + 8 DMA warps) with 16
    pipeline stages and 4x stage unroll trades off peak throughput in favor of
    minimal latency.  Adapted from the TensorRT-LLM ``tinygemm2`` kernel.

    Parameters
    ----------
    input : torch.Tensor
        Input activations of shape ``(batch_size, input_features)``.  Must be
        bfloat16, contiguous.  ``input_features`` must be a multiple of 64.
    weight : torch.Tensor
        Weight matrix of shape ``(output_features, input_features)``.  Must be
        bfloat16, contiguous (row-major).  ``output_features`` must be a multiple
        of 16.
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(batch_size, output_features)``.
        Must be bfloat16, contiguous.  Mutated in place.
    bias : Optional[torch.Tensor]
        Optional bias vector of shape ``(output_features,)``.  Must be bfloat16,
        contiguous.  If ``None``, zero bias is used.
    use_pdl : bool
        Enable Programmatic Dependent Launch (stream serialization).  When
        ``True``, the kernel uses ``cudaGridDependencySynchronize()`` to overlap
        DMA with the preceding kernel's compute.  Only enable when ALL preceding
        stream operations also use PDL, otherwise the kernel hangs.  The kernel
        may read ``weight`` and ``bias`` before that dependency wait, so the
        preceding kernel must not write them; only ``input`` may be its output.
        Defaults to ``False``.

    Notes
    -----
    Requires SM90+ (Hopper or newer).  Raises ``ValueError`` if tensor
    dimensions, dtypes, or alignment constraints are violated.

    On SM100/SM103 (B200/B300 class) devices the bias path dispatches to
    ``tinygemm2_sm100`` — generated variants of the same kernel with
    bit-identical outputs and lower latency (see
    ``csrc/tinygemm2_sm100.cu``).  Set ``FLASHINFER_DISABLE_TINYGEMM2_SM100=1``
    to force the reference implementation everywhere.

    On the same devices, shapes with large weights (up to 64 batch rows) and
    narrow outputs over long inputs dispatch to ``tinygemm2_tcgen05`` instead:
    generated tcgen05 programs whose outputs are bit-identical to the
    reference's (``flashinfer.jit.cake_tinygemm2_tcgen05``).  Set
    ``FLASHINFER_DISABLE_TINYGEMM2_TCGEN05=1`` to keep the kernels above.

    On SM90 (H100/H200 class) devices, the shapes the measured windows name
    dispatch to ``tinygemm2_wgmma``: generated wgmma programs whose outputs are
    bit-identical to the reference's (``flashinfer.jit.cake_tinygemm2_wgmma``).
    Set ``FLASHINFER_DISABLE_TINYGEMM2_WGMMA=1`` to keep the reference kernel.
    """
    program = _tinygemm2_wgmma_program(
        input, weight, bias is not None, use_pdl
    ) or _tinygemm2_tcgen05_program(input, weight, out, bias is not None, use_pdl)
    if program is not None:
        _tinygemm_bf16_cake(program, input, weight, bias, out, backend="cake")
    elif bias is None:
        get_tinygemm2_module().tinygemm2_nobias_op(input, weight, out, use_pdl)
    elif _use_tinygemm2_sm100(input.device):
        get_tinygemm2_sm100_module().tinygemm2_sm100_op(
            input, weight, bias, out, use_pdl
        )
    else:
        get_tinygemm2_module().tinygemm2_op(input, weight, bias, out, use_pdl)
