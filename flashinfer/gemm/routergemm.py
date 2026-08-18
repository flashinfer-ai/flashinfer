from ..api_logging import flashinfer_api
from ..trace.templates.gemm import (
    mm_M1_16_K6144_N256_trace,
    mm_M1_16_K7168_N256_trace,
    mm_M1_16_trace,
    tinygemm_bf16_trace,
)
from flashinfer.jit import (
    gen_dsv3_router_gemm_module,
    gen_tinygemm2_module,
    gen_tinygemm2_sm100_module,
)
import functools
import os
from types import SimpleNamespace
from typing import Optional
import torch
from flashinfer.utils import (
    get_compute_capability,
    register_custom_op,
    supported_compute_capability,
    backend_requirement,
    version_at_least,
)

# The kernel walks K in iterations of VPT (16 / sizeof(bfloat16) = 8) *
# kBlockSize (128) = 1024 elements, and every load is a 16-byte vector load, so
# the hidden dim has to be a whole number of iterations. The expert count, by
# contrast, is only the grid dimension -- any positive value works, which is why
# this path covers DeepSeek-V3 (256), Kimi-K2 (384) and Kimi-K3 (896) alike
# without enumerating them.
_ROUTER_GEMM_K_MULTIPLE = 1024
_ROUTER_GEMM_MIN_TOKENS = 1
_ROUTER_GEMM_MAX_TOKENS = 16
_ROUTER_GEMM_OUT_DTYPES = (torch.float32, torch.bfloat16)

# SM90 is included because the kernel is plain FMA plus warp shuffles; the only
# architecture-specific piece is PDL, which the kernel already guards on
# __CUDA_ARCH__ >= 900. SGLang dispatches its equivalent kernel at SM90+, so
# matching that range is what lets it drop its in-tree copy.
_ROUTER_GEMM_SUPPORTED_ARCHS = [90, 100, 103, 107]


def _router_gemm_shape_checks(
    mat_a,
    mat_b,
    out,
    launch_with_pdl,
    expected_hidden_dim=None,
    expected_num_experts=None,
    expected_out_dtype=None,
):
    """Validate a router-GEMM call.

    ``expected_*`` pin an axis to a single value for the fixed-shape aliases.
    When left as ``None`` the generic constraint is applied instead.
    """
    # Dimension checks
    if mat_a.dim() != 2:
        raise ValueError("mat_a must be a 2D tensor")
    if mat_b.dim() != 2:
        raise ValueError("mat_b must be a 2D tensor")
    if out.dim() != 2:
        raise ValueError("out must be a 2D tensor")

    # Stride checks (check these before dimension checks to give better error messages)
    if mat_a.stride(1) != 1:
        raise ValueError("mat_a must be row-major")
    if out.stride(1) != 1:
        raise ValueError("out must be row-major")
    if mat_b.stride(0) != 1:
        raise ValueError("mat_b must be column-major")
    # The kernel indexes expert n at mat_b + n * hidden_dim, so a column-major
    # view that is not also densely packed (a slice of a wider weight matrix,
    # say) would silently read the wrong columns.
    if mat_b.stride(1) != mat_b.shape[0]:
        raise ValueError("mat_b must be column-major and contiguous")

    if mat_a.shape[1] != mat_b.shape[0]:
        raise ValueError("mat_a.shape[1] must be equal to mat_b.shape[0]")
    if out.shape[0] != mat_a.shape[0]:
        raise ValueError("out.shape[0] must be equal to mat_a.shape[0]")
    if out.shape[1] != mat_b.shape[1]:
        raise ValueError("out.shape[1] must be equal to mat_b.shape[1]")

    # Problem size checks
    min_tokens = _ROUTER_GEMM_MIN_TOKENS
    max_tokens = _ROUTER_GEMM_MAX_TOKENS
    if mat_a.shape[0] < min_tokens or mat_a.shape[0] > max_tokens:
        raise ValueError(
            f"mat_a.shape[0] (num_tokens) must be between {min_tokens} and {max_tokens}"
        )
    if expected_hidden_dim is not None:
        if mat_a.shape[1] != expected_hidden_dim:
            raise ValueError(
                f"mat_a.shape[1] (hidden_dim) must be equal to {expected_hidden_dim}"
            )
    elif mat_a.shape[1] % _ROUTER_GEMM_K_MULTIPLE != 0:
        raise ValueError(
            f"mat_a.shape[1] (hidden_dim) must be a multiple of {_ROUTER_GEMM_K_MULTIPLE}, "
            f"got {mat_a.shape[1]}"
        )
    if expected_num_experts is not None:
        if mat_b.shape[1] != expected_num_experts:
            raise ValueError(
                f"mat_b.shape[1] (num_experts) must be equal to {expected_num_experts}"
            )
    elif mat_b.shape[1] < 1:
        raise ValueError("mat_b.shape[1] (num_experts) must be at least 1")

    # Data type checks
    if mat_a.dtype != torch.bfloat16:
        raise ValueError("mat_a must be a bfloat16 tensor")
    if mat_b.dtype != torch.bfloat16:
        raise ValueError("mat_b must be a bfloat16 tensor")
    if expected_out_dtype is not None:
        if out.dtype != expected_out_dtype:
            raise ValueError(f"out must be a {expected_out_dtype} tensor")
    elif out.dtype not in _ROUTER_GEMM_OUT_DTYPES:
        raise ValueError(
            f"out must be a torch.float32 or torch.bfloat16 tensor, got {out.dtype}"
        )

    return True


@supported_compute_capability(_ROUTER_GEMM_SUPPORTED_ARCHS)
def _mm_M1_16_shape_checks(mat_a, mat_b, out, launch_with_pdl):
    return _router_gemm_shape_checks(mat_a, mat_b, out, launch_with_pdl)


# TODO: other compute capabilities may be supported but are untested
@supported_compute_capability(_ROUTER_GEMM_SUPPORTED_ARCHS)
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
@supported_compute_capability(_ROUTER_GEMM_SUPPORTED_ARCHS)
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
@supported_compute_capability(_ROUTER_GEMM_SUPPORTED_ARCHS)
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


@functools.cache
def get_dsv3_router_gemm_module(num_experts: int, hidden_dim: int, out_float: bool):
    """Build (or fetch from cache) the router GEMM specialized for one shape.

    The expert count, hidden dim and output dtype are compile-time constants in
    the kernel, so each combination is its own small JIT module.
    """
    module = gen_dsv3_router_gemm_module(
        num_experts=num_experts, hidden_dim=hidden_dim, out_float=out_float
    ).build_and_load()

    dtype_tag = "f32" if out_float else "bf16"

    @register_custom_op(
        f"flashinfer::router_gemm_n{num_experts}_k{hidden_dim}_{dtype_tag}",
        mutates_args=["out"],
    )
    def router_gemm(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor,
        launch_with_pdl: bool = True,
    ) -> None:
        module.router_gemm_op(mat_a, mat_b, out, launch_with_pdl)

    return SimpleNamespace(router_gemm=router_gemm)


def _run_router_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool,
) -> None:
    get_dsv3_router_gemm_module(
        num_experts=mat_b.shape[1],
        hidden_dim=mat_a.shape[1],
        out_float=out.dtype == torch.float32,
    ).router_gemm(mat_a, mat_b, out, launch_with_pdl)


@backend_requirement({}, common_check=_mm_M1_16_shape_checks)
@flashinfer_api(trace=mm_M1_16_trace)
def mm_M1_16(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Latency-optimized GEMM for MoE router (expert-gate) projections.

    Computes ``out = mat_a @ mat_b`` for the tall-skinny shape a Mixture-of-Experts
    router produces during decode: a handful of token embeddings against the expert
    routing weights.  One thread block reduces the whole hidden dimension for a
    single expert, which beats a general-purpose GEMM at these sizes because the
    problem is entirely memory-bound on the weight matrix.

    The kernel is specialized at JIT time on ``(num_experts, hidden_dim, out.dtype)``,
    so a given combination is compiled once and cached.  This covers, among others,
    DeepSeek-V3 (``N=256, K=7168``), GLM-MoE-DSA (``N=256, K=6144``), Mistral
    Large 3 (``N=128, K=7168``), Kimi-K2 (``N=384, K=7168``) and Kimi-K3
    (``N=896``).

    Parameters
    ----------
    mat_a : torch.Tensor
        Input token embeddings of shape ``(M, K)``, where ``M`` is the number of
        tokens (1-16) and ``K`` is the hidden dimension (any multiple of 1024).
        Must be bfloat16, row-major (contiguous).
    mat_b : torch.Tensor
        Expert routing weights of shape ``(K, N)``, where ``N`` is the number of
        experts.  Must be bfloat16 and column-major *and* contiguous, i.e. a
        ``(N, K)`` row-major weight matrix passed as ``w.t()``.
    out : torch.Tensor
        Pre-allocated output tensor of shape ``(M, N)`` holding the routing
        scores.  Must be float32 or bfloat16, row-major (contiguous).  Mutated in
        place.
    launch_with_pdl : bool
        Whether to launch the kernel using Programmatic Dependent Launch.
        Defaults to ``True``.

    Notes
    -----
    Requires SM90 (Hopper) or newer.  Raises ``ValueError`` if tensor dimensions,
    strides, or dtypes fall outside the supported range.

    This kernel wins only while ``M`` is small; past that a general-purpose GEMM
    is faster.  The crossover is hardware-dependent -- on Blackwell (SM100/SM103)
    it sits around ``M=4``, on Hopper it holds to the full ``M=16`` range -- so
    callers are expected to gate on token count rather than assume this is always
    the better choice.
    """
    _run_router_gemm(mat_a, mat_b, out, launch_with_pdl)


@backend_requirement({}, common_check=_mm_M1_16_K7168_N128_shape_checks)
@flashinfer_api
def mm_M1_16_K7168_N128(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in Mistral Large 3.

    Fixed-shape alias of :func:`mm_M1_16` for the Mistral Large 3 MoE router
    (``K = 7168``, ``N = 128``, bfloat16 output).  Computes ``out = mat_a @ mat_b``
    where ``mat_a`` is a small batch of token embeddings (1-16 rows) and ``mat_b``
    is the expert routing weight matrix.

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
    Requires SM90 (Hopper) or newer.  The specialized problem-size optimization
    makes this significantly faster than general-purpose GEMM implementations for
    the router op.  Raises ``ValueError`` if tensor dimensions, strides, or dtypes
    do not match the expected Mistral Large 3 configuration.
    """
    _run_router_gemm(mat_a, mat_b, out, launch_with_pdl)


@backend_requirement({}, common_check=_mm_M1_16_K7168_N256_shape_checks)
@flashinfer_api(trace=mm_M1_16_K7168_N256_trace)
def mm_M1_16_K7168_N256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in DeepSeek-V3.

    Fixed-shape alias of :func:`mm_M1_16` for the DeepSeek-V3 MoE router
    (``K = 7168``, ``N = 256``, float32 output).  Computes ``out = mat_a @ mat_b``
    where ``mat_a`` is a small batch of token embeddings (1-16 rows) and ``mat_b``
    is the expert routing weight matrix.

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
    Requires SM90 (Hopper) or newer.  The specialized problem-size optimization
    makes this significantly faster than general-purpose GEMM implementations for
    the router op.  Raises ``ValueError`` if tensor dimensions, strides, or dtypes
    do not match the expected DeepSeek-V3 router configuration.
    """
    _run_router_gemm(mat_a, mat_b, out, launch_with_pdl)


@backend_requirement({}, common_check=_mm_M1_16_K6144_N256_shape_checks)
@flashinfer_api(trace=mm_M1_16_K6144_N256_trace)
def mm_M1_16_K6144_N256(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    out: torch.Tensor,
    launch_with_pdl: bool = True,
) -> None:
    r"""Optimized GEMM for the router operation in GLM-MoE-DSA.

    Fixed-shape alias of :func:`mm_M1_16` for the GLM-MoE-DSA MoE router
    (``K = 6144``, ``N = 256``, float32 output).  Computes ``out = mat_a @ mat_b``
    where ``mat_a`` is a small batch of token embeddings (1-16 rows) and ``mat_b``
    is the expert routing weight matrix.

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
    Requires SM90 (Hopper) or newer.  The specialized problem-size optimization
    makes this significantly faster than general-purpose GEMM implementations for
    the router op.  Raises ``ValueError`` if tensor dimensions, strides, or dtypes
    do not match the expected GLM-MoE-DSA configuration.
    """
    _run_router_gemm(mat_a, mat_b, out, launch_with_pdl)


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


# tinygemm2_sm100: generated SM100/SM103 variants of the same kernel. Loom
# schedules exactly porting csrc/tinygemm2.cu with bit-identical outputs;
# selected automatically for the bias path on B200/B300-class devices. Ring
# depth (stage 4/8/16) is selected inside the binding, mirroring the
# reference launcher convention.


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


# The generated kernels are validated on SM100 (B200) and SM103 (B300/GB300)
# exactly; other 10.x devices (e.g. SM107) pass is_sm100a_supported's
# major==10 predicate but must keep using the reference kernel.
_TINYGEMM2_SM100_SUPPORTED_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))


def _use_tinygemm2_sm100(device: torch.device) -> bool:
    if os.environ.get("FLASHINFER_DISABLE_TINYGEMM2_SM100", "0") == "1":
        return False
    return get_compute_capability(
        device
    ) in _TINYGEMM2_SM100_SUPPORTED_COMPUTE_CAPABILITIES and version_at_least(
        torch.version.cuda, "12.8"
    )


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
        stream operations also use PDL, otherwise the kernel hangs.  Defaults to
        ``False``.

    Notes
    -----
    Requires SM90+ (Hopper or newer).  Raises ``ValueError`` if tensor
    dimensions, dtypes, or alignment constraints are violated.

    On SM100/SM103 (B200/B300 class) devices the bias path dispatches to
    ``tinygemm2_sm100`` — generated variants of the same kernel with
    bit-identical outputs and lower latency (see
    ``csrc/tinygemm2_sm100.cu``).  Set ``FLASHINFER_DISABLE_TINYGEMM2_SM100=1``
    to force the reference implementation everywhere.
    """
    if bias is None:
        get_tinygemm2_module().tinygemm2_nobias_op(input, weight, out, use_pdl)
    elif _use_tinygemm2_sm100(input.device):
        get_tinygemm2_sm100_module().tinygemm2_sm100_op(
            input, weight, bias, out, use_pdl
        )
    else:
        get_tinygemm2_module().tinygemm2_op(input, weight, bias, out, use_pdl)
