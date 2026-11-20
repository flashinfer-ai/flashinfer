"""Numerical and graph contracts for native unquantized projections.

Both the small-row SIMT and broad BF16 tensor-core paths accumulate in
FP32, add bias before final rounding, and reuse geometry/type callables
across live row counts and noncontiguous views.
"""

from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


from b12x.preparation import PreparationSession, PreparedCall
from b12x.gemm import bf16_gemv


@pytest.fixture
def projection_session():
    with PreparationSession(
        device="cuda", autotune=False, compile_workers=2
    ) as session:
        yield session


def _prepare_projection(
    session, x, weight, *, out=None, bias=None, output_dtype=None, override=None
):
    declaration = bf16_gemv.plan(
        bf16_gemv.query_from_call(
            x,
            weight,
            out=out,
            bias=bias,
            output_dtype=output_dtype,
        ),
        override=override,
    )
    session.prepare(
        (
            declaration.request(
                name="projection",
                prepare_call=lambda state: PreparedCall(
                    run=lambda: state.run(x, weight, out=out, bias=bias),
                ),
            ),
        )
    )
    return declaration


def _mm(x, weight, *, out=None, bias=None, output_dtype=None, override=None):
    with PreparationSession(
        device=x.device, autotune=False, compile_workers=2
    ) as session:
        plan = _prepare_projection(
            session,
            x,
            weight,
            out=out,
            bias=bias,
            output_dtype=output_dtype,
            override=override,
        )
        return bf16_gemv.mm(
            x, weight, out=out, bias=bias, output_dtype=output_dtype, plan=plan
        )


def _op():
    return _mm


@cuda_required
@pytest.mark.parametrize("with_bias", [False, True])
def test_prepared_torch_projection_reuses_dynamic_graphs(with_bias):
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation import require_prepared

    torch.manual_seed(41)
    source = torch.randn(65, 1024, device="cuda", dtype=torch.bfloat16) * 0.125
    weight = torch.randn(512, 1024, device="cuda", dtype=torch.bfloat16) * 0.125
    bias = torch.randn(512, device="cuda", dtype=torch.bfloat16) if with_bias else None
    output = torch.empty(65, 512, device="cuda", dtype=torch.bfloat16)
    with PreparationSession(
        device=source.device, autotune=False, compile_workers=0
    ) as session:
        plan = _prepare_projection(
            session,
            source,
            weight,
            out=output,
            bias=bias,
            override=bf16_gemv.GemvConfig(backend="torch"),
        )
        session.freeze()
        state = require_prepared(plan, "gemm.bf16_gemv", source.device)
        launcher = state.launcher
        with kernel_resolution_guard("prepared cuBLAS projection"):
            for rows in (1, 3, 8, 17, 65):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    bf16_gemv.mm(
                        source[:rows], weight, out=output[:rows], bias=bias, plan=plan
                    )
                source.neg_()
                output.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                expected = source[:rows].float() @ weight.float().T
                if bias is not None:
                    expected += bias.float()
                torch.testing.assert_close(
                    output[:rows].float(), expected, rtol=0.01, atol=0.01
                )
                assert torch.isfinite(output[:rows]).all()
                assert torch.count_nonzero(output[:rows])
                assert torch.isnan(output[rows:]).all()
                assert state.launcher is launcher
                graph.reset()


def _assert_matches_f32_ref(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    ref = x.float() @ w.float().t()
    assert y.dtype == torch.bfloat16
    assert y.shape == (x.shape[0], w.shape[0])
    # f32 accumulation on both sides; the only expected difference is the
    # final bf16 rounding plus reduction-order noise far below it.
    torch.testing.assert_close(y.float(), ref, rtol=1e-2, atol=1e-2)


@cuda_required
@pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
@pytest.mark.parametrize(
    "n,k", [(64, 5120), (96, 5120), (128, 2048), (112, 1024), (1, 5120)]
)
def test_small_n_gemv_matches_reference(m, n, k):
    op = _op()
    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f32_ref(y, x, w)


@cuda_required
def test_last_element_contributes():
    """Regression: the strided K loop must cover the entire row."""
    op = _op()
    device = torch.device("cuda")
    m, n, k = 1, 64, 5120
    x = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
    w = torch.zeros(n, k, device=device, dtype=torch.bfloat16)
    x[0, k - 1] = 3.0
    w[n - 1, k - 1] = 2.0
    y = op(x, w)
    assert y[0, n - 1].item() == pytest.approx(6.0)
    assert y[0, : n - 1].abs().max().item() == 0.0


@cuda_required
@pytest.mark.parametrize("n", [32, 384, 512])
def test_long_k_row_tiles_reuse_graph_and_preserve_fp32_projection(
    n, projection_session
):
    """Prepared long-K projections retain FP32 output with runtime row counts."""
    from b12x.gemm import bf16_gemv

    torch.manual_seed(419132)
    source = torch.randn(17, 5120, device="cuda").bfloat16() * 0.125
    weight = torch.randn(n, 5120, device="cuda").bfloat16() * 0.125
    bias = torch.linspace(-0.25, 0.25, n, device="cuda")
    storage = torch.full((18, n + 8), 123.0, device="cuda")
    output = storage[:17, :n]
    plan = _prepare_projection(
        projection_session, source, weight, out=output, bias=bias
    )
    projection_session.freeze()
    try:
        for rows in (1, 3, 7, 8, 9, 17):
            storage.fill_(123)
            bf16_gemv.mm(source[:rows], weight, out=output[:rows], bias=bias, plan=plan)
            oracle = source[:rows].double() @ weight.double().T + bias.double()
            torch.testing.assert_close(
                output[:rows].double(), oracle, rtol=2e-5, atol=2e-5
            )
            assert (storage[rows:] == 123).all()
            assert (storage[:, n:] == 123).all()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_gemv.mm(source[:8], weight, out=output[:8], bias=bias, plan=plan)
        source.neg_()
        bias.add_(0.03125)
        storage.fill_(123)
        graph.replay()
        oracle = source[:8].double() @ weight.double().T + bias.double()
        torch.testing.assert_close(output[:8].double(), oracle, rtol=2e-5, atol=2e-5)
        assert (storage[8:] == 123).all()
        assert (storage[:, n:] == 123).all()
    finally:
        graph.reset()


@cuda_required
def test_noncontiguous_x():
    """The native scalar path must read a strided column view correctly."""
    op = _op()
    torch.manual_seed(3)
    device = torch.device("cuda")
    big = torch.randn(4, 2 * 2048, device=device, dtype=torch.bfloat16)
    x = big[:, ::2]  # non-contiguous (4, 2048)
    w = torch.randn(96, 2048, device=device, dtype=torch.bfloat16)
    y = op(x, w)
    _assert_matches_f32_ref(y, x.contiguous(), w)


@cuda_required
@pytest.mark.parametrize(
    "input_dtype,weight_dtype,output_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.float32, torch.float32),
        (torch.float32, torch.float32, torch.bfloat16),
    ],
)
def test_unquantized_bias_and_live_rows_reuse_native_graph(
    input_dtype,
    weight_dtype,
    output_dtype,
    projection_session,
):
    from b12x.gemm import bf16_gemv

    device = torch.device("cuda")
    torch.manual_seed(41091)
    capacity, n, k = 17, 97, 131
    # Odd K and column-strided sources exercise the non-vectorized native path.
    source = torch.randn(capacity, k * 2, device=device, dtype=input_dtype)[:, ::2]
    weight = torch.randn(n, k, device=device, dtype=weight_dtype) * 0.125
    bias = torch.linspace(-0.03, 0.04, n, device=device, dtype=torch.float32)
    output = torch.empty(capacity, n + 3, device=device, dtype=output_dtype)

    plan = _prepare_projection(
        projection_session, source, weight, bias=bias, out=output[:, :n]
    )

    def launch(rows):
        return bf16_gemv.mm(
            source[:rows], weight, bias=bias, out=output[:rows, :n], plan=plan
        )

    launch(1)
    torch.cuda.synchronize()
    projection_session.freeze()
    try:
        for rows in (0, 1, 7, capacity):
            output.fill_(123)
            actual = launch(rows)
            expected = (source[:rows].double() @ weight.double().T + bias.double()).to(
                output_dtype
            )
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-5 if output_dtype == torch.float32 else 1e-2,
                atol=2e-5 if output_dtype == torch.float32 else 1e-2,
            )
            torch.testing.assert_close(
                output[:, n:], torch.full_like(output[:, n:], 123)
            )
            torch.testing.assert_close(
                output[rows:], torch.full_like(output[rows:], 123)
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = launch(capacity)
        address = captured.data_ptr()
        source.mul_(0.5)
        bias.add_(0.03125)
        graph.replay()
        torch.cuda.synchronize()
        assert captured.data_ptr() == address
        expected = (source.double() @ weight.double().T + bias.double()).to(
            output_dtype
        )
        torch.testing.assert_close(
            captured,
            expected,
            rtol=1e-5 if output_dtype == torch.float32 else 1e-2,
            atol=2e-5 if output_dtype == torch.float32 else 1e-2,
        )
    finally:
        graph.reset()


@cuda_required
@pytest.mark.parametrize("n", [384, 512, 1024])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_prefill_projection_reuses_warm_kernel_and_preserves_live_graph_inputs(
    n,
    output_dtype,
    projection_session,
):
    """Both dispatch regimes share an unquantized contract and caller ownership."""
    from b12x.gemm import bf16_gemv

    torch.manual_seed(415122)
    capacity, k = 4096, 5120
    source = torch.randn(capacity, k, device="cuda").bfloat16()
    weight = (torch.randn(n, k, device="cuda") / k**0.5).bfloat16()
    output = torch.empty(capacity, n, device="cuda", dtype=output_dtype)
    source_copy, weight_copy = source.clone(), weight.clone()
    plan = _prepare_projection(projection_session, source, weight, out=output)
    projection_session.freeze()
    try:
        for rows in (1, 17, 255, 256, 513, capacity):
            output.fill_(float("nan"))
            bf16_gemv.mm(source[:rows], weight, out=output[:rows], plan=plan)
            expected = source[:rows].double() @ weight.double().T
            if output_dtype == torch.float32:
                torch.testing.assert_close(
                    output[:rows].double(), expected, atol=1e-6, rtol=1e-6
                )
            else:
                torch.testing.assert_close(
                    output[:rows].double(), expected, atol=0.004, rtol=0.004
                )
            assert torch.isnan(output[rows:]).all()
        torch.testing.assert_close(source, source_copy, atol=0, rtol=0)
        torch.testing.assert_close(weight, weight_copy, atol=0, rtol=0)
        output.fill_(float("nan"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_gemv.mm(source[:513], weight, out=output[:513], plan=plan)
        source.mul_(0.5)
        weight.mul_(0.25)
        graph.replay()
        torch.cuda.synchronize()
        expected = source[:513].double() @ weight.double().T
        torch.testing.assert_close(
            output[:513].double(),
            expected,
            atol=1e-6 if output_dtype == torch.float32 else 0.004,
            rtol=1e-6 if output_dtype == torch.float32 else 0.004,
        )
        assert torch.isnan(output[513:]).all()
    finally:
        graph.reset()


@cuda_required
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("m,n,k", [(257, 2048, 1024), (129, 1025, 513)])
def test_broad_bf16_projection(output_dtype, m, n, k):
    """Aligned prefill and simultaneous M/N/K tails retain BF16 operands."""
    from b12x.gemm import bf16_gemv

    torch.manual_seed(41092)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125
    actual = _mm(x, weight, output_dtype=output_dtype)
    expected = (x.double() @ weight.double().T).to(output_dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-5 if output_dtype == torch.float32 else 1e-2,
        atol=2e-5 if output_dtype == torch.float32 else 1e-2,
    )


@cuda_required
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_bf16_bias_is_added_before_output_rounding(output_dtype):
    """A rounded GEMM followed by bias would lose the entire residual."""
    from b12x.gemm import bf16_gemv

    m, n, k = 129, 1025, 72
    x = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    weight[:, 0] = 1.0
    # The last eight-element vector lies in a partial K64 tile.
    weight[:, -1] = 2**-8
    bias = torch.full((n,), -1.0, device="cuda", dtype=torch.float32)
    actual = _mm(x, weight, bias=bias, output_dtype=output_dtype)
    torch.testing.assert_close(actual, torch.full_like(actual, 2**-8), rtol=0, atol=0)


@cuda_required
@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
)
def test_fp32_operand_is_not_rounded_for_tensor_cores(input_dtype, weight_dtype):
    """Low FP32 bits survive on either operand for broad multi-row inputs."""
    from b12x.gemm import bf16_gemv

    m, n, k = 33, 65, 72
    x = torch.zeros(m, k, device="cuda", dtype=input_dtype)
    weight = torch.zeros(n, k, device="cuda", dtype=weight_dtype)
    x[:, -1] = 1.0 + (2**-12 if input_dtype == torch.float32 else 0)
    weight[:, -1] = 1.0 + (2**-12 if weight_dtype == torch.float32 else 0)
    bias = torch.full((n,), -1.0, device="cuda", dtype=torch.float32)
    actual = _mm(x, weight, bias=bias, output_dtype=torch.float32)
    torch.testing.assert_close(actual, torch.full_like(actual, 2**-12), rtol=0, atol=0)


@cuda_required
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_prepared_layouts_cover_live_rows_under_freeze(
    output_dtype, projection_session
):
    from b12x.gemm import bf16_gemv

    torch.manual_seed(41093)
    capacity, n, k = 257, 1031, 136
    sources = [
        torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16) * 0.125,
        (torch.randn(capacity, 2 * k + 1, device="cuda", dtype=torch.bfloat16) * 0.125)[
            :, 1::2
        ],
    ]
    weights = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125,
        (torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.125).T,
    ]
    bias = torch.linspace(-0.25, 0.25, n, device="cuda", dtype=torch.float32)
    storage = torch.empty(capacity + 1, n + 3, device="cuda", dtype=output_dtype)
    output = storage[:capacity, 1 : n + 1]
    plans = [
        _prepare_projection(projection_session, x, weight, out=output, bias=bias)
        for x, weight in zip(sources, weights, strict=True)
    ]
    projection_session.freeze()
    try:
        for x, weight, plan in zip(sources, weights, plans, strict=True):
            for rows in (0, 1, 8, 9, 127, 128, 129, capacity):
                storage.fill_(123)
                actual = bf16_gemv.mm(
                    x[:rows], weight, bias=bias, out=output[:rows], plan=plan
                )
                expected = (x[:rows].double() @ weight.double().T + bias.double()).to(
                    output_dtype
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-5 if output_dtype == torch.float32 else 1e-2,
                    atol=2e-5 if output_dtype == torch.float32 else 1e-2,
                )
                torch.testing.assert_close(
                    storage[:, 0], torch.full_like(storage[:, 0], 123)
                )
                torch.testing.assert_close(
                    storage[:, n + 1 :], torch.full_like(storage[:, n + 1 :], 123)
                )
                torch.testing.assert_close(
                    storage[rows:], torch.full_like(storage[rows:], 123)
                )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = bf16_gemv.mm(
                sources[1], weights[1], bias=bias, out=output, plan=plans[1]
            )
        sources[1].mul_(0.5)
        bias.add_(0.03125)
        graph.replay()
        torch.cuda.synchronize()
        assert actual.data_ptr() == output.data_ptr()
        expected = (sources[1].double() @ weights[1].double().T + bias.double()).to(
            output_dtype
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-5 if output_dtype == torch.float32 else 1e-2,
            atol=2e-5 if output_dtype == torch.float32 else 1e-2,
        )
        torch.testing.assert_close(
            storage[capacity:], torch.full_like(storage[capacity:], 123)
        )
    finally:
        graph.reset()


@cuda_required
def test_prefill_router_topk_agrees_with_fp64_and_scalar_projection():
    """A 384-expert router keeps the same six selected expert identifiers."""
    from b12x.gemm import bf16_gemv
    from b12x.gemm.bf16_gemv._tuning import GemvConfig

    torch.manual_seed(415123)
    source = torch.randn(4096, 5120, device="cuda").bfloat16()
    weight = (torch.randn(384, 5120, device="cuda") / 5120**0.5).bfloat16()
    scalar = torch.empty(4096, 384, device="cuda")
    _mm(source, weight, out=scalar, override=GemvConfig(backend="simt"))
    result = _mm(source, weight, output_dtype=torch.float32)
    oracle = source.double() @ weight.double().T
    for value in (scalar, result):
        torch.testing.assert_close(value.double(), oracle, rtol=1e-6, atol=1e-6)
        assert torch.equal(
            value.topk(6, dim=-1).indices, oracle.topk(6, dim=-1).indices
        )


@cuda_required
@pytest.mark.parametrize("n", [384, 512])
def test_prefill_projection_retains_small_terms_between_cancelling_large_terms(n):
    """Compensated carry preserves exact representable BF16 products in FP32."""
    from b12x.gemm import bf16_gemv

    source = torch.ones(256, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.zeros(n, 5120, device="cuda", dtype=torch.bfloat16)
    weight[:, ::32] = 1024
    weight[:, 1::32] = 0.015625
    weight[:, 31::32] = -1024
    result = _mm(source, weight, output_dtype=torch.float32)
    expected = source.double() @ weight.double().T
    torch.testing.assert_close(result.double(), expected, atol=0, rtol=0)
