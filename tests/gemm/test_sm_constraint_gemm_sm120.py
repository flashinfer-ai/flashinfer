import pytest
import torch

from flashinfer.triton import sm_constraint_gemm
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize(
    "kernel,dtype,out_dtype,subtile",
    [
        ("gemm", torch.float32, torch.float32, False),
        ("gemm_persistent", torch.float32, torch.float32, False),
        ("gemm", torch.float16, torch.float16, False),
        ("gemm", torch.bfloat16, torch.bfloat16, False),
        ("gemm", torch.float8_e4m3fn, torch.bfloat16, False),
        ("gemm_persistent", torch.float16, torch.float16, False),
        ("gemm_persistent", torch.bfloat16, torch.bfloat16, False),
        ("gemm_persistent", torch.float8_e4m3fn, torch.bfloat16, False),
        *[
            ("gemm_descriptor_persistent", dtype, out_dtype, subtile)
            for dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
            for out_dtype in (
                dtype if dtype != torch.float8_e4m3fn else torch.bfloat16,
                torch.float32,
                torch.float8_e4m3fn,
            )
            for subtile in (False, True)
        ],
    ],
)
@pytest.mark.parametrize("shape", [(256, 256, 256), (129, 160, 144)])
@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_sm120_gemm_shared_memory(
    kernel, dtype, out_dtype, subtile, shape, beta, use_cuda_graph
):
    """Check SM120 GEMM correctness and resource limits in eager and graph modes."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    if get_compute_capability(torch.device("cuda")) != (12, 0):
        pytest.skip("Requires SM120's shared-memory limit")

    torch.manual_seed(120)
    m, n, k = shape
    a = (torch.randn((m, k), device="cuda") / 8).to(dtype)
    b = (torch.randn((n, k), device="cuda") / 8).to(dtype)
    c = torch.empty((m, n), device="cuda", dtype=out_dtype)
    alpha = 0.5
    reference = (alpha * (a.float() @ b.float().T) + beta * 2).to(out_dtype)
    kwargs = {}
    if kernel != "gemm":
        # Force each persistent program to process multiple tiles.
        kwargs["num_sms"] = 1
    if kernel == "gemm_descriptor_persistent":
        kwargs["EPILOGUE_SUBTILE"] = subtile
    else:
        b = b.T  # Also exercise a non-contiguous input.
    fn = getattr(sm_constraint_gemm, kernel)

    def run():
        """Reset C to isolate stage selection from the separate beta=0/NaN issue."""
        c.fill_(2)
        return fn(a, b, c=c, alpha=alpha, beta=beta, out_dtype=out_dtype, **kwargs)

    output = run()
    if use_cuda_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()

    assert output.data_ptr() == c.data_ptr()
    tol = 1e-2 if out_dtype == torch.bfloat16 else 1e-3
    rtol = 0.125 if out_dtype == torch.float8_e4m3fn else tol
    torch.testing.assert_close(c.float(), reference.float(), atol=tol, rtol=rtol)
