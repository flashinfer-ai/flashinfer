import torch
import pytest
import torch.nn.functional as F
from flashinfer.utils import get_compute_capability


def _skip_if_not_sm90():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 9:
        pytest.skip("tinygemm2 requires SM90+")


# Positive tests — parameterized correctness checks
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize(
    "output_features,input_features",
    [
        (16, 64),
        (256, 7168),
        (2048, 7168),
        (128, 4096),
    ],
)
def test_tinygemm_bf16(batch_size, output_features, input_features):
    _skip_if_not_sm90()
    from flashinfer.gemm import tinygemm_bf16

    input = torch.randn(batch_size, input_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(batch_size, output_features, device="cuda", dtype=torch.bfloat16)

    tinygemm_bf16(input, weight, out, bias=bias)

    # Reference in FP32 for accuracy
    ref = F.linear(input.float(), weight.float(), bias.float()).bfloat16()

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


# No-bias test — validates bias=None zero-fill path
@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_tinygemm_bf16_no_bias(batch_size):
    _skip_if_not_sm90()
    from flashinfer.gemm import tinygemm_bf16

    input_features = 256
    output_features = 128

    input = torch.randn(batch_size, input_features, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.bfloat16
    )
    out = torch.empty(batch_size, output_features, device="cuda", dtype=torch.bfloat16)

    tinygemm_bf16(input, weight, out)

    ref = (input.float() @ weight.float().T).bfloat16()

    cos_sim = F.cosine_similarity(
        ref.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


# PDL tests — back-to-back launches with programmatic dependent launch
@pytest.mark.parametrize("num_launches", [2, 4, 8])
@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_tinygemm_bf16_pdl_back_to_back(num_launches, batch_size):
    """Test use_pdl=True with back-to-back launches on a clean stream.

    PDL (Programmatic Dependent Launch) enables overlapping DMA of kernel N
    with compute of kernel N-1. The first kernel's cudaGridDependencySynchronize()
    sees no previous PDL grid and returns immediately. Each subsequent kernel
    waits for the previous one's cudaTriggerProgrammaticLaunchCompletion() signal.

    We sync the device before launching to ensure no non-PDL ops are pending
    on the stream, then fire all PDL kernels back-to-back without host sync
    in between.
    """
    _skip_if_not_sm90()
    from flashinfer.gemm import tinygemm_bf16

    input_features = 4096
    output_features = 256

    # Pre-allocate everything before the PDL launch burst
    inputs = [
        torch.randn(batch_size, input_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_launches)
    ]
    weight = torch.randn(
        output_features, input_features, device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn(output_features, device="cuda", dtype=torch.bfloat16)
    outs = [
        torch.empty(batch_size, output_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_launches)
    ]

    # Ensure no pending non-PDL work on the stream
    torch.cuda.synchronize()

    # Fire all PDL kernels back-to-back (no host sync between them)
    for i in range(num_launches):
        tinygemm_bf16(inputs[i], weight, outs[i], bias=bias, use_pdl=True)

    # Sync and check correctness for every launch
    torch.cuda.synchronize()

    for i in range(num_launches):
        ref = F.linear(inputs[i].float(), weight.float(), bias.float()).bfloat16()
        cos_sim = F.cosine_similarity(
            ref.reshape(-1).float(), outs[i].reshape(-1).float(), dim=0
        )
        assert cos_sim > 0.99, (
            f"Launch {i}/{num_launches}: cosine similarity {cos_sim:.6f} < 0.99"
        )
