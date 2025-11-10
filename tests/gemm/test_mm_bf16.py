import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bf16
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("m", [1, 48, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_mm_bf16(m: int, n: int, k: int, res_dtype: torch.dtype):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    cc_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_bf16.is_compute_capability_supported(cc_number):
        pytest.skip(
            f"mm_bf16 requires one of the following compute capabilities: "
            f"{sorted(mm_bf16._supported_ccs)}. "
            f"Detected sm{cc_number}."
        )

    torch.manual_seed(42)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([k, n], device="cuda", dtype=torch.bfloat16)
    reference = torch.mm(a.float(), b.float())

    out = torch.empty([m, n], device="cuda", dtype=res_dtype)
    with autotune():
        mm_bf16(a, b, out=out, out_dtype=res_dtype)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.float().reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
