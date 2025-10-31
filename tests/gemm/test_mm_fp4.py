import pytest
import torch
import torch.nn.functional as F
from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
)
from flashinfer.utils import get_compute_capability, LibraryError
from flashinfer.gemm import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR


def _test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not mm_fp4.is_backend_supported(backend, compute_capability_number):
        pytest.skip(
            f"Skipping test for {backend} because it is not supported on compute capability {compute_capability_number}."
        )

    if backend == "trtllm":
        if res_dtype == torch.float16:
            pytest.skip("Skipping test for trtllm fp4 with float16")
        if compute_capability[0] in [11, 12]:
            pytest.skip("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if not use_nvfp4 and backend not in ["cudnn", "auto"]:
        pytest.skip("mx_fp4 is only supported for cudnn and auto backends")

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    reference = torch.mm(input, mat2.T)

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    try:
        with autotune(auto_tuning):
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
                skip_check=False,
            )

        cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
        assert cos_sim > 0.97
    except LibraryError as e:
        # TODO: Remove this check once cuDNN backend version is updated to 9.14.0
        if str(e) == CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR:
            pytest.xfail(str(e))
        else:
            pytest.fail(str(e))


# TODO: Consdier splitting this function up for the various backends
@pytest.mark.parametrize("m", [1, 48, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["trtllm", "cudnn", "cutlass"])
@pytest.mark.parametrize("use_128x4_sf_layout", [False, True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Non-auto backends
    _test_mm_fp4(
        m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
    )


# Split tests for checking auto functionality
@pytest.mark.parametrize("m", [1, 48, 256, 512])
@pytest.mark.parametrize("n", [256, 512])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_128x4_sf_layout", [True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4_backend_auto(
    m, n, k, res_dtype, use_128x4_sf_layout, auto_tuning, fp4_type
):
    # Some test cases for auto backend.
    _test_mm_fp4(m, n, k, res_dtype, "auto", use_128x4_sf_layout, auto_tuning, fp4_type)


if __name__ == "__main__":
    pytest.main([__file__])
