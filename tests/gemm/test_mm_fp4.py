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
from flashinfer.utils import (
    get_compute_capability,
    is_sm12x_supported,
    version_at_least,
    LibraryError,
)
from flashinfer.gemm.gemm_base import CUDNN_FP4_MXFP4_SM120_CUDNN_VERSION_ERROR


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
    if backend == "cute-dsl":
        if not use_128x4_sf_layout:
            pytest.skip("cute_dsl backend only supports 128x4 SF layout")
        if compute_capability[0] not in [10]:
            pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    if backend == "b12x":
        if not use_128x4_sf_layout:
            pytest.skip("b12x backend only supports 128x4 SF layout")
        if compute_capability[0] != 12:
            pytest.skip("b12x backend only supports SM120/SM121 GPUs.")
        if not use_nvfp4:
            pytest.skip("b12x backend only supports NVFP4 (sf_vec_size=16).")
        if torch.version.cuda and int(torch.version.cuda.split(".")[0]) < 13:
            pytest.skip("b12x backend requires CUDA 13+.")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if not use_nvfp4 and backend not in ["cudnn", "auto", "cute-dsl"]:
        pytest.skip("mx_fp4 is only supported for cudnn, cute-dsl, and auto backends")

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
@pytest.mark.parametrize(
    "m",
    [1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 15, 16, 17, 20, 24, 31, 32, 48, 64, 128, 256, 512],
)
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["trtllm", "cudnn", "cutlass", "cute-dsl", "b12x"])
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


# Regression (#3560): b12x must accept ragged K (real floor K%32==0, not tile_k=128).
# K=192 (packed_k=96) is the shape #3560 broke; both auto_tuning values hit distinct paths.
@pytest.mark.parametrize("k", [96, 192])
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_mm_fp4_b12x_ragged_k(k, auto_tuning):
    _test_mm_fp4(
        m=64,
        n=512,
        k=k,
        res_dtype=torch.bfloat16,
        backend="b12x",
        use_128x4_sf_layout=True,
        auto_tuning=auto_tuning,
        fp4_type="nvfp4",
    )


# K % 32 != 0 violates TMA 16-byte alignment; explicit b12x must reject cleanly.
def test_mm_fp4_b12x_misaligned_k_raises():
    device = torch.device("cuda")
    if not (
        is_sm12x_supported(device) and version_at_least(torch.version.cuda, "13.0")
    ):
        pytest.skip("b12x backend requires SM120/SM121 + CUDA 13+.")
    m, n, k = 64, 512, 112  # k % 32 == 16
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    g_in = (448 * 6) / a.float().abs().nan_to_num().max()
    g_w = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_s = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_s = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    res = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 32"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            1.0 / (g_in * g_w),
            torch.bfloat16,
            res,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="b12x",
            use_nvfp4=True,
            skip_check=False,
        )


def test_mm_fp4_cute_dsl_misaligned_n_raises():
    device = torch.device("cuda")
    if get_compute_capability(device)[0] != 10:
        pytest.skip("cute_dsl backend only supports SM100/SM103 GPUs.")
    m, n, k = 16, 130, 128  # n % 8 == 2
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    g_in = (448 * 6) / a.float().abs().nan_to_num().max()
    g_w = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_s = nvfp4_quantize(
        a, g_in, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_s = nvfp4_quantize(
        b, g_w, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    res = torch.empty([m, n], device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="N % 8 == 0"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_s,
            b_s.T,
            1.0 / (g_in * g_w),
            torch.bfloat16,
            res,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cute-dsl",
            use_nvfp4=True,
            skip_check=False,
        )


if __name__ == "__main__":
    pytest.main([__file__])
