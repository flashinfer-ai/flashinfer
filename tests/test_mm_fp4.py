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


# TODO: Consdier splitting this function up for the various backends
@pytest.mark.parametrize("m", [1, 48, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["trtllm", "cudnn", "cutlass"])
@pytest.mark.parametrize("use_128x4_sf_layout", [False, True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4"])
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    if backend == "trtllm" and res_dtype == torch.float16:
        pytest.skip("Skipping test for trtllm fp4 with float16")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if auto_tuning and backend == "cudnn":
        pytest.skip("Skipping test for cudnn fp4 with auto_tuning=True")
    if fp4_type == "mxfp4" and backend != "cudnn":
        pytest.skip("mx_fp4 is only supported for cudnn backend")

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    use_nvfp4 = fp4_type == "nvfp4"

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

    reference = torch.mm(input, mat2.T)

    alpha = 1.0 / (global_sf_input * global_sf_mat2)
    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    with autotune(auto_tuning):
        mm_fp4(
            input_fp4,
            mat2_fp4.T,
            input_inv_s,
            mat2_inv_s.T,
            alpha if use_nvfp4 else None,
            res_dtype,
            res,
            block_size=16 if use_nvfp4 else 32,
            use_8x4_sf_layout=not use_128x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
        )

    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.97


if __name__ == "__main__":
    pytest.main([__file__])
