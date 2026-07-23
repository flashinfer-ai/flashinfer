"""SM103-specific correctness tests for CUTLASS NVFP4 GEMM."""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, autotune, mm_fp4, nvfp4_quantize
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability


# The 63 SM103 tactics contain 36 generic K128/K256 entries followed by 27
# native K768 entries. The tests below cover both families without forcing a tactic.
_NATIVE_K768_TACTICS = range(36, 63)
_GENERIC_K128_K256_TACTICS = range(36)


def _skip_if_not_sm103() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM103 GPU")
    if get_compute_capability(torch.device("cuda")) != (10, 3):
        pytest.skip("Requires an SM103 GPU")


def test_mm_fp4_sm103_cutlass_epilogue() -> None:
    """The public API must produce accurate output through SM103 CUTLASS."""

    _skip_if_not_sm103()

    # This compact shape selected native K768 tactic 51 in 20/20 independent
    # public-API autotuning runs across four B300 GPUs. No tactic is forced.
    m, n, k = 512, 6144, 6144
    torch.manual_seed(103768)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(
        a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    base_alpha = torch.reciprocal(a_global_sf * b_global_sf).float()
    reference = torch.mm(a, b.T).float()
    tuner = AutoTuner.get()
    tuner.clear_cache()
    selected_tactic_checked = False

    for out_dtype in (torch.bfloat16, torch.float16):
        unit_alpha_norm = None
        for alpha_factor in (1.0, 0.73125, -0.34375):
            alpha = base_alpha * alpha_factor
            output = torch.empty((m, n), device="cuda", dtype=out_dtype)

            # The optimized epilogue requires a 256-bit-aligned output.
            assert output.data_ptr() % 32 == 0
            assert output.stride(0) * output.element_size() % 32 == 0

            # A single exact bucket makes the cache entry for this invocation
            # unambiguous and avoids profiling unrelated M buckets in CI.
            with autotune(True, tuning_buckets=(m,)):
                result = mm_fp4(
                    a_fp4,
                    b_fp4.T,
                    a_sf,
                    b_sf.T,
                    alpha=alpha,
                    out_dtype=out_dtype,
                    out=output,
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend="cutlass",
                    use_nvfp4=True,
                )

            if not selected_tactic_checked:
                matching_entries = [
                    (key, runner_id, tactic)
                    for key, (runner_id, tactic, _) in tuner.profiling_cache.items()
                    if key.custom_op == "fp4_gemm" and key.nearest_profile[0][0] == m
                ]
                assert len(matching_entries) == 1, (
                    "expected one fp4_gemm autotuner entry for the exact M bucket, "
                    f"got {matching_entries}"
                )
                key, runner_id, tactic = matching_entries[0]
                assert key.runner_class_name == "CutlassFp4GemmRunner"
                assert runner_id == 0
                assert tactic in _NATIVE_K768_TACTICS, (
                    f"autotuner selected generic tactic {tactic}; expected a native "
                    "SM103 K768 tactic"
                )
                selected_tactic_checked = True

            assert result.data_ptr() == output.data_ptr()
            assert torch.isfinite(output).all(), (
                f"non-finite output for out_dtype={out_dtype}, "
                f"alpha_factor={alpha_factor}"
            )

            cosine = F.cosine_similarity(
                (reference * alpha_factor).reshape(-1),
                output.float().reshape(-1),
                dim=0,
            )
            assert cosine > 0.98, (
                f"cosine={cosine.item():.6f} for out_dtype={out_dtype}, "
                f"alpha_factor={alpha_factor}"
            )

            output_norm = output.float().norm().item()
            if alpha_factor == 1.0:
                unit_alpha_norm = output_norm
            else:
                assert unit_alpha_norm is not None
                observed_scale = output_norm / unit_alpha_norm
                assert observed_scale == pytest.approx(
                    abs(alpha_factor), rel=0.01, abs=0.005
                )


def test_mm_fp4_sm103_generic_alignment_dispatch() -> None:
    """K128/K256 must dispatch Store256 or TMA according to output alignment."""

    _skip_if_not_sm103()

    m, n, k = 512, 512, 512
    torch.manual_seed(103128256)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
    a_fp4, a_sf = nvfp4_quantize(
        a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_sf = nvfp4_quantize(
        b, b_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    base_alpha = torch.reciprocal(a_global_sf * b_global_sf).float()
    reference = torch.mm(a, b.T).float()
    tuner = AutoTuner.get()

    def make_misaligned_output(dtype: torch.dtype):
        guard_elements = 16
        storage = torch.full(
            (m * n + 2 * guard_elements,),
            -1234.0,
            device="cuda",
            dtype=dtype,
        )
        offset = ((16 - storage.data_ptr() % 32) % 32) // storage.element_size()
        if offset < guard_elements:
            offset += 32 // storage.element_size()
        output = storage[offset : offset + m * n].view(m, n)
        assert output.is_contiguous()
        assert output.data_ptr() % 32 == 16
        return storage, output, offset

    def invoke(output: torch.Tensor, alpha: torch.Tensor):
        with autotune(True, tuning_buckets=(m,)):
            result = mm_fp4(
                a_fp4,
                b_fp4.T,
                a_sf,
                b_sf.T,
                alpha=alpha,
                out_dtype=output.dtype,
                out=output,
                block_size=16,
                use_8x4_sf_layout=False,
                backend="cutlass",
                use_nvfp4=True,
            )
        assert result.data_ptr() == output.data_ptr()
        return result

    for out_dtype in (torch.bfloat16, torch.float16):
        # Check both cache transitions because output pointer alignment is not part of the
        # autotuner cache key: aligned -> misaligned and misaligned -> aligned.
        for first_alignment in ("aligned", "misaligned"):
            tuner.clear_cache()
            selected_tactic_checked = False
            for alpha_factor in (1.0, 0.73125, -0.34375):
                aligned = torch.empty((m, n), device="cuda", dtype=out_dtype)
                storage, misaligned, offset = make_misaligned_output(out_dtype)
                assert aligned.data_ptr() % 32 == 0
                alpha = base_alpha * alpha_factor

                if first_alignment == "aligned":
                    aligned_result = invoke(aligned, alpha)
                    misaligned_result = invoke(misaligned, alpha)
                else:
                    misaligned_result = invoke(misaligned, alpha)
                    aligned_result = invoke(aligned, alpha)

                if not selected_tactic_checked:
                    matching_entries = [
                        (key, runner_id, tactic)
                        for key, (runner_id, tactic, _) in tuner.profiling_cache.items()
                        if key.custom_op == "fp4_gemm"
                        and key.nearest_profile[0][0] == m
                    ]
                    assert len(matching_entries) == 1
                    key, runner_id, tactic = matching_entries[0]
                    assert key.runner_class_name == "CutlassFp4GemmRunner"
                    assert runner_id == 0
                    if first_alignment == "aligned":
                        assert tactic in _GENERIC_K128_K256_TACTICS
                    selected_tactic_checked = True

                assert torch.equal(aligned_result, misaligned_result)
                for result in (aligned_result, misaligned_result):
                    cosine = F.cosine_similarity(
                        (reference * alpha_factor).reshape(-1),
                        result.float().reshape(-1),
                        dim=0,
                    )
                    assert cosine > 0.98

                sentinel = torch.tensor(
                    -1234.0, device=storage.device, dtype=storage.dtype
                )
                assert torch.all(storage[:offset] == sentinel)
                assert torch.all(storage[offset + misaligned.numel() :] == sentinel)

    non_contiguous = torch.empty((n, m), device="cuda", dtype=torch.bfloat16).T
    assert non_contiguous.shape == (m, n)
    assert not non_contiguous.is_contiguous()
    with pytest.raises(ValueError, match="requires a contiguous output tensor"):
        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_sf,
            b_sf.T,
            alpha=base_alpha,
            out_dtype=non_contiguous.dtype,
            out=non_contiguous,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )
