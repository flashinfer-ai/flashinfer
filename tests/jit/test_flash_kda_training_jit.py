# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import hashlib
import re

import pytest

from flashinfer.jit import flash_kda, flash_kda_training


def _kernel_symbols(source: str) -> set[str]:
    return set(re.findall(r"kernel_flashkda_[A-Za-z0-9_]+", source))


@pytest.mark.parametrize(
    ("target", "arch_flag"),
    [
        ("sm100a", "-gencode=arch=compute_100a,code=sm_100a"),
        ("sm103a", "-gencode=arch=compute_103a,code=sm_103a"),
    ],
)
def test_flash_kda_training_jit_spec(target, arch_flag):
    flash_kda_training.gen_flash_kda_training_module.cache_clear()
    uri = flash_kda_training.get_flash_kda_training_uri(target)
    spec = flash_kda_training.gen_flash_kda_training_module(target)
    assert re.fullmatch(rf"flash_kda_training_[0-9a-f]{{10}}_{target}", uri)
    assert spec.name == uri
    assert [source.name for source in spec.sources] == [
        "flashkda_training_forward_v483_binding.cu",
        "flashkda_training_paired_binding.cu",
        "flashkda_training_fallback_binding.cu",
        "flashkda_training_c16.cu",
        "flashkda_training_aux.cu",
        "flashkda_training_final_state.cu",
        f"training_fallback_pointer_{target.replace('sm', 'sm_', 1)}.cu",
        f"training_grouped_row_wg8_pointer_{target.replace('sm', 'sm_', 1)}.cu",
    ]
    assert all(source.is_file() for source in spec.sources)
    common = flash_kda_training._get_csrc_dir() / "flashkda_binding_common.cuh"
    source_digest = hashlib.sha256(
        b"\0".join(source.read_bytes() for source in (*spec.sources, common))
    ).hexdigest()[:10]
    assert uri == f"flash_kda_training_{source_digest}_{target}"
    assert arch_flag in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    (
        legacy_binding,
        paired_binding,
        fallback_binding,
        c16,
        auxiliary,
        final_state,
        fallback,
        grouped_row,
    ) = (source.read_text() for source in spec.sources)
    assert '#include "flashkda_training_forward_v483.cu"' not in legacy_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_forward" in legacy_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_forward" in paired_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_backward" in paired_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_c32_forward" in fallback_binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_training_c32_backward" in fallback_binding
    assert _kernel_symbols(c16) == {
        "kernel_flashkda_forward_checkpoint_c16",
        "kernel_flashkda_backward_persistent_c16",
    }
    assert _kernel_symbols(auxiliary) == {
        "kernel_flashkda_refine_forgetting_horizons",
        "kernel_flashkda_backward_param_reduce_c16_partial",
        "kernel_flashkda_grouped_qk_expand",
        "kernel_flashkda_grouped_qk_reduce",
    }
    assert _kernel_symbols(final_state) == {
        "kernel_flashkda_blackwell_prefill_fp32_state_initial"
    }
    assert _kernel_symbols(fallback) == {
        "kernel_flashkda_backward_preprocess",
        "kernel_flashkda_forward_checkpoint_rows_f32",
        "kernel_flashkda_backward_reverse_rows",
        "kernel_flashkda_backward_finalize_tokens",
        "kernel_flashkda_backward_gate_reduce_split",
        "kernel_flashkda_bf16_fused_m128_unsplit",
        "kernel_flashkda_bf16_fused_m128",
        "kernel_flashkda_backward_state_checkpoint_fallback_c32",
        "kernel_flashkda_backward_boundary_c32_tcgen_m64",
        "kernel_flashkda_backward_boundary_c32_tcgen",
        "kernel_flashkda_backward_local_c32_tcgen",
        "kernel_flashkda_backward_map_finalize_c32",
    }
    assert _kernel_symbols(grouped_row) == {
        "kernel_flashkda_backward_preprocess_bf16_norm",
        "kernel_flashkda_forward_checkpoint_rows_bf16_wg4",
        "kernel_flashkda_backward_reverse_wg8",
        "kernel_flashkda_backward_finalize_tokens_bf16_norm",
    }
    assert "kernel_flashkda_bf16_fused_m128_unsplit" in fallback_binding
    assert "run_training_grouped_row_forward" in fallback_binding
    assert "run_training_grouped_row_backward" in fallback_binding
    assert "materialize_public_forward" in fallback_binding
    assert "kGroupedRowReverseSmemBytes = 12416" in fallback_binding
    assert "use_split_work_items" in fallback_binding
    assert "seq_order" in fallback_binding
    assert "#define SPLIT_WORK_ITEMS 0" in fallback
    assert "#define SPLIT_WORK_ITEMS 1" in fallback
    for workspace in (
        "dq_normalized",
        "dk_normalized",
        "dlog_decay",
        "dbeta_active",
    ):
        assert f"clear row {workspace}" in fallback_binding
    flash_kda_training.gen_flash_kda_training_module.cache_clear()


def test_training_forward_frozen_specialization_contract():
    csrc_dir = flash_kda_training._get_csrc_dir()
    c16 = (csrc_dir / "flashkda_training_c16.cu").read_text()
    final_state = (csrc_dir / "flashkda_training_final_state.cu").read_text()
    paired_binding = (csrc_dir / "flashkda_training_paired_binding.cu").read_text()
    assert "STORE_BETA_ACTIVE 1" in c16
    assert "G_INPUT_BF16 1" in c16
    assert "STORE_FINAL_STATE 1" in c16
    assert "#define validate_outputs 0" in c16
    assert "USE_INITIAL_STATE 1" in final_state
    assert "STORE_FINAL_STATE 1" in final_state
    assert "ENABLE_CHECKPOINTS 0" in final_state
    assert "CastFinalState" not in paired_binding
    assert "final_output_scratch" in paired_binding
    assert "dl_float32" in paired_binding


def test_paired_backward_binding_has_no_forward_recompute_symbols():
    """Every route consumes context produced by the paired forward."""

    paired_binding = (
        flash_kda_training._get_csrc_dir() / "flashkda_training_paired_binding.cu"
    ).read_text()
    backward = paired_binding.split("void RunTrainingBackward(", 1)[1]
    backward = backward.split("}  // namespace flash_kda_training_paired", 1)[0]
    assert "run_training_forward" not in backward
    assert not re.search(r"kernel_flashkda_[A-Za-z0-9_]*forward", backward)
    assert "RunLow(" not in backward
    assert "RunHigh(" not in backward

    fallback_binding = (
        flash_kda_training._get_csrc_dir() / "flashkda_training_fallback_binding.cu"
    ).read_text()
    row_backward = fallback_binding.split("void RunTrainingRowBackward(", 1)[1]
    row_backward = row_backward.split("void RunTrainingGroupedRowForward(", 1)[0]
    grouped_row_backward = fallback_binding.split(
        "void RunTrainingGroupedRowBackward(", 1
    )[1]
    grouped_row_backward = grouped_row_backward.split("void RunTrainingC32Forward(", 1)[
        0
    ]
    c32_backward = fallback_binding.split("void RunTrainingC32Backward(", 1)[1]
    c32_backward = c32_backward.split("}  // namespace flash_kda_training_fallback", 1)[
        0
    ]
    for fallback_backward in (row_backward, grouped_row_backward, c32_backward):
        assert "LaunchAccurateForward" not in fallback_backward
        assert "run_training_forward" not in fallback_backward


@pytest.mark.parametrize("target", ["sm100a", "sm100f"])
def test_checkpoint_n16_prefill_jit_spec(target):
    flash_kda.gen_flash_kda_module.cache_clear()
    uri = flash_kda.get_flash_kda_uri("m128_n16_checkpoint", target)
    spec = flash_kda.gen_flash_kda_m128_n16_checkpoint_module(target)
    assert re.fullmatch(
        rf"flash_kda_bf16_m128_n16_checkpoint_[0-9a-f]{{10}}_{target}", uri
    )
    assert spec.name == uri
    assert len(spec.sources) == 1
    binding = spec.sources[0].read_text()
    body = (
        flash_kda._get_flash_kda_csrc_dir()
        / "flashkda_bf16_fused_m128_n16_checkpoint.cu"
    ).read_text()
    assert spec.sources[0].name == (
        "flashkda_bf16_fused_m128_n16_checkpoint_binding.cu"
    )
    assert (
        "checkpoint N16 descriptor_storage must provide at least 896 bytes" in binding
    )
    assert "PublishCheckpointMap" in binding
    assert "state_checkpoints_tma" in body
    assert "flashkda_checkpoint_generated_LoomTensorMap" in binding
    common = flash_kda._get_flash_kda_csrc_dir() / "flashkda_binding_common.cuh"
    source_digest = hashlib.sha256(
        b"\0".join(
            source.read_bytes()
            for source in (
                flash_kda._get_flash_kda_csrc_dir()
                / "flashkda_bf16_fused_m128_n16_checkpoint.cu",
                spec.sources[0],
                common,
            )
        )
    ).hexdigest()[:10]
    assert uri == f"flash_kda_bf16_m128_n16_checkpoint_{source_digest}_{target}"
    flash_kda.gen_flash_kda_module.cache_clear()
