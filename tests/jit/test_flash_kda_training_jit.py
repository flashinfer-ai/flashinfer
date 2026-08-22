# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import re

import pytest

from flashinfer.jit import flash_kda, flash_kda_training


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
    assert len(spec.sources) == 1
    assert spec.sources[0].name == "flashkda_training_forward_v483_binding.cu"
    assert spec.sources[0].is_file()
    assert arch_flag in spec.extra_cuda_cflags
    assert "-use_fast_math" in spec.extra_cuda_cflags
    binding = spec.sources[0].read_text()
    assert '#include "flashkda_training_forward_v483.cu"' in binding
    assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_forward" in binding
    flash_kda_training.gen_flash_kda_training_module.cache_clear()


def test_training_forward_frozen_specialization_contract():
    csrc_dir = flash_kda_training._get_csrc_dir()
    body = (csrc_dir / "flashkda_training_forward_v483.cu").read_text()
    binding = (csrc_dir / "flashkda_training_forward_v483_binding.cu").read_text()
    assert "kernel_flashkda_forward_checkpoint_c16" in body
    assert "beta_active_out" in body
    assert "final_state" in body
    assert "CastFinalState" in binding
    assert "dl_float32" in binding


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
    flash_kda.gen_flash_kda_module.cache_clear()
