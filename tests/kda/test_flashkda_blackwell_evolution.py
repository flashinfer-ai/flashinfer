"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch

from flashinfer.jit.flash_kda_evolution import (
    FLASH_KDA_EVOLUTION_VARIANTS,
    gen_flash_kda_evolution_module,
)
from flashinfer.kda_evolution import prepare_flash_kda_evolution


def test_flashkda_evolution_profile_is_frozen():
    assert len(FLASH_KDA_EVOLUTION_VARIANTS) == 26
    assert (
        sum(
            metadata.has_tile_schedule
            for metadata in FLASH_KDA_EVOLUTION_VARIANTS.values()
        )
        == 4
    )
    assert (
        sum(
            metadata.value_rows == 64
            for metadata in FLASH_KDA_EVOLUTION_VARIANTS.values()
        )
        == 1
    )
    metadata = FLASH_KDA_EVOLUTION_VARIANTS["vtile_f1_t8192_h96_p1_s96"]
    assert metadata.value_rows == 128
    assert not metadata.has_tile_schedule
    assert metadata.kernel_symbol.endswith("vtile_f1_t8192_h96_p1_s96")


@pytest.mark.parametrize("target", ["sm100a", "sm100f"])
@pytest.mark.parametrize("variant", FLASH_KDA_EVOLUTION_VARIANTS)
def test_flashkda_evolution_jit_spec_has_one_generated_binding(variant, target):
    spec = gen_flash_kda_evolution_module(variant, target)
    assert spec.name.endswith(target)
    assert len(spec.sources) == 1
    assert spec.sources[0].name == "binding.cu"
    assert "--ptxas-options=-O1" in spec.extra_cuda_cflags


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("seq_lens", "seed", "expected_variant"),
    (
        ((8192,), 20260826, "vtile_f1_t8192_h96_p1_s96"),
        ((1300, 547, 2048, 963, 271, 3063), 10001, "m128_h96_p0_s1"),
    ),
    ids=("fixed-8192", "mixed"),
)
def test_flashkda_evolution_h96_matches_public_backend(
    seq_lens, seed, expected_variant
):
    from flashinfer.kda import recurrent_kda
    from flashinfer.kda_prefill import _select_flash_kda_prefill_target

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100 or SM103")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (1, sum(seq_lens), 96, 128)
    q = (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
    )
    k = (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
    )
    v = (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
    )
    g = (
        torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)
        * 0.1
    )
    beta = torch.randn(
        shape[:-1], dtype=torch.bfloat16, device=device, generator=generator
    )
    A_log = (
        torch.randn((96,), dtype=torch.float32, device=device, generator=generator)
        * 0.1
    )
    dt_bias = (
        torch.randn((96, 128), dtype=torch.float32, device=device, generator=generator)
        * 0.1
    )
    initial = (
        torch.randn(
            (len(seq_lens), 96, 128, 128),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.01
    )
    offsets = [0]
    for seq_len in seq_lens:
        offsets.append(offsets[-1] + seq_len)
    cu_seqlens = (
        torch.tensor(offsets, dtype=torch.int64, device=device)
        if len(seq_lens) > 1
        else None
    )

    expected_initial = initial.clone()
    expected_out, expected_state = recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state=expected_initial,
        output_final_state=True,
        use_gate_in_kernel=True,
        scale=1.0 / math.sqrt(128),
        lower_bound=-5.0,
        beta_is_logit=True,
        backend="cake",
        cu_seqlens=cu_seqlens,
    )

    actual_out = torch.empty_like(q)
    actual_state = torch.empty_like(initial)
    prepared = prepare_flash_kda_evolution(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial,
        actual_out,
        actual_state,
        scale=1.0 / math.sqrt(128),
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
    )
    assert prepared.variant == expected_variant
    assert prepared.target == _select_flash_kda_prefill_target(device)
    prepared.launch()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    assert expected_state is not None
    torch.testing.assert_close(actual_state, expected_state, atol=1e-2, rtol=1e-2)
