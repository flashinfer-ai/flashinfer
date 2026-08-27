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
    load_flash_kda_evolution_module,
)


def test_flashkda_evolution_profile_is_frozen():
    metadata = FLASH_KDA_EVOLUTION_VARIANTS["vtile_f1_t8192_h96_p1_s96"]
    assert metadata.value_rows == 128
    assert not metadata.has_tile_schedule
    assert metadata.grid_x == 96
    assert metadata.kernel_symbol.endswith("vtile_f1_t8192_h96_p1_s96")


@pytest.mark.parametrize("target", ["sm100a", "sm100f"])
def test_flashkda_evolution_jit_spec_has_one_generated_binding(target):
    spec = gen_flash_kda_evolution_module("vtile_f1_t8192_h96_p1_s96", target)
    assert spec.name.endswith(target)
    assert len(spec.sources) == 1
    assert spec.sources[0].name == "binding.cu"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashkda_evolution_h96_fixed_8192_matches_public_backend():
    from flashinfer.kda import recurrent_kda
    from flashinfer.kda_prefill import _select_flash_kda_prefill_target

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("requires SM100 or SM103")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260826)
    shape = (1, 8192, 96, 128)
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
    beta = torch.sigmoid(
        torch.randn(
            shape[:-1], dtype=torch.bfloat16, device=device, generator=generator
        )
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
            (1, 96, 128, 128), dtype=torch.bfloat16, device=device, generator=generator
        )
        * 0.01
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
        scale=1.0 / math.sqrt(128),
        lower_bound=-5.0,
        backend="cake",
    )

    actual_out = torch.empty_like(q)
    actual_state = torch.empty_like(initial)
    cu_seqlens = torch.tensor([0, 8192], dtype=torch.int64, device=device)
    seq_order = torch.tensor([0], dtype=torch.int32, device=device)
    dummy_i32 = torch.empty((1,), dtype=torch.int32, device=device)
    descriptor_storage = torch.empty((6 * 128,), dtype=torch.uint8, device=device)
    target = _select_flash_kda_prefill_target(device)
    module = load_flash_kda_evolution_module("vtile_f1_t8192_h96_p1_s96", target)
    stream_ptr = int(torch.cuda.current_stream(device).cuda_stream)
    module.run(
        q,
        k,
        v,
        g,
        beta,
        beta,
        A_log,
        dt_bias,
        cu_seqlens,
        seq_order,
        dummy_i32,
        dummy_i32,
        initial,
        actual_out,
        actual_state,
        descriptor_storage,
        1,
        96,
        96,
        1,
        1,
        1.0 / math.sqrt(128),
        -5.0,
        stream_ptr,
    )
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    assert expected_state is not None
    torch.testing.assert_close(actual_state, expected_state, atol=1e-2, rtol=1e-2)
