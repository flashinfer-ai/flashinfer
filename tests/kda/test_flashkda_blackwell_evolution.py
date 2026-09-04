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
from types import SimpleNamespace

import pytest
import torch

from flashinfer.jit.flash_kda_evolution import (
    FLASH_KDA_EVOLUTION_VARIANTS,
    _module_ident,
    gen_flash_kda_evolution_module,
)
from flashinfer.kda_evolution import (
    PreparedFlashKDAEvolution,
    _EVOLUTION_WINNER_SHAPES,
    _route,
    _use_evolution_route,
    _uses_production_general,
    prepare_flash_kda_evolution,
)


_BLACKWELL_ROUTE_CONFIGS = (
    pytest.param((10, 0), 148, id="b200"),
    pytest.param((10, 3), 148, id="b300"),
    pytest.param((10, 0), 152, id="gb200"),
    pytest.param((10, 3), 152, id="gb300"),
)


@pytest.mark.parametrize(
    ("num_heads", "seq_lens", "packed", "expected"),
    (
        (96, (8192,), False, True),
        (96, (1300, 547, 2048, 963, 271, 3063), True, True),
        (96, (1024,) * 8, True, True),
        (64, (8192,), False, True),
        (64, (1300, 547, 2048, 963, 271, 3063), True, True),
        (64, (1024,) * 8, True, True),
        (32, (8192,), False, False),
        (32, (1300, 547, 2048, 963, 271, 3063), True, True),
        (96, (1024,) * 16, True, True),
        (96, (1024,) * 32, True, True),
        (96, (1024,) * 64, True, True),
        (96, (1024,) * 128, True, True),
        (96, (1024,) * 256, True, True),
        (96, (64, 128, 256), True, True),
        (96, (17, 33, 65), True, True),
        (16, (16384,), False, False),
        (16, (32768,), False, False),
        (16, (65536,), False, False),
        (8, (65536,), False, False),
        (4, (65536,), False, False),
        (4, tuple(range(1, 16)), True, False),
        (1, (1048576,), False, False),
        (96, (37,), False, True),
        (96, (97,), False, True),
        (96, (16,), True, False),
        (96, (16, 16), True, False),
        (1, (131072,), False, False),
        (1, (131072,), True, False),
        (1, (524288, 524288), True, False),
    ),
)
def test_flashkda_evolution_hybrid_route_manifest(
    num_heads, seq_lens, packed, expected
):
    assert _use_evolution_route(seq_lens, num_heads, not packed) is expected


@pytest.mark.parametrize(("compute_capability", "sm_count"), _BLACKWELL_ROUTE_CONFIGS)
def test_flashkda_evolution_n128_uses_production_general_only_at_148_sms(
    compute_capability, sm_count
):
    sequence_lengths = (1024,) * 128

    assert _use_evolution_route(sequence_lengths, 96, False)
    assert _uses_production_general(
        sequence_lengths,
        96,
        False,
        compute_capability=compute_capability,
        sm_count=sm_count,
        use_initial_state=True,
        store_final_state=True,
    ) is (sm_count == 148)


@pytest.mark.parametrize(("compute_capability", "sm_count"), _BLACKWELL_ROUTE_CONFIGS)
def test_flashkda_evolution_irregular_route_stays_generated(
    compute_capability, sm_count
):
    sequence_lengths = (17, 33, 65)

    assert _use_evolution_route(sequence_lengths, 96, False)
    assert not _uses_production_general(
        sequence_lengths,
        96,
        False,
        compute_capability=compute_capability,
        sm_count=sm_count,
        use_initial_state=True,
        store_final_state=True,
    )


@pytest.mark.parametrize(("compute_capability", "sm_count"), _BLACKWELL_ROUTE_CONFIGS)
def test_flashkda_evolution_fixed_h64_keeps_independent_value_split(
    compute_capability, sm_count
):
    sequence_lengths = (8192,)

    assert _use_evolution_route(sequence_lengths, 64, True)
    assert not _uses_production_general(
        sequence_lengths,
        64,
        True,
        compute_capability=compute_capability,
        sm_count=sm_count,
        use_initial_state=True,
        store_final_state=True,
    )


@pytest.mark.parametrize(("compute_capability", "sm_count"), _BLACKWELL_ROUTE_CONFIGS)
@pytest.mark.parametrize(
    ("fixed_layout", "num_heads", "sequence_lengths"),
    tuple(
        sorted(
            _EVOLUTION_WINNER_SHAPES
            - {
                (False, 96, (1024,) * 128),
                (False, 96, (17, 33, 65)),
                (True, 64, (8192,)),
            }
        )
    ),
)
def test_flashkda_evolution_other_frozen_routes_stay_generated(
    fixed_layout,
    num_heads,
    sequence_lengths,
    compute_capability,
    sm_count,
):
    assert not _uses_production_general(
        sequence_lengths,
        num_heads,
        fixed_layout,
        compute_capability=compute_capability,
        sm_count=sm_count,
        use_initial_state=True,
        store_final_state=True,
    )


def test_flashkda_evolution_profile_is_frozen():
    assert len(FLASH_KDA_EVOLUTION_VARIANTS) == 29
    assert (
        sum(
            metadata.has_tile_schedule
            for metadata in FLASH_KDA_EVOLUTION_VARIANTS.values()
        )
        == 7
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


def test_prepared_flashkda_evolution_rejects_cross_stream_launch(monkeypatch):
    class FakeModule:
        def __init__(self):
            self.calls = []

        def run(self, *args):
            self.calls.append(args)

    streams = iter(
        (
            SimpleNamespace(cuda_stream=17),
            SimpleNamespace(cuda_stream=17),
            SimpleNamespace(cuda_stream=23),
        )
    )
    seen_devices = []

    def current_stream(device):
        seen_devices.append(device)
        return next(streams)

    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    prepared = PreparedFlashKDAEvolution.__new__(PreparedFlashKDAEvolution)
    prepared.route = "evolution"
    prepared._device = torch.device("cuda:1")
    prepared._launch_stream_ptr = None
    prepared._prepare_descriptors = True
    prepared._args = ()
    prepared._launch_scalars = ()
    prepared.module = FakeModule()

    prepared.launch()
    prepared.launch()
    with pytest.raises(RuntimeError, match="stream used by the first launch"):
        prepared.launch()

    assert prepared.module.calls == [(1, 17), (0, 17)]
    assert seen_devices == [prepared._device] * 3


def test_flashkda_evolution_module_ident_covers_all_included_sources(tmp_path):
    metadata = FLASH_KDA_EVOLUTION_VARIANTS["m128_h96_p1_s166"]
    (tmp_path / f"{metadata.source_stem}.cu").write_bytes(b"body")
    (tmp_path / "cake_flashkda_blackwell_evolution_binding.cuh").write_bytes(b"binding")
    common = tmp_path / "flashkda_binding_common.cuh"
    common.write_bytes(b"common-v1")
    first = _module_ident(tmp_path, metadata)
    common.write_bytes(b"common-v2")
    assert _module_ident(tmp_path, metadata) != first


@pytest.mark.parametrize(
    ("num_heads", "multiprocessor_count", "expected_variant", "expected_stride"),
    (
        (64, 148, "m128_h64_p1_s126", 126),
        (96, 148, "m128_h96_p1_s173", 173),
        (64, 152, "m128_h64_p1_s114", 114),
        (96, 152, "m128_h96_p1_s166", 166),
    ),
)
def test_flashkda_evolution_routes_persistent_scalar_by_sm_count(
    monkeypatch,
    num_heads,
    multiprocessor_count,
    expected_variant,
    expected_stride,
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=multiprocessor_count),
    )
    route = _route(
        (1300, 547, 2048, 963, 271, 3063),
        num_heads,
        False,
        torch.device("cpu"),
    )

    assert route.variant == expected_variant
    assert route.grid_x == multiprocessor_count
    assert route.tile_schedule.numel() == multiprocessor_count * expected_stride
    assert int(route.tile_schedule_counts.max()) == expected_stride


@pytest.mark.parametrize("target", ["sm100a", "sm100f"])
@pytest.mark.parametrize("variant", FLASH_KDA_EVOLUTION_VARIANTS)
def test_flashkda_evolution_jit_spec_has_one_generated_binding(variant, target):
    spec = gen_flash_kda_evolution_module(variant, target)
    assert spec.name.endswith(target)
    assert len(spec.sources) == 1
    assert spec.sources[0].name == "binding.cu"
    assert "--use_fast_math" in spec.extra_cuda_cflags
    assert "--ptxas-options=-O1" not in spec.extra_cuda_cflags


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("seq_lens", "packed", "seed", "expected_variant", "alias_state"),
    (
        ((8192,), False, 20260826, "vtile_f1_t8192_h96_p1_s96", False),
        (
            (1300, 547, 2048, 963, 271, 3063),
            True,
            10001,
            "persistent-h96-mixed",
            False,
        ),
        ((17, 33, 65), True, 11016, "m128_h96_p0_s1", False),
        ((17, 33, 65), True, 11016, "m128_h96_p0_s1", True),
        ((16,), False, 11017, "cake_dispatcher", False),
        ((16,), True, 11018, "cake_dispatcher", False),
    ),
    ids=(
        "fixed-8192",
        "mixed",
        "irregular-evolution",
        "irregular-evolution-alias",
        "fixed-t16-cake",
        "packed-n1-t16-cake",
    ),
)
def test_flashkda_evolution_h96_matches_public_backend(
    seq_lens, packed, seed, expected_variant, alias_state
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
        torch.tensor(offsets, dtype=torch.int64, device=device) if packed else None
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
    actual_state = initial if alias_state else torch.empty_like(initial)
    actual_initial_before = initial.clone()
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
    if expected_variant == "persistent-h96-mixed":
        multiprocessor_count = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
        expected_variant = {
            148: "m128_h96_p1_s173",
            152: "m128_h96_p1_s166",
        }[multiprocessor_count]
    assert prepared.variant == expected_variant
    assert prepared.target == _select_flash_kda_prefill_target(device)
    assert prepared.route == (
        "cake" if expected_variant == "cake_dispatcher" else "evolution"
    )
    prepared.launch()
    torch.cuda.synchronize(device)

    if not alias_state:
        torch.testing.assert_close(initial, actual_initial_before, atol=0, rtol=0)
    torch.testing.assert_close(actual_out, expected_out, atol=1e-2, rtol=1e-2)
    assert expected_state is not None
    torch.testing.assert_close(actual_state, expected_state, atol=1e-2, rtol=1e-2)
