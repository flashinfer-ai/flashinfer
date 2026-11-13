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

# The MOSCA catalog (flashinfer/experimental/mosca/catalog.py) against the code:
# every kernel is registered under its name, every feature maps to a config
# switch and (for device features) to a macro that gates code in the kernel
# header and that the JIT emits, and the coverage check accepts exactly the
# validated cells. No GPU needed.

from __future__ import annotations

import dataclasses
import re
import warnings
from pathlib import Path

import pytest

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental import mosca
from flashinfer.experimental.mosca import catalog
from flashinfer.experimental.mosca.megamoe import (
    DSV4_PRO_DEP8_TILE_RULE,
    KERNEL_NAME,
    TOKEN_LAYOUTS,
    Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
)
from flashinfer.experimental.mosca.megamoe import backend as mosca_backend
from flashinfer.experimental.mosca.megamoe import jit as mosca_jit
from flashinfer.moe_ep import FleetParams
from flashinfer.moe_ep.core.kernel import registry

PACKAGE = Path(mosca.__file__).resolve().parent
KERNEL = catalog.SM100_FP8_FP4_MEGAMOE
B200 = (10, 0)


def make_config(**kw) -> Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig:
    kw.setdefault("activation_clamp", 10.0)
    return Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(intermediate_size=3072, top_k=6, **kw)


def dsv4_fleet(cap: int = 512) -> FleetParams:
    return FleetParams(num_experts=384, max_tokens_per_rank=cap, token_hidden_size=7168)


def test_every_catalog_kernel_is_registered_under_its_name() -> None:
    assert [k.name for k in catalog.KERNELS] == [KERNEL_NAME]
    for kernel in catalog.KERNELS:
        assert kernel.name in registry._available_mega_kernels()
        assert (PACKAGE / kernel.family).is_dir()
        assert catalog.get_kernel(kernel.name) is kernel
    with pytest.raises(KeyError):
        catalog.get_kernel("no_such_kernel")


def test_features_name_their_switch_and_files() -> None:
    fields = {
        f.name for f in dataclasses.fields(Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig)
    }
    for feature in KERNEL.features:
        switch_field = feature.switch.split("=", 1)[0]
        assert switch_field in fields, feature.name
        assert feature.result_preserving
        for rel in feature.where:
            if rel.startswith("csrc/") and not (PACKAGE / "csrc").is_dir():
                continue  # release wheel: no kernel sources
            assert (PACKAGE / rel).is_file(), rel


def test_device_feature_macros_gate_code_and_are_emitted_by_the_jit() -> None:
    header = PACKAGE / "csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh"
    if not header.is_file():
        pytest.skip("experimental kernel sources are not installed (release wheel)")
    text = header.read_text()
    device = [f for f in KERNEL.features if f.macro is not None]
    assert {f.macro for f in device} == set(mosca_jit.FEATURE_MACROS.values())
    for feature in device:
        assert re.search(rf"^\s*#\s*ifdef\s+{feature.macro}\b", text, re.M), (
            feature.macro
        )
    base = dict(
        num_max_tokens_per_rank=1920,
        hidden=7168,
        intermediate_hidden=3072,
        num_experts=384,
        num_topk=6,
        num_ranks=8,
        num_sms=148,
        block_m=16,
        block_n=128,
        block_k=128,
        store_block_m=8,
        sf_block_m=128,
        sf_block_n=128,
        gran_k=32,
        num_ring_tokens=19200,
        num_sf_ring_tokens=307200,
        swizzle_acts_mode=128,
        swizzle_weights_mode=128,
        num_stages=10,
        smem_size=219844,
        num_dispatch_threads=128,
        num_non_epilogue_threads=128,
        num_epilogue_threads=256,
        num_bytes_per_pull=7168,
        activation_clamp=10.0,
        fast_math=True,
        small_t=False,
        sparse_count=False,
        gather=False,
    )
    off = mosca_jit.render_config_header(mosca_jit.KernelParams(**base))
    assert not any(m in off for m in mosca_jit.FEATURE_MACROS.values())
    for switch, macro in mosca_jit.FEATURE_MACROS.items():
        on = mosca_jit.render_config_header(
            mosca_jit.KernelParams(**{**base, switch: True})
        )
        assert f"#define {macro}\n" in on


def test_validated_cells_are_consistent() -> None:
    names = {f.name for f in KERNEL.features}
    assert KERNEL.validated
    for cell in KERNEL.validated:
        assert set(cell.features) <= names
        assert cell.token_layout in TOKEN_LAYOUTS
        assert cell.correctness


@pytest.mark.parametrize(
    "kwargs, ep, cc, covered",
    [
        (dict(), 8, B200, True),
        (dict(token_layout="replicated"), 8, B200, True),
        (dict(token_layout="replicated"), 4, B200, True),
        (dict(sparse_count=True), 8, B200, True),
        # A tile rule keeps outputs, but no rule's speed is validated here.
        (dict(sparse_count=True, tile_rule=DSV4_PRO_DEP8_TILE_RULE), 8, B200, False),
        (dict(small_t=False), 1, B200, True),
        (dict(), 2, B200, False),  # EP2 was not validated
        (dict(), 8, (10, 3), False),  # B300 was not validated
        (dict(token_layout="replicated", sparse_count=True), 8, B200, False),
        (
            dict(token_layout="replicated", tile_rule=DSV4_PRO_DEP8_TILE_RULE),
            8,
            B200,
            False,
        ),
    ],
)
def test_coverage_check(kwargs, ep, cc, covered) -> None:
    mosca_backend._WARNED_OUTSIDE_COVERAGE.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = mosca_backend.warn_if_outside_validated_coverage(
            make_config(**kwargs), dsv4_fleet(), ep, cc
        )
        mosca_backend.warn_if_outside_validated_coverage(
            make_config(**kwargs), dsv4_fleet(), ep, cc
        )
    assert result is covered
    outside = [
        w
        for w in caught
        if issubclass(w.category, ExperimentalWarning)
        and "validated coverage" in str(w.message)
    ]
    assert len(outside) == (0 if covered else 1)  # once per configuration


def test_other_shapes_are_outside_coverage() -> None:
    fleet = FleetParams(
        num_experts=256, max_tokens_per_rank=512, token_hidden_size=7168
    )
    assert (
        KERNEL.find_cell(
            num_experts=fleet.num_experts,
            hidden=7168,
            intermediate=2048,
            top_k=8,
            ep_size=8,
            token_layout="partitioned",
            compute_capability=B200,
        )
        is None
    )


def test_enabled_features_follow_the_config() -> None:
    assert mosca_backend.enabled_features(make_config(small_t=False)) == ()
    assert mosca_backend.enabled_features(
        make_config(
            sparse_count=True,
            token_layout="replicated",
            tile_rule=DSV4_PRO_DEP8_TILE_RULE,
        )
    ) == ("small_t", "sparse_count", "in_kernel_all_gather", "tile_rule")
