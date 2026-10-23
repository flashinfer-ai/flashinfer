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

The KCP registry (``cake_kcp_jit``) and manifests of the Cake packed KDA training package: the records against the
frozen kernel ABIs, the program names against the identity rule, ``build.json`` (the shipped programs, never a
test-only one) and ``build.test.json`` (the test-only programs, loadable only under ``CAKE_KCP_TEST_KNOBS=1``).
No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import cake_package

cake_build = cake_package("cake_kda_train", "cake_build")
cake_jit = cake_package("cake_kda_train", "cake_jit")
kcp_jit = cake_package("cake_kda_train", "cake_kcp_jit")

PACKAGE_DIR = Path(cake_build.__file__).parent
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]

# The frozen kernel argument lists (name, kind) of the four KCP kernel packages.
FWDH_PLAN = [
    ["buffer", "w_tma"], ["buffer", "kg_tma"], ["buffer", "u"], ["buffer", "gk"], ["parameter", "num_heads"],
    ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"], ["buffer", "walk_seg"], ["buffer", "h0"],
    ["buffer", "h0_index"], ["parameter", "h0_hg"], ["parameter", "h0_n"], ["buffer", "out"], ["buffer", "out_index"],
    ["parameter", "out_hg"], ["parameter", "out_n"], ["parameter", "tile_stride"], ["buffer", "h_tape"],
    ["buffer", "v_new"], *GRID,
]  # fmt: skip
DHU_PLAN = [
    ["buffer", "kg_tma"], ["buffer", "qg_tma"], ["buffer", "w_tma"], ["buffer", "do_tma"], ["buffer", "dv1"],
    ["buffer", "gk"], ["parameter", "num_heads"], ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"],
    ["parameter", "scale"], ["buffer", "walk_seg"], ["buffer", "dht"], ["buffer", "dht_index"], ["parameter", "dht_hg"],
    ["parameter", "dht_n"], ["buffer", "out"], ["buffer", "out_index"], ["parameter", "out_hg"], ["parameter", "out_n"],
    ["parameter", "tile_stride"], ["buffer", "dh_tape"], ["buffer", "dv2"], *GRID,
]  # fmt: skip
TRANSFER_PLAN = [
    ["buffer", "w_tma"], ["buffer", "kg_tma"], ["buffer", "u"], ["buffer", "gk"], ["parameter", "num_heads"],
    ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"], ["buffer", "walk_seg"], ["buffer", "out"],
    ["buffer", "out_index"], ["parameter", "out_hg"], ["parameter", "out_n"], ["parameter", "tile_stride"],
    ["parameter", "init_mode"], ["parameter", "use_u"], ["parameter", "write_transposed"], *GRID,
]  # fmt: skip
MERGE_PLAN = [
    ["buffer", "m"], ["buffer", "c"], ["buffer", "x0"], ["buffer", "y"], ["buffer", "chain_start"],
    ["buffer", "elem_m"], ["buffer", "elem_c"], ["buffer", "elem_y"], ["buffer", "x0_index"], ["buffer", "y_final"],
    ["parameter", "ncols"], ["parameter", "transpose_m"], ["parameter", "write_states"], ["parameter", "cols_per_cta"],
    *GRID,
]  # fmt: skip
PLANS = {
    "kcp_fwdh_offset": FWDH_PLAN,
    "kcp_fwdh_tape": FWDH_PLAN,
    "kcp_dhu_offset": DHU_PLAN,
    "kcp_dhu_tape": DHU_PLAN,
    "kcp_transfer_walk": TRANSFER_PLAN,
    "kcp_affine_merge_c8": MERGE_PLAN,
    "kcp_affine_merge_c16": MERGE_PLAN,
    "kcp_transfer_walk_three_mma": TRANSFER_PLAN,
    "kcp_transfer_walk_plain_tf32": TRANSFER_PLAN,
    "kcp_transfer_walk_bf16x3": TRANSFER_PLAN,
    "kcp_transfer_walk_bf16x3_hw64": TRANSFER_PLAN,
}
# The identity rule: ``cake_kda_train_<20 hex of sha256("<tart module>\\n<sorted JSON of the build keywords>")>``.
_W = "sm100_sm103_sm107"
IDENTITIES = {
    "kcp_fwdh_offset": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_kcp_fwdh_v1",
        {"program": "offset"},
    ),
    "kcp_fwdh_tape": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_kcp_fwdh_v1",
        {"program": "tape"},
    ),
    "kcp_dhu_offset": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_kcp_dhu_v1",
        {"program": "offset"},
    ),
    "kcp_dhu_tape": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_kcp_dhu_v1",
        {"program": "tape"},
    ),
    "kcp_transfer_walk": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 0},
    ),
    "kcp_affine_merge_c8": (
        f"tart.kda_chunk_kcp_affine_merge.{_W}.kda_chunk_kcp_affine_merge_fp32_v1",
        {"cols_per_cta": 8},
    ),
    "kcp_affine_merge_c16": (
        f"tart.kda_chunk_kcp_affine_merge.{_W}.kda_chunk_kcp_affine_merge_fp32_v1",
        {"cols_per_cta": 16},
    ),
    "kcp_transfer_walk_three_mma": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 1},
    ),
    "kcp_transfer_walk_plain_tf32": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 2},
    ),
    "kcp_transfer_walk_bf16x3": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_bf16x3_v1",
        {"vh": 128},
    ),
    "kcp_transfer_walk_bf16x3_hw64": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_bf16x3_v1",
        {"vh": 64},
    ),
}


def _records() -> dict[str, dict]:
    return {**kcp_jit.MODULES, **kcp_jit.TEST_MODULES}


def test_the_tables_are_disjoint_and_complete() -> None:
    assert set(kcp_jit.MODULES) == set(PLANS) - set(kcp_jit.TEST_MODULES)
    assert set(kcp_jit.TEST_MODULES) == {
        "kcp_transfer_walk_three_mma",
        "kcp_transfer_walk_plain_tf32",
        "kcp_transfer_walk_bf16x3",
        "kcp_transfer_walk_bf16x3_hw64",
    }
    assert kcp_jit.ARCHES == ARCHES == tuple(cake_jit.ARCHES)
    assert kcp_jit.PROGRAMS == (
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_dhu_offset",
        "kcp_dhu_tape",
        "kcp_transfer_walk",
        "kcp_affine_merge",
    )
    assert kcp_jit.PRECISION_MODES == (0, 1, 2)
    assert kcp_jit.TRANSFER_WALKS == ("tf32_hilo", "bf16x3", "bf16x3_hw64")


@pytest.mark.parametrize("name", sorted(PLANS))
def test_record_matches_the_frozen_abi(name: str) -> None:
    record = _records()[name]
    assert record["arg_plan"] == PLANS[name], name
    assert record["arches"] == list(ARCHES)
    assert record["ffi_entry"] == "run"
    assert record["kernel_symbol"] == f"kernel_{record['library']}"
    assert (
        record["launch"]["cooperative"] is False
        and record["launch"]["use_pdl"] is False
    )
    assert record["launch"]["cluster"] == (1, 1, 1)
    assert record["launch"]["block"] == (
        (128, 1, 1) if record["program"] == "kcp_affine_merge" else (192, 1, 1)
    )
    assert isinstance(record["launch"]["dynamic_smem_bytes"], int)
    expected_name = (
        record["program"]
        if record["variant"] is None
        else f"{record['program']}_{record['variant']}"
    )
    assert name == expected_name
    kept = {parameter for kind, parameter in record["arg_plan"] if kind != "grid"}
    assert set(record["logical"]) == kept
    for kind, parameter in record["arg_plan"]:
        if kind == "grid":
            continue
        vocabulary = (
            kcp_jit.LOGICAL_TENSORS if kind == "buffer" else kcp_jit.LOGICAL_SCALARS
        )
        assert record["logical"][parameter] in vocabulary, (name, parameter)


def test_program_names_follow_the_identity_rule() -> None:
    for name, (module, kwargs) in IDENTITIES.items():
        digest = hashlib.sha256(
            f"{module}\n{json.dumps(kwargs, sort_keys=True)}".encode()
        ).hexdigest()
        assert _records()[name]["library"] == f"cake_kda_train_{digest[:20]}", name
    libraries = [record["library"] for record in _records().values()]
    assert len(set(libraries)) == len(libraries), "every record is its own program"
    assert not set(libraries) & set(cake_jit.MODULES), (
        "no KCP program shares a name with a single-rank one"
    )


def test_selection_and_grids() -> None:
    assert kcp_jit.select_record("kcp_affine_merge", ncols=128) == "kcp_affine_merge_c8"
    assert (
        kcp_jit.select_record("kcp_affine_merge", ncols=256) == "kcp_affine_merge_c16"
    )
    with pytest.raises(ValueError):
        kcp_jit.select_record("kcp_affine_merge", ncols=64)
    assert kcp_jit.select_record("kcp_fwdh_offset") == "kcp_fwdh_offset"
    assert kcp_jit.launch_grid("kcp_fwdh_offset", n_walk=3, num_heads=4) == (12, 1, 1)
    assert kcp_jit.launch_grid("kcp_transfer_walk_bf16x3", n_walk=3, num_heads=4) == (
        12,
        1,
        1,
    )
    assert kcp_jit.launch_grid(
        "kcp_transfer_walk_bf16x3_hw64", n_walk=3, num_heads=4
    ) == (24, 1, 1)
    assert kcp_jit.launch_grid("kcp_affine_merge_c16", n_chains=5, ncols=256) == (
        80,
        1,
        1,
    )
    assert kcp_jit.launch_grid("kcp_affine_merge_c8", n_chains=2, ncols=128) == (
        32,
        1,
        1,
    )
    with pytest.raises(ValueError):
        kcp_jit.launch_grid("kcp_fwdh_offset", n_walk=0, num_heads=4)


def test_test_record_selection_by_knobs() -> None:
    assert kcp_jit.test_record() is None
    assert kcp_jit.test_record(0, "tf32_hilo") is None
    assert kcp_jit.test_record(1)[0] == "kcp_transfer_walk_three_mma"
    assert kcp_jit.test_record(2)[0] == "kcp_transfer_walk_plain_tf32"
    assert kcp_jit.test_record(0, "bf16x3")[0] == "kcp_transfer_walk_bf16x3"
    assert kcp_jit.test_record(0, "bf16x3_hw64")[0] == "kcp_transfer_walk_bf16x3_hw64"
    with pytest.raises(ValueError, match="name two test-only programs"):
        kcp_jit.test_record(1, "bf16x3")
    with pytest.raises(ValueError):
        kcp_jit.test_record(3)
    with pytest.raises(ValueError):
        kcp_jit.test_record(0, "fp32")


def test_shipped_manifest_lists_the_kcp_programs_and_no_test_only_one() -> None:
    manifest = cake_build.manifest()
    libraries = cake_build.libraries_by_name()
    for name, record in kcp_jit.MODULES.items():
        for arch in ARCHES:
            library = kcp_jit.library_name(name, arch)
            assert library == f"{record['library']}_{arch}"
            entry = libraries[library]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert "--use_fast_math" in entry["nvcc_flags"]
            for source in entry["sources"]:
                path = PACKAGE_DIR / source
                assert path.is_file(), source
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == entry["sha256"][source]
                ), source
    names = {entry["name"] for entry in manifest["libraries"]}
    test_only = {
        f"{record['library']}_{arch}"
        for record in kcp_jit.TEST_MODULES.values()
        for arch in ARCHES
    }
    assert not names & test_only


def test_test_manifest_is_the_test_only_programs_under_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = PACKAGE_DIR / kcp_jit.TEST_MANIFEST
    if not path.is_file():
        pytest.skip(f"{kcp_jit.TEST_MANIFEST} is not shipped here")
    kcp_jit.test_manifest.cache_clear()
    monkeypatch.delenv(kcp_jit.TEST_KNOBS_ENV, raising=False)
    assert kcp_jit.test_knobs_enabled() is False
    with pytest.raises(RuntimeError, match=kcp_jit.TEST_KNOBS_ENV):
        kcp_jit.test_manifest()
    monkeypatch.setenv(kcp_jit.TEST_KNOBS_ENV, "1")
    kcp_jit.test_manifest.cache_clear()
    assert kcp_jit.test_knobs_enabled() is True
    document = kcp_jit.test_manifest()
    shipped = cake_build.manifest()
    assert document["schema"] == shipped["schema"] == "cake-build/1"
    assert (
        document["package"]
        == shipped["package"]
        == "flashinfer.experimental.cake_kda_train"
    )
    # the shipped manifest is re-rendered by every kernel update of the package, the test-only manifest only when its
    # programs change: the two generator records share their form, not their stamp
    assert (
        set(document["generator"])
        == set(shipped["generator"])
        == {"cake_revision", "cuda", "rendered_at"}
    )
    names = {entry["name"] for entry in document["libraries"]}
    assert names == {
        f"{record['library']}_{arch}"
        for record in kcp_jit.TEST_MODULES.values()
        for arch in ARCHES
    }
    for entry in document["libraries"]:
        assert "--use_fast_math" in entry["nvcc_flags"]
        for source in entry["sources"]:
            source_path = PACKAGE_DIR / source
            assert source_path.is_file(), source
            assert (
                hashlib.sha256(source_path.read_bytes()).hexdigest()
                == entry["sha256"][source]
            ), source
    for name in kcp_jit.TEST_MODULES:
        for arch in ARCHES:
            assert kcp_jit.test_library_name(name, arch) in names
    with pytest.raises(RuntimeError):
        kcp_jit.test_library_name("kcp_transfer_walk_three_mma", "sm_90a")
    kcp_jit.test_manifest.cache_clear()
