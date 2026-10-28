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

The KCP registry (``kcp_jit``) of the Cake causal convolution training package: the records against the frozen
kernel ABIs and the identity rule, the routes and the selection by channel count, and the ``build.json`` entries of
the KCP programs.  No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import cake_package

cake_build = cake_package("cake_causal_conv1d_train", "cake_build")
cake_jit = cake_package("cake_causal_conv1d_train", "cake_jit")
kcp_jit = cake_package("cake_causal_conv1d_train", "kcp_jit")

PACKAGE_DIR = Path(cake_build.__file__).parent
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
_W = "sm100_sm103_sm107"
FWD = f"tart.causal_conv1d_kcp_fwd_bf16_silu.{_W}.causal_conv1d_kcp_fwd"
BWD = f"tart.causal_conv1d_kcp_bwd_bf16_silu.{_W}.causal_conv1d_kcp_bwd"
HEAD = f"tart.causal_conv1d_kcp_bwd_bf16_silu.{_W}.causal_conv1d_kcp_bwd_head"

FWD_PLAN = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "y"], ["buffer", "hist"], ["buffer", "hist_row_off"],
    ["parameter", "S"], ["parameter", "C"], ["parameter", "num_docs"], ["parameter", "search_steps"],
    ["parameter", "ct_vec"], ["parameter", "n_sub"], ["parameter", "n_ch_tiles"], *GRID,
]  # fmt: skip
BWD_SCALARS = [
    ["parameter", "S"], ["parameter", "C"], ["parameter", "num_docs"], ["parameter", "search_steps"],
    ["parameter", "ct_vec"], ["parameter", "n_sub"], ["parameter", "n_ch_tiles"],
]  # fmt: skip
BWD_BUFFERS = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "dy"], ["buffer", "dx"], ["buffer", "dw"],
    ["buffer", "partial"], ["buffer", "hist"], ["buffer", "hist_row_off"], ["buffer", "dpre_tail"],
    ["buffer", "tail_row_off"],
]  # fmt: skip
BWD_PLAN = [*BWD_BUFFERS, *BWD_SCALARS, *GRID]
BWD_TMA_PLAN = [
    *BWD_BUFFERS,
    *BWD_SCALARS,
    ["buffer", "x_map"],
    ["buffer", "dy_map"],
    *GRID,
]
HEAD_PLAN = [
    ["buffer", "x"], ["buffer", "w"], ["buffer", "cu"], ["buffer", "dy"], ["buffer", "hist"], ["buffer", "hist_row_off"],
    ["buffer", "dpre_head"], ["buffer", "head_row_off"], ["parameter", "S"], ["parameter", "C"],
    ["parameter", "num_docs"], *GRID,
]  # fmt: skip

# program name -> (module, build keywords, arches, role, plan, ctas_per_sm)
PROGRAMS = {
    "cake_causal_conv1d_train_675940a7a4fc0dba4b53": (
        FWD,
        {},
        ARCHES,
        "kcp_fwd",
        FWD_PLAN,
        4,
    ),
    "cake_causal_conv1d_train_5288d132a5cc927ae729": (
        BWD,
        {"vec": 8, "copy": "tma2d"},
        ("sm_100a", "sm_103a"),
        "kcp_bwd",
        BWD_TMA_PLAN,
        3,
    ),
    "cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039": (
        BWD,
        {"vec": 8, "min_blocks": 2},
        ("sm_107a",),
        "kcp_bwd",
        BWD_PLAN,
        2,
    ),
    "cake_causal_conv1d_train_7485529a1755e5bd3036": (
        BWD,
        {"vec": 8, "copy": "tma2d", "stage_tokens": 8, "min_blocks": 2},
        ARCHES,
        "kcp_bwd",
        BWD_TMA_PLAN,
        2,
    ),
    "cake_causal_conv1d_train_17cd42fb05079bbbf840": (
        HEAD,
        {},
        ARCHES,
        "kcp_bwd_head",
        HEAD_PLAN,
        5,
    ),
}
BIG_C_MIN = {"sm_100a": 512, "sm_103a": 512, "sm_107a": 1024}


def _identity(module: str, kwargs: dict) -> str:
    identity = {"build": {key: kwargs[key] for key in sorted(kwargs)}, "module": module}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"cake_causal_conv1d_train_{digest[:20]}"


@pytest.mark.cpu_only
def test_program_names_follow_the_identity_rule() -> None:
    assert set(kcp_jit.MODULES) == set(PROGRAMS)
    for name, (module, kwargs, *_rest) in PROGRAMS.items():
        assert _identity(module, kwargs) == name, name
    assert not set(kcp_jit.MODULES) & set(cake_jit.MODULES), (
        "the KCP programs are new builds, not the zero-history ones"
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_record_matches_the_frozen_abi(name: str) -> None:
    _module, _kwargs, arches, role, plan, ctas_per_sm = PROGRAMS[name]
    record = kcp_jit.MODULES[name]
    assert record["library"] == record["kernel"] == name
    assert record["kernel_symbol"] == f"kernel_{name}"
    assert record["arches"] == list(arches)
    assert record["role"] == role
    assert record["arg_plan"] == plan
    assert record["ffi_entry"] == "run"
    assert record["grid_rule"]["kind"] == "co_resident"
    assert record["grid_rule"]["ctas_per_sm"] == ctas_per_sm
    assert record["grid_rule"]["geometry"] == {"threads": 128, "vec": 8}
    assert record["grid_rule"]["cooperative"] is (role == "kcp_bwd")
    assert record["launch"]["cooperative"] is (role == "kcp_bwd")
    assert (
        record["launch"]["block"] == (128, 1, 1)
        and record["launch"]["use_pdl"] is False
    )
    kept = {parameter for kind, parameter in record["arg_plan"] if kind != "grid"}
    assert set(record["logical"]) == kept
    vocabulary = set(kcp_jit.LOGICAL_TENSORS) | set(kcp_jit.LOGICAL_SCALARS)
    assert set(record["logical"].values()) <= vocabulary
    if role == "kcp_bwd":
        assert record["workspace_rule"] == {
            "kind": "partial_rows",
            "taps": kcp_jit.WIDTH,
        }
    else:
        assert record["workspace_rule"] is None


@pytest.mark.cpu_only
def test_routes_and_selection() -> None:
    assert kcp_jit.KINDS == ("kcp_fwd", "kcp_bwd", "kcp_bwd_head")
    for arch in ARCHES:
        routes = kcp_jit.ROUTES[arch]
        assert set(routes) == set(kcp_jit.KINDS)
        assert routes["kcp_fwd"] == ["cake_causal_conv1d_train_675940a7a4fc0dba4b53"]
        assert routes["kcp_bwd_head"] == [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840"
        ]
        fallback = (
            "cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039"
            if arch == "sm_107a"
            else "cake_causal_conv1d_train_5288d132a5cc927ae729"
        )
        big = "cake_causal_conv1d_train_7485529a1755e5bd3036"
        assert routes["kcp_bwd"] == [fallback, big]
        assert kcp_jit.route_modules(arch, "kcp_bwd") == (fallback, big)
        assert kcp_jit.select_rule(kcp_jit.MODULES[big]) == {
            "channels": BIG_C_MIN,
            "kind": "channels_at_least",
        }
        assert kcp_jit.select_rule(kcp_jit.MODULES[fallback]) is None
        assert kcp_jit.selected_modules(arch, "kcp_bwd", C=BIG_C_MIN[arch] - 8) == (
            fallback,
        )
        assert kcp_jit.selected_modules(arch, "kcp_bwd", C=BIG_C_MIN[arch]) == (big,)
        assert kcp_jit.selected_modules(arch, "kcp_bwd", C=8192) == (big,)
        assert kcp_jit.selected_modules(arch, "kcp_fwd", C=256) == (
            routes["kcp_fwd"][0],
        )
        assert kcp_jit.workspace_record(arch)["role"] == "kcp_bwd"
    head = kcp_jit.MODULES["cake_causal_conv1d_train_17cd42fb05079bbbf840"]
    assert kcp_jit.head_grid(head, n_frag=3, C=4096) == (3 * 4, 1, 1)
    assert kcp_jit.head_grid(head, n_frag=0, C=4096) == (1, 1, 1)


@pytest.mark.cpu_only
def test_build_manifest_lists_the_kcp_programs() -> None:
    libraries = cake_build.libraries_by_name()
    for name, record in kcp_jit.MODULES.items():
        for arch in record["arches"]:
            entry = libraries[kcp_jit.library_name(name, arch)]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
            ]
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert {"--prec-div=true", "--prec-sqrt=true", "--ftz=false"} <= set(
                entry["nvcc_flags"]
            )
            for source in entry["sources"]:
                path = PACKAGE_DIR / source
                assert path.is_file(), source
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == entry["sha256"][source]
                ), source
        for arch in set(ARCHES) - set(record["arches"]):
            assert f"{name}_{arch}" not in libraries, (name, arch)
            with pytest.raises(RuntimeError, match="is not delivered for"):
                kcp_jit.library_name(name, arch)
