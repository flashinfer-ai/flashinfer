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

Tests of the Cake Kimi-K3 AttnRes backend package (SM100 / SM103): the host policy
and its route tables, the registry and the generated sources, the launch plans and,
on a supported device, the numerical contract against the independent FP32 reference
(output within atol 8e-2 / rtol 3e-2, prefix and snapshot bank bit-exact).

This suite exercises the backend package alone: it names the package once, through
``tests.test_helpers.cake_package``, and imports nothing else from flashinfer, so a
build that vendors the package under another import path runs it unchanged after
pointing that table entry at its package.  The ``flashinfer.kimi_k3_attn_res`` facade
is covered by ``test_cake_kimi_k3_attn_res_api.py``.
"""

import inspect
import json
import pathlib
import re

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

cb = cake_package("cake_kimi_k3_attn_res", "cake_backend")
cake_build = cake_package("cake_kimi_k3_attn_res", "cake_build")
cake_jit = cake_package("cake_kimi_k3_attn_res", "cake_jit")
ARCHES = cb.ARCHES
HIDDEN_SIZE = cb.HIDDEN_SIZE
MAX_BLOCKS = cb.MAX_BLOCKS
NATIVE_SM_COUNTS = cb.NATIVE_SM_COUNTS
SUPPORTED_COMPUTE_CAPABILITIES = cb.SUPPORTED_COMPUTE_CAPABILITIES
common_path_eligible = cb.common_path_eligible
plan_route = cb.plan_route
prepare_kimi_k3_attn_res = cb.prepare_kimi_k3_attn_res
reference_kimi_k3_attn_res = cb.reference_kimi_k3_attn_res
KERNELS = cake_jit.KERNELS
MODULES = cake_jit.MODULES
ROUTES = cake_jit.ROUTES

# The Kimi-K3 AttnRes evaluation contract: output within (atol 8e-2, rtol 3e-2)
# of the independent FP32 reference; prefix / snapshot bank bit-exact.
ATOL = 8e-2
RTOL = 3e-2
# Reference physical configuration of the measured policy (B200 / B300) and every SM count the
# programs are qualified on (now including the 152-SM GB300 / GB200 parts).  All grids derive from
# the count: the native ports launch one CTA per SM on NATIVE_SM_COUNTS, the K = 0 path 2x / 3x
# the count at its promoted cells, the persistent programs min(M, SMs) (balanced 128 at M 256-512).
SM_COUNT = 148
SM_COUNTS = (148, 152)
assert SM_COUNTS == NATIVE_SM_COUNTS
#: POLICY_ROWS grid marker: the device's SM count.
NATIVE = "sms"
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
PRIMARY_K = (0, 1, 4, 8)

# Representative cells of the measured policy per architecture:
# (arch, M, K, pdl) -> (kind, schedule_id, grid_x)
POLICY_ROWS = [
    ("sm_100a", 1, 0, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 1),
    ("sm_100a", 16, 0, True, "small_m", "small_m_direct_cta256_regres_fp32x2", 16),
    ("sm_100a", 32, 0, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 32),
    ("sm_100a", 8, 1, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 8),
    ("sm_100a", 16, 4, True, "small_m", "small_m_cluster2_cta128_regres_fp32x2", 32),
    ("sm_100a", 256, 0, True, "small_m", "small_m_direct_cta256_regres_fp32x2", 256),
    ("sm_100a", 1024, 0, False, "k0_tma", "k0_tma_persistent_ws288_vec128_fp32x2", 444),
    ("sm_100a", 2048, 0, True, "k0_tma", "k0_tma_persistent_ws288_vec128_fp32x2", 148),
    ("sm_100a", 256, 8, False, "native", "native_k8_nc3_d2_ws288_grid148", NATIVE),
    ("sm_100a", 1, 5, False, "small_m", "small_m_cluster4_cta64_regres_fp32x2", 4),
    ("sm_100a", 64, 4, True, "small_m", "small_m_direct_cta256_regres_fp32x2", 64),
    ("sm_100a", 1, 7, False, "small_m", "small_m_cluster4_cta64_regres_fp32x2", 4),
    ("sm_100a", 1, 7, True, "small_m", "small_m_cluster4_cta64_regres_fp32x2", 4),
    ("sm_100a", 16, 8, True, "small_m", "small_m_cluster4_cta64_regres_fp32x2", 64),
    ("sm_100a", 16, 8, False, "small_m", "small_m_cluster4_cta64_regres_fp32x2", 64),
    ("sm_100a", 512, 1, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 512),
    (
        "sm_100a",
        1024,
        1,
        False,
        "persistent",
        "trtllm_persistent_ws288_nc2_d2_vec128_fp32x2_early_consume",
        148,
    ),
    (
        "sm_100a",
        4096,
        4,
        False,
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2_early_consume",
        148,
    ),
    (
        "sm_100a",
        4096,
        5,
        True,
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2",
        148,
    ),
    (
        "sm_100a",
        1024,
        4,
        False,
        "native",
        "native_m128_nc3_d2_ws288_grid148",
        NATIVE,
    ),
    ("sm_103a", 4, 0, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 4),
    ("sm_103a", 32, 0, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 32),
    ("sm_103a", 1, 3, True, "small_m", "small_m_direct_cta256_regres_fp32x2", 1),
    ("sm_103a", 16, 4, False, "small_m", "small_m_cluster2_cta128_regres_fp32x2", 32),
    (
        "sm_103a",
        1024,
        0,
        False,
        "k0_tma",
        "k0_tma_persistent_ws288_vec128_fp32x2",
        444,
    ),
    ("sm_103a", 1, 7, False, "small_m", "small_m_cluster2_cta128_regres_fp32x2", 2),
    ("sm_103a", 8, 8, True, "small_m", "small_m_cluster2_cta128_regres_fp32x2", 16),
    (
        "sm_103a",
        1024,
        4,
        False,
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2_early_consume",
        148,
    ),
    (
        "sm_103a",
        256,
        4,
        False,
        "native",
        "native_m128_nc3_d2_ws288_grid148",
        NATIVE,
    ),
    ("sm_103a", 256, 8, True, "native", "native_k8_nc3_d2_ws288_grid148", NATIVE),
    ("sm_103a", 1, 7, True, "small_m", "small_m_cluster2_cta128_regres_fp32x2", 2),
    (
        "sm_103a",
        4096,
        5,
        False,
        "persistent",
        "trtllm_persistent_ws288_nc4_d3_vec128_fp32x2_early_consume_relaxed_producer_wait",
        148,
    ),
    ("sm_103a", 256, 1, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 256),
    ("sm_103a", 512, 0, False, "small_m", "small_m_direct_cta256_regres_fp32x2", 512),
    ("sm_103a", 512, 1, True, "small_m", "small_m_direct_cta256_regres_fp32x2", 512),
    (
        "sm_103a",
        512,
        4,
        True,
        "persistent",
        "trtllm_persistent_ws288_nc3_d2_vec128_fp32x2_early_consume",
        128,
    ),
    (
        "sm_103a",
        16384,
        8,
        False,
        "persistent",
        "trtllm_persistent_ws288_nc5_d2_vec128_fp32x2_early_consume",
        148,
    ),
]


# Round 2 (the ``R2_*`` route tables): the POLICY_ROWS cells whose program depends on the SM count,
# (arch, M, K, sm_count) -> (kind, schedule_id, grid_x).  Empty after the step-3 acceptance table: the
# dense cluster-4 small-M cells of sm_103a (the only SM-count-dependent dense cells of round 2) were
# withdrawn, so every POLICY_ROWS cell takes the same program at 148 and 152 SMs.
_ROUND_2_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {}

# Round 5: the POLICY_ROWS cells whose program depends on the SM count after the round-5 route tables.
# sm_103a (4096, 5) takes the held nc3 depth-3 K = 5 program at 152 SMs (``R2_PERSISTENT_KEY_CELLS``)
# and keeps the early-consume nc4 program at 148 SMs.
_ROUND_5_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {
    ("sm_103a", 4096, 5, 152): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2",
        152,
    ),
}


# Round 6: the POLICY_ROWS cells the convergence-screen tables move (``R2_PERSISTENT_KEY_CELLS``):
# sm_100a (4096, 4) takes the held nc3 depth-3 K = 4 program; sm_103a (1024, 4) the held packed-add
# program at 148 SMs and the early packed-add fast-math program at 152 SMs; sm_103a (4096, 5) at 148
# SMs the held relaxed-producer-wait program; sm_103a (512, 4) at 152 SMs the early depth-3 program.
_ROUND_6_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {
    ("sm_100a", 4096, 4, 148): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2",
        148,
    ),
    ("sm_103a", 1024, 4, 148): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2_prefix_bf16_add_packed_inputs_prefix_shared_addr",
        148,
    ),
    ("sm_103a", 1024, 4, 152): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2_early_consume_prefix_bf16_add_packed_inputs_prefix_shared_addr",
        152,
    ),
    ("sm_103a", 4096, 5, 148): (
        "persistent",
        "trtllm_persistent_ws288_nc4_d3_vec128_fp32x2_relaxed_producer_wait",
        148,
    ),
    ("sm_103a", 512, 4, 152): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2_early_consume",
        128,
    ),
}


def _expected_at(sm_count: int, grid_x, schedule_id: str) -> tuple[int, str]:
    """The (grid, schedule id) a POLICY_ROWS cell takes on ``sm_count`` SMs: the native ports and the
    SM-count-multiple grids scale with the count (148 -> sm_count, 444 -> 3 x sm_count); the token-bound,
    balanced-128 and cluster grids do not."""
    if grid_x == NATIVE:
        return sm_count, schedule_id.replace("grid148", f"grid{sm_count}")
    if grid_x % SM_COUNT == 0:
        return grid_x // SM_COUNT * sm_count, schedule_id
    return grid_x, schedule_id


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch,M,K,pdl,kind,schedule_id,grid_x", POLICY_ROWS)
def test_plan_route_policy(arch, M, K, pdl, kind, schedule_id, grid_x, sm_count):
    grid_x, schedule_id = _expected_at(sm_count, grid_x, schedule_id)
    kind, schedule_id, grid_x = (_ROUND_2_ROWS | _ROUND_5_ROWS | _ROUND_6_ROWS).get(
        (arch, M, K, sm_count), (kind, schedule_id, grid_x)
    )
    plan = plan_route(arch, sm_count, M, K, pdl)
    assert plan.kind == kind
    # Round r5: --use_fast_math is a program axis of the small-M / persistent programs (schedule id
    # ``_fastmath``, kernel key ``_fm``); the rows pin the program apart from that axis.
    assert plan.schedule_id.removesuffix("_fastmath") == schedule_id
    assert plan.schedule_id.endswith("_fastmath") is ("_fm" in plan.kernel_key)
    # A round-6 exact cell carries its fast-math axis in its table key, not in the fast-math bands.
    if (
        kind in ("small_m", "persistent")
        and (arch, M, K, sm_count) not in _ROUND_6_ROWS
    ):
        assert plan.schedule_id.endswith("_fastmath") is cb._fast_math(
            arch, M, K, kind, sm_count
        )
    assert plan.grid_x == grid_x
    assert plan.use_pdl is pdl
    # PDL is a launch argument of every generated program, not a program axis.
    assert "pdl" not in plan.kernel_key
    assert f".pdl{int(pdl)}." in plan.route_id
    assert plan.route_id.startswith(plan.schedule_id + ".")


@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize(
    "M,K,early",
    (
        # K2: the early-release band M768..2048; held below, between and above.
        # Round 2 at 148 SMs: M768 / M1536 run the early register-carry programs, M2048 the held one
        # (``persistent:k2_nc4_d2_f011010000000000``; early again at 152 SMs), M3072 / M6144 the held
        # register-carry program; M4096 runs the held fast-math nc3 program without the fence.
        (767, 2, False),
        (768, 2, True),
        (1536, 2, True),
        (2048, 2, False),
        (2049, 2, False),
        (3072, 2, False),
        (4096, 2, False),
        (4097, 2, False),
        (6144, 2, False),
        # K4: the nc5 band M513..1023 + the pow2 cells {512, 2048, 4096}; held at 1024..2047 and above 4096.
        (512, 4, True),
        (513, 4, True),
        (768, 4, True),
        (1023, 4, True),
        # round 6: the held packed-add depth-3 program at 148 SMs
        (1024, 4, False),
        (1025, 4, False),
        (1536, 4, False),
        (2048, 4, True),
        (3072, 4, False),
        (4096, 4, True),
        (6144, 4, False),
    ),
)
def test_plan_route_sm103_k2_k4_consumed_release_bands(M, K, pdl, early):
    """Round r4 (K2 / K4 screening, Cake 9d069424c49): on sm_103a the early consumed-stage release
    of the persistent K2 program wins only at M768..2048 (1.3-2.6 % vs the held program, bit-identical)
    and loses from M12288 on; the K4 nc5 program releases early at M513..1023. The first persistent
    flag is the release policy; the band edges are the dispatcher / evaluator / FI mirror constants."""
    plan = plan_route("sm_103a", SM_COUNT, M, K, pdl)
    assert plan.kind == "persistent", plan
    # round r5: the fast-math axis (``_fm``) and the later suffixes follow the flag tuple
    flags = re.match(r"^persistent:k\d+_nc\d+_d\d+_f([01]+)", plan.kernel_key).group(1)
    assert (flags[0] == "1") is early, (plan.kernel_key, early)
    if (M, K) == (2048, 2):
        at_152 = plan_route("sm_103a", 152, M, K, pdl)
        assert at_152.kernel_key == "persistent:k2_nc4_d2_f111010000000000"


def _r2_cells(table_name):
    """(arch, sm_count, M, K, value) of every cell of a round-2 ``ROUTES`` table."""
    for arch, by_sm in ROUTES[table_name].items():
        for sm_count, pairs in by_sm.items():
            for cell in pairs:
                if isinstance(cell[0], (list, tuple)):
                    (M, K), value = cell
                else:
                    (M, K), value = cell, None
                yield arch, int(sm_count), int(M), int(K), value


def test_round_2_cells_take_their_table_program():
    """Every cell of the round-2 tables (``ROUTES["R2_*"]``) resolves to the program the table names,
    exactly (no registered-variant fallback), with the launch geometry of its kind."""
    seen = 0
    for arch, sm_count, M, K, value in _r2_cells("R2_PERSISTENT_KEY_CELLS"):
        for pdl in (False, True):
            key = value[int(pdl)] if isinstance(value, (list, tuple)) else value
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert plan.kind == "persistent" and plan.fallback_from is None, (plan, key)
            assert plan.kernel_key == key, (
                arch,
                sm_count,
                M,
                K,
                pdl,
                plan.kernel_key,
                key,
            )
            assert plan.threads == cb.PERSISTENT_THREADS and 1 <= plan.grid_x <= max(
                M, sm_count
            )
            flags = cb._PERSISTENT_KEY.match(key)["bits"]
            if flags[cb._PERSISTENT_FLAG_REGISTER_CARRY_ONLY] == "1":
                assert flags[cb._PERSISTENT_FLAG_PREFIX_BF16_ADD] == "1"
                # the fence-free twin (_crfnone) appends _authored_fences
                schedule = plan.schedule_id.split("_fastmath")[0]
                assert schedule.removesuffix("_authored_fences").endswith(
                    "_register_carry_only"
                )
            seen += 1
    for arch, sm_count, M, K, port in _r2_cells("R2_NATIVE_CELLS"):
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert (plan.kind, plan.kernel_key, plan.grid_x) == (
                "native",
                f"native_{port}",
                sm_count,
            )
            assert plan.fallback_from is None and plan.threads == cb.PERSISTENT_THREADS
            seen += 1
    # Round 5: a write cell names the dense key of the program whose snapshot-write variant the cell
    # runs (block K written), or one key per PDL mode; the dense cell itself keeps its own program.
    for arch, sm_count, M, K, value in _r2_cells("WRITE_PERSISTENT_KEY_CELLS"):
        for pdl in (False, True):
            key = value[int(pdl)] if isinstance(value, (list, tuple)) else value
            plan = cb._plan_route_exact(arch, sm_count, M, K, pdl, block_write_idx=K)
            assert plan.kind == "persistent" and plan.fallback_from is None, (plan, key)
            assert plan.kernel_key == f"{key}_write", (
                arch,
                sm_count,
                M,
                K,
                pdl,
                plan.kernel_key,
            )
            assert (
                plan.schedule_id.endswith("_write")
                and plan.threads == cb.PERSISTENT_THREADS
            )
            assert 1 <= plan.grid_x <= max(M, sm_count)
            dense = plan_route(arch, sm_count, M, K, pdl)
            # Round 6: the (256, 7) write cells of sm_100a and of sm_103a at 148 SMs run the
            # snapshot-write variant of the program their dense cell runs.
            on_dense = (arch, sm_count, M, K) in {
                ("sm_100a", 148, 256, 7),
                ("sm_103a", 148, 256, 7),
            }
            assert (dense.kernel_key == key) is on_dense, (dense, key)
            seen += 1
    # Rounds 6 and 17 add the (256, 4), (256, 7), (1024, 1), (1024, 4) and (1024, 7) write cells
    # (sm_100a: (256, 7), (1024, 1), (1024, 4) and (1024, 7); sm_103a: (256, 4), (256, 7) and
    # (1024, 7)); round 19 the sm_103a held-release (``_crheld``) K1 cells (1024, 1) at 148 SMs and
    # (4096, 1) at 152 SMs, one key per PDL mode (off, on).
    assert list(_r2_cells("WRITE_PERSISTENT_KEY_CELLS")) == [
        ("sm_100a", 148, 256, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        ("sm_100a", 148, 1024, 1, "persistent:k1_nc2_d3_f110000000000000_fm"),
        ("sm_100a", 148, 1024, 4, "persistent:k4_nc3_d2_f111000000000000"),
        ("sm_100a", 148, 1024, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        ("sm_100a", 148, 4096, 4, "persistent:k4_nc3_d2_f111000000000000"),
        ("sm_103a", 148, 256, 4, "persistent:k4_nc3_d2_f111000000000000_fm"),
        ("sm_103a", 148, 256, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        (
            "sm_103a",
            148,
            1024,
            1,
            [
                "persistent:k1_nc4_d3_f010000000000010_crheld",
                "persistent:k1_nc4_d3_f010000000100000_crheld",
            ],
        ),
        ("sm_103a", 148, 1024, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        ("sm_103a", 148, 4096, 4, "persistent:k4_nc3_d2_f111000000000000"),
        ("sm_103a", 152, 256, 4, "persistent:k4_nc3_d2_f111000000000000_fm"),
        ("sm_103a", 152, 256, 7, "persistent:k7_nc4_d2_f110000000000000_fm"),
        ("sm_103a", 152, 1024, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        (
            "sm_103a",
            152,
            4096,
            1,
            [
                "persistent:k1_nc4_d2_f010000000000010_fm_crheld",
                "persistent:k1_nc4_d2_f010000000100000_fm_crheld",
            ],
        ),
        ("sm_103a", 152, 4096, 4, "persistent:k4_nc3_d2_f111000000000000"),
    ]
    for arch, sm_count, M, K, key in _r2_cells("R2_PAIR_CELLS"):
        regs = key.endswith(("_cr232", "_cr200_pk1"))
        threads = 384 if regs else 320
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert (plan.kind, plan.kernel_key, plan.grid_x) == ("pair", key, sm_count)
            assert plan.threads == threads and plan.fallback_from is None
            assert plan.schedule_id == (
                f"pair_ws{threads}_nc2_prefix_bf16_add_fastmath"
                + ("_consumer_regs232" if key.endswith("_cr232") else "")
                + ("_consumer_regs200_park_acc1" if key.endswith("_cr200_pk1") else "")
                + f"_grid{sm_count}"
            )
            seen += 1
    for name, write in (
        ("R2_SMALL_M_CLUSTER_CELLS", False),
        ("R2_SMALL_M_WRITE_CLUSTER_CELLS", True),
    ):
        for arch, sm_count, M, K, cluster in _r2_cells(name):
            for pdl in (False, True):
                plan = plan_route(
                    arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
                )
                assert plan.kind == "small_m" and plan.fallback_from is None, plan
                assert plan.kernel_key.startswith(f"small_m_cluster{cluster}:k{K}"), (
                    plan
                )
                assert plan.kernel_key.endswith("_write") is write
                assert (plan.grid_x, plan.threads) == (M * cluster, 256 // cluster)
                seen += 1
    for name, write in (
        ("R2_SMALL_M_FAST_MATH_CELLS", False),
        ("R2_SMALL_M_WRITE_FAST_MATH_CELLS", True),
    ):
        for arch, sm_count, M, K, _none in _r2_cells(name):
            for pdl in (False, True):
                plan = plan_route(
                    arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
                )
                assert plan.kind == "small_m" and plan.fallback_from is None, plan
                assert "_fm" in plan.kernel_key and "_fastmath" in plan.schedule_id, (
                    plan
                )
                seen += 1
    assert seen > 0
    if MODULES:
        for arch, _sm, _m, _k, value in _r2_cells("R2_PERSISTENT_KEY_CELLS"):
            for key in value if isinstance(value, (list, tuple)) else (value,):
                assert key in KERNELS[arch], (arch, key)
    # The K1 M6144 cell of sm_103a names one key per PDL mode (the packed output-norm cache at PDL off).
    for sm_count in SM_COUNTS:
        off = plan_route("sm_103a", sm_count, 6144, 1, False).kernel_key
        on = plan_route("sm_103a", sm_count, 6144, 1, True).kernel_key
        assert (off, on) == (
            "persistent:k1_nc4_d2_f110000000000010",
            "persistent:k1_nc4_d2_f110000000000000",
        )
    # Cells round 2 left alone keep the legacy program: the GB300 K6 M3 small-M cell (the B300 K6
    # M192 cell runs the parked-accumulator pair program,
    # ``test_park_acc_pair_cells_take_the_384_thread_programs``). Round 6 moved the record-only K2
    # M8192 (152) / M4096 (148) cells to the K2 register-carry programs without the consumed-release
    # fence.
    assert (
        plan_route("sm_103a", 152, 3, 6, False).kernel_key == "small_m_cluster2:k6_fm"
    )
    assert (
        plan_route("sm_103a", 152, 8192, 2, False).kernel_key
        == "persistent:k2_nc3_d2_f011010000000000_crfnone"
    )
    assert (
        plan_route("sm_103a", 148, 4096, 2, False).kernel_key
        == "persistent:k2_nc3_d2_f011010000000000_fm_crfnone"
    )
    # Round 3: the sm_103a K5 cells M = 192 / 256 / 768 and the sm_100a K5 cells M = 192 / 256 / 384 /
    # 768 / 1024 / 1536 / 2048 run the native K5 port (one CTA per SM, one two-architecture program)
    # at the qualified SM counts and both PDL modes; M = 384 on sm_103a keeps its round-2 persistent
    # key at 152 SMs (round 6 gives 148 SMs an nc3 fast-math program) and M = 512 on sm_100a runs the
    # round-6 early-release program.  Round 20 moves the M = 192 cells of both architectures to the
    # ``_cr232`` pair program (``test_consumer_regs_pair_cells_take_the_384_thread_programs``).
    for sm_count in SM_COUNTS:
        for M in (256, 768):
            for pdl in (False, True):
                plan = plan_route("sm_103a", sm_count, M, 5, pdl)
                assert (
                    plan.kind,
                    plan.kernel_key,
                    plan.grid_x,
                    plan.fallback_from,
                ) == (
                    "native",
                    "native_k5",
                    sm_count,
                    None,
                ), (sm_count, M, pdl, plan)
                assert plan.schedule_id == f"native_k5_nc3_d2_ws288_grid{sm_count}"
        assert plan_route("sm_103a", sm_count, 384, 5, False).kernel_key == (
            "persistent:k5_nc3_d2_f111000000000000_fm"
            if sm_count == 148
            else "persistent:k5_nc4_d2_f111000000000000"
        )
    for M in (256, 384, 768, 1024, 1536, 2048):
        for pdl in (False, True):
            plan = plan_route("sm_100a", 148, M, 5, pdl)
            assert (plan.kind, plan.kernel_key, plan.grid_x, plan.fallback_from) == (
                "native",
                "native_k5",
                148,
                None,
            ), (M, pdl, plan)
            assert plan.schedule_id == "native_k5_nc3_d2_ws288_grid148"
    assert (
        plan_route("sm_100a", 148, 512, 5, False).kernel_key
        == "persistent:k5_nc4_d2_f111000000000000"
    )
    # Round 3: twelve sm_100a K2 / K3 cells run the fence-authored twins of their persistent
    # programs: the key of the fenced program plus ``_crfnone`` (a registered program of its own,
    # never a fallback across that suffix), the schedule label plus ``_authored_fences``, the same
    # grid; K3 M = 1536 and the band interiors keep the fenced programs.  Round 6 moved the K3 cells
    # from M = 3072 up to the held-release twin and the K3 M2048 / K2 M12288 / M16384 cells to the
    # bf16-prefix-add twin, and gives sm_103a its own nc3 K2 ``_crfnone`` programs on four large-M
    # cells (below); every other large-M sm_103a K2 / K3 cell keeps a fenced program, and the
    # sm_100a nc4 twins stay sm_100a only.
    sm100_fence_free = {
        (2048, 3): "011000000000000",
        (3072, 3): "010000000000000",
        (6144, 3): "010000000000000",
        (8192, 3): "010000000000000",
        (12288, 3): "010000000000000",
        (16384, 3): "010000000000000",
        (3072, 2): "110000000000000",
        (4096, 2): "110000000000000",
        (6144, 2): "110000000000000",
        (8192, 2): "110000000000000",
        (12288, 2): "011000000000000",
        (16384, 2): "011000000000000",
    }
    # Round 20 gives the K3 M2048 cell the packed register carry of its program (``_rcp``).
    sm100_packed = {(2048, 3)}
    for (M, K), flags in sm100_fence_free.items():
        packed = (M, K) in sm100_packed
        for pdl in (False, True):
            plan = plan_route("sm_100a", 148, M, K, pdl)
            assert (plan.kind, plan.kernel_key, plan.fallback_from) == (
                "persistent",
                f"persistent:k{K}_nc4_d2_f{flags}_crfnone" + ("_rcp" if packed else ""),
                None,
            ), (M, K, pdl, plan)
            assert plan.schedule_id == cb._persistent_schedule_id(
                4, 2, flags, True, packed_register_carry=packed
            )
            assert (plan.grid_x, plan.threads) == (
                cb._grid(M, 148, 4)[0],
                cb.PERSISTENT_THREADS,
            )
    for M, K in ((1024, 3), (1536, 3), (2560, 3), (5000, 2)):
        for pdl in (False, True):
            plan = plan_route("sm_100a", 148, M, K, pdl)
            assert plan.kind == "persistent" and plan.fallback_from is None, (
                M,
                K,
                pdl,
                plan,
            )
            assert not plan.kernel_key.endswith("_crfnone"), (M, K, pdl, plan)
            assert not plan.schedule_id.endswith("_authored_fences")
    round_6 = {
        (148, 4096, 2): "persistent:k2_nc3_d2_f011010000000000_fm_crfnone",
        (148, 16384, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
        (152, 8192, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
        (152, 12288, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
    }
    for sm_count in SM_COUNTS:
        for M in (2048, 3072, 4096, 6144, 8192, 12288, 16384):
            for K in (2, 3):
                key = plan_route("sm_103a", sm_count, M, K, False).kernel_key
                if (sm_count, M, K) in round_6:
                    assert key == round_6[sm_count, M, K], (sm_count, M, K, key)
                else:
                    assert not key.endswith("_crfnone"), (sm_count, M, K, key)
    # The sm_100a (384, 3) / (1024, 3) cells at 148 SMs run programs built for two CTAs per SM
    # (``_cta2``): a one-deep chunk ring with the early consumed release, launched on min(M, 2 x SMs)
    # CTAs.
    for M, key in (
        (384, "persistent:k3_nc4_d1_f111000000000000_fm_cta2"),
        (1024, "persistent:k3_nc4_d1_f110000000000000_cta2"),
    ):
        for pdl in (False, True):
            plan = plan_route("sm_100a", 148, M, 3, pdl)
            assert (plan.kind, plan.kernel_key, plan.grid_x) == (
                "persistent",
                key,
                296,
            ), (M, pdl, plan)
            assert plan.schedule_id.endswith(
                "_fastmath_two_ctas_per_sm"
                if key.endswith("_fm_cta2")
                else "_two_ctas_per_sm"
            )
            assert plan.route_id.endswith(".two_ctas_per_sm")
    if MODULES:
        assert KERNELS["sm_100a"]["native_k5"] == KERNELS["sm_103a"]["native_k5"]
        assert MODULES[KERNELS["sm_103a"]["native_k5"]]["arches"] == [
            "sm_100a",
            "sm_103a",
        ]
        for key in sorted(
            {
                f"persistent:k{K}_nc4_d2_f{flags}_crfnone"
                + ("_rcp" if (M, K) in sm100_packed else "")
                for (M, K), flags in sm100_fence_free.items()
            }
        ):
            assert key in KERNELS["sm_100a"] and key not in KERNELS["sm_103a"]
            assert MODULES[KERNELS["sm_100a"][key]]["arches"] == ["sm_100a"]
            assert (
                MODULES[KERNELS["sm_100a"][key]]["schedule_id"]
                .removesuffix("_packed_register_carry")
                .endswith("_authored_fences")
            )


def test_held_release_cells_take_the_held_programs():
    """Round 19: a ``_crheld`` key spells flag bit 0 as 0 and names the program of the same key
    with bit 0 set, built without the early consumed release; its schedule label adds
    ``_held_release``.  The sm_103a (1024, 5) cell runs the held K5 program at both SM counts and
    PDL modes, and the held K1 write cells keep the label ahead of ``_fastmath`` / ``_write``."""
    for sm_count in SM_COUNTS:
        for pdl in (False, True):
            plan = plan_route("sm_103a", sm_count, 1024, 5, pdl)
            assert (
                plan.kernel_key == "persistent:k5_nc4_d3_f001000000000000_fm_crheld"
            ), (sm_count, pdl, plan)
            assert plan.schedule_id.endswith("_held_release_fastmath"), plan
            assert "_early_consume" not in plan.schedule_id, plan
    for sm_count, M, suffix in ((148, 1024, ""), (152, 4096, "_fastmath")):
        for pdl in (False, True):
            plan = plan_route("sm_103a", sm_count, M, 1, pdl, block_write_idx=1)
            assert plan.kernel_key.endswith("_crheld_write"), (sm_count, pdl, plan)
            assert plan.schedule_id.endswith(f"_held_release{suffix}_write"), plan


def test_packed_register_carry_cells_take_the_packed_programs():
    """Round 20: a ``_rcp`` key names the K = 3 register-carry program of the same key that keeps
    each token's bf16 source words in registers instead of TMEM; its schedule label adds
    ``_packed_register_carry`` ahead of ``_fastmath``.  The sm_100a (256, 3), (1536, 3) and
    (2048, 3) cells and the sm_103a (256, 3), (512, 3) and (1536, 3) cells run it at both PDL
    modes (sm_103a at both SM counts)."""
    cells = [("sm_100a", 148, M) for M in (256, 1536, 2048)] + [
        ("sm_103a", sm_count, M) for sm_count in SM_COUNTS for M in (256, 512, 1536)
    ]
    for arch, sm_count, M in cells:
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, 3, pdl)
            assert plan.kind == "persistent" and plan.fallback_from is None, plan
            assert plan.kernel_key.endswith("_rcp"), (arch, sm_count, M, pdl, plan)
            assert "_packed_register_carry" in plan.schedule_id, plan
            assert plan.schedule_id.endswith(
                "_packed_register_carry_fastmath"
                if "_fm_" in plan.kernel_key
                else "_packed_register_carry"
            ), plan
            assert MODULES[KERNELS[arch][plan.kernel_key]]["schedule_id"] == (
                plan.schedule_id
            ), plan


def test_a_packed_register_carry_key_never_falls_back_across_the_carry():
    """A fallback keeps the ``_rcp`` line: an unregistered ``_rcp`` key resolves to a registered
    ``_rcp`` program or stays exact, and an unregistered key without the suffix never resolves to
    a ``_rcp`` program."""
    for arch, table in KERNELS.items():
        for K in range(MAX_BLOCKS + 1):
            for packed in (False, True):
                key = f"persistent:k{K}_nc4_d2_f000000000000001" + (
                    "_rcp" if packed else ""
                )
                assert key not in table
                plan = cb.RoutePlan(
                    "persistent", key, 148, cb.PERSISTENT_THREADS, "", "", arch, False
                )
                resolved = cb._resolve_registered(plan, 148, 4096)
                if resolved.kernel_key != key:
                    assert resolved.kernel_key.endswith("_rcp") == packed, resolved
                    assert ("_packed_register_carry" in resolved.schedule_id) == packed


def test_consumer_regs_pair_cells_take_the_384_thread_programs():
    """Round 20: a ``_cr232`` pair key is the pair kernel built at ``consumer_regs`` 232 (three
    whole warpgroups, 384 threads); its schedule label adds ``_consumer_regs232`` ahead of the grid.
    The sm_100a (192, 4), (192, 5) and (192, 8) cells and the sm_103a (192, 4) and (192, 5) cells at
    both SM counts run it at both PDL modes; no cell runs the 320-thread pair program."""
    cells = [("sm_100a", 148, K) for K in (4, 5, 8)] + [
        ("sm_103a", sm_count, K) for sm_count in SM_COUNTS for K in (4, 5)
    ]
    for arch, sm_count, K in cells:
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, 192, K, pdl)
            assert (plan.kind, plan.kernel_key) == (
                "pair",
                f"pair:k{K}_nc2_pbf16a_fm_cr232",
            ), (arch, sm_count, K, pdl, plan)
            assert plan.threads == cb.PAIR_CONSUMER_REGS_THREADS == 384, plan
            assert plan.schedule_id == (
                f"pair_ws384_nc2_prefix_bf16_add_fastmath_consumer_regs232_grid{sm_count}"
            ), plan
            name = KERNELS[arch][plan.kernel_key]
            defines = _compile_defines(
                _LIBRARIES_BY_NAME[cake_jit.library_name(name, arch)]
            )
            assert int(defines["CAKE_BIND_THREADS"]) == 384, (name, defines)
    assert all(
        key.endswith(("_cr232", "_cr200_pk1")) for *_, key in _r2_cells("R2_PAIR_CELLS")
    )


def test_park_acc_pair_cells_take_the_384_thread_programs():
    """A ``_cr200_pk1`` pair key is the pair kernel built at ``consumer_regs`` 200 with
    ``park_acc`` 1 (three whole warpgroups, 384 threads; each consumer thread's accumulators wait in
    tensor memory while the prefix chunk streams); its schedule label adds
    ``_consumer_regs200_park_acc1`` ahead of the grid. The sm_100a (192, 6) and (192, 7) cells at
    148 SMs and the sm_103a (192, 6), (192, 7) and (192, 8) cells at both SM counts run it at both
    PDL modes."""
    cells = [("sm_100a", 148, K) for K in (6, 7)] + [
        ("sm_103a", sm_count, K) for sm_count in SM_COUNTS for K in (6, 7, 8)
    ]
    for arch, sm_count, K in cells:
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, 192, K, pdl)
            assert (plan.kind, plan.kernel_key) == (
                "pair",
                f"pair:k{K}_nc2_pbf16a_fm_cr200_pk1",
            ), (arch, sm_count, K, pdl, plan)
            assert plan.threads == cb.PAIR_CONSUMER_REGS_THREADS == 384, plan
            assert plan.schedule_id == (
                "pair_ws384_nc2_prefix_bf16_add_fastmath_consumer_regs200_park_acc1"
                f"_grid{sm_count}"
            ), plan
            name = KERNELS[arch][plan.kernel_key]
            defines = _compile_defines(
                _LIBRARIES_BY_NAME[cake_jit.library_name(name, arch)]
            )
            assert int(defines["CAKE_BIND_THREADS"]) == 384, (name, defines)
    assert sorted(
        (arch, sm_count, M, K)
        for arch, sm_count, M, K, key in _r2_cells("R2_PAIR_CELLS")
        if key.endswith("_pk1")
    ) == sorted((arch, sm_count, 192, K) for arch, sm_count, K in cells)


@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize(
    "arch,M,nc,depth",
    (
        # sm_100a: nc4 depth 3 from the band edge (round r4 closing screening); the r1 M4096 cell keeps nc3 depth 3.
        # The M1536 cell itself runs the native K5 port since round 3 phase C (asserted exactly in
        # test_round_2_cells_take_their_table_program); M1537 is the first persistent row of the depth-3 band.
        ("sm_100a", 1535, 4, 2),
        ("sm_100a", 1537, 4, 3),
        ("sm_100a", 2049, 4, 3),
        ("sm_100a", 3072, 4, 3),
        ("sm_100a", 4096, 3, 3),
        ("sm_100a", 8192, 4, 3),
        ("sm_100a", 16384, 4, 3),
        # sm_103a: nc4 depth 3 from M1536 (round r4 direction 2).
        ("sm_103a", 1535, 4, 2),
        ("sm_103a", 1536, 4, 3),
        ("sm_103a", 4096, 4, 3),
        ("sm_103a", 16384, 4, 3),
    ),
)
def test_plan_route_k5_depth3_bands(arch, M, nc, depth, pdl):
    """Round r4: the persistent K5 program runs four sources per chunk with a depth-3 pipeline from the
    per-architecture band edge (same-GPU ABBA, bit-identical, 1.3-3.8 % faster than depth 2); the band
    edges are the dispatcher / evaluator / FI mirror constants. Since round 3 phase C the sm_100a exact
    cells M 192 / 256 / 384 / 768 / 1024 / 1536 / 2048 (K = 5) run the native K5 port instead (asserted
    in test_round_2_cells_take_their_table_program), so the sm_100a rows here sit beside those cells."""
    plan = plan_route(arch, SM_COUNT, M, 5, pdl)
    assert plan.kind == "persistent", plan
    assert f"_nc{nc}_d{depth}_" in plan.schedule_id, (plan.schedule_id, nc, depth)


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize(
    "M,K",
    ((1, 1), (16, 7), (64, 4), (256, 7), (256, 4), (1024, 1), (1024, 4), (4096, 4)),
)
def test_plan_route_snapshot_write_takes_the_write_variants(arch, M, K, pdl):
    """Round r4 (direction 4 / 4b): the block-boundary snapshot write (block K written) runs
    the write variant of the dense cell's program - the small-M write programs inside the
    dense family's M table, the persistent write variant above it (also where the dense cell
    runs a native port, e.g. sm_100a K4 M1024 / sm_103a K4 M256)."""
    plan = cb._plan_route_exact(arch, SM_COUNT, M, K, pdl, block_write_idx=K)
    dense = cb._persistent_plan_exact(arch, SM_COUNT, M, K, pdl, write=True)
    max_m = cb._SMALL_M_DIRECT_MAX_M[arch].get(K)
    assert plan.fallback_from is None
    assert f".k{K}.delta1.write1.norm1.pdl{int(pdl)}." in plan.route_id
    assert plan.schedule_id.endswith("_write") and plan.kernel_key.endswith("_write")
    if max_m is not None and max_m >= M:
        cluster = cb._small_m_cluster(arch, M, K, num_sms=SM_COUNT, write=True)
        nc = cb._small_m_sources_per_chunk(arch, M, K)
        nc_suffix = "" if nc is None else f"_nc{nc}"
        fm = cb._fast_math(arch, M, K, "small_m_write", SM_COUNT)
        family = "direct" if cluster == 1 else f"cluster{cluster}"
        assert plan.kind == "small_m"
        assert (
            plan.kernel_key
            == f"small_m_{family}:k{K}{nc_suffix}{'_fm' if fm else ''}_write"
        )
        assert plan.grid_x == M * cluster and plan.threads == 256 // cluster
        assert plan.schedule_id.endswith(
            f"_regres_fp32x2{nc_suffix}{'_fastmath' if fm else ''}_write"
        )
    else:
        assert plan.kind == "persistent"
        assert plan.kernel_key == f"{dense.kernel_key}_write"
        assert plan.schedule_id == f"{dense.schedule_id}_write"
        assert (plan.grid_x, plan.threads) == (dense.grid_x, dense.threads)
        assert plan.route_id == dense.route_id.replace(
            dense.schedule_id, plan.schedule_id, 1
        ).replace(".write0.", ".write1.", 1)
    for bad_idx in (K - 1, K + 1, MAX_BLOCKS):
        with pytest.raises(ValueError):
            cb._plan_route_exact(arch, SM_COUNT, M, K, pdl, block_write_idx=bad_idx)


def test_write_cell_programs_serve_their_cells_only():
    """A ``WRITE_PERSISTENT_KEY_CELLS`` program is the exact program of its cells and never the
    registered-variant fallback of another snapshot-write plan (round 5 pins the (4096, 4) write
    cell to a depth-2 program that would otherwise be the closest registered variant of every
    unregistered depth-2 K = 4 write key), so pinning a write cell changes no other route."""
    cells = {}
    for arch, sm_count, M, K, value in _r2_cells("WRITE_PERSISTENT_KEY_CELLS"):
        for pdl in (False, True):
            key = value[int(pdl)] if isinstance(value, (list, tuple)) else value
            cells[(arch, sm_count, M, K, pdl)] = f"{key}_write"
    pinned = {
        arch: {k for (a, _s, _m, _k, _p), k in cells.items() if a == arch}
        for arch in ARCHES
    }
    assert all(pinned[arch] for arch in ARCHES)
    assert pinned == {arch: cb._cell_pinned_write_keys(arch) for arch in ARCHES}
    seen = 0
    for arch in ARCHES:
        for sm_count in SM_COUNTS:
            for K in range(1, MAX_BLOCKS):
                for pdl in (False, True):
                    for M in _table_edge_token_counts():
                        plan = plan_route(arch, sm_count, M, K, pdl, block_write_idx=K)
                        if (arch, sm_count, M, K, pdl) in cells:
                            assert plan.kernel_key == cells[arch, sm_count, M, K, pdl]
                            assert plan.fallback_from is None
                            continue
                        assert plan.kernel_key not in pinned[arch], (
                            arch,
                            sm_count,
                            M,
                            K,
                            pdl,
                            plan.kernel_key,
                            plan.fallback_from,
                        )
                        seen += plan.fallback_from is not None
    assert seen


def test_snapshot_write_fallback_stays_in_the_write_family(monkeypatch):
    """A registered-variant fallback of a snapshot-write plan only ever selects another
    small-M write program (the snapshot store exists nowhere else)."""
    # round r5: the write twin's fast-math class is part of the key; the fallback never crosses it either
    fm = "_fm" if cb._fast_math("sm_100a", 64, 7, "small_m_write") else ""
    table = {
        f"small_m_direct:k7{fm}_write": "m1",
        f"small_m_direct:k7{fm}": "m2",
        "persistent:k7_nc4_d2_f110000000000000": "m3",
    }
    monkeypatch.setattr(cb, "KERNELS", {"sm_100a": table})
    exact = cb._plan_route_exact("sm_100a", SM_COUNT, 64, 7, False, block_write_idx=7)
    assert exact.kernel_key not in table
    resolved = cb._resolve_registered(exact, SM_COUNT, 64)
    assert resolved.kernel_key == f"small_m_direct:k7{fm}_write"
    assert resolved.fallback_from == exact.kernel_key
    assert (
        resolved.route_id.endswith(".registered_fallback")
        and ".write1." in resolved.route_id
    )
    monkeypatch.setattr(cb, "KERNELS", {"sm_100a": {f"small_m_direct:k7{fm}": "m2"}})
    unresolved = cb._resolve_registered(exact, SM_COUNT, 64)
    assert (
        unresolved.kernel_key == exact.kernel_key and unresolved.fallback_from is None
    )


def test_plan_route_bootstrap_variants():
    # The bootstrap key encodes the semantics (K, delta, snapshot write, output norm) and never
    # substitutes; the three registered variants are K0 without delta with the write, K4
    # without delta and K8 without the output norm.
    plan = plan_route(
        None, 0, 1, 0, False, has_delta=False, block_write_idx=0, common_path=False
    )
    assert plan.kind == "direct"
    assert plan.kernel_key == "direct:k0_d0_w1_n1"
    assert plan.grid_x == 1 and plan.threads == 256
    plan = plan_route(None, 0, 17, 4, False, has_delta=False, common_path=False)
    assert plan.kind == "direct"
    assert plan.kernel_key == "direct:k4_d0_w0_n1"
    assert plan.grid_x == 17 and plan.threads == 256
    plan = plan_route(
        "sm_100a", SM_COUNT, 7, 8, True, apply_output_norm=False, common_path=False
    )
    assert plan.kernel_key == "direct:k8_d1_w0_n0"
    if MODULES:
        registered = ("direct:k0_d0_w1_n1", "direct:k4_d0_w0_n1", "direct:k8_d1_w0_n0")
        for arch in ARCHES:
            assert all(key in KERNELS[arch] for key in registered)


def test_plan_route_rejects_bad_shapes():
    with pytest.raises(ValueError):
        plan_route("sm_100a", SM_COUNT, 0, 4, False)
    with pytest.raises(ValueError):
        plan_route("sm_100a", SM_COUNT, 1, MAX_BLOCKS + 1, False)
    with pytest.raises(ValueError):
        plan_route("sm_90a", SM_COUNT, 1, 4, False)


def test_native_ports_route_on_the_qualified_sm_counts():
    # M = 256 / K = 8 is a native_k8 cell (M = 1 belongs to the small-M direct kernel); the port launches one
    # CTA per SM on every qualified count and keeps its kernel key (one registered program)
    keys = set()
    for sm_count in SM_COUNTS:
        plan = plan_route("sm_100a", sm_count, 256, 8, False)
        assert plan.kind == "native"
        assert plan.grid_x == sm_count
        assert plan.schedule_id == f"native_k8_nc3_d2_ws288_grid{sm_count}"
        keys.add(plan.kernel_key)
    assert keys == {"native_k8"}
    # any other count takes the persistent family with a grid derived from the actual count
    plan = plan_route("sm_100a", 132, 256, 8, False)
    assert plan.kind == "persistent"
    assert plan.grid_x <= 132


_CSRC = pathlib.Path(cake_build.__file__).resolve().parent / "csrc"
_LIBRARIES_BY_NAME = cake_build.libraries_by_name() if MODULES else {}


def _record_kernel_source(name):
    """The kernel ``.cu`` of a registry record, resolved through ``build.json``.  After
    the fast-math and per-architecture folds a record's device source is one shared file
    named for another record, so a record's kernel is the first source of any of its
    ``build.json`` entries."""
    arch = next(iter(MODULES[name]["arches"]))
    library = _LIBRARIES_BY_NAME[cake_jit.library_name(name, arch)]
    return _CSRC.parent / library["sources"][0]


_TMEM_PROGRAMS = sorted(
    name
    for name in MODULES
    if "tcgen05.dealloc" in _record_kernel_source(name).read_text()
)


@pytest.mark.parametrize("name", _TMEM_PROGRAMS)
def test_tmem_programs_drain_loads_before_the_dealloc(name):
    # The persistent and native programs read TMEM with tcgen05.ld and free the columns
    # with tcgen05.dealloc after the cleanup barrier. A register dependency on a tcgen05.ld
    # result does not complete the load; only tcgen05.wait::ld does (PTX ISA, tcgen05 memory
    # consistency model), so a wait::ld must come before the barrier that precedes the dealloc.
    text = _record_kernel_source(name).read_text()
    dealloc = text.index("tcgen05.dealloc")
    cleanup = text.rindex("barrier.sync", 0, dealloc)
    assert "tcgen05.ld." in text[:cleanup]
    assert "tcgen05.wait::ld" in text[text.rindex("tcgen05.ld.", 0, cleanup) : cleanup]


def test_small_m_table_boundary():
    for arch in ("sm_100a", "sm_103a"):
        for K in range(MAX_BLOCKS + 1):
            max_m = cb._SMALL_M_DIRECT_MAX_M[arch].get(K)
            if max_m is None:
                assert plan_route(arch, SM_COUNT, 1, K, False).kind != "small_m"
                continue
            bands = cb._SMALL_M_CLUSTER[arch].get(K, ())
            assert list(bands) == sorted(bands)
            assert all(cs in (2, 4) and 1 <= mm <= max_m for mm, cs in bands)
            chunk_bands = cb._SMALL_M_CHUNK_BANDS[arch].get(K, ())
            assert list(chunk_bands) == sorted(chunk_bands)
            assert all(1 <= nc <= 8 and 1 <= mm <= max_m for mm, nc in chunk_bands)
            edges = {mm for mm, _cs in bands} | {
                mm + 1 for mm, _cs in bands if mm < max_m
            }
            edges |= {mm for mm, _nc in chunk_bands} | {
                mm + 1 for mm, _nc in chunk_bands if mm < max_m
            }
            for m in sorted({1, max_m} | edges):
                at = cb._plan_route_exact(arch, SM_COUNT, m, K, False)
                if (m, K) in cb._R5_SMALL_M_EXCLUDED_CELLS[arch]:
                    # round r5: the dense cell left the table for its persistent program
                    assert at.kind == "persistent"
                    continue
                cluster = cb._small_m_cluster(arch, m, K, num_sms=SM_COUNT)
                nc = cb._small_m_sources_per_chunk(arch, m, K)
                fm = cb._fast_math(arch, m, K, "small_m", SM_COUNT)
                suffix = ("" if nc is None else f"_nc{nc}") + ("_fm" if fm else "")
                key = (
                    f"small_m_direct:k{K}{suffix}"
                    if cluster == 1
                    else f"small_m_cluster{cluster}:k{K}{suffix}"
                )
                assert at.schedule_id.endswith(
                    ("" if nc is None else f"_nc{nc}") + ("_fastmath" if fm else "")
                )
                assert (at.kind, at.kernel_key, at.grid_x, at.threads) == (
                    "small_m",
                    key,
                    m * cluster,
                    cb.DIRECT_THREADS // cluster,
                )
            at = plan_route(arch, SM_COUNT, max_m, K, False)
            # the direct kernel is SM-count agnostic and PDL only changes the launch argument; a
            # round-r5 cell at max_m runs the persistent program its SM count's table names
            sm_count = 132 if at.kind == "small_m" else SM_COUNT
            assert (
                plan_route(arch, sm_count, max_m, K, True).kernel_key == at.kernel_key
            )
            assert plan_route(arch, SM_COUNT, max_m + 1, K, False).kind != "small_m"


def test_persistent_key_is_the_complete_flag_tuple():
    # M 1024 is the first K1 token count above the small-M table on both architectures
    a = plan_route("sm_103a", SM_COUNT, 1024, 1, True)
    b = plan_route("sm_103a", SM_COUNT, 1024, 1, False)
    for plan in (a, b):
        assert plan.kind == "persistent"
        assert re.fullmatch(r"persistent:k1_nc\d_d\d_f[01]{15}(_fm)?", plan.kernel_key)
    # The PDL mode only changes the launch argument and the route id.
    assert a.route_id != b.route_id


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
def test_registered_keys_cover_the_measured_grid_when_programs_exist(arch, sm_count):
    if not MODULES:
        pytest.skip("no generated programs registered in this checkout")
    for pdl in (False, True):
        for M in TOKEN_COUNTS:
            for K in PRIMARY_K:
                plan = plan_route(arch, sm_count, M, K, pdl)
                assert plan.kernel_key in KERNELS[arch], (
                    arch,
                    sm_count,
                    M,
                    K,
                    pdl,
                    plan.kernel_key,
                )
        for M in (1, 4096):
            for K in range(MAX_BLOCKS + 1):
                plan = plan_route(arch, sm_count, M, K, pdl)
                assert plan.kernel_key in KERNELS[arch], (
                    arch,
                    sm_count,
                    M,
                    K,
                    pdl,
                    plan.kernel_key,
                )
    for record in MODULES.values():
        pairs = [tuple(pair) for pair in record["arg_plan"]]
        assert pairs[-4:] == [
            ("launch", cake_jit.LAUNCH_ARGUMENT),
            ("grid", "grid_x"),
            ("grid", "grid_y"),
            ("grid", "grid_z"),
        ], record["library"]
        assert all(kind in ("buffer", "parameter") for kind, _ in pairs[:-4]), record[
            "library"
        ]
        assert {name for kind, name in pairs[:-4]} <= set(record["logical"]), record[
            "library"
        ]
        assert record["arches"] and set(record["arches"]) <= set(ARCHES)


# The five shared launch bindings under ``csrc/``, one per argument-plan form.  Every
# ``build.json`` entry names its form's binding in ``sources[1]`` and carries the
# binding's selectors in ``nvcc_flags``: ``-DCAKE_PROGRAM_KERNEL`` (the kernel symbol
# its device source defines) and the ``-DCAKE_BIND_*`` launch literals the binding
# refuses to compile without.
_LAUNCH_BINDINGS = (
    "cake_kimi_k3_attn_res_common_binding.cu",
    "cake_kimi_k3_attn_res_small_m_cluster_binding.cu",
    "cake_kimi_k3_attn_res_small_m_direct_binding.cu",
    "cake_kimi_k3_attn_res_direct_binding.cu",
    "cake_kimi_k3_attn_res_k0_tma_binding.cu",
)
_BINDING_ALWAYS_REQUIRED = {
    "CAKE_PROGRAM_KERNEL",
    "CAKE_BIND_SMEM",
    "CAKE_BIND_THREADS",
}
_BINDING_REQUIRED_MACRO = re.compile(r"^#ifndef (\w+)\n#error ", re.M)
_KERNEL_DEFINITION = re.compile(r"^(kernel_\w+|CAKE_PROGRAM_KERNEL)\(", re.M)
_PROGRAM_MACRO_DEFAULT = re.compile(r"^#define CAKE_PROGRAM_KERNEL (\w+)$", re.M)


def _launch_binding(record):
    """The shared launch binding of a registry record, by its kind and kernel key."""
    kind = record["kind"]
    if kind == "small_m":
        form = "cluster" if "cluster" in record["kernel_key"] else "direct"
        return f"cake_kimi_k3_attn_res_small_m_{form}_binding.cu"
    if kind in ("direct", "k0_tma"):
        return f"cake_kimi_k3_attn_res_{kind}_binding.cu"
    return "cake_kimi_k3_attn_res_common_binding.cu"


def _binding_required(binding):
    """The ``-D`` names a shared launch binding refuses to compile without."""
    text = (_CSRC / binding).read_text(encoding="utf-8")
    return set(_BINDING_REQUIRED_MACRO.findall(text))


def _compile_defines(entry):
    """``name -> value`` of the ``-D`` flags of a ``build.json`` entry."""
    return dict(
        flag[2:].split("=", 1) if "=" in flag else (flag[2:], "1")
        for flag in entry["nvcc_flags"]
        if flag.startswith("-D")
    )


def _defined_kernel_symbol(entry):
    """The kernel symbol the entry's device source defines under its ``-D`` flags: the
    literal name at the definition, or the ``-DCAKE_PROGRAM_KERNEL`` value (else the
    source's ``#ifndef`` default) when the definition spells the macro."""
    text = (_CSRC.parent / entry["sources"][0]).read_text(encoding="utf-8")
    definition = _KERNEL_DEFINITION.search(text)
    assert definition is not None, entry["name"]
    if definition.group(1) != "CAKE_PROGRAM_KERNEL":
        return definition.group(1)
    default = _PROGRAM_MACRO_DEFAULT.search(text)
    defines = _compile_defines(entry)
    return defines.get("CAKE_PROGRAM_KERNEL", default and default.group(1))


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding sources
    under ``csrc/`` of this package, their digests and a closure digest; the registry
    itself records the host plan only.  ``--use_fast_math`` is an entry flag exactly
    for the fast-math programs (an ``_fm`` key, the native ports and the pair kernel).
    """
    if not MODULES:
        pytest.skip("no generated programs registered in this checkout")
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == cake_jit.__name__.rsplit(".", 1)[0]
    libraries = cake_build.libraries_by_name()
    package = pathlib.Path(cake_build.__file__).parent
    referenced = set()
    for name, record in MODULES.items():
        assert record["library"] == name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        assert record["kind"] in (
            "direct",
            "small_m",
            "persistent",
            "k0_tma",
            "native",
            "pair",
        )
        fast_math = bool(
            re.search(
                r"_fm(_crfnone)?(_crheld)?(_rcp)?(_write)?(_cta2)?$",
                record["kernel_key"],
            )
        )
        fast_math = fast_math or record["kind"] in ("native", "pair")
        for arch in record["arches"]:
            library = cake_jit.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            binding = _launch_binding(record)
            assert entry["sources"][1] == f"csrc/{binding}", name
            shared = re.fullmatch(
                r"csrc/(cake_kimi_k3_attn_res_[0-9a-f]{20})_kernel\.cu",
                entry["sources"][0],
            )
            assert shared is not None, entry["sources"][0]
            assert MODULES[shared.group(1)]["kind"] == record["kind"], name
            defines = _compile_defines(entry)
            assert defines.get("CAKE_PROGRAM_KERNEL") == f"kernel_{name}", name
            assert _binding_required(binding) <= set(defines), (name, arch)
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert ("--use_fast_math" in entry["nvcc_flags"]) is fast_math, (name, arch)
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()
    for arch, table in KERNELS.items():
        for key, name in table.items():
            assert (
                MODULES[name]["kernel_key"] == key and arch in MODULES[name]["arches"]
            )


def test_shared_sources_keep_distinct_program_symbols():
    """Every library of the family loads into one process, so no two libraries of one
    architecture may export one kernel symbol.  Each ``build.json`` entry selects its
    own through ``-DCAKE_PROGRAM_KERNEL``, equal to the symbol its device source
    defines (the literal name, or the macro at the definition with the entry's value),
    and the 187 records' symbols are distinct per architecture."""
    libraries = cake_build.libraries_by_name()
    symbols_by_arch = {}
    for library, entry in libraries.items():
        arch = entry["arch"]
        name = library[: -len(arch) - 1]
        assert name in MODULES and arch in MODULES[name]["arches"], library
        symbol = _compile_defines(entry).get("CAKE_PROGRAM_KERNEL")
        assert symbol == f"kernel_{name}", (library, symbol)
        assert _defined_kernel_symbol(entry) == symbol, library
        symbols_by_arch.setdefault(arch, set()).add(symbol)
    for arch, symbols in symbols_by_arch.items():
        assert len(symbols) == sum(e["arch"] == arch for e in libraries.values()), arch
    assert set().union(*symbols_by_arch.values()) == {
        f"kernel_{name}" for name in MODULES
    }


def test_launch_bindings_take_each_record_from_its_compile_flags():
    """One shared launch binding per argument-plan form.  Every ``build.json`` entry
    names one of the five in ``sources[1]`` and carries exactly the ``-DCAKE_BIND_*``
    selectors its binding refuses to compile without, each with an integer or
    identifier value; the two the kernel key determines (the last snapshot row
    ``K - 1 + write``; the write-index arm) equal the key's.  ``TVM_FFI_CHECK``
    stringises its condition unexpanded, so the bindings keep the selectors out of
    its conditions and route a condition that needs one through
    ``CAKE_CHECK_EXPANDED``."""
    required = {}
    for binding in _LAUNCH_BINDINGS:
        text = (_CSRC / binding).read_text(encoding="utf-8")
        required[binding] = _binding_required(binding)
        assert required[binding] >= _BINDING_ALWAYS_REQUIRED, binding
        for line in text.splitlines():
            if line.lstrip().startswith("TVM_FFI_CHECK("):
                assert not set(re.findall(r"\w+", line)) & required[binding], line
    libraries = cake_build.libraries_by_name()
    for name, record in MODULES.items():
        binding = _launch_binding(record)
        key = record["kernel_key"]
        for arch in record["arches"]:
            entry = libraries[cake_jit.library_name(name, arch)]
            assert entry["sources"][1] == f"csrc/{binding}", (name, arch)
            defines = _compile_defines(entry)
            selectors = {macro for macro in defines if macro.startswith("CAKE_BIND_")}
            assert selectors == required[binding] - {"CAKE_PROGRAM_KERNEL"}, name
            for macro in required[binding]:
                assert re.fullmatch(r"-?\d+|[A-Za-z_]\w*", defines[macro]), macro
            if "CAKE_BIND_KM" in defines:
                K = int(re.search(r":k(\d+)", key).group(1))
                write = key.endswith("_write")
                assert int(defines["CAKE_BIND_KM"]) == K - 1 + write, name
            if "CAKE_BIND_WRITE_IDX" in defines:
                write_idx = "1" if "_w1_" in key else "0"
                assert defines["CAKE_BIND_WRITE_IDX"] == write_idx, name


# Every token count a decoder step can present in the small / mid range, plus the measured
# powers of two and a few large non-powers of two.
DENSE_TOKEN_COUNTS = (
    tuple(range(1, 601))
    + TOKEN_COUNTS[10:]
    + (
        1000,
        1536,
        3000,
        6144,
        10000,
        16383,
    )
)


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
def test_every_common_path_route_resolves_to_a_registered_program(arch, sm_count):
    if not MODULES:
        pytest.skip("no generated programs registered in this checkout")
    registered = KERNELS[arch]
    for pdl in (False, True):
        for K in range(MAX_BLOCKS + 1):
            for M in DENSE_TOKEN_COUNTS:
                exact = cb._plan_route_exact(arch, sm_count, M, K, pdl)
                plan = plan_route(arch, sm_count, M, K, pdl)
                cell = (arch, sm_count, M, K, pdl, exact.kernel_key, plan.kernel_key)
                assert plan.kernel_key in registered, cell
                assert plan.kind in (
                    "small_m",
                    "native",
                    "pair",
                    "k0_tma",
                    "persistent",
                ), cell
                if exact.kernel_key in registered:
                    # a registered variant always runs exactly as the tables say
                    assert plan == exact and plan.fallback_from is None, cell
                    continue
                assert plan.fallback_from == exact.kernel_key, cell
                assert plan.route_id.endswith(".registered_fallback"), cell
                assert plan.kind in ("small_m", "persistent"), cell
                assert plan.kernel_key.split(":k", 1)[1].split("_", 1)[0] == str(K), (
                    cell
                )
                assert (plan.arch, plan.use_pdl) == (arch, pdl), cell
                if plan.kind == "small_m":
                    cluster = cb.DIRECT_THREADS // plan.threads
                    assert cluster in (1, 2, 4), cell
                    assert plan.kernel_key.startswith(
                        "small_m_direct:"
                        if cluster == 1
                        else f"small_m_cluster{cluster}:"
                    ), cell
                    assert plan.grid_x == M * cluster, cell
                else:
                    assert plan.threads == cb.PERSISTENT_THREADS and plan.grid_x >= 1, (
                        cell
                    )
    # The measured cells never substitute.
    for row_arch, M, K, pdl, _kind, _schedule_id, _grid_x in POLICY_ROWS:
        if row_arch == arch:
            assert plan_route(arch, sm_count, M, K, pdl).fallback_from is None, (
                sm_count,
                M,
                K,
                pdl,
            )


# First token count of every dense route band: the M at which ``plan_route`` changes its kernel
# key on some architecture, SM count in SM_COUNTS or PDL setting, per K.  The bands follow the
# route tables (the small-M table and its cluster / chunk / fast-math bands, the round-r5 cells,
# the round-2 / round-3 / round-6 / round-20 cells, the release bands, the nc / depth edges, the native and
# pair cells) and, for K8 on sm_100a, the SM count itself (the release flips at M = SMs + 1).
# ``test_gpu_rows_are_the_route_band_starts`` recomputes the set from the tables.
# fmt: off
_ROUTE_BAND_START_ROWS = {
    0: (1, 6, 8, 9, 16, 17, 32, 33, 192, 193, 257, 513),
    1: (
        1, 513, 768, 769, 1024, 1025, 1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145,
        8192, 8193, 12288, 12289, 16384,
    ),
    2: (
        1, 192, 193, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537, 2048, 2049, 3072,
        3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    3: (
        1, 9, 12, 13, 16, 17, 24, 25, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 192, 193, 256, 257,
        384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097,
        6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    4: (
        1, 5, 17, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537,
        2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145,
    ),
    5: (
        1, 5, 9, 12, 13, 16, 17, 33, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024,
        1025, 1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289,
        16384,
    ),
    6: (
        1, 2, 5, 17, 33, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536,
        1537, 2048, 2049, 3072, 3073, 4096, 4097,
    ),
    7: (
        1, 5, 17, 33, 65, 129, 192, 193, 256, 257, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537,
        2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 12288, 12289,
    ),
    8: (
        1, 3, 6, 7, 17, 65, 149, 153, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025,
        1536, 1537, 2048, 2049, 3072, 3073,
    ),
}
# fmt: on

# Every registered program is exercised at the edge of the band that selects it, plus a few
# measured cells and off-grid token counts.
GPU_ROWS = sorted(
    {(M, K) for K, starts in _ROUTE_BAND_START_ROWS.items() for M in starts}
    | {
        (16, 8),
        (64, 4),
        (256, 1),
        (512, 1),
        # round-2 cells: the pair kernel, the native K7 / K8 / M128 ports, the register-carry and
        # early + packed-add persistent programs, the per-PDL K1 M6144 cell; round 3: the sm_103a
        # K5 M192 / M256 / M768 and sm_100a K5 M192 ... M2048 cells of the native K5 port and the
        # twelve sm_100a K2 / K3 fence-authored cells (every one of them a band start above)
        (192, 4),
        (192, 6),
        (192, 7),
        (192, 8),
        (384, 8),
        (256, 5),
        (256, 7),
        (384, 4),
        (384, 5),
        (768, 2),
        (768, 5),
        (1024, 2),
        (1536, 1),
        (1536, 2),
        (2048, 2),
        (3072, 2),
        (4096, 2),
        (6144, 1),
        (6144, 2),
        (8192, 2),
        (12288, 2),
        # off-grid token counts (registered-variant fallback where this checkout lacks the exact program)
        (7, 5),
        (64, 6),
        (128, 7),
        (140, 4),
        (200, 8),
        (300, 3),
        (300, 5),
        (1000, 1),
        (1000, 2),
        (16384, 8),
    }
)


def _table_edge_token_counts(m_max: int = TOKEN_COUNTS[-1]) -> list[int]:
    """Every token count at which the route can change its kernel key, with its neighbours: the
    small / mid range, every integer literal of the planner (table edges, cells, minimum /
    maximum M, the balanced-grid bounds) and the SM counts.  A band can only start at one of
    these, so sweeping them finds every band start of M 1..``m_max``."""
    literals = {int(n) for n in re.findall(r"\b\d+\b", inspect.getsource(cb))}
    literals |= {int(n) for n in re.findall(r"\b\d+\b", json.dumps(ROUTES))}
    counts = set(range(1, 601)) | {m_max}
    for edge in literals | set(TOKEN_COUNTS) | set(SM_COUNTS):
        counts |= {edge - 1, edge, edge + 1}
    return sorted(M for M in counts if 1 <= M <= m_max)


def test_gpu_rows_are_the_route_band_starts():
    rows = set()
    token_counts = _table_edge_token_counts()
    for arch in ARCHES:
        for sm_count in SM_COUNTS:
            for K in range(MAX_BLOCKS + 1):
                for pdl in (False, True):
                    previous = None
                    for M in token_counts:
                        key = plan_route(arch, sm_count, M, K, pdl).kernel_key
                        if key != previous:
                            rows.add((M, K))
                            previous = key
    expected = {(M, K) for K, starts in _ROUTE_BAND_START_ROWS.items() for M in starts}
    assert rows == expected, (sorted(rows - expected), sorted(expected - rows))


# ---------------------------------------------------------------------------
# Device tests
# ---------------------------------------------------------------------------


def _device_arch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(
        tuple(torch.cuda.get_device_capability(device))
    )
    if arch is None:
        pytest.skip("SM100 / SM103 required")
    if not MODULES or arch not in KERNELS:
        pytest.skip("no generated programs registered for this architecture")
    return device, arch


def _make_inputs(
    M, K, *, has_delta=True, apply_output_norm=True, seed=31000, row_padding=0
):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def padded(shape):
        storage = torch.empty(
            (*shape[:-1], shape[-1] + row_padding), device="cuda", dtype=torch.bfloat16
        )
        return storage[..., : shape[-1]]

    prefix = padded((M, HIDDEN_SIZE)).uniform_(-1.0, 1.0, generator=g)
    blocks = padded((M, MAX_BLOCKS, HIDDEN_SIZE)).uniform_(-1.0, 1.0, generator=g)
    delta = (
        padded((M, HIDDEN_SIZE)).uniform_(-0.015625, 0.015625, generator=g)
        if has_delta
        else None
    )
    norm_weight = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16).normal_(
        1.0, 0.05, generator=g
    )
    qk_weight = torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16).normal_(
        0.0, 0.02, generator=g
    )
    output_norm_weight = (
        torch.empty(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16).normal_(
            1.0, 0.05, generator=g
        )
        if apply_output_norm
        else None
    )
    out = torch.full(
        (M, HIDDEN_SIZE), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    return dict(
        prefix=prefix,
        delta=delta,
        blocks=blocks,
        norm_weight=norm_weight,
        qk_weight=qk_weight,
        output_norm_weight=output_norm_weight,
        out=out,
        num_blocks=K,
    )


def _clone(inputs):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()}


def _check(inputs, expected):
    torch.testing.assert_close(
        inputs["out"].float(), expected["out"].float(), atol=ATOL, rtol=RTOL
    )
    assert torch.isfinite(inputs["out"].float()).all()
    assert torch.equal(
        inputs["prefix"].view(torch.int16), expected["prefix"].view(torch.int16)
    )
    assert torch.equal(
        inputs["blocks"].view(torch.int16), expected["blocks"].view(torch.int16)
    )


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("M,K", GPU_ROWS)
def test_matches_reference(M, K, pdl):
    device, arch = _device_arch()
    if not cb.generated_program_available(device, M, K, enable_pdl=pdl):
        pytest.skip("program not registered for this cell")
    inputs = _make_inputs(M, K)
    expected = _clone(inputs)
    reference_kimi_k3_attn_res(
        **{k: v for k, v in expected.items() if k != "num_blocks"}, num_blocks=K
    )
    read_only = {
        k: inputs[k].clone()
        for k in ("delta", "norm_weight", "qk_weight", "output_norm_weight")
    }
    runner = prepare_kimi_k3_attn_res(
        **{k: v for k, v in inputs.items() if k != "num_blocks"},
        num_blocks=K,
        enable_pdl=pdl,
    )
    assert runner.plan.kind in ("small_m", "native", "pair", "k0_tma", "persistent")
    assert runner.plan.kernel_key in KERNELS[arch]
    if runner.plan.fallback_from is not None:
        assert runner.plan.fallback_from not in KERNELS[arch]
    assert runner.launch() is inputs["out"]
    torch.cuda.synchronize()
    _check(inputs, expected)
    for k, v in read_only.items():
        assert torch.equal(inputs[k], v)


@pytest.mark.parametrize(
    "M,K,has_delta,write_idx,apply_output_norm",
    [
        (1, 0, False, 0, True),
        (17, 4, False, -1, True),
        (7, 8, True, -1, False),
    ],
)
def test_semantic_variants_take_the_bootstrap(
    M, K, has_delta, write_idx, apply_output_norm
):
    device, arch = _device_arch()
    if not cb.generated_program_available(
        device,
        M,
        K,
        common_path=False,
        has_delta=has_delta,
        block_write_idx=write_idx,
        apply_output_norm=apply_output_norm,
    ):
        pytest.skip("program not registered for this variant")
    inputs = _make_inputs(
        M, K, has_delta=has_delta, apply_output_norm=apply_output_norm
    )
    expected = _clone(inputs)
    reference_kimi_k3_attn_res(
        **{k: v for k, v in expected.items() if k != "num_blocks"},
        num_blocks=K,
        block_write_idx=write_idx,
    )
    runner = prepare_kimi_k3_attn_res(
        **{k: v for k, v in inputs.items() if k != "num_blocks"},
        num_blocks=K,
        block_write_idx=write_idx,
    )
    assert runner.plan.kind == "direct"
    runner.launch()
    torch.cuda.synchronize()
    _check(inputs, expected)


@pytest.mark.parametrize(
    "M,K",
    [
        (17, 4),
        (3, 7),
        (1, 1),
        (300, 4),
        (1024, 1),
        (1024, 4),
        (4096, 1),
        (1, 4),
        (4, 7),
        (16, 4),
        (16, 7),
    ],
)
@pytest.mark.parametrize("pdl", [False, True])
def test_snapshot_write_matches_reference_on_the_write_variants(M, K, pdl):
    """Round r4: the block-boundary snapshot write (delta + output norm, block K written) runs the
    write variant of the dense program; the written snapshot and the prefix are bit-exact, the
    output within tolerance, every other block byte preserved."""
    device, arch = _device_arch()
    if not cb.generated_program_available(
        device, M, K, enable_pdl=pdl, block_write_idx=K
    ):
        pytest.skip("small-M write program not registered for this cell")
    inputs = _make_inputs(M, K)
    assert common_path_eligible(
        **{k: v for k, v in inputs.items() if k != "num_blocks"},
        num_blocks=K,
        block_write_idx=K,
    )
    expected = _clone(inputs)
    reference_kimi_k3_attn_res(
        **{k: v for k, v in expected.items() if k != "num_blocks"},
        num_blocks=K,
        block_write_idx=K,
    )
    runner = prepare_kimi_k3_attn_res(
        **{k: v for k, v in inputs.items() if k != "num_blocks"},
        num_blocks=K,
        block_write_idx=K,
        enable_pdl=pdl,
    )
    assert runner.plan.kind in ("small_m", "persistent")
    assert runner.plan.kernel_key.endswith("_write")
    assert ".write1." in runner.plan.route_id
    runner.launch()
    torch.cuda.synchronize()
    _check(inputs, expected)
    assert torch.equal(
        inputs["blocks"][:, K, :].view(torch.int16), inputs["prefix"].view(torch.int16)
    )


def test_row_padded_layout_leaves_the_dense_path():
    # A row-padded operand leaves the dense path for the bootstrap.  Its variant with
    # delta and the output norm (direct:k4_d1_w0_n1) has no registered program, so
    # preparing the call raises NotImplementedError instead of launching.
    _device_arch()
    inputs = _make_inputs(5, 4, row_padding=64)
    kwargs = {k: v for k, v in inputs.items() if k != "num_blocks"}
    assert not common_path_eligible(**kwargs, num_blocks=4, block_write_idx=-1)
    with pytest.raises(NotImplementedError):
        prepare_kimi_k3_attn_res(**kwargs, num_blocks=4)


def test_launch_makes_no_allocation_and_replays_in_a_graph():
    device, arch = _device_arch()
    M, K = 64, 4
    if not cb.generated_program_available(device, M, K):
        pytest.skip("program not registered for this cell")
    inputs = _make_inputs(M, K)
    pristine = _clone(inputs)
    runner = prepare_kimi_k3_attn_res(
        **{k: v for k, v in inputs.items() if k != "num_blocks"}, num_blocks=K
    )
    runner.launch()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    runner.launch()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    # Graph replay follows the device contents of the bound tensors.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        runner.launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            runner.launch()
    torch.cuda.synchronize()
    inputs["prefix"].copy_(pristine["prefix"])
    inputs["blocks"].copy_(pristine["blocks"])
    inputs["out"].fill_(float("nan"))
    expected = _clone(pristine)
    reference_kimi_k3_attn_res(
        **{k: v for k, v in expected.items() if k != "num_blocks"}, num_blocks=K
    )
    graph.replay()
    torch.cuda.synchronize()
    _check(inputs, expected)


def test_prepare_rejects_bad_bindings():
    device, arch = _device_arch()
    inputs = _make_inputs(2, 4)
    with pytest.raises(ValueError):
        prepare_kimi_k3_attn_res(
            **{
                **{k: v for k, v in inputs.items() if k != "num_blocks"},
                "out": inputs["out"].float(),
            },
            num_blocks=4,
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_attn_res(
            **{k: v for k, v in inputs.items() if k != "num_blocks"}, num_blocks=9
        )
    with pytest.raises(ValueError):
        prepare_kimi_k3_attn_res(
            **{k: v for k, v in inputs.items() if k != "num_blocks"},
            num_blocks=4,
            block_write_idx=8,
        )
