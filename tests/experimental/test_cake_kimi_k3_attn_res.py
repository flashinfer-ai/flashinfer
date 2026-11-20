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
``tests.cake.test_helpers.cake_package``, and imports nothing else from flashinfer, so a
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

from tests.cake.test_helpers.cake_package import cake_package

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
# sm_100a (4096, 4) takes the held nc3 depth-3 K = 4 program; sm_103a (1024, 4) the early packed-add
# fast-math program at 152 SMs (round 26 moves its 148-SM cell, ``_ROUND_26_ROWS``); sm_103a (4096, 5)
# at 148 SMs the held relaxed-producer-wait program.  (Round 6 gave sm_103a (512, 4) at 152 SMs the
# early depth-3 program; round 32 moves it, ``_ROUND_32_ROWS``.)
_ROUND_6_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {
    ("sm_100a", 4096, 4, 148): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d3_vec128_fp32x2",
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
}

# Round 23: the POLICY_ROWS (256, 8) cells run the pair kernel at 384 threads: the ``consumer_regs``
# 232 program on sm_100a at 148 SMs, the parked-accumulator program on sm_103a at both SM counts.
# The sm_100a cell keeps the native K8 port at 152 SMs.
_ROUND_23_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {
    ("sm_100a", 256, 8, 148): (
        "pair",
        "pair_ws384_nc2_prefix_bf16_add_fastmath_consumer_regs232_grid148",
        148,
    ),
    ("sm_103a", 256, 8, 148): (
        "pair",
        "pair_ws384_nc2_prefix_bf16_add_fastmath_consumer_regs200_park_acc1_grid148",
        148,
    ),
    ("sm_103a", 256, 8, 152): (
        "pair",
        "pair_ws384_nc2_prefix_bf16_add_fastmath_consumer_regs200_park_acc1_grid152",
        152,
    ),
}

# Round 26: the sm_103a (512, 4) cell at 148 SMs (round 24: the ``consumer_regs`` 232 pair program)
# and the sm_103a (1024, 4) cell at 148 SMs (round 6: the held packed-add depth-3 program) run the
# packed register carry of the early packed-add fast-math nc3 depth-2 program (``_rcp``).
_ROUND_26_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {
    ("sm_103a", 512, 4, 148): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d2_vec128_fp32x2_early_consume_prefix_bf16_add_packed_inputs_prefix_shared_addr_packed_register_carry",
        128,
    ),
    ("sm_103a", 1024, 4, 148): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d2_vec128_fp32x2_early_consume_prefix_bf16_add_packed_inputs_prefix_shared_addr_packed_register_carry",
        148,
    ),
}

# Round 32: the sm_103a (512, 4) cell at 152 SMs runs the packed register carry program of its
# 148-SM cell (``_ROUND_26_ROWS``).  The round-32 ``_clc`` cells are ``_ROUND_32_CLC_CELLS_152``.
_ROUND_32_ROWS: dict[tuple[str, int, int, int], tuple[str, str, int]] = {
    ("sm_103a", 512, 4, 152): (
        "persistent",
        "trtllm_persistent_ws288_nc3_d2_vec128_fp32x2_early_consume_prefix_bf16_add_packed_inputs_prefix_shared_addr_packed_register_carry",
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
    kind, schedule_id, grid_x = (
        _ROUND_2_ROWS
        | _ROUND_5_ROWS
        | _ROUND_6_ROWS
        | _ROUND_23_ROWS
        | _ROUND_26_ROWS
        | _ROUND_32_ROWS
    ).get((arch, M, K, sm_count), (kind, schedule_id, grid_x))
    if (sm_count == SM_COUNT and (M, K) in _CLC_CELLS[arch]) or (
        sm_count == 152
        and arch == "sm_103a"
        and (M, K) in _ROUND_32_34_35_CLC_CELLS_152
    ):
        # rounds 28 / 29 / 32: the cell runs the token-queue build (``_clc``) of its program, one
        # CTA per token
        schedule_id, grid_x = f"{schedule_id}_clc_token_queue", M
    plan = plan_route(arch, sm_count, M, K, pdl)
    assert plan.kind == kind
    # Round r5: --use_fast_math is a program axis of the small-M / persistent programs (schedule id
    # ``_fastmath``, kernel key ``_fm``); the rows pin the program apart from that axis.  The pair
    # kernel is always fast-math and spells it ahead of its register and grid suffixes.
    # Round 35: a fast-math token-queue program spells ``_fastmath`` before ``_clc_token_queue``.
    # Rounds 38 and 39: a ``_LEVER_CELLS`` cell runs the lever build of the row's program, whose
    # label adds the lever's (test_lever_cells_take_their_lever_programs checks it exactly).
    lever = _lever_cell(arch, sm_count, M, K, False, plan.kernel_key)[1]
    token_queue = plan.schedule_id.endswith("_clc_token_queue")
    program_schedule = plan.schedule_id.removesuffix(lever).removesuffix(
        "_clc_token_queue"
    )
    assert (
        program_schedule.removesuffix("_fastmath")
        + ("_clc_token_queue" if token_queue else "")
        == schedule_id
    )
    assert ("_fastmath" in plan.schedule_id) is ("_fm" in plan.kernel_key)
    # A round-6 / round-26 exact cell carries its fast-math axis in its table key, not in the
    # fast-math bands.
    if (
        kind in ("small_m", "persistent")
        and (arch, M, K, sm_count)
        not in _ROUND_6_ROWS | _ROUND_26_ROWS | _ROUND_32_ROWS
    ):
        assert program_schedule.endswith("_fastmath") is cb._fast_math(
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
        # round 35: the (1536, 2) cell runs the token-queue build of the held program
        (1536, 2, False),
        (2048, 2, False),
        (2049, 2, False),
        (3072, 2, False),
        (4096, 2, False),
        (4097, 2, False),
        (6144, 2, False),
        # K4: the nc5 band M513..1023 + the pow2 cells {2048, 4096}; held at 1025..2047 and above 4096.
        # Round 24 ran the (512, 4) cell on the ``consumer_regs`` 232 pair program; round 26 runs it
        # on the early packed register-carry program
        # (``test_packed_register_carry_cells_take_the_packed_programs``).
        (513, 4, True),
        # round 23: the (768, 4) cell runs the ``consumer_regs`` 232 pair program
        # (``test_consumer_regs_pair_cells_take_the_384_thread_programs``); its band neighbours keep
        # the early-release program
        (767, 4, True),
        (769, 4, True),
        (1023, 4, True),
        # round 26: the early packed register-carry program at 148 SMs (round 6: the held packed-add
        # depth-3 program)
        (1024, 4, True),
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
        # round 35: the token-queue build of the held program at 152 SMs too
        assert at_152.kernel_key == "persistent:k2_nc4_d2_f011010000000000_clc"


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


def _write_key(key):
    """The key of the snapshot-write variant of dense key ``key``: ``_write`` goes before the
    ``_clc[_cla][_fpf][_fpl][_fpw][r<R>][_tokb]`` tail (round 29; rounds 37 and 38 add the lever
    suffixes), so the write variant of a token-queue program is ``..._write_clc`` and that of an
    ``_fplr2`` program ``..._write_fplr2``."""
    tail = re.search(
        r"(_clc(_cla)?(_fpf)?)?(_fpl)?(_fpw)?((?<=_fp[lw])r[1-9][0-9]*)?(_tokb)?$", key
    ).group(0)
    return f"{key.removesuffix(tail)}_write{tail}" if tail else f"{key}_write"


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
                # (a ``_clc`` program appends ``_clc_token_queue``)
                # (a round-38 or round-39 lever program appends its lever's label)
                lever = _lever_cell(arch, sm_count, M, K, False, key)[1]
                schedule = (
                    plan.schedule_id.removesuffix(lever)
                    .split("_fastmath")[0]
                    .removesuffix("_clc_token_queue")
                )
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
            assert plan.kernel_key == _write_key(key), (
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
            # snapshot-write variant of the program their dense cell runs; round 29: so do the
            # sm_100a (4096, 1), sm_103a (4096, 1) write cells at 148 SMs (``_clc``); rounds 33 and
            # 34: so do the token-queue write cells whose dense cell runs the same ``_clc`` program.
            # Round 38 builds the sm_103a (256, 7) cells at 148 SMs with ``_fplr3`` (dense) and
            # ``_fplr2`` (write), so that write cell leaves its dense cell's program; round 39 does
            # the same to the sm_100a (256, 7) cells at 148 SMs and builds both sm_100a (256, 4)
            # cells at 148 SMs with ``_fplr3``, so that write cell runs its dense cell's program.
            on_dense = (arch, sm_count, M, K) in {
                ("sm_100a", 148, 256, 4),
                ("sm_100a", 148, 4096, 1),
                ("sm_100a", 148, 4096, 7),
                ("sm_103a", 148, 4096, 1),
                ("sm_103a", 148, 4096, 7),
                ("sm_103a", 152, 4096, 1),
                ("sm_103a", 152, 4096, 7),
            }
            assert (dense.kernel_key == key) is on_dense, (dense, key)
            seen += 1
    # Rounds 6 and 17 add the (256, 4), (256, 7), (1024, 1), (1024, 4) and (1024, 7) write cells
    # (sm_100a: (256, 7), (1024, 1), (1024, 4) and (1024, 7); sm_103a: (256, 4), (256, 7) and
    # (1024, 7)); round 19 the sm_103a held-release (``_crheld``) K1 cells (1024, 1) at 148 SMs and
    # (4096, 1) at 152 SMs, one key per PDL mode (off, on); round 29 the token-queue (``_clc``)
    # write cells sm_100a [(4096, 1)], sm_103a [(4096, 1)] at 148 SMs; round 33 the token-queue
    # write cells sm_100a [(4096, 4), (4096, 7)] and sm_103a [(4096, 4), (4096, 7)] at 148 SMs and
    # sm_103a [(4096, 1), (4096, 4), (4096, 7)] at 152 SMs ((4096, 1) leaves its held program);
    # round 34 the token-queue write cells sm_100a [(1024, 4)] at 148 SMs and sm_103a [(1024, 1)]
    # at 152 SMs; round 37 builds the sm_103a held K1 cell (1024, 1) at 148 SMs with ``_fpl``;
    # round 38 builds the sm_103a write cells (256, 4), (256, 7) at 148 SMs and (256, 7) at 152
    # SMs with ``_fplr2``; round 39 the sm_100a write cells (256, 4) (``_fplr3``) and (256, 7)
    # (``_fplr2``) and the sm_103a token-queue write cell (1024, 4) (``_fplr3``) at 148 SMs.
    assert list(_r2_cells("WRITE_PERSISTENT_KEY_CELLS")) == [
        ("sm_100a", 148, 256, 4, "persistent:k4_nc3_d2_f110000000000000_fm_fplr3"),
        ("sm_100a", 148, 256, 7, "persistent:k7_nc4_d2_f111000000000000_fm_fplr2"),
        ("sm_100a", 148, 1024, 1, "persistent:k1_nc4_d2_f110000000000000_clc"),
        ("sm_100a", 148, 1024, 4, "persistent:k4_nc3_d2_f111000000000000_clc"),
        ("sm_100a", 148, 1024, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        ("sm_100a", 148, 4096, 1, "persistent:k1_nc4_d2_f110000000000000_clc"),
        ("sm_100a", 148, 4096, 4, "persistent:k4_nc3_d2_f111000000000000_clc"),
        ("sm_100a", 148, 4096, 7, "persistent:k7_nc4_d2_f110000000000000_clc"),
        ("sm_103a", 148, 256, 4, "persistent:k4_nc3_d2_f111000000000000_fm_fplr2"),
        ("sm_103a", 148, 256, 7, "persistent:k7_nc4_d2_f111000000000000_fm_fplr2"),
        (
            "sm_103a",
            148,
            1024,
            1,
            [
                "persistent:k1_nc4_d3_f010000000000010_crheld_fpl",
                "persistent:k1_nc4_d3_f010000000100000_crheld_fpl",
            ],
        ),
        ("sm_103a", 148, 1024, 4, "persistent:k4_nc3_d2_f111000000000000_clc_fplr3"),
        ("sm_103a", 148, 1024, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        (
            "sm_103a",
            148,
            4096,
            1,
            [
                "persistent:k1_nc4_d2_f110000000000010_clc",
                "persistent:k1_nc4_d2_f110000000100000_clc",
            ],
        ),
        ("sm_103a", 148, 4096, 4, "persistent:k4_nc3_d2_f111000000000000_clc"),
        ("sm_103a", 148, 4096, 7, "persistent:k7_nc4_d2_f110000000000000_clc"),
        ("sm_103a", 152, 256, 4, "persistent:k4_nc3_d2_f111000000000000_fm"),
        ("sm_103a", 152, 256, 7, "persistent:k7_nc4_d2_f110000000000000_fm_fplr2"),
        (
            "sm_103a",
            152,
            1024,
            1,
            [
                "persistent:k1_nc4_d2_f110000000000010_clc",
                "persistent:k1_nc4_d2_f110000000100000_clc",
            ],
        ),
        ("sm_103a", 152, 1024, 4, "persistent:k4_nc3_d2_f111000000000000_clc"),
        ("sm_103a", 152, 1024, 7, "persistent:k7_nc4_d2_f111000000000000_fm"),
        (
            "sm_103a",
            152,
            4096,
            1,
            [
                "persistent:k1_nc4_d2_f110000000000010_clc",
                "persistent:k1_nc4_d2_f110000000100000_clc",
            ],
        ),
        ("sm_103a", 152, 4096, 4, "persistent:k4_nc3_d2_f111000000000000_clc"),
        ("sm_103a", 152, 4096, 7, "persistent:k7_nc4_d2_f110000000000000_clc"),
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
    # Round 29: a ``_CLC_CELLS`` cell runs the token-queue build (``_clc``) of the pinned program.
    routed = (6144, 1) in _CLC_CELLS["sm_103a"]
    for sm_count in SM_COUNTS:
        # Round 32: the cell runs the ``_clc`` build at 152 SMs too.
        clc = (
            "_clc"
            if (routed and sm_count == SM_COUNT)
            or (sm_count == 152 and (6144, 1) in _ROUND_32_CLC_CELLS_152)
            else ""
        )
        off = plan_route("sm_103a", sm_count, 6144, 1, False).kernel_key
        on = plan_route("sm_103a", sm_count, 6144, 1, True).kernel_key
        assert (off, on) == (
            f"persistent:k1_nc4_d2_f110000000000010{clc}",
            f"persistent:k1_nc4_d2_f110000000000000{clc}",
        )
    # Cells round 2 left alone keep the legacy program: the GB300 K6 M3 small-M cell (the B300 K6
    # M192 cell runs the parked-accumulator pair program,
    # ``test_park_acc_pair_cells_take_the_384_thread_programs``). Round 6 moved the record-only K2
    # M8192 (152) / M4096 (148) cells to the K2 register-carry programs without the consumed-release
    # fence.
    assert (
        plan_route("sm_103a", 152, 3, 6, False).kernel_key == "small_m_cluster2:k6_fm"
    )
    clc = "_clc" if (8192, 2) in _ROUND_32_CLC_CELLS_152 else ""
    assert (
        plan_route("sm_103a", 152, 8192, 2, False).kernel_key
        == f"persistent:k2_nc3_d2_f011010000000000_crfnone{clc}"
    )
    clc = "_clc" if (4096, 2) in _CLC_CELLS["sm_103a"] else ""
    assert (
        plan_route("sm_103a", 148, 4096, 2, False).kernel_key
        == f"persistent:k2_nc3_d2_f011010000000000_fm_crfnone{clc}"
    )
    # Round 3: the sm_100a K5 cells M = 256 / 384 / 1024 / 1536 / 2048 run the native K5 port (one
    # CTA per SM) at 148 SMs and both PDL modes; M = 384 on sm_103a kept its round-2 persistent key
    # at 152 SMs until round 33 moved it to the held K5 packed register carry program of (1024, 5)
    # (round 6 gives 148 SMs an nc3 fast-math program).  Round 20 moves the M = 192 cells
    # of both architectures and round 23 the sm_103a M = 256 / 768 cells and the sm_100a M = 768
    # cell to the ``_cr232`` pair program
    # (``test_consumer_regs_pair_cells_take_the_384_thread_programs``); round 23 gives the sm_100a
    # M = 512 cell the packed register carry of its early-release program (``_rcp``).
    for sm_count in SM_COUNTS:
        for M in (256, 768):
            for pdl in (False, True):
                plan = plan_route("sm_103a", sm_count, M, 5, pdl)
                assert (plan.kind, plan.kernel_key, plan.grid_x) == (
                    "pair",
                    "pair:k5_nc2_pbf16a_fm_cr232",
                    sm_count,
                ), (sm_count, M, pdl, plan)
        # round 38: both SM counts run the ``_fplr3`` build of the (384, 5) program
        assert plan_route("sm_103a", sm_count, 384, 5, False).kernel_key == (
            "persistent:k5_nc3_d2_f111000000000000_fm_fplr3"
            if sm_count == 148
            else "persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp_fplr3"
        )
    for M in (256, 384, 1024, 1536, 2048):
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
        plan_route("sm_100a", 148, 768, 5, False).kernel_key
        == "pair:k5_nc2_pbf16a_fm_cr232"
    )
    # round 39: the ``_fplr3`` build of the (512, 5) program
    assert (
        plan_route("sm_100a", 148, 512, 5, False).kernel_key
        == "persistent:k5_nc4_d2_f111000000000000_rcp_fplr3"
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
        token_queue = (M, K) in _CLC_CELLS["sm_100a"]
        for pdl in (False, True):
            plan = plan_route("sm_100a", 148, M, K, pdl)
            assert (plan.kind, plan.kernel_key, plan.fallback_from) == (
                "persistent",
                f"persistent:k{K}_nc4_d2_f{flags}_crfnone"
                + ("_rcp" if packed else "")
                + ("_clc" if token_queue else ""),
                None,
            ), (M, K, pdl, plan)
            schedule = cb._persistent_schedule_id(
                4, 2, flags, True, packed_register_carry=packed
            )
            assert plan.schedule_id == schedule + (
                "_clc_token_queue" if token_queue else ""
            )
            assert (plan.grid_x, plan.threads) == (
                M if token_queue else cb._grid(M, 148, 4)[0],
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
        # round 35: the token-queue builds of these programs at two more cells
        (148, 2048, 2): "persistent:k2_nc3_d2_f011010000000000_fm_crfnone",
        (152, 16384, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
    }
    for sm_count in SM_COUNTS:
        for M in (2048, 3072, 4096, 6144, 8192, 12288, 16384):
            for K in (2, 3):
                key = plan_route("sm_103a", sm_count, M, K, False).kernel_key
                if (sm_count == SM_COUNT and (M, K) in _CLC_CELLS["sm_103a"]) or (
                    sm_count == 152 and (M, K) in _ROUND_32_34_35_CLC_CELLS_152
                ):
                    assert key.endswith("_clc"), (sm_count, M, K, key)
                    key = key.removesuffix("_clc")
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
        # Round 23: no sm_103a cell runs the native K5 port; it ships for sm_100a only.  Round 29:
        # a fence-authored program whose every cell runs its ``_clc`` build need not ship.
        assert "native_k5" not in KERNELS["sm_103a"]
        assert MODULES[KERNELS["sm_100a"]["native_k5"]]["arches"] == ["sm_100a"]
        for key in sorted(
            {
                f"persistent:k{K}_nc4_d2_f{flags}_crfnone"
                + ("_rcp" if (M, K) in sm100_packed else "")
                for (M, K), flags in sm100_fence_free.items()
                if (M, K) not in _CLC_CELLS["sm_100a"]
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
    PDL modes (round 23: with the packed register carry, ``_rcp``), and the held K1 write cell keeps
    the label ahead of ``_write`` (round 33 moves the 152-SM (4096, 1) write cell to a token-queue
    program, ``_ROUND_33_CLC_WRITE_CELLS``)."""
    for sm_count in SM_COUNTS:
        for pdl in (False, True):
            plan = plan_route("sm_103a", sm_count, 1024, 5, pdl)
            # Round 39: at 148 SMs the cell runs the ``_fplr2`` build of the held program
            key, lever = _lever_cell(
                "sm_103a", sm_count, 1024, 5, False, plan.kernel_key
            )
            assert key == "persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp", (
                sm_count,
                pdl,
                plan,
            )
            assert plan.schedule_id.removesuffix(lever).endswith(
                "_held_release_packed_register_carry_fastmath"
            ), plan
            assert "_early_consume" not in plan.schedule_id, plan
    # Round 37: the held K1 write cell runs the ``_fpl`` build of its program, whose label sits
    # between ``_held_release`` and ``_write`` (``_ROUND_37_CELLS``).
    for pdl in (False, True):
        plan = plan_route("sm_103a", 148, 1024, 1, pdl, block_write_idx=1)
        assert plan.kernel_key.endswith("_crheld_write_fpl"), (pdl, plan)
        assert plan.schedule_id.endswith(
            "_held_release_first_chunk_prefetch_l2_lanes_write"
        ), plan


def test_packed_register_carry_cells_take_the_packed_programs():
    """Round 20: a ``_rcp`` key names the register-carry program of the same key that keeps each
    token's bf16 source words in registers instead of TMEM; its schedule label adds
    ``_packed_register_carry`` ahead of ``_fastmath``.  The K = 3 cells sm_100a (256, 3), (1536, 3)
    and (2048, 3) and sm_103a (256, 3), (512, 3) and (1536, 3) run it at both PDL modes (sm_103a at
    both SM counts); round 23 adds the K = 4 / 5 cells sm_100a (1536, 4) and (512, 5) and sm_103a
    (1024, 5) at both SM counts; round 25 adds the sm_103a (384, 4) cell at 148 SMs; round 26 adds
    the sm_103a (512, 4), (1024, 4) and (512, 5) cells at 148 SMs; round 31 the sm_103a (384, 4)
    cell at 152 SMs; round 32 the sm_103a (384, 3) and (512, 4) cells at 152 SMs; round 33 the
    sm_103a (384, 5) and (512, 5) cells at 152 SMs; round 35 moves the sm_103a (1536, 3) cells to a
    token-queue program."""
    cells = (
        [("sm_100a", 148, M, 3) for M in (256, 1536, 2048)]
        + [("sm_103a", sm_count, M, 3) for sm_count in SM_COUNTS for M in (256, 512)]
        + [("sm_100a", 148, 1536, 4), ("sm_100a", 148, 512, 5)]
        + [("sm_103a", sm_count, 1024, 5) for sm_count in SM_COUNTS]
        + [("sm_103a", sm_count, 384, 4) for sm_count in SM_COUNTS]
        + [("sm_103a", 148, M, 4) for M in (512, 1024)]
        + [("sm_103a", 148, 512, 5)]
        + [("sm_103a", 152, 384, 3), ("sm_103a", 152, 512, 4)]
        + [("sm_103a", 152, 384, 5), ("sm_103a", 152, 512, 5)]
    )
    for arch, sm_count, M, K in cells:
        # Round 37: the sm_103a (512, 3) cell at 148 SMs runs the ``_fpl`` build of the program
        fpl = (arch, sm_count, M, K) == ("sm_103a", 148, 512, 3)
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, K, pdl)
            # Rounds 38 and 39: a ``_LEVER_CELLS`` cell runs the lever build of the program
            key, lever = _lever_cell(arch, sm_count, M, K, False, plan.kernel_key)
            assert plan.kind == "persistent" and plan.fallback_from is None, plan
            assert key.endswith("_rcp_fpl" if fpl else "_rcp"), (
                arch,
                sm_count,
                M,
                K,
                pdl,
                plan,
            )
            assert "_packed_register_carry" in plan.schedule_id, plan
            assert (
                plan.schedule_id.removesuffix(lever)
                .removesuffix("_first_chunk_prefetch_l2_lanes" if fpl else "")
                .endswith(
                    "_packed_register_carry_fastmath"
                    if "_fm_" in plan.kernel_key
                    else "_packed_register_carry"
                )
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


# Round 28: the cells that run the cluster-launch-control token-queue programs (``_clc``) at 148 SMs
# (B200 / B300): M 4096, 6144 and 8192 for K 6, 7 and 8, less the sm_100a (8192, 6) cell; their keys
# and schedule labels.
_CLC_CELLS_148 = frozenset((M, K) for M in (4096, 6144, 8192) for K in (6, 7, 8))
_CLC_KEYS = {
    6: "persistent:k6_nc4_d2_f110000000000000_clc",
    7: "persistent:k7_nc4_d2_f110000000000000_clc",
    8: "persistent:k8_nc5_d2_f110000000000000_clc",
}
_CLC_SCHEDULES = {
    6: "trtllm_persistent_ws288_nc4_d2_vec128_fp32x2_early_consume_clc_token_queue",
    7: "trtllm_persistent_ws288_nc4_d2_vec128_fp32x2_early_consume_clc_token_queue",
    8: "trtllm_persistent_ws288_nc5_d2_vec128_fp32x2_early_consume_clc_token_queue",
}
# Round 29: more 148-SM cells run the ``_clc`` build of the program they ran before.  Per
# architecture, (M, K) -> the key of that program (a pair: the keys at PDL off and on).  A dense
# cell runs the key + ``_clc``: the round-28 programs on more cells, and new token-queue builds of
# other programs.  A snapshot-write cell (block K written; ``WRITE_PERSISTENT_KEY_CELLS`` names the
# dense key + ``_clc``) names its write program and runs ``..._write_clc``.
_ROUND_29_CLC_CELLS = {
    "sm_100a": {
        (3072, 6): "persistent:k6_nc4_d2_f110000000000000",
        (3072, 7): "persistent:k7_nc4_d2_f111000000000000_fm",
        (3072, 8): "persistent:k8_nc5_d2_f110000000000000",
        (4096, 1): "persistent:k1_nc4_d2_f110000000000000",
        (4096, 2): "persistent:k2_nc4_d2_f110000000000000_crfnone",
        (4096, 4): "persistent:k4_nc3_d3_f010000000000000",
        (6144, 1): "persistent:k1_nc4_d2_f110000000000000",
        (6144, 2): "persistent:k2_nc4_d2_f110000000000000_crfnone",
        (6144, 3): "persistent:k3_nc4_d2_f010000000000000_crfnone",
        (6144, 4): "persistent:k4_nc3_d3_f110000000000000",
        (6144, 5): "persistent:k5_nc4_d3_f010000000000000",
        (8192, 1): "persistent:k1_nc2_d2_f110000000000000",
        (8192, 2): "persistent:k2_nc4_d2_f110000000000000_crfnone",
        (8192, 3): "persistent:k3_nc4_d2_f010000000000000_crfnone",
        (8192, 4): "persistent:k4_nc3_d3_f110000000000000",
        (8192, 5): "persistent:k5_nc4_d3_f010000000000000",
        (8192, 6): "persistent:k6_nc4_d2_f110000000000000",
        (12288, 1): "persistent:k1_nc4_d2_f110000000000000",
        (12288, 2): "persistent:k2_nc4_d2_f011000000000000_crfnone",
        (12288, 3): "persistent:k3_nc4_d2_f010000000000000_crfnone",
        (12288, 7): "persistent:k7_nc4_d2_f110000000000000",
        (16384, 1): "persistent:k1_nc4_d2_f110000000000000",
        (16384, 2): "persistent:k2_nc4_d2_f011000000000000_crfnone",
        (16384, 3): "persistent:k3_nc4_d2_f010000000000000_crfnone",
    },
    "sm_103a": {
        (3072, 6): "persistent:k6_nc4_d2_f110000000000000",
        (3072, 8): "persistent:k8_nc5_d2_f110000000000000",
        (4096, 1): (
            "persistent:k1_nc4_d2_f110000000000010",
            "persistent:k1_nc4_d2_f110000000100000",
        ),
        (4096, 2): "persistent:k2_nc3_d2_f011010000000000_fm_crfnone",
        (4096, 3): "persistent:k3_nc4_d2_f011000000000000",
        (4096, 4): "persistent:k4_nc3_d3_f110000000000000",
        (4096, 5): "persistent:k5_nc4_d3_f000000000000000",
        (6144, 1): (
            "persistent:k1_nc4_d2_f110000000000010",
            "persistent:k1_nc4_d2_f110000000000000",
        ),
        (6144, 2): "persistent:k2_nc4_d2_f011010000000000",
        (6144, 3): "persistent:k3_nc4_d2_f011000000000000",
        (6144, 4): "persistent:k4_nc5_d2_f011001000000000_fm",
        (8192, 1): (
            "persistent:k1_nc4_d2_f110000000000010",
            "persistent:k1_nc4_d2_f110000000100000",
        ),
        (8192, 2): "persistent:k2_nc4_d2_f011010000000000",
        (8192, 3): "persistent:k3_nc4_d2_f011000000000000",
        (8192, 5): "persistent:k5_nc3_d3_f010000000000000",
        (12288, 1): (
            "persistent:k1_nc4_d2_f110000000000010_fm",
            "persistent:k1_nc4_d2_f110000000000000_fm",
        ),
        (12288, 2): "persistent:k2_nc4_d2_f011010000000000",
        (12288, 3): "persistent:k3_nc4_d2_f011000000000000",
        (12288, 5): "persistent:k5_nc4_d3_f010000000000000",
        (12288, 6): "persistent:k6_nc4_d2_f110000000000000",
        (12288, 7): "persistent:k7_nc4_d2_f110000000000000",
        (12288, 8): "persistent:k8_nc5_d2_f110000000000000",
        (16384, 1): (
            "persistent:k1_nc4_d2_f110000000000010",
            "persistent:k1_nc4_d2_f110000000000000",
        ),
        (16384, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
        (16384, 3): "persistent:k3_nc4_d2_f011000000000000",
        (16384, 5): "persistent:k5_nc4_d3_f010000000000000",
        (16384, 6): "persistent:k6_nc4_d2_f110000000000000",
        (16384, 7): "persistent:k7_nc4_d2_f110000000000000",
        (16384, 8): "persistent:k8_nc5_d2_f110000000000000",
    },
}
_ROUND_29_CLC_WRITE_CELLS = {
    "sm_100a": {
        (4096, 1): "persistent:k1_nc4_d2_f110000000000000_write",
    },
    "sm_103a": {
        (4096, 1): (
            "persistent:k1_nc4_d2_f110000000000010_write",
            "persistent:k1_nc4_d2_f110000000100000_write",
        ),
    },
}
# Round 32: more 148-SM cells run the ``_clc`` build of the program they ran before (B200: (4096, 5)
# and (12288, 6)), per architecture (M, K) -> the key of that program.
_ROUND_32_CLC_CELLS_148 = {
    "sm_100a": {
        (4096, 5): "persistent:k5_nc3_d3_f010000000000000",
        (12288, 6): "persistent:k6_nc4_d2_f110000000000000",
    },
    "sm_103a": {},
}
# Round 33: two more 148-SM cells run a ``_clc`` program (B200: (2048, 7), the build of its fast-math
# program; B300: (3072, 7), the round-28 K7 program), per architecture (M, K) -> the key of the
# program without the suffix.
_ROUND_33_CLC_CELLS_148 = {
    "sm_100a": {
        (2048, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
    },
    "sm_103a": {
        (3072, 7): "persistent:k7_nc4_d2_f110000000000000",
    },
}
# Round 33: snapshot-write cells that run the ``_write_clc`` build of a write program, per
# architecture and SM count, (M, K) -> that write program (a pair: the keys at PDL off and on):
# B200 and B300 (4096, 4) and (4096, 7) at 148 SMs, GB300 (4096, 1), (4096, 4) and (4096, 7).
_ROUND_33_CLC_WRITE_CELLS = {
    "sm_100a": {
        148: {
            (4096, 4): "persistent:k4_nc3_d2_f111000000000000_write",
            (4096, 7): "persistent:k7_nc4_d2_f110000000000000_write",
        },
        152: {},
    },
    "sm_103a": {
        148: {
            (4096, 4): "persistent:k4_nc3_d2_f111000000000000_write",
            (4096, 7): "persistent:k7_nc4_d2_f110000000000000_write",
        },
        152: {
            (4096, 1): (
                "persistent:k1_nc4_d2_f110000000000010_write",
                "persistent:k1_nc4_d2_f110000000100000_write",
            ),
            (4096, 4): "persistent:k4_nc3_d2_f111000000000000_write",
            (4096, 7): "persistent:k7_nc4_d2_f110000000000000_write",
        },
    },
}
# Round 34: more 148-SM cells run the ``_clc`` build of the program they ran before (B200: eleven
# cells; B300: sixteen cells, six of them on three new token-queue builds of fast-math programs),
# per architecture (M, K) -> the key of the program without the suffix (a pair: the keys at PDL
# off and on).
_ROUND_34_CLC_CELLS_148 = {
    "sm_100a": {
        (768, 6): "persistent:k6_nc4_d2_f110000000000000",
        (768, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
        (1536, 6): "persistent:k6_nc4_d2_f110000000000000",
        (1536, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
        (2048, 1): "persistent:k1_nc2_d2_f110000000000000",
        (2048, 4): "persistent:k4_nc3_d3_f110000000000000",
        (3072, 1): "persistent:k1_nc4_d2_f110000000000000",
        (3072, 2): "persistent:k2_nc4_d2_f110000000000000_crfnone",
        (3072, 3): "persistent:k3_nc4_d2_f010000000000000_crfnone",
        (3072, 4): "persistent:k4_nc3_d3_f110000000000000",
        (3072, 5): "persistent:k5_nc4_d3_f010000000000000",
    },
    "sm_103a": {
        (1024, 1): (
            "persistent:k1_nc4_d2_f110000000000010_fm",
            "persistent:k1_nc4_d2_f110000000100000_fm",
        ),
        (1024, 6): "persistent:k6_nc4_d2_f111000000000000_fm",
        (1536, 4): "persistent:k4_nc5_d2_f011001000000000_fm",
        (1536, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
        (2048, 1): (
            "persistent:k1_nc4_d2_f110000000000010_fm",
            "persistent:k1_nc4_d2_f110000000100000_fm",
        ),
        (2048, 3): "persistent:k3_nc4_d2_f011000000000000",
        (2048, 4): "persistent:k4_nc3_d3_f110000000000000",
        (2048, 5): "persistent:k5_nc4_d3_f010000000000000",
        (2048, 6): "persistent:k6_nc4_d2_f111000000000000_fm",
        (2048, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
        (3072, 1): (
            "persistent:k1_nc4_d2_f110000000000010_fm",
            "persistent:k1_nc4_d2_f110000000000000_fm",
        ),
        (3072, 2): "persistent:k2_nc4_d2_f011010000000000",
        (3072, 3): "persistent:k3_nc4_d2_f011000000000000",
        (3072, 4): "persistent:k4_nc5_d2_f011001000000000_fm",
        (3072, 5): "persistent:k5_nc3_d3_f010000000000000",
        (6144, 5): "persistent:k5_nc3_d3_f010000000000000",
    },
}
# Round 34: snapshot-write cells that run the ``_write_clc`` build of their write program, per
# architecture and SM count, (M, K) -> that write program (a pair: the keys at PDL off and on):
# B200 (1024, 4) at 148 SMs, GB300 (1024, 1) at 152 SMs.
_ROUND_34_CLC_WRITE_CELLS = {
    "sm_100a": {
        148: {
            (1024, 4): "persistent:k4_nc3_d2_f111000000000000_write",
        },
        152: {},
    },
    "sm_103a": {
        148: {},
        152: {
            (1024, 1): (
                "persistent:k1_nc4_d2_f110000000000010_write",
                "persistent:k1_nc4_d2_f110000000100000_write",
            ),
        },
    },
}
# Round 35: more 148-SM cells run a ``_clc`` program (B200: 7 cells; B300: 13 cells), the
# token-queue build of the program they ran before or of another program of their K, per
# architecture (M, K) -> the key of that program without the suffix (a pair: the keys at PDL
# off and on).
_ROUND_35_CLC_CELLS_148 = {
    "sm_100a": {
        (768, 1): "persistent:k1_nc4_d2_f110000000000000_fm",
        (768, 8): "persistent:k8_nc5_d2_f110000000000000",
        (1024, 1): "persistent:k1_nc2_d2_f110000000000000_fm",
        (1536, 1): "persistent:k1_nc4_d2_f110000000000000_fm",
        (1536, 8): "persistent:k8_nc5_d2_f110000000000000",
        (2048, 2): "persistent:k2_nc4_d2_f011010000000000",
        (2048, 6): "persistent:k6_nc4_d2_f111000000000000_fm",
    },
    "sm_103a": {
        (768, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
        (768, 8): "persistent:k8_nc5_d2_f110000000000000",
        (1024, 2): "persistent:k2_nc4_d2_f011010000000000",
        (1536, 1): (
            "persistent:k1_nc4_d2_f110000000000010_fm",
            "persistent:k1_nc4_d2_f110000000100000_fm",
        ),
        (1536, 2): "persistent:k2_nc4_d2_f011010000000000",
        (1536, 3): "persistent:k3_nc4_d2_f011000000000000",
        (1536, 5): "persistent:k5_nc4_d3_f010000000000000",
        (1536, 6): "persistent:k6_nc4_d2_f111000000000000_fm",
        (1536, 8): "persistent:k8_nc5_d2_f110000000000000",
        (2048, 2): "persistent:k2_nc3_d2_f011010000000000_fm_crfnone",
        (8192, 4): "persistent:k4_nc5_d2_f010001000000000",
        (12288, 4): "persistent:k4_nc5_d2_f010001000000000",
        (16384, 4): "persistent:k4_nc5_d2_f010001000000000",
    },
}
# Round 35: snapshot-write cells that run the ``_write_clc`` build of a write program, per
# architecture and SM count, (M, K) -> that write program: B200 (1024, 1) at 148 SMs, B300
# (1024, 4) at 148 SMs and GB300 (1024, 4) at 152 SMs.
_ROUND_35_CLC_WRITE_CELLS = {
    "sm_100a": {
        148: {
            (1024, 1): "persistent:k1_nc4_d2_f110000000000000_write",
        },
        152: {},
    },
    "sm_103a": {
        148: {
            (1024, 4): "persistent:k4_nc3_d2_f111000000000000_write",
        },
        152: {
            (1024, 4): "persistent:k4_nc3_d2_f111000000000000_write",
        },
    },
}
_CLC_CELLS = {
    "sm_100a": (_CLC_CELLS_148 - {(8192, 6)})
    | set(_ROUND_29_CLC_CELLS["sm_100a"])
    | set(_ROUND_32_CLC_CELLS_148["sm_100a"])
    | set(_ROUND_33_CLC_CELLS_148["sm_100a"])
    | set(_ROUND_34_CLC_CELLS_148["sm_100a"])
    | set(_ROUND_35_CLC_CELLS_148["sm_100a"]),
    "sm_103a": _CLC_CELLS_148
    | set(_ROUND_29_CLC_CELLS["sm_103a"])
    | set(_ROUND_32_CLC_CELLS_148["sm_103a"])
    | set(_ROUND_33_CLC_CELLS_148["sm_103a"])
    | set(_ROUND_34_CLC_CELLS_148["sm_103a"])
    | set(_ROUND_35_CLC_CELLS_148["sm_103a"]),
}
# Round 31: the sm_103a cells at 152 SMs (GB300) that run the ``_clc`` build of a round-28
# program, (M, K) -> the key of that program; (4096, 6), (4096, 7) and (6144, 7) ran the f111
# fast-math program before.
_ROUND_31_CLC_CELLS_152 = {
    (4096, 6): "persistent:k6_nc4_d2_f110000000000000",
    (4096, 7): "persistent:k7_nc4_d2_f110000000000000",
    (4096, 8): "persistent:k8_nc5_d2_f110000000000000",
    (6144, 6): "persistent:k6_nc4_d2_f110000000000000",
    (6144, 7): "persistent:k7_nc4_d2_f110000000000000",
    (6144, 8): "persistent:k8_nc5_d2_f110000000000000",
    (8192, 6): "persistent:k6_nc4_d2_f110000000000000",
    (8192, 7): "persistent:k7_nc4_d2_f110000000000000",
    (8192, 8): "persistent:k8_nc5_d2_f110000000000000",
}
# Round 32: the sm_103a cells at 152 SMs (GB300) that run the ``_clc`` build of the program they
# ran before, (M, K) -> the key of that program (a pair: the keys at PDL off and on).
_ROUND_32_CLC_CELLS_152 = {
    (4096, 1): (
        "persistent:k1_nc4_d2_f110000000000010",
        "persistent:k1_nc4_d2_f110000000100000",
    ),
    (4096, 2): "persistent:k2_nc4_d2_f011010000000000",
    (4096, 3): "persistent:k3_nc4_d2_f011000000000000",
    (4096, 4): "persistent:k4_nc3_d3_f110000000000000",
    (4096, 5): "persistent:k5_nc3_d3_f010000000000000",
    (6144, 1): (
        "persistent:k1_nc4_d2_f110000000000010",
        "persistent:k1_nc4_d2_f110000000000000",
    ),
    (6144, 2): "persistent:k2_nc4_d2_f011010000000000",
    (6144, 3): "persistent:k3_nc4_d2_f011000000000000",
    (6144, 4): "persistent:k4_nc5_d2_f011001000000000_fm",
    (8192, 1): (
        "persistent:k1_nc4_d2_f110000000000010",
        "persistent:k1_nc4_d2_f110000000100000",
    ),
    (8192, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
    (8192, 3): "persistent:k3_nc4_d2_f011000000000000",
    (8192, 5): "persistent:k5_nc3_d3_f010000000000000",
    (12288, 1): (
        "persistent:k1_nc4_d2_f110000000000010_fm",
        "persistent:k1_nc4_d2_f110000000000000_fm",
    ),
    (12288, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
    (12288, 3): "persistent:k3_nc4_d2_f011000000000000",
    (12288, 5): "persistent:k5_nc4_d3_f010000000000000",
    (12288, 6): "persistent:k6_nc4_d2_f110000000000000",
    (12288, 8): "persistent:k8_nc5_d2_f110000000000000",
    (16384, 1): (
        "persistent:k1_nc4_d2_f110000000000010",
        "persistent:k1_nc4_d2_f110000000000000",
    ),
    (16384, 3): "persistent:k3_nc4_d2_f011000000000000",
    (16384, 5): "persistent:k5_nc4_d3_f010000000000000",
    (16384, 6): "persistent:k6_nc4_d2_f110000000000000",
    (16384, 7): "persistent:k7_nc4_d2_f110000000000000",
    (16384, 8): "persistent:k8_nc5_d2_f110000000000000",
}
# Round 34: the sm_103a cells at 152 SMs (GB300) that run a ``_clc`` program at both PDL modes:
# the token-queue build of the program they ran before, or of the f110 program of their K
# ((M, K) -> the key of that program, a pair: the keys at PDL off and on).
_ROUND_34_CLC_CELLS_152 = {
    (768, 7): "persistent:k7_nc4_d2_f110000000000000",
    (768, 8): "persistent:k8_nc5_d2_f110000000000000",
    (1536, 6): "persistent:k6_nc4_d2_f110000000000000",
    (1536, 8): "persistent:k8_nc5_d2_f110000000000000",
    (2048, 3): "persistent:k3_nc4_d2_f011000000000000",
    (2048, 4): "persistent:k4_nc3_d3_f110000000000000",
    (2048, 5): "persistent:k5_nc3_d3_f010000000000000",
    (2048, 6): "persistent:k6_nc4_d2_f110000000000000",
    (2048, 7): "persistent:k7_nc4_d2_f110000000000000",
    (2048, 8): "persistent:k8_nc5_d2_f110000000000000",
    (3072, 1): (
        "persistent:k1_nc4_d2_f110000000000010_fm",
        "persistent:k1_nc4_d2_f110000000000000_fm",
    ),
    (3072, 2): "persistent:k2_nc4_d2_f011010000000000",
    (3072, 3): "persistent:k3_nc4_d2_f011000000000000",
    (3072, 4): "persistent:k4_nc5_d2_f011001000000000_fm",
    (3072, 5): "persistent:k5_nc3_d3_f010000000000000",
    (3072, 8): "persistent:k8_nc5_d2_f110000000000000",
    (6144, 5): "persistent:k5_nc4_d3_f010000000000000",
}
# Round 34: the sm_103a K1 cells at 152 SMs (GB300) that run the ``_clc`` build at PDL off only,
# (M, K) -> (the key of that program at PDL off, the key the cell keeps at PDL on).
_ROUND_34_CLC_CELLS_152_PDL_OFF = {
    (1024, 1): (
        "persistent:k1_nc4_d2_f110000000000010_fm",
        "persistent:k1_nc4_d2_f110000000100000_fm",
    ),
    (2048, 1): (
        "persistent:k1_nc4_d2_f110000000000010_fm",
        "persistent:k1_nc4_d2_f110000000100000_fm",
    ),
}
# Round 35: the sm_103a cells at 152 SMs (GB300) that run a ``_clc`` program at both PDL modes
# ((M, K) -> the key of that program without the suffix, a pair: the keys at PDL off and on).
# (768, 7) leaves the round-34 program for the fast-math one; the K1 cells (1024, 1) and
# (2048, 1) of ``_ROUND_34_CLC_CELLS_152_PDL_OFF`` now run the token-queue build at PDL on too.
_ROUND_35_CLC_CELLS_152 = {
    (768, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
    (1024, 1): (
        "persistent:k1_nc4_d2_f110000000000010_fm",
        "persistent:k1_nc4_d2_f110000000100000_fm",
    ),
    (1024, 6): "persistent:k6_nc4_d2_f111000000000000_fm",
    (1024, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
    (1536, 2): "persistent:k2_nc4_d2_f011010000000000",
    (1536, 3): "persistent:k3_nc4_d2_f011000000000000",
    (1536, 4): "persistent:k4_nc5_d2_f011001000000000_fm",
    (1536, 5): "persistent:k5_nc3_d3_f010000000000000",
    (1536, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
    (2048, 1): (
        "persistent:k1_nc4_d2_f110000000000010_fm",
        "persistent:k1_nc4_d2_f110000000100000_fm",
    ),
    (2048, 2): "persistent:k2_nc4_d2_f011010000000000",
    (3072, 6): "persistent:k6_nc4_d2_f111000000000000_fm",
    (3072, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
    (8192, 4): "persistent:k4_nc5_d2_f010001000000000",
    (12288, 4): "persistent:k4_nc5_d2_f010001000000000",
    (12288, 7): "persistent:k7_nc4_d2_f110000000000000_fm",
    (16384, 2): "persistent:k2_nc3_d2_f011010000000000_crfnone",
    (16384, 4): "persistent:k4_nc5_d2_f010001000000000",
}
# The cells at 152 SMs that rounds 32, 34 and 35 move to a ``_clc`` program at both PDL modes.
_ROUND_32_34_35_CLC_CELLS_152 = (
    _ROUND_32_CLC_CELLS_152.keys()
    | _ROUND_34_CLC_CELLS_152.keys()
    | _ROUND_35_CLC_CELLS_152.keys()
)


def _at_pdl(value, pdl):
    """A route-table value at one PDL mode (a pair holds the keys at PDL off and on)."""
    return value[int(pdl)] if isinstance(value, (list, tuple)) else value


def _clc_static_key(arch, M, K, pdl):
    """The key of the program whose ``_clc`` build the ``_CLC_CELLS`` cell runs at PDL ``pdl``."""
    if (M, K) in _ROUND_29_CLC_CELLS[arch]:
        return _at_pdl(_ROUND_29_CLC_CELLS[arch][(M, K)], pdl)
    if (M, K) in _ROUND_32_CLC_CELLS_148[arch]:
        return _at_pdl(_ROUND_32_CLC_CELLS_148[arch][(M, K)], pdl)
    if (M, K) in _ROUND_33_CLC_CELLS_148[arch]:
        return _at_pdl(_ROUND_33_CLC_CELLS_148[arch][(M, K)], pdl)
    if (M, K) in _ROUND_34_CLC_CELLS_148[arch]:
        return _at_pdl(_ROUND_34_CLC_CELLS_148[arch][(M, K)], pdl)
    if (M, K) in _ROUND_35_CLC_CELLS_148[arch]:
        return _at_pdl(_ROUND_35_CLC_CELLS_148[arch][(M, K)], pdl)
    return _CLC_KEYS[K].removesuffix("_clc")


def test_clc_cells_take_the_token_queue_programs():
    """Rounds 28 and 29: a ``_clc`` key names the program of the same key without the suffix built
    with cluster launch control token scheduling.  It launches one CTA per token (grid ``M``, grid
    policy ``clc_token_queue``); each resident CTA computes its own token, then the token of every
    pending CTA it cancels.  Its schedule label adds ``_clc_token_queue`` to its program's.  At 148
    SMs the ``_CLC_CELLS`` of each architecture run it at both PDL modes (round 28: the (4096, K),
    (6144, K) and (8192, K) cells for K 6, 7 and 8, less the sm_100a (8192, 6) cell; round 29:
    ``_ROUND_29_CLC_CELLS``), and the ``_ROUND_29_CLC_WRITE_CELLS`` snapshot writes run the
    ``_write_clc`` build of their write program; at 152 SMs the sm_103a
    ``_ROUND_31_CLC_CELLS_152`` run it (round 31) and the ``_ROUND_32_CLC_CELLS_152`` (round 32;
    round 32 also adds the ``_ROUND_32_CLC_CELLS_148``, round 33 the ``_ROUND_33_CLC_CELLS_148``);
    round 33 runs the ``_write_clc`` build of the ``_ROUND_33_CLC_WRITE_CELLS`` snapshot writes at
    148 and 152 SMs; round 34 adds the ``_ROUND_34_CLC_CELLS_148``, the ``_ROUND_34_CLC_CELLS_152``,
    the ``_ROUND_34_CLC_CELLS_152_PDL_OFF`` (at PDL off) and the ``_ROUND_34_CLC_WRITE_CELLS``;
    round 35 adds the ``_ROUND_35_CLC_CELLS_148``, the ``_ROUND_35_CLC_CELLS_152`` (over the
    round-34 entries of its cells) and the ``_ROUND_35_CLC_WRITE_CELLS``; no other cell, SM count
    or snapshot-write plan runs one."""

    def clc_cells(table_name):
        # (a lever build of a ``_clc`` program, ``_LEVER_CELLS``, is a token-queue program too)
        write = table_name == "WRITE_PERSISTENT_KEY_CELLS"
        return {
            (arch, sm_count, M, K)
            for arch, sm_count, M, K, value in _r2_cells(table_name)
            if any(
                _lever_cell(arch, sm_count, M, K, write, str(key))[0].endswith("_clc")
                for key in (value if isinstance(value, (list, tuple)) else (value,))
            )
        }

    assert clc_cells("R2_PERSISTENT_KEY_CELLS") == {
        (arch, 148, M, K) for arch, cells in _CLC_CELLS.items() for M, K in cells
    } | {
        ("sm_103a", 152, M, K)
        for M, K in _ROUND_31_CLC_CELLS_152.keys()
        | _ROUND_32_CLC_CELLS_152.keys()
        | _ROUND_34_CLC_CELLS_152.keys()
        | _ROUND_34_CLC_CELLS_152_PDL_OFF.keys()
        | _ROUND_35_CLC_CELLS_152.keys()
    }
    assert clc_cells("WRITE_PERSISTENT_KEY_CELLS") == {
        (arch, 148, M, K)
        for arch, cells in _ROUND_29_CLC_WRITE_CELLS.items()
        for M, K in cells
    } | {
        (arch, sm_count, M, K)
        for round_cells in (
            _ROUND_33_CLC_WRITE_CELLS,
            _ROUND_34_CLC_WRITE_CELLS,
            _ROUND_35_CLC_WRITE_CELLS,
        )
        for arch, by_sm in round_cells.items()
        for sm_count, cells in by_sm.items()
        for M, K in cells
    }
    for arch, cells in _CLC_CELLS.items():
        for M, K in sorted(cells):
            for pdl in (False, True):
                static_key = _clc_static_key(arch, M, K, pdl)
                static = cb._persistent_plan_for_key(arch, 148, M, K, pdl, static_key)
                plan = plan_route(arch, 148, M, K, pdl)
                assert plan.kind == "persistent" and plan.fallback_from is None, plan
                assert plan.kernel_key == f"{static_key}_clc", (arch, M, K, pdl, plan)
                assert plan.grid_x == M and plan.threads == cb.PERSISTENT_THREADS, plan
                assert plan.schedule_id == f"{static.schedule_id}_clc_token_queue", plan
                if (M, K) in _CLC_CELLS_148:
                    assert plan.schedule_id == _CLC_SCHEDULES[K], plan
                assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
                if MODULES:
                    assert MODULES[KERNELS[arch][plan.kernel_key]]["schedule_id"] == (
                        plan.schedule_id
                    ), plan
                    if (M, K) in _CLC_CELLS_148:
                        # the program without the suffix ships too (the cells beside these run it)
                        assert static_key in KERNELS[arch], plan
                others = [plan_route(arch, 148, M + 1, K, pdl)]
                if arch != "sm_103a" or (
                    (M, K) not in _ROUND_31_CLC_CELLS_152
                    and (M, K) not in _ROUND_32_CLC_CELLS_152
                    and (M, K) not in _ROUND_34_CLC_CELLS_152
                    and (pdl or (M, K) not in _ROUND_34_CLC_CELLS_152_PDL_OFF)
                    and (M, K) not in _ROUND_35_CLC_CELLS_152
                ):
                    others.append(plan_route(arch, 152, M, K, pdl))
                if (
                    K < MAX_BLOCKS
                    and (M, K) not in _ROUND_29_CLC_WRITE_CELLS[arch]
                    and (M, K) not in _ROUND_33_CLC_WRITE_CELLS[arch][148]
                    and (M, K) not in _ROUND_34_CLC_WRITE_CELLS[arch][148]
                    and (M, K) not in _ROUND_35_CLC_WRITE_CELLS[arch][148]
                ):
                    # the snapshot write of block K (K = MAX_BLOCKS has no block to write)
                    others.append(plan_route(arch, 148, M, K, pdl, block_write_idx=K))
                for other in others:
                    assert not other.kernel_key.endswith("_clc"), (arch, M, K, other)
                    assert "_clc_token_queue" not in other.schedule_id, other
                    if other.kind == "persistent":
                        assert other.grid_x < M, other
    clc_152 = {}
    for round_cells in (
        _ROUND_31_CLC_CELLS_152,
        _ROUND_32_CLC_CELLS_152,
        _ROUND_34_CLC_CELLS_152,
    ):
        assert not clc_152.keys() & round_cells.keys()
        clc_152 |= round_cells
    # Round 35 moves (768, 7) to another ``_clc`` program (its other cells are new here).
    assert clc_152.keys() & _ROUND_35_CLC_CELLS_152.keys() == {(768, 7)}
    clc_152 |= _ROUND_35_CLC_CELLS_152
    for (M, K), value in sorted(clc_152.items()):
        for pdl in (False, True):
            static_key = _at_pdl(value, pdl)
            static = cb._persistent_plan_for_key("sm_103a", 152, M, K, pdl, static_key)
            plan = plan_route("sm_103a", 152, M, K, pdl)
            assert plan.kind == "persistent" and plan.fallback_from is None, plan
            assert plan.kernel_key == f"{static_key}_clc", (M, K, pdl, plan)
            assert plan.grid_x == M and plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.schedule_id == f"{static.schedule_id}_clc_token_queue", plan
            if (M, K) in _ROUND_31_CLC_CELLS_152:
                assert plan.schedule_id == _CLC_SCHEDULES[K], plan
            assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
            if MODULES:
                assert MODULES[KERNELS["sm_103a"][plan.kernel_key]]["schedule_id"] == (
                    plan.schedule_id
                ), plan
                if (M, K) in _ROUND_31_CLC_CELLS_152:
                    assert static_key in KERNELS["sm_103a"], plan
            others = [
                plan_route("sm_103a", 152, M - 1, K, pdl),
                plan_route("sm_103a", 152, M + 1, K, pdl),
            ]
            if (
                K < MAX_BLOCKS
                and (M, K) not in _ROUND_33_CLC_WRITE_CELLS["sm_103a"][152]
                and (M, K) not in _ROUND_34_CLC_WRITE_CELLS["sm_103a"][152]
                and (M, K) not in _ROUND_35_CLC_WRITE_CELLS["sm_103a"][152]
            ):
                others.append(plan_route("sm_103a", 152, M, K, pdl, block_write_idx=K))
            for other in others:
                assert not other.kernel_key.endswith("_clc"), (M, K, other)
                assert "_clc_token_queue" not in other.schedule_id, other
                if other.kind == "persistent":
                    assert other.grid_x < M, other
    for arch, cells in _ROUND_29_CLC_WRITE_CELLS.items():
        for (M, K), value in sorted(cells.items()):
            for pdl in (False, True):
                write_key = _at_pdl(value, pdl)
                dense = cb._persistent_plan_for_key(
                    arch, 148, M, K, pdl, f"{write_key.removesuffix('_write')}_clc"
                )
                plan = plan_route(arch, 148, M, K, pdl, block_write_idx=K)
                assert plan.kind == "persistent" and plan.fallback_from is None, plan
                assert plan.kernel_key == f"{write_key}_clc", (arch, M, K, pdl, plan)
                assert (plan.grid_x, plan.threads) == (M, cb.PERSISTENT_THREADS), plan
                assert plan.schedule_id == f"{dense.schedule_id}_write", plan
                assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
                assert ".write1." in plan.route_id, plan
                if MODULES:
                    assert MODULES[KERNELS[arch][plan.kernel_key]]["schedule_id"] == (
                        plan.schedule_id
                    ), plan
                others = [plan_route(arch, 148, M + 1, K, pdl, block_write_idx=K)]
                if (M, K) not in _ROUND_33_CLC_WRITE_CELLS[arch][152]:
                    others.append(plan_route(arch, 152, M, K, pdl, block_write_idx=K))
                for other in others:
                    assert not other.kernel_key.endswith("_clc"), (arch, M, K, other)
                    assert "_clc_token_queue" not in other.schedule_id, other
    write_cells_33 = [
        (arch, sm_count, M, K, value)
        for arch, by_sm in _ROUND_33_CLC_WRITE_CELLS.items()
        for sm_count, cells in by_sm.items()
        for (M, K), value in sorted(cells.items())
    ]
    for arch, sm_count, M, K, value in write_cells_33:
        for pdl in (False, True):
            write_key = _at_pdl(value, pdl)
            dense_key = f"{write_key.removesuffix('_write')}_clc"
            dense = cb._persistent_plan_for_key(arch, sm_count, M, K, pdl, dense_key)
            plan = plan_route(arch, sm_count, M, K, pdl, block_write_idx=K)
            assert plan.kind == "persistent" and plan.fallback_from is None, plan
            assert plan.kernel_key == f"{write_key}_clc", (arch, M, K, pdl, plan)
            assert (plan.grid_x, plan.threads) == (M, cb.PERSISTENT_THREADS), plan
            assert plan.schedule_id == f"{dense.schedule_id}_write", plan
            assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
            assert ".write1." in plan.route_id, plan
            if MODULES:
                record = MODULES[KERNELS[arch][plan.kernel_key]]
                assert record["schedule_id"] == plan.schedule_id, plan
            other = plan_route(arch, sm_count, M + 1, K, pdl, block_write_idx=K)
            assert not other.kernel_key.endswith("_clc"), (arch, M, K, other)
            assert "_clc_token_queue" not in other.schedule_id, other
    # Round 34: the GB300 K1 cells that run the ``_clc`` build at PDL off keep their program at PDL on
    # (round 35 moves them to the token-queue build at PDL on too: the 152-SM loop checks them).
    pdl_off_cells = sorted(
        (cell, keys)
        for cell, keys in _ROUND_34_CLC_CELLS_152_PDL_OFF.items()
        if cell not in _ROUND_35_CLC_CELLS_152
    )
    for (M, K), (static_key, kept_key) in pdl_off_cells:
        static = cb._persistent_plan_for_key("sm_103a", 152, M, K, False, static_key)
        plan = plan_route("sm_103a", 152, M, K, False)
        assert plan.kind == "persistent" and plan.fallback_from is None, plan
        assert plan.kernel_key == f"{static_key}_clc", (M, K, plan)
        assert plan.grid_x == M and plan.threads == cb.PERSISTENT_THREADS, plan
        assert plan.schedule_id == f"{static.schedule_id}_clc_token_queue", plan
        assert plan.route_id.endswith(".pdl0.clc_token_queue"), plan
        if MODULES:
            assert MODULES[KERNELS["sm_103a"][plan.kernel_key]]["schedule_id"] == (
                plan.schedule_id
            ), plan
        kept = plan_route("sm_103a", 152, M, K, True)
        assert kept.kernel_key == kept_key and kept.fallback_from is None, kept
        assert "_clc_token_queue" not in kept.schedule_id, kept
        for other in (
            plan_route("sm_103a", 152, M - 1, K, False),
            plan_route("sm_103a", 152, M + 1, K, False),
        ):
            assert not other.kernel_key.endswith("_clc"), (M, K, other)
            assert "_clc_token_queue" not in other.schedule_id, other
    write_cells_34_35 = [
        (arch, sm_count, M, K, value)
        for round_cells in (_ROUND_34_CLC_WRITE_CELLS, _ROUND_35_CLC_WRITE_CELLS)
        for arch, by_sm in round_cells.items()
        for sm_count, cells in by_sm.items()
        for (M, K), value in sorted(cells.items())
    ]
    for arch, sm_count, M, K, value in write_cells_34_35:
        for pdl in (False, True):
            write_key = _at_pdl(value, pdl)
            dense_key = f"{write_key.removesuffix('_write')}_clc"
            dense = cb._persistent_plan_for_key(arch, sm_count, M, K, pdl, dense_key)
            plan = plan_route(arch, sm_count, M, K, pdl, block_write_idx=K)
            # Round 39: the sm_103a (1024, 4) cell at 148 SMs runs the ``_fplr3`` build
            key, lever = _lever_cell(arch, sm_count, M, K, True, plan.kernel_key)
            assert plan.kind == "persistent" and plan.fallback_from is None, plan
            assert key == f"{write_key}_clc", (arch, M, K, pdl, plan)
            assert (plan.grid_x, plan.threads) == (M, cb.PERSISTENT_THREADS), plan
            assert plan.schedule_id == f"{dense.schedule_id}{lever}_write", plan
            assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
            assert ".write1." in plan.route_id, plan
            if MODULES:
                record = MODULES[KERNELS[arch][plan.kernel_key]]
                assert record["schedule_id"] == plan.schedule_id, plan
            other = plan_route(arch, sm_count, M + 1, K, pdl, block_write_idx=K)
            assert not other.kernel_key.endswith("_clc"), (arch, M, K, other)
            assert "_clc_token_queue" not in other.schedule_id, other


# Round 31 (GB300): the sm_103a cells that change program, (SM count, M, K) -> (kernel
# key, kind, grid) at both PDL modes: at 152 SMs the nine ``_clc`` cells, the K 0 cells (768, 0)
# and (1536, 0) (the dense small-M program with fast-math), (384, 4) (packed register carry) and
# the parked-accumulator pair cells (512, 6) and (512, 8); at 148 SMs the K 0 cells (the small-M
# inclusion of sm_103a has no SM-count key).
_ROUND_31_CELLS = {
    (148, 768, 0): ("small_m_direct:k0_fm", "small_m", 768),
    (148, 1536, 0): ("small_m_direct:k0_fm", "small_m", 1536),
    (152, 384, 4): ("persistent:k4_nc3_d2_f111000000000000_fm_rcp", "persistent", 128),
    (152, 512, 6): ("pair:k6_nc2_pbf16a_fm_cr200_pk1", "pair", 152),
    (152, 512, 8): ("pair:k8_nc2_pbf16a_fm_cr200_pk1", "pair", 152),
    (152, 768, 0): ("small_m_direct:k0_fm", "small_m", 768),
    (152, 1536, 0): ("small_m_direct:k0_fm", "small_m", 1536),
    (152, 4096, 6): ("persistent:k6_nc4_d2_f110000000000000_clc", "persistent", 4096),
    (152, 4096, 7): ("persistent:k7_nc4_d2_f110000000000000_clc", "persistent", 4096),
    (152, 4096, 8): ("persistent:k8_nc5_d2_f110000000000000_clc", "persistent", 4096),
    (152, 6144, 6): ("persistent:k6_nc4_d2_f110000000000000_clc", "persistent", 6144),
    (152, 6144, 7): ("persistent:k7_nc4_d2_f110000000000000_clc", "persistent", 6144),
    (152, 6144, 8): ("persistent:k8_nc5_d2_f110000000000000_clc", "persistent", 6144),
    (152, 8192, 6): ("persistent:k6_nc4_d2_f110000000000000_clc", "persistent", 8192),
    (152, 8192, 7): ("persistent:k7_nc4_d2_f110000000000000_clc", "persistent", 8192),
    (152, 8192, 8): ("persistent:k8_nc5_d2_f110000000000000_clc", "persistent", 8192),
}


def test_round_31_cells_take_their_programs():
    """Round 31: each ``_ROUND_31_CELLS`` cell runs its program at both PDL modes, registered and
    without a fallback, on its kind's grid and thread count; (1024, 0) and (2048, 0) keep the
    ``k0_tma`` program on sm_103a at both SM counts."""
    threads = {
        "persistent": cb.PERSISTENT_THREADS,
        "pair": cb.PAIR_CONSUMER_REGS_THREADS,
        "small_m": 256,
    }
    for (sm_count, M, K), (key, kind, grid_x) in sorted(_ROUND_31_CELLS.items()):
        for pdl in (False, True):
            plan = plan_route("sm_103a", sm_count, M, K, pdl)
            # Round 38: (152, 384, 4) runs the ``_fplr3`` build of its program
            static, _ = _lever_cell("sm_103a", sm_count, M, K, False, plan.kernel_key)
            assert (plan.kind, static, plan.grid_x) == (kind, key, grid_x), (
                sm_count,
                M,
                K,
                pdl,
                plan,
            )
            assert plan.threads == threads[kind] and plan.fallback_from is None, plan
            assert ("_fastmath" in plan.schedule_id) is ("_fm" in key), plan
            if MODULES:
                assert key in KERNELS["sm_103a"], key
    for sm_count in SM_COUNTS:
        for M in (1024, 2048):
            for pdl in (False, True):
                plan = plan_route("sm_103a", sm_count, M, 0, pdl)
                assert plan.kernel_key == "k0_tma", (sm_count, M, pdl, plan)


# Round 32: the cells that change program, (architecture, SM count, M, K) -> (kernel key or a pair
# of keys at PDL off and on, grid) at both PDL modes: on sm_100a at 148 SMs (B200) the ``_clc``
# builds of (4096, 5) and (12288, 6); on sm_103a at 152 SMs (GB300) 25 ``_clc`` builds of the
# programs the cells ran before and the packed register carry programs of (384, 3) and (512, 4).
_ROUND_32_CELLS = {
    ("sm_100a", 148, 4096, 5): ("persistent:k5_nc3_d3_f010000000000000_clc", 4096),
    ("sm_100a", 148, 12288, 6): ("persistent:k6_nc4_d2_f110000000000000_clc", 12288),
    ("sm_103a", 152, 384, 3): ("persistent:k3_nc2_d2_f111000000000000_fm_rcp", 128),
    ("sm_103a", 152, 512, 4): ("persistent:k4_nc3_d2_f111000000000000_fm_rcp", 128),
    ("sm_103a", 152, 4096, 1): (
        (
            "persistent:k1_nc4_d2_f110000000000010_clc",
            "persistent:k1_nc4_d2_f110000000100000_clc",
        ),
        4096,
    ),
    ("sm_103a", 152, 4096, 2): ("persistent:k2_nc4_d2_f011010000000000_clc", 4096),
    ("sm_103a", 152, 4096, 3): ("persistent:k3_nc4_d2_f011000000000000_clc", 4096),
    ("sm_103a", 152, 4096, 4): ("persistent:k4_nc3_d3_f110000000000000_clc", 4096),
    ("sm_103a", 152, 4096, 5): ("persistent:k5_nc3_d3_f010000000000000_clc", 4096),
    ("sm_103a", 152, 6144, 1): (
        (
            "persistent:k1_nc4_d2_f110000000000010_clc",
            "persistent:k1_nc4_d2_f110000000000000_clc",
        ),
        6144,
    ),
    ("sm_103a", 152, 6144, 2): ("persistent:k2_nc4_d2_f011010000000000_clc", 6144),
    ("sm_103a", 152, 6144, 3): ("persistent:k3_nc4_d2_f011000000000000_clc", 6144),
    ("sm_103a", 152, 6144, 4): ("persistent:k4_nc5_d2_f011001000000000_fm_clc", 6144),
    ("sm_103a", 152, 8192, 1): (
        (
            "persistent:k1_nc4_d2_f110000000000010_clc",
            "persistent:k1_nc4_d2_f110000000100000_clc",
        ),
        8192,
    ),
    ("sm_103a", 152, 8192, 2): (
        "persistent:k2_nc3_d2_f011010000000000_crfnone_clc",
        8192,
    ),
    ("sm_103a", 152, 8192, 3): ("persistent:k3_nc4_d2_f011000000000000_clc", 8192),
    ("sm_103a", 152, 8192, 5): ("persistent:k5_nc3_d3_f010000000000000_clc", 8192),
    ("sm_103a", 152, 12288, 1): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000000000_fm_clc",
        ),
        12288,
    ),
    ("sm_103a", 152, 12288, 2): (
        "persistent:k2_nc3_d2_f011010000000000_crfnone_clc",
        12288,
    ),
    ("sm_103a", 152, 12288, 3): ("persistent:k3_nc4_d2_f011000000000000_clc", 12288),
    ("sm_103a", 152, 12288, 5): ("persistent:k5_nc4_d3_f010000000000000_clc", 12288),
    ("sm_103a", 152, 12288, 6): ("persistent:k6_nc4_d2_f110000000000000_clc", 12288),
    ("sm_103a", 152, 12288, 8): ("persistent:k8_nc5_d2_f110000000000000_clc", 12288),
    ("sm_103a", 152, 16384, 1): (
        (
            "persistent:k1_nc4_d2_f110000000000010_clc",
            "persistent:k1_nc4_d2_f110000000000000_clc",
        ),
        16384,
    ),
    ("sm_103a", 152, 16384, 3): ("persistent:k3_nc4_d2_f011000000000000_clc", 16384),
    ("sm_103a", 152, 16384, 5): ("persistent:k5_nc4_d3_f010000000000000_clc", 16384),
    ("sm_103a", 152, 16384, 6): ("persistent:k6_nc4_d2_f110000000000000_clc", 16384),
    ("sm_103a", 152, 16384, 7): ("persistent:k7_nc4_d2_f110000000000000_clc", 16384),
    ("sm_103a", 152, 16384, 8): ("persistent:k8_nc5_d2_f110000000000000_clc", 16384),
}


def test_round_32_cells_take_their_programs():
    """Round 32: each ``_ROUND_32_CELLS`` cell runs its persistent program at both PDL modes,
    registered and without a fallback, on its grid with the persistent thread count."""
    for (arch, sm_count, M, K), (value, grid_x) in sorted(_ROUND_32_CELLS.items()):
        for pdl in (False, True):
            key = _at_pdl(value, pdl)
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert (plan.kind, plan.kernel_key, plan.grid_x) == (
                "persistent",
                key,
                grid_x,
            ), (arch, sm_count, M, K, pdl, plan)
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            assert ("_fastmath" in plan.schedule_id) is ("_fm" in key), plan
            assert plan.schedule_id.endswith("_clc_token_queue") is key.endswith(
                "_clc"
            ), plan
            if MODULES:
                assert key in KERNELS[arch], (arch, key)


# Round 33: the cells that change program, (architecture, SM count, M, K, snapshot write of block K)
# -> (kernel key or the (PDL off, PDL on) pair, grid) at both PDL modes: on sm_100a at 148 SMs (B200)
# the ``_clc`` build of the (2048, 7) fast-math program and the ``_write_clc`` builds of the
# (4096, 4) and (4096, 7) write programs; on sm_103a at 148 SMs (B300) the K7 ``_clc`` program at
# (3072, 7) and the (4096, 4) / (4096, 7) ``_write_clc`` builds; at 152 SMs (GB300) the held K5
# packed register carry program at (384, 5) and (512, 5) and the (4096, 1) / (4096, 4) / (4096, 7)
# ``_write_clc`` builds.
_ROUND_33_CELLS = {
    ("sm_100a", 148, 2048, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        2048,
    ),
    ("sm_100a", 148, 4096, 4, True): (
        "persistent:k4_nc3_d2_f111000000000000_write_clc",
        4096,
    ),
    ("sm_100a", 148, 4096, 7, True): (
        "persistent:k7_nc4_d2_f110000000000000_write_clc",
        4096,
    ),
    ("sm_103a", 148, 3072, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_clc",
        3072,
    ),
    ("sm_103a", 148, 4096, 4, True): (
        "persistent:k4_nc3_d2_f111000000000000_write_clc",
        4096,
    ),
    ("sm_103a", 148, 4096, 7, True): (
        "persistent:k7_nc4_d2_f110000000000000_write_clc",
        4096,
    ),
    ("sm_103a", 152, 384, 5, False): (
        "persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp",
        128,
    ),
    ("sm_103a", 152, 512, 5, False): (
        "persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp",
        128,
    ),
    ("sm_103a", 152, 4096, 1, True): (
        (
            "persistent:k1_nc4_d2_f110000000000010_write_clc",
            "persistent:k1_nc4_d2_f110000000100000_write_clc",
        ),
        4096,
    ),
    ("sm_103a", 152, 4096, 4, True): (
        "persistent:k4_nc3_d2_f111000000000000_write_clc",
        4096,
    ),
    ("sm_103a", 152, 4096, 7, True): (
        "persistent:k7_nc4_d2_f110000000000000_write_clc",
        4096,
    ),
}


def test_round_33_cells_take_their_programs():
    """Round 33: each ``_ROUND_33_CELLS`` cell runs its persistent program at both PDL modes,
    registered and without a fallback, on its grid with the persistent thread count; a write cell
    runs the snapshot write of block K."""
    for (arch, sm_count, M, K, write), (value, grid_x) in sorted(
        _ROUND_33_CELLS.items()
    ):
        for pdl in (False, True):
            key = _at_pdl(value, pdl)
            plan = plan_route(
                arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
            )
            # Round 38: (152, 384, 5) runs the ``_fplr3`` build of its program
            static = _lever_cell(arch, sm_count, M, K, write, plan.kernel_key)[0]
            assert (plan.kind, static, plan.grid_x) == (
                "persistent",
                key,
                grid_x,
            ), (arch, sm_count, M, K, write, pdl, plan)
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            assert ("_fastmath" in plan.schedule_id) is ("_fm" in key), plan
            token_queue = "_clc_token_queue" in plan.schedule_id
            assert token_queue is key.endswith("_clc"), plan
            assert plan.schedule_id.endswith("_write") is write, plan
            assert (".write1." in plan.route_id) is write, plan
            if MODULES:
                assert key in KERNELS[arch], (arch, key)


# Round 34: the cells that change program, (architecture, SM count, M, K, snapshot write of block K)
# -> (kernel key or the (PDL off, PDL on) pair, grid): on sm_100a at 148 SMs (B200) the ``_clc``
# builds of eleven dense programs and the ``_write_clc`` build of the (1024, 4) write program; on
# sm_103a at 148 SMs (B300) sixteen ``_clc`` builds (six on three new fast-math builds); at 152 SMs
# (GB300) nineteen dense ``_clc`` programs (the K1 (1024, 1) and (2048, 1) cells at PDL off only:
# they keep their program at PDL on) and the ``_write_clc`` build of the (1024, 1) write programs.
_ROUND_34_CELLS = {
    ("sm_100a", 148, 768, 6, False): ("persistent:k6_nc4_d2_f110000000000000_clc", 768),
    ("sm_100a", 148, 768, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        768,
    ),
    ("sm_100a", 148, 1024, 4, True): (
        "persistent:k4_nc3_d2_f111000000000000_write_clc",
        1024,
    ),
    ("sm_100a", 148, 1536, 6, False): (
        "persistent:k6_nc4_d2_f110000000000000_clc",
        1536,
    ),
    ("sm_100a", 148, 1536, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        1536,
    ),
    ("sm_100a", 148, 2048, 1, False): (
        "persistent:k1_nc2_d2_f110000000000000_clc",
        2048,
    ),
    ("sm_100a", 148, 2048, 4, False): (
        "persistent:k4_nc3_d3_f110000000000000_clc",
        2048,
    ),
    ("sm_100a", 148, 3072, 1, False): (
        "persistent:k1_nc4_d2_f110000000000000_clc",
        3072,
    ),
    ("sm_100a", 148, 3072, 2, False): (
        "persistent:k2_nc4_d2_f110000000000000_crfnone_clc",
        3072,
    ),
    ("sm_100a", 148, 3072, 3, False): (
        "persistent:k3_nc4_d2_f010000000000000_crfnone_clc",
        3072,
    ),
    ("sm_100a", 148, 3072, 4, False): (
        "persistent:k4_nc3_d3_f110000000000000_clc",
        3072,
    ),
    ("sm_100a", 148, 3072, 5, False): (
        "persistent:k5_nc4_d3_f010000000000000_clc",
        3072,
    ),
    ("sm_103a", 148, 1024, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm_clc",
        ),
        1024,
    ),
    ("sm_103a", 148, 1024, 6, False): (
        "persistent:k6_nc4_d2_f111000000000000_fm_clc",
        1024,
    ),
    ("sm_103a", 148, 1536, 4, False): (
        "persistent:k4_nc5_d2_f011001000000000_fm_clc",
        1536,
    ),
    ("sm_103a", 148, 1536, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        1536,
    ),
    ("sm_103a", 148, 2048, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm_clc",
        ),
        2048,
    ),
    ("sm_103a", 148, 2048, 3, False): (
        "persistent:k3_nc4_d2_f011000000000000_clc",
        2048,
    ),
    ("sm_103a", 148, 2048, 4, False): (
        "persistent:k4_nc3_d3_f110000000000000_clc",
        2048,
    ),
    ("sm_103a", 148, 2048, 5, False): (
        "persistent:k5_nc4_d3_f010000000000000_clc",
        2048,
    ),
    ("sm_103a", 148, 2048, 6, False): (
        "persistent:k6_nc4_d2_f111000000000000_fm_clc",
        2048,
    ),
    ("sm_103a", 148, 2048, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        2048,
    ),
    ("sm_103a", 148, 3072, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000000000_fm_clc",
        ),
        3072,
    ),
    ("sm_103a", 148, 3072, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        3072,
    ),
    ("sm_103a", 148, 3072, 3, False): (
        "persistent:k3_nc4_d2_f011000000000000_clc",
        3072,
    ),
    ("sm_103a", 148, 3072, 4, False): (
        "persistent:k4_nc5_d2_f011001000000000_fm_clc",
        3072,
    ),
    ("sm_103a", 148, 3072, 5, False): (
        "persistent:k5_nc3_d3_f010000000000000_clc",
        3072,
    ),
    ("sm_103a", 148, 6144, 5, False): (
        "persistent:k5_nc3_d3_f010000000000000_clc",
        6144,
    ),
    ("sm_103a", 152, 768, 7, False): ("persistent:k7_nc4_d2_f110000000000000_clc", 768),
    ("sm_103a", 152, 768, 8, False): ("persistent:k8_nc5_d2_f110000000000000_clc", 768),
    ("sm_103a", 152, 1024, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm",
        ),
        1024,
    ),
    ("sm_103a", 152, 1024, 1, True): (
        (
            "persistent:k1_nc4_d2_f110000000000010_write_clc",
            "persistent:k1_nc4_d2_f110000000100000_write_clc",
        ),
        1024,
    ),
    ("sm_103a", 152, 1536, 6, False): (
        "persistent:k6_nc4_d2_f110000000000000_clc",
        1536,
    ),
    ("sm_103a", 152, 1536, 8, False): (
        "persistent:k8_nc5_d2_f110000000000000_clc",
        1536,
    ),
    ("sm_103a", 152, 2048, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm",
        ),
        2048,
    ),
    ("sm_103a", 152, 2048, 3, False): (
        "persistent:k3_nc4_d2_f011000000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 2048, 4, False): (
        "persistent:k4_nc3_d3_f110000000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 2048, 5, False): (
        "persistent:k5_nc3_d3_f010000000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 2048, 6, False): (
        "persistent:k6_nc4_d2_f110000000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 2048, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 2048, 8, False): (
        "persistent:k8_nc5_d2_f110000000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 3072, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000000000_fm_clc",
        ),
        3072,
    ),
    ("sm_103a", 152, 3072, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        3072,
    ),
    ("sm_103a", 152, 3072, 3, False): (
        "persistent:k3_nc4_d2_f011000000000000_clc",
        3072,
    ),
    ("sm_103a", 152, 3072, 4, False): (
        "persistent:k4_nc5_d2_f011001000000000_fm_clc",
        3072,
    ),
    ("sm_103a", 152, 3072, 5, False): (
        "persistent:k5_nc3_d3_f010000000000000_clc",
        3072,
    ),
    ("sm_103a", 152, 3072, 8, False): (
        "persistent:k8_nc5_d2_f110000000000000_clc",
        3072,
    ),
    ("sm_103a", 152, 6144, 5, False): (
        "persistent:k5_nc4_d3_f010000000000000_clc",
        6144,
    ),
}


def test_round_34_cells_take_their_programs():
    """Round 34: each ``_ROUND_34_CELLS`` cell runs its persistent program at both PDL modes,
    registered and without a fallback, on its grid with the persistent thread count (a ``_clc``
    program on one CTA per token); a write cell runs the snapshot write of block K."""
    for (arch, sm_count, M, K, write), (value, grid_x) in sorted(
        _ROUND_34_CELLS.items()
    ):
        if (arch, sm_count, M, K, write) in _ROUND_35_CELLS:
            continue  # round 35 moves the cell (test_round_35_cells_take_their_programs)
        for pdl in (False, True):
            key = _at_pdl(value, pdl)
            plan = plan_route(
                arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
            )
            token_queue = key.endswith("_clc")
            assert (plan.kind, plan.kernel_key) == ("persistent", key), (
                arch,
                sm_count,
                M,
                K,
                write,
                pdl,
                plan,
            )
            if token_queue:
                assert plan.grid_x == grid_x, plan
            else:
                assert 1 <= plan.grid_x <= sm_count, plan
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            assert ("_fastmath" in plan.schedule_id) is ("_fm" in key), plan
            assert ("_clc_token_queue" in plan.schedule_id) is token_queue, plan
            assert plan.schedule_id.endswith("_write") is write, plan
            assert (".write1." in plan.route_id) is write, plan
            if MODULES:
                assert key in KERNELS[arch], (arch, key)


# Round 35: the cells that change program, (architecture, SM count, M, K, snapshot write of block K)
# -> (kernel key or the (PDL off, PDL on) pair, grid): ``_clc`` programs on sm_100a at 148 SMs
# (B200, 8 cells), on sm_103a at 148 SMs (B300, 14 cells) and at 152 SMs (GB300,
# 19 cells), one of them a snapshot write per configuration.
_ROUND_35_CELLS = {
    ("sm_100a", 148, 768, 1, False): (
        "persistent:k1_nc4_d2_f110000000000000_fm_clc",
        768,
    ),
    ("sm_100a", 148, 768, 8, False): ("persistent:k8_nc5_d2_f110000000000000_clc", 768),
    ("sm_100a", 148, 1024, 1, False): (
        "persistent:k1_nc2_d2_f110000000000000_fm_clc",
        1024,
    ),
    ("sm_100a", 148, 1024, 1, True): (
        "persistent:k1_nc4_d2_f110000000000000_write_clc",
        1024,
    ),
    ("sm_100a", 148, 1536, 1, False): (
        "persistent:k1_nc4_d2_f110000000000000_fm_clc",
        1536,
    ),
    ("sm_100a", 148, 1536, 8, False): (
        "persistent:k8_nc5_d2_f110000000000000_clc",
        1536,
    ),
    ("sm_100a", 148, 2048, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        2048,
    ),
    ("sm_100a", 148, 2048, 6, False): (
        "persistent:k6_nc4_d2_f111000000000000_fm_clc",
        2048,
    ),
    ("sm_103a", 148, 768, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        768,
    ),
    ("sm_103a", 148, 768, 8, False): ("persistent:k8_nc5_d2_f110000000000000_clc", 768),
    ("sm_103a", 148, 1024, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        1024,
    ),
    ("sm_103a", 148, 1024, 4, True): (
        "persistent:k4_nc3_d2_f111000000000000_write_clc",
        1024,
    ),
    ("sm_103a", 148, 1536, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm_clc",
        ),
        1536,
    ),
    ("sm_103a", 148, 1536, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        1536,
    ),
    ("sm_103a", 148, 1536, 3, False): (
        "persistent:k3_nc4_d2_f011000000000000_clc",
        1536,
    ),
    ("sm_103a", 148, 1536, 5, False): (
        "persistent:k5_nc4_d3_f010000000000000_clc",
        1536,
    ),
    ("sm_103a", 148, 1536, 6, False): (
        "persistent:k6_nc4_d2_f111000000000000_fm_clc",
        1536,
    ),
    ("sm_103a", 148, 1536, 8, False): (
        "persistent:k8_nc5_d2_f110000000000000_clc",
        1536,
    ),
    ("sm_103a", 148, 2048, 2, False): (
        "persistent:k2_nc3_d2_f011010000000000_fm_crfnone_clc",
        2048,
    ),
    ("sm_103a", 148, 8192, 4, False): (
        "persistent:k4_nc5_d2_f010001000000000_clc",
        8192,
    ),
    ("sm_103a", 148, 12288, 4, False): (
        "persistent:k4_nc5_d2_f010001000000000_clc",
        12288,
    ),
    ("sm_103a", 148, 16384, 4, False): (
        "persistent:k4_nc5_d2_f010001000000000_clc",
        16384,
    ),
    ("sm_103a", 152, 768, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        768,
    ),
    ("sm_103a", 152, 1024, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm_clc",
        ),
        1024,
    ),
    ("sm_103a", 152, 1024, 4, True): (
        "persistent:k4_nc3_d2_f111000000000000_write_clc",
        1024,
    ),
    ("sm_103a", 152, 1024, 6, False): (
        "persistent:k6_nc4_d2_f111000000000000_fm_clc",
        1024,
    ),
    ("sm_103a", 152, 1024, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        1024,
    ),
    ("sm_103a", 152, 1536, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        1536,
    ),
    ("sm_103a", 152, 1536, 3, False): (
        "persistent:k3_nc4_d2_f011000000000000_clc",
        1536,
    ),
    ("sm_103a", 152, 1536, 4, False): (
        "persistent:k4_nc5_d2_f011001000000000_fm_clc",
        1536,
    ),
    ("sm_103a", 152, 1536, 5, False): (
        "persistent:k5_nc3_d3_f010000000000000_clc",
        1536,
    ),
    ("sm_103a", 152, 1536, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        1536,
    ),
    ("sm_103a", 152, 2048, 1, False): (
        (
            "persistent:k1_nc4_d2_f110000000000010_fm_clc",
            "persistent:k1_nc4_d2_f110000000100000_fm_clc",
        ),
        2048,
    ),
    ("sm_103a", 152, 2048, 2, False): (
        "persistent:k2_nc4_d2_f011010000000000_clc",
        2048,
    ),
    ("sm_103a", 152, 3072, 6, False): (
        "persistent:k6_nc4_d2_f111000000000000_fm_clc",
        3072,
    ),
    ("sm_103a", 152, 3072, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        3072,
    ),
    ("sm_103a", 152, 8192, 4, False): (
        "persistent:k4_nc5_d2_f010001000000000_clc",
        8192,
    ),
    ("sm_103a", 152, 12288, 4, False): (
        "persistent:k4_nc5_d2_f010001000000000_clc",
        12288,
    ),
    ("sm_103a", 152, 12288, 7, False): (
        "persistent:k7_nc4_d2_f110000000000000_fm_clc",
        12288,
    ),
    ("sm_103a", 152, 16384, 2, False): (
        "persistent:k2_nc3_d2_f011010000000000_crfnone_clc",
        16384,
    ),
    ("sm_103a", 152, 16384, 4, False): (
        "persistent:k4_nc5_d2_f010001000000000_clc",
        16384,
    ),
}


def test_round_35_cells_take_their_programs():
    """Round 35: each ``_ROUND_35_CELLS`` cell runs its persistent ``_clc`` program at both PDL
    modes, registered and without a fallback, on one CTA per token with the persistent thread
    count; a write cell runs the snapshot write of block K."""
    for (arch, sm_count, M, K, write), (value, grid_x) in sorted(
        _ROUND_35_CELLS.items()
    ):
        for pdl in (False, True):
            key = _at_pdl(value, pdl)
            plan = plan_route(
                arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
            )
            # Round 39: the sm_103a (1024, 4) write cell at 148 SMs runs the ``_fplr3`` build
            static = _lever_cell(arch, sm_count, M, K, write, plan.kernel_key)[0]
            assert (plan.kind, static) == ("persistent", key), (
                arch,
                sm_count,
                M,
                K,
                write,
                pdl,
                plan,
            )
            assert key.endswith("_clc") and plan.grid_x == grid_x, plan
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            assert ("_fastmath" in plan.schedule_id) is ("_fm" in key), plan
            assert "_clc_token_queue" in plan.schedule_id, plan
            assert plan.schedule_id.endswith("_write") is write, plan
            assert (".write1." in plan.route_id) is write, plan
            if MODULES:
                assert key in KERNELS[arch], (arch, key)


# Round 36: two cells run the ``_clc`` build of the program they ran before, built with the CLC
# lookahead (``_cla``: the producer requests a trip's successor at the start of the trip) and, on
# GB300, with an L2 prefetch of the first token's rows at kernel entry (``_fpf``), on one CTA per
# token at both PDL modes: (arch, SM count, M, K) -> (key, grid).
_ROUND_36_CELLS = {
    ("sm_100a", 148, 12288, 5): (
        "persistent:k5_nc4_d3_f110000000000000_clc_cla",
        12288,
    ),
    ("sm_103a", 152, 384, 2): (
        "persistent:k2_nc3_d2_f011010000000000_fm_crfnone_clc_cla_fpf",
        384,
    ),
}


def test_round_36_cells_take_their_programs():
    """Round 36: each ``_ROUND_36_CELLS`` cell runs its ``_clc_cla`` (B200 (12288, 5)) or
    ``_clc_cla_fpf`` (GB300 (384, 2)) program at both PDL modes, registered and without a
    fallback, on one CTA per token with the persistent thread count.  Its schedule label is the
    label of the program without ``_clc`` plus ``_clc_token_queue``, ``_clc_lookahead`` and, for an
    ``_fpf`` key, ``_first_chunk_prefetch_l2``."""
    for (arch, sm_count, M, K), (key, grid_x) in sorted(_ROUND_36_CELLS.items()):
        fpf = key.endswith("_clc_cla_fpf")
        assert fpf or key.endswith("_clc_cla"), key
        static = key.removesuffix("_fpf").removesuffix("_cla").removesuffix("_clc")
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert (plan.kind, plan.kernel_key) == ("persistent", key), (
                arch,
                sm_count,
                M,
                K,
                pdl,
                plan,
            )
            assert plan.grid_x == grid_x == M, plan
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            label = cb._persistent_plan_for_key(
                arch, sm_count, M, K, pdl, static
            ).schedule_id
            label += "_clc_token_queue_clc_lookahead"
            label += "_first_chunk_prefetch_l2" if fpf else ""
            assert plan.schedule_id == label, plan
            assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
            if MODULES:
                assert MODULES[KERNELS[arch][key]]["kernel_key"] == key, (arch, key)


# Round 37: seven cells run the program they ran before built with one opt-in lever, at both PDL
# modes: ``_clc_tokb`` (B200 (16384, 7) and (12288, 4): the ``_clc`` build, one CTA per token, with
# the producer alone reading the CLC responses and handing each trip's token to the consumers
# through shared memory), ``_fpl`` / ``_fpw`` (B300: the first token's L2 prefetches issued off the
# barrier / TMEM setup path, one row per lane of warp 8 or one row per warp over warps 2-8, on the
# program's own grid).  The per-token work and the output are the program's without the suffix.
# (arch, SM count, M, K, snapshot write) -> ((key at PDL off, key at PDL on), grid).
_ROUND_37_CELLS = {
    ("sm_100a", 148, 12288, 4, False): (
        ("persistent:k4_nc3_d3_f110000000000000_clc_tokb",) * 2,
        12288,
    ),
    ("sm_100a", 148, 16384, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_clc_tokb",) * 2,
        16384,
    ),
    ("sm_103a", 148, 384, 2, False): (
        ("persistent:k2_nc3_d2_f011010000000000_fm_crfnone_fpw",) * 2,
        128,
    ),
    ("sm_103a", 148, 512, 3, False): (
        ("persistent:k3_nc4_d2_f111000000000000_fm_rcp_fpl",) * 2,
        128,
    ),
    ("sm_103a", 148, 768, 1, False): (
        ("persistent:k1_nc4_d2_f111010000000000_fm_fpl",) * 2,
        148,
    ),
    ("sm_103a", 148, 768, 2, False): (
        ("persistent:k2_nc3_d2_f111010000000000_fm_fpl",) * 2,
        148,
    ),
    ("sm_103a", 148, 1024, 1, True): (
        (
            "persistent:k1_nc4_d3_f010000000000010_crheld_write_fpl",
            "persistent:k1_nc4_d3_f010000000100000_crheld_write_fpl",
        ),
        148,
    ),
}
# The lever suffix of a round-37 key and the schedule label it adds to its program's label.
_ROUND_37_LEVERS = {
    "_clc_tokb": "_clc_token_queue_clc_token_broadcast",
    "_fpl": "_first_chunk_prefetch_l2_lanes",
    "_fpw": "_first_chunk_prefetch_l2_warps",
}


def test_round_37_cells_take_their_lever_programs():
    """Round 37: each ``_ROUND_37_CELLS`` cell runs its lever program at both PDL modes,
    registered and without a fallback, with the persistent thread count: a ``_clc_tokb`` program
    on one CTA per token, an ``_fpl`` / ``_fpw`` program on the grid of its program without the
    suffix.  Its schedule label is that program's label plus the lever's (before ``_write`` for a
    snapshot-write cell)."""
    for (arch, sm_count, M, K, write), (keys, grid_x) in sorted(
        _ROUND_37_CELLS.items()
    ):
        for pdl in (False, True):
            key = keys[int(pdl)]
            lever = next(s for s in _ROUND_37_LEVERS if key.endswith(s))
            static = key.removesuffix(lever).removesuffix("_write" if write else "")
            base = cb._persistent_plan_for_key(arch, sm_count, M, K, pdl, static)
            plan = plan_route(
                arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
            )
            assert (plan.kind, plan.kernel_key) == ("persistent", key), (
                arch,
                sm_count,
                M,
                K,
                pdl,
                plan,
            )
            assert plan.grid_x == grid_x, plan
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            label = base.schedule_id + _ROUND_37_LEVERS[lever]
            label += "_write" if write else ""
            assert plan.schedule_id == label, plan
            if lever == "_clc_tokb":
                assert grid_x == M, plan
                assert plan.route_id.endswith(f".pdl{int(pdl)}.clc_token_queue"), plan
            else:
                assert grid_x == base.grid_x, (plan, base)
                route = base.route_id.replace(base.schedule_id, label, 1)
                if write:
                    route = route.replace(".write0.", ".write1.", 1)
                assert plan.route_id == route, plan
            if MODULES:
                record = MODULES[KERNELS[arch][key]]
                assert (record["kernel_key"], record["schedule_id"]) == (key, label), (
                    arch,
                    key,
                )


def test_a_registered_fallback_never_resolves_to_a_lever_program(monkeypatch):
    """An ``_fpl``, ``_fpw`` (also with ``r<R>``) or ``_tokb`` program serves the cells that name
    it only: an unregistered key without the suffix never resolves to one, however close its
    fields."""
    for arch, M, K, static, lever in (
        ("sm_103a", 768, 1, "persistent:k1_nc4_d2_f111010000000000_fm", "_fpl"),
        ("sm_103a", 384, 2, "persistent:k2_nc3_d2_f011010000000000_fm_crfnone", "_fpw"),
        ("sm_100a", 16384, 7, "persistent:k7_nc4_d2_f110000000000000", "_clc_tokb"),
        ("sm_103a", 512, 7, "persistent:k7_nc4_d2_f110000000000000_fm", "_fpwr2"),
    ):
        plan = cb._persistent_plan_for_key(arch, SM_COUNT, M, K, False, static)
        monkeypatch.setattr(cb, "KERNELS", {arch: {static + lever: "m1"}})
        unresolved = cb._resolve_registered(plan, SM_COUNT, M)
        assert unresolved == plan, (arch, lever, unresolved)


# Round 38: twenty-five cells run the program they ran before built with a first-token L2
# prefetch lever, at both PDL modes, on the program's own grid: ``_fpl`` (one row per lane of
# warp 8), ``_fplr<R>`` / ``_fpwr<R>`` (the lane / warp prefetches limited to the first token's
# first R rows in load order).  The per-token work and the output are the program's without the
# suffix.  (arch, SM count, M, K, snapshot write) -> ((key at PDL off, key at PDL on), grid).
_ROUND_38_CELLS = {
    ("sm_100a", 148, 512, 2, False): (
        ("persistent:k2_nc3_d2_f111010000000000_fm_crfnone_fpl",) * 2,
        128,
    ),
    ("sm_100a", 148, 768, 2, False): (
        ("persistent:k2_nc4_d2_f111010000000000_fm_fpl",) * 2,
        148,
    ),
    ("sm_100a", 148, 1024, 2, False): (
        ("persistent:k2_nc3_d2_f011010000000000_fm_fpl",) * 2,
        148,
    ),
    ("sm_103a", 148, 256, 4, True): (
        ("persistent:k4_nc3_d2_f111000000000000_fm_write_fplr2",) * 2,
        128,
    ),
    ("sm_103a", 148, 256, 7, False): (
        ("persistent:k7_nc4_d2_f111000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 256, 7, True): (
        ("persistent:k7_nc4_d2_f111000000000000_fm_write_fplr2",) * 2,
        128,
    ),
    ("sm_103a", 148, 384, 4, False): (
        ("persistent:k4_nc3_d2_f111000000000000_fm_rcp_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 384, 5, False): (
        ("persistent:k5_nc3_d2_f111000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 384, 6, False): (
        ("persistent:k6_nc4_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 384, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 512, 5, False): (
        ("persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 512, 6, False): (
        ("persistent:k6_nc4_d2_f111000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 512, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 1024, 4, False): (
        ("persistent:k4_nc3_d2_f111000000000000_fm_rcp_fplr3",) * 2,
        148,
    ),
    ("sm_103a", 152, 256, 7, False): (
        ("persistent:k7_nc4_d2_f111000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 152, 256, 7, True): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_write_fplr2",) * 2,
        128,
    ),
    ("sm_103a", 152, 384, 4, False): (
        ("persistent:k4_nc3_d2_f111000000000000_fm_rcp_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 152, 384, 5, False): (
        ("persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 152, 384, 6, False): (
        ("persistent:k6_nc4_d2_f111000000000000_fm_fplr2",) * 2,
        128,
    ),
    ("sm_103a", 152, 384, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_fpwr3",) * 2,
        128,
    ),
    ("sm_103a", 152, 512, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_fpwr2",) * 2,
        128,
    ),
    ("sm_103a", 152, 768, 1, False): (
        ("persistent:k1_nc4_d2_f011000000000000_fm_fpl",) * 2,
        152,
    ),
    ("sm_103a", 152, 768, 2, False): (
        ("persistent:k2_nc3_d2_f111010000000000_fm_fpl",) * 2,
        152,
    ),
    ("sm_103a", 152, 1024, 4, False): (
        ("persistent:k4_nc3_d3_f111000000000000_fm_fplr3",) * 2,
        152,
    ),
    ("sm_103a", 152, 1536, 1, False): (
        ("persistent:k1_nc4_d2_f111010000000000_fm_fpl",) * 2,
        152,
    ),
}
# The lever suffix of a round-38 key and the schedule label it adds to its program's label.
_ROUND_38_LEVERS = {
    "_fpl": "_first_chunk_prefetch_l2_lanes",
    "_fplr2": "_first_chunk_prefetch_l2_lanes_first_chunk_prefetch_l2_rows2",
    "_fplr3": "_first_chunk_prefetch_l2_lanes_first_chunk_prefetch_l2_rows3",
    "_fpwr2": "_first_chunk_prefetch_l2_warps_first_chunk_prefetch_l2_rows2",
    "_fpwr3": "_first_chunk_prefetch_l2_warps_first_chunk_prefetch_l2_rows3",
}
# Round 39: thirteen more cells run the program they ran before built with a row-capped
# first-token L2 prefetch (``_fplr<R>`` / ``_fpwr<R>``), at both PDL modes, on the program's own
# grid; the sm_103a snapshot-write (1024, 4) cell at 148 SMs runs the ``_fplr3`` build of its
# token-queue program (one CTA per token).  Same layout as ``_ROUND_38_CELLS``.
_ROUND_39_CELLS = {
    ("sm_100a", 148, 256, 4, False): (
        ("persistent:k4_nc3_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_100a", 148, 256, 4, True): (
        ("persistent:k4_nc3_d2_f110000000000000_fm_write_fplr3",) * 2,
        128,
    ),
    ("sm_100a", 148, 256, 7, False): (
        ("persistent:k7_nc4_d2_f111000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_100a", 148, 256, 7, True): (
        ("persistent:k7_nc4_d2_f111000000000000_fm_write_fplr2",) * 2,
        128,
    ),
    ("sm_100a", 148, 384, 4, False): (
        ("persistent:k4_nc3_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_100a", 148, 384, 6, False): (
        ("persistent:k6_nc4_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_100a", 148, 384, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_fpwr2",) * 2,
        128,
    ),
    ("sm_100a", 148, 384, 8, False): (
        ("persistent:k8_nc5_d2_f110000000000000_fm_fplr2",) * 2,
        148,
    ),
    ("sm_100a", 148, 512, 5, False): (
        ("persistent:k5_nc4_d2_f111000000000000_rcp_fplr3",) * 2,
        128,
    ),
    ("sm_100a", 148, 512, 7, False): (
        ("persistent:k7_nc4_d2_f110000000000000_fm_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 512, 4, False): (
        ("persistent:k4_nc3_d2_f111000000000000_fm_rcp_fplr3",) * 2,
        128,
    ),
    ("sm_103a", 148, 1024, 4, True): (
        ("persistent:k4_nc3_d2_f111000000000000_write_clc_fplr3",) * 2,
        1024,
    ),
    ("sm_103a", 148, 1024, 5, False): (
        ("persistent:k5_nc4_d3_f001000000000000_fm_crheld_rcp_fplr2",) * 2,
        148,
    ),
}
# Every cell that runs a first-token prefetch lever build of the program it ran before.
_LEVER_CELLS = {**_ROUND_38_CELLS, **_ROUND_39_CELLS}


def _lever_cell(arch, sm_count, M, K, write, key):
    """(the key without its lever suffix, the schedule label of the lever) for the key a
    ``_LEVER_CELLS`` cell runs; ``(key, "")`` for any other cell."""
    if (arch, sm_count, M, K, write) not in _LEVER_CELLS:
        return key, ""
    lever = next(s for s in _ROUND_38_LEVERS if key.endswith(s))
    return key.removesuffix(lever), _ROUND_38_LEVERS[lever]


def test_lever_cells_take_their_lever_programs():
    """Rounds 38 and 39: each ``_LEVER_CELLS`` cell runs its lever program at both PDL modes,
    registered and without a fallback, with the persistent thread count, on the grid of its
    program without the suffix.  Its schedule label is that program's label plus the lever's
    (before ``_write`` for a snapshot-write cell)."""
    for (arch, sm_count, M, K, write), (keys, grid_x) in sorted(_LEVER_CELLS.items()):
        for pdl in (False, True):
            key = keys[int(pdl)]
            lever = next(s for s in _ROUND_38_LEVERS if key.endswith(s))
            static = key.removesuffix(lever)
            if write:  # ``_write`` sits before the ``_clc`` group of a token-queue key
                static = static.replace("_write", "", 1)
            base = cb._persistent_plan_for_key(arch, sm_count, M, K, pdl, static)
            plan = plan_route(
                arch, sm_count, M, K, pdl, block_write_idx=K if write else -1
            )
            assert (plan.kind, plan.kernel_key) == ("persistent", key), (
                arch,
                sm_count,
                M,
                K,
                pdl,
                plan,
            )
            assert plan.grid_x == grid_x == base.grid_x, (plan, base)
            assert plan.threads == cb.PERSISTENT_THREADS, plan
            assert plan.fallback_from is None, plan
            label = base.schedule_id + _ROUND_38_LEVERS[lever]
            label += "_write" if write else ""
            assert plan.schedule_id == label, plan
            route = base.route_id.replace(base.schedule_id, label, 1)
            if write:
                route = route.replace(".write0.", ".write1.", 1)
            assert plan.route_id == route, plan
            if MODULES:
                record = MODULES[KERNELS[arch][key]]
                assert (record["kernel_key"], record["schedule_id"]) == (key, label), (
                    arch,
                    key,
                )


def test_a_registered_fallback_never_resolves_to_a_clc_program(monkeypatch):
    """A ``_clc`` program covers every token only on its cell's grid of ``M`` CTAs, so the
    registered-variant fallback never resolves to one: an unregistered ``_clc`` cell resolves to
    the program without the suffix on that program's own grid, and an unregistered key without the
    suffix never resolves to a ``_clc`` program, however close its fields."""
    clc, static = _CLC_KEYS[6], _CLC_KEYS[6].removesuffix("_clc")
    exact = cb._plan_route_exact("sm_103a", SM_COUNT, 4096, 6, False)
    assert (exact.kernel_key, exact.grid_x) == (clc, 4096), exact
    monkeypatch.setattr(cb, "KERNELS", {"sm_103a": {static: "m1"}})
    resolved = cb._resolve_registered(exact, SM_COUNT, 4096)
    assert (resolved.kernel_key, resolved.fallback_from) == (static, clc), resolved
    assert resolved.grid_x == SM_COUNT, resolved
    assert "_clc_token_queue" not in resolved.schedule_id, resolved
    assert resolved.route_id.endswith(".token_or_full_sm.registered_fallback"), resolved
    plan = cb._persistent_plan_for_key("sm_103a", SM_COUNT, 4096, 6, False, static)
    other = "persistent:k6_nc3_d2_f010000000000000"
    monkeypatch.setattr(cb, "KERNELS", {"sm_103a": {clc: "m1", other: "m2"}})
    resolved = cb._resolve_registered(plan, SM_COUNT, 4096)
    assert (resolved.kernel_key, resolved.fallback_from) == (other, static), resolved
    monkeypatch.setattr(cb, "KERNELS", {"sm_103a": {clc: "m1"}})
    unresolved = cb._resolve_registered(plan, SM_COUNT, 4096)
    assert unresolved == plan, unresolved


def test_consumer_regs_pair_cells_take_the_384_thread_programs():
    """Round 20: a ``_cr232`` pair key is the pair kernel built at ``consumer_regs`` 232 (three
    whole warpgroups, 384 threads); its schedule label adds ``_consumer_regs232`` ahead of the grid.
    The sm_100a (192, 4), (192, 5) and (192, 8) cells and the sm_103a (192, 4) and (192, 5) cells at
    both SM counts run it at both PDL modes; round 23 adds the sm_100a (768, 4), (768, 5) and
    (256, 8) cells and the sm_103a (768, 4), (256, 5) and (768, 5) cells at both SM counts; round 24
    adds the sm_100a (512, 4) and (512, 8) cells and the sm_103a (512, 4) cell at 148 SMs, which
    round 26 moves to the packed register-carry program.  No cell runs the 320-thread pair program."""
    cells = (
        [("sm_100a", 148, 192, K) for K in (4, 5, 8)]
        + [("sm_103a", sm_count, 192, K) for sm_count in SM_COUNTS for K in (4, 5)]
        + [("sm_100a", 148, M, K) for M, K in ((768, 4), (768, 5), (256, 8))]
        + [
            ("sm_103a", sm_count, M, K)
            for sm_count in SM_COUNTS
            for M, K in ((768, 4), (256, 5), (768, 5))
        ]
        + [("sm_100a", 148, 512, K) for K in (4, 8)]
    )
    for arch, sm_count, M, K in cells:
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert (plan.kind, plan.kernel_key) == (
                "pair",
                f"pair:k{K}_nc2_pbf16a_fm_cr232",
            ), (arch, sm_count, M, K, pdl, plan)
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
    assert sorted(
        (arch, sm_count, M, K)
        for arch, sm_count, M, K, key in _r2_cells("R2_PAIR_CELLS")
        if key.endswith("_cr232")
    ) == sorted(cells)


def test_park_acc_pair_cells_take_the_384_thread_programs():
    """A ``_cr200_pk1`` pair key is the pair kernel built at ``consumer_regs`` 200 with
    ``park_acc`` 1 (three whole warpgroups, 384 threads; each consumer thread's accumulators wait in
    tensor memory while the prefix chunk streams); its schedule label adds
    ``_consumer_regs200_park_acc1`` ahead of the grid. The sm_100a (192, 6) and (192, 7) cells at
    148 SMs and the sm_103a (192, 6), (192, 7) and (192, 8) cells at both SM counts run it at both
    PDL modes; round 23 adds the sm_100a (256, 6) cell and the sm_103a (256, 6), (768, 6) and
    (256, 8) cells at both SM counts; round 24 adds the sm_100a (512, 6) and sm_103a (512, 8) cells
    at 148 SMs; round 31 adds the sm_103a (512, 6) and (512, 8) cells at 152 SMs."""
    cells = (
        [("sm_100a", 148, 192, K) for K in (6, 7)]
        + [("sm_103a", sm_count, 192, K) for sm_count in SM_COUNTS for K in (6, 7, 8)]
        + [("sm_100a", 148, 256, 6)]
        + [
            ("sm_103a", sm_count, M, K)
            for sm_count in SM_COUNTS
            for M, K in ((256, 6), (768, 6), (256, 8))
        ]
        + [("sm_100a", 148, 512, 6), ("sm_103a", 148, 512, 8)]
        + [("sm_103a", 152, 512, K) for K in (6, 8)]
    )
    for arch, sm_count, M, K in cells:
        for pdl in (False, True):
            plan = plan_route(arch, sm_count, M, K, pdl)
            assert (plan.kind, plan.kernel_key) == (
                "pair",
                f"pair:k{K}_nc2_pbf16a_fm_cr200_pk1",
            ), (arch, sm_count, M, K, pdl, plan)
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
    ) == sorted(cells)


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
    if (M, K) in (
        _ROUND_33_CLC_WRITE_CELLS[arch][SM_COUNT].keys()
        | _ROUND_34_CLC_WRITE_CELLS[arch][SM_COUNT].keys()
        | _ROUND_35_CLC_WRITE_CELLS[arch][SM_COUNT].keys()
    ):
        # Rounds 33 to 35: the cell runs the token-queue build of its write program, one CTA
        # per token
        # (test_clc_cells_take_the_token_queue_programs checks it exactly); round 39 builds the
        # sm_103a (1024, 4) cell with ``_fplr3``.
        key, lever = _lever_cell(arch, SM_COUNT, M, K, True, plan.kernel_key)
        assert plan.schedule_id.endswith(f"_clc_token_queue{lever}_write"), plan
        assert key.endswith("_write_clc") and plan.grid_x == M, plan
        return
    if (arch, SM_COUNT, M, K, True) in _ROUND_37_CELLS:
        # Round 37: the cell runs the ``_fpl`` build of its write program
        # (test_round_37_cells_take_their_lever_programs checks it exactly).
        assert plan.schedule_id.endswith("_first_chunk_prefetch_l2_lanes_write"), plan
        assert plan.kernel_key.endswith("_write_fpl"), plan
        return
    if (arch, SM_COUNT, M, K, True) in _LEVER_CELLS:
        # Rounds 38 and 39: the cell runs a lever build of its write program
        # (test_lever_cells_take_their_lever_programs checks it exactly).
        key, lever = _lever_cell(arch, SM_COUNT, M, K, True, plan.kernel_key)
        assert plan.schedule_id.endswith(f"{lever}_write"), plan
        assert key.endswith("_write"), plan
        return
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
            cells[(arch, sm_count, M, K, pdl)] = _write_key(key)
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
    # M = 384 / K = 8 is the native_k8 cell of sm_103a (M = 1 belongs to the small-M direct kernel);
    # the port launches one CTA per SM on every qualified count and keeps its kernel key (one
    # registered program)
    keys = set()
    for sm_count in SM_COUNTS:
        plan = plan_route("sm_103a", sm_count, 384, 8, False)
        assert plan.kind == "native"
        assert plan.grid_x == sm_count
        assert plan.schedule_id == f"native_k8_nc3_d2_ws288_grid{sm_count}"
        keys.add(plan.kernel_key)
    assert keys == {"native_k8"}
    # sm_100a M = 256 / K = 8 runs the consumer_regs 232 pair program at 148 SMs (round 23) and the
    # native K8 port at 152 SMs, which ships for sm_100a too (one two-architecture program)
    assert (
        plan_route("sm_100a", 148, 256, 8, False).kernel_key
        == "pair:k8_nc2_pbf16a_fm_cr232"
    )
    plan = plan_route("sm_100a", 152, 256, 8, False)
    assert (plan.kind, plan.kernel_key, plan.grid_x) == ("native", "native_k8", 152)
    assert plan.schedule_id == "native_k8_nc3_d2_ws288_grid152"
    if MODULES:
        assert KERNELS["sm_100a"]["native_k8"] == KERNELS["sm_103a"]["native_k8"]
        assert MODULES[KERNELS["sm_100a"]["native_k8"]]["arches"] == [
            "sm_100a",
            "sm_103a",
        ]
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
            # PDL only changes the launch argument; a round-r5 cell at max_m runs the persistent
            # program its SM count's table names
            assert (
                plan_route(arch, SM_COUNT, max_m, K, True).kernel_key == at.kernel_key
            )
            if at.kind == "small_m":
                # the direct kernel is SM-count agnostic apart from the fast-math cells each SM
                # count's table names (round 23: sm_100a (256, 0) at 148 SMs)
                other = plan_route(arch, 132, max_m, K, True).kernel_key
                assert other.removesuffix("_fm") == at.kernel_key.removesuffix("_fm")
                assert other.endswith("_fm") is cb._fast_math(
                    arch, max_m, K, "small_m", 132
                )
            assert plan_route(arch, SM_COUNT, max_m + 1, K, False).kind != "small_m"


def test_persistent_key_is_the_complete_flag_tuple():
    # M 1024 is the first K1 token count above the small-M table on both architectures
    a = plan_route("sm_103a", SM_COUNT, 1024, 1, True)
    b = plan_route("sm_103a", SM_COUNT, 1024, 1, False)
    for plan in (a, b):
        assert plan.kind == "persistent"
        # round 34: the B300 (1024, 1) cell runs the token-queue build (``_clc``) of its program
        assert re.fullmatch(
            r"persistent:k1_nc\d_d\d_f[01]{15}(_fm)?(_clc)?", plan.kernel_key
        )
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
                r"_fm(_crfnone)?(_crheld)?(_rcp)?(_write)?(_cta2)?"
                r"(_clc(_cla)?(_fpf)?)?(_fpl)?(_fpw)?((?<=_fp[lw])r[1-9][0-9]*)?(_tokb)?$",
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
    and the records' symbols are distinct per architecture."""
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
                tail = re.search(
                    r"(_clc(_cla)?(_fpf)?)?(_fpl)?(_fpw)?((?<=_fp[lw])r[1-9][0-9]*)?(_tokb)?$",
                    key,
                ).group(0)
                write = key.removesuffix(tail).endswith("_write")
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
# the round-2 / round-3 / round-6 / round-20 / round-23 / round-24 / round-25 / round-26 / round-28
# / round-29 cells (among them the single-cell fast-math cells of K 0 and the token-queue ``_clc``
# cells of rounds 28 and 29), the release bands, the nc / depth edges, the native and pair cells)
# and, for K8 on sm_100a, the SM count itself (the release flips at M = SMs + 1).
# ``test_gpu_rows_are_the_route_band_starts`` recomputes the set from the tables.
# fmt: off
_ROUTE_BAND_START_ROWS = {
    0: (
        1, 9, 12, 13, 16, 17, 24, 25, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 192, 193, 256, 257,
        384, 385, 512, 513, 768, 769, 1536, 1537,
    ),
    1: (
        1, 513, 768, 769, 1024, 1025, 1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145,
        8192, 8193, 12288, 12289, 16384,
    ),
    2: (
        1, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537, 2048, 2049,
        3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    3: (
        1, 9, 12, 13, 16, 17, 24, 25, 32, 33, 48, 49, 64, 65, 96, 97, 128, 129, 192, 193, 256, 257,
        384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097,
        6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    4: (
        1, 5, 17, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536, 1537,
        2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    5: (
        1, 5, 9, 12, 13, 16, 17, 33, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024,
        1025, 1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289,
        16384,
    ),
    6: (
        1, 2, 5, 17, 33, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536,
        1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    7: (
        1, 5, 17, 33, 65, 129, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025, 1536,
        1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289, 16384,
    ),
    8: (
        1, 3, 6, 7, 17, 65, 149, 153, 192, 193, 256, 257, 384, 385, 512, 513, 768, 769, 1024, 1025,
        1536, 1537, 2048, 2049, 3072, 3073, 4096, 4097, 6144, 6145, 8192, 8193, 12288, 12289, 16384,
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


@pytest.mark.parametrize("pdl", [False, True])
def test_clc_program_matches_the_reference_and_its_static_program(pdl, monkeypatch):
    """Round 28 on a 148-SM part (B200 / B300), round 31 on a 152-SM part (GB300): the (4096, 6)
    cell launches its ``_clc`` program on 4096 CTAs.  Every token's output matches the FP32
    reference (``out`` starts as NaN, so a token no CTA computed fails), and the output and the
    updated prefix equal, bit for bit, those of the program without the suffix launched on its own
    one-CTA-per-SM grid from the same state."""
    device, arch = _device_arch()
    M, K = 4096, 6
    sm_count = cb.sm_count_for(device)
    if sm_count not in SM_COUNTS:
        pytest.skip("the _clc cells are routed at 148 and 152 SMs")
    if not cb.generated_program_available(device, M, K, enable_pdl=pdl):
        pytest.skip("program not registered for this cell")
    inputs = _make_inputs(M, K)
    pristine = _clone(inputs)
    expected = _clone(inputs)
    reference_kimi_k3_attn_res(
        **{k: v for k, v in expected.items() if k != "num_blocks"}, num_blocks=K
    )
    runner = prepare_kimi_k3_attn_res(
        **{k: v for k, v in inputs.items() if k != "num_blocks"},
        num_blocks=K,
        enable_pdl=pdl,
    )
    assert runner.plan.kernel_key == _CLC_KEYS[K] and runner.plan.grid_x == M, (
        runner.plan
    )
    assert runner.plan.fallback_from is None
    assert runner.launch() is inputs["out"]
    torch.cuda.synchronize()
    _check(inputs, expected)
    static_key = _CLC_KEYS[K].removesuffix("_clc")
    static_plan = cb._persistent_plan_for_key(arch, sm_count, M, K, pdl, static_key)
    assert static_plan.grid_x == sm_count, static_plan
    monkeypatch.setattr(cb, "plan_route", lambda *args, **kwargs: static_plan)
    static = _clone(pristine)
    static_runner = prepare_kimi_k3_attn_res(
        **{k: v for k, v in static.items() if k != "num_blocks"},
        num_blocks=K,
        enable_pdl=pdl,
    )
    assert static_runner.module_name == KERNELS[arch][static_key]
    static_runner.launch()
    torch.cuda.synchronize()
    for name in ("out", "prefix", "blocks"):
        assert torch.equal(
            static[name].view(torch.int16), inputs[name].view(torch.int16)
        ), name


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
        (256, 4),
        (256, 7),
        (1024, 1),
        (1024, 4),
        (4096, 1),
        (4096, 4),
        (4096, 7),
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
    key = runner.plan.kernel_key
    tail = re.search(
        r"(_clc(_cla)?(_fpf)?)?(_fpl)?(_fpw)?((?<=_fp[lw])r[1-9][0-9]*)?(_tokb)?$", key
    ).group(0)
    assert key.removesuffix(tail).endswith("_write")
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
