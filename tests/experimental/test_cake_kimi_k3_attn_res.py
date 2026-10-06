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

import re

import pytest
import torch

from flashinfer.experimental.cake_kimi_k3_attn_res import cake_backend as cb
from flashinfer.experimental.cake_kimi_k3_attn_res.cake_backend import (
    ARCHES,
    HIDDEN_SIZE,
    MAX_BLOCKS,
    NATIVE_SM_COUNTS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    common_path_eligible,
    plan_route,
    reference_kimi_k3_attn_res,
)
from flashinfer.experimental.cake_kimi_k3_attn_res.cake_jit import KERNELS, MODULES
from flashinfer.kimi_k3_attn_res import kimi_k3_attn_res, prepare_kimi_k3_attn_res

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
    plan = plan_route(arch, sm_count, M, K, pdl)
    assert plan.kind == kind
    assert plan.schedule_id == schedule_id
    assert plan.grid_x == grid_x
    assert plan.use_pdl is pdl
    # PDL is a launch argument of every generated program, not a program axis.
    assert "pdl" not in plan.kernel_key
    assert f".pdl{int(pdl)}." in plan.route_id
    assert plan.route_id.startswith(schedule_id + ".")


@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize(
    "M,K,early",
    (
        # K2: the early-release band M768..2048 (+ the M4096 cell); held below, between and above.
        (767, 2, False),
        (768, 2, True),
        (1536, 2, True),
        (2048, 2, True),
        (2049, 2, False),
        (3072, 2, False),
        (4096, 2, True),
        (4097, 2, False),
        (6144, 2, False),
        # K4: the nc5 band M513..1023 + the pow2 cells {512, 1024, 2048, 4096}; held at 1025..2047 and above 4096.
        (512, 4, True),
        (513, 4, True),
        (768, 4, True),
        (1023, 4, True),
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
    flags = plan.kernel_key.rsplit("_f", 1)[1]
    assert (flags[0] == "1") is early, (plan.kernel_key, early)


@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize(
    "arch,M,nc,depth",
    (
        # sm_100a: nc4 depth 3 from the band edge (round r4 closing screening); the r1 M4096 cell keeps nc3 depth 3.
        ("sm_100a", 1535, 4, 2),
        ("sm_100a", 1536, 4, 3),
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
    edges are the dispatcher / evaluator / FI mirror constants."""
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
        cluster = cb._small_m_cluster(arch, M, K)
        nc = cb._small_m_sources_per_chunk(arch, M, K)
        nc_suffix = "" if nc is None else f"_nc{nc}"
        family = "direct" if cluster == 1 else f"cluster{cluster}"
        assert plan.kind == "small_m"
        assert plan.kernel_key == f"small_m_{family}:k{K}{nc_suffix}_write"
        assert plan.grid_x == M * cluster and plan.threads == 256 // cluster
        assert plan.schedule_id.endswith(f"_regres_fp32x2{nc_suffix}_write")
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


def test_snapshot_write_fallback_stays_in_the_write_family(monkeypatch):
    """A registered-variant fallback of a snapshot-write plan only ever selects another
    small-M write program (the snapshot store exists nowhere else)."""
    table = {
        "small_m_direct:k7_write": "m1",
        "small_m_direct:k7": "m2",
        "persistent:k7_nc4_d2_f110000000000000": "m3",
    }
    monkeypatch.setattr(cb, "KERNELS", {"sm_100a": table})
    exact = cb._plan_route_exact("sm_100a", SM_COUNT, 64, 7, False, block_write_idx=7)
    assert exact.kernel_key not in table
    resolved = cb._resolve_registered(exact, SM_COUNT, 64)
    assert resolved.kernel_key == "small_m_direct:k7_write"
    assert resolved.fallback_from == exact.kernel_key
    assert (
        resolved.route_id.endswith(".registered_fallback")
        and ".write1." in resolved.route_id
    )
    monkeypatch.setattr(cb, "KERNELS", {"sm_100a": {"small_m_direct:k7": "m2"}})
    unresolved = cb._resolve_registered(exact, SM_COUNT, 64)
    assert (
        unresolved.kernel_key == exact.kernel_key and unresolved.fallback_from is None
    )


def test_plan_route_bootstrap_variants():
    plan = plan_route(
        None, 0, 17, 4, False, has_delta=False, block_write_idx=4, common_path=False
    )
    assert plan.kind == "direct"
    assert plan.kernel_key == "direct:k4_d0_w1_n1"
    assert plan.grid_x == 17 and plan.threads == 256
    plan = plan_route(
        "sm_100a", SM_COUNT, 7, 8, True, apply_output_norm=False, common_path=False
    )
    assert plan.kernel_key == "direct:k8_d1_w0_n0"


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
                cluster = cb._small_m_cluster(arch, m, K)
                nc = cb._small_m_sources_per_chunk(arch, m, K)
                suffix = "" if nc is None else f"_nc{nc}"
                key = (
                    f"small_m_direct:k{K}{suffix}"
                    if cluster == 1
                    else f"small_m_cluster{cluster}:k{K}{suffix}"
                )
                assert at.schedule_id.endswith(suffix)
                assert (at.kind, at.kernel_key, at.grid_x, at.threads) == (
                    "small_m",
                    key,
                    m * cluster,
                    cb.DIRECT_THREADS // cluster,
                )
            at = plan_route(arch, SM_COUNT, max_m, K, False)
            # the direct kernel is SM-count agnostic and PDL only changes the launch argument
            assert plan_route(arch, 132, max_m, K, True).kernel_key == at.kernel_key
            assert plan_route(arch, SM_COUNT, max_m + 1, K, False).kind != "small_m"


def test_persistent_key_is_the_complete_flag_tuple():
    # M 1024 is the first K1 token count above the small-M table on both architectures
    a = plan_route("sm_103a", SM_COUNT, 1024, 1, True)
    b = plan_route("sm_103a", SM_COUNT, 1024, 1, False)
    for plan in (a, b):
        assert plan.kind == "persistent"
        assert re.fullmatch(r"persistent:k1_nc\d_d\d_f[01]{15}", plan.kernel_key)
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
        assert record["tma_workspace_bytes"] == 0
        assert record["arches"] and set(record["arches"]) <= set(ARCHES)


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
                assert plan.kind in ("small_m", "native", "k0_tma", "persistent"), cell
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
                elif plan.schedule_id.endswith("_one_token_per_cta"):
                    assert (plan.grid_x, plan.threads) == (M, cb.PERSISTENT_THREADS), (
                        cell
                    )
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


def _route_band_start_rows(m_max: int = 16384):
    """(M, K) of every token count where the dense route changes its kernel key on any
    architecture / PDL setting: the first row of every route band, so each registered
    program is exercised at the edge of the band that selects it."""
    rows = set()
    for arch in ARCHES:
        for sm_count in SM_COUNTS:
            for K in range(MAX_BLOCKS + 1):
                for pdl in (False, True):
                    previous = None
                    for M in range(1, m_max + 1):
                        key = plan_route(arch, sm_count, M, K, pdl).kernel_key
                        if key != previous:
                            rows.add((M, K))
                            previous = key
    return rows


GPU_ROWS = sorted(
    _route_band_start_rows()
    | {
        (16, 8),
        (64, 4),
        (256, 1),
        (512, 1),
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
    assert runner.plan.kind in ("small_m", "native", "k0_tma", "persistent")
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
        (17, 4, True, 2, True),
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
    "M,K", [(17, 4), (3, 7), (1, 1), (300, 4), (1024, 1), (1024, 4)]
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


def test_row_padded_layout_takes_the_bootstrap():
    device, arch = _device_arch()
    inputs = _make_inputs(5, 4, row_padding=64)
    assert not common_path_eligible(
        **{k: v for k, v in inputs.items() if k not in ("num_blocks",)},
        num_blocks=4,
        block_write_idx=-1,
    )
    if not cb.generated_program_available(device, 5, 4, common_path=False):
        pytest.skip("bootstrap program not registered")
    expected = _clone(inputs)
    reference_kimi_k3_attn_res(
        **{k: v for k, v in expected.items() if k != "num_blocks"}, num_blocks=4
    )
    kimi_k3_attn_res(
        **{k: v for k, v in inputs.items() if k != "num_blocks"}, num_blocks=4
    )
    torch.cuda.synchronize()
    _check(inputs, expected)


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
    with pytest.raises(ValueError):
        prepare_kimi_k3_attn_res(
            **{k: v for k, v in inputs.items() if k != "num_blocks"},
            num_blocks=4,
            backend="triton",
        )
