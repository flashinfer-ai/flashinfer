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
    NATIVE_GRID,
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
SM_COUNT = 148
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
    ("sm_100a", 256, 8, False, "native", "native_k8_nc3_d2_ws288_grid148", NATIVE_GRID),
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
        NATIVE_GRID,
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
        3 * SM_COUNT,
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
        NATIVE_GRID,
    ),
    ("sm_103a", 256, 8, True, "native", "native_k8_nc3_d2_ws288_grid148", NATIVE_GRID),
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


@pytest.mark.parametrize("arch,M,K,pdl,kind,schedule_id,grid_x", POLICY_ROWS)
def test_plan_route_policy(arch, M, K, pdl, kind, schedule_id, grid_x):
    plan = plan_route(arch, SM_COUNT, M, K, pdl)
    assert plan.kind == kind
    assert plan.schedule_id == schedule_id
    assert plan.grid_x == grid_x
    assert plan.use_pdl is pdl
    # PDL is a launch argument of every generated program, not a program axis.
    assert "pdl" not in plan.kernel_key
    assert f".pdl{int(pdl)}." in plan.route_id
    assert plan.route_id.startswith(schedule_id + ".")


@pytest.mark.parametrize("arch", ("sm_100a", "sm_103a"))
@pytest.mark.parametrize("pdl", (False, True))
@pytest.mark.parametrize(
    "M,K", ((1, 1), (16, 7), (64, 4), (256, 7), (1024, 1), (4096, 4))
)
def test_plan_route_snapshot_write_takes_the_small_m_write_programs(arch, M, K, pdl):
    """Round r4: the block-boundary snapshot write (block K written) runs the small-M write
    programs at every M - the cluster / chunk bands of the dense family inside its M table,
    one CTA per token above it."""
    plan = cb._plan_route_exact(arch, SM_COUNT, M, K, pdl, block_write_idx=K)
    cluster = cb._small_m_cluster(arch, M, K)
    nc = cb._small_m_sources_per_chunk(arch, M, K)
    nc_suffix = "" if nc is None else f"_nc{nc}"
    family = "direct" if cluster == 1 else f"cluster{cluster}"
    assert plan.kind == "small_m"
    assert plan.kernel_key == f"small_m_{family}:k{K}{nc_suffix}_write"
    assert plan.grid_x == M * cluster and plan.threads == 256 // cluster
    assert plan.schedule_id.endswith(f"_regres_fp32x2{nc_suffix}_write")
    assert f".k{K}.delta1.write1.norm1.pdl{int(pdl)}." in plan.route_id
    assert plan.fallback_from is None
    with pytest.raises(ValueError):
        cb._plan_route_exact(arch, SM_COUNT, M, K, pdl, block_write_idx=K - 1)
    with pytest.raises(ValueError):
        cb._plan_route_exact(arch, SM_COUNT, M, K, pdl, block_write_idx=MAX_BLOCKS)


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


def test_native_ports_require_148_sms():
    # M = 256 / K = 8 is a native_k8 cell on 148 SMs (M = 1 belongs to the small-M direct kernel)
    assert plan_route("sm_100a", SM_COUNT, 256, 8, False).kind == "native"
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


@pytest.mark.parametrize("arch", ARCHES)
def test_registered_keys_cover_the_measured_grid_when_programs_exist(arch):
    if not MODULES:
        pytest.skip("no generated programs registered in this checkout")
    for pdl in (False, True):
        for M in TOKEN_COUNTS:
            for K in PRIMARY_K:
                plan = plan_route(arch, SM_COUNT, M, K, pdl)
                assert plan.kernel_key in KERNELS[arch], (
                    arch,
                    M,
                    K,
                    pdl,
                    plan.kernel_key,
                )
        for M in (1, 4096):
            for K in range(MAX_BLOCKS + 1):
                plan = plan_route(arch, SM_COUNT, M, K, pdl)
                assert plan.kernel_key in KERNELS[arch], (
                    arch,
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


@pytest.mark.parametrize("arch", ARCHES)
def test_every_common_path_route_resolves_to_a_registered_program(arch):
    if not MODULES:
        pytest.skip("no generated programs registered in this checkout")
    registered = KERNELS[arch]
    for pdl in (False, True):
        for K in range(MAX_BLOCKS + 1):
            for M in DENSE_TOKEN_COUNTS:
                exact = cb._plan_route_exact(arch, SM_COUNT, M, K, pdl)
                plan = plan_route(arch, SM_COUNT, M, K, pdl)
                cell = (arch, M, K, pdl, exact.kernel_key, plan.kernel_key)
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
            assert plan_route(arch, SM_COUNT, M, K, pdl).fallback_from is None, (
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


GPU_ROWS = [
    (1, 0),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (16, 8),
    (64, 4),
    (256, 1),
    (256, 8),
    (512, 1),
    (1024, 4),
    # table variants this checkout does not register (registered-variant fallback)
    (7, 5),
    (17, 5),
    (64, 6),
    (128, 7),
    (140, 4),
    (200, 8),
    (300, 3),
    (300, 5),
    (1000, 1),
    (1000, 2),
    (4096, 5),
    (16384, 8),
]


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


@pytest.mark.parametrize("M,K", [(17, 4), (3, 7), (1, 1), (300, 4), (1024, 1)])
@pytest.mark.parametrize("pdl", [False, True])
def test_snapshot_write_matches_reference_on_the_small_m_write_programs(M, K, pdl):
    """Round r4: the block-boundary snapshot write (delta + output norm, block K written) runs a
    small-M write program; the written snapshot and the prefix are bit-exact, the output within
    tolerance, every other block byte preserved."""
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
    assert runner.plan.kind == "small_m"
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
