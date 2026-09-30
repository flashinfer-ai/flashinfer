from b12x.moe._shared.kernels.w4a16.kernel import (
    _TC_DECODE_MAX_M,
    _TC_DECODE_PACK_COLLIDING_PAIRS,
    _TC_DECODE_PACK_SM_COVERAGE_CAP,
    _TC_DECODE_PACK_SM_COVERAGE_DENOMINATOR,
    _TC_DECODE_PACK_SM_COVERAGE_NUMERATOR,
    _LARGE_BATCH_TILE_CONFIGS,
    _SMALL_BATCH_TILE_CONFIGS,
    _candidate_tile_fits,
    _w4a16_num_regs,
    _w4a16_tc_decode_preferred,
)


def _fits(
    *,
    tile_k: int,
    tile_n: int,
    cta_threads: int,
    allow_qualified_fc2_tile: bool = False,
) -> bool:
    return _candidate_tile_fits(
        problem_n=4096,
        problem_k=512,
        cta_m_blocks=1,
        tile_n=tile_n,
        tile_k=tile_k,
        cta_threads=cta_threads,
        max_shared_mem=1 << 30,
        scale_format="e8m0_k32",
        weight_layout="modelopt",
        weight_bits=4,
        allow_qualified_fc2_tile=allow_qualified_fc2_tile,
    )


def test_wave_balanced_fc2_tile_is_valid_as_an_explicit_pin() -> None:
    assert _fits(
        tile_k=32,
        tile_n=512,
        cta_threads=256,
        allow_qualified_fc2_tile=True,
    )


def test_wave_balanced_fc2_tile_is_rejected_for_fc1() -> None:
    assert not _fits(tile_k=32, tile_n=512, cta_threads=256)


def test_other_sub64_k_tiles_remain_unsupported() -> None:
    assert not _fits(tile_k=32, tile_n=256, cta_threads=128)
    assert not _fits(tile_k=16, tile_n=512, cta_threads=128)


def test_block_48_route_tile_has_a_resource_model() -> None:
    assert (
        _w4a16_num_regs(
            cta_threads=256,
            cta_m_blocks=3,
            cta_n_blocks=8,
            cta_k_blocks=8,
            uses_m_block_8=False,
            weight_layout="trellis_t256",
        )
        == 255
    )


def test_wide_n_single_route_tile_has_a_resource_model() -> None:
    assert (
        _w4a16_num_regs(
            cta_threads=256,
            cta_m_blocks=1,
            cta_n_blocks=16,
            cta_k_blocks=4,
            uses_m_block_8=False,
        )
        == 255
    )


def test_every_generic_route_tile_has_a_resource_model() -> None:
    for route_block_size in (8, 16, 32, 48, 64):
        cta_m_blocks = (route_block_size + 15) // 16
        configs = (
            _LARGE_BATCH_TILE_CONFIGS
            if cta_m_blocks > 1
            else _SMALL_BATCH_TILE_CONFIGS
        )
        for tile_k, tile_n, cta_threads in configs:
            assert _w4a16_num_regs(
                cta_threads=cta_threads,
                cta_m_blocks=cta_m_blocks,
                cta_n_blocks=tile_n // 16,
                cta_k_blocks=tile_k // 16,
                uses_m_block_8=route_block_size == 8,
            ) > 0


def test_tc_decode_planner_keeps_underfilled_direct_route() -> None:
    assert _w4a16_tc_decode_preferred(m=8, topk=6, num_experts=256, sms=188)


def test_tc_decode_planner_caps_route_coverage_on_large_gpus() -> None:
    assert _w4a16_tc_decode_preferred(m=6, topk=8, num_experts=256, sms=188)
    assert not _w4a16_tc_decode_preferred(m=7, topk=8, num_experts=256, sms=188)


def test_tc_decode_coverage_cap_preserves_smaller_gpu_policy() -> None:
    for sms in range(1, _TC_DECODE_PACK_SM_COVERAGE_CAP + 1):
        for m in range(1, _TC_DECODE_MAX_M + 1):
            for topk in (1, 4, 6, 8, 16):
                for num_experts in (1, 64, 128, 256, 512):
                    routed_rows = m * topk
                    uncapped_pack_has_reuse = (
                        routed_rows * _TC_DECODE_PACK_SM_COVERAGE_DENOMINATOR
                        >= sms * _TC_DECODE_PACK_SM_COVERAGE_NUMERATOR
                        and routed_rows * (routed_rows - 1)
                        >= 2
                        * _TC_DECODE_PACK_COLLIDING_PAIRS
                        * num_experts
                    )
                    assert _w4a16_tc_decode_preferred(
                        m=m,
                        topk=topk,
                        num_experts=num_experts,
                        sms=sms,
                    ) is not uncapped_pack_has_reuse


def test_tc_decode_planner_packs_near_full_machine_with_expected_reuse() -> None:
    assert not _w4a16_tc_decode_preferred(m=7, topk=6, num_experts=256, sms=48)
    assert not _w4a16_tc_decode_preferred(m=8, topk=6, num_experts=256, sms=48)


def test_tc_decode_planner_keeps_lower_coverage_direct_route() -> None:
    assert _w4a16_tc_decode_preferred(m=6, topk=6, num_experts=256, sms=48)


def test_tc_decode_planner_keeps_low_collision_direct_route() -> None:
    assert _w4a16_tc_decode_preferred(m=8, topk=6, num_experts=512, sms=48)
