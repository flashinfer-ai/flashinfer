from types import SimpleNamespace

from flashinfer.gemm.kernels.cutile._tune_select import TIE_REL_TOL, rank_measurements


def _measure(mean_us, error_margin_us, occupancy, block_m=64, block_n=64):
    return SimpleNamespace(
        mean_us=mean_us,
        error_margin_us=error_margin_us,
        config=SimpleNamespace(
            occupancy=occupancy,
            num_ctas=1,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=64,
            GROUP_SIZE_M=8,
        ),
    )


def _key(cfg):
    return (
        cfg.occupancy,
        cfg.num_ctas,
        cfg.BLOCK_M * cfg.BLOCK_N,
        cfg.BLOCK_M,
        cfg.BLOCK_K,
        cfg.GROUP_SIZE_M,
    )


def test_ci_overlap_tie_prefers_key_order():
    # occupancy=8 measures fastest but within combined error margins of
    # occupancy=1, so the deterministic key decides.
    measurements = [
        _measure(52.3, 0.5, occupancy=8),
        _measure(52.6, 0.5, occupancy=1),
    ]
    ranked = rank_measurements(measurements, _key)
    assert [m.config.occupancy for m in ranked] == [1, 8]


def test_rel_tol_floor_catches_narrow_ci():
    # Gap exceeds the combined error margins but stays under TIE_REL_TOL of
    # the best mean, so it is still treated as a tie.
    best, second = 51.4, 52.0
    assert second - best > 0.2 + 0.3
    assert second - best < TIE_REL_TOL * best
    measurements = [
        _measure(best, 0.2, occupancy=8),
        _measure(second, 0.3, occupancy=1),
    ]
    ranked = rank_measurements(measurements, _key)
    assert [m.config.occupancy for m in ranked] == [1, 8]


def test_clear_winner_stays_first():
    measurements = [
        _measure(50.0, 0.1, occupancy=8),
        _measure(60.0, 0.1, occupancy=1),
    ]
    ranked = rank_measurements(measurements, _key)
    assert [m.config.occupancy for m in ranked] == [8, 1]


def test_non_tied_tail_keeps_latency_order():
    measurements = [
        _measure(52.3, 0.5, occupancy=8),
        _measure(52.6, 0.5, occupancy=1),
        _measure(70.0, 0.5, occupancy=1, block_m=256),
        _measure(60.0, 0.5, occupancy=4, block_m=128),
    ]
    ranked = rank_measurements(measurements, _key)
    assert [m.mean_us for m in ranked] == [52.6, 52.3, 60.0, 70.0]


def test_input_order_does_not_matter():
    measurements = [
        _measure(52.6, 0.5, occupancy=1),
        _measure(60.0, 0.5, occupancy=4, block_m=128),
        _measure(52.3, 0.5, occupancy=8),
    ]
    forward = rank_measurements(measurements, _key)
    backward = rank_measurements(list(reversed(measurements)), _key)
    assert [m.mean_us for m in forward] == [m.mean_us for m in backward]
    assert forward[0].config.occupancy == 1


def test_all_measurements_are_returned():
    measurements = [_measure(50.0 + i, 0.5, occupancy=1 + i % 4) for i in range(8)]
    ranked = rank_measurements(measurements, _key)
    assert len(ranked) == len(measurements)
    assert {id(m) for m in ranked} == {id(m) for m in measurements}


def test_single_measurement():
    measurements = [_measure(52.3, 0.5, occupancy=8)]
    ranked = rank_measurements(measurements, _key)
    assert ranked == measurements
