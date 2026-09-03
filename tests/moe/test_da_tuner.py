"""Host-only tests for distribution-aware MoE tuning policy."""

from flashinfer.fused_moe.da_tuner import FullOpMeasurementCache


def test_full_op_measurement_pair_uses_abba_order_and_reuses_cache():
    measurements = FullOpMeasurementCache()
    order = []
    first_values = iter((1.0, 3.0))
    second_values = iter((2.0, 4.0))

    def measure_first():
        order.append("a")
        return next(first_values)

    def measure_second():
        order.append("b")
        return next(second_values)

    result = measurements.measure_counterbalanced_pair(
        ("shape", "da"),
        measure_first,
        ("shape", "noda"),
        measure_second,
    )

    assert order == ["a", "b", "b", "a"]
    assert result == (2.0, 3.0)
    assert measurements.count == 2
    assert measurements.measure_counterbalanced_pair(
        ("shape", "da"),
        measure_first,
        ("shape", "noda"),
        measure_second,
    ) == (2.0, 3.0)
    assert order == ["a", "b", "b", "a"]
