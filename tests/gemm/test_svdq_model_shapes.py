"""CPU-only checks for the shared SVDQuant model-shape registry."""

from typing import Final

from flashinfer.testing.svdq_model_shapes import (
    MODEL_SHAPE_CASES,
    MODEL_SHAPE_COUNT,
    ShapeCase,
)


H3_TEXT_LENGTHS: Final[tuple[int, ...]] = (537, 1935, 6913)
H3_OMNI_LENGTHS: Final[tuple[int, ...]] = (61056, 73984, 82752)
H3_BLOCK_NK_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (21504, 5376),
    (5376, 7168),
    (28672, 5376),
    (5376, 14336),
)


def test_model_shape_registry_is_immutable() -> None:
    # Given: the shared registry is imported by tests, benchmarks, and tools.
    # When: its public container type is inspected.
    # Then: callers cannot mutate the process-wide source of truth in place.
    assert isinstance(MODEL_SHAPE_CASES, tuple)


def test_h3_official_example_shapes_are_registered() -> None:
    # Given: all recurring Linear shapes from the three official H3 examples.
    expected_h3_shapes: set[ShapeCase] = {
        (m, n, k)
        for m in (*H3_TEXT_LENGTHS, *H3_OMNI_LENGTHS)
        for n, k in H3_BLOCK_NK_SHAPES
    }
    expected_h3_shapes.update((m, 5376, 5120) for m in H3_TEXT_LENGTHS)

    # When: the expected H3 workload is compared with the shared registry.
    missing_shapes = expected_h3_shapes.difference(MODEL_SHAPE_CASES)

    # Then: every H3 shape is available through --shape-set model.
    assert missing_shapes == set()


def test_model_shape_count_includes_h3_batch() -> None:
    # Given: the existing 44 rows and 27 new unique H3 rows.
    expected_shape_count = 71

    # When: the registry's declared and materialized sizes are inspected.
    actual_shape_counts = (MODEL_SHAPE_COUNT, len(MODEL_SHAPE_CASES))

    # Then: both expose the complete four-batch workload.
    assert actual_shape_counts == (expected_shape_count, expected_shape_count)
