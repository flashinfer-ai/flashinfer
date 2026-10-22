"""Shared model-shape coverage for the NVFP4 SVDQuant kernels.

The cases in this module are the executable source of truth corresponding to
``docs/model_shape_requirements.md``.  Keep correctness tests and benchmarks on
this single, sorted collection so overlapping model workloads run only once.
"""

from typing import Final, TypeAlias


ShapeCase: TypeAlias = tuple[int, int, int]

MODEL_SHAPE_ALIGNMENT: Final[int] = 32
MODEL_LORA_RANKS: Final[tuple[int, ...]] = (32, 64, 96, 128)
MODEL_SHAPE_COUNT: Final[int] = 71

_H3_TEXT_M_VALUES: Final[tuple[int, ...]] = (537, 1935, 6913)
_H3_OMNI_M_VALUES: Final[tuple[int, ...]] = (61056, 73984, 82752)
_H3_BLOCK_NK_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (21504, 5376),
    (5376, 7168),
    (28672, 5376),
    (5376, 14336),
)


def _build_model_shape_cases() -> tuple[ShapeCase, ...]:
    """Build the deduplicated model registry without mutable module state."""
    cases: set[ShapeCase] = set()

    def add_cases(
        m_values: tuple[int, ...],
        nk_shapes: tuple[tuple[int, int], ...],
    ) -> None:
        cases.update((m, n, k) for m in m_values for n, k in nk_shapes)

    # Qwen-Image image stream and representative text-stream lengths.
    add_cases(
        (64, 256, 512, 1024, 4096, 6889, 9216, 16384),
        ((3072, 3072), (12288, 3072), (3072, 12288)),
    )

    # Wan2.1 14B and Wan2.2 A14B latent-token paths and text K/V projection.
    add_cases(
        (7800, 32760, 75600),
        ((5120, 5120), (13824, 5120), (5120, 13824)),
    )
    cases.add((512, 5120, 5120))

    # Wan2.1 1.3B latent-token paths and text K/V projection.
    add_cases(
        (7800, 32760),
        ((1536, 1536), (8960, 1536), (1536, 8960)),
    )
    cases.add((512, 1536, 1536))

    # Wan2.2 TI2V-5B latent-token paths and text K/V projection. The latter
    # overlaps Qwen-Image's (512, 3072, 3072) case and is deduplicated here.
    add_cases(
        (27280,),
        ((3072, 3072), (14336, 3072), (3072, 14336)),
    )
    cases.add((512, 3072, 3072))

    # MiniMax-H3 official 768p examples.
    add_cases(_H3_TEXT_M_VALUES, _H3_BLOCK_NK_SHAPES)
    add_cases(_H3_OMNI_M_VALUES, _H3_BLOCK_NK_SHAPES)
    add_cases(_H3_TEXT_M_VALUES, ((5376, 5120),))
    return tuple(sorted(cases))


MODEL_SHAPE_CASES: Final[tuple[ShapeCase, ...]] = _build_model_shape_cases()
MODEL_M_VALUES: Final[tuple[int, ...]] = tuple(
    sorted({m for m, _, _ in MODEL_SHAPE_CASES})
)

if len(MODEL_SHAPE_CASES) != MODEL_SHAPE_COUNT:
    raise AssertionError(
        f"expected {MODEL_SHAPE_COUNT} unique model shapes, got {len(MODEL_SHAPE_CASES)}"
    )
if tuple(sorted(set(MODEL_SHAPE_CASES))) != MODEL_SHAPE_CASES:
    raise AssertionError("MODEL_SHAPE_CASES must be sorted and unique")
if any(
    m <= 0 or n <= 0 or k <= 0 or n % MODEL_SHAPE_ALIGNMENT or k % MODEL_SHAPE_ALIGNMENT
    for m, n, k in MODEL_SHAPE_CASES
):
    raise AssertionError(
        f"model shapes must be positive and N/K must be multiples of {MODEL_SHAPE_ALIGNMENT}"
    )
if any(rank <= 0 or rank % MODEL_SHAPE_ALIGNMENT for rank in MODEL_LORA_RANKS):
    raise AssertionError(
        f"model LoRA ranks must be positive multiples of {MODEL_SHAPE_ALIGNMENT}"
    )

__all__ = [
    "MODEL_LORA_RANKS",
    "MODEL_M_VALUES",
    "MODEL_SHAPE_ALIGNMENT",
    "MODEL_SHAPE_CASES",
    "MODEL_SHAPE_COUNT",
    "ShapeCase",
]
