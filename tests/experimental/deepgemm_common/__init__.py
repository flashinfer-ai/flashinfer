"""Shared random-input builders and references for the DeepGEMM-family tests."""

from .random_inputs import (
    E2M1_VALUES,
    TOLERANCES,
    assert_close,
    e2m1_pack,
    e2m1_unpack,
    random_fp4_operand,
    random_fp8_block2d,
    random_fp8_blockwise,
    random_fp8_ue8m0,
    reference_gemm,
    seeded_generator,
    ue8m0_pack_words,
    ue8m0_unpack_words,
)

__all__ = [
    "E2M1_VALUES",
    "TOLERANCES",
    "assert_close",
    "e2m1_pack",
    "e2m1_unpack",
    "random_fp4_operand",
    "random_fp8_block2d",
    "random_fp8_blockwise",
    "random_fp8_ue8m0",
    "reference_gemm",
    "seeded_generator",
    "ue8m0_pack_words",
    "ue8m0_unpack_words",
]
