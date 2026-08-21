import argparse
import sys
from pathlib import Path

import pytest


BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARK_ROOT))

from routines.gemm import _dynamic_mxfp8_problem_bytes, parse_gemm_args  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routine")
    parser.add_argument("--verbose", action="count", default=0)
    return parser


def test_mm_mxfp8_dynamic_quant_defaults_to_trtllm() -> None:
    args = parse_gemm_args(
        [
            "--routine",
            "mm_mxfp8",
            "--m",
            "4",
            "--n",
            "2688",
            "--k",
            "4096",
            "--dynamic_quant",
        ],
        _parser(),
    )
    assert args.backends == ["trtllm"]


@pytest.mark.parametrize(
    ("dynamic_quant_layout", "backend"),
    [
        ("auto", "trtllm"),
        ("8x4", "trtllm"),
        ("128x4", "trtllm"),
        ("128x4", "cute-dsl"),
    ],
)
def test_mm_mxfp8_accepts_supported_dynamic_quant_layout_backend_pair(
    dynamic_quant_layout: str,
    backend: str,
) -> None:
    args = parse_gemm_args(
        [
            "--routine",
            "mm_mxfp8",
            "--m",
            "4",
            "--n",
            "2688",
            "--k",
            "4096",
            "--backends",
            backend,
            "--dynamic_quant",
            "--dynamic_quant_layout",
            dynamic_quant_layout,
        ],
        _parser(),
    )
    assert args.dynamic_quant is True
    assert args.dynamic_quant_layout == dynamic_quant_layout
    assert args.backends == [backend]


@pytest.mark.parametrize(
    ("dynamic_quant_layout", "backend"),
    [("auto", "cute-dsl"), ("8x4", "cute-dsl"), ("128x4", "cutlass")],
)
def test_mm_mxfp8_rejects_dynamic_quant_layout_backend_mismatch(
    dynamic_quant_layout: str,
    backend: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"--dynamic_quant_layout {dynamic_quant_layout} supports only",
    ):
        parse_gemm_args(
            [
                "--routine",
                "mm_mxfp8",
                "--m",
                "4",
                "--n",
                "2688",
                "--k",
                "4096",
                "--backends",
                backend,
                "--dynamic_quant",
                "--dynamic_quant_layout",
                dynamic_quant_layout,
            ],
            _parser(),
        )


def test_dynamic_mxfp8_problem_bytes_includes_quantization_traffic() -> None:
    assert _dynamic_mxfp8_problem_bytes(4, 128, 256, out_itemsize=2) == 38_976


def test_mm_mxfp8_dynamic_quant_rejects_non_bf16_output() -> None:
    with pytest.raises(ValueError, match="bfloat16 output"):
        parse_gemm_args(
            [
                "--routine",
                "mm_mxfp8",
                "--m",
                "4",
                "--n",
                "2688",
                "--k",
                "4096",
                "--backends",
                "trtllm",
                "--out_dtype",
                "float16",
                "--dynamic_quant",
            ],
            _parser(),
        )


def test_mm_mxfp8_dynamic_quant_accepts_fp16_for_fixed_cute_dsl() -> None:
    args = parse_gemm_args(
        [
            "--routine",
            "mm_mxfp8",
            "--m",
            "4",
            "--n",
            "2688",
            "--k",
            "4096",
            "--backends",
            "cute-dsl",
            "--out_dtype",
            "float16",
            "--dynamic_quant",
            "--dynamic_quant_layout",
            "128x4",
        ],
        _parser(),
    )
    assert args.out_dtype == "float16"
