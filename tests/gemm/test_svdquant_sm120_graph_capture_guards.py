"""Source guards for stream attributes that CUDA Graph capture forbids."""

from pathlib import Path
from typing import Final

_BINDING_PATH: Final = (
    Path(__file__).resolve().parents[2]
    / "csrc"
    / "nvfp4_svdquant_gemm_cutlass_sm120.cu"
)


def test_persisting_l2_hints_are_skipped_during_stream_capture() -> None:
    source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "cudaStreamIsCapturing(stream, &capture_status)" in source
    assert "state.persisting_l2t_stream != stream" in source
    assert "if (!is_capturing)" in source

    linear = source[source.index("void nvfp4_svdquant_linear_sm120(") :]
    linear = linear[: linear.index("\n#endif")]
    assert "configure_case7_persisting_l2 = !is_capturing" in linear
    assert linear.count("if (configure_case7_persisting_l2)") == 3
