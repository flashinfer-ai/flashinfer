"""Run the hardware checks in tests/experimental from the standalone study."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "tests/experimental/mla_fp8_sm120/check.py"
        ),
        run_name="__main__",
    )
