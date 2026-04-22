"""
fi_trace + sglang example: run one inference pass in sglang with the
flashinfer backend and verify trace JSONs are produced.

sglang calls flashinfer APIs (rmsnorm, RoPE, attention, GEMM, activation,
sampling) during a forward pass; every ``@flashinfer_api(trace=...)``
decorated call writes a trace JSON when ``FLASHINFER_TRACE_DUMP=1`` is set.

Uses the locally cached Llama-3.2-3B-Instruct. One inference pass (prefill
+ one decode step) is sufficient to exercise most of the instrumented
flashinfer APIs.
"""

import os
import shutil
from pathlib import Path


# Must be set before any flashinfer / sglang import.
SAVE_DIR = Path(__file__).parent / "fi_trace_out_sglang"
os.environ["FLASHINFER_TRACE_DUMP_DIR"] = str(SAVE_DIR)
os.environ["FLASHINFER_TRACE_DUMP"] = "1"
# Disable cubin cache download to avoid network hit.
os.environ.setdefault("SGLANG_SKIP_CUBIN_DOWNLOAD", "1")

if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)

from sglang.srt.entrypoints.engine import Engine  # noqa: E402


def main() -> None:
    model = os.environ.get("FI_TRACE_SGLANG_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    print(f"Loading sglang Engine with model={model} (attention_backend=flashinfer)")
    engine = Engine(
        model_path=model,
        attention_backend="flashinfer",
        disable_cuda_graph=True,  # keep the first call on the Python path
        mem_fraction_static=0.5,
        tp_size=1,
        disable_radix_cache=True,
        log_level="warning",
    )

    prompts = ["The capital of France is"]
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 4,
        "top_k": 50,
        "top_p": 0.9,
    }
    print("Running one inference pass…")
    outputs = engine.generate(prompts, sampling_params)
    for p, out in zip(prompts, outputs, strict=True):
        text = out.get("text") if isinstance(out, dict) else out
        print(f"  prompt: {p!r}")
        print(f"  output: {text!r}")

    engine.shutdown()

    json_files = sorted(SAVE_DIR.glob("*.json"))
    print()
    print(f"Produced {len(json_files)} trace JSON files in {SAVE_DIR}:")
    for f in json_files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
