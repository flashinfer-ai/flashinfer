# Experimental SM120 materialized FP8 prefill

Local, explicit SGLang forward-hook integration for RTX PRO 5000 (SM120).
This is not a public FlashInfer API or an automatically selected backend.
It depends on the matching SGLang FA4 compatibility and one-shot prefix changes.

Opt in through the SGLang forward hook `materialized_fp8_hook.create_hook`,
with this experimental directory on the Python module search path.
Supported layout: 20 heads, Q/K dimension 256, V dimension 256, causal
one-shot MLA prefill, BF16 input/output, E4M3 intermediate tensors and
FP32 accumulation. Queries shorter than 1024 tokens use the existing path.
Decode is unaffected. The model and business-quality limits are documented
in the paired SGLang optimization delivery guide.

Fuses K/rotary concatenation with block-scaled Q/K/V preparation and uses
FP8 QK/PV attention. Initial PV uses K16; later commits carry measured K32,
TMA, instruction/layout and tile-scheduling optimizations separately.

Historical service measurement at concurrency 8: 5.42 to 5.78 QPS (+6.64%).
This measures the whole prefill combination, not isolated kernel operations.
Correctness coverage is in `tests/experimental/mla_fp8_sm120/test_materialized_prefill.py`.
