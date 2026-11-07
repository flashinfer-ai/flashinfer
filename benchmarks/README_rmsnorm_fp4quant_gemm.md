# RMSNorm/NVFP4 producer before a real cuBLASLt GEMM

This caller measures a synthetic BF16 single-layer workload:
`RMSNorm + NVFP4 quantization -> cuBLASLt NVFP4 GEMM -> BF16 output`.
It is not connected to a model or serving stack. It measures layer latency,
not model throughput, output quality, or service latency.

The installed CuTe implementation is compared with the unmodified CuTe source
from commit `5d0c89eacae6ca08f2a1ce92eba557bbad7a1bfc`. Both preserve the
existing BF16 `x * weight` intermediate rounding. The Triton prototype's
different FP32 arithmetic is not used for this comparison.

The optimization aligns each thread's reduction fragment with a 16-element
quantization block, then reuses the BF16 registers during scale generation and
packing. This removes the explicit shared staging and second X load. K=4096
uses 128 threads per row (two scale blocks per thread); K=7168 retains its
128-thread choice after a 64-thread variant regressed. A final CTA barrier is
unnecessary on this path: the reduction already synchronizes partial sums,
and there are no later shared-memory writes. The scale-block loop is statically
unrolled for register indexing. FP16/MXFP4 fallbacks retain their arithmetic
but share this unrolling change; their performance has not been established
by the two-width full-chain matrix.

## Reproduce

Install FlashInfer's development dependencies, including CuTe DSL, in a
CUDA 13 environment with an NVFP4-capable GPU. The recorded GPU is RTX 5090
(SM120). The optimization is restricted to SM120 BF16 NVFP4, single-CTA rows.
Other architectures have not been performance-validated here.

From the repository root, choose a writable directory outside the package:

```bash
export CHAIN_WORK=/tmp/flashinfer-rmsnorm-chain
mkdir -p "$CHAIN_WORK"
git show 5d0c89eacae6ca08f2a1ce92eba557bbad7a1bfc:flashinfer/cute_dsl/rmsnorm_fp4quant.py > "$CHAIN_WORK/baseline.py"

# CUDA_LIB_DIR should contain libcublasLt.so.13 and libcudart.so.13.
export CUDA_LIB_DIR="$CUDA_HOME/lib64"
g++ -x c++ -shared -fPIC -O3 benchmarks/rmsnorm_fp4quant_cublaslt.cu -x none \
  -I"$CUDA_HOME/include" -Wl,-rpath,"$CUDA_LIB_DIR" \
  "$CUDA_LIB_DIR/libcublasLt.so.13" "$CUDA_LIB_DIR/libcudart.so.13" \
  -o "$CHAIN_WORK/bridge.so"

python benchmarks/bench_rmsnorm_fp4quant_gemm.py \
  --baseline-source "$CHAIN_WORK/baseline.py" --library "$CHAIN_WORK/bridge.so" \
  --full --seed 73 --scale 32 > "$CHAIN_WORK/forward-s73-g32.jsonl"
```

The bridge contains host calls only; no CUDA device kernel is compiled by g++.
For pip CUDA installations, set `CUDA_LIB_DIR` to the toolkit's `lib` directory.
Repeat with seeds 73/109, scales 1/32, and both normal order and `--reverse`:
15 shapes × 2 seeds × 2 scales × 2 execution orders = 120 paired cases.
Use `--allocate-outputs` for a separate allocation control. Do not pool its
measurements with preallocated-buffer measurements.

## Measurement contract

- Shapes are `(M, N, K)`. The matrix includes small decode batches and large
  row counts; there is no workload-frequency weighting or model attribution.
- Both producers use identical inputs, RMSNorm weights, activation scale,
  FP4 GEMM weights, output buffer, workspace and cuBLASLt algorithm instance.
  The primary measurement additionally uses the same activation/scale buffer
  addresses. The allocation control allocates producer outputs separately.
  Both use the original API's default device-dependent PDL setting.
- Weight quantization, allocations, compilation, heuristic selection and
  correctness checks happen outside the timed CUDA Graphs. This does not
  measure eager Python/dispatch/allocation overhead.
- Each CUDA Graph contains 50 actual producer-plus-GEMM chains. CUDA events
  enclose 100 graph replays; six timing rounds alternate which implementation
  runs first. Report the median for each producer and their ratio. These are
  direct complete-chain timings, not sums of isolated kernel measurements.
- Inputs/weights are reused. This is a resident-input repeated-graph benchmark,
  not a cold-cache benchmark. GPU clocks are observed, not locked.
- GEMM computes `Y^T = W X^T`: both operand data and scale pointers are swapped
  to produce contiguous `(M, N)` BF16 output. The same heuristic result is used
  on both sides. Global scaling is undone by
  `alpha = 1 / (activation_global_scale * weight_global_scale)`.
- An independent unpacker decodes packed E2M1 data and swizzled E4M3 scales;
  256 sampled output entries are checked against FP64 dot products of the
  actual quantized operands. All output elements are also checked against
  the baseline chain. These checks establish implementation agreement, not
  equivalence to an unquantized model.

Raw records include source/library hashes, every timing sample, selected GEMM
algorithm ID and numerical errors. Small gains must be interpreted alongside
run-order sensitivity, clocks, and the separate allocation control.

## Correctness and memory checks

```bash
python -m pytest tests/norm/test_rmsnorm_fp4_quant_cute_dsl.py -q
compute-sanitizer --tool memcheck --error-exitcode 91 \
  python -m pytest tests/norm/test_rmsnorm_fp4_quant_cute_dsl.py -q \
  -k 'padded_rows_graph_replay or global_scale_value_consistency or nvfp4_swizzled_vs_unswizzled or auto_allocation_matches_preallocated'
compute-sanitizer --tool racecheck --error-exitcode 92 \
  python -m pytest tests/norm/test_rmsnorm_fp4_quant_cute_dsl.py -q \
  -k padded_rows_graph_replay
```
