# SM90 BF16 sliding-window MLA

The public entry is `flashinfer.mla.batch_decode_with_kv_cache_mla_swa`.
This is a dense Page64 latent cache, not SM120's sparse-index packed FP8 API.

## Qualified contract

- SM90 with cluster launch and at least 231936 bytes opt-in shared memory.
- BF16 Q `[B, sq, 64, 1088]`, KV `[pages, 64, 1088]`, output `[B, sq, 64, 1024]`.
- Local B in `{1,2,4,8,16}`, sq in `{1,4}`, window_left=512 (513 tokens inclusive).
- CUDA int32 page table and total KV lengths; causal Q positions are `L-sq+i`.
- Q/KV and the WGMMA probability tile are BF16. Scores, PV accumulators
  and all split-reduction payloads/calculations are FP32; final output is BF16.
- Caller-owned byte workspace is conservative across all tuning candidates.
- Warm eagerly before capture; autotune is optional and uses the standard
  FlashInfer context/cache. Serving does not read metadata values on the host.
- No gradients, LSE output, cache quantization, DCP, cross attention, or prefill
  support is implied. Invalid metadata values are caller errors.

## Qualified tactic catalogue

| ID | M | S | Mainloop | Reduction |
|---|---:|---:|---|---|
|0|8|9|Register softmax, owner PV|Global FP32, cooperative or separate combine|
|1|8|5|Register softmax, owner PV|Global FP32, cooperative or separate combine|
|2|8|3|Register softmax, owner PV|Global FP32, cooperative or separate combine|
|3|16|2|Helper softmax, 6 lookahead slices|DSM FP32, natural fragment, per-WG ready/ACK|
|4|32|2|All-WG softmax|DSM FP32, natural fragment|
|5|32|1|All-WG softmax, 2 lookahead slices|Direct|
|6|16|4|Helper softmax, 6 lookahead slices|Global FP32, cooperative or separate combine|

The default uses tactic 6 for `(local B,sq)=(8,1)/(2,4)` (512 query-head
rows). The 1024-row shapes retain tactic 3; increasing splits there crosses
a main-kernel wave boundary and regresses. Autotune may choose any qualified
tactic and uses a versioned cache key. The workspace capacity is unchanged.

For Global FP32 tactics, a cooperative grid fuses the combine when there are
at most 512 query-head rows and the entire grid fits the device's queried
cooperative-launch capacity. Otherwise the same tactic uses a separate FP32
combine kernel. Both paths use the caller's existing workspace and retain FP32
partials until the final BF16 output write. No per-invocation counters or
workspace clearing are required.

Each CTA's query-head rows belong to one query token. Its page range is derived
from that token's causal window, avoiding a request-wide union's extra masked
page when `sq=4`. M32 tactics reuse the dead RoPE KV slice for scores; M32S1
also caches row rescaling factors and prefetches two lookahead slices.

The device implementation retains the validated kernel's arithmetic and
pipeline, with formatting normalized to FlashInfer's style. Torch-dependent
host code is replaced by `launch.cuh` and the TVM-FFI binding.
New tactics must be explicitly qualified; this is not an arbitrary
M x S parameter product. The normal JitSpec is shared by JIT and AOT builds.

## Reproduce

```bash
pytest tests/attention/test_mla_swa_sm90.py tests/trace/test_mla_swa.py
python benchmarks/bench_mla_swa_sm90.py --refcheck --output results/mla_swa.json
```

The benchmark uses only FlashInfer and PyTorch, times the complete operation
including any split combine with interleaved CUDA Graph replays, and checks an
independent FP64 reference. Its PyTorch gather/mask baseline is not SGLang's
Triton fallback; do not compare that baseline's speedup to historical SGLang
fallback speedups. Use `--autotune --autotune-cache results/mla_swa_tuning.json`
to benchmark startup-tuned dispatch instead of the default tactic catalogue.
