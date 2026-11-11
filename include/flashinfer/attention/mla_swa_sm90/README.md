# SM90 BF16 sliding-window MLA

The public entry is `flashinfer.mla.batch_decode_with_kv_cache_mla_swa`.
This is a dense Page64 latent cache, not SM120's sparse-index packed FP8 API.

## Qualified contract

- SM90 with cluster launch and at least 223584 bytes opt-in dynamic shared memory.
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

## Frozen tactic catalogue

| ID | M | S | Mainloop | Reduction |
|---|---:|---:|---|---|
|0|8|9|Register softmax, owner PV|Global FP32 + combine|
|1|8|5|Register softmax, owner PV|Global FP32 + combine|
|2|8|3|Register softmax, owner PV|DSM FP32, row-packed|
|3|16|2|Helper softmax, 4 lookahead slices|DSM FP32, natural fragment|
|4|32|2|All-WG softmax|DSM FP32, natural fragment|
|5|32|1|All-WG softmax|Direct|

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
