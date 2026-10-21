# Static PTX Kimi Delta Attention

These four SM103a PTX programs are imported from
[`NVlabs/kda`, branch `260927-kda-for-kda`](https://github.com/NVlabs/kda/tree/ea37ebaff74c88a2545751dcb8ea8ef6c6251b67),
`ptx/kda_ptx/shims/`. `manifest.json` records the exact source revision and
the SHA-256 of each retained `.ptx` file; their bytes are unchanged. The
imported per-program TVM FFI shims and argument plans are replaced by one
shared host shim, `shim.cc`, and one argument plan, `programs.json`, whose
per-program entries (kernel symbol, embedded-cubin identifier, dynamic shared
memory, TMA box sizes, handoff-flag argument, cluster width) are prepended as
`#define` lines when the shim is compiled for a program.
The original kernel lineage is `humanfia/kda-for-kda` branch
`yahui-2.89x-ptx`, revision `82a6a79`, including its slow-decay FTZ fix.
The MIT license is shipped as `LICENSE.kda-for-kda.txt`.

The public entry point is `flashinfer.recurrent_kda(..., backend="ptx")`.
`flashinfer/kda_kernels/ptx/` contains the imported host scheduling code;
`flashinfer/kda_prefill_ptx.py` adapts it to FlashInfer's state, output,
validation, and workspace contracts. Integration changes:

- Remove approximate split planning and expected-norm substitution. Splits
  exchange the complete FP32 recurrent state.
- Clear handoff flags before every launch, including eager launches after
  CUDA Graph replay.
- Assemble PTX ISA 9.2 with ptxas >= 13.2 and embed the cubin in TVM FFI.
  Cache identity includes the PTX and assembler binary; temporary outputs
  are unique and published atomically outside the installed package.
- Retain each prepared launch's TMA descriptor table and input tensors.
  Descriptor encoding/upload takes place during planning, outside capture.
- Restrict the public contract to H64/H96, D128, SM103a, >= 32 total tokens, sequence lengths
  <= 16384, and finite FP32 initial state with absolute values <= 4096.
  Other shapes remain available through FlashInfer's existing backends.

The six INT21 workloads have 8192 total tokens, H96/H64, and sequence lengths
`[8192]`, `[1300, 547, 2048, 963, 271, 3063]`, or `[1024] * 8`.
Run `benchmarks/bench_recurrent_kda_ptx.py` for per-shape absolute latency,
correctness against FLA, raw MoonshotAI/FlashKDA latency, and geometric mean
speedup. See `docs/api/kda.rst` for installation and an example.
