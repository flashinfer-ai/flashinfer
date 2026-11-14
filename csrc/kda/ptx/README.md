# Static PTX Kimi Delta Attention

The original four SM103a PTX programs are imported from
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

- Remove approximate split planning and expected-norm substitution. Piece
  handoffs exchange complete FP32 state. Fixed-split correction retains all
  tokens, using TF32 products for final-state composition and BF16 state
  operands for output correction, both with FP32 accumulation.
- Clear handoff flags before every launch, including eager launches after
  CUDA Graph replay.
- Prefer driver JIT for PTX ISA 9.2 when the CUDA driver supports 13.2 or
  newer. Older drivers retain the ptxas >= 13.2 cubin fallback. Assembled
  cache identity includes the PTX and assembler binary; temporary outputs
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

## Optimized INT21 schedules

The four optimized PTX payloads are recorded in `manifest.json` under
`optimized_programs`, with SHA-256 checksums. Publication comments are
normalized; the instruction stream is unchanged and has a separate checksum. They reuse the shared
host shim and per-plan TMA descriptor ownership.

On a 148-SM SM103 device, the six INT21 layouts use
`flashinfer/kda_kernels/ptx/optimized/`. Other layouts retain the original
scheduler. The public API and explicit `backend="ptx"` selection are unchanged.

- A fixed sequence splits into an exact prefix and parallel zero-state and
  identity-state tail recurrences. A Triton correction kernel composes the
  tail output and final FP32 state with the prefix state; no recurrent history
  is discarded. State composition uses TF32 products, and output correction
  uses BF16 state operands, with FP32 accumulation and the validated numeric
  tolerances.
- Mixed and uniform layouts use revised piece packing, state handoffs, and
  the updated uniform-family PTX pipeline. Partial pieces use identity-token
  padding with masked output stores.
- The PTX pipeline prefetches V into L2, overlaps state conversion with MMA,
  and uses the retained deferred-decay variants.
- Each prepared plan owns its mutable workspace. Preparation does not execute
  the recurrence or modify the caller's state. The first eager call warms the
  plan before capture. Native FP32 stores update the caller's state directly,
  after the corresponding chain has consumed its initial state.

Approximate splitting and expected-norm substitution are disabled by the
adapter. Planning, compilation, and descriptor upload remain outside graph
capture. Stream dependencies join dual-tail completion before launching
fixed-split correction, preventing correction CTAs from starving unscheduled
dual producers. The redundant dual-completion counters are disabled. The benchmark includes all captured GPU work, including tail
correction. It reports GPU activity spans rather
than CPU-inclusive end-to-end latency.

The optimized manifest pins the first public export in
[`smoothsmooth/flashinfer`, revision `93bb64c3`](https://github.com/smoothsmooth/flashinfer/tree/93bb64c3e8352e696aaa153718d8716fe85b8c29/csrc/kda/ptx).
These checked-in PTX files are the authoritative hand-optimized assembly
source, rather than output reproducible from a separate high-level generator.
The manifest records both file checksums and comment-normalized instruction
checksums. Rebuild a native image with CUDA 13.2 or newer, for example:

```bash
ptxas -arch=sm_103a -o optimized.cubin csrc/kda/ptx/optimized_varlen_uniform_h64_t60.ptx
```
