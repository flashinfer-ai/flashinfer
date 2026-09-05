# PatchShift BF16 3x3x3 Conv3d on SM100a

## Scope

The proposed operator is one generic convolution compute primitive. It has no
model-specific cache, concatenation, fallback, or layout-conversion behavior.

The initial contract is intentionally narrow:

- input: contiguous BF16 NDHWC `[N, D, H, W, C]`;
- weight: a separately prepacked representation derived from one logical
  3x3x3 weight tensor;
- output: contiguous BF16 NDHWC `[N, D, H, W, K]`;
- FP32 accumulation;
- kernel 3x3x3, padding 1, stride 1, dilation 1, groups 1;
- positive extents with `C % 8 == 0`;
- NVIDIA SM100a/B200 only until other architectures are independently tested.

Bias and training/backward support are out of scope for the first PR.

## Layering

```text
Python API and validation                  flashinfer/conv3d/
JIT source selection and sm_100a flags     flashinfer/jit/
TVM-FFI, streams, descriptors, dispatch    csrc/patchshift_conv3d/
framework-independent device code          include/flashinfer/conv3d/patchshift/
correctness tests                          tests/conv3d/
performance comparison                     benchmarks/bench_patchshift_conv3d.py
```

The device implementation remains one CUDA translation unit through
`kernels.cuh`. Splitting the detail headers into separately compiled objects is
not a neutral refactor for this template-heavy kernel and requires PTX plus
resource-table comparison.

## Weight and workspace lifecycle

The standalone program packed weights and allocated TensorMap storage inside
its process. A library API must instead make these lifetimes explicit:

1. prepack a static logical weight once;
2. allocate descriptor workspace owned by the caller or an explicit plan and
   bind it to that packed-weight storage;
3. update pointer-dependent input TensorMaps when the input address changes,
   and prepare a new workspace before changing the packed-weight address;
4. preserve caller-stream ordering for every launch, including routes whose
   disjoint main and auxiliary tiles execute on plan-owned internal streams;
5. perform no hidden synchronization or process termination in the hot launch.

Descriptor preparation explicitly synchronizes its caller stream because it
copies stack-built CUDA TensorMap descriptors to device storage. It is a cold
setup operation and must be called before CUDA graph capture. The subsequent
launch updates input addresses on the current stream. For exact M32/M64 output
tails and the measured C96 P-tail route, it then records a fork on that stream,
runs the disjoint main and auxiliary intervals on two nonblocking streams owned
by the prepared workspace, and joins both completion events back into the
caller's stream. The streams and events are materialized during preparation, so
the same topology is legal inside an outer CUDA Graph capture. Other routes run
entirely on the current stream.

One prepared workspace must not be submitted concurrently by independent host
threads or caller streams. Prepare a separate workspace for each concurrent
call site or separately captured graph. The ID18 cluster-A spatial-edge route
remains sequential in the library because its standalone overlap relies on a
CUDA Graph launch-completion dependency rather than an ordinary stream event.

This separation keeps weight transformation outside the measured compute path
and makes CUDA graph behavior reviewable.

## Dispatch ownership

All measured shape thresholds remain in `csrc/patchshift_conv3d/select_policy.inl`.
Kernel files implement routes but do not select themselves. The route order is:

1. C16 and output-channel main/tail decomposition;
2. exact M32/M64 small-grid and D1 micro paths;
3. compact spatial tails;
4. C96 hybrid;
5. logical-M256 cluster-B;
6. C64/K64 single-CTA path;
7. C32 general fallback.

## Public lifecycle

The Python API separates static and dynamic work:

1. `pack_patchshift_conv3d_weight(weight)` packs logical `[K,C,3,3,3]` BF16
   weights once;
2. `prepare_patchshift_conv3d(input, packed_weight, K)` allocates and fills the
   descriptor workspace for an input geometry;
3. `patchshift_conv3d(input, packed_weight, workspace, K)` performs the hot,
   CUDA-graph-capturable launch.

The public API validates dtype, layout, device, packed-buffer size, workspace
alignment, packed-weight identity, and exact compute capability 10.0 before
dispatch.
