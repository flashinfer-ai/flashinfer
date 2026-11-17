# NCCL EP 0.3 integration

The package requires `nccl-extensions>=0.2.0`, which ships EP library 0.3.
Admission uses the loaded EP library version. The adapter additionally admits
LL hidden sizes 3072 and 3584 on that
library; older providers retain the legacy seven-size guard. EP 0.3 uses runtime
JIT compilation, so provide its kernel headers, NCCL device headers, a compatible
`nvcc`, and a writable `NCCL_EP_JIT_CACHE_DIR`. Other JIT sizes remain outside
the adapter's admitted set. The CUDA-major requirement remains unchanged. The
8192-token HT capacity also matches the EP 0.3 provider limit.

Pass `FleetAlgoKnobNcclEpConfig` in the Fleet's `algo_knobs` to set a group-wide
`num_topk` bound or shared `max_num_sms`. The new release also accepts independent
`dispatch_num_sms`, `combine_num_sms`, `ht_flow` values `auto`, `push_pull`, or
`pull_push`, and `meta_exchange_policy` values `auto`,
`require_fused`, or `force_scan`. Unsupported fields fail early on older
providers. A positive group top-k bounds every handle created by that Fleet.

HT uses Flat layout, which supports both flows. The default leaves flow
selection to the provider. A handle-time received-token counter requires
metadata before dispatch and is incompatible with `require_fused`; under
`auto` the provider can select its unfused fallback.

Set `eager_recv=True` on EP0.3 to size receive payload and routing arrays from
the actual received-row count. This mode reads that count on the host and
allocates compact arrays for expert compute. An empty receiver uses one inactive
row with masked routing. Handle creation, updates and dispatch reject CUDA graph
capture in this mode. The default fixed-capacity mode supports capture and
avoids the host count read. Eager sizing is HT-only and cannot use
`require_fused`. It does not reduce the provider's internal staging reservation.

HT Expert-Major and quantized combine require a separate compute-layout
integration; they are not exposed by this Flat adapter.
The adapter also does not expose native dispatch-quantization recipes or
window-backed zero-copy tensors, experimental LL NVFP4 combine, AFD role APIs,
prepared-handle import, or an overflow-policy selector. The provider's default
overflow policy still applies. CUDA 12, other LL JIT sizes, and the provider's
opt-in CFT and NUCS paths are outside this qualification. These limits concern
the split NCCL backend; native MegaMoE uses its own communication path.

Fault-tolerance query/update use the C ABI because their Python counterparts
take an EP Tensor and a host NumPy array rather than pointers. Mask updates keep
their pinned host input unchanged until its asynchronous copy finishes; reuse,
topology replacement and destruction wait for that copy when needed.
Normal LL group destruction still barriers across the original communicator
and can block after a peer exits.
