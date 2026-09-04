# Head-Chunk Ulysses Transport Primitives

## Scope

This design extends `UlyssesCommunicator` with allocation-stable destination
passing and opt-in head-chunk transport primitives. It does not add a model
scheduler, choose a chunk schedule, create process groups or side streams, or
change the default whole-tensor Ulysses path.

The intended framework pipeline is:

```text
input stream:   pack+A2A(chunk 0)  pack+A2A(chunk 1)  pack+A2A(chunk 2)
compute stream:                    attention(chunk 0) attention(chunk 1) ...
output stream:                                        A2A+merge(chunk 0) ...
```

Input and output communication must use separate communicators and reusable
workspaces when they can be in flight at the same time. One communicator
continues to permit at most one in-flight collective.

## Public building blocks

| API | Purpose |
| --- | --- |
| `UlyssesWorkspace` | Own fixed NCCL send and receive staging buffers. |
| `create_workspace()` | Size a workspace from a communicator or a smaller chunk bound. |
| `scatter_heads(..., out=, workspace=)` | Use caller-owned output and NCCL staging for ordinary Ulysses. |
| `gather_heads(..., out=, workspace=)` | Reverse ordinary Ulysses without hot-path allocation. |
| `pack_ulysses_qkv_head_chunk(...)` | Pack a destination-local head band from independent positive-strided Q/K/V views. |
| `merge_ulysses_output_head_chunk(...)` | Merge a received compact band into a full output. |
| `scatter_qkv_head_chunk(...)` | Fuse Q/K/V packing with one input all-to-all. |
| `gather_output_head_chunk(...)` | Pack, all-to-all, and merge one attention-output band. |

All operations enqueue on the caller's current CUDA stream. The standalone
pack operation allocates only if `out` is omitted; the standalone merge
requires `out`. Communicator operations are allocation-stable after warmup
when both `out` and a sufficiently large workspace are supplied.

## Layout

For world size `W`, global head count `H`, local head count `HL = H / W`, and
a selected band `[head_offset, head_offset + HC)`:

```text
Q, K, V input:       [B, S_local, H, D]
packed local QKV:    [B, S_local, W * HC, 3 * D]
input A2A output:    [B, W * S_local, HC, 3 * D]
attention output:    [B, W * S_local, HC, D]
output A2A payload:  [W, B, S_local, HC, D]
full merged output:  [B, S_local, H, D]
```

The NCCL input pack writes send-major `[W, B, S_local, HC, 3 * D]` directly
into workspace, avoiding a separate `permute().contiguous()` allocation. For
`B == 1`, the receive-major NCCL buffer is already a valid view of the public
output layout. If no explicit output is supplied, the method returns this
workspace view; its lifetime ends when that workspace is reused.

## Compatibility and safety boundary

The primitives are deliberately opt-in. Frameworks own admission and must
make one rank-consistent decision before entering a data collective.

Initial integrations should require:

- pure Ulysses context parallelism with `world_size > 1`;
- equal Q/K/V head geometry (no implicit GQA or MLA head remapping);
- `H % world_size == 0` and a non-overlapping schedule summing exactly to
  `H / world_size`;
- an attention backend instance valid for every scheduled `head_count`;
- separate input/output communicator and workspace when directions overlap;
- valid device, dtype, positive strides, output shape, and capacity on every
  rank;
- model-owned varlen or padding metadata that remains identical to the
  ordinary attention path.

The implementation rejects byte-range overlap between outputs, inputs, and
workspace storage. Payloads are capped at the int32 element range used by the
transport ABI, while Triton source address arithmetic uses int64 so a legal
positive-strided view cannot wrap its source offset.

Tensor parallelism reduces the effective attention heads available after
Ulysses, approximately `H / (TP * Ulysses)`. Expert parallelism does not
change this layout, but concurrent MoE communication can contend for the same
links. Ring Attention and Attention2D have different collective ordering and
metadata semantics and are outside this implementation.

Head count alone is not a profitability rule. A long sequence can make even
one-head attention chunks large enough to hide communication. Conversely,
short sequences with many heads can still lose to launch and event overhead.
Production schedulers should use an offline `(GPU, topology, backend, dtype,
sequence, local_heads)` whitelist, not a universal head threshold. Unknown
shapes stay on whole-tensor Ulysses.

There is no safe rank-local fallback after one rank has entered a collective.
An error at that point aborts the distributed call; fallback can only be
selected jointly before the call or at the next initialization.

## Backend behavior

- NCCL uses caller-owned send/receive workspace in the opt-in path.
- The existing NVLink backend already owns IPC staging for ordinary Ulysses;
  `out` is honored but the optional ordinary workspace is not consumed.
- Head-chunk NVLink transport reuses caller workspace for local fused input or
  compact output around the existing raw all-to-all.
- FP16, BF16, and FP32 follow the communicator's existing dtype contract.
  FP8 QKV/output communication is not part of this change.
- No process group is created inside FlashInfer.

## Validation

The test suite covers:

- world sizes 1, 2, 3, and 4 on NCCL;
- FP16 and BF16 head-chunk transport;
- independent non-contiguous fused-projection Q/K/V views;
- uneven schedules `[3, 8, 3]` and `[5, 5, 4]`;
- batch sizes 1 and 2;
- default and non-default current streams;
- destination identity, direct receive-workspace lifetime, capacity, dtype,
  shape, and storage-overlap failures;
- stable workspace pointers and no PyTorch CUDA allocator events after
  warmup in explicit-buffer communication paths;
- exact reconstruction against ordinary scatter/gather references.

The NVLink head-chunk path requires a separate NVLink-machine run before an
upstream PR can mark that backend validated. The local SM120 PCIe environment
can only exercise its topology rejection/fallback behavior.

## Reference performance

The benchmark uses BF16 PyTorch SDPA and reports rank-maximum latency. The
ordinary baseline already uses destination passing and reusable workspace, so
the comparison isolates QKV fusion, head chunking, and overlap instead of
allocator noise. These are operator-pipeline measurements, not model E2E.

RTX PRO 6000 Blackwell Server Edition, NCCL over PCIe, 3 warmups and 7 timed
iterations:

| Physical shape `[B, S, H, D]` | Ulysses | Schedule | Ordinary median | Overlap median | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| `[1, 37760, 56, 128]` | 2 | `[7, 14, 7]` | 99.332 ms | 89.486 ms | 1.110x |
| `[1, 37760, 56, 128]` | 4 | `[2, 10, 2]` | 50.881 ms | 47.589 ms | 1.069x |
| `[1, 39808, 56, 128]` | 2 | `[7, 14, 7]` | 107.418 ms | 99.553 ms | 1.079x |
| `[1, 39808, 56, 128]` | 4 | `[2, 10, 2]` | 57.195 ms | 51.524 ms | 1.110x |

All compared outputs had zero maximum absolute difference in these runs. The
benchmark treats the full physical sequence as dense. Model integrations are
still responsible for preserving true `cu_seqlens` and tail-padding semantics.

An SM120 NCU run on the U4 10-head QKV pack measured 341.34 us, 1.48 TB/s
DRAM throughput, 92.74% of peak memory throughput, and 11.72% compute
throughput. An aggregate Nsys capture attributed about 69.5% of GPU kernel
time to NCCL SendRecv, 28.5% to attention, and about 0.6% to the three layout
kernels. The next material optimization target is transport exposure and
resource contention, not arithmetic inside pack/merge.

## Non-goals and follow-up work

- The benchmark-local three-stream executor is a reference, not a runtime API.
- Schedule selection and TP/EP/CP policy belong in an integrating framework.
- CUDA Graph capture needs explicit workspace/stream lifetime tests.
- GQA/MLA needs an explicit Q-to-KV head mapping API.
- FP8 communication needs independent scale format, numerical, and quality
  review; FP8 attention compute with BF16 communication can integrate first.
- A custom PCIe P2P or topology-aware transport may reduce the dominant NCCL
  cost, but should be developed independently from these layout primitives.
