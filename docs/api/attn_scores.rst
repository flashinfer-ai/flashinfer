.. _apiattnscores:

flashinfer.attn_scores
======================

Paged MQA logits ("attention scores") kernels for datacentre Blackwell
(SM100/SM103) and Rubin (SM107).

These compute, for every request and KV position, the per-head weighted sum of
rectified query-key scores that a sparse-attention indexer uses to choose which
KV tokens to keep:

.. math::

  \mathrm{logits}[t, p] = \sum_h w_{t,h} \cdot \mathrm{relu}\left( q_{t,h} \cdot k_p \right)

Two things about that expression are easy to get wrong:

* The rectifier is applied **per head, before** the weighted sum -- not to the
  summed score. Because the weights may be negative, the result may be too.
* :func:`fp8_paged_mqa_logits` additionally multiplies by the per-token FP32 KV
  scale carried in the trailing scale region of each ``kv_fused`` block, so its
  result is
  :math:`s_p` times the above. :func:`fp4_paged_mqa_logits` has no such
  per-position factor: MXFP4 block scales are folded into the dequantised
  values themselves. The asymmetry is algebraic -- a scale uniform over a
  token's whole K vector factors out of the dot product (and, being positive,
  commutes with the ReLU), while per-32-element-group scales on both operands
  do not.

The variants also differ beyond the scales. :func:`fp4_paged_mqa_logits` takes
the extra ``q_sf`` scale tensor (plus ``sf_vec_size`` and
``is_kv_sf_interleaved``) and packs two FP4 values per ``q`` byte. Only
:func:`fp8_paged_mqa_logits` exposes ``acc_dtype`` (fp4's MMA accumulator is
fixed float32), and the default ``output_dtype`` is float32 (fp8) versus
bfloat16 (fp4). fp8 accepts parametric shapes while fp4 requires exactly
``num_heads=64``, ``head_dim=128``. Finally, ``next_n=4`` is single-pass on
every supported arch for fp8, but for fp4 only on Rubin (SM107) --
SM100/SM103 run it as two internal KV passes; build any caller
``schedule_meta`` with the schedule helper's matching ``next_n``/``variant``
arguments. Paging arguments, the output contract, and
``out=``/``schedule_meta`` semantics are identical across the pair.

FP8 selects 64-bit store offsets when the physical output span, including row
padding, reaches ``2**31`` elements. Logical dimensions still fit int32. This
specialization has small-output and lowering coverage. The H64/Q8192/1M
benchmark uses a full-score buffer of ``2**33`` FP32 elements (about 8.6 billion),
selects the Int64 store path, and passes all-row FP32 checks before and after
timing. There is no dedicated unit test at that scale. FP4 continues to reject
spans of ``2**31`` elements or larger.

.. note::

  Within each row's scheduled extent the kernels write unconditionally and
  apply no causal or length mask; columns beyond that extent may never be
  written at all.  Output row ``b*next_n + t`` is meaningful only for KV
  positions ``0 .. seq_lens[b] - next_n + t`` inclusive (the newest slot
  ``t = next_n - 1`` sees the whole sequence, each earlier slot one position
  fewer); callers must mask or slice past that per-slot limit themselves.  A
  request with ``seq_lens[b] == 0`` is skipped entirely and its output
  rows are never written -- when passing ``out=``, initialise it if you
  intend to read those rows.

.. currentmodule:: flashinfer

Paged MQA Logits
----------------

.. autosummary::
  :toctree: ../generated

  fp8_paged_mqa_logits
  fp4_paged_mqa_logits

.. autofunction:: fp8_paged_mqa_logits

.. autofunction:: fp4_paged_mqa_logits

FP8 MQA TopK
------------

The selective strategy requires CuTe DSL and supports SM100 with CUDA 12.8 or
newer, and SM103 with CUDA 12.9 or newer. Construction rejects other compute
capabilities before allocating its workspace.

The TopK wrapper has an explicit, lifetime-fixed ``strategy`` with common
constructor, planning and layer-call signatures. Both reserve capacities at
construction, bind prefixes/maps with ``plan`` once per step and call ``run``
for each layer. Full planning reads prefixes on the host and must run outside
capture; selective planning refreshes device metadata and is capturable.
``"full"`` composes paged logits with public
GVR-v2 TopK512, TopK1024 or TopK2048; ``"selective"`` uses bounded candidate publication and
exact repair with fixed internal selectors. There is no automatic backend
selection or fallback. Full owns GVR-v2 scratch; separate wrappers own
separate buffers across graph capture/replay.
Both the wrapper constructor and :func:`fp8_paged_mqa_logits` accept
``math_mode="relu_strict"`` (default) or ``"relu_fast"``. The wrapper's
read-only policy applies to every scoring phase and cannot change across
plan/replan or graph replay. Fast FP32 epilogues use packed ``x + abs(x)``
and one half-scale compensation; intermediate overflow, tiny-scale underflow
and exceptional-value behavior can differ from the default strict NaN-aware
ReLU with the full scale. Fast is an explicit opt-in to those tradeoffs. FP16
epilogue arithmetic is unchanged in either mode; FP16 output alone does not
select FP16 arithmetic. TopK is exact for the selected scoring recipe, not
a promise of bitwise equality between recipes.
Selective treats a final NaN score as negative infinity consistently in
sampling, candidate publication and exact repair. NaNs tie with genuine
negative infinity, so valid indices from that tie can fill TopK when fewer
than K higher scores exist. Short-row padding remains -1. The ordinary
scorer and the full strategy keep their existing NaN behavior.
The FP8 logits trace reference models FP32 epilogue and accumulation with
FP32 or FP16 final output storage. It rejects other traced epilogue or
accumulator precisions; the public scorer still supports its documented
FP16 arithmetic independently of that trace-reference restriction.
The selective sampling, candidate, repair, radix, and
finalization phases remain private implementation details that are independently
testable and profileable. Packed ragged batches are supported, with TopK512/1024/2048
for full and every integer TopK from 1 through 2048 for selective. Rows
shorter than K retain all causal keys and pad the fixed-width suffix with
``-1``, as ``top_k_varlen`` does. Query and KV sequence lengths
remain runtime metadata rather than CuTe compile-cache keys.

The immutable ``compression_ratio`` constructor option defaults to 1 and
accepts 4 for the selective strategy (not full). Ratio 4 consumes KV and
scales that the caller has already compressed; it neither compresses inputs
nor reduces the number of query rows. ``kv_indptr`` still counts original tokens.
For each request with original length L and Q queries, physical length is
``L // 4`` and query row i sees compressed slots
``[0, (L - Q + i + 1) // 4)``. Each request is floored independently before
forming physical prefixes. Page maps and returned indices address compressed
slots; incomplete trailing blocks are excluded. Only the first
``min(top_k, eligible_slots)`` entries are valid, and the remaining suffix is
``-1``. Empty rows are entirely ``-1``.

Selective specializes heads, dimension, page size, TopK and math at construction,
which also reserves the step workspace for fixed capacities and one KV pool layout.
Pass the layout as a ``kv_layout`` exemplar, whose shape, strides, dtype and device
are read without retaining the tensor, or as ``num_pages`` and ``device`` for a
contiguous pool with standard strides. Per-step ``plan()`` binds the device
prefixes and page map once; serialized layer calls share one scoring workspace
without host prefix-value reads::

    wrapper = FP8PagedMQATopKWrapper(
        strategy="selective", num_heads=8, head_dim=128, page_size=128,
        top_k=512, compression_ratio=4,
        max_context_length=1048576, max_chunk_size=148, max_batch_size=148,
        kv_layout=kv_layer_0,  # or num_pages=..., device=...
    )
    wrapper.prepare_pool(kv_layer_0)  # once, outside capture: optional warmup
    wrapper.plan(qo_indptr, kv_indptr, block_table, num_query_rows=148)
    wrapper.run(q0, weights0, kv_layer_0, out=indices0)
    wrapper.run(q1, weights1, kv_layer_1, out=indices1)

For serving integration, allocate the indexer KV cache first, then construct
the selective wrapper and prepare it once, with ``prepare_pool`` on any layer
pool or one eager ``run``, before graph capture. Recreate the wrapper when the
engine reallocates its cache. For same-layout layer-pool swaps within an
allocation, discard the old caller graphs, call ``clear_pools()`` and recapture.
Equal layout includes shape, strides, dtype and device.
A caller graph captured over ``plan``/``run`` addresses the wrapper's workspace
and the pools it captured. Keep the wrapper and those pools alive for as long as
the graph may replay; replaying it after the wrapper is released reads and
writes memory the allocator may have handed to another owner.
Release wrappers and their plans outside every CUDA graph capture, including
captures owned by other wrappers: retiring private metadata unloads CUDA libraries.

Construction requires CUDA and allocates staging, metadata, output and status
buffers without compiling kernels. The first ``prepare_pool`` (also called by
an eager ``run``) compiles metadata kernels and allocates/compiles the shared
scoring workspace. A ``plan`` before that first preparation stages inputs only;
pool preparation validates them before scoring. Prepare a pool before reading
the initial step's status or capturing ``plan``. Subsequent ``plan`` calls
refresh and validate metadata immediately. The ``num_pages``/``device`` form
uses standard contiguous strides, including singleton axes; use ``kv_layout``
to describe a different contiguous singleton-axis layout.

``max_chunk_size`` bounds the total supplied query rows, not rows per request.
``max_context_length`` counts original tokens per request; the page table has
``max_batch_size`` rows and enough columns for each request's physical capacity.
Every layer pool must have the constructed layout's shape, strides, dtype and
device, including the same physical page count. Each run validates that layout.
Kernels compile once per layout, at the first ``prepare_pool`` or eager ``run``.
After that, caller capture binds any unseen equal-layout pool by building its
launch arguments, with no compilation, host value read or allocation, so later
layers need no per-pool preparation. Eager ``run`` lazily builds an owned graph
per pool, keyed on address plus layout, and keeps those pools alive until
``clear_pools()`` or wrapper destruction; ``prepare_pool`` builds that eager
graph ahead of time and is otherwise optional. Pools seen only under capture
add no owned graph. For same-layout layer-pool swaps, discard caller graphs
using the old pools and call ``clear_pools()`` outside capture. This
synchronizes outstanding device work and releases all cached pool graphs,
launch owners and pool tensors while preserving the compiled kernels, tensor
workspace and step status; then recapture.
The graph configuration is selected from capacities, not active lengths.
Sampling stays enabled when physical context capacity exceeds 6144, including
for shorter steps. Device prefixes may change in place, but each step must
call ``plan()`` or replay a graph that includes it before running its layers.
Changing prefixes alone is insufficient when replaying a run-only graph.
Inactive output rows and short-row suffixes are ``-1``.

Use a wrapper per batch bucket rather than one global maximum. Metadata refresh
work depends on planned row/request capacities and page-table width as well as
the active step. Oversized buckets can therefore increase work even when only
a few rows are active. Eager run also includes staging and host enqueue work;
its CUDA-event envelope may include host-launch gaps and is not a sum of kernel
durations. Caller graph replay and direct staging-buffer use have different
costs and should be measured separately for the intended deployment.

``input_buffers()`` exposes owned Q, weight, prefix and page-table staging tensors.
Pass matching views to run/plan to avoid their copies. Optional ``out`` is a non-overlapping
contiguous int32 tensor with the supplied query extent and TopK width.
``error``, ``repair_error`` and ``active_shape`` are device buffers; inspect them
after a dependency or synchronization. Invalid step metadata suppresses scoring
and produces ``-1`` rows rather than a synchronous host exception. The validation
status persists across layers; ``repair_error`` is an OR-accumulated flag,
with zero indicating no error, rather than a count of failed rows. It persists across layer
calls. Both statuses reset only at the next ``plan``. Its
``num_query_rows`` defaults to ``max_chunk_size``; each layer must supply that
many Q rows, while the final device prefix may be smaller.

Sampling is selected from the physical capacity:
``max_context_length // compression_ratio > 6144``. Smaller buckets publish
all eligible keys without SAMPLE or the threshold ladder. Larger buckets keep
sampling enabled on shorter steps; changing lengths does not change the
captured route. Capacity validation bounds every admitted physical row.

The device validation ``error`` word uses these bits:

.. list-table:: Selective step validation
   :header-rows: 1

   * - Bit
     - Meaning
   * - 1
     - Staged query-row or batch extent exceeds the planned bounds, or is negative.
   * - 2
     - Prefixes do not start at zero, are negative/decreasing, or a request has
       fewer original KV tokens than query rows.
   * - 4
     - A query prefix exceeds the supplied row extent, a request exceeds its
       original-context capacity, or its physical pages exceed the table width.
   * - 8
     - A live page-map index is outside ``[0, pool_page_count)``.

Bit 1 suppresses all further checks. Bits 2 and 4 may combine; page IDs are
checked only after those checks pass. Any device error publishes zero active
rows and sanitized prefixes, and selection fails closed to ``-1``. Tensor
layout/type/shape and alias errors instead raise on the host before staging
copies. ``repair_error`` is a separate OR-accumulated flag, not this bit field
or a count of failed rows.

``FLASHINFER_VALIDATE_INPUTS=1`` enables additional synchronized value checks
in the prepared selective launch entrypoints outside capture. It is a developer
debugging option, not a replacement for the capacity API's always-active
device-side step validation.

Both strategies use ``plan(qo_indptr, kv_indptr, block_table,
num_query_rows=...)`` and ``run(q, weights, kv_fused, out=...)``. Specialization
belongs to the constructor, and the map is bound only by plan. The helpers
``prepare_pool``, ``clear_pools``, ``input_buffers`` and the device status
properties are available for both strategies. Full requires plan before pool
preparation and a prepared pool before capturing run; every new full plan
invalidates the previous full pool graphs. Selective can prepare the layout
before its first plan and bind additional equal-layout pools during capture.

Capacities and the pool layout are fixed for the wrapper's lifetime; construct
another wrapper to change them, and capture new caller graphs for it. Pool
changes do not allocate another selective tensor workspace. Full pools own
separate score and GVR workspaces. Results and staging views
stay valid for the wrapper's lifetime, but later runs overwrite their contents. Supply a separate ``out`` per layer, or
consume a plan-owned result before the next layer. Use one wrapper per concurrent
stream; calls sharing a workspace must not overlap. Eager calls replay each
pool's owned graph. Caller capture emits plan/run operations directly,
including any staging copies; captured source addresses and supplied extents
are fixed until recapture. For complete single-layer benchmarks include
plan in the captured workload. Multi-layer timing should include it once
per step, not once per layer.

Full uses the same per-step and per-layer signatures, supports ratio1 and
TopK512/1024/2048, and reads plan prefixes on the host. Selective traces use
separate ordinary/compressed schemas with constructor, plan and run
input initializers, plus strict reference/check functions.

Unlike planners that read device prefixes on the host, selective ``plan()``
has no host value reads or synchronization and may run inside a caller's CUDA
graph. Construct the selective wrapper outside capture before the first step.

The native sample-ladder and conditional-repair helpers can load from an
installed architecture-specific JIT-cache provider. These helper artifacts
do not contain every CuTe-DSL specialization used by the wrapper. In
particular, ``FLASHINFER_DISABLE_JIT=1`` also rejects a missing cached
CuTe-DSL radix module. Installing the two native helpers alone therefore
does not enable a cold-cache, compilation-free wrapper deployment; matching
CuTe artifacts must already be available through their cache mechanism.

.. autosummary::

  FP8PagedMQATopKWrapper

.. autoclass:: FP8PagedMQATopKWrapper
  :members:

Reproduce full/GVR-v2 versus selective comparisons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, the in-tree benchmark composes the public FP8 scorer
and ``top_k_varlen(backend="gvr_2")`` for its full arm. This supports TopK512
TopK1024 and TopK2048 with ratio1 or ratio4 without expanding the full wrapper's
ratio1 API. For example::

    # H8 ratio-4 prefill, then request-local decode.
    python benchmarks/bench_fp8_mqa_selective_logits.py --q 8192 \
        --kv 8192 --num-heads 8 --top-k 512 --compression-ratio 4
    python benchmarks/bench_fp8_mqa_selective_logits.py --batch-size 148 --q 1 \
        --kv 8192 --num-heads 8 --top-k 512 --compression-ratio 4

    # H64 ratio-1 prefill, then request-local decode.
    python benchmarks/bench_fp8_mqa_selective_logits.py --q 8192 \
        --kv 8192 --num-heads 64 --top-k 2048
    python benchmarks/bench_fp8_mqa_selective_logits.py --batch-size 148 --q 1 \
        --kv 8192 --num-heads 64 --top-k 2048

``--kv`` accepts multiple original-token context lengths. Ratio4 generates
already-compressed independent KV per request; it does not run a compressor.
The full arm groups queries only within each uniform request, refreshes all
group page tables with one batched gather per step, and passes a natural-width
score view to GVR-v2 while preserving padded allocation/row pitch. Prefill
defaults to H8/T16 or H64/T4 when Q is divisible by that tile; otherwise T1.
``--full-query-tile``, ``--full-epi-subtiles`` and ``--full-w-cache-regs`` expose
explicit scorer configuration. Unsupported geometry fails rather than falling
back. A single output row requires physical width divisible by four for the
public GVR-v2 route.

The benchmark records separate matched per-step and per-layer graph pairs.
The full layer contains the public scorer and GVR-v2; the selective layer contains
``run()``, including counter resets, publication, selection, repair and output.
The full step derives row/group bounds and request IDs from device prefixes,
gathers the live page table and invokes the capturable public schedule builder.
The selective step invokes ``plan()``, including the caller page-table copy and
metadata/schedule refresh. Q, weight and prefix staging copies are outside both
pairs. Query topology is uniform and fixed within a cell; capacities fit the
declared row, batch and context dimensions exactly.
Every row is checked against
chunked independent FP32 score math with TF32 disabled before and after timing,
including distinct indices, request-local bounds and ratio-specific suffixes.
Reference generation, wrapper construction, pool preparation and JIT are outside
timing; per-step planning is timed. JSON includes all
raw samples, inclusive interpolated p10/p90, workspace totals, correctness
results and Torch/CUDA versions. Workspace counts reachable owned tensors,
excluding inputs and outputs; it is not an allocator or device peak.

See the benchmark's module documentation and ``--help`` for its measurement
protocol and reporting options. Derived multi-layer estimates are not directly
measured multi-layer latency.

Successful benchmarks print Markdown latency/workspace tables in the same
invocation, while ``--output`` keeps raw JSON. Regenerate the report with
``python benchmarks/bench_fp8_mqa_selective_logits.py --summarize results/*.json``.
This reporting mode requires neither Torch nor CUDA. It checks positive finite
paired raw samples and exact-repair status, and rejects incomplete runs or
duplicate cells. Keep measurements from different environments separate.

Scheduling and Setup
--------------------

The persistent kernels need a per-call CTA work assignment. It is computed on
the GPU by default, so the whole dispatch can be captured in a CUDA graph with
no host round-trip. :func:`compute_paged_mqa_logits_schedule` lets a caller
build it once and pass it back via ``schedule_meta=``. Pass the helper the
same ``variant`` and ``next_n`` as the main call: it applies the kernel's
internal scheduling policy for that configuration, so the returned tensor
(treat it as opaque) is correct on every architecture.

.. warning::

  A reused schedule is only valid while the contents of ``seq_lens`` are
  unchanged for every request. Recompute it whenever ``seq_lens`` changes,
  including on every CUDA-graph replay.

.. autosummary::
  :toctree: ../generated

  compute_paged_mqa_logits_schedule
  min_block_table_width
  padded_seq_len
  precompile_paged_mqa_logits
