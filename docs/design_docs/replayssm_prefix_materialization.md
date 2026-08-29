# ReplaySSM prefix-state materialization

Status: initial implementation, August 2026. This records the implemented
low-level contract; it is not yet a general public API commitment.

## Purpose

Materialize a durable Mamba-2 selective-state checkpoint at a prefix boundary.
The operation starts with a saved state and its historical ReplaySSM ring
entries, applies a caller-selected prefix of those entries, and writes a new
state. It never mutates the source state, replay rings, or ring metadata.

## Semantics

One invocation covers all participating layers and requests. A single-layer
call uses `num_layers == 1`; there is no separate per-layer API.

For request `b` and layer `l`, `src_slots[l, b]` selects the source state and
ring entry, and `dst_slots[l, b]` selects the destination state. The caller
provides one logical ring position for every layer in the invocation:

* `flush_count[b] < 0`: no writes, including no state-scale writes.
* `flush_count[b] == 0`: copy source state and scale representation exactly.
* `flush_count[b] > 0`: replay exactly that many oldest entries, beginning at
  `ring_start[b]`, into source state and store the result at destination.

The caller owns boundary convention and validates selected counts. The op has
no accepted-token, alignment, or ring-cursor update semantics.

## Provisional data interface

Each storage family is independently addressable by a GPU-resident per-layer
`int64` pointer table plus a per-layer outer-slot stride:

```text
state_ptrs[L],       state_slot_strides[L]
x_cache_ptrs[L],     x_cache_slot_strides[L]
B_cache_ptrs[L],     B_cache_slot_strides[L]
dt_cache_ptrs[L],    dt_cache_slot_strides[L]
A_ptrs[L]
state_scale_ptrs[L], state_scale_slot_strides[L]  # only for quantized state
src_slots[L, B], dst_slots[L, B]                 # int32
ring_start[B], flush_count[B]                     # int32
```

The layer slot tables are fully explicit rather than encoding a cache-group
rule. Initial support requires homogeneous `H`, `G`, `D`, `S`, `R`, and dtype
specialization. Outer slot spacing need not be compact. Within a slot, each
family uses the existing checkpointing-SSU packed layout: state is
`[head, D, S]`, x and dt are `[head, R, ...]`, and B is `[group, R, S]`.

## Metadata ownership

The initial kernel consumes bespoke, non-layer `ring_start[B]` and
`flush_count[B]`. It does not read or write vLLM's tracker tensors. This keeps
the kernel independent of whether an integration commits trackers at step end
or at the next forward, while allowing a tracker-reading wrapper later.

## Implementation constraints

Use a dedicated replay-only driver with the existing inline-template recurrence
and quantization helpers; do not duplicate the numerical recurrence. Zero
count uses an explicit raw-copy branch so it cannot consume Philox randomness
or re-quantize state. Preserve the existing forward specializations: any
required 8-bit source/destination I/O split is compile-time specialized and
must be checked for unchanged forward resource use and SASS.

The initial launch is one four-warp CTA per `(request, layer, head)` with
`D_PER_CTA == DIM`; it has no persistent work queue, D split, or autotuning
surface. `DIM` is a JIT geometry specialization rather than a runtime tuning
knob. This is intentional for the infrequent alignment-crossing path. The
kernel is explicitly non-PDL (`ENABLE_PDL=false`) and emits no PDL dependency
or completion signals.

The 8-bit implementation follows the currently optimized checkpointing-SSU
two-pass path and currently requires `DIM=64, DSTATE=128`. Its source scale
read is a compile-time variation so it may read the immutable source state
while writing destination scales; the original forward specialization remains
bit-identical after this extension.
