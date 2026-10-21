# Prepared GDN prefill

This experimental b12x operation accepts packed BF16 Q/K/V projections, raw
BF16 decay/update projections, and caller-owned FP32 recurrent state. Key and
value dimensions are 128; each key head serves three value heads.

`Caps(intermediate_dtype=torch.bfloat16)` uses 16-token tensor-core chunks.
This is the default. `Caps(intermediate_dtype=torch.float32)` explicitly
selects a CuTe device-side recurrence loop with FP32 Q/K normalization and
state arithmetic. Both variants return BF16 output, preserve the model's
BF16 sigmoid-update boundary, and write FP32 final/checkpoint state. Precision
is part of the preparation identity and cannot be selected by autotuning.

```python
caps = gdn_prefill.Caps(
    device="cuda:0", max_tokens=512, max_seqs=4, max_state_slots=13,
    key_heads=16, value_heads=48, checkpoint_export=True,
    null_state_index=0, intermediate_dtype=torch.float32,
)
plan = gdn_prefill.plan(caps, invocation=gdn_prefill.invocation_from_tensors(**args))
# Prepare the declaration, bind caller-owned tensors/scratch, then capture run().
```

The FP32 variant uses a fixed native launch with device-resident sequence
counts and boundaries. It retains state in registers across all tokens of a
sequence and stores only its requested checkpoint and final state. Its scratch
contains metadata capacity only; it has no prepared-tile ring. Serial token
dependencies and FP32 scalar arithmetic can reduce throughput relative to the
BF16 tensor-core path; callers must qualify their shapes and numerical needs.

State slots use `[slot, value_head, value_dim, key_dim]` with contiguous slot
contents and a positive outer stride. A null initial slot supplies zero state
without reading its contents. Null destination slots remain untouched. Empty
sequences copy initial state to final state. Checkpoint export requires a
positive offset divisible by 16; offset zero disables export. Packed metadata
must identify in-range, conflict-free state slots. The FP32 variant zeros padded
output rows. The BF16 chunk variant writes live rows; callers initialize padded
capacity when downstream operators read it.

Compilation and workspace allocation finish before CUDA graph capture.
`run()` exposes one custom operator to `torch.compile`, declaring mutations
of caller-owned scratch, recurrent state, and output. Kernel selection and
precision remain fixed by the prepared plan.
Captured replay can change device counts, packed boundaries and slot indices
without compilation, host reads or allocation. Runtime execution never
substitutes a PyTorch reference or another precision variant.
