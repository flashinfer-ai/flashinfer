# Experimental NVFP4 attention

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection.

Use `flashinfer.prefill.prepare_nvfp4_attention(q, k, v, out, backend="cake")`
on an SM103 GPU. `q`, `k` and `v` are BF16 `[B,H,S,128]` tensors with any
strides (an `[B,S,H,128]` buffer viewed as `[B,H,S,128]` is read in place);
`out` is a contiguous BF16 `[B,H,S,128]` tensor. The route supports noncausal
attention with positive extents and `S` divisible by 512.

Preparation quantizes the inputs to E2M1 with E4M3 block scales (groups of 16
along the head dimension for `q`/`k`, along the sequence for `v`), packs the
scale tiles into the MMA operand layout and returns an `NVFP4AttentionRunner`.
Calling the runner executes QK, softmax and PV attention, writing the
caller-owned output with no CUDA allocation:

```python
attention = prepare_nvfp4_attention(q, k, v, out, backend="cake")
attention()  # Execute attention and write out.
```

Prepare a new runner when input values or bindings change. CUDA Graph capture
belongs to the caller.

The runner's `main_kwargs` exposes the packed tensors and host launch
metadata, and `out` is the exact output passed by the caller. Tensor-map
descriptors are encoded by the host binding and passed by value for each
launch. The implementation does not mutate descriptors on the device.

`cake_backend.pack_nvfp4_attention_inputs(q, k, v)` exposes the packing alone
(any device); `tests/experimental/test_cake_nvfp4_attention.py` holds its
bit-exact oracle and the validated shape set, and
`benchmarks/bench_cake_nvfp4_attention.py` reports preparation and attention
timings per shape. See `examples/cake_nvfp4_attention.py` for a complete example.
