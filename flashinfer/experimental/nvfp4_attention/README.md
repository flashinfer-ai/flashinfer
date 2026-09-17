# Experimental NVFP4 attention

Both this API and its Cake backend are experimental and may change without
backward compatibility. Calling the API explicitly opts into the experimental
feature and emits FlashInfer's experimental API warning. There is no automatic
backend selection.

Use `flashinfer.prefill.prepare_nvfp4_attention(q, k, v, out, backend="cake")`
to prepare contiguous BF16 `[B,H,S,128]` inputs on an SM103 GPU. The current
route supports noncausal attention with positive extents and S divisible by 512.
Preparation packs E2M1 values with E4M3 block scales and returns an
`NVFP4AttentionRunner`. Calling the runner executes QK, softmax and PV attention,
writing the caller-owned BF16 output with no CUDA allocation:

```python
attention = prepare_nvfp4_attention(q, k, v, out, backend="cake")
attention()  # Execute attention and write out.
```

Prepare a new runner when input values or bindings change. CUDA Graph capture
belongs to the caller.

The runner's `main_kwargs` exposes packed tensors and host launch
metadata, and `out` is the exact output passed by the caller. Tensor-map
descriptors are encoded by the host binding and passed by value for each
launch. The implementation does not mutate descriptors on the device.

See `examples/cake_nvfp4_attention.py` for a complete example and
`tests/experimental/test_cake_nvfp4_attention.py` for the validated shape set.
