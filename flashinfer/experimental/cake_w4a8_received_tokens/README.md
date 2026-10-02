# Experimental prepared W4A8 routed MoE

Owner: @hzfan. Tracking: [#4254](https://github.com/flashinfer-ai/flashinfer/issues/4254).
Both the prepared API and its Cake backend are experimental.
The prepared call has a distinct lifetime contract: it owns packed weights and
workspace, retains caller tensor addresses, and exposes an allocation-free
`run()` for repeated calls and external CUDA Graph capture.

Supported configuration: 152-SM GB300 (SM103a), H=3072, I=5120, 512 global
experts, 32 contiguous local experts, top-k 8, and 1..8192 received token rows.
Empty ranks are unsupported. Inputs are contiguous CUDA tensors. Input values
and routing contents may change in place; addresses and weights remain fixed.
IDs retain all original route slots, including duplicates and remote experts.
Output is a BF16 local expert partial; dispatch and global combine are caller
operations. Each call submits routing, FC1/SwiGLU/quantization, FC2 and local
finalize kernels with PDL.

Opt in by calling
`flashinfer.fused_moe.prepare_fp4_block_scale_routed_moe(..., backend="cake")`.
This route never participates in automatic dispatch, autotuning or trace apply.
It requires PyTorch with CUDA and CUTLASS CuTe DSL with SM103a support.
Serialize calls sharing one prepared workspace.

Run the example:

```bash
python -m flashinfer.experimental.cake_w4a8_received_tokens.cake_example
pytest tests/experimental/test_cake_w4a8_received_tokens.py
```

The example documents tensor construction and validates the output against an
independent PyTorch reference. Preparation includes compilation and packing;
warm up before capture. The API and backend can change or be removed without
compatibility guarantees.
