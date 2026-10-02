# Experimental BF16 MoK training adapter

This is a **synthetic toy setting**, not a full-model training result.
The explicit `flashinfer.mok.prepare_mok_bf16` API returns a BF16 Mixture of
Kittens functional adapter. It covers scheduling, forward, and backward,
including input gradients, router-score gradients, and all six expert-weight
gradients. Shared-expert weight gradients remain source-rank local.

The implementation uses standalone CUDA sources and FlashInfer's TVM-FFI JIT.
It preserves the original launch boundaries: three scheduler kernels,
metadata gathering and peer barriers, one fused forward, one fused backward,
separate output/input-gradient epilogues, and empty-expert gradient zeroing.
Dispatch, GEMMs, SwiGLU, combine, and backward ring replay stay inside the
fused kernels. Framework copies, resets, and the router-gradient clone remain
part of a complete iteration.

## Use

The package includes its caller-owned workspace and metadata layer; no separate
MoK installation or extension is needed. A CUDA toolkit supporting the selected
device and PyTorch with symmetric-memory multicast support are required.
Distributed ranks must share one peer-accessible NVLink domain.

```python
from flashinfer.mok import prepare_mok_bf16, create_mok_bf16_workspace

config, workspace = create_mok_bf16_workspace(
    group=group, device=device, num_local_tokens=1024, hidden_size=6144, topk=8,
)
backend = prepare_mok_bf16(ep_size=16, local_experts=16, topk=8)
# Using caller-owned inputs and weights:
schedule = backend.build_schedule(workspace, config, ids, num_local_experts=16)
y, context = backend.forward(config, workspace, schedule, x, scores, *weights)
grads = backend.backward(config, workspace, schedule, context, dy, x, scores, *weights)
```

`weights` contains shared gate/up/down followed by routed gate/up/down.
`grads` contains input, router scores, routed gate/up/down, then shared
gate/up/down gradients. Prepare and warm up before CUDA Graph capture.
All ranks must execute matching collective calls. Keep workspace allocations
and peer mappings alive while the adapter or a captured graph can use them.
Use a separate workspace and adapter for concurrent executions.

Calling the experimental API is the opt-in; it emits an experimental warning.
There is no automatic dispatch or AOT registration.

## Toy contract

- Scheduler layouts `(EP, local experts, top-k)`: `(1, 4, 2)`, `(4, 4, 2)`,
  and `(16, 16, 8)`. EP1/EP4 are small diagnostic settings.
- BF16 inputs, expert weights, outputs and weight gradients; FP32 positive
  router scores; contiguous int64 expert IDs. Each token selects distinct,
  valid expert IDs. Scores are already normalized/scaled by the caller.
- Token rows, hidden width, and intermediate width are positive multiples of
  256; the workspace requires at least 512 local tokens. Communication-SM
  counts are positive and even, leaving compute SMs available.
- Mini-batches are multiples of 256; macro-batches are multiples of the
  mini-batch. Schedule capacity must hold every padded route; overflow traps.
- The EP16 toy workload has 16,384 global tokens (1,024 per rank), hidden width
  6,144, intermediate width 2,048, 256 routed experts, and one shared expert.
  Its mini/macro sizes are 4,096/32,768. Sequence length here is only a way to
  specify token count: no attention or sequence-dependent operation is tested.
- A larger macro ring uses more memory and can avoid forward-context
  recomputation inside backward. A comparison that changes this setting must
  report it, alongside peak memory, and time both complete iterations with
  the same CUDA Graph policy.

SwiGLU clipping, MXFP8, a separate recompute API, optimizer steps, router-network
backpropagation, and shared-gradient all-reduce are outside this adapter.

## Validation and lifecycle

Run `pytest tests/experimental/test_mok_bf16.py` for the independent
BF16-rounding reference, empty experts, multiple ring lengths, variable
widths, repeated eager execution, and changed-input CUDA Graph replay.
The runnable distributed example is `examples/mok_bf16_toy.py`.

Owner: Haozheng Fan. This draft needs a public tracking issue and a maintainer-
agreed release target before experimental admission. Graduation requires
full distributed qualification of the exported runtime and resolution of
positive-window sanitizer deadlock-detector failures; numerical checks alone
do not clear that gate. No whole-model or public performance claim is made.

The CUDA schedules derive from Cursor Research's Apache-2.0 Mixture of Kittens
implementation at the revision above. Copyright and modification notices are
retained in the generated source files.
