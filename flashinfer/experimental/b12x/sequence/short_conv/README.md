# Prepared raw-input short convolution

This experimental b12x operation computes causal depthwise convolution and
SiLU on SM120/SM121 using the Triton history and convolution kernels shared
with PLE. Importing `b12x.sequence.short_conv` opts into the operation.
Inputs and history contain raw BF16 projections. FP32 convolution accumulation
rounds to BF16 before SiLU; there is no PLE normalization, gate, or residual
addition. Qwen GDN uses 10240 channels, four taps, and dilation one.

The plan fixes token, request, and state-slot capacities. Device-resident counts
and packed query boundaries choose live work without host synchronization.
Decode, prefill, and mixed modes use the same prepared/bound lifecycle.
All buffers are caller owned. Only the outer history-slot stride may be padded;
input and output rows, weights, and metadata are contiguous. Use
`invocation_from_tensors` to prepare actual history strides and alignments.

For kernel size `K`, dilation `d`, and speculative capacity `S`, history has
shape `[slots, channels, d * (K - 1) + S]` in chronological order. Prefill
commits every query row. Decode commits its first row and retains the remaining
speculative inputs in the tail. The next call reads a history window beginning
at `num_accepted_tokens - 1`; ordinary decode uses one. Fresh flags ignore dirty
history. Negative slots produce zero output and never mutate state; zero-token
requests leave their entire state untouched. Padded output rows are zeroed.

The caller guarantees bounded device counts, monotone query boundaries spanning
the live token range, valid accepted counts, and unique in-range nonnegative
state slots. Inactive metadata may contain arbitrary values. Input, output,
history, and scratch must not alias. Prepared plans, graph operands, and scratch
must remain alive until replay has drained. Correctness and allocation stability
are tested; throughput is not qualified.

This example captures a four-token GDN prefill with caller-owned history:

```python
import torch
from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import short_conv

device = torch.device("cuda", torch.cuda.current_device())
channels, tokens = 10240, 4
args = dict(
    x=torch.randn((tokens, channels), dtype=torch.bfloat16, device=device),
    conv_weight=torch.randn((channels, 4), dtype=torch.bfloat16, device=device),
    conv_state=torch.zeros((1, channels, 3), dtype=torch.bfloat16, device=device),
    out=torch.empty((tokens, channels), dtype=torch.bfloat16, device=device),
    query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
    state_slot_ids=torch.zeros(1, dtype=torch.int64, device=device),
    state_is_fresh=torch.ones(1, dtype=torch.bool, device=device),
    num_accepted_tokens=torch.ones(1, dtype=torch.int32, device=device),
    num_seqs=torch.ones(1, dtype=torch.int32, device=device),
    num_tokens=torch.full((1,), tokens, dtype=torch.int32, device=device),
)
plan = short_conv.plan(
    short_conv.Caps(
        device=device,
        mode="prefill",
        channels=channels,
        kernel_size=4,
        max_tokens=tokens,
        max_seqs=1,
        max_state_slots=1,
    ),
    invocation=short_conv.invocation_from_tensors(**args),
)
bindings = []


def prepare_call(state):
    scratch = {
        spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    }
    binding = state.bind(plan=plan, scratch=scratch, **args)
    bindings.append(binding)
    return PreparedCall(run=lambda: state.run(binding), output=binding.out)


with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
    session.prepare((plan.request(name="gdn_conv", prepare_call=prepare_call),))
    session.freeze()
    binding = bindings[0]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        short_conv.run(binding)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(args["conv_state"][0], args["x"][-3:].T, rtol=0, atol=0)
    graph.reset()
```
