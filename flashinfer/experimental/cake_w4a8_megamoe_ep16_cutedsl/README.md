# Experimental W4A8 EP16 MegaMoE through CuTeDSL

This experimental API runs one fused forward on 16 mutually NVLink-accessible
SM103a GPUs, each with 152 SMs. It uses MXFP4 weights, MXFP8 activations and K32
UE8M0 scales, with BF16 caller inputs and outputs. Model geometry is fixed:
hidden size 3072, intermediate size 5120, 512 experts and top-k 8.

```python
from flashinfer.moe_ep.cake_w4a8_megamoe_ep16_cutedsl import (
    CakeW4A8MegaMoeEp16CuteDsl,
    preprocess_cake_w4a8_megamoe_ep16_cutedsl_weights,
)

prepared = preprocess_cake_w4a8_megamoe_ep16_cutedsl_weights(weights)
session = CakeW4A8MegaMoeEp16CuteDsl(prepared, topk_ids)
forward = session.prepare(x, router_weights, out=output)
forward()
```

Calling this API explicitly opts into experimental behavior. It does not
participate in automatic backend selection and offers no compatibility
guarantee. The implementation is JIT-only, with no AOT registration.

All ranks must construct sessions and invoke forwards in the same order.
Sessions support equal local token counts from 0 through 384. Routing IDs are
cloned during setup; create a new session to change IDs or token count.
Duplicate IDs remain distinct weighted contributions. Caller activations and
normalized finite FP32 router weights may change on every forward.
`prepare` binds their storage and the setup stream, encodes tensor maps, and
retains all referenced storage. Tensor contents remain live. Call it again
when pointers change. `session.forward(x, router_weights, out=output)` also
works and caches its most recent bindings after validating the arguments.

The complete forward, including input quantization, dispatch, NVLink transfers,
FC1/SwiGLU, intermediate quantization, FC2, ordered combine and workspace cleanup,
uses one CuTeDSL kernel launch per GPU. Static FP4 weights are repacked without
changing bits; scale values and expert ownership are preserved. Weights are
prepared once outside forward.
Output must be caller-owned BF16 storage disjoint from inputs and workspace.
Use the setup stream and serialize calls. Graph capture and concurrent session
use are unsupported.

One compiled entry and image serve all token counts and routes. Group rank zero
uses FlashInfer's CuTeDSL JIT cache and broadcasts the exact object during
setup. Every rank loads that object; token count and rank remain runtime
arguments. The CUDA driver loads the exact image embedded in that CuTe object.
A small host C++ extension holds the fixed cluster launch configuration; all
device code is compiled through CuTe DSL. NVSHMEM must provide both unicast
peer mappings and a multicast mapping for the completion signal.
DeepGEMM is required for static weight preparation and numerical comparison.

The intended validation stack uses CuTeDSL 4.7, CUDA 13.3 and the compatible
DeepGEMM revision `559d79fb6994a58b8a15b4b93bf13ccc16edf247`. Newer DeepGEMM
revisions can reject the FC1 width of 10240; changing that reference requires
separate compatibility validation.

Run `examples/cake_w4a8_megamoe_ep16_cutedsl.py` using `torchrun` across the
16 GPUs. Repository correctness tests are in
`tests/experimental/test_cake_w4a8_megamoe_ep16_cutedsl.py` and compare against
DeepGEMM at `atol = rtol = 1e-2`, with relative L2 below 0.02. They include
repeated forwards, duplicate routes, empty input, 17-token extreme values and
384-token all-hot routing. Performance comparisons must include the complete
forward and identify the dependency revisions, graph use and kernel counts, compiled
image and timing reduction used for every arm.

## DeepGEMM with CUDA Graph baseline

The baseline is **DeepGEMM with CUDA Graph**, using FlashInfer
[`2f1dd17d`](https://github.com/flashinfer-ai/flashinfer/blob/2f1dd17d0c40badb0a866d5ca370ac5b428eed91/flashinfer/moe_ep/backends/mega/kernel/sm100/fp8_fp4_bf16_deepgemm/backend.py)
and DeepGEMM
[`559d79fb`](https://github.com/deepseek-ai/DeepGEMM/tree/559d79fb6994a58b8a15b4b93bf13ccc16edf247).
The FlashInfer backend stages and quantizes BF16 inputs, then calls
`deep_gemm.fp8_fp4_mega_moe`. Both operations belong to the measured forward.

Run one process per GPU across 16 mutually NVLink-accessible GB300 GPUs.
Set `FLASHINFER_MEGA_FUSED_STAGE=1` and `FLASHINFER_MOE_EP_KNOB_CACHE=0`
before importing FlashInfer. With the distributed process group and symmetric
memory initialized, construct the baseline from the same canonical weights,
activations and routes as CuTeDSL:

```python
from flashinfer.moe_ep import (
    BootstrapConfig, FleetParams, MegaConfig, MoEEpMegaLayer, MoEEpTensors,
    Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
)

tensors = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=router_weights)
layer = MoEEpMegaLayer(
    bootstrap=BootstrapConfig(world_size=16, rank=rank, device=0),
    fleet_params=FleetParams(
        num_experts=512, max_tokens_per_rank=x.shape[0], token_hidden_size=3072,
    ),
    weights=weights,
    backend=MegaConfig(
        megakernel=Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
            intermediate_size=5120, top_k=8, activation_clamp=None, fast_math=True,
        ),
        quantize_input=True, preprocess_weights=True,
    ),
)
```

Here `weights` is the canonical `MoEEpWeights` input, before CuTeDSL's static
pair-major repacking. Complete lazy setup before capturing one full forward:

```python
import torch
import torch.distributed as dist

layer.forward(tensors)
torch.cuda.synchronize()
dist.barrier()
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for _ in range(3):
        layer.forward(tensors)
        stream.synchronize()
torch.cuda.current_stream().wait_stream(stream)
torch.cuda.synchronize()
dist.barrier()
graph = torch.cuda.CUDAGraph(keep_graph=True)
with torch.cuda.graph(graph, stream=stream):
    baseline_output = layer.forward(tensors)
graph.instantiate()
# Invoke this callable for each measured complete forward:
baseline_forward = graph.replay
```

Keep the layer, tensors, graph and output alive through measurement. Graph
construction and static weight preparation are setup; quantization, staging,
communication, computation, output completion and cleanup stay in the graph.
CUPTI verifies **two kernels per graph replay**, versus **one CuTeDSL kernel
per complete forward on each GPU**. The same CuTeDSL image serves every row.

Measure cold-L2 CUPTI complete spans, including copies, memsets and gaps:
100 ms warmup, 1000 ms measurement, three balanced-order groups per row.
Take the maximum over 16 ranks for each sample, then the median within each
group and the median of three groups. Only the new CuTeDSL feature paths are
added over the pinned FlashInfer dependency tree; its nine shared support
modules are verified byte-identical. Correctness is also tested separately
from the public checkout. This setup describes the baseline used for the
reported numbers; graph capture alone is not the complete timing harness.
