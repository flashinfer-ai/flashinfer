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
uses one eager CuTeDSL kernel launch. Static FP4 weights are repacked without
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
384-token all-hot routing. Export validation and measured performance are
pending; no results from another executable are claimed here.
