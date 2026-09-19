Cake SiTU backend
=================

``flashinfer.fused_moe.cutlass_fused_moe(..., backend="cake")`` executes the
complete BF16-input routed SiTU operation: group-16 NVFP4 input quantization,
routing, FC1, SiTU, intermediate NVFP4 requantization, FC2, and BF16 weighted
finalization. It supports SM100a/SM103a with TP8-local H=3584, I=384, E=896,
top-k=16 and 1 through 16384 tokens. Expert IDs are int32 and routing weights
are BF16. The caller supplies valid expert IDs in [0,896); routing values never
participate in host dispatch. Output is local to the TP rank; collectives belong
to the caller. The existing CUTLASS default and Cake warp-decode API are unchanged.

Prepared weights and scales
---------------------------

The Cake selection uses the TRTLLM shuffled group-16 NVFP4 layout, including
the fused FC1 gate/up row permutation and block-scale interleave. Ordinary
CUTLASS packed weight bytes are a different layout and must not be reused.
Prepare the TRTLLM FP4 weights once with ``TrtllmFp4Config.prepare_weights``
using ``activation=SiTU()`` or supply tensors already in that exact layout.
Storage is uint8 ``[896,768,1792]`` for FC1 and ``[896,3584,192]`` for FC2;
block-scale storage is uint8 or float8_e4m3fn ``[896,768,224]`` and
``[896,3584,24]`` respectively. All tensors are contiguous.

The existing six-entry NVFP4 ``quant_scales`` list is passed directly:

1. ``qX``: one float32 activation encoding scale.
2. FC1 block scales in the prepared layout.
3. ``dX*dW13``: float32 ``[896]`` FC1 dequantization scales.
4. ``qA``: float32 ``[896]`` intermediate encoding scales.
5. FC2 block scales in the prepared layout.
6. ``dA*dW2``: float32 ``[896]`` FC2 dequantization scales.

``qX=1/dX`` and ``qA=1/dA``. Expand a shared intermediate scale to a contiguous
896-element tensor during model preparation; submission performs no scale
arithmetic or broadcast allocation. Weight preparation's scales must be carried
with its packed bytes. Default SiTU is
``(4*tanh(g/4)*sigmoid(g))*(25*tanh(u/25))``. Existing ``situ_beta`` and
``situ_linear_beta`` arguments can supply caller-owned float32 ``[896]`` gate
and linear smoothing parameters directly. Omitting them uses the workspace's
prepared 4/25 constants.

Caller-owned execution
----------------------

.. code-block:: python

   from flashinfer.fused_moe import (
       cutlass_fused_moe, cutlass_fused_moe_workspace_size,
       cutlass_fused_moe_prepare_workspace,
   )
   import torch
   from flashinfer.tllm_enums import ActivationType

   # x, expert_ids, route_weights, w13, w2 and the six scales are prepared.
   nbytes = cutlass_fused_moe_workspace_size(
       max_tokens, 3584, 384, 896, 16,
       x_dtype=torch.bfloat16, weight_dtype=torch.uint8,
       activation_type=ActivationType.Situ, tp_size=8, backend="cake",
   )
   workspace = torch.empty(nbytes, dtype=torch.uint8, device=x.device)
   output = torch.empty_like(x)
   cutlass_fused_moe_prepare_workspace(workspace, x.shape[0], backend="cake")
   cutlass_fused_moe(
       x, expert_ids, route_weights, w13, w2, torch.bfloat16, quant_scales,
       activation_type=ActivationType.Situ, tp_size=8, backend="cake",
       workspace_buffer=workspace, output=output,
   )

The size query uses host metadata only. Explicit preparation initializes default
SiTU constants and loads the route before capture; it allocates no tensor
storage. Prepare every token count before its first submission. A maximum-size
buffer can hold every smaller prepared shape. Preparation and use must have
normal stream ordering; each concurrently executing call needs separate output
and workspace storage. Submission owns no graph and makes no CUDA allocation.
The caller may capture or replay the complete call. The generated per-stage PDL
and tensor-map ABI are preserved; ``enable_pdl=False`` is unsupported.
