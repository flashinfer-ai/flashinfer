# Experimental W4A8 MegaMoE EP16

This explicitly selected backend performs BF16 input quantization, EP dispatch,
MXFP4-weight FC1, SwiGLU and MXFP8 intermediate quantization, FC2, remote route
return and ordered BF16 output reduction in one CUDA launch.

The fixed geometry is H=3072, I=5120, 512 experts, top-k=8 and 32 contiguous
experts per rank. Both MXFP4 E2M1 weights and MXFP8 E4M3 activations use K32
UE8M0 scales. FC1 rounds through BF16 before SwiGLU; router weights are applied
before FC2 input quantization; BF16 route contributions are combined in slot
order using FP32. Fast-math arithmetic and unclamped activation match the
native DeepGEMM MegaMoE backend.

## Requirements and limits

- Exactly 16 SM103a GPUs with 152 physical SMs each and mutually accessible
  NVLink memory in one NVL72; one NCCL process per GPU.
- PyTorch NVSHMEM symmetric memory and CUDA 13.3 are the tested environment.
  DeepGEMM is needed for weight preparation and reference tests, not forward.
- 0–384 tokens per rank, equal token counts across ranks. Routing may contain
  duplicate experts. All-hot routing and ring wrap are covered by tests.
- Routing is fixed when the session is constructed. Setup clones the IDs;
  create another session to change routing. Inputs and router weights remain
  dynamic between forwards. Prepared weights must remain immutable.
- Approximately 514 MiB of symmetric workspace per rank, plus weights.
- All ranks must construct and invoke sessions in the same order. Use the
  setup CUDA stream, serially, and keep the session and tensor storage alive
  until its work finishes. CUDA graph capture and concurrent forwards on one
  session are unsupported. Failure on one rank requires aborting the job.
- This is a JIT-only experimental API, with no automatic backend selection.

## Usage

After initializing NCCL and selecting each rank's CUDA device:

```python
from flashinfer.moe_ep.cake_w4a8_megamoe_ep16 import (
    CakeW4A8MegaMoeEp16,
    preprocess_cake_w4a8_megamoe_ep16_weights,
)
from flashinfer.moe_ep.weights import MoEWeightPack

# w13_bf16: [32, 10240, 3072]; w2_bf16: [32, 3072, 5120].
# Setup is collective; IDs are int64 [tokens, 8], with values in [0, 512).
weights = preprocess_cake_w4a8_megamoe_ep16_weights(
    MoEWeightPack(w13_bf16, w2_bf16)
)
session = CakeW4A8MegaMoeEp16(weights, topk_ids)
session.forward(x_bf16, router_weights_fp32, out=output_bf16)
```

Prequantized MXFP4 `MoEWeightPack` inputs are also accepted, with packed
uint8/int8 weights and uint8 UE8M0 scales (or their exact FP32 powers of two)
in canonical K32 layout.
Weight preparation and session allocation are setup. Quantization of caller
activations is always inside the forward; it is not a prequantized-input API.

Distributed reference tests:

```bash
torchrun --nnodes=4 --nproc-per-node=4 --node-rank=$NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=29500 \
  -m pytest tests/experimental/test_cake_w4a8_megamoe_ep16.py -q
```

Use the same command on every node with its node rank. Tests compare to
native MXFP4/MXFP8 DeepGEMM and check 32 identical consecutive outputs,
empty inputs, tails, duplicates, all-hot ring wrap and immutable inputs.

A runnable synthetic-input example is `examples/cake_w4a8_megamoe_ep16.py`;
launch it with the same torchrun topology. For the six-row balanced/hotset50
comparison, run `benchmarks/bench_cake_w4a8_megamoe_ep16.py --output results/w4a8`
with that topology. The benchmark includes input quantization and all forward
work, alternates arm order, and reports three groups of cold-L2 CUPTI
measurements with a 100 ms warmup and a 1,000 ms measurement budget.
