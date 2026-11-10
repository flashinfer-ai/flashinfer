# SM120 native QMMA reference

`NativeQmmaReferenceLauncher` independently loads packed E2M1 operands and raw
E4M3 scales from global memory. A single warp computes a 16-feature × 8-token
tile with ascending K64 `mma.sync` operations, using a persistent FP32
accumulator. It uses no MegaMoE kernel, scheduler, TMA/SMEM layout, production
MMA helper or captured target result. Output is FP32 [tokens, features].

The reference deliberately places weights in MMA A and activations in MMA B,
matching the native instruction orientation of SM120 MegaMoE. The PTX lane and
scale mappings follow NVIDIA's specification:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16864

Requires CUDA nvcc supporting compute_120a, SM120 hardware and PyTorch.
The compiled shared library is cached under `NATIVE_QMMA_CACHE` (default:
`~/.cache/native_qmma_reference`), guarded by a process lock. Compile flags target
only sm_120a. No binaries are stored in the source tree.

```python
from moe_sm120_nvfp4_split.native_qmma_reference import NativeQmmaReferenceLauncher
out_fp32 = NativeQmmaReferenceLauncher()(
    a=activation_fp4, sfa=activation_raw_sf,
    b=weight_fp4, sfb=weight_raw_sf, n=hidden, k=reduction_k,
)
```

`compute_megamoe_reference_nvfp4(..., gemm_backend="native_qmma")` applies this
to both FC1 and FC2. Routing, SwiGLU, quantization and BF16 output construction
remain independently computed by the existing reference. It returns per-route
terms; use `flat_topk_bf16_reference` for ordinary top-k, or
`owner_grouped_bf16_reference` for the explicit owner-grouped contract.

`gemm_backend="torch_fp64"` retains the previous arithmetic reference. Native
byte equality and agreement with the independent numerical reference must be
reported separately. Passing the native reference does not mean passing the
historical `0.05 + 0.05 * abs(reference)` numerical threshold. Exact real-to-BF16
rounding and exact-to-FP32-to-BF16 are also different contracts.

The published ABI2735 candidate retains per-reader FC1 shared-read fencing,
warp rendezvous, and all consumer release arrivals.
Generic FC2 no longer resets the accumulator after each K128 tile or performs
software partial/correction summation. Existing N16 dual FC2 and both N16
correctness fixes are retained. Selection uses the frozen v15 candidate model and scoped paired policies;
out-of-domain cases retain compatibility fallback. See `SELECTOR.md` for
model scope, numerical permission and validation provenance.

The experiment's `audit_runner.py` runs full four-rank byte checks against this
reference, reports independent Torch FP64 numerical differences, and records
transport checks separately. Ordinary direct peer-store has no independent
pre-store transport oracle; owner-grouped mode can audit actual FC2 staging.
`test_reference.py` checks data/scale mappings and tails against exactly
representable independent FP64 products. `exact_dot_runner.py` re-encodes saved
historical operands for numerical-boundary diagnostics.

Standalone full-kernel validation (normal runner arguments apply):

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m moe_sm120_nvfp4_split.check_native_reference \
  --num_tokens_per_rank 64 --hidden 4096 --intermediate 4096 \
  --num_topk 6 --num_total_experts 256 --enable_static_expert_shape \
  --comm_backend p2p_direct --split_launch green_graph --gate_up_clamp 10
```

The standalone entry reports `independent_numerical_accuracy=NOT_CHECKED`; use
the normal `mega_runner` or the experiment audit to check that separate contract.
For an explicit Torch FP32 reference, call the full reference API with
`gemm_backend="torch_fp32"` and disable TF32. The default remains `torch_fp64`.
