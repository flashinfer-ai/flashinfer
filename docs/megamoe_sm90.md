# SM90 W4A8 MegaMoE

`flashinfer.megamoe` runs expert-parallel MoE on NVLink-connected Hopper GPUs,
using NVFP4/MXFP4 weights, FP8 E4M3 activations and BF16 outputs. The kernels
perform token dispatch, two expert GEMMs, clamped SwiGLU and top-k reduction.

| Weights | Family | Launches | EP ranks |
| --- | --- | ---: | --- |
| NVFP4 | fused | 1 | 4 or 8 |
| NVFP4 | split | 2: L1, then L2 | 4 or 8 |
| MXFP4 | fused | 1 | Powers of two from 1 to 32 |

## Usage

Allocate `MegaMoESymmBuffer` collectively before graph capture. Prepare weights
with `transform_nvfp4_weights_for_mega_moe` or
`transform_mxfp4_weights_for_mega_moe`; all ranks stage inputs and call matching
kernels in matching order.

```python
from flashinfer import megamoe

buf.stage(x_fp8, x_sf, topk_idx, topk_weights)
megamoe.nvfp4_mega_moe(y, nv_w1, nv_w2, buf, kernel_family="auto")
# Or, with MXFP4 weights:
megamoe.mxfp4_mega_moe(y, mx_w1, mx_w2, buf)
```

NVFP4 supports 256 or 384 total experts. The process group selects EP4 or EP8;
no separate execution API is needed. EP4 uses the generic fused schedule;
EP8 retains its measured tuning policies. Auto selection uses fused through
256 tokens per rank, then split.
Input scales are FP32 per 128 channels; routing uses INT64 expert indices
(`-1` masks a route) and FP32 weights. Hidden width is a multiple of 256 and
intermediate width a multiple of 128. Raw weight scales use UINT8 E4M3 per
16 values for NVFP4 or E8M0 per 32 values for MXFP4.

## Performance

Cold L2 before each call; one MoE call per graph; CUDA events; maximum time
across ranks per call, then the median of 50 samples. Split timing includes
both launches. Packing, compilation and cache clearing are outside timing.
`M` is tokens per rank; capacity is the allocated maximum tokens per rank.

| Model | Hidden | Intermediate | Experts | Top-k |
| --- | ---: | ---: | ---: | ---: |
| Flash | 4096 | 2048 | 256 | 6 |
| MiMo | 6144 | 2048 | 384 | 8 |
| Pro | 7168 | 3072 | 384 | 6 |

<details>
<summary>All 82 workloads: H20 and H200 latency</summary>

| Kernel | Model / EP | M / rank | Capacity | H20 (us) | H200 (us) |
| --- | --- | ---: | ---: | ---: | ---: |
| NVFP4 fused | Flash / 8 | 1 | 384 | 90.960 | 81.280 |
| NVFP4 fused | Flash / 8 | 24 | 384 | 254.016 | 215.440 |
| NVFP4 fused | Flash / 8 | 32 | 384 | 258.624 | 218.944 |
| NVFP4 fused | Flash / 8 | 64 | 384 | 288.208 | 232.496 |
| NVFP4 fused | Flash / 8 | 96 | 384 | 327.056 | 227.824 |
| NVFP4 fused | Mimo / 8 | 1 | 384 | 151.376 | 136.560 |
| NVFP4 fused | Mimo / 8 | 24 | 384 | 509.616 | 422.048 |
| NVFP4 fused | Mimo / 8 | 32 | 384 | 519.904 | 426.096 |
| NVFP4 fused | Mimo / 8 | 64 | 384 | 579.584 | 458.912 |
| NVFP4 fused | Mimo / 8 | 96 | 384 | 629.312 | 456.368 |
| NVFP4 fused | Pro / 8 | 1 | 384 | 169.152 | 161.856 |
| NVFP4 fused | Pro / 8 | 24 | 384 | 849.856 | 703.008 |
| NVFP4 fused | Pro / 8 | 32 | 384 | 890.464 | 706.304 |
| NVFP4 fused | Pro / 8 | 64 | 384 | 930.048 | 727.424 |
| NVFP4 fused | Pro / 8 | 96 | 384 | 972.128 | 746.960 |
| NVFP4 split | Flash / 8 | 384 | 4224 | 891.280 | 339.296 |
| NVFP4 split | Flash / 8 | 1024 | 4224 | 1,588.000 | 601.840 |
| NVFP4 split | Flash / 8 | 4096 | 4224 | 5,223.680 | 1,824.592 |
| NVFP4 split | Mimo / 8 | 384 | 4224 | 1,637.664 | 635.584 |
| NVFP4 split | Mimo / 8 | 1024 | 4224 | 3,027.472 | 1,207.888 |
| NVFP4 split | Mimo / 8 | 4096 | 4224 | 10,236.048 | 3,887.568 |
| NVFP4 split | Pro / 8 | 384 | 4224 | 2,026.080 | 949.056 |
| NVFP4 split | Pro / 8 | 1024 | 4224 | 4,358.480 | 1,622.128 |
| NVFP4 split | Pro / 8 | 4096 | 4224 | 13,342.016 | 6,017.504 |
| MXFP4 fused | Flash / 8 | 1 | 4224 | 98.144 | 97.296 |
| MXFP4 fused | Flash / 8 | 24 | 4224 | 271.648 | 197.600 |
| MXFP4 fused | Flash / 8 | 32 | 4224 | 270.912 | 192.016 |
| MXFP4 fused | Flash / 8 | 64 | 4224 | 289.104 | 202.672 |
| MXFP4 fused | Flash / 8 | 96 | 4224 | 310.480 | 220.752 |
| MXFP4 fused | Flash / 8 | 384 | 4224 | 801.184 | 393.040 |
| MXFP4 fused | Flash / 8 | 1024 | 4224 | 1,504.720 | 704.352 |
| MXFP4 fused | Flash / 8 | 4096 | 4224 | 5,195.056 | 2,406.736 |
| MXFP4 fused | Mimo / 8 | 1 | 4224 | 131.008 | 127.344 |
| MXFP4 fused | Mimo / 8 | 24 | 4224 | 501.568 | 339.296 |
| MXFP4 fused | Mimo / 8 | 32 | 4224 | 561.024 | 360.208 |
| MXFP4 fused | Mimo / 8 | 64 | 4224 | 582.112 | 392.960 |
| MXFP4 fused | Mimo / 8 | 96 | 4224 | 626.848 | 434.160 |
| MXFP4 fused | Mimo / 8 | 384 | 4224 | 1,436.624 | 647.120 |
| MXFP4 fused | Mimo / 8 | 1024 | 4224 | 2,857.840 | 1,355.680 |
| MXFP4 fused | Mimo / 8 | 4096 | 4224 | 10,136.752 | Pending* |
| MXFP4 fused | Pro / 8 | 1 | 4224 | 221.840 | 157.376 |
| MXFP4 fused | Pro / 8 | 24 | 4224 | 787.984 | 519.264 |
| MXFP4 fused | Pro / 8 | 32 | 4224 | 836.160 | 550.400 |
| MXFP4 fused | Pro / 8 | 64 | 4224 | 935.568 | 614.960 |
| MXFP4 fused | Pro / 8 | 96 | 4224 | 975.072 | 637.424 |
| MXFP4 fused | Pro / 8 | 384 | 4224 | 1,708.752 | 750.672 |
| MXFP4 fused | Pro / 8 | 1024 | 4224 | 4,049.184 | 1,810.048 |
| MXFP4 fused | Pro / 8 | 4096 | 4224 | 13,173.744 | 29,917.920 |
| MXFP4 fused | Flash / 8 | 8 | 384 | 214.080 | 171.856 |
| MXFP4 fused | Flash / 8 | 16 | 384 | 238.896 | 182.560 |
| MXFP4 fused | Flash / 8 | 32 | 384 | 271.248 | 193.280 |
| MXFP4 fused | Flash / 8 | 48 | 384 | 280.800 | 197.504 |
| MXFP4 fused | Flash / 8 | 64 | 384 | 289.040 | 202.880 |
| MXFP4 fused | Flash / 8 | 96 | 384 | 307.504 | 220.672 |
| MXFP4 fused | Flash / 8 | 112 | 384 | 318.624 | 221.040 |
| MXFP4 fused | Flash / 8 | 128 | 384 | 351.120 | 221.904 |
| MXFP4 fused | Flash / 8 | 160 | 384 | 438.320 | 236.144 |
| MXFP4 fused | Flash / 8 | 192 | 384 | 440.304 | 229.760 |
| MXFP4 fused | Flash / 8 | 256 | 384 | 457.088 | 1,060.400 |
| MXFP4 fused | Flash / 8 | 384 | 384 | 814.176 | 1,155.168 |
| MXFP4 fused | Flash / 4 | 8 | 768 | 261.088 | 211.152 |
| MXFP4 fused | Flash / 4 | 16 | 768 | 380.096 | 267.680 |
| MXFP4 fused | Flash / 4 | 32 | 768 | 438.016 | 286.512 |
| MXFP4 fused | Flash / 4 | 48 | 768 | 451.552 | 306.160 |
| MXFP4 fused | Flash / 4 | 64 | 768 | 496.848 | 330.000 |
| MXFP4 fused | Flash / 4 | 96 | 768 | 518.560 | 340.400 |
| MXFP4 fused | Flash / 4 | 128 | 768 | 535.520 | 351.376 |
| MXFP4 fused | Flash / 4 | 160 | 768 | 555.216 | 384.320 |
| MXFP4 fused | Flash / 4 | 192 | 768 | 573.584 | 386.256 |
| MXFP4 fused | Flash / 4 | 224 | 768 | 589.936 | 386.784 |
| MXFP4 fused | Flash / 4 | 256 | 768 | 614.192 | 388.768 |
| MXFP4 fused | Flash / 4 | 384 | 768 | 857.952 | 394.448 |
| MXFP4 fused | Flash / 4 | 512 | 768 | 866.240 | 406.992 |
| MXFP4 fused | Flash / 4 | 768 | 768 | 1,519.440 | 694.912 |
| MXFP4 fused | Mimo / 4 | 8 | 384 | 527.168 | 361.936 |
| MXFP4 fused | Mimo / 4 | 32 | 384 | 854.064 | 549.344 |
| MXFP4 fused | Mimo / 4 | 64 | 384 | 1,008.064 | 647.968 |
| MXFP4 fused | Mimo / 4 | 128 | 384 | 1,101.408 | 711.840 |
| MXFP4 fused | Mimo / 4 | 192 | 384 | 1,216.400 | 791.616 |
| MXFP4 fused | Mimo / 4 | 240 | 384 | 1,204.288 | 795.072 |
| MXFP4 fused | Mimo / 4 | 288 | 384 | 1,304.784 | 797.056 |
| MXFP4 fused | Mimo / 4 | 384 | 384 | 1,804.144 | 803.552 |

</details>


<details>
<summary>NVFP4 EP4: 9 workloads (capacity 384)</summary>

| Model | Family | M / rank | H20 (us) | H200 (us) |
| --- | --- | ---: | ---: | ---: |
| Flash | fused | 1 | 95.344 | 80.928 |
| Flash | fused | 96 | 842.656 | 399.984 |
| Flash | split | 384 | 1,051.264 | 512.256 |
| MiMo | fused | 1 | 163.888 | 131.968 |
| MiMo | fused | 96 | 1,758.592 | 831.920 |
| MiMo | split | 384 | 2,232.464 | 1,068.272 |
| Pro | fused | 1 | 202.512 | 175.536 |
| Pro | fused | 96 | 3,007.920 | 1,401.072 |
| Pro | split | 384 | 3,766.832 | 1,752.096 |

</details>

H200 EP8 regression against the original NVFP4 kernels covers 12 points
(Flash/MiMo/Pro, M=1/32/96/384, capacity 384): latency changes range from
-1.07% to +0.64%. Flash M96 is 227.808 us before and
227.440 us after. EP4 results use the generic schedule without EP4-specific tuning.

## Validation

```bash
pytest tests/moe/test_megamoe_fp4_ep.py tests/moe/test_megamoe_fp4_weights.py tests/moe/test_megamoe_fp4_sm90.py
```

CPU plan/cache and weight-layout checks pass. EP4 numerical, receive-count,
workspace-reuse and CUDA graph checks pass on H20 and H200, including Flash,
MiMo and Pro shapes and both push-dispatch barrier modes. H200 EP8 regression
outputs match the original NVFP4 fused/split kernels bitwise.

Reproduce EP4 latency with the table's cache and timing protocol:

```bash
python benchmarks/bench_megamoe_fp4_sm90.py --ep 4 --formats nvfp4 \
  --shapes flash mimo pro --tokens 1 96 384 --cold-l2 --repeats 1 --rounds 50
```

The benchmark defaults to warm-cache repeated calls; pass `--cold-l2` and
`--repeats 1` for one cold-L2 invocation per sample.
