# vLLM SM120 integration and host sanitizer validation

Date: 2026-08-13
PR: #4495 (`e63cf54`)
Host: RTX PRO 6000 Blackwell Server Edition, SM120, driver 595.80
Container: `vllm024_hpc_sm120_groupgemm_backup`
vLLM: 0.24.0

## vLLM integration

The experimental `flashinfer_b12x_direct` backend was wired into vLLM's
NVFP4 oracle and modular MoE dispatcher. The test used synthetic ModelOpt
layout weights with `E=64`, `top-k=8`, `H=2048`, `I=512`, and BF16 hidden
states, matching the Qwen benchmark shape. Both W4A16 and NVFP4 were tested
for `M=1,2,4,8`.

Every case passed:

- model-load scale normalization;
- vLLM `FusedMoEKernel.apply` prepare/dispatch/finalize;
- eager repeatability;
- Direct expert CUDA Graph replay;
- complete vLLM modular-dispatch CUDA Graph replay.

The legacy `flashinfer_b12x` mapping still resolves to
`FlashInferB12xExperts`; only the explicit `flashinfer_b12x_direct` backend
resolves to `FlashInferB12xDirectExperts`.

The full-model serving benchmark was not run for this backend: the server's
`/data/models` contains BF16/FP8 Qwen checkpoints but no NVFP4/W4A16 checkpoint,
and the requested JoyAI model is not present. The synthetic test therefore
validates the kernel/dispatcher contract, not end-to-end model quality or
throughput.

## Compute Sanitizer

The complete CUDA 13.2 Compute Sanitizer directory was copied from the host to
`/usr/local/cuda-13.2/compute-sanitizer` in the container. The binary checksum
matches the host (`dcec87e3437d8127cc58a0b186d4830831ce18a0cd4c5846ac84eac396312172`).
The tool reports version `2026.1.0.0 (build 37182542)`.

The container's driver/runtime combination emits a `cuGetProcAddress` version
compatibility diagnostic (`cudaVersion 13030` vs driver `13020`). The runs use
`--report-api-errors no` so this initialization diagnostic does not obscure the
kernel reports.

| Tool | Mode | M | Result | Details |
|---|---|---:|---|---|
| memcheck | W4A16 | 1, 8 | PASS | `ERROR SUMMARY: 0 errors` |
| memcheck | NVFP4 | 1, 8 | PASS | `ERROR SUMMARY: 0 errors` |
| racecheck | NVFP4 | 1 | PASS | `RACECHECK SUMMARY: 0 hazards displayed` |
| racecheck | W4A16 | 1 | FOLLOW-UP | 612 shared-memory WAR hazards in the CuteDSL W4A16 kernel; one hazard class is reported as an error between a read by thread 31 and a write by thread 0. |
| synccheck | W4A16/NVFP4 | 1 | FOLLOW-UP | The first W4A16 warm-up causes the target process to return `cudaErrorIllegalAddress`; sanitizer itself reports `ERROR SUMMARY: 0 errors`. |

The normal W4A16 racecheck run lists 612 shared-memory WAR hazards. With
async-copy race tracking disabled, the log displays 100 hazard records (192
errors and 200 warnings in that report); these are different
display/classification totals, not evidence that the underlying shared-memory
hazard disappeared. The warning is tied to the kernel's
warp-level asynchronous pipeline. The W4A16 racecheck result should remain a
follow-up item rather than being treated as a clean sanitizer pass. Eager and
CUDA Graph correctness checks, as well as memcheck, pass for the same shape.

## Reproduction

The vLLM integration test uses the existing validation worktree and can be
run after setting the FlashInfer JIT environment:

```bash
export FLASHINFER_CUDA_ARCH_LIST=12.0
export FLASHINFER_EXTRA_CUDAFLAGS=-I/tmp/flashinfer-nvrtc-include
export FLASHINFER_EXTRA_LDFLAGS=-L/tmp/flashinfer-cuda-libs
export LD_LIBRARY_PATH=/tmp/flashinfer-cuda-libs:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/workspace/flashinfer_b12x_direct_quantized_pr_validation_20260813:$PYTHONPATH
cd /workspace/flashinfer_b12x_direct_quantized_pr_validation_20260813
.venv/bin/python test_vllm_integration.py
```

For memcheck, use the host-copied binary and add
`--report-api-errors no --error-exitcode 1`.
