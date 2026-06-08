"""vLLM Marlin nvfp4 W4A16 GEMM adapter for flashinfer_benchmark.

Lets the ``mm_w4a16_fp4`` routine benchmark vLLM's Marlin nvfp4 (4-bit weight,
bf16 activation) dense GEMM side-by-side with FlashInfer's own w4a16 backends,
for an apples-to-apples *performance* comparison.

Why it's apples-to-apples: both kernels do the same work -- dequantize an FP4
weight to bf16 and run a bf16 tensor-core matmul with fp32 accumulate.  With
identical M/N/K and dtypes, the benchmark's FLOPs (``2*m*n*k``) and bytes
(``A + (k/2)*n FP4 weight + (k/16)*n FP8 scales + out``) formulas hold for both
-- the Marlin repacked weight ``(K/16, N*2)`` int32 is exactly ``k*n/2`` bytes,
matching the packed-FP4 term -- so TFLOPS / TB/s are directly comparable.

vLLM is imported lazily so the benchmark only needs it when the ``marlin``
backend is actually requested.  The Marlin weights come from vLLM's
``rand_marlin_weight_nvfp4_like`` at the same (N, K) magnitude as FlashInfer's
weight.  The kernel does identical work regardless of the exact FP4 bit values,
so this is a faithful perf comparison, but it is NOT bit-identical to
FlashInfer's quantization of the same weight -- hence the caller checks the
Marlin output against its own dequantized reference, not FlashInfer's gold.
"""

import torch

MARLIN_BACKEND = "marlin"
GROUP_SIZE = 16  # nvfp4 always uses group_size 16


def is_vllm_marlin_available():
    """Return ``(ok: bool, detail: str)``.

    Lazy-imports vLLM's Marlin nvfp4 helpers and checks the compiled op is
    present.  ``detail`` carries the failure reason when ``ok`` is False.
    """
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: F401
            apply_fp4_marlin_linear,
            rand_marlin_weight_nvfp4_like,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (  # noqa: F401
            marlin_make_workspace_new,
        )
    except Exception as e:  # ImportError, or vLLM platform-init failure
        return False, f"{type(e).__name__}: {e}"
    if not hasattr(torch.ops._C, "marlin_gemm"):
        return False, "torch.ops._C.marlin_gemm missing (vLLM _C extension not built)"
    return True, "ok"


def prepare_marlin_weights(w, device, group_size=GROUP_SIZE):
    """Build Marlin-layout nvfp4 weights for a ``(N, K)`` bf16 weight ``w``.

    One-shot (NOT timed).  Returns a dict with the repacked int32 weight, the
    processed FP8 scales, the float32 global scale, a Marlin workspace, and the
    dequantized reference ``w_ref`` (shape ``(K, N)``) used for the caller's
    self-consistency refcheck.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        rand_marlin_weight_nvfp4_like,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )

    w_ref, qweight, scales, global_scale = rand_marlin_weight_nvfp4_like(w, group_size)
    workspace = marlin_make_workspace_new(device)
    return {
        "qweight": qweight,
        "scales": scales,
        "global_scale": global_scale,
        "workspace": workspace,
        "w_ref": w_ref,
    }


def make_marlin_runner(prep, n, k):
    """Return ``run(a) -> out`` invoking vLLM's Marlin nvfp4 GEMM.

    Output dtype follows the activation dtype (bf16); no cast is added inside
    the timed region.  Mirrors the ``run(a)`` closure shape of the FlashInfer
    backends so it drops straight into the benchmark's timing loop.
    """
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
    )

    qweight = prep["qweight"]
    scales = prep["scales"]
    global_scale = prep["global_scale"]
    workspace = prep["workspace"]

    def run(a):
        return apply_fp4_marlin_linear(
            input=a,
            weight=qweight,
            weight_scale=scales,
            weight_global_scale=global_scale,
            workspace=workspace,
            size_n=n,
            size_k=k,
        )

    return run
