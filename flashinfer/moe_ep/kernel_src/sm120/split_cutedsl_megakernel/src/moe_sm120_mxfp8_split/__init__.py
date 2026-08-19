"""SM120 Split-MegaMoE MXFP4-weight/MXFP8-activation swap-AB package."""

from .api import (
    KERNEL_CACHE_ABI,
    MegaMoECompileSpec,
    MegaMoEProblemSpec,
    Sm120JitConfig,
    SplitKernelBuildOptions,
    SplitKernelBundle,
    build_split_kernels,
    compile_combine_reduce,
    select_compile_spec,
)


def __getattr__(name):
    """Load CuTe/CUDA-heavy compatibility exports only when requested."""

    if name == "Sm120DispatchFc1Kernel":
        from .kernel_dispatch_fc1 import Sm120DispatchFc1Kernel

        return Sm120DispatchFc1Kernel
    if name == "Sm120Fc2CombineKernel":
        from .kernel_fc2_combine import Sm120Fc2CombineKernel

        return Sm120Fc2CombineKernel
    if name == "compile_topk_reduce":
        return compile_combine_reduce
    if name in ("GreenContextStreams", "NativeGreenContextGraph"):
        from .runtime import green_context

        return getattr(green_context, name)
    raise AttributeError(name)


__all__ = [
    "KERNEL_CACHE_ABI",
    "MegaMoECompileSpec",
    "MegaMoEProblemSpec",
    "Sm120JitConfig",
    "SplitKernelBuildOptions",
    "SplitKernelBundle",
    "build_split_kernels",
    "compile_combine_reduce",
    "select_compile_spec",
    # Compatibility exports, loaded lazily through __getattr__.
    "GreenContextStreams",
    "NativeGreenContextGraph",
    "Sm120DispatchFc1Kernel",
    "Sm120Fc2CombineKernel",
    "compile_topk_reduce",
]
