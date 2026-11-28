"""SM120 Split-MegaMoE NVFP4-weight/NVFP4-activation swap-AB package."""

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
]

from .execution_plan import CollectiveExecutionPlan, select_collective_execution_plan
__all__ += ["CollectiveExecutionPlan", "select_collective_execution_plan"]

from .tuning_config import TuningConfig, load_tuning_config, hardware_identity
__all__ += ["TuningConfig", "load_tuning_config", "hardware_identity"]
