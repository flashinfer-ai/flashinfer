# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Torch-facing dispatch for the vendored PrimsTS dense GEMM kernel.

The device program has module globals because CuTe captures Python constants.
This wrapper never mutates a shared loaded module: one isolated module is
created for each immutable config and a compiled program is cached by that
same config.  That is what makes concurrent fused epilogues safe.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.utils import (
    backend_requirement,
    get_compute_capability,
    supported_compute_capability,
)

from .config import PrimsTsGemmConfig
from .support import nvfp4_128x4_numel as _nvfp4_128x4_numel
from .support import validate_dense_gemm
from .tactics import derived_ab_stages, fallback_cluster

_COMPILED: dict[tuple[PrimsTsGemmConfig, int, int, bool], object] = {}
_KERNEL_MODULES: dict[PrimsTsGemmConfig, ModuleType] = {}
_COMPILE_LOCK = threading.Lock()
_SCALAR_CACHE: dict[tuple[str, int | None, float], torch.Tensor] = {}
_SCALED_OUTPUT_FACTOR_CACHE: dict[tuple[str, int | None, int, float], torch.Tensor] = {}
_MAX_ACTIVE_CLUSTERS: dict[tuple[str, int | None, int], int] = {}

_FP8 = torch.float8_e4m3fn
_SUPPORTED_ARCHES = {100, 103, 107}


@supported_compute_capability([100, 103, 107])
def _check_gemm(
    a: Optional[torch.Tensor] = None,
    a_packed: Optional[torch.Tensor] = None,
    **_: object,
) -> bool:
    operand = a if a is not None else a_packed
    if not isinstance(operand, torch.Tensor) or not operand.is_cuda:
        raise ValueError("PrimsTS GEMM inputs must be CUDA tensors")
    return True


def _validate_cuda_tensors(*tensors: Optional[torch.Tensor]) -> torch.device:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        raise ValueError("at least one tensor is required")
    device = present[0].device
    for tensor in present:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("all PrimsTS GEMM tensors must be CUDA tensors")
        if tensor.device != device:
            raise ValueError(f"all tensors must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError("PrimsTS GEMM tensors must be contiguous")
    cc = get_compute_capability(device)
    arch = cc[0] * 10 + cc[1]
    if arch not in _SUPPORTED_ARCHES:
        raise RuntimeError(
            f"PrimsTS dense GEMM supports SM100, SM103, and SM107, got SM{arch}"
        )
    return device


def _output_format(out_dtype: torch.dtype) -> str:
    if out_dtype == torch.bfloat16:
        return "bf16"
    if out_dtype == _FP8:
        return "fp8_e4m3"
    if out_dtype == torch.uint8:
        return "nvfp4_e2m1"
    raise ValueError(
        "out_dtype must be torch.bfloat16, torch.float8_e4m3fn, or torch.uint8"
    )


def _scalar(device: torch.device, value: float) -> torch.Tensor:
    key = (device.type, device.index, float(value))
    result = _SCALAR_CACHE.get(key)
    if result is None:
        result = torch.tensor([value], device=device, dtype=torch.float32)
        _SCALAR_CACHE[key] = result
    return result


def _scaled_output_factor(scale: torch.Tensor, factor: float) -> torch.Tensor:
    """Return the kernel-local input-dequant times output-encode factor."""
    key = (scale.device.type, scale.device.index, scale.data_ptr(), float(factor))
    result = _SCALED_OUTPUT_FACTOR_CACHE.get(key)
    if result is None:
        result = torch.empty_like(scale)
        _SCALED_OUTPUT_FACTOR_CACHE[key] = result
    torch.mul(scale, factor, out=result)
    return result


def _output(
    device: torch.device,
    m: int,
    logical_n: int,
    output_format: str,
    out: Optional[torch.Tensor],
    output_quant_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
    if output_format == "bf16":
        if output_quant_scale is not None:
            raise ValueError("output_quant_scale is only valid for FP8 or NVFP4 output")
        expected_shape, expected_dtype = (m, logical_n), torch.bfloat16
        if out is None:
            return (
                torch.empty(expected_shape, device=device, dtype=expected_dtype),
                None,
                False,
            )
        if (
            out.device != device
            or out.dtype != expected_dtype
            or tuple(out.shape) != expected_shape
        ):
            raise ValueError(f"out must be BF16 with shape {expected_shape}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        return out, None, False

    if output_format == "fp8_e4m3":
        expected_shape, expected_dtype = (m, logical_n), _FP8
    else:
        if logical_n % 16:
            raise ValueError(
                "NVFP4 output requires its logical N dimension to be divisible by 16"
            )
        expected_shape, expected_dtype = (m, logical_n // 2), torch.uint8
    if out is None:
        out = torch.empty(expected_shape, device=device, dtype=expected_dtype)
    elif (
        out.device != device
        or out.dtype != expected_dtype
        or tuple(out.shape) != expected_shape
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"out must be contiguous {expected_dtype} with physical shape {expected_shape}"
        )

    allocated_scale = output_quant_scale is None
    if output_quant_scale is None:
        # A unit encode scale is always correct and is graph-capture friendly.
        # Applications may provide a calibrated scalar to improve precision.
        output_quant_scale = _scalar(device, 1.0)
    if output_quant_scale.device != device or output_quant_scale.dtype != torch.float32:
        raise ValueError(
            "output_quant_scale must be a CUDA float32 tensor on the input device"
        )
    if output_quant_scale.numel() != 1 or not output_quant_scale.is_contiguous():
        raise ValueError(
            "output_quant_scale must be contiguous with exactly one element"
        )
    return out, output_quant_scale, allocated_scale


def _load_kernel(config: PrimsTsGemmConfig) -> ModuleType:
    module = _KERNEL_MODULES.get(config)
    if module is not None:
        return module
    with _COMPILE_LOCK:
        module = _KERNEL_MODULES.get(config)
        if module is not None:
            return module
        tag = f"flashinfer.prims_ts.gemm._kernel_{abs(hash(config)):x}"
        spec = importlib.util.spec_from_file_location(
            tag, Path(__file__).with_name("kernel.py")
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load PrimsTS GEMM kernel source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[tag] = module
        spec.loader.exec_module(module)
        module._set_io_dtypes(
            "fp8" if config.operand_format == "fp8_e4m3" else "fp4",
            {"bf16": "bf16", "fp8_e4m3": "fp8", "nvfp4_e2m1": "fp4"}[
                config.output_format
            ],
        )
        module.use_per_token_channel_scale = config.operand_format == "fp8_e4m3"
        module.use_gated_activation = config.epilogue == "swiglu"
        module.use_fused_qknorm_rope = config.epilogue == "qkv_qknorm_rope"
        module.use_block_major_k = False
        module.use_tma_store = False
        if config.tmem_overlap and (
            config.operand_format != "nvfp4_e2m1"
            or config.output_format != "bf16"
            or config.epilogue != "linear"
        ):
            raise ValueError("tmem_overlap supports NVFP4 BF16 linear epilogues only")
        module.use_nvfp4_tmem_overlap = config.tmem_overlap
        module._tile_k_override = config.tile_k
        if config.nvfp4_mma_k not in (64, 96):
            raise ValueError("nvfp4_mma_k must be 64 or 96")
        if config.nvfp4_mma_k == 96 and (
            config.arch != 103 or config.operand_format != "nvfp4_e2m1"
        ):
            raise ValueError("MMA-K=96 requires SM103 NVFP4 operands")
        module.nvfp4_mma_k = config.nvfp4_mma_k
        module._tile_n_override = config.tile_n
        default_epilogue_warps = (
            8 if config.epilogue != "linear" or config.tmem_overlap else 4
        )
        module.num_epilogue_warps = (
            default_epilogue_warps
            if config.epilogue_warps is None
            else config.epilogue_warps
        )
        if module.num_epilogue_warps not in (4, 8):
            raise ValueError("epilogue_warps must be 4 or 8")
        if config.epilogue != "linear" and module.num_epilogue_warps != 8:
            raise ValueError("fused epilogues require 8 epilogue warps")
        module.threads_in_epilogue = module.num_epilogue_warps * 32
        forced_stages = derived_ab_stages(
            config.operand_format,
            config.epilogue,
            config.nvfp4_mma_k,
            config.tile_n,
            config.tile_k,
        )
        if config.ab_stages is None:
            module.ab_stages = forced_stages
        elif config.ab_stages not in (1, 2, 4, 5, 6):
            raise ValueError(
                f"ab_stages must be one of 1, 2, 4, 5, 6, got {config.ab_stages}"
            )
        elif (
            config.nvfp4_mma_k == 96
            and config.tile_k == 768
            and config.ab_stages != forced_stages
        ):
            raise ValueError(
                "NVFP4 MMA-K=96 with tile_k=768 forces "
                f"ab_stages={forced_stages}, got {config.ab_stages}"
            )
        else:
            module.ab_stages = config.ab_stages
        cm, cn, ck = config.cluster_shape
        module.cluster_shape_mnk = (cm, cn, ck)
        module.cluster_m, module.cluster_n, module.cluster_size = cm, cn, cm * cn
        module.num_pairs = (cm * cn) // module.num_mma_ctas
        module.num_pair_rows, module.num_pair_cols = cm // module.num_mma_ctas, cn
        module._a_mcast_template = sum(
            1 << (module.num_mma_ctas * col) for col in range(cn)
        )
        module._b_mcast_template = sum(
            1 << (cn * module.num_mma_ctas * row)
            for row in range(cm // module.num_mma_ctas)
        )
        module.fallback_cluster_shape_mnk = fallback_cluster(config.cluster_shape)
        module.use_clc_dynamic_scheduler = config.scheduler == "clc_dynamic"
        module._refresh_input_dependent_config()
        module._validate_cluster_shape(
            module.cluster_shape_mnk, option_name="cluster_shape_mnk"
        )
        if module.fallback_cluster_shape_mnk is not None:
            module._validate_fallback_cluster_shape(
                module.fallback_cluster_shape_mnk, module.cluster_shape_mnk
            )
        _KERNEL_MODULES[config] = module
        return module


def _compile(
    config: PrimsTsGemmConfig,
    module: ModuleType,
    args: tuple[object, ...],
) -> object:
    activation = args[0]
    if not isinstance(activation, torch.Tensor):
        raise TypeError("the PrimsTS compile activation must be a Torch tensor")
    device_index = activation.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    max_active_clusters = int(args[4])
    # The checker runs inside compile, and a cache hit never rebuilds the
    # TaskManager. Keep a checked compile distinct from the production one.
    debug_checks = bool(module._prims_ts_debug_checks_enabled())
    compile_key = (config, max_active_clusters, device_index, debug_checks)
    compiled = _COMPILED.get(compile_key)
    if compiled is not None:
        return compiled
    with _COMPILE_LOCK:
        compiled = _COMPILED.get(compile_key)
        if compiled is not None:
            return compiled

        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream

        from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        (
            a,
            b,
            c,
            mnk,
            max_active_clusters,
            bias,
            x_scale,
            weight_scale,
            q_norm,
            k_norm,
            cos_sin,
            positions,
            sfa,
            sfb,
            sf_c,
            scale_c,
            scale_gate,
            qkv_scale,
        ) = args

        def compile_kernel():
            return cute.compile[cute.FrontendNext](
                module.host_function,
                _adapt(a),
                _adapt(b),
                _adapt(c),
                mnk,
                max_active_clusters,
                make_fake_stream(use_tvm_ffi_env_stream=True),
                _adapt(bias, 16),
                cutlass.Float32(1.0),
                _adapt(x_scale, 16),
                _adapt(weight_scale, 16),
                _adapt(q_norm, 16),
                _adapt(k_norm, 16),
                _adapt(cos_sin, 16),
                _adapt(positions, 16),
                _adapt(sfa, 16),
                _adapt(sfb, 16),
                _adapt(sf_c, 16),
                _adapt(scale_c, 4),
                _adapt(scale_gate, 4),
                _adapt(qkv_scale, 4),
                options="--opt-level 2 --enable-tvm-ffi",
            )

        def field(value: object) -> str:
            if value is None:
                return "x"
            if isinstance(value, tuple):
                return "x".join(str(item) for item in value)
            return str(value)

        kernel_name = "_".join(
            (
                f"sm{config.arch}",
                config.operand_format,
                config.output_format,
                config.epilogue,
                f"bias{int(config.has_bias)}",
                f"hd{field(config.head_dim)}",
                f"neox{field(config.is_neox)}",
                f"tm{config.tile_m}",
                f"tn{config.tile_n}",
                f"tk{config.tile_k}",
                f"f4mk{config.nvfp4_mma_k}",
                f"epiw{field(config.epilogue_warps)}",
                f"overlap{int(config.tmem_overlap)}",
                f"abs{module.ab_stages}",
                f"cluster{field(config.cluster_shape)}",
                config.scheduler,
                f"bucket{field(config.shape_bucket)}",
                f"qkvs{int(config.has_qkv_scale)}",
                f"mac{max_active_clusters}",
            )
        )
        if debug_checks:
            kernel_name += "_dbg"
        compiled = build_and_load_cute_dsl_kernel(
            "prims_ts_dense_gemm",
            kernel_name,
            compile_kernel,
            extra_key_files=(
                __file__,
                str(Path(__file__).with_name("config.py")),
                str(Path(module.__file__)),
            ),
        )
        _COMPILED[compile_key] = compiled
        return compiled


def _max_active_clusters(
    config: PrimsTsGemmConfig,
    module: ModuleType,
    device: torch.device,
) -> int:
    """Return the cached device limit for one cluster specialization."""
    import cutlass.utils as cutlass_utils

    cluster_size = (
        module.cluster_shape_mnk[0]
        * module.cluster_shape_mnk[1]
        * module.cluster_shape_mnk[2]
    )
    active_cluster_key = (device.type, device.index, cluster_size)
    max_active_clusters = _MAX_ACTIVE_CLUSTERS.get(active_cluster_key)
    if max_active_clusters is None:
        # HardwareInfo itself JIT-compiles a tiny query kernel. Serialize its
        # first construction too: concurrent steady-state launches must never
        # enter the DSL compiler.
        with _COMPILE_LOCK:
            max_active_clusters = _MAX_ACTIVE_CLUSTERS.get(active_cluster_key)
            if max_active_clusters is None:
                with torch.cuda.device(device):
                    max_active_clusters = (
                        cutlass_utils.HardwareInfo().get_max_active_clusters(
                            cluster_size
                        )
                    )
                _MAX_ACTIVE_CLUSTERS[active_cluster_key] = max_active_clusters
    if max_active_clusters <= 0:
        raise RuntimeError(f"no active cluster supports cluster size {cluster_size}")
    return max_active_clusters


def _adapt(tensor: Optional[torch.Tensor], alignment: int = 1):
    """Create a dynamic CuTe wrapper while compiling the TVM-FFI launcher."""
    if tensor is None:
        return None
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(tensor, assumed_align=alignment).mark_layout_dynamic()


def _launch(
    config: PrimsTsGemmConfig,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    mnk: tuple[int, int, int],
    *,
    bias: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    q_norm: Optional[torch.Tensor] = None,
    k_norm: Optional[torch.Tensor] = None,
    cos_sin: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    sfa: Optional[torch.Tensor] = None,
    sfb: Optional[torch.Tensor] = None,
    sf_c: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    scale_c: Optional[torch.Tensor] = None,
    scale_gate: Optional[torch.Tensor] = None,
    qkv_scale: Optional[torch.Tensor] = None,
) -> None:
    module = _load_kernel(config)
    device = a.device
    max_active_clusters = _max_active_clusters(config, module, device)
    values = (
        a,
        b,
        c,
        mnk,
        max_active_clusters,
        bias,
        x_scale,
        weight_scale,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        sfa,
        sfb,
        sf_c,
        scale_c,
        scale_gate,
        qkv_scale,
    )
    compiled = _compile(config, module, values)
    compiled(
        a,
        b,
        c,
        mnk,
        bias,
        float(scale),
        x_scale,
        weight_scale,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        sfa,
        sfb,
        sf_c,
        scale_c,
        scale_gate,
        qkv_scale,
    )


class PreparedFp4Linear:
    """Prepared dense FP4 launch path for a fixed projection.

    Static kernel configuration, weights, weight scales, optional bias, and
    Q/K normalization weights are resolved once. Steady-state calls pass Torch
    tensors directly to a cached TVM-FFI launcher on the caller's current
    stream; DLPack adaptation is confined to compilation.
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        weight_block_scale: torch.Tensor,
        input_global_scale: float,
        weight_global_scale: float,
        *,
        epilogue: str,
        out_dtype: torch.dtype = torch.bfloat16,
        output_quant_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        q_norm: Optional[torch.Tensor] = None,
        k_norm: Optional[torch.Tensor] = None,
        qkv_scale: Optional[torch.Tensor] = None,
        head_dim: Optional[int] = None,
        is_neox: Optional[bool] = None,
        mma_k: int = 64,
        tile_k: int = 256,
        tmem_overlap: bool = False,
    ) -> None:
        device = _validate_cuda_tensors(
            weight_packed,
            weight_block_scale,
            output_quant_scale,
            bias,
            q_norm,
            k_norm,
            qkv_scale,
        )
        if (
            weight_packed.dtype != torch.uint8
            or weight_packed.ndim != 2
            or not weight_packed.is_contiguous()
        ):
            raise ValueError("weight_packed must be a contiguous rank-2 uint8 tensor")
        n, packed_k = weight_packed.shape
        k = packed_k * 2
        logical_n = n // 2 if epilogue == "swiglu" else n
        output_format = _output_format(out_dtype)
        cc = get_compute_capability(device)
        arch = cc[0] * 10 + cc[1]
        validate_dense_gemm(
            arch=arch,
            operand_format="nvfp4_e2m1",
            epilogue=epilogue,
            n=n,
            k=k,
            out_dtype=out_dtype,
            weight_scale=weight_block_scale,
            a_global_scale=input_global_scale,
            weight_global_scale=weight_global_scale,
            bias=bias,
            q_norm=q_norm,
            k_norm=k_norm,
            head_dim=head_dim,
            is_neox=is_neox,
            qkv_scale=qkv_scale,
        )
        self.config = PrimsTsGemmConfig(
            arch,
            "nvfp4_e2m1",
            output_format,
            epilogue,
            bias is not None,
            head_dim,
            is_neox,
            tile_k=tile_k,
            nvfp4_mma_k=mma_k,
            tmem_overlap=tmem_overlap,
            has_qkv_scale=qkv_scale is not None,
        )
        self.device = device
        self.n = n
        self.k = k
        self.logical_n = logical_n
        self.output_format = output_format
        self.alpha = float(input_global_scale) * float(weight_global_scale)
        self._out_dtype = out_dtype
        # Defaults stay unpinned so a later autotune pass can replace them.
        # __init__ still loads this config: callers inspect the module, and
        # an untuned launch uses the same kernel via tactic -1.
        self._pinned = mma_k != 64 or tile_k != 256 or tmem_overlap
        self._module = _load_kernel(self.config)
        self._max_active_clusters = _max_active_clusters(
            self.config, self._module, device
        )

        # Retain static tensor owners for the prepared projection.
        self.weight_packed = weight_packed
        self.weight_block_scale = weight_block_scale
        self.bias = bias
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.qkv_scale = qkv_scale
        self._return_quant_scale = False
        self.output_quant_scale = None
        self._scale_c_tensor = None
        self._scale_gate_tensor = None
        if output_format == "nvfp4_e2m1":
            self._return_quant_scale = output_quant_scale is None
            self.output_quant_scale = (
                _scalar(device, 1.0)
                if output_quant_scale is None
                else output_quant_scale
            )
            if (
                self.output_quant_scale.device != device
                or self.output_quant_scale.dtype != torch.float32
                or self.output_quant_scale.numel() != 1
                or not self.output_quant_scale.is_contiguous()
            ):
                raise ValueError(
                    "output_quant_scale must be one contiguous CUDA float32 value"
                )
            # This work happens once when the projection is prepared, rather
            # than launching a scale multiply before every MLP-up GEMM.
            self._scale_c_tensor = _scaled_output_factor(
                self.output_quant_scale, self.alpha
            )
            self._scale_gate_tensor = _scalar(device, self.alpha)

    def __call__(
        self,
        a_packed: torch.Tensor,
        a_block_scale: torch.Tensor,
        *,
        cos_sin: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ):
        """Launch the prepared projection for one activation."""
        if (
            a_packed.device != self.device
            or a_packed.dtype != torch.uint8
            or a_packed.ndim != 2
            or a_packed.shape[1] * 2 != self.k
            or not a_packed.is_contiguous()
        ):
            raise ValueError(
                f"a_packed must be contiguous uint8 [M, {self.k // 2}] on {self.device}"
            )
        m = a_packed.shape[0]
        if a_block_scale.device != self.device:
            raise ValueError(f"a_block_scale must be on {self.device}")
        validate_dense_gemm(
            arch=self.config.arch,
            operand_format="nvfp4_e2m1",
            epilogue=self.config.epilogue,
            n=self.n,
            k=self.k,
            out_dtype=self._out_dtype,
            m=m,
            a_scale=a_block_scale,
            weight_scale=self.weight_block_scale,
            a_global_scale=1.0,
            weight_global_scale=1.0,
            bias=self.bias,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            cos_sin=cos_sin,
            positions=positions,
            head_dim=self.config.head_dim,
            is_neox=self.config.is_neox,
            qkv_scale=self.qkv_scale,
        )
        if self.config.epilogue == "qkv_qknorm_rope":
            if (
                cos_sin is None
                or cos_sin.device != self.device
                or cos_sin.dtype != torch.float32
                or tuple(cos_sin.shape) != (m, self.config.head_dim)
            ):
                raise ValueError("cos_sin has an invalid prepared-QKV contract")
            if (
                positions is None
                or positions.device != self.device
                or positions.dtype != torch.int64
                or tuple(positions.shape) != (m,)
            ):
                raise ValueError("positions has an invalid prepared-QKV contract")

        c, _, _ = _output(
            self.device,
            m,
            self.logical_n,
            self.output_format,
            out,
            self.output_quant_scale,
        )
        sf_c = None
        if self.output_format == "nvfp4_e2m1":
            sf_c = torch.empty(
                (_nvfp4_128x4_numel(m, self.logical_n),),
                device=self.device,
                dtype=torch.uint8,
            )
        if not self._pinned:
            from .runner import dense_gemm_op_name, launch_dense_gemm

            launch_dense_gemm(
                dense_gemm_op_name("nvfp4_e2m1", self.config.epilogue),
                arch=self.config.arch,
                operand_format="nvfp4_e2m1",
                output_format=self.output_format,
                epilogue=self.config.epilogue,
                has_bias=self.config.has_bias,
                head_dim=self.config.head_dim,
                is_neox=self.config.is_neox,
                has_qkv_scale=self.config.has_qkv_scale,
                a=a_packed,
                weight=self.weight_packed,
                output=c,
                bias=self.bias,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                cos_sin=cos_sin,
                positions=positions,
                sfa=a_block_scale,
                sfb=self.weight_block_scale,
                sf_c=sf_c,
                scale=self.alpha,
                scale_c=self._scale_c_tensor,
                scale_gate=self._scale_gate_tensor,
                qkv_scale=self.qkv_scale,
            )
            if self.output_format == "nvfp4_e2m1":
                if self._return_quant_scale:
                    return c, sf_c, self.output_quant_scale
                return c, sf_c
            return c

        values = (
            a_packed,
            self.weight_packed,
            c,
            (m, self.n, self.k),
            self._max_active_clusters,
            self.bias,
            None,
            None,
            self.q_norm,
            self.k_norm,
            cos_sin,
            positions,
            a_block_scale,
            self.weight_block_scale,
            sf_c,
            self._scale_c_tensor,
            self._scale_gate_tensor,
            self.qkv_scale,
        )
        compiled = _compile(self.config, self._module, values)
        compiled(
            a_packed,
            self.weight_packed,
            c,
            (m, self.n, self.k),
            self.bias,
            self.alpha,
            None,
            None,
            self.q_norm,
            self.k_norm,
            cos_sin,
            positions,
            a_block_scale,
            self.weight_block_scale,
            sf_c,
            self._scale_c_tensor,
            self._scale_gate_tensor,
            self.qkv_scale,
        )
        if self.output_format == "nvfp4_e2m1":
            if self._return_quant_scale:
                return c, sf_c, self.output_quant_scale
            return c, sf_c
        return c


def prepare_fp4_linear(
    weight_packed,
    weight_block_scale,
    input_global_scale,
    weight_global_scale,
    bias=None,
    *,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    mma_k: int = 64,
    tile_k: int = 256,
    tmem_overlap: bool = False,
) -> PreparedFp4Linear:
    """Prepare a fixed packed-NVFP4 linear projection.

    On SM103, ``mma_k=96`` enables 3x NVFP4. ``tile_k=256`` uses separate
    256-element data and 384-element scale pipelines; ``tile_k=768`` selects
    the large-tile baseline. The default K=64 path remains available on all
    supported devices. ``tmem_overlap=True`` uses eight epilogue warps and
    alternating accumulator windows to overlap the next tile's MMA with
    output stores. It requires ``tile_k=256`` and BF16 output and is opt-in.
    """
    return PreparedFp4Linear(
        weight_packed,
        weight_block_scale,
        input_global_scale,
        weight_global_scale,
        epilogue="linear",
        out_dtype=out_dtype,
        output_quant_scale=output_quant_scale,
        bias=bias,
        mma_k=mma_k,
        tile_k=tile_k,
        tmem_overlap=tmem_overlap,
    )


def prepare_fp4_linear_swiglu(
    weight_packed,
    weight_block_scale,
    input_global_scale,
    weight_global_scale,
    bias=None,
    *,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    mma_k: int = 64,
    tile_k: int = 256,
) -> PreparedFp4Linear:
    """Prepare a fixed packed-NVFP4 linear+SwiGLU projection."""
    return PreparedFp4Linear(
        weight_packed,
        weight_block_scale,
        input_global_scale,
        weight_global_scale,
        epilogue="swiglu",
        out_dtype=out_dtype,
        output_quant_scale=output_quant_scale,
        bias=bias,
        mma_k=mma_k,
        tile_k=tile_k,
    )


def prepare_fp4_qkv_qknorm_rope(
    weight_packed,
    weight_block_scale,
    input_global_scale,
    weight_global_scale,
    q_norm_weight,
    k_norm_weight,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    is_neox=False,
    mma_k: int = 64,
    tile_k: int = 256,
    qkv_scale=None,
) -> PreparedFp4Linear:
    """Prepare a fixed packed-NVFP4 QKV+QKNorm+RoPE projection."""
    if num_q_heads != num_kv_heads:
        raise ValueError("the copied QKV specialization requires equal Q/KV heads")
    if weight_packed.shape[0] != 3 * num_q_heads * head_dim:
        raise ValueError("weight_packed must contain three equal Q/K/V head groups")
    return PreparedFp4Linear(
        weight_packed,
        weight_block_scale,
        input_global_scale,
        weight_global_scale,
        epilogue="qkv_qknorm_rope",
        q_norm=q_norm_weight,
        k_norm=k_norm_weight,
        qkv_scale=qkv_scale,
        head_dim=head_dim,
        is_neox=is_neox,
        mma_k=mma_k,
        tile_k=tile_k,
    )


def _fp8(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    epilogue: str,
    out_dtype: torch.dtype,
    output_quant_scale: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    q_norm: Optional[torch.Tensor] = None,
    k_norm: Optional[torch.Tensor] = None,
    cos_sin: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    is_neox: Optional[bool] = None,
    qkv_scale: Optional[torch.Tensor] = None,
    config: Optional[PrimsTsGemmConfig] = None,
    op_name: str = "prims_ts_fp8_linear",
):
    device = _validate_cuda_tensors(
        a,
        weight,
        a_scale,
        weight_scale,
        bias,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        out,
        output_quant_scale,
        qkv_scale,
    )
    if a.dtype != _FP8 or weight.dtype != _FP8 or a.ndim != 2 or weight.ndim != 2:
        raise ValueError("a and weight must be rank-2 torch.float8_e4m3fn tensors")
    m, k = a.shape
    n, wk = weight.shape
    if wk != k:
        raise ValueError(
            f"weight must have logical shape [N, {k}], got {tuple(weight.shape)}"
        )
    logical_n = n // 2 if epilogue == "swiglu" else n
    fmt = _output_format(out_dtype)
    cc = get_compute_capability(device)
    arch = cc[0] * 10 + cc[1]
    validate_dense_gemm(
        arch=arch,
        operand_format="fp8_e4m3",
        epilogue=epilogue,
        n=n,
        k=k,
        out_dtype=out_dtype,
        m=m,
        a_scale=a_scale,
        weight_scale=weight_scale,
        bias=bias,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_sin=cos_sin,
        positions=positions,
        head_dim=head_dim,
        is_neox=is_neox,
        qkv_scale=qkv_scale,
    )
    c, qscale, allocated_scale = _output(
        device, m, logical_n, fmt, out, output_quant_scale
    )
    identity = dict(
        arch=arch,
        operand_format="fp8_e4m3",
        output_format=fmt,
        epilogue=epilogue,
        has_bias=bias is not None,
        head_dim=head_dim,
        is_neox=is_neox,
        has_qkv_scale=qkv_scale is not None,
    )
    scale_gate = _scalar(device, 1.0) if qscale is not None else None
    if config is not None:
        if (
            config.arch != arch
            or config.operand_format != "fp8_e4m3"
            or config.output_format != fmt
            or config.epilogue != epilogue
            or config.has_bias != (bias is not None)
            or config.head_dim != head_dim
            or config.is_neox != is_neox
            or config.has_qkv_scale != (qkv_scale is not None)
        ):
            raise ValueError(
                "config must match the operand, output, and epilogue properties "
                "of this fp8 GEMM invocation"
            )
        _launch(
            config,
            a,
            weight,
            c,
            (m, n, k),
            bias=bias,
            x_scale=a_scale,
            weight_scale=weight_scale,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_sin=cos_sin,
            positions=positions,
            scale_c=qscale,
            scale_gate=scale_gate,
            qkv_scale=qkv_scale,
        )
    else:
        from .runner import launch_dense_gemm

        launch_dense_gemm(
            op_name,
            a=a,
            weight=weight,
            output=c,
            bias=bias,
            a_scale=a_scale,
            weight_scale=weight_scale,
            q_norm=q_norm,
            k_norm=k_norm,
            cos_sin=cos_sin,
            positions=positions,
            scale_c=qscale,
            scale_gate=scale_gate,
            qkv_scale=qkv_scale,
            **identity,
        )
    return (c, qscale) if allocated_scale else c


def _fp4(
    a_packed: torch.Tensor,
    a_block_scale: torch.Tensor,
    a_global_scale: float,
    weight_packed: torch.Tensor,
    weight_block_scale: torch.Tensor,
    weight_global_scale: float,
    bias: Optional[torch.Tensor],
    epilogue: str,
    out_dtype: torch.dtype,
    output_quant_scale: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    q_norm: Optional[torch.Tensor] = None,
    k_norm: Optional[torch.Tensor] = None,
    cos_sin: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    is_neox: Optional[bool] = None,
    qkv_scale: Optional[torch.Tensor] = None,
    op_name: str = "prims_ts_fp4_linear",
):
    device = _validate_cuda_tensors(
        a_packed,
        a_block_scale,
        weight_packed,
        weight_block_scale,
        bias,
        q_norm,
        k_norm,
        cos_sin,
        positions,
        out,
        output_quant_scale,
        qkv_scale,
    )
    if (
        a_packed.dtype != torch.uint8
        or weight_packed.dtype != torch.uint8
        or a_packed.ndim != 2
        or weight_packed.ndim != 2
    ):
        raise ValueError("packed NVFP4 operands must be rank-2 uint8 tensors")
    m, packed_k = a_packed.shape
    n, packed_weight_k = weight_packed.shape
    k = packed_k * 2
    if packed_weight_k != packed_k:
        raise ValueError("A and weight packed K dimensions must match")
    logical_n = n // 2 if epilogue == "swiglu" else n
    fmt = _output_format(out_dtype)
    cc = get_compute_capability(device)
    arch = cc[0] * 10 + cc[1]
    validate_dense_gemm(
        arch=arch,
        operand_format="nvfp4_e2m1",
        epilogue=epilogue,
        n=n,
        k=k,
        out_dtype=out_dtype,
        m=m,
        a_scale=a_block_scale,
        weight_scale=weight_block_scale,
        a_global_scale=a_global_scale,
        weight_global_scale=weight_global_scale,
        bias=bias,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_sin=cos_sin,
        positions=positions,
        head_dim=head_dim,
        is_neox=is_neox,
        qkv_scale=qkv_scale,
    )
    c, qscale, allocated_scale = _output(
        device, m, logical_n, fmt, out, output_quant_scale
    )
    sf_c = None
    if fmt == "nvfp4_e2m1":
        sf_c = torch.empty(
            (_nvfp4_128x4_numel(m, logical_n),),
            device=device,
            dtype=torch.uint8,
        )
    alpha = float(a_global_scale) * float(weight_global_scale)
    scale_c = _scaled_output_factor(qscale, alpha) if qscale is not None else None
    scale_gate = _scalar(device, alpha) if qscale is not None else None
    from .runner import launch_dense_gemm

    launch_dense_gemm(
        op_name,
        arch=arch,
        operand_format="nvfp4_e2m1",
        output_format=fmt,
        epilogue=epilogue,
        has_bias=bias is not None,
        head_dim=head_dim,
        is_neox=is_neox,
        has_qkv_scale=qkv_scale is not None,
        a=a_packed,
        weight=weight_packed,
        output=c,
        bias=bias,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_sin=cos_sin,
        positions=positions,
        sfa=a_block_scale,
        sfb=weight_block_scale,
        sf_c=sf_c,
        scale=alpha,
        scale_c=scale_c,
        scale_gate=scale_gate,
        qkv_scale=qkv_scale,
    )
    if fmt == "nvfp4_e2m1":
        result = (c, sf_c, qscale) if allocated_scale else (c, sf_c)
        return result
    return (c, qscale) if allocated_scale else c


@flashinfer_api
@backend_requirement({}, common_check=_check_gemm)
def fp8_linear(
    a,
    weight,
    a_scale,
    weight_scale,
    bias=None,
    *,
    qkv_scale=None,
    config=None,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    out=None,
):
    """FP8 [M,K] times [N,K] with per-token and per-channel FP32 scales."""
    return _fp8(
        a,
        weight,
        a_scale,
        weight_scale,
        bias,
        "linear",
        out_dtype,
        output_quant_scale,
        out,
        qkv_scale=qkv_scale,
        config=config,
        op_name="prims_ts_fp8_linear",
    )


@flashinfer_api
@backend_requirement({}, common_check=_check_gemm)
def fp8_linear_swiglu(
    a,
    weight,
    a_scale,
    weight_scale,
    bias=None,
    *,
    config=None,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    out=None,
):
    """FP8 fused SwiGLU. Weight, scale, and bias rows are adjacent gate/activation pairs."""
    return _fp8(
        a,
        weight,
        a_scale,
        weight_scale,
        bias,
        "swiglu",
        out_dtype,
        output_quant_scale,
        out,
        config=config,
        op_name="prims_ts_fp8_swiglu",
    )


@flashinfer_api
@backend_requirement({}, common_check=_check_gemm)
def fp8_qkv_qknorm_rope(
    a,
    qkv_weight,
    a_scale,
    qkv_weight_scale,
    q_norm_weight,
    k_norm_weight,
    cos_sin,
    positions,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    is_neox=False,
    qkv_scale=None,
    config=None,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    out=None,
):
    if num_q_heads != num_kv_heads:
        raise ValueError(
            "the copied QKV specialization requires num_q_heads == num_kv_heads"
        )
    if qkv_weight.shape[0] != 3 * num_q_heads * head_dim:
        raise ValueError("qkv_weight must be packed as three equal Q/K/V head groups")
    return _fp8(
        a,
        qkv_weight,
        a_scale,
        qkv_weight_scale,
        None,
        "qkv_qknorm_rope",
        out_dtype,
        output_quant_scale,
        out,
        q_norm_weight,
        k_norm_weight,
        cos_sin,
        positions,
        head_dim,
        is_neox,
        qkv_scale,
        config,
        op_name="prims_ts_fp8_qkv",
    )


@flashinfer_api
@backend_requirement({}, common_check=_check_gemm)
def fp4_linear(
    a_packed,
    a_block_scale,
    a_global_scale,
    weight_packed,
    weight_block_scale,
    weight_global_scale,
    bias=None,
    *,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    out=None,
):
    """Packed NVFP4 [M,K] times [N,K]. Block scales are contiguous 1D 128x4 E4M3 buffers."""
    return _fp4(
        a_packed,
        a_block_scale,
        a_global_scale,
        weight_packed,
        weight_block_scale,
        weight_global_scale,
        bias,
        "linear",
        out_dtype,
        output_quant_scale,
        out,
        op_name="prims_ts_fp4_linear",
    )


@flashinfer_api
@backend_requirement({}, common_check=_check_gemm)
def fp4_linear_swiglu(
    a_packed,
    a_block_scale,
    a_global_scale,
    weight_packed,
    weight_block_scale,
    weight_global_scale,
    bias=None,
    *,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    out=None,
):
    """Packed NVFP4 fused SwiGLU. Block scales are contiguous 1D 128x4 E4M3 buffers."""
    return _fp4(
        a_packed,
        a_block_scale,
        a_global_scale,
        weight_packed,
        weight_block_scale,
        weight_global_scale,
        bias,
        "swiglu",
        out_dtype,
        output_quant_scale,
        out,
        op_name="prims_ts_fp4_swiglu",
    )


@flashinfer_api
@backend_requirement({}, common_check=_check_gemm)
def fp4_qkv_qknorm_rope(
    a_packed,
    a_block_scale,
    a_global_scale,
    qkv_weight_packed,
    qkv_weight_block_scale,
    qkv_weight_global_scale,
    q_norm_weight,
    k_norm_weight,
    cos_sin,
    positions,
    *,
    num_q_heads,
    num_kv_heads,
    head_dim,
    is_neox=False,
    qkv_scale=None,
    out_dtype=torch.bfloat16,
    output_quant_scale=None,
    out=None,
):
    """Packed NVFP4 QKV with RMSNorm and RoPE. Block scales are contiguous 1D 128x4 E4M3 buffers."""
    if num_q_heads != num_kv_heads:
        raise ValueError(
            "the copied QKV specialization requires num_q_heads == num_kv_heads"
        )
    if qkv_weight_packed.shape[0] != 3 * num_q_heads * head_dim:
        raise ValueError(
            "qkv_weight_packed must be packed as three equal Q/K/V head groups"
        )
    return _fp4(
        a_packed,
        a_block_scale,
        a_global_scale,
        qkv_weight_packed,
        qkv_weight_block_scale,
        qkv_weight_global_scale,
        None,
        "qkv_qknorm_rope",
        out_dtype,
        output_quant_scale,
        out,
        q_norm_weight,
        k_norm_weight,
        cos_sin,
        positions,
        head_dim,
        is_neox,
        qkv_scale,
        op_name="prims_ts_fp4_qkv",
    )
