"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

AOT build script for FlashInfer.

NOTE (Zihao): The following modules are intentionally excluded from the AOT build:
- gen_pod_module
- gen_deepgemm_sm100_module (it doesn't involve host-side compilation)
"""

import argparse
import os
import shutil
from itertools import product
from pathlib import Path
from typing import List, Tuple, Iterator
import importlib

import torch
import torch.version

from .activation import act_func_def_str, gen_act_and_mul_module
from .fp8_quantization import gen_mxfp8_quantization_sm100_module
from .cascade import gen_cascade_module
from .fp4_quantization import (
    gen_fp4_quantization_sm90_module,
    gen_fp4_quantization_sm100_module,
    gen_fp4_quantization_sm103_module,
    gen_fp4_quantization_sm110_module,
    gen_fp4_quantization_sm120_module,
    gen_fp4_quantization_sm121_module,
)
from .fused_moe import (
    gen_cutlass_fused_moe_sm100_module,
    gen_cutlass_fused_moe_sm90_module,
    gen_trtllm_gen_fused_moe_sm100_module,
)
from .gemm import (
    gen_gemm_module,
    gen_gemm_sm90_module,
    gen_gemm_sm100_module,
    gen_gemm_sm100_module_cutlass_fp4,
    gen_gemm_sm100_module_cutlass_fp8,
    gen_gemm_sm100_module_tgv,
    gen_gemm_sm120_module,
    gen_gemm_sm120_module_cutlass_fp4,
    gen_trtllm_gen_gemm_module,
)
from .jit import JitSpec, build_jit_specs
from .jit import env as jit_env
from .jit import (
    gen_batch_attention_module,
    gen_batch_decode_module,
    gen_batch_mla_module,
    gen_batch_prefill_module,
    gen_cudnn_fmha_module,
    gen_fmha_cutlass_sm100a_module,
    gen_single_decode_module,
    gen_single_prefill_module,
    gen_trtllm_gen_fmha_module,
)
from .mla import gen_mla_module
from .norm import gen_norm_module
from .page import gen_page_module
from .quantization import gen_quantization_module
from .rope import gen_rope_module
from .sampling import gen_sampling_module
from .tllm_utils import gen_trtllm_utils_module
from .utils import gen_logging_module, version_at_least
from .xqa import gen_xqa_module
from .compilation_context import CompilationContext


def gen_fa2(
    dtype_qo: torch.dtype,
    dtype_kv: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> Iterator[JitSpec]:
    if dtype_qo.itemsize == dtype_kv.itemsize and dtype_qo != dtype_kv:
        return
    if dtype_qo.itemsize == 1:
        return  # fp8 tensor cores not supported in fa2

    yield gen_single_prefill_module(
        backend="fa2",
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=False,
    )

    yield gen_batch_prefill_module(
        backend="fa2",
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        dtype_idx=torch.int32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=False,
    )

    yield gen_single_decode_module(
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )

    yield gen_batch_decode_module(
        dtype_q=dtype_qo,
        dtype_kv=dtype_kv,
        dtype_o=dtype_qo,
        dtype_idx=torch.int32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )


def gen_fa3(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
) -> Iterator[JitSpec]:
    if dtype_q != dtype_kv:
        return  # fa3 template do not support mixed precision
    if dtype_q.itemsize == 2:
        if dtype_q != dtype_o:
            return  # for fp16, dtype_o must be the same as dtype_q/dtype_kv

    if dtype_kv.itemsize == 1:
        if head_dim_qk == 192 or head_dim_qk == 64:
            return  # (192, 128) & (64, 64) not supported for fp8 yet.

    yield gen_batch_prefill_module(
        backend="fa3",
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        dtype_o=dtype_o,
        dtype_idx=torch.int32,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        pos_encoding_mode=0,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
        use_fp16_qk_reduction=False,
    )


def gen_attention(
    f16_dtype_: List[torch.dtype],
    f8_dtype_: List[torch.dtype],
    fa2_head_dim_: List[Tuple[int, int]],
    fa3_head_dim_: List[Tuple[int, int]],
    use_sliding_window_: List[bool],
    use_logits_soft_cap_: List[bool],
    has_sm90: bool,
    has_sm100: bool,
    add_gemma: bool,
    add_oai_oss: bool,
) -> Iterator[JitSpec]:
    head_dim_ckv = 512
    head_dim_kpe = 64

    # FA2 MHA / MQA / GQA
    for (
        (head_dim_qk, head_dim_vo),
        dtype_qo,
        dtype_kv,
        use_sliding_window,
        use_logits_soft_cap,
    ) in product(
        fa2_head_dim_,
        f16_dtype_,
        f16_dtype_ + f8_dtype_,
        use_sliding_window_,
        use_logits_soft_cap_,
    ):
        yield from gen_fa2(
            dtype_qo=dtype_qo,
            dtype_kv=dtype_kv,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
        )
        yield gen_batch_attention_module(
            dtype_q=dtype_qo,
            dtype_kv=dtype_kv,
            dtype_o=dtype_qo,
            dtype_idx=torch.int32,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            pos_encoding_mode=0,
            # use_sliding_window=use_sliding_window,
            use_logits_soft_cap=use_logits_soft_cap,
            use_profiler=False,
        )

    # FA3 MHA / MQA / GQA
    if has_sm90:
        for (
            (head_dim_qk, head_dim_vo),
            dtype_qkv,
            dtype_o,
            use_sliding_window,
            use_logits_soft_cap,
        ) in product(
            fa3_head_dim_,
            f16_dtype_ + f8_dtype_,
            f16_dtype_,
            use_sliding_window_,
            use_logits_soft_cap_,
        ):
            yield from gen_fa3(
                dtype_q=dtype_qkv,
                dtype_kv=dtype_qkv,
                dtype_o=dtype_o,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                use_sliding_window=use_sliding_window,
                use_logits_soft_cap=use_logits_soft_cap,
            )

    # Gemma
    if add_gemma:
        for (
            dtype_qo,
            dtype_kv,
            (use_sliding_window, use_logits_soft_cap),
        ) in product(
            f16_dtype_,
            f16_dtype_ + f8_dtype_,
            [(True, True)],
        ):
            yield from gen_fa2(
                dtype_qo=dtype_qo,
                dtype_kv=dtype_kv,
                head_dim_qk=256,
                head_dim_vo=256,
                use_sliding_window=use_sliding_window,
                use_logits_soft_cap=use_logits_soft_cap,
            )
        if has_sm90:
            for (
                dtype_qkv,
                dtype_o,
                (use_sliding_window, use_logits_soft_cap),
            ) in product(
                f16_dtype_ + f8_dtype_,
                f16_dtype_,
                [(True, True)],
            ):
                yield from gen_fa3(
                    dtype_q=dtype_qkv,
                    dtype_kv=dtype_qkv,
                    dtype_o=dtype_o,
                    head_dim_qk=256,
                    head_dim_vo=256,
                    use_sliding_window=use_sliding_window,
                    use_logits_soft_cap=use_logits_soft_cap,
                )

    # OAI OSS
    if add_oai_oss:
        from .jit.attention import gen_batch_prefill_attention_sink_module

        for dtype in f16_dtype_:
            for backend in ["fa2", "fa3"]:
                for use_swa in [True, False]:
                    yield gen_batch_prefill_attention_sink_module(
                        backend=backend,
                        dtype_q=dtype,
                        dtype_kv=dtype,
                        dtype_o=dtype,
                        dtype_idx=torch.int32,
                        head_dim_qk=64,
                        head_dim_vo=64,
                        pos_encoding_mode=0,
                        use_sliding_window=use_swa,
                    )

    # fmha_cutlass_sm100a
    # NOTE: currently there's only one uri.
    if has_sm100:
        yield gen_fmha_cutlass_sm100a_module(
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            dtype_o=torch.bfloat16,
            dtype_idx=torch.int32,
            head_dim_qk=128,
            head_dim_vo=128,
            pos_encoding_mode=0,
            use_sliding_window=False,
            use_logits_soft_cap=False,
        )

        # trtllm_gen_fmha
        yield gen_trtllm_gen_fmha_module()

    # MLA
    # NOTE: fp8 kv not supported in MLA
    mla_backend_ = ["fa2"] + (["fa3"] if has_sm90 else [])
    for dtype_qo in f16_dtype_:
        for backend in mla_backend_:
            yield gen_batch_mla_module(
                backend=backend,
                dtype_q=dtype_qo,
                dtype_kv=dtype_qo,
                dtype_o=dtype_qo,
                dtype_idx=torch.int32,
                head_dim_ckv=head_dim_ckv,
                head_dim_kpe=head_dim_kpe,
                use_profiler=False,
            )

    # MLA SM100
    if has_sm100:
        yield gen_mla_module()


def gen_xqa(
    use_fp16_: List[bool],
    token_per_page_: List[int],
    head_size_: List[int],
    head_grp_size_: List[int],
    use_sliding_window_: List[bool],
    has_sm90: bool,
) -> Iterator[JitSpec]:
    """Generate XQA modules for various configurations."""
    if not has_sm90:
        return  # XQA requires SM90+

    for (
        use_fp16,
        token_per_page,
        head_size,
        head_grp_size,
        use_sliding_window,
    ) in product(
        use_fp16_,
        token_per_page_,
        head_size_,
        head_grp_size_,
        use_sliding_window_,
    ):
        # Skip invalid configurations
        if head_size % 16 != 0 or head_size > 256 or head_size < 16:
            continue
        if token_per_page not in [16, 32, 64, 128]:
            continue

        yield gen_xqa_module(
            use_fp16=use_fp16,
            token_per_page=token_per_page,
            head_size=head_size,
            head_grp_size=head_grp_size,
            use_sliding_window=use_sliding_window,
        )


def gen_all_modules(
    f16_dtype_: List[torch.dtype],
    f8_dtype_: List[torch.dtype],
    fa2_head_dim_: List[Tuple[int, int]],
    fa3_head_dim_: List[Tuple[int, int]],
    use_sliding_window_: List[bool],
    use_logits_soft_cap_: List[bool],
    sm_capabilities: dict,
    add_comm: bool,
    add_gemma: bool,
    add_oai_oss: bool,
    add_moe: bool,
    add_act: bool,
    add_misc: bool,
    add_xqa: bool,
) -> List[JitSpec]:
    jit_specs: List[JitSpec] = []
    has_sm90 = sm_capabilities.get("sm90", False)
    has_sm100 = sm_capabilities.get("sm100", False)
    has_sm103 = sm_capabilities.get("sm103", False)
    has_sm110 = sm_capabilities.get("sm110", False)
    has_sm120 = sm_capabilities.get("sm120", False)
    has_sm121 = sm_capabilities.get("sm121", False)

    jit_specs += list(
        gen_attention(
            f16_dtype_,
            f8_dtype_,
            fa2_head_dim_,
            fa3_head_dim_,
            use_sliding_window_,
            use_logits_soft_cap_,
            has_sm90,
            has_sm100,
            add_gemma,
            add_oai_oss,
        )
    )

    if add_act:
        for act_name in act_func_def_str:
            jit_specs.append(gen_act_and_mul_module(act_name))

    if add_moe:
        jit_specs.append(gen_gemm_module())
        if has_sm90:
            jit_specs.append(gen_gemm_sm90_module())
            jit_specs.append(gen_fp4_quantization_sm90_module())
            jit_specs.append(gen_cutlass_fused_moe_sm90_module())
        if has_sm100:
            jit_specs.append(gen_fp4_quantization_sm100_module())
            jit_specs.append(gen_cutlass_fused_moe_sm100_module())
            jit_specs.append(gen_gemm_sm100_module())
            jit_specs.append(gen_gemm_sm100_module_cutlass_fp4())
            jit_specs.append(gen_gemm_sm100_module_cutlass_fp8())
            # Add TGV GEMM modules for both bf16 and fp16
            jit_specs.append(gen_gemm_sm100_module_tgv(torch.bfloat16))
            jit_specs.append(gen_gemm_sm100_module_tgv(torch.float16))
            jit_specs.append(gen_mxfp8_quantization_sm100_module())
            jit_specs.append(gen_trtllm_gen_gemm_module())
            jit_specs.append(gen_trtllm_gen_fused_moe_sm100_module())
        if has_sm103:
            jit_specs.append(gen_fp4_quantization_sm103_module())
        if has_sm110:
            jit_specs.append(gen_fp4_quantization_sm110_module())
        if has_sm120:
            jit_specs.append(gen_fp4_quantization_sm120_module())
            jit_specs.append(gen_gemm_sm120_module())
            jit_specs.append(gen_gemm_sm120_module_cutlass_fp4())
        if has_sm121:
            jit_specs.append(gen_fp4_quantization_sm121_module())

    if add_comm:
        from .comm import gen_trtllm_comm_module, gen_vllm_comm_module
        from .comm.nvshmem import gen_nvshmem_module
        from .comm.trtllm_alltoall import gen_comm_alltoall_module
        from .comm.trtllm_mnnvl_ar import gen_trtllm_mnnvl_comm_module

        jit_specs.append(gen_nvshmem_module())
        jit_specs.append(gen_comm_alltoall_module())
        if has_sm100:
            jit_specs.append(gen_trtllm_comm_module())
            jit_specs.append(gen_trtllm_mnnvl_comm_module())
        jit_specs.append(gen_vllm_comm_module())

    if add_misc:
        jit_specs += [
            gen_cascade_module(),
            gen_norm_module(),
            gen_page_module(),
            gen_quantization_module(),
            gen_rope_module(),
            gen_sampling_module(),
        ]
        if has_sm90:
            jit_specs.append(gen_trtllm_utils_module())

    if add_xqa:
        # Define XQA configurations to iterate over
        xqa_use_fp16_ = [True, False]  # fp16 and bf16
        xqa_token_per_page_ = [16, 32, 64, 128]
        xqa_head_size_ = [64, 128, 256]
        xqa_head_grp_size_ = [1, 2, 4, 8]  # Different group sizes for MQA/GQA

        jit_specs += list(
            gen_xqa(
                xqa_use_fp16_,
                xqa_token_per_page_,
                xqa_head_size_,
                xqa_head_grp_size_,
                use_sliding_window_,
                has_sm90,
            )
        )

    # Add cuDNN FMHA module
    jit_specs.append(gen_cudnn_fmha_module())

    # dedup
    names = set()
    ret: List[JitSpec] = []
    for jit_spec in jit_specs:
        if jit_spec.name not in names:
            names.add(jit_spec.name)
            ret.append(jit_spec)
    return ret


def copy_built_kernels(
    jit_specs: List[JitSpec],
    out_dir: Path,
) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    for jit_spec in jit_specs:
        src = jit_env.FLASHINFER_JIT_DIR / jit_spec.name / f"{jit_spec.name}.so"
        dst = out_dir / jit_spec.name / f"{jit_spec.name}.so"
        dst.parent.mkdir(exist_ok=False, parents=False)
        shutil.copy2(src, dst)


def parse_bool(s: str) -> bool:
    if s.lower() in ("true", "1"):
        return True
    elif s.lower() in ("false", "0"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}")


def parse_head_dim(head_dim: str) -> Tuple[int, int]:
    qo, kv = map(int, head_dim.split(","))
    return qo, kv


def get_default_config():
    """Get default AOT configuration"""
    return {
        "fa2_head_dim": [(64, 64), (128, 128), (256, 256)],
        "fa3_head_dim": [(192, 128), (128, 128), (64, 64), (256, 256)],
        "f16_dtype": [torch.float16, torch.bfloat16],
        "f8_dtype": [torch.float8_e4m3fn],
        "use_sliding_window": [False, True],
        "use_logits_soft_cap": [False, True],
        "add_comm": True,
        "add_gemma": True,
        "add_oai_oss": True,
        "add_moe": True,
        "add_act": True,
        "add_misc": True,
        "add_xqa": True,
    }


def detect_sm_capabilities():
    """Detect SM capabilities"""
    import torch.version

    compilation_context = CompilationContext()
    gencode_flags_list = compilation_context.get_nvcc_flags_list(
        supported_major_versions=None
    )

    def has_sm(compute: str, version: str) -> bool:
        if not any(compute in flag for flag in gencode_flags_list):
            return False
        if torch.version.cuda is None:
            return True
        return version_at_least(torch.version.cuda, version)

    return {
        "sm90": has_sm("compute_90", "12.3"),
        "sm100": has_sm("compute_100", "12.8"),
        "sm103": has_sm("compute_103", "12.8"),
        "sm110": has_sm("compute_110", "12.9"),
        "sm120": has_sm("compute_120", "13.0"),
        "sm121": has_sm("compute_121", "13.0"),
    }


def register_default_modules() -> int:
    """Register the default set of modules"""
    config = get_default_config()
    sm_capabilities = detect_sm_capabilities()

    jit_specs = gen_all_modules(
        config["f16_dtype"],
        config["f8_dtype"],
        config["fa2_head_dim"],
        config["fa3_head_dim"],
        config["use_sliding_window"],
        config["use_logits_soft_cap"],
        sm_capabilities,
        config["add_comm"],
        config["add_gemma"],
        config["add_oai_oss"],
        config["add_moe"],
        config["add_act"],
        config["add_misc"],
        config["add_xqa"],
    )
    return len(jit_specs)


def main():
    parser = argparse.ArgumentParser(
        description="Ahead-of-Time (AOT) build all modules"
    )
    parser.add_argument("--out-dir", type=Path, help="Output directory")
    parser.add_argument("--build-dir", type=Path, help="Build directory")
    parser.add_argument(
        "--fa2-head-dim",
        nargs="*",
        help="FA2 head dim pair of qk and vo, separated by comma",
    )
    parser.add_argument(
        "--fa3-head-dim",
        nargs="*",
        help="FA3 head dim pair of qk and vo, separated by comma",
    )
    parser.add_argument(
        "--f16-dtype",
        nargs="*",
        choices=["float16", "bfloat16"],
        help="16-bit data type",
    )
    parser.add_argument(
        "--f8-dtype",
        nargs="*",
        choices=["float8_e4m3fn", "float8_e5m2"],
        help="8-bit data type",
    )
    parser.add_argument(
        "--use-sliding-window", nargs="*", help="Use sliding window attention"
    )
    parser.add_argument("--use-logits-soft-cap", nargs="*", help="Use logits soft cap")
    parser.add_argument(
        "--add-comm",
        type=parse_bool,
        help="Add communication kernels (trtllm_comm, vllm_comm)",
    )
    parser.add_argument(
        "--add-gemma",
        type=parse_bool,
        help="Add kernels for Gemma Model (head_dim=256, use_sliding_window, use_logits_soft_cap)",
    )
    parser.add_argument(
        "--add-oai-oss",
        type=parse_bool,
        help="Add kernels for OAI OSS Model (head_dim=64, use_sliding_window)",
    )
    parser.add_argument("--add-moe", type=parse_bool, help="Add MoE kernels")
    parser.add_argument("--add-act", type=parse_bool, help="Add activation kernels")
    parser.add_argument("--add-misc", type=parse_bool, help="Add miscellaneous kernels")
    parser.add_argument(
        "--add-xqa", type=parse_bool, help="Add XQA (Cross-Query Attention) kernels"
    )
    args = parser.parse_args()

    # Start with default configuration
    project_root = Path(__file__).resolve().parents[1]
    config = get_default_config()
    out_dir = (
        project_root / "aot-ops"
        if importlib.util.find_spec("flashinfer") is None
        else jit_env.FLASHINFER_WORKSPACE_DIR / "cached_ops"
    )
    build_dir = project_root / "build" / "aot"

    # Override with command line arguments
    if args.out_dir:
        out_dir = Path(args.out_dir)
    if args.build_dir:
        build_dir = Path(args.build_dir)
    if args.fa2_head_dim:
        config["fa2_head_dim"] = [parse_head_dim(dim) for dim in args.fa2_head_dim]
    if args.fa3_head_dim:
        config["fa3_head_dim"] = [parse_head_dim(dim) for dim in args.fa3_head_dim]
    if args.f16_dtype:
        config["f16_dtype"] = [getattr(torch, dtype) for dtype in args.f16_dtype]
    if args.f8_dtype:
        config["f8_dtype"] = [getattr(torch, dtype) for dtype in args.f8_dtype]
    if args.use_sliding_window:
        config["use_sliding_window"] = [parse_bool(s) for s in args.use_sliding_window]
    if args.use_logits_soft_cap:
        config["use_logits_soft_cap"] = [
            parse_bool(s) for s in args.use_logits_soft_cap
        ]

    for key in [
        "add_comm",
        "add_gemma",
        "add_oai_oss",
        "add_moe",
        "add_act",
        "add_misc",
        "add_xqa",
    ]:
        arg_value = getattr(args, key, None)
        if arg_value is not None:
            config[key] = arg_value

    # Cuda Arch
    if "FLASHINFER_CUDA_ARCH_LIST" not in os.environ:
        raise RuntimeError("Please explicitly set env var FLASHINFER_CUDA_ARCH_LIST.")

    sm_capabilities = detect_sm_capabilities()

    # Update data dir
    # if flashinfer installed, do not change these paths, otherwise change to the local paths

    if importlib.util.find_spec("flashinfer") is None:
        # NOTE(Zihao): this script can run before or after flashinfer is installed, if flashinfer is not installed, use the local paths
        jit_env.FLASHINFER_CSRC_DIR = project_root / "csrc"
        jit_env.FLASHINFER_INCLUDE_DIR = project_root / "include"
        jit_env.CUTLASS_INCLUDE_DIRS = [
            project_root / "3rdparty" / "cutlass" / "include",
            project_root / "3rdparty" / "cutlass" / "tools" / "util" / "include",
        ]
        jit_env.SPDLOG_INCLUDE_DIR = project_root / "3rdparty" / "spdlog" / "include"

    # Update workdir
    jit_env.FLASHINFER_WORKSPACE_DIR = build_dir
    jit_env.FLASHINFER_JIT_DIR = build_dir / "cached_ops"
    jit_env.FLASHINFER_GEN_SRC_DIR = build_dir / "generated"
    jit_env.FLASHINFER_JIT_DIR.mkdir(parents=True, exist_ok=True)
    jit_env.FLASHINFER_GEN_SRC_DIR.mkdir(parents=True, exist_ok=True)

    # Print summary
    print("AOT build summary:")
    print("  out_dir:", out_dir)
    print("  build_dir:", build_dir)
    print("  fa2_head_dim:", config["fa2_head_dim"])
    print("  fa3_head_dim:", config["fa3_head_dim"])
    print("  f16_dtype:", config["f16_dtype"])
    print("  f8_dtype:", config["f8_dtype"])
    print("  use_sliding_window:", config["use_sliding_window"])
    print("  use_logits_soft_cap:", config["use_logits_soft_cap"])
    print("  FLASHINFER_CUDA_ARCH_LIST:", os.environ["FLASHINFER_CUDA_ARCH_LIST"])
    print("  SM capabilities detected:")
    for sm_name, has_sm in sm_capabilities.items():
        if has_sm:
            print(f"    {sm_name}: True")
    for key in [
        "add_comm",
        "add_gemma",
        "add_oai_oss",
        "add_moe",
        "add_act",
        "add_misc",
        "add_xqa",
    ]:
        print(f"  {key}:", config[key])

    # Generate JIT specs
    print("Generating JIT specs...")
    jit_specs = [gen_logging_module()]
    jit_specs += gen_all_modules(
        config["f16_dtype"],
        config["f8_dtype"],
        config["fa2_head_dim"],
        config["fa3_head_dim"],
        config["use_sliding_window"],
        config["use_logits_soft_cap"],
        sm_capabilities,
        config["add_comm"],
        config["add_gemma"],
        config["add_oai_oss"],
        config["add_moe"],
        config["add_act"],
        config["add_misc"],
        config["add_xqa"],
    )
    print("Total ops:", len(jit_specs))

    # Build
    build_jit_specs(jit_specs, verbose=True, skip_prebuilt=False)

    # Copy built kernels
    copy_built_kernels(jit_specs, out_dir)
    print("AOT kernels saved to:", out_dir)


if __name__ == "__main__":
    main()
