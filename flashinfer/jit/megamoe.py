"""JIT module generation for the three SM90 FP4 MegaMoE paths.

The kernel resolves its shared-memory layout, task schedule and dispatch
protocol at compile time, so one module is built per (shape, plan) pair. The
plan is chosen in ``flashinfer.megamoe.heuristics`` and rendered into
``megamoe_sm90_nvfp4_config.inc`` here.
"""

import os
from pathlib import Path

import jinja2

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, write_if_different


def _csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "megamoe"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "megamoe"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        f"MegaMoE CUDA sources not found (checked {installed}, {checkout})"
    )


def _gen_fp4_module(uri, template_name, filenames, render_args, use_fast_math):
    """Shared config rendering, source staging and CUDA build policy."""
    csrc = _csrc_dir()
    directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    directory.mkdir(parents=True, exist_ok=True)
    template = jinja2.Template((csrc / f"{template_name}.jinja").read_text())
    write_if_different(
        directory / f"{template_name}.inc", template.render(**render_args)
    )
    sources = []
    for filename in filenames:
        source = directory / filename
        write_if_different(source, (csrc / filename).read_text())
        sources.append(source)
    return gen_jit_spec(
        uri,
        sources,
        extra_cuda_cflags=[
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=--register-usage-level=5",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
        ],
        extra_cflags=["-std=c++20"],
        extra_include_paths=[directory, csrc, jit_env.FLASHINFER_CSRC_DIR],
        use_fast_math=use_fast_math,
    )


def get_megamoe_sm90_nvfp4_uri(
    plan,
    num_sms: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
    fast_math: bool,
    num_ranks: int = 8,
) -> str:
    return (
        f"megamoe_sm90_nvfp4_r{num_ranks}_sm{num_sms}_e{num_experts}_k{num_topk}"
        f"_h{hidden}_i{intermediate_hidden}_cap{num_max_tokens_per_rank}"
        f"_bm{plan.block_m}_bn{plan.block_n}_epw{plan.num_experts_per_wave}"
        f"_st{plan.num_stages}"
        f"_{'swap' if plan.swap_ab else 'noswap'}"
        f"{'_rs' if plan.rs_swap_ab else ''}"
        f"{'_il' if plan.use_interleaved_scheduler else '_static'}"
        f"{'_push' if plan.push_dispatch else ''}"
        f"{'_noclean' if plan.no_clean_barrier else ''}"
        f"{'_mode2' if plan.use_mode2_row_decoder else ''}"
        f"{'_single' if plan.single_active_dispatch_warp else ''}"
        f"_clamp{activation_clamp:g}"
        f"{'_fm' if fast_math else ''}"
    )


def gen_megamoe_sm90_nvfp4_module(
    plan,
    num_sms: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation_clamp: float,
    fast_math: bool,
    num_ranks: int = 8,
) -> JitSpec:
    uri = get_megamoe_sm90_nvfp4_uri(
        plan,
        num_sms,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
        fast_math,
        num_ranks=num_ranks,
    )
    # Fused compiler fast-math follows the source policy.
    return _gen_fp4_module(
        uri,
        "megamoe_sm90_nvfp4_config",
        ("megamoe_sm90_nvfp4_fused.cu",),
        dict(
            num_sms=num_sms,
            num_ranks=num_ranks,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts_per_wave=plan.num_experts_per_wave,
            block_m=plan.block_m,
            block_n=plan.block_n,
            num_max_pool_tokens=plan.num_max_pool_tokens,
            num_padded_sf_pool_tokens=plan.num_padded_sf_pool_tokens,
            num_stages=plan.num_stages,
            smem_size=plan.smem_size,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
            swap_ab=plan.swap_ab,
            rs_swap_ab=plan.rs_swap_ab,
            single_active_dispatch_warp=plan.single_active_dispatch_warp,
            use_mode2_row_decoder=plan.use_mode2_row_decoder,
            use_interleaved_scheduler=plan.use_interleaved_scheduler,
            push_dispatch=plan.push_dispatch,
            no_clean_barrier=plan.no_clean_barrier,
            spin_debug=os.environ.get("FLASHINFER_MEGAMOE_SPIN_DEBUG", "0") == "1",
        ),
        fast_math,
    )


# ---------------------------------------------------------------------------
# Split (BN128) family
# ---------------------------------------------------------------------------


def _split_shape_uri(
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
) -> str:
    return (
        f"sm{num_sms}_r{num_ranks}_e{num_experts}_k{num_topk}"
        f"_h{hidden}_i{intermediate_hidden}_cap{num_max_tokens_per_rank}"
    )


def gen_megamoe_sm90_nvfp4_split_plan_module(
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
) -> JitSpec:
    """Template-free module that reports the L1/L2 pipeline depth and shared
    memory budget. Those are kernel template parameters, so they must be known
    before the (slow) kernel module is built."""
    uri = "megamoe_sm90_nvfp4_split_plan_" + _split_shape_uri(
        num_sms,
        num_ranks,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    csrc = _csrc_dir()
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    os.makedirs(gen_directory, exist_ok=True)
    dest = gen_directory / "megamoe_sm90_nvfp4_split_plan.cu"
    for filename in (
        "megamoe_sm90_nvfp4_split_plan.cu",
        "megamoe_sm90_nvfp4_split_plan.cuh",
    ):
        with open(csrc / filename) as f:
            write_if_different(gen_directory / filename, f.read())
    return gen_jit_spec(
        uri,
        [dest],
        extra_cuda_cflags=["-std=c++20", "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++20"],
        extra_include_paths=[gen_directory, csrc, jit_env.FLASHINFER_CSRC_DIR],
    )


def get_megamoe_sm90_nvfp4_split_uri(
    plan,
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
    activation_clamp,
    fast_math,
) -> str:
    return (
        "megamoe_sm90_nvfp4_split_"
        + _split_shape_uri(
            num_sms,
            num_ranks,
            num_max_tokens_per_rank,
            num_experts,
            num_topk,
            hidden,
            intermediate_hidden,
        )
        + f"_epw{plan.num_experts_per_wave}"
        + f"_l1st{plan.l1_num_stages}_l2st{plan.l2_num_stages}"
        + ("_dq" if plan.dispatch_dequant else "")
        + ("_l2ac" if plan.l2_arrival_counter else "")
        + f"_clamp{activation_clamp:g}"
        + ("_fm" if fast_math else "")
    )


def gen_megamoe_sm90_nvfp4_split_module(
    plan,
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
    activation_clamp,
    fast_math,
) -> JitSpec:
    uri = get_megamoe_sm90_nvfp4_split_uri(
        plan,
        num_sms,
        num_ranks,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
        fast_math,
    )
    # Split kFastMath selects source intrinsics; retain denormals in this family.
    return _gen_fp4_module(
        uri,
        "megamoe_sm90_nvfp4_split_config",
        ("megamoe_sm90_nvfp4_split.cu",),
        dict(
            num_sms=num_sms,
            num_ranks=num_ranks,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_experts_per_wave=plan.num_experts_per_wave,
            num_max_pool_tokens=plan.num_max_pool_tokens,
            num_padded_sf_pool_tokens=plan.num_padded_sf_pool_tokens,
            l1_num_stages=plan.l1_num_stages,
            l1_smem_size=plan.l1_smem_size,
            l2_num_stages=plan.l2_num_stages,
            l2_smem_size=plan.l2_smem_size,
            dispatch_dequant=plan.dispatch_dequant,
            l2_arrival_counter=plan.l2_arrival_counter,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        ),
        False,
    )


def gen_megamoe_sm90_layout_module() -> JitSpec:
    csrc = _csrc_dir()
    return gen_jit_spec(
        "megamoe_sm90_layout",
        [csrc / "megamoe_sm90_layout.cu"],
        extra_include_paths=[csrc, jit_env.FLASHINFER_CSRC_DIR],
        extra_cuda_cflags=["-std=c++20", "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++20"],
    )


def gen_megamoe_sm90_mxfp4_module(
    plan,
    num_sms,
    num_ranks,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    hidden,
    intermediate_hidden,
    activation_clamp,
    fast_math,
) -> JitSpec:
    # Include every compile-time policy in the key, including rank width.
    uri = get_megamoe_sm90_nvfp4_uri(
        plan,
        num_sms,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        hidden,
        intermediate_hidden,
        activation_clamp,
        fast_math,
        num_ranks=num_ranks,
    ).replace("nvfp4", "mxfp4")
    return _gen_fp4_module(
        uri,
        "megamoe_sm90_mxfp4_config",
        ("megamoe_sm90_mxfp4_fused.cu",),
        dict(
            plan=plan,
            num_sms=num_sms,
            num_ranks=num_ranks,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_experts=num_experts,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        ),
        fast_math,
    )
