"""JIT module generation for the SM90 FP4 MegaMoE paths and the SM120 MXFP4 x MXFP8 nosync MegaMoE
(the ``flashinfer.moe_ep`` backend ``sm120_mxfp4_mxfp8_bf16_nosync``).

The kernel resolves its shared-memory layout, task schedule and dispatch
protocol at compile time, so one module is built per (shape, plan) pair. The
plan is chosen in ``flashinfer.megamoe.heuristics`` and rendered into
``megamoe_sm90_nvfp4_config.inc`` here.
"""

import os
from pathlib import Path

import jinja2

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags, write_if_different


def _csrc_dir() -> Path:
    """Directory holding the MegaMoE CUDA sources, installed or in a source checkout."""
    installed = jit_env.FLASHINFER_CSRC_DIR / "megamoe"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "megamoe"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        f"MegaMoE CUDA sources not found (checked {installed}, {checkout})"
    )


def _gen_fp4_module(
    uri,
    template_name,
    filenames,
    render_args,
    use_fast_math,
    extra_include_paths=(),
    extra_cuda_cflags=(),
):
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
            *extra_cuda_cflags,
        ],
        extra_cflags=["-std=c++20"],
        extra_include_paths=[
            directory,
            csrc,
            jit_env.FLASHINFER_CSRC_DIR,
            *extra_include_paths,
        ],
        use_fast_math=use_fast_math,
    )


# NCCL GIN on the host-proxy backend; the other backends are compiled out, as
# the SM120 MegaMoE does, so the persistent kernel does not carry their code.
_GIN_PROXY_CFLAGS = (
    "-DFI_MEGAMOE_WITH_NCCL_GIN=1",
    "-DNCCL_GIN_GDAKI_ENABLE=0",
    "-DNCCL_GIN_GPI_ENABLE=0",
    "-D_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE",
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
    """Module key of the NVFP4 fused kernel: shape, rank width and every compile-time plan policy."""
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
    """Render the fused config for ``plan`` and return the JIT spec of the NVFP4 fused kernel module."""
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
    """Shape part shared by the split-family module keys."""
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
    """Module key of the NVFP4 split kernel: shape, pipeline depths and dispatch policies of ``plan``."""
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
    """Render the split config for ``plan`` and return the JIT spec of the NVFP4 split kernel module.

    Compiler fast-math stays off here; ``fast_math`` only selects source intrinsics.
    """
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
    """JIT spec of the helper that computes the symmetric buffer layout and byte offsets."""
    csrc = _csrc_dir()
    return gen_jit_spec(
        "megamoe_sm90_layout",
        [csrc / "megamoe_sm90_layout.cu"],
        extra_include_paths=[csrc, jit_env.FLASHINFER_CSRC_DIR],
        extra_cuda_cflags=["-std=c++20", "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++20"],
    )


def _nccl_paths():
    """NCCL headers and the versioned library, from ``NCCL_HOME`` or the ``nvidia-nccl`` wheel."""
    root = os.environ.get("NCCL_HOME")
    home = Path(root).expanduser() if root else None
    if home is None:
        try:
            import nvidia.nccl  # type: ignore[import-not-found]

            home = Path(nvidia.nccl.__path__[0])
        except (ImportError, AttributeError, IndexError):
            home = None
    if home is None:
        raise FileNotFoundError(
            "NCCL development files not found; set NCCL_HOME or install the nvidia-nccl wheel"
        )
    include = home / "include"
    lib_dir = next(
        (
            path
            for path in (home / "lib", home / "lib64", home)
            if any(path.glob("libnccl.so*"))
        ),
        home / "lib",
    )
    libraries = sorted(lib_dir.glob("libnccl.so.*")) or sorted(
        lib_dir.glob("libnccl.so")
    )
    if not (include / "nccl_device.h").is_file() or not libraries:
        raise FileNotFoundError(
            f"NCCL device API headers or libnccl not found under {home}"
        )
    return include, libraries[-1]


def gen_megamoe_sm90_transport_module() -> JitSpec:
    """JIT spec of the NCCL-window transport: symmetric windows, their load/store
    peer pointers and the device communicator that carries inter-node puts."""
    csrc = _csrc_dir()
    include, library = _nccl_paths()
    return gen_jit_spec(
        "megamoe_sm90_transport_v2",
        [csrc / "megamoe_sm90_transport.cu"],
        extra_include_paths=[csrc, jit_env.FLASHINFER_CSRC_DIR, include],
        extra_cuda_cflags=["-std=c++20", "--expt-relaxed-constexpr"],
        extra_cflags=["-std=c++20"],
        extra_ldflags=[str(library), "-ldl"],
    )


def get_megamoe_sm90_mxfp4_uri(
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
    in_kernel_weight_copy: bool = False,
    num_nodes: int = 1,
    weight_tile_bytes: int = 8704,
) -> str:
    """Module key of the MXFP4 fused kernel: every compile-time policy, the rank
    width, weight-record format, replica-copy mode and node count."""
    if weight_tile_bytes not in (8704, 8448, 16384):
        raise ValueError("unsupported MXFP4 weight-record size")
    if weight_tile_bytes == 8448 and not plan.rs_swap_ab:
        raise ValueError("compact MXFP4 weights require register-source math")
    if weight_tile_bytes == 16384 and plan.rs_swap_ab:
        raise ValueError("predecoded FP8 weights require shared-source math")
    if in_kernel_weight_copy and (num_nodes > 1 or weight_tile_bytes != 8704):
        raise ValueError("expert slots require single-node canonical MXFP4 storage")
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
    uri += f"_wt{weight_tile_bytes}"
    uri += "_eplbcopy" if in_kernel_weight_copy else ""
    uri += f"_nodes{num_nodes}" if num_nodes > 1 else ""
    return uri


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
    in_kernel_weight_copy: bool = False,
    num_nodes: int = 1,
    weight_tile_bytes: int = 8704,
) -> JitSpec:
    """JIT spec of the MXFP4 fused kernel, with or without the in-kernel weight copy,
    for ranks on one node or spread over ``num_nodes`` nodes."""
    uri = get_megamoe_sm90_mxfp4_uri(
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
        in_kernel_weight_copy,
        num_nodes,
        weight_tile_bytes,
    )
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
            in_kernel_weight_copy=in_kernel_weight_copy,
            num_nodes=num_nodes,
            weight_tile_bytes=weight_tile_bytes,
        ),
        fast_math,
        extra_include_paths=[_nccl_paths()[0]] if num_nodes > 1 else (),
        extra_cuda_cflags=_GIN_PROXY_CFLAGS if num_nodes > 1 else (),
    )


def get_megamoe_sm90_eplb_uri(num_ranks, num_experts, num_redundant_slots) -> str:
    """Module key of the replica-plan kernels, one per slot geometry."""
    return f"megamoe_sm90_eplb_r{num_ranks}_e{num_experts}_s{num_redundant_slots}"


def gen_megamoe_sm90_eplb_module(
    num_ranks, num_experts, num_redundant_slots
) -> JitSpec:
    """Replica-plan kernels of the MXFP4 fused kernel: the logical-to-physical
    remap of the top-k ids and the separate weight copy of the changed redundant
    expert slots. One module per slot geometry."""
    return _gen_fp4_module(
        get_megamoe_sm90_eplb_uri(num_ranks, num_experts, num_redundant_slots),
        "megamoe_sm90_eplb_config",
        ("megamoe_sm90_eplb.cu",),
        dict(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_redundant_slots=num_redundant_slots,
        ),
        False,
    )


def _sm120_nccl():
    # NCCL 2.30.7 headers + library (the Device API GIN), located as for the moe_ep cake backend
    from flashinfer.moe_ep.shims.sm120.cake_mxfp4.cake_jit import nccl_installation

    return nccl_installation()


def gen_megamoe_sm120_mxfp4_module(
    fast_math: bool, gin_type: int | None = None
) -> JitSpec:
    """SM120 MXFP4 nosync MegaMoE. Shapes are runtime arguments: one module per fast-math policy,
    P2P only (``gin_type`` None) or with NIC (GIN) pairs built for one NCCL GIN backend, so that
    the kernel inlines only that backend's put / signal code."""
    csrc = _csrc_dir()
    uri = "megamoe_sm120_mxfp4"
    cflags = ["-std=c++17"]
    ldflags: list[str] = []
    includes: list[str | Path] = [csrc, jit_env.FLASHINFER_CSRC_DIR]
    if gin_type is not None:
        from flashinfer.moe_ep.shims.sm120.cake_mxfp4.cake_jit import GIN_BACKENDS

        names = {t: (name, flags) for name, (t, flags) in GIN_BACKENDS.items()}
        if gin_type not in names:
            raise ValueError(f"unsupported NCCL GIN type {gin_type}")
        backend, backend_flags = names[gin_type]
        include, library = _sm120_nccl()
        uri += f"_gin_{backend}"
        cflags += [
            "-DFLASHINFER_MEGAMOE_SM120_GIN=1",
            "-D_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE",
            *backend_flags,
        ]
        ldflags = [str(library), f"-Wl,-rpath,{library.parent}"]
        includes.append(include)
    uri += "_fm" if fast_math else ""
    directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "megamoe_sm120_mxfp4.cu"
    write_if_different(source, (csrc / "megamoe_sm120_mxfp4.cu").read_text())
    return gen_jit_spec(
        uri,
        [source],
        extra_cuda_cflags=[*cflags, *sm120a_nvcc_flags],
        extra_ldflags=ldflags or None,
        extra_include_paths=includes,
        use_fast_math=fast_math,
    )


def gen_megamoe_sm120_gin_session_module() -> JitSpec:
    """SM120 MegaMoE NIC transport session: NCCL window, LSA peer pointers, device communicator."""
    include, library = _sm120_nccl()
    uri = "megamoe_sm120_gin_session"
    directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "megamoe_sm120_gin_session.cu"
    write_if_different(
        source, (_csrc_dir() / "megamoe_sm120_gin_session.cu").read_text()
    )
    return gen_jit_spec(
        uri,
        [source],
        extra_cuda_cflags=["-std=c++17", *sm120a_nvcc_flags],
        extra_ldflags=[str(library), f"-Wl,-rpath,{library.parent}"],
        extra_include_paths=[include, jit_env.FLASHINFER_CSRC_DIR],
    )
