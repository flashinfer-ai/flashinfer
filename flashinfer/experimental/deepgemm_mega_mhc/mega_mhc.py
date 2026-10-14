"""Prepared fused projection, Sinkhorn, residual mixing and RMSNorm on SM100a/SM103a.

One generated source per schedule class (shifted x shared scale layout x the
unroll and partition choices of a split-count class) serves every supported
architecture; the loader compiles it with the exact flag set of the device it
runs on and passes the split count, its FP8 norm partition count and the device
SM count on the compile line. The token count is a run-time argument: the split
count and the launch grid are chosen here from the same rules the producer uses.
"""

from __future__ import annotations

import functools
from typing import Any

GENERATED_ROOT = "csrc/experimental/deepgemm_mega_mhc/generated"
CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
MAX_TOKENS = 1 << 20
SPLIT_BARRIER_WORDS = 2 * (MAX_TOKENS // 64) * 16

# Registry written by the exporter: one record per delivered source, and the
# schedule keys (hidden, split count, shifted, shared scale layout) each source
# serves with its compile-line defines.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_mega_mhc_078e89653dee8bca9319": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_078e89653dee8bca9319_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_078e89653dee8bca9319_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_078e89653dee8bca9319",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [20, 27],
    },
    "cake_mega_mhc_18f70125e9a55ebad54f": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_18f70125e9a55ebad54f_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_18f70125e9a55ebad54f_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_18f70125e9a55ebad54f",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [16],
    },
    "cake_mega_mhc_35a28241805e52d8d025": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_35a28241805e52d8d025_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_35a28241805e52d8d025_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_35a28241805e52d8d025",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [16, 20, 27],
    },
    "cake_mega_mhc_45d5adebe0a596a1e4a3": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_45d5adebe0a596a1e4a3_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_45d5adebe0a596a1e4a3_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_45d5adebe0a596a1e4a3",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [40],
    },
    "cake_mega_mhc_557f419314de9efca58e": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_557f419314de9efca58e_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_557f419314de9efca58e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_557f419314de9efca58e",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [40],
    },
    "cake_mega_mhc_7d69b17877f1cfa74fef": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_7d69b17877f1cfa74fef_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_7d69b17877f1cfa74fef_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_7d69b17877f1cfa74fef",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [20, 27],
    },
    "cake_mega_mhc_b9fd6d16636225d37ec9": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_b9fd6d16636225d37ec9_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_b9fd6d16636225d37ec9_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_b9fd6d16636225d37ec9",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": False,
        "shared_sf_block_m": 0,
        "num_splits": [16],
    },
    "cake_mega_mhc_f7faf2e5631cadf7cdd7": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f7faf2e5631cadf7cdd7_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f7faf2e5631cadf7cdd7_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_f7faf2e5631cadf7cdd7",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [16, 20, 27],
    },
    "cake_mega_mhc_f87d829e07967a853c8e": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f87d829e07967a853c8e_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f87d829e07967a853c8e_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_f87d829e07967a853c8e",
        "hidden": 5120,
        "shifted": True,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [40],
    },
    "cake_mega_mhc_f9701d865ea1f932513b": {
        "sources": [
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f9701d865ea1f932513b_kernel.cu",
            "csrc/experimental/deepgemm_mega_mhc/generated/cake_mega_mhc_f9701d865ea1f932513b_binding.cu",
        ],
        "kernel_symbol": "kernel_cake_mega_mhc_f9701d865ea1f932513b",
        "hidden": 5120,
        "shifted": False,
        "shared_sf": True,
        "shared_sf_block_m": 224,
        "num_splits": [40],
    },
}
ROUTES: list[dict[str, Any]] = [
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_35a28241805e52d8d025",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_f7faf2e5631cadf7cdd7",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_b9fd6d16636225d37ec9",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 16,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_18f70125e9a55ebad54f",
        "defines": {"NUM_SPLITS": 16, "FP8_NORM_PARTITIONS": 2},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_35a28241805e52d8d025",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_f7faf2e5631cadf7cdd7",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_7d69b17877f1cfa74fef",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 20,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_078e89653dee8bca9319",
        "defines": {"NUM_SPLITS": 20, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_35a28241805e52d8d025",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_f7faf2e5631cadf7cdd7",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_7d69b17877f1cfa74fef",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 27,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_078e89653dee8bca9319",
        "defines": {"NUM_SPLITS": 27, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": False,
        "shared_sf": False,
        "program": "cake_mega_mhc_45d5adebe0a596a1e4a3",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": False,
        "shared_sf": True,
        "program": "cake_mega_mhc_f9701d865ea1f932513b",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": True,
        "shared_sf": False,
        "program": "cake_mega_mhc_557f419314de9efca58e",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
    {
        "hidden": 5120,
        "num_splits": 40,
        "shifted": True,
        "shared_sf": True,
        "program": "cake_mega_mhc_f87d829e07967a853c8e",
        "defines": {"NUM_SPLITS": 40, "FP8_NORM_PARTITIONS": 10},
    },
]
ARG_PLAN: list[list[str]] = [
    ["tma_buffer", "residual_map"],
    ["tma_buffer", "x_map"],
    ["tma_buffer", "fn_map"],
    ["tma_buffer", "post_map"],
    ["tma_buffer", "comb_map"],
    ["tma_buffer", "prev_map"],
    ["tma_buffer", "new_residual_map"],
    ["tma_buffer", "y_map"],
    ["buffer", "mix_scales"],
    ["buffer", "mix_bases"],
    ["buffer", "new_prev_mix"],
    ["buffer", "new_post_mix"],
    ["buffer", "new_comb_res_mix"],
    ["buffer", "rmsnorm_weight"],
    ["buffer", "new_residual"],
    ["buffer", "y_bf16"],
    ["buffer", "y_fp8"],
    ["buffer", "y_primary_sf"],
    ["buffer", "y_shared_sf"],
    ["buffer", "scratch"],
    ["buffer", "split_barriers"],
    ["buffer", "launch_epochs"],
    ["parameter", "num_tokens"],
    ["parameter", "hc_norm_eps"],
    ["parameter", "hc_pre_eps"],
    ["parameter", "hc_post_scale"],
    ["parameter", "sinkhorn_eps"],
    ["parameter", "num_sinkhorn_iters"],
    ["parameter", "rmsnorm_eps"],
    ["parameter", "rmsnorm_scale"],
    ["parameter", "primary_sf_stride_token"],
    ["parameter", "primary_sf_stride_word"],
    ["parameter", "shared_sf_stride_word"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
COMPILE_FLAGS: list[str] = ["--use_fast_math"]
ARCHES: list[str] = ["sm_100a", "sm_103a"]


def num_splits(num_tokens, hidden, num_sms, deterministic=False):
    """Fewest splits reaching the shortest longest task (deterministic mode fixes 16)."""
    if deterministic:
        return 16
    num_m_blocks = (num_tokens + 63) // 64
    blocks = hidden // 64
    max_splits = min(64, max(16, num_sms // num_m_blocks))
    longest = (blocks + max_splits - 1) // max_splits
    return (blocks + longest - 1) // longest


def num_launch_sms(
    num_tokens, hidden, num_sms, splits, shifted, store_fp8, has_gemm_sf, deterministic
):
    """Active CTAs of the launch; the kernel keeps its NUM_SMS scheduler stride."""
    if (
        num_sms in (148, 152)
        and hidden == 5120
        and not deterministic
        and shifted
        and store_fp8
        and num_tokens in (1, 64, 65)
    ):
        # Shifted FP8 T1/T64/T65 launch only the CTAs some role can reach.
        projection_ctas = ((num_tokens + 63) // 64) * splits
        mix_ctas = (num_tokens + 3) // 4
        norm_partitions = hidden // 512 if splits > 16 and hidden // 256 > 8 else 2
        if norm_partitions > 2:
            norm_partitions = min(
                norm_partitions, max(2, num_sms * 4 // ((num_tokens + 1) // 2))
            )
        norm_ctas = (((num_tokens + 1) // 2) * norm_partitions + 3) // 4
        return min(num_sms, max(projection_ctas, mix_ctas, norm_ctas))
    if (
        num_sms in (148, 152)
        and hidden == 5120
        and not deterministic
        and not shifted
        and num_tokens == 65
        and store_fp8
        and has_gemm_sf
    ):
        return 80
    return num_sms


def fp8_norm_partitions(hidden, splits):
    """Hidden partitions the FP8 Norm workers split a token pair into."""
    return hidden // 512 if splits > 16 and hidden // 256 > 8 else 2


@functools.cache
def device_facts(device_index):
    """(architecture, SM count) of ``cuda:<device_index>``, queried once per process."""
    import torch

    from ...utils import get_compute_capability, get_device_sm_count

    device = torch.device("cuda", device_index)
    capability = get_compute_capability(device)
    arch = CAPABILITIES.get(capability)
    if arch is None or arch not in ARCHES:
        raise RuntimeError(
            f"Mega mHC has no exported programs for compute capability {capability}; "
            f"supported: {ARCHES}"
        )
    return arch, get_device_sm_count(device)


def select_program(hidden, splits, shifted, shared_sf):
    """(source name, compile-line defines) serving the schedule key, or raise."""
    for route in ROUTES:
        if (
            route["hidden"] == hidden
            and route["num_splits"] == splits
            and route["shifted"] == shifted
            and route["shared_sf"] == shared_sf
        ):
            return route["program"], dict(route["defines"])
    raise NotImplementedError(
        f"No exported mHC program for hidden={hidden}, num_splits={splits}, "
        f"shifted={shifted}, shared_sf={shared_sf}"
    )


@functools.cache
def load_program(name, arch, num_sms, defines):
    """Build ``name`` for ``arch`` with the route's defines and ``NUM_SMS`` on the compile line."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

    arch_flags = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]
    values = (*defines, ("NUM_SMS", num_sms))
    spec = gen_jit_spec(
        name="_".join(
            [name, arch, *(f"{key.lower()}{value}" for key, value in values)]
        ),
        sources=[
            env.FLASHINFER_CSRC_DIR / path.removeprefix("csrc/")
            for path in PROGRAMS[name]["sources"]
        ],
        extra_cuda_cflags=[
            *arch_flags,
            *COMPILE_FLAGS,
            *(f"-D{key}={value}" for key, value in values),
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            env.FLASHINFER_CSRC_DIR / GENERATED_ROOT.removeprefix("csrc/"),
            env.FLASHINFER_CSRC_DIR,
            env.FLASHINFER_INCLUDE_DIR,
        ],
        use_fast_math=False,
    )
    return spec.build_and_load(), str(spec.get_library_path())


def _new_outputs(
    x, residual, post_mix, comb_res_mix, shifted_prev_mix, sf_layout, block_m
):
    import torch

    tokens, hidden = x.shape
    outputs = dict(
        new_residual=torch.empty_like(residual),
        new_post_mix=torch.empty_like(post_mix),
        new_comb_res_mix=torch.empty_like(comb_res_mix),
        y_bf16=torch.empty_like(x),
    )
    if shifted_prev_mix is not None:
        outputs["new_prev_mix"] = torch.empty_like(shifted_prev_mix)
    if sf_layout != "bf16":
        outputs["y_fp8"] = torch.empty(
            (tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device
        )
        if sf_layout == "col":
            outputs["y_gemm_sf"] = torch.empty_strided(
                (tokens, hidden // 128),
                (1, (tokens + 3) // 4 * 4),
                dtype=torch.int32,
                device=x.device,
            )
        else:
            outputs["y_routed_sf"] = torch.empty(
                (tokens, hidden // 128), dtype=torch.int32, device=x.device
            )
            rows = (tokens + block_m - 1) // block_m * ((block_m + 127) // 128 * 128)
            storage = torch.empty_strided(
                (rows, hidden // 128), (1, rows), dtype=torch.int32, device=x.device
            )
            outputs.update(
                y_shared_sf_storage=storage,
                y_shared_sf=storage[:tokens],
                shared_sf_block_m=block_m,
            )
    return outputs


def _raw_storage(tensor):
    import torch

    words = (
        tensor.untyped_storage().nbytes() // tensor.element_size()
        - tensor.storage_offset()
    )
    return tensor.as_strided((words,), (1,)).view(torch.uint32)


class MegaMHCPlan:
    """Bind a fused mHC operation and reuse its private invocation state.

    x/residual/rmsnorm_weight are BF16; projection and mixing coefficients are
    FP32. The four residual routes produce BF16 normalized output plus FP8
    E4M3 output and packed per-32 UE8M0 scales. ``sf_layout='col'`` stores
    column-major scale words; ``'extra'`` also stores the shared-expert padded
    row permutation. Any token count in [1, 2**20] is accepted; the split
    count follows the token count (or ``deterministic``), and the program for
    that split count must be exported for ``hidden``.

    Scratch, split_barriers and launch_epochs may be caller-owned. Fresh
    barriers/epochs must be zeroed once before first use and retained across
    ordinary calls and CUDA Graph replay. Epoch storage always has num_sms
    entries, including CTAs omitted by a selected launch grid. One plan must
    not execute concurrently on multiple streams. run() uses the current
    PyTorch stream and performs no device allocation or workspace reset.
    """

    def __init__(
        self,
        *,
        x,
        residual,
        post_mix,
        comb_res_mix,
        shifted_prev_mix,
        fn,
        mix_scales,
        mix_bases,
        rmsnorm_weight,
        hc_mult=4,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=20,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
        sf_layout="col",
        shared_sf_block_m=224,
        out=None,
        deterministic=False,
        scratch=None,
        split_barriers=None,
        launch_epochs=None,
    ):
        import torch

        if x.device.type != "cuda":
            raise RuntimeError("Mega mHC requires a CUDA device")
        if x.ndim != 2 or hc_mult != 4 or sf_layout not in ("bf16", "col", "extra"):
            raise ValueError(
                "Expected x[T,H], four residual routes and col/extra/bf16 scale layout"
            )
        tokens, hidden = x.shape
        if hidden % 1024 or not 1 <= tokens <= MAX_TOKENS or num_sinkhorn_iters < 1:
            raise ValueError(
                "Expected H divisible by 1024, 1 <= T <= 2**20 and positive Sinkhorn iterations"
            )
        if sf_layout == "extra" and shared_sf_block_m <= 0:
            raise ValueError("shared_sf_block_m must be positive")
        if sf_layout == "bf16":
            raise NotImplementedError(
                "Exported mHC programs store the FP8 output; use sf_layout 'col' or 'extra'"
            )
        arch, sms = device_facts(x.device.index)
        shifted = shifted_prev_mix is not None
        splits = num_splits(tokens, hidden, sms, deterministic)
        shared_sf = sf_layout == "extra"
        program, defines = select_program(hidden, splits, shifted, shared_sf)
        block_m = shared_sf_block_m if shared_sf else 0
        if shared_sf and block_m != PROGRAMS[program]["shared_sf_block_m"]:
            raise NotImplementedError(
                f"Exported mHC programs use shared_sf_block_m={PROGRAMS[program]['shared_sf_block_m']}"
            )
        launch_sms = num_launch_sms(
            tokens,
            hidden,
            sms,
            splits,
            shifted,
            True,
            sf_layout == "col",
            deterministic,
        )
        self.config = dict(num_splits=splits, num_sms=sms, num_launch_sms=launch_sms)
        self.program = program
        self.defines = defines

        def require(name, tensor, dtype, shape, *, contiguous=True):
            if (
                tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or tensor.device != x.device
            ):
                raise ValueError(
                    f"{name} must be {dtype}{shape} on the input CUDA device"
                )
            if contiguous and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        for name, tensor, dtype, shape in (
            ("x", x, torch.bfloat16, (tokens, hidden)),
            ("residual", residual, torch.bfloat16, (tokens, 4, hidden)),
            ("post_mix", post_mix, torch.float32, (tokens, 4, 1)),
            ("comb_res_mix", comb_res_mix, torch.float32, (tokens, 4, 4)),
            ("fn", fn, torch.float32, (24, 4 * hidden)),
            ("mix_scales", mix_scales, torch.float32, (3,)),
            ("mix_bases", mix_bases, torch.float32, (24,)),
            ("rmsnorm_weight", rmsnorm_weight, torch.bfloat16, (hidden,)),
        ):
            require(name, tensor, dtype, shape)
        if shifted:
            require("shifted_prev_mix", shifted_prev_mix, torch.float32, (tokens, 4, 1))
        if out is None:
            out = _new_outputs(
                x,
                residual,
                post_mix,
                comb_res_mix,
                shifted_prev_mix,
                sf_layout,
                block_m,
            )
        for name, dtype, shape in (
            ("new_residual", torch.bfloat16, (tokens, 4, hidden)),
            ("new_post_mix", torch.float32, (tokens, 4, 1)),
            ("new_comb_res_mix", torch.float32, (tokens, 4, 4)),
            ("y_bf16", torch.bfloat16, (tokens, hidden)),
            ("y_fp8", torch.float8_e4m3fn, (tokens, hidden)),
        ):
            require(name, out[name], dtype, shape)
        if shifted:
            require("new_prev_mix", out["new_prev_mix"], torch.float32, (tokens, 4, 1))
        primary = out["y_gemm_sf" if sf_layout == "col" else "y_routed_sf"]
        require(
            "primary scale words",
            primary,
            torch.int32,
            (tokens, hidden // 128),
            contiguous=False,
        )
        expected_stride = (
            (1, (tokens + 3) // 4 * 4) if sf_layout == "col" else (hidden // 128, 1)
        )
        if primary.stride() != expected_stride:
            raise ValueError(f"Primary scale strides must be {expected_stride}")
        if shared_sf:
            rows = (tokens + block_m - 1) // block_m * ((block_m + 127) // 128 * 128)
            storage, shared = out["y_shared_sf_storage"], out["y_shared_sf"]
            require(
                "shared scale storage",
                storage,
                torch.int32,
                (rows, hidden // 128),
                contiguous=False,
            )
            require(
                "shared scale view",
                shared,
                torch.int32,
                (tokens, hidden // 128),
                contiguous=False,
            )
            if (
                storage.stride() != (1, rows)
                or shared.stride() != (1, rows)
                or storage.data_ptr() != shared.data_ptr()
                or out["shared_sf_block_m"] != block_m
            ):
                raise ValueError(
                    "Shared scales require the padded column-major storage and its first-T-row view"
                )
        scratch_shape = ((tokens + 63) // 64 * splits * (1536 + 128),)
        if scratch is None:
            scratch = torch.empty(scratch_shape, dtype=torch.float32, device=x.device)
        if split_barriers is None:
            split_barriers = torch.zeros(
                (SPLIT_BARRIER_WORDS,), dtype=torch.uint64, device=x.device
            )
        if launch_epochs is None:
            launch_epochs = torch.zeros((sms,), dtype=torch.uint64, device=x.device)
        require("scratch", scratch, torch.float32, scratch_shape)
        require("split_barriers", split_barriers, torch.uint64, (SPLIT_BARRIER_WORDS,))
        require("launch_epochs", launch_epochs, torch.uint64, (sms,))
        dummy_u32 = torch.empty(1, dtype=torch.uint32, device=x.device)
        dummy_float = torch.empty(1, dtype=torch.float32, device=x.device)
        shared = out["y_shared_sf"] if shared_sf else dummy_u32
        self.bindings = dict(
            residual_map=residual,
            x_map=x,
            fn_map=fn,
            post_map=post_mix.view(tokens, 4),
            comb_map=comb_res_mix.view(tokens, 16),
            prev_map=(shifted_prev_mix if shifted else post_mix).view(tokens, 4),
            new_residual_map=out["new_residual"],
            y_map=out["y_bf16"] if shifted else x,
            mix_scales=mix_scales,
            mix_bases=mix_bases,
            new_prev_mix=out["new_prev_mix"] if shifted else dummy_float,
            new_post_mix=out["new_post_mix"],
            new_comb_res_mix=out["new_comb_res_mix"],
            rmsnorm_weight=rmsnorm_weight,
            new_residual=out["new_residual"],
            y_bf16=out["y_bf16"],
            y_fp8=out["y_fp8"].view(torch.uint8),
            y_primary_sf=_raw_storage(primary),
            y_shared_sf=_raw_storage(shared),
            scratch=scratch,
            split_barriers=split_barriers,
            launch_epochs=launch_epochs,
            num_tokens=tokens,
            hc_norm_eps=hc_norm_eps,
            hc_pre_eps=hc_pre_eps,
            hc_post_scale=hc_post_scale,
            sinkhorn_eps=sinkhorn_eps,
            num_sinkhorn_iters=num_sinkhorn_iters,
            rmsnorm_eps=rmsnorm_eps,
            rmsnorm_scale=rmsnorm_scale,
            primary_sf_stride_token=primary.stride(0),
            primary_sf_stride_word=primary.stride(1),
            shared_sf_stride_word=shared.stride(1) if shared_sf else 0,
            grid_x=launch_sms,
            grid_y=1,
            grid_z=1,
        )
        module, self.library_path = load_program(
            program, arch, sms, tuple(sorted(defines.items()))
        )
        args = tuple(self.bindings[name] for _kind, name in ARG_PLAN)
        self._submission = (module["run"], args)
        self._retained = (
            module,
            x,
            residual,
            post_mix,
            comb_res_mix,
            shifted_prev_mix,
            fn,
            mix_scales,
            mix_bases,
            rmsnorm_weight,
            dummy_u32,
            dummy_float,
        )
        self.outputs, self.scratch, self.split_barriers, self.launch_epochs = (
            out,
            scratch,
            split_barriers,
            launch_epochs,
        )

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.outputs
