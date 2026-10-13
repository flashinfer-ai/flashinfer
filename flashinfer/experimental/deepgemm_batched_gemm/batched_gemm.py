"""Prepared per-head FP8 projections with BF16 or dynamic FP8 output on SM100a/SM103a."""

from __future__ import annotations

import functools
from typing import Any

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# SM counts the two pinned BF16 schedules were selected on (B200, GB300). Other
# devices route BF16 output to the general schedule, which is not exported.
PINNED_NUM_SMS = (148, 152)

# Generated programs: one source pair per physical schedule, compiled for every
# listed architecture with that architecture's exact flags.
PROGRAMS: dict[str, dict[str, Any]] = {
    "cake_deepgemm_batched_gemm_44b3db5dcd94945d6530": {
        "sources": [
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_44b3db5dcd94945d6530_kernel.cu",
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_44b3db5dcd94945d6530_binding.cu",
        ],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_batched_gemm_4fed225332cf21378870": {
        "sources": [
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_4fed225332cf21378870_kernel.cu",
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_4fed225332cf21378870_binding.cu",
        ],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_batched_gemm_9c78b5deea1e2a3f6d18": {
        "sources": [
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_9c78b5deea1e2a3f6d18_kernel.cu",
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_9c78b5deea1e2a3f6d18_binding.cu",
        ],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_batched_gemm_c1767ea2214a0b0df04b": {
        "sources": [
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_c1767ea2214a0b0df04b_kernel.cu",
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_c1767ea2214a0b0df04b_binding.cu",
        ],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_batched_gemm_df7ac68be53c4ef83047": {
        "sources": [
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_df7ac68be53c4ef83047_kernel.cu",
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_df7ac68be53c4ef83047_binding.cu",
        ],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_deepgemm_batched_gemm_ed2c9a5f2dfe74719dc3": {
        "sources": [
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_ed2c9a5f2dfe74719dc3_kernel.cu",
            "experimental/deepgemm_batched_gemm/generated/cake_deepgemm_batched_gemm_ed2c9a5f2dfe74719dc3_binding.cu",
        ],
        "arg_plan": [
            ["tma_buffer", "A"],
            ["tma_buffer", "B"],
            ["tma_buffer", "SFA"],
            ["tma_buffer", "SFB"],
            ["tma_buffer", "D"],
            ["buffer", "SFD"],
            ["parameter", "M"],
            ["parameter", "grid_m"],
            ["parameter", "sfd_stride"],
            ["parameter", "alpha"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "compile_flags": [],
        "arches": ["sm_100a", "sm_103a"],
    },
}
# Schedule key (route, tile, stage count, epilogue) -> program.
ROUTES: dict[str, str] = {
    "bf16_t128_bm64_bn128_s10_bf16": "cake_deepgemm_batched_gemm_c1767ea2214a0b0df04b",
    "bf16_t4_bm16_bn128_s12_bf16": "cake_deepgemm_batched_gemm_df7ac68be53c4ef83047",
    "general_bm128_bn128_s5_alpha": "cake_deepgemm_batched_gemm_ed2c9a5f2dfe74719dc3",
    "n256_bm128_bn256_s5_fp8": "cake_deepgemm_batched_gemm_4fed225332cf21378870",
    "swap_ab_bm16_bn128_s12_fp8": "cake_deepgemm_batched_gemm_44b3db5dcd94945d6530",
    "swap_ab_bm64_bn128_s10_fp8": "cake_deepgemm_batched_gemm_9c78b5deea1e2a3f6d18",
}


def device_arch(device):
    """Generated-program architecture for ``device`` (raises when none is exported)."""
    import torch

    device = torch.device(device)
    catalogued = sorted(
        {arch for record in PROGRAMS.values() for arch in record["arches"]}
    )
    if device.type != "cuda":
        raise RuntimeError("Batched FP8 projection requires a CUDA device")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = _ARCHES.get(capability)
    if arch is None or arch not in catalogued:
        raise RuntimeError(
            f"Batched FP8 projection has no exported programs for compute capability "
            f"{capability}; exported architectures: {catalogued}"
        )
    return arch


def supported_num_sms(arch):
    """SM counts on which every exported schedule, including the pinned BF16 ones, is selected."""
    return PINNED_NUM_SMS


def _source_layout(tokens, num_heads, inner, width, num_sms):
    """FP8 tile layout the projection selects for ``tokens`` rows.

    Mirrors the DeepGEMM SM100 layout heuristic (``get_layout_candidates`` /
    ``get_layout_info`` / ``compare``) for the three exported FP8 layouts:
    ``(swap_ab, block_m, block_n, num_stages)``. ``None`` means the heuristic
    selects another layout and the general 128x128 schedule applies.
    """
    if (num_heads, inner, width) != (8, 4096, 1024):
        return None
    best_key, best_layout = None, None
    for swap_ab in (False, True):
        block_ms = (
            range(16, 257, 16)
            if swap_ab
            else (32 if tokens <= 32 else 64 if tokens <= 64 else 128,)
        )
        block_ns = (128,) if swap_ab else (16, *range(32, 257, 32))
        for cluster_m in (1, 2):
            if swap_ab and cluster_m == 2:
                continue
            for cluster_n in (1, 2):
                cluster_size = cluster_m * cluster_n
                if (
                    cluster_size > 2
                    or (not swap_ab and cluster_n == 2)
                    or num_sms % cluster_size
                ):
                    continue
                for block_m in block_ms:
                    if (
                        block_m // cluster_n % 8
                        or ((tokens + block_m - 1) // block_m) % cluster_m
                    ):
                        continue
                    for block_n in block_ns:
                        if (
                            block_n // cluster_m % 8
                            or ((width + block_n - 1) // block_n) % cluster_n
                        ):
                            continue
                        sf_cols = ((block_m + 127) // 128 + (block_n + 127) // 128) * 4
                        if (block_m if swap_ab else block_n) + sf_cols > 512:
                            continue
                        store_n = (
                            128
                            if swap_ab
                            else next(
                                value
                                for value in (128, 64, 32, 16)
                                if block_n % value == 0
                            )
                        )
                        if store_n % 32:
                            continue
                        blocks = (
                            ((tokens + block_m - 1) // block_m)
                            * ((width + block_n - 1) // block_n)
                            * num_heads
                        )
                        waves = (blocks + num_sms - 1) // num_sms
                        utilization = blocks % num_sms or num_sms
                        key = (
                            waves != 1,
                            -cluster_size,
                            waves,
                            -utilization,
                            block_m + block_n,
                            block_m * block_n,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_layout = (
                                swap_ab,
                                block_m,
                                block_n,
                                cluster_m,
                                cluster_n,
                            )
    if best_layout not in (
        (False, 128, 256, 2, 1),
        (True, 16, 128, 1, 2),
        (True, 64, 128, 1, 2),
    ):
        return None
    swap_ab, block_m, block_n, cluster_m, cluster_n = best_layout
    smem_cd = (16 * block_n if swap_ab else min(128, block_m) * 128) * 2
    smem_extra = smem_cd + 32 * 8 * 3 + 2 * 8 * 3 + 8 + 4
    smem_per_stage = (block_m // cluster_n + block_n // cluster_m) * 128
    smem_per_stage += (
        (((block_m + 127) // 128 + (block_n + 127) // 128) * 128) * 128 // 32
    )
    stages = min((232448 - smem_extra) // smem_per_stage, 32)
    return swap_ab, block_m, block_n, stages


def route_config(tokens, num_heads, inner, width, num_sms, epilogue, num_stages=5):
    """Schedule and launch geometry of one projection: route, tile, stages, grid.

    The selection depends on the token count, the SM count and the epilogue
    only; M, the M tile count and the grid are launch arguments of the
    selected program, so every token count maps onto one of a few programs.
    """
    layout = None
    if (epilogue, num_stages) == ("fp8", 5):
        layout = _source_layout(tokens, num_heads, inner, width, num_sms)
    block_m, block_n = 128, 128
    geometry = (tokens, num_heads, inner, width, epilogue, num_stages)
    if num_sms in PINNED_NUM_SMS and geometry == (128, 8, 4096, 1024, "bf16", 5):
        route, block_m, block_n, num_stages = "bf16_t128", 64, 128, 10
    elif num_sms in PINNED_NUM_SMS and geometry == (4, 8, 4096, 1024, "bf16", 5):
        route, block_m, block_n, num_stages = "bf16_t4", 16, 128, 12
    elif layout is None:
        route = "general"
    else:
        swap_ab, block_m, block_n, num_stages = layout
        route = "swap_ab" if swap_ab else "n256"
    grid_m = (tokens + block_m - 1) // block_m
    launch_ctas = num_sms
    if route == "swap_ab":
        # Only the working CTAs are launched, keeping the two-CTA pairing intact.
        tiles = grid_m * (width // block_n) * num_heads
        launch_ctas = min(num_sms, tiles + tiles % 2)
    return dict(
        route=route,
        block_m=block_m,
        block_n=block_n,
        block_k=128,
        num_stages=num_stages,
        num_sms=num_sms,
        epilogue=epilogue,
        grid=(launch_ctas, 1, 1),
        grid_m=grid_m,
    )


def schedule_key(config):
    """Program key of a route configuration (the ``ROUTES`` key)."""
    return (
        f"{config['route']}_bm{config['block_m']}_bn{config['block_n']}"
        f"_s{config['num_stages']}_{config['epilogue']}"
    )


def _nvcc_flags(arch):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


@functools.cache
def jit_spec(name, arch):
    """JIT build specification of program ``name`` for ``arch`` (exact-architecture flags)."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = PROGRAMS[name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[env.FLASHINFER_CSRC_DIR / path for path in record["sources"]],
        extra_cuda_cflags=[
            *_nvcc_flags(arch),
            *record["compile_flags"],
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,
    )


@functools.cache
def load_program(arch, name):
    return jit_spec(name, arch).build_and_load()


def _pack_scales(scales, padded_rows):
    import torch

    heads, rows, groups = scales.shape
    exponents = (scales.contiguous().view(torch.int32).to(torch.int64) >> 23) & 255
    groups_of_four = exponents.reshape(heads, rows, groups // 4, 4)
    shifts = torch.arange(4, dtype=torch.int64, device=scales.device) * 8
    words = (groups_of_four << shifts).sum(-1).to(torch.int32)
    storage = torch.zeros(
        (heads, groups // 4, padded_rows), dtype=torch.int32, device=scales.device
    )
    storage[:, :, :rows] = words.permute(0, 2, 1)
    return storage.reshape(heads * (groups // 4), padded_rows).view(torch.uint32)


class BatchedGemmPlan:
    """Prepared A[T,H,K] @ B[H,N,K] -> output[T,H,N].

    Inputs are E4M3 with positive power-of-two FP32 scales per A token/K128
    and B N128/K128 block. Scale packing and output allocation occur during
    preparation. Operand values may change between runs; prepare a new plan
    when the input scales change. Dynamic FP8 returns E4M3 values and packed
    per-32 UE8M0 scale words. BF16 optionally applies runtime alpha. Submit
    on the current stream; do not concurrently reuse one output. The
    ``descriptor_workspace`` argument is accepted and ignored.
    """

    def __init__(
        self,
        a,
        b,
        *,
        output_fp8=True,
        alpha=None,
        out=None,
        output_scales=None,
        descriptor_workspace=None,
    ):
        import torch

        # accepted for signature stability; tensor maps travel by value
        del descriptor_workspace

        aq, asf = a
        bq, bsf = b
        arch = device_arch(aq.device)
        if (
            aq.ndim != 3
            or bq.ndim != 3
            or aq.dtype != torch.float8_e4m3fn
            or bq.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("Expected E4M3 A[T,H,K] and B[H,N,K]")
        tokens, heads, inner = aq.shape
        width = bq.shape[1]
        if (
            bq.shape[0] != heads
            or bq.shape[2] != inner
            or tokens < 1
            or width % 128
            or inner % 512
        ):
            raise ValueError(
                "Expected matching H/K, T>=1, N divisible by128 and K divisible by512"
            )
        if (
            asf.dtype != torch.float32
            or bsf.dtype != torch.float32
            or tuple(asf.shape) != (tokens, heads, inner // 128)
            or tuple(bsf.shape) != (heads, width // 128, inner // 128)
        ):
            raise ValueError("Expected FP32 A[T,H,K/128] and B[H,N/128,K/128] scales")
        if (
            any(t.device != aq.device for t in (bq, asf, bsf))
            or not aq.is_contiguous()
            or not bq.is_contiguous()
        ):
            raise ValueError(
                "Operands must be contiguous; operands and scales must share a CUDA device"
            )
        if output_fp8 and alpha is not None:
            raise ValueError(
                "Dynamic FP8 output and runtime alpha are separate epilogues"
            )
        if not output_fp8 and output_scales is not None:
            raise ValueError("BF16 output has no output scales")
        epilogue = "fp8" if output_fp8 else "alpha" if alpha is not None else "bf16"
        sms = torch.cuda.get_device_properties(aq.device).multi_processor_count
        self.config = route_config(tokens, heads, inner, width, sms, epilogue)
        key = schedule_key(self.config)
        program = ROUTES.get(key)
        if program is None:
            raise NotImplementedError(
                f"The {self.config['route']} schedule ({key}) selected for T={tokens}, "
                f"H={heads}, K={inner}, N={width} with {epilogue} output on {sms} SMs "
                "is not exported"
            )
        self.program = program
        cfg = self.config
        dtype = torch.float8_e4m3fn if output_fp8 else torch.bfloat16
        if out is None:
            out = torch.empty((tokens, heads, width), dtype=dtype, device=aq.device)
        if (
            out.dtype != dtype
            or tuple(out.shape) != (tokens, heads, width)
            or not out.is_contiguous()
            or out.device != aq.device
        ):
            raise ValueError(
                "Output must have the selected dtype and contiguous [T,H,N] layout on the input device"
            )
        if output_fp8:
            shape = (tokens, heads * width // 128)
            stride = (1, (tokens + 3) // 4 * 4)
            if output_scales is None:
                output_scales = torch.empty_strided(
                    shape, stride, dtype=torch.int32, device=aq.device
                )
            if (
                output_scales.dtype != torch.int32
                or output_scales.device != aq.device
                or tuple(output_scales.shape) != shape
                or tuple(output_scales.stride()) != stride
            ):
                raise ValueError(
                    "Output scales must be int32 [T,H*N/128], column-major with T padded to4"
                )
            words = (
                output_scales.untyped_storage().nbytes() // output_scales.element_size()
                - output_scales.storage_offset()
            )
            sf_storage = output_scales.as_strided((words,), (1,)).view(torch.uint32)
            sf_stride = output_scales.stride(1)
        else:
            sf_storage = torch.empty((1,), dtype=torch.uint32, device=aq.device)
            sf_stride = 0
        packed_a = _pack_scales(asf.permute(1, 0, 2), (tokens + 127) // 128 * 128)
        packed_b = _pack_scales(bsf.repeat_interleave(128, dim=1), width)
        self.bindings = dict(
            A=aq.view(torch.uint8),
            B=bq.view(torch.uint8),
            SFA=packed_a,
            SFB=packed_b,
            D=out.view(torch.uint8) if output_fp8 else out,
            SFD=sf_storage,
            M=tokens,
            grid_m=cfg["grid_m"],
            sfd_stride=sf_stride,
            alpha=1.0 if alpha is None else float(alpha),
            grid_x=cfg["grid"][0],
            grid_y=cfg["grid"][1],
            grid_z=cfg["grid"][2],
        )
        module = load_program(arch, program)
        record = PROGRAMS[program]
        args = tuple(self.bindings[name] for _kind, name in record["arg_plan"])
        self._submission = (module[record["ffi_entry"]], args)
        self._retained = (module, a, b, out, output_scales, sf_storage, self.bindings)
        self.values, self.scales = out, output_scales
        self.output = (out, output_scales) if output_fp8 else out

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.output
