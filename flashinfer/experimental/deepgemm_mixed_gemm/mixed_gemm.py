"""Prepared mixed FP8 E4M3 x packed FP4 E2M1 GEMM with BF16 output on SM100a/SM103a.

Eight generated programs, each one source compiled for both architectures:
four normal schedules (BM128 with the N tile 128/160/224/256), three swap-AB
schedules for small M (BN128 with the M tile 16/32/48) and the BK128 schedule
with per-128 A scales (BM128 x BN224). Tile counts and the SM count are runtime
arguments; the route is chosen from the problem shape by ``select_route``.
``PROGRAMS`` and ``ROUTES`` are populated by the generated-program export; do
not edit them by hand.
"""

from __future__ import annotations

import functools

PROGRAMS = {
    "cake_deepgemm_mixed_gemm_20bec5dc16309a77b8c4": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_20bec5dc16309a77b8c4_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_20bec5dc16309a77b8c4_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_215099de8776807fb433": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_215099de8776807fb433_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_215099de8776807fb433_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_7951ef4c2b789eb6ced5": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_7951ef4c2b789eb6ced5_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_7951ef4c2b789eb6ced5_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_7c41f9e1650b40806f5e": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_7c41f9e1650b40806f5e_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_7c41f9e1650b40806f5e_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_91df6d765e8a0d343acd": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_91df6d765e8a0d343acd_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_91df6d765e8a0d343acd_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_a0f125c27e5a860a3f90": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_a0f125c27e5a860a3f90_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_a0f125c27e5a860a3f90_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_ac5a27628d43b5aec93d": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_ac5a27628d43b5aec93d_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_ac5a27628d43b5aec93d_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
    "cake_deepgemm_mixed_gemm_ce64deed53f017813730": {
        "sources": [
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_ce64deed53f017813730_kernel.cu",
            "experimental/deepgemm_mixed_gemm/cake_deepgemm_mixed_gemm_ce64deed53f017813730_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            [
                "tma_buffer",
                "A",
            ],
            [
                "tma_buffer",
                "B",
            ],
            [
                "tma_buffer",
                "SFA",
            ],
            [
                "tma_buffer",
                "SFB",
            ],
            [
                "tma_buffer",
                "C_tma",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "N",
            ],
            [
                "parameter",
                "K",
            ],
            [
                "parameter",
                "grid_m",
            ],
            [
                "parameter",
                "grid_n",
            ],
            [
                "parameter",
                "K_tiles",
            ],
            [
                "parameter",
                "SCHED_DIV",
            ],
            [
                "parameter",
                "TAIL_DIV",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
    },
}

ROUTES = {
    "bk128_bm128_bn224_s6_ga128": "cake_deepgemm_mixed_gemm_215099de8776807fb433",
    "normal_bm128_bn128_s7_ga32": "cake_deepgemm_mixed_gemm_7c41f9e1650b40806f5e",
    "normal_bm128_bn160_s7_ga32": "cake_deepgemm_mixed_gemm_7951ef4c2b789eb6ced5",
    "normal_bm128_bn224_s6_ga32": "cake_deepgemm_mixed_gemm_a0f125c27e5a860a3f90",
    "normal_bm128_bn256_s5_ga32": "cake_deepgemm_mixed_gemm_20bec5dc16309a77b8c4",
    "swap_ab_bm16_bn128_s12_ga32": "cake_deepgemm_mixed_gemm_ce64deed53f017813730",
    "swap_ab_bm32_bn128_s11_ga32": "cake_deepgemm_mixed_gemm_91df6d765e8a0d343acd",
    "swap_ab_bm48_bn128_s10_ga32": "cake_deepgemm_mixed_gemm_ac5a27628d43b5aec93d",
}

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
BLOCK_K = 128
GROUP_M, GROUP_N = 16, 8  # persistent raster groups of the normal / swap-AB schedules
# route name -> (family, block_m, block_n, store_n)
_SCHEDULES = {
    "normal_bm128_bn128_s7_ga32": ("normal", 128, 128, 64),
    "normal_bm128_bn160_s7_ga32": ("normal", 128, 160, 32),
    "normal_bm128_bn224_s6_ga32": ("normal", 128, 224, 32),
    "normal_bm128_bn256_s5_ga32": ("normal", 128, 256, 64),
    "swap_ab_bm16_bn128_s12_ga32": ("swap_ab", 16, 128, 0),
    "swap_ab_bm32_bn128_s11_ga32": ("swap_ab", 32, 128, 0),
    "swap_ab_bm48_bn128_s10_ga32": ("swap_ab", 48, 128, 0),
    "bk128_bm128_bn224_s6_ga128": ("bk128", 128, 224, 32),
}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def aligned_mn(extent: int) -> int:
    """MN extent of the prepacked UE8M0 scale words (four bytes per word, 16-byte TMA rows)."""
    return _ceil_div(extent, 4) * 4


def scale_words(k: int, gran_k: int) -> int:
    """Packed-K words: one word holds four UE8M0 bytes covering ``4 * gran_k`` K elements."""
    return _ceil_div(k, 4 * gran_k)


def _fits(name: str, m: int, n: int, k: int) -> bool:
    family, block_m, block_n, store_n = _SCHEDULES[name]
    grid_m, grid_n = _ceil_div(m, block_m), _ceil_div(n, block_n)
    if family == "swap_ab":
        return n >= 128 and grid_n % 2 == 0
    sfn = _ceil_div(block_n, 128) * 128 if family == "normal" else block_n
    return grid_m % 2 == 0 and n >= max(block_n // 2, store_n) and aligned_mn(n) >= sfn


def _rank(name: str, m: int, n: int, k: int, num_sms: int):
    """Source ``SM100ArchSpec`` ordering: fewest waves, fullest last wave, smaller tile."""
    _family, block_m, block_n, _store_n = _SCHEDULES[name]
    blocks = _ceil_div(m, block_m) * _ceil_div(n, block_n)
    waves = _ceil_div(blocks, num_sms)
    last = blocks % num_sms
    return (
        waves,
        -(num_sms if last == 0 else last),
        block_m + block_n,
        block_m * block_n,
    )


def select_route(m: int, n: int, k: int, *, num_sms: int, gran_k_a: int = 32) -> str:
    """Route name for one problem; ``NotImplementedError`` when no schedule can raster it."""
    if k <= 0 or k % BLOCK_K or m <= 0 or n <= 0:
        raise NotImplementedError(
            f"mixed FP8xFP4 GEMM needs M, N > 0 and K a multiple of {BLOCK_K}; got M={m}, N={n}, K={k}"
        )
    if gran_k_a == 128:
        family = "bk128"
    elif gran_k_a == 32:
        family = "swap_ab" if m <= 128 else "normal"
    else:
        raise ValueError(
            "gran_k_a must be 32 (per-32 A scales) or 128 (per-128 A scales)"
        )
    candidates = [
        name for name, schedule in _SCHEDULES.items() if schedule[0] == family
    ]
    if family == "normal" and _ceil_div(m, 128) % 2:
        candidates = ["swap_ab_bm48_bn128_s10_ga32"]
    fitting = [name for name in candidates if _fits(name, m, n, k)]
    if not fitting:
        raise NotImplementedError(
            f"no mixed FP8xFP4 GEMM schedule rasters M={m}, N={n}, K={k}, gran_k_a={gran_k_a}: M<=128 needs an even "
            "ceil(N/128); M>128 needs an even ceil(M/128) or an even ceil(N/128); gran_k_a=128 needs an even "
            "ceil(M/128) and N>=224"
        )
    return min(fitting, key=lambda name: _rank(name, m, n, k, num_sms))


def route_geometry(name: str, m: int, n: int, k: int, gran_k_a: int) -> dict[str, int]:
    family, block_m, block_n, _store_n = _SCHEDULES[name]
    grid_m, grid_n = _ceil_div(m, block_m), _ceil_div(n, block_n)
    # Divisor of the persistent raster: swap-AB groups GROUP_N N tiles per M tile, the other
    # schedules GROUP_M M tiles per N tile.
    # TAIL_DIV splits the short last group; full groups split by the power-of-two group size.
    if family == "swap_ab":
        sched_divisor, tail_divisor = grid_m * GROUP_N, grid_n % GROUP_N or GROUP_N
    else:
        sched_divisor, tail_divisor = grid_n * GROUP_M, grid_m % GROUP_M or GROUP_M
    return dict(
        grid_m=grid_m,
        grid_n=grid_n,
        K_tiles=k // BLOCK_K,
        sched_divisor=sched_divisor,
        tail_divisor=tail_divisor,
        sfa_words=scale_words(k, gran_k_a),
        sfa_mn=aligned_mn(m),
        sfb_words=scale_words(k, 32),
        sfb_mn=aligned_mn(n),
    )


def _fast_divmod(divisor: int):
    """Three-integer carrier (divisor, multiplier, shift) of the kernel's fast unsigned divmod."""
    from tvm_ffi import Shape

    if divisor == 1:
        return Shape((1, 0, 0))
    p = 31 + (divisor - 1).bit_length()
    return Shape((divisor, (((1 << p) + divisor - 1) // divisor) & 0xFFFFFFFF, p - 32))


@functools.cache
def device_facts(device_index: int) -> tuple[str, int]:
    """(architecture, SM count) of one device, queried once; raises when no program is exported for it."""
    import torch
    from flashinfer.utils import get_compute_capability, get_device_sm_count

    device = torch.device("cuda", device_index)
    capability = get_compute_capability(device)
    arch = _ARCHES.get(capability)
    if arch is None:
        raise RuntimeError(
            f"Mixed FP8xFP4 GEMM has no exported programs for compute capability {capability}; "
            f"supported: {sorted(_ARCHES.values())}"
        )
    return arch, get_device_sm_count(device)


def _nvcc_flags(arch: str):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


def jit_spec(arch: str, name: str):
    """JIT spec of one shared program compiled with ``arch``'s exact flags (the arch is in the spec name)."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = PROGRAMS[name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[env.FLASHINFER_CSRC_DIR / source for source in record["sources"]],
        extra_cuda_cflags=[*_nvcc_flags(arch), *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,
    )


@functools.cache
def load_program(arch: str, name: str):
    return jit_spec(arch, name).build_and_load()[PROGRAMS[name]["ffi_entry"]]


class MixedGemmPlan:
    """Prepared ``out[:m] = A @ B.T`` (FP8 E4M3 A, packed FP4 E2M1 B, prepacked UE8M0 scales, BF16 out).

    Preparation validates the operands, chooses the route for the problem shape
    and this device's SM count, and loads the program; ``run()`` submits the one
    kernel launch on the current stream with the tensor maps encoded by value.
    Changing tensor addresses or layouts requires a new plan; contents may change.
    """

    def __init__(self, a, b, a_scales, b_scales, *, m=None, out=None, gran_k_a=32):
        import torch

        if (
            a.ndim != 2
            or b.ndim != 2
            or a.dtype not in (torch.float8_e4m3fn, torch.uint8)
            or b.dtype not in (torch.int8, torch.uint8)
        ):
            raise ValueError(
                "A must be E4M3 FP8 (or raw uint8) [M, K]; B must be packed E2M1 bytes [N, K/2]"
            )
        if a.device.type != "cuda":
            raise RuntimeError("Mixed FP8xFP4 GEMM requires CUDA tensors")
        n, k = b.shape[0], b.shape[1] * 2
        if a.shape[1] != k:
            raise ValueError("A/B logical K dimensions must agree")
        m = a.shape[0] if m is None else int(m)
        if not 1 <= m <= a.shape[0]:
            raise ValueError(f"m must be in [1, {a.shape[0]}] (the rows of A), got {m}")
        arch, num_sms = device_facts(a.device.index)
        self.route = select_route(m, n, k, num_sms=num_sms, gran_k_a=gran_k_a)
        self.program = ROUTES[self.route]
        geometry = route_geometry(self.route, m, n, k, gran_k_a)
        for tensor, shape, label in (
            (a_scales, (geometry["sfa_words"], geometry["sfa_mn"]), "A scales"),
            (b_scales, (geometry["sfb_words"], geometry["sfb_mn"]), "B scales"),
        ):
            if (
                tensor.dtype not in (torch.int32, torch.uint32)
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(
                    f"{label} must be packed UE8M0 int32/uint32 words of shape {shape}, got {tuple(tensor.shape)}"
                )
        if out is None:
            out = a.new_empty((m, n), dtype=torch.bfloat16)
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n):
            raise ValueError(f"out must be BF16 of shape {(m, n)}")
        tensors = (a, b, a_scales, b_scales, out)
        if any(t.device != a.device or not t.is_contiguous() for t in tensors):
            raise ValueError(
                "Operands, packed scales and output must be contiguous on one CUDA device"
            )
        self.grid = (num_sms, 1, 1)
        self.output = self.storage = out
        bindings = dict(
            A=a.view(torch.uint8),
            B=b.view(torch.uint8),
            SFA=a_scales.view(torch.uint32),
            SFB=b_scales.view(torch.uint32),
            C_tma=out,
            M=m,
            N=n,
            K=k,
            grid_m=geometry["grid_m"],
            grid_n=geometry["grid_n"],
            K_tiles=geometry["K_tiles"],
            SCHED_DIV=_fast_divmod(geometry["sched_divisor"]),
            TAIL_DIV=_fast_divmod(geometry["tail_divisor"]),
            grid_x=self.grid[0],
            grid_y=self.grid[1],
            grid_z=self.grid[2],
        )
        self._entry = load_program(arch, self.program)
        self._args = tuple(
            bindings[name] for _kind, name in PROGRAMS[self.program]["arg_plan"]
        )
        self._retained = tensors

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            self._entry(*self._args)
        return self.output
