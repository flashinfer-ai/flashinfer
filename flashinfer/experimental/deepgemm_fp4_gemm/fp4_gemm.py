# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0
"""Prepared native packed FP4 GEMM with FP32 alpha and BF16 output on SM100a/SM103a.

Eight generated programs, each one source compiled for both architectures:
three swap-AB schedules for small M (BN128 with the M tile 16/32/48) and five
normal schedules (BM128 with the N tile 16/128/160/224/256). Tile counts and
the SM count are runtime arguments; the route is chosen from the problem shape
by ``select_route``. ``PROGRAMS`` and ``ROUTES`` are populated by the
generated-program export; do not edit them by hand.
"""

from __future__ import annotations

import functools

PROGRAMS = {
    "cake_deepgemm_fp4_gemm_40f53c88e86797021dc8": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_40f53c88e86797021dc8_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_40f53c88e86797021dc8_binding.cu",
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
                "parameter",
                "alpha",
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
    "cake_deepgemm_fp4_gemm_7f6481216fbfb264acf6": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_7f6481216fbfb264acf6_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_7f6481216fbfb264acf6_binding.cu",
        ],
        "compile_flags": [
            "--ptxas-options=--register-usage-level=10",
        ],
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
                "parameter",
                "alpha",
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
    "cake_deepgemm_fp4_gemm_a51b18d814bd13cb5875": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_a51b18d814bd13cb5875_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_a51b18d814bd13cb5875_binding.cu",
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
                "alpha",
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
    "cake_deepgemm_fp4_gemm_b8f21c5cd2f70ffc6f24": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_b8f21c5cd2f70ffc6f24_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_b8f21c5cd2f70ffc6f24_binding.cu",
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
                "alpha",
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
    "cake_deepgemm_fp4_gemm_bc83e664ecb731739d12": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_bc83e664ecb731739d12_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_bc83e664ecb731739d12_binding.cu",
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
                "alpha",
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
    "cake_deepgemm_fp4_gemm_e6bafe7ceb6a5cfd40fc": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_e6bafe7ceb6a5cfd40fc_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_e6bafe7ceb6a5cfd40fc_binding.cu",
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
                "alpha",
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
    "cake_deepgemm_fp4_gemm_e6c9fb59c73ea5fce24c": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_e6c9fb59c73ea5fce24c_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_e6c9fb59c73ea5fce24c_binding.cu",
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
                "alpha",
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
    "cake_deepgemm_fp4_gemm_f6e26f62fc441e353075": {
        "sources": [
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_f6e26f62fc441e353075_kernel.cu",
            "experimental/deepgemm_fp4_gemm/cake_deepgemm_fp4_gemm_f6e26f62fc441e353075_binding.cu",
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
                "alpha",
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
    "normal_bm128_bn128_s7": "cake_deepgemm_fp4_gemm_bc83e664ecb731739d12",
    "normal_bm128_bn160_s7": "cake_deepgemm_fp4_gemm_b8f21c5cd2f70ffc6f24",
    "normal_bm128_bn16_s7": "cake_deepgemm_fp4_gemm_e6c9fb59c73ea5fce24c",
    "normal_bm128_bn224_s6": "cake_deepgemm_fp4_gemm_7f6481216fbfb264acf6",
    "normal_bm128_bn256_s6": "cake_deepgemm_fp4_gemm_40f53c88e86797021dc8",
    "swap_ab_bm16_bn128_s11": "cake_deepgemm_fp4_gemm_f6e26f62fc441e353075",
    "swap_ab_bm32_bn128_s10": "cake_deepgemm_fp4_gemm_e6bafe7ceb6a5cfd40fc",
    "swap_ab_bm48_bn128_s10": "cake_deepgemm_fp4_gemm_a51b18d814bd13cb5875",
}

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
BLOCK_K = 256
# route name -> (family, block_m, block_n, stages, store_n)
_SCHEDULES = {
    "swap_ab_bm16_bn128_s11": ("swap_ab", 16, 128, 11, 0),
    "swap_ab_bm32_bn128_s10": ("swap_ab", 32, 128, 10, 0),
    "swap_ab_bm48_bn128_s10": ("swap_ab", 48, 128, 10, 0),
    "normal_bm128_bn16_s7": ("normal", 128, 16, 7, 16),
    "normal_bm128_bn128_s7": ("normal", 128, 128, 7, 64),
    "normal_bm128_bn160_s7": ("normal", 128, 160, 7, 32),
    "normal_bm128_bn224_s6": ("normal", 128, 224, 6, 32),
    "normal_bm128_bn256_s6": ("normal", 128, 256, 6, 32),
}


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def aligned_mn(extent: int) -> int:
    """MN extent of the prepacked UE8M0 scale words (four bytes per word, 16-byte TMA rows)."""
    return _ceil_div(extent, 4) * 4


def scale_words(k: int) -> int:
    """Packed-K words: one word holds four UE8M0 bytes covering 128 K elements."""
    return _ceil_div(k, 128)


def _fits(name: str, m: int, n: int, k: int) -> bool:
    family, block_m, block_n, _stages, store_n = _SCHEDULES[name]
    grid_m, grid_n = _ceil_div(m, block_m), _ceil_div(n, block_n)
    if family == "swap_ab":
        return n >= 128 and grid_n % 2 == 0
    return grid_m % 2 == 0 and n >= max(block_n // 2, store_n) and aligned_mn(n) >= _ceil_div(block_n, 128) * 128


def _rank(name: str, m: int, n: int, k: int, num_sms: int):
    """Source ``SM100ArchSpec`` ordering: fewest waves, fullest last wave, smaller tile."""
    _family, block_m, block_n, _stages, _store_n = _SCHEDULES[name]
    blocks = _ceil_div(m, block_m) * _ceil_div(n, block_n)
    waves = _ceil_div(blocks, num_sms)
    last = blocks % num_sms
    return (waves, -(num_sms if last == 0 else last), block_m + block_n, block_m * block_n)


def select_route(m: int, n: int, k: int, *, num_sms: int) -> str:
    """Route name for one problem; ``NotImplementedError`` when no schedule can raster it."""
    if k <= 0 or k % BLOCK_K or m <= 0 or n <= 0:
        raise NotImplementedError(f"native FP4 GEMM needs M, N > 0 and K a multiple of {BLOCK_K}; got M={m}, N={n}, K={k}")
    family = "swap_ab" if m <= 128 else "normal"
    candidates = [name for name, schedule in _SCHEDULES.items() if schedule[0] == family]
    if family == "normal" and _ceil_div(m, 128) % 2:
        candidates = ["swap_ab_bm48_bn128_s10"]
    fitting = [name for name in candidates if _fits(name, m, n, k)]
    if not fitting:
        raise NotImplementedError(
            f"no native FP4 GEMM schedule rasters M={m}, N={n}, K={k}: M<=128 needs N>=128 and an even "
            "ceil(N/128); M>128 needs an even ceil(M/128) or an even ceil(N/128)"
        )
    return min(fitting, key=lambda name: _rank(name, m, n, k, num_sms))


def route_geometry(name: str, m: int, n: int, k: int) -> dict[str, int]:
    _family, block_m, block_n, _stages, _store_n = _SCHEDULES[name]
    return dict(grid_m=_ceil_div(m, block_m), grid_n=_ceil_div(n, block_n), K_tiles=k // BLOCK_K,
                sfa_words=scale_words(k), sfa_mn=aligned_mn(m), sfb_words=scale_words(k), sfb_mn=aligned_mn(n))


def route_stages(name: str) -> int:
    return _SCHEDULES[name][3]


@functools.cache
def device_facts(device_index: int) -> tuple[str, int]:
    """(architecture, SM count) of one device; raises when no program is exported for it."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = _ARCHES.get(capability)
    if arch is None:
        raise RuntimeError(
            f"Native FP4 GEMM has no exported programs for compute capability {capability}; "
            f"supported: {sorted(_ARCHES.values())}"
        )
    return arch, torch.cuda.get_device_properties(device_index).multi_processor_count


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


class Fp4GemmPlan:
    """Prepared ``out[:m] = alpha * A @ B.T`` (packed E2M1 A and B, prepacked UE8M0 scales, BF16 out).

    Preparation validates the operands, chooses the route for the problem shape
    and this device's SM count, and loads the program; ``run()`` submits the one
    kernel launch on the current stream with the tensor maps encoded by value.
    Changing tensor addresses or layouts requires a new plan; contents may change.
    """

    def __init__(self, a, b, a_scales, b_scales, *, m=None, alpha=1.0, out=None, num_stages=None,
                 block_n=128, epilogue_store_n=32, descriptor_workspace=None):
        import torch

        if a.ndim != 2 or b.ndim != 2 or a.dtype not in (torch.int8, torch.uint8) or b.dtype not in (torch.int8, torch.uint8):
            raise ValueError("A/B must be packed int8/uint8 matrices [rows, K/2] and [N, K/2]")
        if a.device.type != "cuda":
            raise RuntimeError("Native FP4 GEMM requires CUDA tensors")
        n, k = b.shape[0], b.shape[1] * 2
        if a.shape[1] * 2 != k:
            raise ValueError("A/B packed K dimensions must agree")
        m = a.shape[0] if m is None else int(m)
        if not 1 <= m <= a.shape[0]:
            raise ValueError(f"m must be in [1, {a.shape[0]}] (the rows of A), got {m}")
        if block_n != 128 or epilogue_store_n != 32:
            raise NotImplementedError("the N tile and store width are selected per problem shape; pass the defaults")
        arch, num_sms = device_facts(a.device.index)
        self.route = select_route(m, n, k, num_sms=num_sms)
        if num_stages is not None and int(num_stages) != route_stages(self.route):
            raise NotImplementedError(
                f"the selected schedule uses {route_stages(self.route)} stages; pass num_stages=None or that count"
            )
        self.program = ROUTES[self.route]
        geometry = route_geometry(self.route, m, n, k)
        for tensor, shape, label in (
            (a_scales, (geometry["sfa_words"], geometry["sfa_mn"]), "A scales"),
            (b_scales, (geometry["sfb_words"], geometry["sfb_mn"]), "B scales"),
        ):
            if tensor.dtype not in (torch.int32, torch.uint32) or tuple(tensor.shape) != shape:
                raise ValueError(f"{label} must be packed UE8M0 int32/uint32 words of shape {shape}, got {tuple(tensor.shape)}")
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (m, n):
            raise ValueError(f"out must be BF16 of shape {(m, n)}")
        tensors = (a, b, a_scales, b_scales, out)
        if any(t.device != a.device or not t.is_contiguous() for t in tensors):
            raise ValueError("Operands, packed scales and output must be contiguous on one CUDA device")
        self.grid = (num_sms, 1, 1)
        self.output = self.storage = out
        self.descriptor_workspace = descriptor_workspace
        bindings = dict(
            A=a.view(torch.uint8), B=b.view(torch.uint8), SFA=a_scales.view(torch.uint32), SFB=b_scales.view(torch.uint32),
            C_tma=out, M=m, N=n, K=k, grid_m=geometry["grid_m"], grid_n=geometry["grid_n"], K_tiles=geometry["K_tiles"],
            alpha=float(alpha), grid_x=self.grid[0], grid_y=self.grid[1], grid_z=self.grid[2],
        )
        self._entry = load_program(arch, self.program)
        self._args = tuple(bindings[name] for _kind, name in PROGRAMS[self.program]["arg_plan"])
        self._retained = tensors

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            self._entry(*self._args)
        return self.output
