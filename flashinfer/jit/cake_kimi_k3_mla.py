"""JIT loader for the CAKE-generated Kimi-K3 MLA FP8 paged-attention kernels (SM100 / SM103).

The generated CUDA lives in ``csrc/cake_kimi_k3_mla/``: one ``*_kernel.cu`` + ``*_binding.cu``
pair per physical program, compiled with the exact flag set of the architecture the device
runs (small per-architecture lowering differences are ``#if __CUDA_ARCH__`` guards inside the
one source).  ``MODULES`` holds one record per program (sources, compile flags, FFI entry,
argument plan, supported architectures) and ``KERNELS`` maps a logical kernel key
(``main_rt16`` .. ``main_rt96``, ``main_wide``, ``reduce_w1`` / ``reduce_w2`` / ``reduce_w4``,
``reduce_cta``) to its program, or to one program per architecture when the two lowerings do
not fold (the wide decode kernel: its sm_103a variant computes the row max with the TMEM
reduction load).  Both literals are written by the Cake exporter; regenerate them, never edit
them by hand.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

_ARCH_NVCC_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
PACKAGE_DIR = "cake_kimi_k3_mla"

# Written by the exporter (see the module docstring).
MODULES: dict[str, dict[str, Any]] = {
    "cake_kimi_k3_mla_fp8_paged_attention_037deeabf96bd0d2d769": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_037deeabf96bd0d2d769_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_037deeabf96bd0d2d769_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_213d59cceca7f0a817c0": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_213d59cceca7f0a817c0_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_213d59cceca7f0a817c0_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_3ca2519f90933160bbcb": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_3ca2519f90933160bbcb_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_3ca2519f90933160bbcb_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_5e6f67c2c89c6eb24061": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_5e6f67c2c89c6eb24061_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_5e6f67c2c89c6eb24061_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_kr"],
            ["tma_buffer", "tmap_v"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_977c0b337b3f4232a99f": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_977c0b337b3f4232a99f_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_977c0b337b3f4232a99f_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_a1945d8ab3d522beb089": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_a1945d8ab3d522beb089_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_a1945d8ab3d522beb089_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_abc850f024ed156e98ca": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_abc850f024ed156e98ca_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_abc850f024ed156e98ca_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_b1c5eb5cb28aa248c2a9": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_b1c5eb5cb28aa248c2a9_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_b1c5eb5cb28aa248c2a9_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "cum_seq_lens_q"],
            ["parameter", "batch"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "bmm2_scale"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["parameter", "tile_rows"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_e1bdfbc717f1844c99a8": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_e1bdfbc717f1844c99a8_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_e1bdfbc717f1844c99a8_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_kr"],
            ["tma_buffer", "tmap_v"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "O"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["parameter", "m_tiles"],
            ["parameter", "n_full_items"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_f2ed3823ac88848234c4": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_f2ed3823ac88848234c4_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_f2ed3823ac88848234c4_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
    "cake_kimi_k3_mla_fp8_paged_attention_f6f8a85d05a12ec611f2": {
        "sources": [
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_f6f8a85d05a12ec611f2_kernel.cu",
            "cake_kimi_k3_mla/cake_kimi_k3_mla_fp8_paged_attention_f6f8a85d05a12ec611f2_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "tmap_q"],
            ["tma_buffer", "tmap_qr"],
            ["tma_buffer", "tmap_k"],
            ["tma_buffer", "tmap_kr"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "seq_lens"],
            ["buffer", "cum_seq_lens_q"],
            ["buffer", "page_table"],
            ["parameter", "softmax_scale_log2"],
            ["parameter", "bmm2_scale"],
            ["parameter", "num_heads"],
            ["parameter", "num_split"],
            ["parameter", "max_pages_per_seq"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "arches": ["sm_100a", "sm_103a"],
    },
}
KERNELS: dict[str, str | dict[str, str]] = {
    "main_rt16": "cake_kimi_k3_mla_fp8_paged_attention_f2ed3823ac88848234c4",
    "main_rt32": "cake_kimi_k3_mla_fp8_paged_attention_f6f8a85d05a12ec611f2",
    "main_rt48": "cake_kimi_k3_mla_fp8_paged_attention_3ca2519f90933160bbcb",
    "main_rt64": "cake_kimi_k3_mla_fp8_paged_attention_abc850f024ed156e98ca",
    "main_rt96": "cake_kimi_k3_mla_fp8_paged_attention_a1945d8ab3d522beb089",
    "main_wide": {
        "sm_100a": "cake_kimi_k3_mla_fp8_paged_attention_5e6f67c2c89c6eb24061",
        "sm_103a": "cake_kimi_k3_mla_fp8_paged_attention_e1bdfbc717f1844c99a8",
    },
    "reduce_cta": "cake_kimi_k3_mla_fp8_paged_attention_037deeabf96bd0d2d769",
    "reduce_w1": "cake_kimi_k3_mla_fp8_paged_attention_977c0b337b3f4232a99f",
    "reduce_w2": "cake_kimi_k3_mla_fp8_paged_attention_b1c5eb5cb28aa248c2a9",
    "reduce_w4": "cake_kimi_k3_mla_fp8_paged_attention_213d59cceca7f0a817c0",
}


def supported_arches() -> frozenset[str]:
    """Architectures with generated programs (``sm_100a``, ``sm_103a``)."""
    return frozenset(arch for record in MODULES.values() for arch in record["arches"])


def get_cake_kimi_k3_mla_kernel(key: str, *, arch: str) -> dict[str, Any]:
    """Return the program record (plus ``name``) that serves kernel ``key`` on ``arch``."""
    try:
        entry = KERNELS[key]
    except KeyError as exc:
        raise ValueError(f"CAKE Kimi-K3 MLA has no generated kernel {key!r}") from exc
    name = entry if isinstance(entry, str) else entry.get(arch)
    record = MODULES.get(name)
    if record is None or arch not in record["arches"]:
        arches = (
            sorted(entry.keys())
            if isinstance(entry, dict)
            else MODULES[entry]["arches"]
        )
        raise ValueError(
            f"CAKE Kimi-K3 MLA kernel {key!r} is not generated for {arch} (supported: {', '.join(arches)})"
        )
    return dict(record, name=name)


def _csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / PACKAGE_DIR
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / PACKAGE_DIR
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "CAKE Kimi-K3 MLA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


@functools.cache
def gen_cake_kimi_k3_mla_module(name: str, arch: str) -> JitSpec:
    """JIT spec of one program (device + binding translation units) for ``arch``."""
    try:
        record = MODULES[name]
    except KeyError as exc:
        raise ValueError(f"CAKE Kimi-K3 MLA has no generated program {name!r}") from exc
    if arch not in record["arches"]:
        raise ValueError(
            f"CAKE Kimi-K3 MLA program {name!r} is not generated for {arch}"
        )
    if arch not in _ARCH_NVCC_FLAGS:
        raise ValueError(f"unsupported CAKE Kimi-K3 MLA architecture: {arch}")
    csrc_dir = _csrc_dir()
    sources = [csrc_dir / Path(src).name for src in record["sources"]]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "CAKE Kimi-K3 MLA generated sources were not found: "
            + ", ".join(str(path) for path in missing)
        )
    spec = gen_jit_spec(
        name=f"cake_kimi_k3_mla_{name}_{arch.replace('_', '')}",
        sources=sources,
        extra_cuda_cflags=[
            *_ARCH_NVCC_FLAGS[arch],
            *record["compile_flags"],
            *record.get("host_linkage_flags", ()),
        ],
        # The generated record owns the fast-math decision.
        use_fast_math=False,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _include_dir()],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated CAKE Kimi-K3 MLA {name} JIT spec: {spec.name}")
    return spec


@functools.cache
def get_cake_kimi_k3_mla_module(name: str, arch: str):
    loaded = gen_cake_kimi_k3_mla_module(name, arch).build_and_load()
    logger.info(f"Loaded CAKE Kimi-K3 MLA {name} module for {arch}")
    return loaded


__all__ = [
    "KERNELS",
    "MODULES",
    "gen_cake_kimi_k3_mla_module",
    "get_cake_kimi_k3_mla_kernel",
    "get_cake_kimi_k3_mla_module",
    "supported_arches",
]
