# SPDX-License-Identifier: Apache-2.0
"""JIT-only task specialization of the native XQA source."""

import functools
from pathlib import Path

import torch

from ...compilation_context import CompilationContext
from ...jit import env as jit_env
from ...jit.core import gen_jit_spec
from ...jit.xqa import xqa_nvcc_flags
from .geometry import check_head_config, check_query_tile, max_table_width


def gen_module(
    persistent: bool,
    *,
    direct: bool = False,
    page_size: int = 32,
    head_group_ratio: int = 6,
    multi_query: bool = False,
    query_tile: int = 16,
):
    max_table_width(page_size)
    check_head_config(1, head_group_ratio)
    check_query_tile(query_tile, 2 if multi_query else 1)
    if direct:
        persistent = False
    source = Path(__file__).resolve().parent / "csrc"
    flags = CompilationContext().get_nvcc_flags_list(supported_major_versions=[12])
    return gen_jit_spec(
        f"experimental_xqa_decode_buckets_r{head_group_ratio}_d256_p{page_size}_persistent_{int(persistent)}_direct_{int(direct)}_spec_{int(multi_query)}_m{query_tile}",
        [source / "binding.cu"],
        extra_cuda_cflags=xqa_nvcc_flags
        + flags
        + [
            "--expt-relaxed-constexpr",
            "-Xptxas=-v",
            "-DHEAD_ELEMS=256",
            f"-DHEAD_GRP_SIZE={head_group_ratio}",
            "-DINPUT_FP16=0",
            "-DCACHE_ELEM_ENUM=3",
            f"-DTOKENS_PER_PAGE={page_size}",
            "-DSLIDING_WINDOW=0",
            "-DLOW_PREC_OUTPUT=0",
            f"-DSPEC_DEC={int(multi_query)}",
            f"-DM_TILESIZE={query_tile}",
            "-DIS_SPEC_DEC_TREE=0",
            "-DFLASHINFER_XQA_TASKS=1",
            f"-DFLASHINFER_XQA_PERSISTENT_TASKS={int(persistent)}",
            f"-DFLASHINFER_XQA_DIRECT_TASKS={int(direct)}",
        ],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR / "xqa"],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def get_module(
    persistent: bool,
    device: torch.device,
    *,
    direct: bool = False,
    page_size: int = 32,
    head_group_ratio: int = 6,
    multi_query: bool = False,
    query_tile: int = 16,
):
    # Configure dynamic shared memory outside capture, on each owning device.
    with torch.cuda.device(device):
        module = gen_module(
            persistent,
            direct=direct,
            page_size=page_size,
            head_group_ratio=head_group_ratio,
            multi_query=multi_query,
            query_tile=query_tile,
        ).build_and_load()
        resources = torch.empty(4, dtype=torch.int64, device="cpu")
        module.configure(resources)
    return module, tuple(resources.tolist())
