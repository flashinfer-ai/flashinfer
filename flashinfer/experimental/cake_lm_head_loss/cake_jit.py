"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags

# Explicit target-owned registration of the generated chunked LM-head + loss
# programs.  One record per architecture.  A record carries ``arch``, the host
# binding profile ``abi`` (the keyword set its kernels expect, see
# ``cake_backend``), the list of kernel ``stages`` it registers, the
# ``geometry`` the kernels were built for (see ``cake_backend.Geometry``: the
# vocabulary columns per row-statistics partial, the GEMM row tile and CTA
# pair, the K block of the weight-gradient GEMM, the element vector of the
# cast, the divisibility ``V`` / ``H`` / the row stride of ``X`` must satisfy
# and the label element type) and one physical entry per stage (translation
# units, compile flags, FFI entry, argument plan, grid rule, launch geometry
# and closure identity).  Populated verbatim by the generated-program export;
# do not edit by hand.
MODULES: dict[str, dict[str, Any]] = {
    "cake_lm_head_loss_sm_100a": {
        "arch": "sm_100a",
        "abi": "lm_head_loss_v1",
        "stages": [
            "gemm_logits",
            "gemm_logits_nostats",
            "row_finalize",
            "loss_reduce",
            "row_grad",
            "gemm_dx",
            "gemm_dx_s2",
            "gemm_dx_s3",
            "gemm_dx_s4",
            "gemm_dw_acc",
            "scale_cast_bf16",
            "scale_cast_f32",
        ],
        "geometry": {
            "stats_tile": 256,
            "row_tile": 128,
            "cta_group": 2,
            "k_block": 64,
            "cast_vec": 8,
            "vocab_multiple": 256,
            "hidden_multiple": 256,
            "ld_multiple": 8,
            "labels_dtype": "int64",
            "hidden": 6144,
            "vocab": 154880,
            "logits_cluster_ctas": 2,
            "dx_cluster_ctas": 2,
            "dw_cluster_ctas": 2,
        },
        "gemm_logits": {
            "module": "cake_lm_head_loss_98495ee25c515bb14cae",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_98495ee25c515bb14cae_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_98495ee25c515bb14cae_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "3f63216ea318ed4fb12bccba5b34d4773d71d7ce4be6a3f1017ba3f7fc07ca34",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*605)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_logits_nostats": {
            "module": "cake_lm_head_loss_2b1925b64416e75c6091",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_2b1925b64416e75c6091_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_2b1925b64416e75c6091_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "a0bb902d2a9fa6785f4ed1e872db1500b7903e0dbec83f1ec09ac3d6550cdd6b",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*605)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "row_finalize": {
            "module": "cake_lm_head_loss_40e27be3789cda6512a1",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_40e27be3789cda6512a1_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_40e27be3789cda6512a1_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "stats"],
                ["buffer", "z"],
                ["buffer", "labels"],
                ["buffer", "infer_logp"],
                ["buffer", "loss_weights"],
                ["buffer", "d_in"],
                ["buffer", "lse"],
                ["buffer", "logp"],
                ["buffer", "d"],
                ["buffer", "term"],
                ["parameter", "rows_c"],
                ["parameter", "row0"],
                ["parameter", "V"],
                ["parameter", "num_tiles"],
                ["parameter", "mode"],
                ["parameter", "loss_div"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "1d478adedc66811ef8755312d4ba23cb1ecd6e24dc27cf2b539fc82bd1dd6dbc",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["rows_c/8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "loss_reduce": {
            "module": "cake_lm_head_loss_f835250d89052888b6ae",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_f835250d89052888b6ae_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_f835250d89052888b6ae_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "term"],
                ["buffer", "loss_acc"],
                ["buffer", "loss_out"],
                ["parameter", "rows_c"],
                ["parameter", "first_chunk"],
                ["parameter", "last_chunk"],
                ["parameter", "mode"],
                ["parameter", "loss_div"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "361010297415b400c5963dfe841334eeb15d33f5bc0ae9bb5a501e9c560d0d3c",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": [1, 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "row_grad": {
            "module": "cake_lm_head_loss_cef7772361b458bb1ee2",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_cef7772361b458bb1ee2_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_cef7772361b458bb1ee2_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "z"],
                ["buffer", "labels"],
                ["buffer", "lse"],
                ["buffer", "d"],
                ["parameter", "row0"],
                ["parameter", "d_off"],
                ["parameter", "V"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "cfeea72e3619a459982c11d16dfa0db4d85f4d74027bb342a087f07051b6b01b",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, V//8/1024)", "rows_c", 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "gemm_dx": {
            "module": "cake_lm_head_loss_485bf80f49ec183cb77b",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_485bf80f49ec183cb77b_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_485bf80f49ec183cb77b_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "c3936bc6264c477e5e28b00fdb47739790707db7fc94cef546cadfdddf256c2a",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*24)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dx_s2": {
            "module": "cake_lm_head_loss_0d2db191075f9ef7263b",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_0d2db191075f9ef7263b_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_0d2db191075f9ef7263b_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "dfcec1c6b3fe6e62b133c7e02b268b786f734527e8d70f40c37374e2c207d18d",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*48)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dx_s3": {
            "module": "cake_lm_head_loss_6b4efd8d73871be4419d",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_6b4efd8d73871be4419d_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_6b4efd8d73871be4419d_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "111988c30f357536151a9a329d0097f868ab0afb244a2009bbd05894949b4997",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*72)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dx_s4": {
            "module": "cake_lm_head_loss_6138e5261199cebb53c0",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_6138e5261199cebb53c0_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_6138e5261199cebb53c0_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "746ed9df0ca9d15d05805c02670649852ee0aa6df28d81e17e0121fc71b4d0f6",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*96)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dw_acc": {
            "module": "cake_lm_head_loss_f7fe5807de393243c5a5",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_f7fe5807de393243c5a5_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_f7fe5807de393243c5a5_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "9d72b7aebbbbd37009d43b8df6c1646c42c196f57adf677306742b152f58b787",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, min(m_tiles//2*24, resident))*2", 1, 1],
            "launch": {"block": [192, 1, 1], "cluster": [2, 1, 1]},
        },
        "scale_cast_bf16": {
            "module": "cake_lm_head_loss_2fbdae03f8d282f1b6ea",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_2fbdae03f8d282f1b6ea_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_2fbdae03f8d282f1b6ea_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "acc"],
                ["buffer", "g"],
                ["buffer", "out"],
                ["parameter", "num_vecs"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "22e9aa4b1aa89201095426489faa890e6f57196079d695e480cd16a237fda0bd",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, min(num_vecs/2048, 65535))", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "scale_cast_f32": {
            "module": "cake_lm_head_loss_809bf68f30658bb0d1d6",
            "sources": [
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_809bf68f30658bb0d1d6_kernel.cu",
                "cake_lm_head_loss/sm_100a/cake_lm_head_loss_809bf68f30658bb0d1d6_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "acc"],
                ["buffer", "g"],
                ["buffer", "out"],
                ["parameter", "num_vecs"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "135db3b48a996cfca75f20f13a894c3ce4a7781fd5c723fa9c3ac37bb0cc2a06",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, min(num_vecs/2048, 65535))", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "c3100e9ec03f3c08d8ddec765f8ac8b75c647c0569416fa87fb8b7ccfc309d7b",
    },
    "cake_lm_head_loss_sm_103a": {
        "arch": "sm_103a",
        "abi": "lm_head_loss_v1",
        "stages": [
            "gemm_logits",
            "gemm_logits_nostats",
            "row_finalize",
            "loss_reduce",
            "row_grad",
            "gemm_dx",
            "gemm_dx_s2",
            "gemm_dx_s3",
            "gemm_dx_s4",
            "gemm_dw_acc",
            "scale_cast_bf16",
            "scale_cast_f32",
        ],
        "geometry": {
            "stats_tile": 256,
            "row_tile": 128,
            "cta_group": 2,
            "k_block": 64,
            "cast_vec": 8,
            "vocab_multiple": 256,
            "hidden_multiple": 256,
            "ld_multiple": 8,
            "labels_dtype": "int64",
            "hidden": 6144,
            "vocab": 154880,
            "logits_cluster_ctas": 2,
            "dx_cluster_ctas": 2,
            "dw_cluster_ctas": 2,
        },
        "gemm_logits": {
            "module": "cake_lm_head_loss_10a742f15238dce376fa",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_10a742f15238dce376fa_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_10a742f15238dce376fa_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["tma_buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "11f62f83e4cd185046ff29491b3f6fabc5af933934f602b9f98ff52932383c72",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*605)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_logits_nostats": {
            "module": "cake_lm_head_loss_47fd8ba4f5d359a9292c",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_47fd8ba4f5d359a9292c_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_47fd8ba4f5d359a9292c_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["tma_buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "e6870b24239c203ca5f7a0c9382362551a6cee04ce929aa9fa30b97e1e1a1607",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*605)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "row_finalize": {
            "module": "cake_lm_head_loss_eadb84f9fd90929a2ed2",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_eadb84f9fd90929a2ed2_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_eadb84f9fd90929a2ed2_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "stats"],
                ["buffer", "z"],
                ["buffer", "labels"],
                ["buffer", "infer_logp"],
                ["buffer", "loss_weights"],
                ["buffer", "d_in"],
                ["buffer", "lse"],
                ["buffer", "logp"],
                ["buffer", "d"],
                ["buffer", "term"],
                ["parameter", "rows_c"],
                ["parameter", "row0"],
                ["parameter", "V"],
                ["parameter", "num_tiles"],
                ["parameter", "mode"],
                ["parameter", "loss_div"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "f4d70a08579804822ed8c7f49b313c5b93bf4f29792a22d83d35c9187bf210fd",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["rows_c/8", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "loss_reduce": {
            "module": "cake_lm_head_loss_045eda29d291bc0ca03e",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_045eda29d291bc0ca03e_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_045eda29d291bc0ca03e_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "term"],
                ["buffer", "loss_acc"],
                ["buffer", "loss_out"],
                ["parameter", "rows_c"],
                ["parameter", "first_chunk"],
                ["parameter", "last_chunk"],
                ["parameter", "mode"],
                ["parameter", "loss_div"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "640bc9bc6efc821fd76dd6deca62175fbcf70f6349b96940a8e4d999654dde74",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": [1, 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "row_grad": {
            "module": "cake_lm_head_loss_9fb02bcaacbe1871aae8",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_9fb02bcaacbe1871aae8_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_9fb02bcaacbe1871aae8_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "z"],
                ["buffer", "labels"],
                ["buffer", "lse"],
                ["buffer", "d"],
                ["parameter", "row0"],
                ["parameter", "d_off"],
                ["parameter", "V"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5d03137bc87ce5a00306435c35db2f54385c5bafb209f4aea6cd04e4ecafc19d",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, V//8/1024)", "rows_c", 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "gemm_dx": {
            "module": "cake_lm_head_loss_31237132a0cf5c909f23",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_31237132a0cf5c909f23_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_31237132a0cf5c909f23_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "536bc3b65d7d53283cda758ab069fa976f06067ae2de72d215f8e25dcd307faf",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*24)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dx_s2": {
            "module": "cake_lm_head_loss_e91e55bd8eea8ce48c3a",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_e91e55bd8eea8ce48c3a_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_e91e55bd8eea8ce48c3a_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "d96d2a73be22e9d1d03e513e562cad2361c8cb99367a32873cc097f7a981c570",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*48)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dx_s3": {
            "module": "cake_lm_head_loss_698bb599ab0d988b560e",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_698bb599ab0d988b560e_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_698bb599ab0d988b560e_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "ee4d8df614ba16084993ab0c9ce25727384485b1d6ab5520f0fc97d60866b4f0",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*72)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dx_s4": {
            "module": "cake_lm_head_loss_b9eee151a0ed95d66320",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_b9eee151a0ed95d66320_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_b9eee151a0ed95d66320_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "5a88c074bc3f4b8e611345ad9777c5b8ef60b6518d1aba61526d96aef663470b",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*96)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "gemm_dw_acc": {
            "module": "cake_lm_head_loss_310d7c7d16f409e18c02",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_310d7c7d16f409e18c02_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_310d7c7d16f409e18c02_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["tma_buffer", "A"],
                ["tma_buffer", "B"],
                ["tma_buffer", "C"],
                ["buffer", "STATS_OUT"],
                ["parameter", "M"],
                ["parameter", "m_tiles"],
                ["parameter", "k_iters"],
                ["parameter", "first_chunk"],
                ["buffer", "WS"],
                ["parameter", "ws_slab"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "554b296002dbd0f94fda27cff3e7198c874f1db09dd2c29cf0827a1c79d4c3cd",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, m_tiles//2*24)*2", 1, 1],
            "launch": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
        },
        "scale_cast_bf16": {
            "module": "cake_lm_head_loss_c98313fd7f02b0482483",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_c98313fd7f02b0482483_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_c98313fd7f02b0482483_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "acc"],
                ["buffer", "g"],
                ["buffer", "out"],
                ["parameter", "num_vecs"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "2b7c0982cde8dfc7c8affac27f5d812c633fcc34a65d2a3cd41e8471d4571e90",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, min(num_vecs/2048, 65535))", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "scale_cast_f32": {
            "module": "cake_lm_head_loss_8ea3692f8bbd9e79b4d6",
            "sources": [
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_8ea3692f8bbd9e79b4d6_kernel.cu",
                "cake_lm_head_loss/sm_103a/cake_lm_head_loss_8ea3692f8bbd9e79b4d6_binding.cu",
            ],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": [
                ["buffer", "acc"],
                ["buffer", "g"],
                ["buffer", "out"],
                ["parameter", "num_vecs"],
                ["grid", "grid_x"],
                ["grid", "grid_y"],
                ["grid", "grid_z"],
            ],
            "closure_sha256": "e06e926da9e457b15f3485090dbb1741bfacf2a77eb8398bd7de4a63522e0900",
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["max(1, min(num_vecs/2048, 65535))", 1, 1],
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "f41f604111968dd454099418cb457e20bacba2616b8355da4bad38e0803fe149",
    },
}

# Kernel stages of one token chunk of the training step, in launch order.
#
# ``gemm_logits``          z_c = bf16(X_c @ W^T) for the chunk's rows plus the
#                          per-(row, vocabulary tile) online (max, sum-exp)
#                          partials of the bf16-rounded logits.
# ``gemm_logits_nostats``  the same GEMM without the statistics (the
#                          recompute of the log-probability backward).
# ``row_finalize``         merges the partials into ``lse`` (every row),
#                          gathers the selected logit, writes ``logp`` (0 on
#                          ignored rows) and, per objective, the per-row
#                          logit-gradient scale ``d`` and loss term.
# ``loss_reduce``          fixed-order sum of the chunk's loss terms into the
#                          loss accumulator (chunk order); the last chunk
#                          writes the finished loss.
# ``row_grad``             dz_c = d_t * (1[v = y_t] - exp(z - lse_t)) in bf16,
#                          in place over ``z_c``; ignored rows become zero.
# ``gemm_dx``              dX_acc[rows] = fp32(dz_c @ W).
# ``gemm_dx_s2`` .. ``_s4``  the same GEMM as 2 / 3 / 4 K-slice work items per output
#                          tile (slice 0 writes dX_acc, slices >= 1 write FP32
#                          workspace slabs the host adds in fixed order); the
#                          host picks the slice count per chunk from its row
#                          count and the SM count.  Optional (a contiguous
#                          prefix may be registered).
# ``gemm_dw_acc``          dW_acc (=|+=) fp32(dz_c^T @ X_c): store on the first
#                          chunk, accumulate afterwards (chunk order = the
#                          reduction order, no atomics).
# ``scale_cast_bf16``      out = bf16(g * acc) over a flat fp32 accumulator
# ``scale_cast_f32``       out = g * acc (fp32) -- the single output cast of
#                          ``dW`` (either) and ``dX`` (bf16) in the backward.
#
# A record registers the subset its program uses; the host refuses an entry
# point whose stages are missing.
STAGES = (
    "gemm_logits",
    "gemm_logits_nostats",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "gemm_dx",
    "gemm_dx_s2",
    "gemm_dx_s3",
    "gemm_dx_s4",
    "gemm_dw_acc",
    "scale_cast_bf16",
    "scale_cast_f32",
)
GEMM_STAGES = ("gemm_logits", "gemm_logits_nostats", "gemm_dx", "gemm_dx_s2", "gemm_dx_s3", "gemm_dx_s4", "gemm_dw_acc")
ROW_STAGES = ("row_finalize", "loss_reduce", "row_grad", "scale_cast_bf16", "scale_cast_f32")
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 only.)"""
    return arch in ARCH_NVCC_FLAGS


def select_module(arch: str) -> str:
    """Return the registered module name for ``arch``."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if len(names) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one chunked LM-head program: {names}"
        )
    if not names:
        raise NotImplementedError(
            f"The generated chunked LM-head + loss program for {arch} is not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#5680)"
        )
    return names[0]


def registered_stages(name: str) -> tuple[str, ...]:
    """Stages a record registers, in launch order."""
    present = tuple(stage for stage in STAGES if stage in MODULES[name])
    declared = tuple(MODULES[name].get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record {name!r} declares stages {declared} but carries {present}"
        )
    return present


def _header_dirs():
    installed = [jit_env.FLASHINFER_CSRC_DIR, jit_env.FLASHINFER_INCLUDE_DIR]
    if (installed[0] / "tvm_ffi_utils.h").is_file() and (
        installed[1] / "flashinfer/layout.cuh"
    ).is_file():
        return installed
    checkout = Path(__file__).resolve().parents[3]
    source = [checkout / "csrc", checkout / "include"]
    if (source[0] / "tvm_ffi_utils.h").is_file() and (
        source[1] / "flashinfer/layout.cuh"
    ).is_file():
        return source
    raise FileNotFoundError("FlashInfer binding headers were not found")


@functools.cache
def gen_cake_lm_head_loss_module(name: str, stage: str):
    record = MODULES[name]
    if not toolchain_supports(record["arch"]):
        raise RuntimeError(
            f"generated chunked LM-head program {name!r} targets {record['arch']}, "
            "which this checkout cannot compile"
        )
    physical = record[stage]
    root = Path(__file__).resolve().parent / "csrc"
    sources = [root / relative for relative in physical["sources"]]
    return gen_jit_spec(
        name=f"{name}_{stage}_" + physical["closure_sha256"][:20],
        sources=sources,
        extra_cuda_cflags=[
            *ARCH_NVCC_FLAGS[record["arch"]],
            *physical["compile_flags"],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_lm_head_loss_module(name: str, stage: str):
    return gen_cake_lm_head_loss_module(name, stage).build_and_load()
