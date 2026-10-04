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
from typing import Any, Optional

from ...jit import env as jit_env
from ...jit.core import gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags
from ...jit.cpp_ext import get_cuda_version

# Explicit target-owned registration of the generated chunked LM-head + loss
# programs, one record per architecture (the kernels take the call geometry
# H / V as launch scalars; one record serves every admissible geometry).  The
# generated-program export writes the compact tables between the two
# ``generated registry``
# markers; ``MODULES`` is their expansion.  A record carries ``arch``, the host
# binding profile ``abi`` (the keyword set its kernels expect, see
# ``cake_backend``), the list of kernel ``stages`` it registers (launch order),
# the ``geometry`` the kernels were built for (see ``cake_backend.Geometry``:
# the vocabulary columns per row-statistics partial, the GEMM row tile and CTA
# pair, the K block of the weight-gradient GEMM, the element vector of the
# cast, the divisibility ``V`` / ``H`` / the row stride of ``X`` must satisfy,
# the label element type and, per GEMM, the cluster width, the work item's
# columns and the default raster height), the program's ``closure_sha256`` and one
# physical entry per stage: ``module`` (the generated program), ``sources``
# (the kernel translation unit and its launch stub, relative to ``csrc/``),
# ``compile_flags``, ``ffi_entry``, ``arg_plan``, ``closure_sha256``,
# ``tma_workspace_bytes``, ``workspace_bytes``, ``grid`` and ``launch``.
#
# Compact form: ``_ARG_PLANS`` and ``_LAUNCHES`` hold the distinct argument
# plans and launch geometries by name; a stage row is
# ``[unit, arg_plan, launch, closure_sha256, grid_x, grid_y, grid_z]`` plus an
# optional trailing mapping of the fields that differ from their defaults
# (``compile_flags`` ``[]``, ``ffi_entry`` ``"run"``, ``tma_workspace_bytes``
# and ``workspace_bytes`` ``0``).  The unit names its two translation units:
# ``csrc/cake_lm_head_loss/cake_lm_head_loss_<unit>_kernel.cu`` (one
# architecture-neutral translation unit per kernel, shared by the records of
# every architecture; the per-architecture lowering sits under ``__CUDA_ARCH__``
# guards) and the launch stub ``..._launch.cu`` that binds the shared launcher
# under ``csrc/cake_lm_head_loss/shim/`` to the kernel.  Do not edit by hand.
# --- generated registry (written by the Cake export; do not edit) ---
_ARG_PLANS: dict[str, list[list[str]]] = {
    "gather_rows_bf16": [["buffer", "x"], ["buffer", "idx_lo"], ["buffer", "count"], ["buffer", "out"], ["parameter", "ld_x_words"], ["parameter", "row_vecs"], ["parameter", "num_rows"], ["parameter", "T"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "gemm_logits": [["tma_buffer", "A"], ["tma_buffer", "B"], ["buffer", "C"], ["buffer", "STATS_OUT"], ["parameter", "M"], ["parameter", "m_tiles"], ["parameter", "k_iters"], ["parameter", "first_chunk"], ["buffer", "WS"], ["parameter", "ws_slab"], ["parameter", "ldc"], ["parameter", "k_slices"], ["parameter", "k_slice_iters"], ["parameter", "group_m"], ["parameter", "group_n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "row_finalize": [["buffer", "stats"], ["buffer", "z"], ["buffer", "labels"], ["buffer", "infer_logp"], ["buffer", "loss_weights"], ["buffer", "d_in"], ["buffer", "lse"], ["buffer", "logp"], ["buffer", "d"], ["buffer", "term"], ["parameter", "rows_c"], ["parameter", "row0"], ["parameter", "V"], ["parameter", "num_tiles"], ["parameter", "mode"], ["parameter", "loss_div"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "loss_reduce": [["buffer", "term"], ["buffer", "loss_acc"], ["buffer", "loss_out"], ["parameter", "rows_c"], ["parameter", "first_chunk"], ["parameter", "last_chunk"], ["parameter", "mode"], ["parameter", "loss_div"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "row_grad": [["buffer", "z"], ["buffer", "labels"], ["buffer", "lse"], ["buffer", "d"], ["parameter", "row0"], ["parameter", "d_off"], ["parameter", "V"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "gemm_dx": [["tma_buffer", "A"], ["tma_buffer", "B"], ["tma_buffer", "C"], ["buffer", "STATS_OUT"], ["parameter", "M"], ["parameter", "m_tiles"], ["parameter", "k_iters"], ["parameter", "first_chunk"], ["buffer", "WS"], ["parameter", "ws_slab"], ["parameter", "ldc"], ["parameter", "k_slices"], ["parameter", "k_slice_iters"], ["parameter", "group_m"], ["parameter", "group_n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "gemm_dx_2": [["tma_buffer", "A"], ["tma_buffer", "B"], ["tma_buffer", "C"], ["buffer", "STATS_OUT"], ["parameter", "M"], ["parameter", "m_tiles"], ["parameter", "k_iters"], ["parameter", "first_chunk"], ["tma_buffer", "WS"], ["parameter", "ws_slab"], ["parameter", "ldc"], ["parameter", "k_slices"], ["parameter", "k_slice_iters"], ["parameter", "group_m"], ["parameter", "group_n"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "slab_sum": [["buffer", "dx"], ["buffer", "ws"], ["parameter", "ws_slab"], ["parameter", "num_vecs"], ["parameter", "n_slabs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "scale_cast_bf16": [["buffer", "acc"], ["buffer", "g"], ["buffer", "out"], ["parameter", "num_vecs"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
    "scale_cast_scatter_bf16": [["buffer", "acc"], ["buffer", "g"], ["buffer", "scan"], ["buffer", "idx_lo"], ["buffer", "out"], ["parameter", "row_vecs"], ["parameter", "num_rows"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
}
_LAUNCHES: dict[str, dict[str, list[int]]] = {
    "b256_c1": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
    "b224_c2": {"block": [224, 1, 1], "cluster": [2, 1, 1]},
    "b128_c1": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
}
_REGISTRY: dict[str, dict[str, Any]] = {
    "cake_lm_head_loss_sm_100a": {
        "arch": "sm_100a",
        "abi": "lm_head_loss_v1",
        "geometry": {"stats_tile": 256, "row_tile": 128, "cta_group": 2, "k_block": 64, "cast_vec": 8, "vocab_multiple": 256, "hidden_multiple": 256, "ld_multiple": 8, "labels_dtype": "int64", "hidden": None, "vocab": None, "logits_cluster_ctas": 2, "dx_cluster_ctas": 2, "dw_cluster_ctas": 2, "logits_item_cols": 256, "dx_item_cols": 512, "dw_item_cols": 512, "logits_group_m": 32, "dx_group_m": 16, "dw_group_m": 8},
        "closure_sha256": "4b29611db786945d4a3feebe397790ea3df0fe509d738969c2addaa0808064a4",
        "stages": {
            "gather_rows_bf16": ["8babf236bbda86bbe13f", "gather_rows_bf16", "b256_c1", "e133292f6aa935803330e3aa9ebec5c9f1c12e806d359d631d216bb16581674c", "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))", 1, 1],
            "gemm_logits": ["00b273c5a38962ee9d08", "gemm_logits", "b224_c2", "31e7d870e968f67accbfe9cf2d0ae9cb00f2c0f5eee0a8b8c55eb72e236db844", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "gemm_logits_mcnt": ["dea74657862f8eaca172", "gemm_logits", "b224_c2", "2ecb5a3520190b7336c52abd3e996dd4ef8fcd901b62cb7c6c2a68a5e70238ca", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "gemm_logits_nostats": ["0051ba2fcabb4de3afa4", "gemm_logits", "b224_c2", "60bf75d28f670365bdfa34cbc67a2d4478e3c10168392b5b672c093011e94fc1", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "gemm_logits_nostats_mcnt": ["e64827f6e1c6ae4a7a3f", "gemm_logits", "b224_c2", "89790cb44afffc35f7f80ac326dea3f643180c348567a18937b473962e060000", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "row_finalize": ["59bf253fd36d24e3a02a", "row_finalize", "b256_c1", "c32275057862be01b41b08043949490e2e2c8b1306e4af6601f046ac056393c5", "rows_c/8", 1, 1],
            "loss_reduce": ["48151c645e612eb5b0b9", "loss_reduce", "b256_c1", "df56e83ae5cce86e74646ce9de196dd04072fe7b681dbe594c12ef351f7e69c1", 1, 1, 1],
            "row_grad": ["1e0b2792d987e6ccf69e", "row_grad", "b128_c1", "593782abf04f125f0df9422f6a8896073856b54214ded3c75e70f93803a5f259", "max(1, V//8/1024)", "rows_c", 1],
            "gemm_dx": ["492ccafa710904c4fdc2", "gemm_dx", "b224_c2", "8dd816a0fab7206e58d0e9b6ba525b3d5b65a533a78d30fad0c4eb13d2fba49e", "max(1, m_tiles//2*(ldc//512))*2", 1, 1],
            "gemm_dx_s": ["f92d92fcce55a5d82479", "gemm_dx_2", "b224_c2", "483c72d0eee932d8091ad1405899e58a6c434e3e09b9b8d65ec183eeaa5894a3", "max(1, m_tiles//2*(ldc//512)*k_slices)*2", 1, 1],
            "gemm_dx_tn256": ["1b119b17d30358410f3c", "gemm_dx", "b224_c2", "fa8dcdc3ec122efbd7070d048be3b41cee0b19a200e631db3cc47141b6741c0c", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dx_s_tn256": ["1e1f9fbcaf2e6b2d1da7", "gemm_dx_2", "b224_c2", "566cea91f961a03e224f1a9afc6df3ad77d2f613c69ca3b2a3c00b513018fb20", "max(1, m_tiles//2*(ldc//512)*k_slices)*4", 1, 1],
            "gemm_dx_st3": ["4b44a4690b790ec43aa7", "gemm_dx", "b224_c2", "00a5e645531a1784e27c55cb524e67b508b5f117999eb7ad33c1055b7b89954f", "max(1, m_tiles//2*(ldc//512))*2", 1, 1],
            "gemm_dx_s_st3": ["34154c354bcd837490ab", "gemm_dx_2", "b224_c2", "3ac0ca505f6cc453939fd7d0d452c7f9ef90d61ec09ed50ad9568b4750c8d22d", "max(1, m_tiles//2*(ldc//512)*k_slices)*2", 1, 1],
            "slab_sum": ["fcb73e342a9efeff71cf", "slab_sum", "b256_c1", "1e005bd929aea8cfe93cf6ed74d342bc35693878d377a3feb407eeac1563803c", "max(1, min(num_vecs/2048, 65535))", 1, 1],
            "gemm_dw_acc": ["a0e3947373eb1b8e1c39", "gemm_logits", "b224_c2", "6ece97669aebae332d940137bd850664a36e22af461b1b6640b45806ea5eba4b", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_acc_gn": ["89ac22cfd82d974eaf92", "gemm_logits", "b224_c2", "e6bf3d84b553c965e577affc817070f436d96c38609ea3646378be04c75842fa", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_cast_bf16": ["bd480eb3887a9ed1a5a5", "gemm_logits", "b224_c2", "fe97673c04b2ceecb75d02e34c7e477923589e140ab5e9bf62defecd71fd3487", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_cast_f32": ["becd4b3abba1b0db2349", "gemm_logits", "b224_c2", "4c0a689a33aa11910766b331a5410021659c53786fc18c65c8704a01c31a10fc", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_cast_f32_gn": ["271a70acf82f0a2501aa", "gemm_logits", "b224_c2", "0c014b51349a31abd73706516b9ce767bf042abc58be5eea1612cf0d2f264915", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "scale_cast_bf16": ["973dc9210174a075d710", "scale_cast_bf16", "b256_c1", "7ef6a9a88743f94bca54512fb38ceb7a02ead175d11a7ab2476dce79ab7160de", "max(1, min(num_vecs/2048, 65535))", 1, 1],
            "scale_cast_f32": ["4ac53e29d72d917205e4", "scale_cast_bf16", "b256_c1", "35aabb345a9690d8de7c1dd94011e189ca5bc33cfbd5f7ccb0f8de9d83701b7c", "max(1, min(num_vecs/2048, 65535))", 1, 1],
            "scale_cast_scatter_bf16": ["1b2d39067a973717f0fb", "scale_cast_scatter_bf16", "b256_c1", "f037f2be6d3be93d75e6e28bf5dbb7e28eb2eb754ca5ac826d96888b99c2fcf1", "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))", 1, 1],
        },
    },
    "cake_lm_head_loss_sm_103a": {
        "arch": "sm_103a",
        "abi": "lm_head_loss_v1",
        "geometry": {"stats_tile": 256, "row_tile": 128, "cta_group": 2, "k_block": 64, "cast_vec": 8, "vocab_multiple": 256, "hidden_multiple": 256, "ld_multiple": 8, "labels_dtype": "int64", "hidden": None, "vocab": None, "logits_cluster_ctas": 2, "dx_cluster_ctas": 2, "dw_cluster_ctas": 2, "logits_item_cols": 256, "dx_item_cols": 512, "dw_item_cols": 512, "logits_group_m": 32, "dx_group_m": 16, "dw_group_m": 2},
        "closure_sha256": "299c35d0de6bb9992e5f74e0ab50e6d24045e03c0427601893c07514fd4d1cd6",
        "stages": {
            "gather_rows_bf16": ["8babf236bbda86bbe13f", "gather_rows_bf16", "b256_c1", "9508fa13120ad6283e09cf05bbe252e0cd3d629f60e47b3b7349dd499f975768", "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))", 1, 1],
            "gemm_logits": ["5fab19dbd297e7c565f5", "gemm_dx", "b224_c2", "08ac9625af81c68e9e25b42568da7536beda6c5d2fa388445c471a7ad11e1ca1", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "gemm_logits_mcnt": ["42d4d8ecadbcd4a473c5", "gemm_dx", "b224_c2", "b893f53f64fcabea9732bc3b6540d551126205d7f1d3bdb5db9a9e54e74a8af8", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "gemm_logits_nostats": ["20f61b6f9af221935f90", "gemm_dx", "b224_c2", "c46239da726d575e33ab71c13eff91f870dfb0db7a1736b2935144ba1904a525", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "gemm_logits_nostats_mcnt": ["4a49f7a07fc5ceb8c4f1", "gemm_dx", "b224_c2", "c3f9f49a48fb34e01539824482191b5a78793e50c616168851021b1b3719728b", "max(1, m_tiles//2*(ldc//256))*2", 1, 1],
            "row_finalize": ["59bf253fd36d24e3a02a", "row_finalize", "b256_c1", "1990f4c1e3f51c89ef754838a10d2f9dde4bfb0b8716ee0b53147e96be7d3f41", "rows_c/8", 1, 1],
            "loss_reduce": ["48151c645e612eb5b0b9", "loss_reduce", "b256_c1", "5fc57e2f87ea52e718f9f3d5275f85d3a089e7417908bcc098c0a844aa32bfdd", 1, 1, 1],
            "row_grad": ["1e0b2792d987e6ccf69e", "row_grad", "b128_c1", "2e4ae8944700e44e52fd30dcf9d93d01dd79fc8dcfbbcfcba4b3c8a74abd7e4f", "max(1, V//8/1024)", "rows_c", 1],
            "gemm_dx": ["492ccafa710904c4fdc2", "gemm_dx", "b224_c2", "fb6cc20a83857879c5ae824fc389c14871fff094acd091b7f260524e66733fd8", "max(1, m_tiles//2*(ldc//512))*2", 1, 1],
            "gemm_dx_s": ["f92d92fcce55a5d82479", "gemm_dx_2", "b224_c2", "48e610a2a7819e495e0c53e3fa0ded343b78eb12c653e7da4b5d0754dce9b95b", "max(1, m_tiles//2*(ldc//512)*k_slices)*2", 1, 1],
            "gemm_dx_tn256": ["1b119b17d30358410f3c", "gemm_dx", "b224_c2", "c5e9deddcbdf25c1e179966d6c5eeea2ca6509095b1ea75e9a3107eed06b9c0d", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dx_s_tn256": ["1e1f9fbcaf2e6b2d1da7", "gemm_dx_2", "b224_c2", "0188bada1ffffdeafdda02606a171d1e568bc7371e7edcbab0810e459bda92ff", "max(1, m_tiles//2*(ldc//512)*k_slices)*4", 1, 1],
            "gemm_dx_st3": ["4b44a4690b790ec43aa7", "gemm_dx", "b224_c2", "b603d587ff628a4e448ebede1e6331409aaf0f6133a83ecaee0126be2ce986b4", "max(1, m_tiles//2*(ldc//512))*2", 1, 1],
            "gemm_dx_s_st3": ["34154c354bcd837490ab", "gemm_dx_2", "b224_c2", "e452f4ecf9ec537bcaec4c49b7b10a252da49e09bba56303f5a8eec7cdb0f7c2", "max(1, m_tiles//2*(ldc//512)*k_slices)*2", 1, 1],
            "slab_sum": ["fcb73e342a9efeff71cf", "slab_sum", "b256_c1", "f733d3d7cfbbc5f5bd0351c71b79d97a9f9d81008472fd8fb963f6ad561e93f3", "max(1, min(num_vecs/2048, 65535))", 1, 1],
            "gemm_dw_acc": ["a0e3947373eb1b8e1c39", "gemm_logits", "b224_c2", "7ae98685938ed3f06d8a92b7a0e3ed8b046db5f5dc991be5fd1b8bd2377655ae", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_acc_gn": ["89ac22cfd82d974eaf92", "gemm_logits", "b224_c2", "3d71a3bffe112d81c250f0c399a413fe9a186af3bab24d548294b4429cd56fe5", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_cast_bf16": ["bd480eb3887a9ed1a5a5", "gemm_logits", "b224_c2", "f0d31c3eb66e7c091e311ac047585632d6d1bd1aef69dd4f0e5bd690dd058287", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_cast_f32": ["becd4b3abba1b0db2349", "gemm_logits", "b224_c2", "ac3a62e5588a5d246ff41f7350e0b1ea1c37b2b0562d6226160ac8679558334b", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "gemm_dw_cast_f32_gn": ["271a70acf82f0a2501aa", "gemm_logits", "b224_c2", "c0fc44a9e3e7ea643f177196e1b6f371093be865135817fb2d8dfc552f603e3e", "max(1, m_tiles//2*(ldc//512))*4", 1, 1],
            "scale_cast_bf16": ["973dc9210174a075d710", "scale_cast_bf16", "b256_c1", "32fd1b809585aa51f34cd65de079ec84bae7b2e1ca3627aec5cf67644b894bf4", "max(1, min(num_vecs/2048, 65535))", 1, 1],
            "scale_cast_f32": ["4ac53e29d72d917205e4", "scale_cast_bf16", "b256_c1", "43777a8dadc72be28de3f03409718ca1f35c9c851f6a28645e9804c9550a1b80", "max(1, min(num_vecs/2048, 65535))", 1, 1],
            "scale_cast_scatter_bf16": ["1b2d39067a973717f0fb", "scale_cast_scatter_bf16", "b256_c1", "ba2c06b91af7afcfab59e981c9e68c12191e737ffdd8f711a58af29092c16c32", "max(1, min(num_rows*row_vecs/2048, num_rows, 65535))", 1, 1],
        },
    },
}
# --- end generated registry ---

STAGE_ROW_FIELDS = (
    "unit",
    "arg_plan",
    "launch",
    "closure_sha256",
    "grid_x",
    "grid_y",
    "grid_z",
)


def _expand_registry(
    registry: dict[str, dict[str, Any]],
    arg_plans: dict[str, list[list[str]]],
    launches: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, Any]]:
    """The per-stage physical records of every program from the compact tables."""
    modules: dict[str, dict[str, Any]] = {}
    for name, compact in registry.items():
        arch = compact["arch"]
        record: dict[str, Any] = {
            "arch": arch,
            "abi": compact["abi"],
            "stages": list(compact["stages"]),
            "geometry": dict(compact["geometry"]),
        }
        for stage, row in compact["stages"].items():
            if len(row) not in (len(STAGE_ROW_FIELDS), len(STAGE_ROW_FIELDS) + 1):
                raise ValueError(f"{name}/{stage}: malformed registry row {row!r}")
            unit, plan, launch, closure, grid_x, grid_y, grid_z = row[
                : len(STAGE_ROW_FIELDS)
            ]
            extra = (
                dict(row[len(STAGE_ROW_FIELDS)])
                if len(row) > len(STAGE_ROW_FIELDS)
                else {}
            )
            physical: dict[str, Any] = {
                "module": f"cake_lm_head_loss_{unit}",
                "sources": [
                    f"cake_lm_head_loss/cake_lm_head_loss_{unit}_kernel.cu",
                    f"cake_lm_head_loss/cake_lm_head_loss_{unit}_launch.cu",
                ],
                "compile_flags": list(extra.pop("compile_flags", [])),
                "ffi_entry": extra.pop("ffi_entry", "run"),
                "arg_plan": [list(item) for item in arg_plans[plan]],
                "closure_sha256": closure,
                "tma_workspace_bytes": int(extra.pop("tma_workspace_bytes", 0)),
                "workspace_bytes": int(extra.pop("workspace_bytes", 0)),
                "grid": [grid_x, grid_y, grid_z],
                "launch": {key: list(value) for key, value in launches[launch].items()},
            }
            physical.update(extra)
            record[stage] = physical
        record["closure_sha256"] = compact["closure_sha256"]
        modules[name] = record
    return modules


MODULES: dict[str, dict[str, Any]] = _expand_registry(_REGISTRY, _ARG_PLANS, _LAUNCHES)

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
# ``gemm_dx_s``            the same GEMM as ``k_slices`` (2 .. 4, a launch scalar)
#                          K-slice work items per output tile (slice 0 writes
#                          dX_acc, slices >= 1 write FP32 workspace slabs the
#                          host adds in fixed order); the host picks the slice
#                          count per chunk from its row count and the SM count.
# ``gemm_dw_acc``          dW_acc (=|+=) fp32(dz_c^T @ X_c): store on the first
#                          chunk, accumulate afterwards (chunk order = the
#                          reduction order, no atomics).
# ``gemm_dw_cast_bf16``    the LAST chunk's dz_c^T @ X_c with the upstream scale
# ``gemm_dw_cast_f32``     and the output cast fused into the epilogue:
#                          dW = cast(g * (dW_acc + tile)) -- or cast(g * tile)
#                          for a one-chunk plan -- run in the backward where
#                          g is known (fuse_dw_cast; bitwise gemm_dw_acc +
#                          scale_cast, one pass over dW fewer).
# ``scale_cast_bf16``      out = bf16(g * acc) over a flat fp32 accumulator
# ``scale_cast_f32``       out = g * acc (fp32) -- the single output cast of
#                          ``dW`` (either) and ``dX`` (bf16) in the backward.
# ``gather_rows_bf16``     chunk 0's row gather of the hidden valid-row count
#                          path: out[r] = X[idx[r]] for r < min(count, num_rows)
#                          with the count read from device memory, exact zeros
#                          after (launched before chunk 0's logits GEMM).
# ``gemm_logits_mcnt``     the logits GEMMs (with / without the statistics)
# ``gemm_logits_nostats_mcnt``  whose valid-row bound is read from device
#                          memory: chunk 0's GEMM of the hidden valid-row count
#                          path, its stores bounded by min(count, M) while M is
#                          the buffer extent min(chunk, T).
#
# A record registers the subset its program uses; the host refuses an entry
# point whose stages are missing.
STAGES = (
    "gather_rows_bf16",
    "gemm_logits",
    "gemm_logits_mcnt",
    "gemm_logits_nostats",
    "gemm_logits_nostats_mcnt",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "gemm_dx",
    "gemm_dx_s",
    "gemm_dx_tn256",
    "gemm_dx_s_tn256",
    "gemm_dx_st3",
    "gemm_dx_s_st3",
    "slab_sum",
    "gemm_dw_acc",
    "gemm_dw_acc_gn",
    "gemm_dw_cast_bf16",
    "gemm_dw_cast_f32",
    "gemm_dw_cast_f32_gn",
    "scale_cast_bf16",
    "scale_cast_f32",
    "scale_cast_scatter_bf16",
)
# The base stages (one per kernel role of the chunk loop).  Every other name in
# ``STAGES`` is a structural FORM of a base GEMM stage -- a distinct kernel body,
# named by the outputs of the host's per-chunk instance rules that select one
# (``cake_backend.stage_variant``): ``_mcnt`` the device-count form of the
# logits GEMMs (the hidden valid-row count path's chunk 0), ``_s`` the K-sliced
# dX GEMM (2 .. 4 K-slice work items per output tile), ``_tn256`` the narrow
# tile and ``_st3`` the 3-deep operand ring of the dX GEMM, ``_gn`` the 2-D
# blocked raster of the weight-gradient accumulate / fp32 fused cast.  The
# call geometry (``H`` / ``V``) and the rules' numeric outputs -- the slice
# count, the raster height ``group_m``, the block width ``group_n`` -- are
# launch scalars of these kernels (``cake_backend.gemm_scalars``), so one
# record per architecture serves every admissible geometry.  A record
# registers the base stages plus the forms its architecture's rules can
# reach.  ``slab_sum`` (the fixed-order slab reduction of a K-sliced dX GEMM)
# and ``scale_cast_scatter_bf16`` (the one-pass scale, cast and scatter of a
# compacted ``dX``) are the fused dX finalize's kernels
# (``cake_backend.DX_FINALIZE_STAGES``); row kernels, geometry-generic.
# ``gather_rows_bf16`` (chunk 0's row gather of the hidden valid-row count path,
# ``cake_backend.HIDDEN_COUNT_STAGE``) is a row kernel too.
BASE_STAGES = (
    "gather_rows_bf16",
    "gemm_logits",
    "gemm_logits_nostats",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "gemm_dx",
    "slab_sum",
    "gemm_dw_acc",
    "gemm_dw_cast_bf16",
    "gemm_dw_cast_f32",
    "scale_cast_bf16",
    "scale_cast_f32",
    "scale_cast_scatter_bf16",
)
GEMM_STAGES = tuple(stage for stage in STAGES if stage.startswith("gemm_"))
ROW_STAGES = (
    "gather_rows_bf16",
    "row_finalize",
    "loss_reduce",
    "row_grad",
    "slab_sum",
    "scale_cast_bf16",
    "scale_cast_f32",
    "scale_cast_scatter_bf16",
)
ARCH_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}


def toolchain_workaround_flags(arch: str) -> list[str]:
    """Extra nvcc flags that work around a toolchain-specific code-generation problem.

    CUDA 13.0's ptxas mis-schedules the sm_103a 2-CTA TMA producer loops of these programs.  At -O3 (and -O2, which
    emits the same code), with a short K loop (<= 16 iterations, i.e. ``chunk_size <= 1024`` for the dW accumulate
    GEMM) the second B-operand ``cp.async.bulk.tensor ... .cta_group::2`` of a stage is rejected by the TMA unit with
    ``cudaErrorIllegalInstruction`` although every operand is legal; at -O1 the structural-form programs' dW cast GEMM
    (the last chunk's ``dz_c^T @ X_c``) faults the same way on every call whose last chunk is partial (``T % chunk_size
    != 0``: the short K loop of the tail chunk).  ptxas -O0 code is correct for both, as is the code of every other
    toolchain (12.9, 13.3, 13.4); -O0 is therefore applied to the sm_103a programs on CUDA 13.0 only.
    """
    if arch != "sm_103a":
        return []
    version = get_cuda_version()
    if (version.major, version.minor) == (13, 0):
        return ["-Xptxas", "-O0"]
    return []


def toolchain_runs_hidden_count(arch: str) -> bool:
    """Does the hidden valid-row count (``cake_backend.hidden_count_eligible``) run on ``arch`` with the nvcc this
    checkout invokes?

    On CUDA 13.0 the sm_103a programs are built at ptxas -O0 (:func:`toolchain_workaround_flags`; -O1 before the
    structural-form programs).  The -O1 code completed
    the focused dW accumulate calls and the test file's shipped call schedule, but with the hidden count's schedule --
    the compaction index, chunk 0's row gather and device-count logits GEMM queued before the count is read back -- the
    test file still reaches a ``cudaErrorIllegalInstruction`` from the cluster launch of the dW accumulate GEMM at its
    shortest K schedule (the one-row tail chunk of a two-chunk call), in every run; with the host count it passes, as
    does the -O0 code with the hidden count.  So on CUDA 13.0 the sm_103a calls take the host-count path (the same
    kernels per row, so bitwise the same outputs; the count and the index are formed on the host before the first
    launch).  Every other architecture / toolchain pair runs the hidden count.  A mitigation of the ptxas 13.0 fault as
    observed, not a root-cause fix.
    """
    if arch != "sm_103a":
        return True
    version = get_cuda_version()
    return (version.major, version.minor) != (13, 0)


def toolchain_supports(arch: str) -> bool:
    """Can the nvcc this checkout invokes emit ``arch``?  (SM100 / SM103 only.)"""
    return arch in ARCH_NVCC_FLAGS


PRIMARY_RECORD = "cake_lm_head_loss_{arch}"  # the architecture's record (geometry-free: H / V are launch scalars)


def record_geometry(record: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """``(hidden, vocab)`` a record's GEMM instances are specialized to (``None`` = any multiple)."""
    geometry = record.get("geometry", {})
    hidden, vocab = geometry.get("hidden"), geometry.get("vocab")
    return (
        None if hidden is None else int(hidden),
        None if vocab is None else int(vocab),
    )


def select_module(
    arch: str, hidden: Optional[int] = None, vocab: Optional[int] = None
) -> str:
    """Return the registered module name for ``arch`` -- the record specialized to the
    ``(hidden, vocab)`` geometry when given (a record pinned to neither accepts any), else
    the sole record of the architecture or its default-geometry record ``PRIMARY_RECORD``."""
    names = [name for name, record in MODULES.items() if record["arch"] == arch]
    if not names:
        raise NotImplementedError(
            f"The generated chunked LM-head + loss program for {arch} is not "
            "registered in this checkout yet (see flashinfer-ai/flashinfer#5680)"
        )
    if hidden is None and vocab is None:
        if len(names) == 1:
            return names[0]
        primary = PRIMARY_RECORD.format(arch=arch)
        if primary in names:
            return primary
        raise NotImplementedError(
            f"{arch} registers more than one chunked LM-head program and no default-geometry "
            f"record {primary!r}: {names}; select by hidden= / vocab="
        )

    def matches(record):
        pinned_h, pinned_v = record_geometry(record)
        return (pinned_h is None or hidden is None or pinned_h == int(hidden)) and (
            pinned_v is None or vocab is None or pinned_v == int(vocab)
        )

    found = [name for name in names if matches(MODULES[name])]
    if len(found) > 1:
        raise NotImplementedError(
            f"{arch} registers more than one chunked LM-head program for H = {hidden}, V = {vocab}: {found}"
        )
    if not found:
        registered = sorted(record_geometry(MODULES[name]) for name in names)
        raise ValueError(
            f"the registered chunked LM-head programs for {arch} are specialized to (H, V) in {registered}, "
            f"got H = {hidden}, V = {vocab}"
        )
    return found[0]


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
            *toolchain_workaround_flags(record["arch"]),
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[root, *[p.parent for p in sources], *_header_dirs()],
        use_fast_math=False,
    )


@functools.cache
def load_cake_lm_head_loss_module(name: str, stage: str):
    return gen_cake_lm_head_loss_module(name, stage).build_and_load()
