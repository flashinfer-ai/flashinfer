"""The one place the CAKE backend test suites name their package.

Every CAKE backend suite (``tests/experimental/test_cake_<package>.py``, ``tests/norm/test_cake_rmsnorm_det_train.py``,
``tests/mhc/test_cake_mhc_det_train.py``, ``tests/experimental/test_mok_*.py``) imports its package through :func:`cake_package` and imports nothing else from
flashinfer; the flashinfer public-API tests of a package live in its ``test_cake_<package>_api.py`` sibling.  A build
that vendors a package under another import path runs the backend suite unchanged after pointing the entry of
:data:`PACKAGES` at its own package (flashinfer issue #6553, ask 9).
"""

from __future__ import annotations

import importlib
from types import ModuleType

# suite / package name -> import path of the package
PACKAGES: dict[str, str] = {
    "cake_rmsnorm_det_train": "flashinfer.cake_rmsnorm_det_train_backend",
    "cake_causal_conv1d_det_train": "flashinfer.cake_causal_conv1d_det_train_backend",
    "cake_gated_act_mxfp8": "flashinfer.cake_gated_act_mxfp8_backend",
    "cake_rmsnorm_silu_gate_det_train": "flashinfer.cake_rmsnorm_silu_gate_det_train_backend",
    "cake_rmsnorm_silu_gate_train": "flashinfer.cake_rmsnorm_silu_gate_train_backend",
    "cake_ulysses": "flashinfer.cake_ulysses_backend",
    "cake_fused_qk_rope_append": "flashinfer.cake_fused_qk_rope_append_backend",
    "cake_sage_block_sparse_attention": "flashinfer.cake_sage_block_sparse_attention_backend",
    "cake_mhc_det_train": "flashinfer.cake_mhc_det_train_backend",
    "cake_dsa_train": "flashinfer.experimental.cake_dsa_train",
    "cake_dsa_indexer_det": "flashinfer.experimental.cake_dsa_indexer_det",
    "cake_dsa_indexer": "flashinfer.experimental.cake_dsa_indexer",
    "cake_lm_head_det_loss": "flashinfer.experimental.cake_lm_head_det_loss",
    "dense_projection_det_gemm": "flashinfer.experimental.dense_projection_det_gemm",
    "cake_moe_grouped_det_gemm": "flashinfer.experimental.cake_moe_grouped_det_gemm",
    # the suite name before the deterministic-family rename (compatibility alias)
    "cake_moe_grouped_gemm": "flashinfer.experimental.cake_moe_grouped_det_gemm",
    # the non-deterministic package (deterministic=False); its key carries the namespace because
    # ``cake_moe_grouped_gemm`` stays the det package's compatibility key above
    "cake.cake_moe_grouped_gemm": "flashinfer.experimental.cake.cake_moe_grouped_gemm",
    "cake_mok_bf16": "flashinfer.experimental.cake_mok_bf16",
    "cake_kda_train": "flashinfer.experimental.cake_kda_train",
    "cake_kda_det_train": "flashinfer.experimental.cake_kda_det_train",
    "cake_kda_det_train_kcp": "flashinfer.experimental.cake_kda_det_train_kcp",
    "cake_gdn_det_train_kcp": "flashinfer.experimental.cake_gdn_det_train_kcp",
    "cake_kimi_k3_attn_res": "flashinfer.experimental.cake_kimi_k3_attn_res",
    "cake_stepfun_moe": "flashinfer.experimental.cake_stepfun_moe",
    "cake_nvfp4_svdquant_gemm": "flashinfer.cake_nvfp4_svdquant_gemm_backend",
    "cake_fused_qk_rope_fp8_append": "flashinfer.cake_fused_qk_rope_fp8_append_backend",
    "nvfp4_attention": "flashinfer.experimental.nvfp4_attention",
    "cake_vsa_sm90": "flashinfer.cake_vsa_sm90_backend",
    "cake_gdp": "flashinfer.cake_gdp_backend",
    "deepgemm_fp8_gemm": "flashinfer.experimental.deepgemm_fp8_gemm",
    "deepgemm_batched_gemm": "flashinfer.experimental.deepgemm_batched_gemm",
    "sm110_xqa": "flashinfer.experimental.sm110_xqa",
    "cake_gdn_det_train": "flashinfer.experimental.cake_gdn_det_train",
    "cake.cake_gdn_train": "flashinfer.experimental.cake.cake_gdn_train",
    "cake_gqa_train": "flashinfer.experimental.cake_gqa_train",
    "cake.cake_mhc_train": "flashinfer.experimental.cake.cake_mhc_train_backend",
}
# Compatibility key: suites that name the causal conv1d package ``cake_causal_conv1d_train`` load the det package.
PACKAGES["cake_causal_conv1d_train"] = PACKAGES["cake_causal_conv1d_det_train"]


def cake_package(name: str, submodule: str | None = None) -> ModuleType:
    """The package module of ``name`` (a key of :data:`PACKAGES`), or its ``submodule`` (``"cake_backend"``, ...).

    ``cake_package("cake_lm_head_det_loss")`` is the package itself; ``cake_package("cake_lm_head_det_loss", "cake_backend")``
    is ``<package>.cake_backend``.  Raises ``KeyError`` for a name the table does not list.
    """
    path = PACKAGES[name]
    if submodule:
        path = f"{path}.{submodule}"
    return importlib.import_module(path)
