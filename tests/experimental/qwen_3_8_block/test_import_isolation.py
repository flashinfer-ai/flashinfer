# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen 3.8 package is as self-contained as the Qwen 3.5 one it builds on: importing the API, probing support and declaring the
decode form never imports the ``cudnn`` package (blocked here); ``import flashinfer.qwen_3_8_block`` loads neither the DSL nor the
vendored tree nor the kernels; the support module imports no kernel module; the closure modules the 3.8 route added import with
``cudnn`` blocked."""

import os
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parents[3]


def _run_isolated(code: str) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(_REPO_ROOT), env.get("PYTHONPATH")))
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    if proc.returncode:
        pytest.fail(
            f"isolated interpreter failed (rc {proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    return proc.stdout


_BLOCK_CUDNN = """
import sys, importlib.abc

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "cudnn" or name.startswith("cudnn."):
            raise ImportError(f"the cuDNN Frontend package is blocked in this test: {name}")
        return None

sys.meta_path.insert(0, _Block())
sys.modules["cudnn"] = None
"""


def test_core_import_is_cheap_and_loads_no_backend():
    out = _run_isolated(
        _BLOCK_CUDNN
        + """
import flashinfer
before = set(sys.modules)
import flashinfer.qwen_3_8_block as q
added = set(sys.modules) - before
assert not any(k == "cutlass" or k.startswith("cutlass.") for k in added), "importing the core module must not load the DSL"
assert not any(k.startswith("flashinfer.experimental.qwen_3_8_block") for k in added), "importing the core module must not load the backend"
assert not any(k.startswith("flashinfer.experimental.qwen_3_5_block") for k in added), "nor the sibling's"
for name in ("qwen_3_8_block_bf16_decode", "prepare_qwen_3_8_block_bf16", "qwen_3_8_block_support", "qwen_3_8_block_pack_weights", "qwen_3_8_block_rope_position_tables", "qwen_3_8_block_rope_tables"):
    assert getattr(q, name).is_experimental is True, name
assert q.Qwen38BlockWrapper.is_experimental is True and q.Qwen38BlockWrapper.plan.is_experimental and q.Qwen38BlockWrapper.run.is_experimental
import torch
g = q.Qwen38BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
sup = q.qwen_3_8_block_support(torch.device("cpu"), g)
assert sup.served is False and sup.source.commit.startswith("977757763")
# the probe on a CPU host reads no vendored tree and imports no kernel module (FlashInfer's own cute_dsl availability probe may load the
# DSL's Python frontend: the Qwen 3.5 package's rule, not this package's)
assert not any(k.startswith("flashinfer.experimental.qwen_3_5_block._cudnn_frost") for k in sys.modules), "the probe on CPU reads no vendored tree"
assert not any(".kernels." in k for k in sys.modules if k.startswith("flashinfer.experimental")), "the probe on CPU imports no kernel module"
assert "cudnn" not in sys.modules or sys.modules["cudnn"] is None
print("OK", sorted(k for k in sys.modules if k.startswith("flashinfer.experimental.qwen_3_8_block")))
"""
    )
    assert out.startswith("OK")


def test_support_module_stays_free_of_the_kernels_and_the_dsl_until_the_vendored_read():
    """``import flashinfer.experimental.qwen_3_8_block.support`` loads no vendored module (the Qwen 3.5 support module, FlashInfer's probes
    and torch only); the tripwire ``check_vendored_rules`` is what imports the tree, inside the vendored read."""
    out = _run_isolated(
        _BLOCK_CUDNN
        + """
import flashinfer
from flashinfer.experimental.qwen_3_8_block import support
mods = sorted(k for k in sys.modules if (".kernels." in k and k.startswith("flashinfer.experimental")) or "_cudnn_frost" in k)
assert not mods, mods
assert support.INDEX_HEAD_DIM == 128 and support.INDEX_QUERY_ROWS_MAX == 64 and support.INDEX_STATE_ROWS_EXTRA == 3
assert support.INDEX_TOP_K_RANGE == (4, 512, 4) and support.PAGE_SIZES == (16, 32, 64, 128) and support.DECODE_CAPTURE_ROWS_MAX == 128
print("OK")
"""
    )
    assert out.startswith("OK")


def test_vendored_tree_declares_the_decode_form_without_cudnn():
    """With the DSL importable: the Qwen 3.8 decode declaration (the in-block indexer over the pools, the two paged side caches, the four
    input-contract arms) builds on host tensors with the ``cudnn`` package blocked -- the closure the 3.8 route added is complete; the
    envelope's decode record carries the indexer rows the support module mirrors."""
    pytest.importorskip("cutlass")
    out = _run_isolated(
        _BLOCK_CUDNN
        + """
import torch
from flashinfer.experimental.qwen_3_5_block._cudnn_frost import gated_attention_block as blk
from flashinfer.experimental.qwen_3_8_block import support
G, Q, Fwd = blk.GatedAttentionBlockGeometry, blk.QsaSpec, blk.GatedAttentionBlockFwd
DT, D, ROPE, DM, HQ, HKV, P, DI = torch.bfloat16, 256, 64, 512, 24, 2, 16, 128
g = G(d_model=DM, h_q=HQ, h_kv=HKV, d_head=D, rope_dim=ROPE, causal_bottom_right=True, qsa=Q(top_k=512, index_source="indexer", index_band=True))
mk = lambda *s: torch.empty(*s, dtype=DT)
w = (mk(g.n_qkvg, DM), torch.ones(D, dtype=DT), torch.ones(D, dtype=DT), mk(DM, HQ * D))
kc, vc = torch.zeros(12, HKV, P, D, dtype=DT), torch.zeros(12, HKV, P, D, dtype=DT)
for fused, tables, slots in ((True, True, True), (True, True, False), (False, False, True), (False, False, False)):
    t = (None, None) if tables else (mk(4, 2, ROPE), mk(4, 2, ROPE))
    b = Fwd(mk(4, 2, DM), *w[:3], *t, w[3], mk(4, 2, DM), g, paged_kv_page_size=P, kv_source="paged", sample_k_cache=kc, sample_v_cache=vc,
            sample_block_table=torch.zeros(4, 4, dtype=torch.int32), fuse_norm_rope=False, fuse_gate=True, fuse_prepare=fused,
            sample_index_kv_cache=torch.zeros(12, P // 4, 1, DI, dtype=DT), sample_index_state_cache=torch.zeros(5, 8, 1, DI, dtype=DT),
            sample_index_state_block_table=torch.zeros(4, 1, dtype=torch.int32), index_state_slot_mapping_present=slots, rope_position_tables_only=tables)
    assert b.indexer_decode and b.fuse_prepare is fused and b.qsa_list_form == "per_row", (fused, tables, slots)
env = Fwd.serving_envelope(g, DT, torch.device("cpu"), device_cc=(10, 7))
d = env.decode
assert d.sparse_core and d.index_sources == ("caller", "indexer") and d.index_list_forms == ("per_row",), (d.index_sources, d.index_list_forms)
assert d.index_head_dims == (support.INDEX_HEAD_DIM,) and d.index_query_rows_max == support.INDEX_QUERY_ROWS_MAX
assert d.index_state_rows_extra == support.INDEX_STATE_ROWS_EXTRA and d.index_top_k_range == support.INDEX_TOP_K_RANGE
assert tuple(d.index_page_sizes) == support.INDEX_PAGE_SIZES and tuple(d.kv_cache_dtypes) == (DT,)
assert all(k in d.declines for k in support.DECLINE_KEYS), [k for k in support.DECLINE_KEYS if k not in d.declines]
from flashinfer.cute_dsl.availability import is_rubin_cute_dsl_available
if is_rubin_cute_dsl_available():
    assert env.cutedsl_ok and d.served and d.fuse_prepare_default is True, (env.cutedsl_reason, d.reason)
else:
    assert not env.cutedsl_ok and d.reason == env.cutedsl_reason
leaked = sorted(k for k in sys.modules if (k == "cudnn" and sys.modules[k] is not None) or k.startswith("cudnn."))
assert not leaked, leaked
print("OK")
"""
    )
    assert out.startswith("OK")


def test_the_closure_modules_the_decode_route_added_import_without_cudnn():
    """Every module the 3.8 route added to the shared vendored tree (the block-sparse core and its lean second body, the paged indexer
    kernels and their DSL-free hosts) imports with the ``cudnn`` package blocked and no GPU; the lazily imported paths outside the
    closure (the e4m3 pool's upcast arm, the DSA top-k) are named and stay out."""
    pytest.importorskip("cutlass")
    out = _run_isolated(
        _BLOCK_CUDNN
        + """
import importlib
root = "flashinfer.experimental.qwen_3_5_block._cudnn_frost."
added = ("sdpa.fwd.sparse_gqa_sm107", "sdpa.fwd.kernels.sm107.sparse_d256_f16", "gated_attention_block.kernels.qsa_prepare",
         "gated_attention_block.kernels.qsa_compress", "gated_attention_block.kernels.qsa_score_paged", "QSA.selection_paged",
         "gated_attention_block.kernels.qsa_topk_decode", "gated_attention_block.qsa_topk", "QSA.selection", "QSA", "_tupledict",
         # the 7301fad21 pin: the lean decode tile, the shared visible-set helper it and the tcgen05 core import, the THD origins module
         "sdpa.fwd.kernels.sm107.sparse_lean_d256_f16", "sdpa.fwd.kernels.sm107._sparse_item_common", "sdpa.fwd.kernels.thd_origins",
         # the 977757763 pin: the DSL-free rules module (the tile_form vocabulary, the lean placement rule in CTAs per chunk) the heuristics re-export
         "sdpa.fwd.sparse_tile_rules")
failed, declined = [], []
for name in added:
    try:
        importlib.import_module(root + name)
    except NotImplementedError as exc:
        (declined if "sm_107a" in str(exc) else failed).append(f"{name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
assert not failed, failed
topk = importlib.import_module(root + "gated_attention_block.qsa_topk")
# the one importlib STRING site of the route (the Rule 7 gate names the kernel module): rewritten into the vendored tree
assert topk.QSA_TOPK_DECODE_KERNEL_MODULE == root + "gated_attention_block.kernels.qsa_topk_decode", topk.QSA_TOPK_DECODE_KERNEL_MODULE
m = importlib.import_module(root + "sdpa.fwd.sparse_gqa_sm107")
# the bare `import cudnn` of this module was rewritten to the vendored root: its data_type enum is the shim's
assert m.cudnn.__name__ == root[:-1] and m.cudnn.data_type.BFLOAT16 == 10, (m.cudnn.__name__,)
for absent in ("gated_attention_block.kernels.qsa_kv_upcast", "deepseek_sparse_attention"):
    try:
        importlib.import_module(root + absent)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"{absent} is vendored but excluded by policy")
leaked = sorted(k for k in sys.modules if (k == "cudnn" and sys.modules[k] is not None) or k.startswith("cudnn."))
assert not leaked, leaked
print("OK", len(added), "declined", len(declined))
"""
    )
    assert out.startswith("OK")
