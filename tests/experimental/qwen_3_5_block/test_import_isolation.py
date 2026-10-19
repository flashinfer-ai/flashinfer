# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The vendored cuDNN Frontend tree is self-contained: importing the API, probing support and declaring plans never imports the
``cudnn`` package (blocked here), and ``import flashinfer.qwen_3_5_block`` loads neither the DSL nor the vendored tree."""

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
import flashinfer.qwen_3_5_block as q
added = set(sys.modules) - before
assert not any(k == "cutlass" or k.startswith("cutlass.") for k in added), "importing the core module must not load the DSL"
assert not any(k.startswith("flashinfer.experimental.qwen_3_5_block") for k in added), "importing the core module must not load the backend"
for name in ("qwen_3_5_block_bf16_prefill", "qwen_3_5_block_bf16_extend", "qwen_3_5_block_bf16_decode", "prepare_qwen_3_5_block_bf16", "qwen_3_5_block_support", "qwen_3_5_block_pack_weights", "qwen_3_5_block_rope_tables"):
    assert getattr(q, name).is_experimental is True, name
assert q.Qwen35BlockWrapper.is_experimental is True and q.Qwen35BlockWrapper.plan.is_experimental and q.Qwen35BlockWrapper.run.is_experimental
import torch
g = q.Qwen35BlockGeometry(d_model=2560, num_qo_heads=24, num_kv_heads=2)
sup = q.qwen_3_5_block_support(torch.device("cpu"), g)
assert sup.served is False and sup.source.commit.startswith("c9f3118a1")
assert "cudnn" not in sys.modules or sys.modules["cudnn"] is None
print("OK", len([k for k in sys.modules if k.startswith("flashinfer.experimental.qwen_3_5_block")]))
"""
    )
    assert out.startswith("OK")


def test_vendored_tree_declares_the_forms_without_cudnn():
    """With the DSL importable (the vendored tree imports it at module level): the three forms DECLARE on host tensors with the
    ``cudnn`` package blocked -- the vendored closure is complete.  The envelope's verdict follows the installed DSL: served with a
    Rubin-capable DSL, a typed decline naming the missing ``sm_107a`` target without one (the CI images' DSL)."""
    pytest.importorskip("cutlass")
    out = _run_isolated(
        _BLOCK_CUDNN
        + """
import torch
from flashinfer.experimental.qwen_3_5_block._cudnn_frost import gated_attention_block as blk
from flashinfer.experimental.qwen_3_5_block import _sdpa_binder  # the Python SDPA launch binder imports (no cudnn)
G, Fwd = blk.GatedAttentionBlockGeometry, blk.GatedAttentionBlockFwd
DT, D, ROPE, DM, HQ, HKV, P = torch.bfloat16, 256, 64, 512, 24, 2, 16
g = G(d_model=DM, h_q=HQ, h_kv=HKV, d_head=D, rope_dim=ROPE, causal_bottom_right=True)
mk = lambda *s: torch.empty(*s, dtype=DT)
w = (mk(g.n_qkvg, DM), torch.ones(D, dtype=DT), torch.ones(D, dtype=DT), mk(DM, HQ * D))
pools = lambda n: (torch.zeros(n, HKV, P, D, dtype=DT), torch.zeros(n, HKV, P, D, dtype=DT))
Fwd(mk(2, 64, DM), *w[:3], mk(2, 64, ROPE), mk(2, 64, ROPE), w[3], mk(2, 64, DM), g, paged_kv_page_size=P, fuse_norm_rope=True, fuse_gate=True)
kc, vc = pools(9)
Fwd(mk(64, DM), *w[:3], mk(64, ROPE), mk(64, ROPE), w[3], mk(64, DM), g, thd=True, num_sequences=2, max_seq_len=64, cu_seqlens=True, token_capacity=64,
    paged_kv_page_size=P, kv_source="paged", sample_k_cache=kc, sample_v_cache=vc, sample_block_table=torch.zeros(2, 4, dtype=torch.int32), fuse_norm_rope=True, fuse_gate=True)
Fwd(mk(4, 2, DM), *w[:3], mk(4, 2, ROPE), mk(4, 2, ROPE), w[3], mk(4, 2, DM), g, paged_kv_page_size=P, kv_source="paged",
    sample_k_cache=kc, sample_v_cache=vc, sample_block_table=torch.zeros(4, 4, dtype=torch.int32), fuse_norm_rope=True, fuse_gate=True)
env = Fwd.serving_envelope(g, DT, torch.device("cpu"), device_cc=(10, 7))
assert env.extend.served == env.decode.served == env.cutedsl_ok, (env.cutedsl_reason, env.extend.reason, env.decode.reason)
from flashinfer.cute_dsl.availability import is_rubin_cute_dsl_available
if is_rubin_cute_dsl_available():
    assert env.cutedsl_ok and env.extend.served and env.decode.served, (env.extend.reason, env.decode.reason)
else:
    assert not env.cutedsl_ok and ("sm_107a" in env.cutedsl_reason or "nvidia-cutlass-dsl" in env.cutedsl_reason), env.cutedsl_reason
    assert env.extend.reason == env.decode.reason == env.cutedsl_reason, (env.extend.reason, env.decode.reason)
leaked = sorted(k for k in sys.modules if (k == "cudnn" and sys.modules[k] is not None) or k.startswith("cudnn."))
assert not leaked, leaked
print("OK")
"""
    )
    assert out.startswith("OK")
