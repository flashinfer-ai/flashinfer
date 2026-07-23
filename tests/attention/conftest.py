"""
Conftest file for attention tests.

Current features:
1.  Bulk-precompile XQA decode kernels before the test file
``test_trtllm_gen_attention_decode_xqa.py`` runs. Bulk-precompile avoids
sequential compilation of kernels that can occupy 95% of the test's wall time.
"""

import os

import pytest
import torch

_XQA_FILE = "test_trtllm_gen_attention_decode_xqa.py"

_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn}


def _executed_xqa_uri(p):
    """Return gen_xqa_module kwargs for an executed xqa case, or None if the
    case is skipped by the backend guards (mirrors the test's pytest.skip logic)."""
    q, kv, o = p["q_dtype"], p["kv_dtype"], p["o_dtype"]
    if p.get("non_contiguous_query"):
        return None  # xqa: no non-contiguous query
    if p.get("skips_softmax"):
        return None  # xqa: no skips_softmax
    if not p.get("uses_shared_paged_kv_idx", True):
        return None  # xqa: needs shared page indices
    if q == "fp8":
        return None  # xqa: only fp16/bf16 query
    if o == "nvfp4" or kv == "nvfp4":
        return None  # xqa: unsupported
    return dict(
        input_dtype=_DT[q],
        kv_cache_dtype=_DT[kv],
        page_size=p["page_size"],
        head_dim=p["head_dim"],
        head_group_ratio=p["head_grp_size"],
        use_sliding_window=(p["window_left"] != -1),
        output_dtype=_DT[o],
        q_seq_len=p["q_len_per_req"],
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    # trylast: run after -k/-m deselection so we only build kernels for cases
    # that will actually execute.
    xqa_items = [it for it in items if it.nodeid.split("::")[0].endswith(_XQA_FILE)]
    if not xqa_items:
        return

    from flashinfer.jit.core import build_jit_specs
    from flashinfer.jit.xqa import gen_xqa_module

    specs = {}
    for it in xqa_items:
        cs = getattr(it, "callspec", None)
        if cs is None:
            continue
        kwargs = _executed_xqa_uri(cs.params)
        if kwargs is None:
            continue
        try:
            spec = gen_xqa_module(**kwargs)
        except Exception:
            continue  # unsupported config the test would also skip
        specs[spec.name] = spec

    specs = list(specs.values())
    if not specs:
        return

    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter:
        reporter.write_line(
            f"[xqa-prebuild] compiling {len(specs)} XQA kernels in parallel..."
        )

    # One ninja graph, built in parallel; skip_prebuilt reuses anything already AOT'd.
    build_jit_specs(specs, verbose=False)

    # Stage each freshly-built .so into the AOT path so build_and_load loads it
    # directly (no per-module ninja dependency scan at test time).
    staged = 0
    for s in specs:
        src = s.jit_library_path
        dst = s.aot_path
        if dst.exists() or not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)  # hardlink: instant, same filesystem
        except OSError:
            import shutil

            shutil.copy2(src, dst)
        staged += 1

    if reporter:
        reporter.write_line(
            f"[xqa-prebuild] staged {staged} kernels into AOT dir; test will load-only."
        )
