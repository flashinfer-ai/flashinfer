"""
Conftest file for attention tests.

Current features:
1.  Bulk-precompile JIT kernels before heavy test files run. Sequential
first-use compilation inside a test session can occupy 85-95% of a file's
wall time (measured on H100 CI: test_trtllm_gen_attention_decode_xqa.py,
test_hopper_fp8_attention.py, test_fmha_v2_prefill.py,
test_batch_prefill_kernels.py). For each registered file we derive the set
of JitSpecs its collected test cases will need (mirroring the tests' own
skip guards), compile them as one parallel ninja graph, and stage the
resulting .so files into the AOT dir so test-time loads skip the
per-module ninja dependency scan entirely.
"""

import contextlib
import logging
import os

import pytest
import torch

logger = logging.getLogger(__name__)

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


def _xqa_specs(items):
    from flashinfer.jit.xqa import gen_xqa_module

    specs = {}
    for it in items:
        cs = getattr(it, "callspec", None)
        if cs is None:
            continue
        kwargs = _executed_xqa_uri(cs.params)
        if kwargs is None:
            continue
        try:
            spec = gen_xqa_module(**kwargs)
        except ValueError as e:
            # Per-config unsupported (bad dtype/page_size/head_dim): skip just
            # this config and keep prebuilding the rest.
            logger.debug("prebuild: skipping unsupported xqa config %s: %s", kwargs, e)
            continue
        specs[spec.name] = spec
    return list(specs.values())


def _hopper_fp8_specs(items):
    """JitSpecs for test_hopper_fp8_attention.py.

    All tests in the file skip on non-SM90A. Module URIs depend only on
    (backend, dtypes, head_dim); seq_len/batch_size/causal/M/N/R/C/num_heads/
    page_size/scale_type do not affect them. The BatchDecode wrapper with
    use_tensor_cores=True routes to the batch *prefill* module, so no decode
    module is needed.
    """
    from flashinfer.utils import is_sm90a_supported

    if not is_sm90a_supported(torch.device("cuda")):
        return []

    from flashinfer.prefill import gen_batch_prefill_module, gen_single_prefill_module
    from flashinfer.quantization import gen_quantization_module

    H, I = torch.half, torch.int32
    specs = {}

    def add(spec):
        specs[spec.name] = spec

    # Dedup on the parameter key BEFORE invoking any generator: spec
    # construction is not free (jinja render + source writes), so it must run
    # once per unique module, not once per collected test item.
    single_keys = set()  # (backend, dtype, head_dim)
    batch_keys = set()  # (dtype, head_dim)
    need_quantization = False
    for it in items:
        cs = getattr(it, "callspec", None)
        if cs is None:
            continue
        p = cs.params
        fn = it.name.split("[")[0]
        hd = p.get("head_dim", 128)
        d = p.get("dtype", torch.float8_e4m3fn)
        if fn == "test_single_prefill":
            # fp16 fa3 reference + fp8 fa3 kernel
            single_keys.add(("fa3", H, hd))
            single_keys.add(("fa3", d, hd))
        elif fn == "test_block_sparse_attention":
            # fa2 custom-mask reference (+ packbits) + fp8 fa3 sparse wrapper
            need_quantization = True
            single_keys.add(("fa2", H, hd))
            batch_keys.add((d, hd))
        else:
            # batch prefill ragged/paged/gqa, tensor-core decode, scale types:
            # fp16 fa3 reference + fp8 fa3 kernel share the batch prefill module
            batch_keys.add((H, hd))
            batch_keys.add((d, hd))
    for backend, d, hd in sorted(single_keys, key=str):
        add(gen_single_prefill_module(backend, d, d, H, hd, hd, 0, False, False, False))
    for d, hd in sorted(batch_keys, key=str):
        add(gen_batch_prefill_module("fa3", d, d, H, I, hd, hd, 0, False, False, False))
    if need_quantization:
        add(gen_quantization_module())
    return list(specs.values())


def _fmha_v2_specs(items):
    """JitSpecs for test_fmha_v2_prefill.py.

    gen_fmha_v2_module URIs depend only on (input_layout, q dtype, o dtype);
    all mask/window/softcap/skip-softmax kernel variants live inside one spec.
    The attention-sink tests additionally compile an fa2/fa3 customize batch
    prefill module (the AttentionSink variant) as reference.
    """
    from flashinfer.jit import gen_customize_batch_prefill_module, gen_fmha_v2_module
    from flashinfer.jit.attention.variants import attention_sink_decl
    from flashinfer.jit.utils import filename_safe_dtype_map
    from flashinfer.utils import is_sm12x_supported, is_sm90a_supported

    device = torch.device("cuda")
    sm90 = is_sm90a_supported(device)
    sm12x = is_sm12x_supported(device)
    if not sm90 and not sm12x:
        return []  # every test in the file skips: FMHA v2 needs SM90a or SM12x

    specs = {}

    def add(spec):
        specs[spec.name] = spec

    # Dedup on the parameter key BEFORE invoking any generator:
    # gen_fmha_v2_module runs a heavyweight source-codegen pass, so it must be
    # called once per unique (layout, dtype, o_dtype), not once per collected
    # test item (~2.4k items would grind for many minutes).
    fmha_keys = set()  # (layout, dtype, o_dtype-or-None)
    sink_keys = set()  # (dtype, use_swa, head_dim)
    need_sm120_module = False
    for it in items:
        cs = getattr(it, "callspec", None)
        if cs is None:
            continue
        p = cs.params
        fn = it.name.split("[")[0]
        if fn == "test_fmha_v2_prefill_deepseek":
            need_sm120_module = sm12x
            continue
        if fn == "test_trtllm_fmha_v2_prefill_sm120_large_head_dim" and not sm12x:
            continue  # test skips on non-SM12x
        dtype = p.get("dtype", torch.float16)
        o_dtype = p.get("o_dtype")
        layout = p.get(
            "input_layout", "SEPARATE_Q_K_V"
        )  # sinks test has no layout param
        is_fp8 = dtype == torch.float8_e4m3fn
        if not sm90 and (is_fp8 or layout == "SEPARATE_Q_K_V"):
            continue  # mirrored from the test's SM12x skip guards
        fmha_keys.add((layout, dtype, o_dtype if is_fp8 else None))
        if fn == "test_trtllm_fmha_v2_prefill_attention_sinks":
            sink_keys.add(
                (dtype, p.get("window_left", -1) >= 0, p.get("head_dim", 128))
            )

    if need_sm120_module:
        from flashinfer.jit import gen_trtllm_fmha_v2_sm120_module

        add(gen_trtllm_fmha_v2_sm120_module())
    for layout, dtype, o_dtype in sorted(fmha_keys, key=str):
        add(gen_fmha_v2_module(layout, dtype, o_dtype))
    for dtype, use_swa, head_dim in sorted(sink_keys, key=str):
        # Reference path: BatchAttentionWithAttentionSinkWrapper(backend="fa3")
        # (see flashinfer/attention/_core.py jit_args construction).
        add(
            gen_customize_batch_prefill_module(
                "fa3",
                f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}_swa_{use_swa}_fa3",
                dtype,  # dtype_q
                dtype,  # dtype_kv
                dtype,  # dtype_o
                torch.int32,  # idtype
                head_dim,
                head_dim,
                ["sink"],
                ["float"],
                ["sm_scale"],
                ["double"],
                "AttentionSink",
                attention_sink_decl["fa3"],
                pos_encoding_mode=0,
                use_sliding_window=use_swa,
                use_fp16_qk_reduction=False,
            )
        )
    return list(specs.values())


def _batch_prefill_specs(items):
    """JitSpecs for test_batch_prefill_kernels.py.

    The file's module-scoped warmup_jit fixture covers the main fp16/fp8-kv
    grids for head_dim 128/256; this collector adds the gap modules measured
    as serial first-use compiles (head_dim 64/512, ALIBI, soft-cap, NVFP4
    uint8-kv, bf16 references, cta-tile probes) and also includes the
    fixture's grid so everything gets staged into the AOT dir (making the
    fixture itself a fast no-op via skip_prebuilt).
    """
    from flashinfer.prefill import gen_batch_prefill_module, gen_single_prefill_module
    from flashinfer.utils import get_compute_capability, is_sm90a_supported

    device = torch.device("cuda")
    cc_major = get_compute_capability(device)[0]
    sm90 = is_sm90a_supported(device)

    H, B, I, U8 = torch.half, torch.bfloat16, torch.int32, torch.uint8
    specs = {}

    def add(spec):
        specs[spec.name] = spec

    def bp(*args, **kwargs):
        add(gen_batch_prefill_module(*args, **kwargs))

    def sp(*args, **kwargs):
        add(gen_single_prefill_module(*args, **kwargs))

    fns = {it.name.split("[")[0] for it in items if getattr(it, "callspec", True)}

    main_grid_fns = {
        "test_batch_prefill_with_paged_kv_cache",
        "test_batch_prefill_with_tuple_paged_kv_cache",
        "test_batch_prefill_with_paged_kv_cache_custom_mask",
        "test_batch_prefill_with_ragged_kv_cache",
    }
    if fns & main_grid_fns:
        # head_dim 64 cases (fixture only builds 128/256); references use
        # single_prefill with backend auto (fa3 on SM90 for pos_encoding 0).
        for p in (0, 1):
            bp("fa2", H, H, H, I, 64, 64, p, False, False, False)
            sp("fa2", H, H, H, 64, 64, p, False, False, False)
        if sm90:
            bp("fa3", H, H, H, I, 64, 64, 0, False, False, False)
            sp("fa3", H, H, H, 64, 64, 0, False, False, False)
        # Fixture grid, included here so it gets AOT-staged.
        try:
            from tests.test_helpers.jit_utils import gen_prefill_attention_modules

            for spec in gen_prefill_attention_modules(
                [torch.float16],
                [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2],
                [128, 256],
                [0, 1],
                [False],
                [False],
                [False],
            ):
                add(spec)
        except ImportError:  # pragma: no cover - fixture still builds these
            pass

    # 16-bit FA2 head_dim > 256 needs SM80+ (tests use
    # skip_if_head_dim_unsupported); don't build those modules on older GPUs.
    hd512_ok = cc_major >= 8

    if hd512_ok and fns & {
        "test_batch_prefill_with_paged_kv_cache_head_dim_512",
        "test_batch_prefill_with_ragged_kv_cache_head_dim_512",
        "test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides",
    }:
        # head_dim 512 forces fa2 for both wrapper and reference.
        for p in (0, 1):
            bp("fa2", H, H, H, I, 512, 512, p, False, False, False)
            sp("fa2", H, H, H, 512, 512, p, False, False, False)

    if "test_batch_prefill_with_ragged_kv_cache_custom_mask" in fns:
        # custom mask forces fa2; ALIBI (=2) and soft-cap are fixture gaps.
        for hd in (128, 256):
            for p in (0, 1, 2):
                for softcap in (False, True):
                    bp("fa2", H, H, H, I, hd, hd, p, False, softcap, False)

    if "test_batch_prefill_with_paged_kv_cache_multi_item_scoring" in fns:
        for softcap in (False, True):
            bp("fa2", H, H, H, I, 128, 128, 1, False, softcap, False)
            sp("fa2", H, H, H, 128, 128, 1, False, softcap, False)

    nvfp4_fns = {
        "test_batch_prefill_with_paged_kv_cache_nvfp4",
        "test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views",
        "test_batch_prefill_with_ragged_kv_cache_nvfp4",
        "test_batch_prefill_with_paged_kv_cache_nvfp4_large_head",
        "test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16",
        "test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head",
        "test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16",
        "test_batch_prefill_with_ragged_kv_cache_nvfp4_large_head",
        "test_batch_prefill_with_ragged_kv_cache_nvfp4_rope_large_head",
    }
    if fns & nvfp4_fns:
        # NVFP4 KV is uint8 (fp4x2_e2m1) and always fa2; references run on the
        # dequantized dtype (bf16 references are not in the fixture grid).
        for q in (H, B):
            bp("fa2", q, U8, q, I, 128, 128, 0, False, False, False)
            sp("fa2", q, q, q, 128, 128, 0, False, False, False)
            if sm90:
                sp("fa3", q, q, q, 128, 128, 0, False, False, False)
            if hd512_ok:
                for p in (0, 1):
                    bp("fa2", q, U8, q, I, 512, 512, p, False, False, False)
                    sp("fa2", q, q, q, 512, 512, p, False, False, False)

    if (
        "test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric" in fns
        and cc_major >= 10
    ):
        for hd_qk, hd_vo in ((512, 256), (256, 128)):
            bp("fa2", B, U8, B, I, hd_qk, hd_vo, 0, False, False, False)

    if "test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256" in fns and hd512_ok:
        bp("fa2", H, H, H, I, 448, 256, 0, False, False, False)
        if cc_major >= 10:
            bp("fa2", H, torch.float8_e4m3fn, H, I, 448, 256, 0, False, False, False)

    return list(specs.values())


# Registered heavy files: test file basename -> callable(items) -> list[JitSpec].
_PREBUILD_SPEC_COLLECTORS = {
    "test_trtllm_gen_attention_decode_xqa.py": _xqa_specs,
    "test_hopper_fp8_attention.py": _hopper_fp8_specs,
    "test_fmha_v2_prefill.py": _fmha_v2_specs,
    "test_batch_prefill_kernels.py": _batch_prefill_specs,
}


def _stage_into_aot(specs):
    """Hardlink (or copy) each freshly-built .so into the AOT path so
    build_and_load loads it directly (no per-module ninja dependency scan at
    test time). Returns the number of staged modules."""
    staged = 0
    for s in specs:
        src = s.jit_library_path
        dst = s.aot_path
        if dst.exists() or not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)  # hardlink: atomic + instant, same filesystem
        except FileExistsError:
            continue  # another worker staged it first; nothing to do
        except OSError:
            # Cross-filesystem (or link unsupported): copy to a temp file in
            # the destination dir, then atomically rename into place so a
            # concurrent worker never observes a partially written .so.
            import shutil
            import tempfile

            fd, tmp = tempfile.mkstemp(dir=str(dst.parent), suffix=".tmp")
            os.close(fd)
            try:
                shutil.copy2(src, tmp)
                os.replace(tmp, dst)  # atomic on the same filesystem
            except OSError:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                continue
        staged += 1
    return staged


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    # trylast: run after -k/-m deselection so we only build kernels for cases
    # that will actually execute.

    # --collect-only enumerates tests without running them, so don't pre-build.
    if config.getoption("--collect-only"):
        return

    from flashinfer.jit.core import build_jit_specs

    reporter = config.pluginmanager.getplugin("terminalreporter")

    for fname, collect_specs in _PREBUILD_SPEC_COLLECTORS.items():
        file_items = [it for it in items if it.nodeid.split("::")[0].endswith(fname)]
        if not file_items:
            continue

        # The whole prebuild is a best-effort optimization. Any failure here
        # must not abort collection for the entire suite.
        try:
            specs = collect_specs(file_items)
            if not specs:
                continue

            if reporter:
                reporter.write_line(
                    f"[jit-prebuild] {fname}: compiling {len(specs)} kernels in parallel..."
                )

            # One ninja graph, built in parallel; skip_prebuilt reuses anything
            # already AOT'd.
            build_jit_specs(specs, verbose=False)

            staged = _stage_into_aot(specs)
        except Exception as e:
            if reporter:
                reporter.write_line(
                    f"[jit-prebuild] {fname}: prebuild failed ({e!r}); falling "
                    "back to per-test JIT compilation."
                )
            continue

        if reporter:
            reporter.write_line(
                f"[jit-prebuild] {fname}: staged {staged} kernels into AOT dir; "
                "tests will load-only."
            )
