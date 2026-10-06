"""Host checks for efficiency domains, captured modes, pins and compile jobs."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping

DEVICES = (
    DeviceIdentity("nvidia", (12, 1), 48, "GB10"),
    DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000"),
)


def _declarations():
    from b12x.gemm import _tuning as dense
    from b12x.gemm.block_fp8_linear import _tuning as linear
    from b12x.sequence.kda_prefill import _tuning as kda
    from b12x.sequence.gdn_prefill import _tuning as gdn
    from b12x.attention.dense_mla import _tuning as mla
    from b12x.attention.varlen import _tuning as varlen
    from b12x.attention.paged import _tuning as gqa
    from b12x.attention.dsa_indexer import _tuning as indexer

    return (
        (dense, dense.DenseGemmQuery, dict(
            recipe="mxfp8", entry_point="gemm.mm", weight_storage="native",
            output_dtype="bfloat16", batch=1, max_rows=8, in_features=4096,
            out_features=4096, expected_m=8,
        )),
        (linear, linear.BlockFp8LinearQuery, dict(
            max_tokens=8, in_features=4096, out_features=4096,
            source_dtype="bfloat16", output_dtype="bfloat16", output_mode="provided",
        )),
        (kda, kda.KdaPrefillQuery, dict(
            heads=64, head_dim=128, model_dtype="bfloat16", state_dtype="float32",
            qk_l2norm=True, checkpoint_export=True, max_tokens=2048, max_seqs=1,
        )),
        (gdn, gdn.GdnPrefillQuery, dict(
            key_heads=8, value_heads=24, head_dim=128, model_dtype="bfloat16",
            state_dtype="float32", qk_l2norm=True, checkpoint_export=True,
            max_tokens=128, max_seqs=1,
        )),
        (mla, mla.DenseMlaQuery, dict(
            mode="decode", q_dtype="bfloat16", kv_dtype="bfloat16", num_q_heads=16,
            qk_head_dim=576, v_head_dim=512, page_size=64, query_rows=1,
            max_batch=1, cache_tokens=131072, physical_record_width=576,
            window_size=None, use_cuda_graph=True, max_page_table_width=2048,
            num_cache_pages=2048, abi=FrozenMapping(),
        )),
        (varlen, varlen.VarlenAttentionQuery, dict(
            variant="varlen", dtype="bfloat16", causal=False, batch_size=1,
            q_heads=16, kv_heads=16, q_head_dim=64, v_head_dim=64,
            query_rows=9216, kv_rows=9216, max_seqlen_q=9216, max_seqlen_k=9216,
        )),
        (gqa, gqa.GqaQuery, dict(
            device="cuda:0", mode="decode", q_dtype="bfloat16", kv_dtype="bfloat16",
            q_heads=64, kv_heads=8, head_dim_qk=128, head_dim_vo=128, page_size=64,
            kv_cache_layout="separate", batch_size=16, query_len=1,
            cache_tokens=65536, window_left=-1, requested_graph_ctas_per_sm=None,
            requested_max_work_items=None, requested_max_partial_rows=None,
            force_split_kv=None, abi=FrozenMapping(), controls=FrozenMapping(),
        )),
        (indexer, indexer.DsaIndexerQuery, dict(
            source_layout="paged", mode="prefill", dtype="bfloat16", kv_dtype="uint8",
            num_q_heads=32, num_idx_heads=1, max_q_rows=96, max_k_rows=16384,
            top_k=512, page_size=256, score_mode="dsa", shared_page_table=True,
            max_page_table_width=64, route="auto", output_physical_slots=False,
            supertile_k=0, prefill_block_k=256, reserve_paged_logits=False,
            paged_logits_k_rows=0, operands=FrozenMapping(), cache_format="mxfp4",
        )),
    )


@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize("device", DEVICES)
def test_efficiency_mode_is_captured_and_keys_selection(index, device, monkeypatch):
    module, query_type, values = _declarations()[index]
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    ordinary = query_type(**values)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
    exhaustive = query_type(**values)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    assert ordinary.exhaustive is False and exhaustive.exhaustive is True
    assert module.TUNING.encode_query(ordinary) != module.TUNING.encode_query(exhaustive)
    assert module.TUNING.parameter_space(ordinary, device).exhaustive is False
    assert module.TUNING.parameter_space(exhaustive, device).exhaustive is True
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        query_type(**values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("recipe,m,n,k,tm,tn,tk,swap,load", (
    ("nvfp4", 64, 4096, 4096, 64, 64, 128, False, "cpasync"),
    ("nvfp4", 64, 64, 256, 64, 64, 128, True, "cpasync"),
    ("nvfp4", 16, 4096, 4096, 128, 64, 128, False, "tma"),
    ("mxfp8", 8, 4096, 4096, 64, 16, 128, True, "tma"),
    ("nvfp4", 512, 8192, 4096, 64, 64, 512, False, "tma"),
    ("mxfp8", 256, 4096, 4096, 16, 64, 128, False, "tma"),
))
def test_dense_efficiency_can_be_lifted_or_pinned(device, recipe, m, n, k, tm, tn, tk, swap, load):
    from b12x.gemm._tuning import DenseGemmConfig, DenseGemmQuery, TUNING
    query = DenseGemmQuery(recipe=recipe, entry_point="gemm.mm", weight_storage="native",
        output_dtype="bfloat16", batch=1, max_rows=m, in_features=k, out_features=n,
        expected_m=m, exhaustive=False)
    config = DenseGemmConfig(backend="cutedsl", tile_m=tm, tile_n=tn, tile_k=tk,
        swap_ab=swap, load_path=load, split_k_slices=1 if recipe == "mxfp8" else None,
        large_m_unroll=False if recipe == "mxfp8" else None, target_occupancy=None)
    with pytest.raises(ValueError, match="efficiency predicates"):
        TUNING.parameter_space(query, device).validate(asdict(config))
    assert TUNING.lower(replace(query, exhaustive=True), device, asdict(config)) == config
    assert TUNING.configure(query, device=device, override=config).pinned == config
    constrained = replace(query, overrides=FrozenMapping({"mma_tiler_mn": (tm, tn)}))
    TUNING.parameter_space(constrained, device).validate(asdict(config))
    with pytest.raises(ValueError):
        TUNING.lower(replace(query, exhaustive=True), device, asdict(replace(config, tile_m=7)))


@pytest.mark.parametrize("device", DEVICES)
def test_dense_retains_bk64_and_gb10_counterexamples(device):
    from b12x.gemm._tuning import DenseGemmConfig, DenseGemmQuery, TUNING
    for recipe, m, n, k, tm, tn, tk, swap in (
        ("mxfp8", 8, 1792, 5120, 128, 64, 64, False),
        ("mxfp8", 12, 48, 2560, 64, 16, 128, True),
        ("block_fp8", 64, 36864, 1024, 16, 128, 128, False),
        ("nvfp4", 16, 28672, 512, 128, 128, 128, False),
        ("nvfp4", 512, 256, 4096, 64, 64, 512, False),
    ):
        query = DenseGemmQuery(recipe=recipe, entry_point="gemm.mm", weight_storage="native",
            output_dtype="bfloat16", batch=1, max_rows=m, in_features=k,
            out_features=n, expected_m=m, exhaustive=False)
        fp8 = recipe in ("mxfp8", "block_fp8")
        config = DenseGemmConfig(backend="cutedsl", tile_m=tm, tile_n=tn, tile_k=tk,
            swap_ab=swap, load_path="tma", split_k_slices=1 if fp8 else None,
            large_m_unroll=False if fp8 else None, target_occupancy=None)
        assert TUNING.lower(query, device, asdict(config)) == config


@pytest.mark.parametrize("device", DEVICES)
def test_kda_keeps_non_power_default_and_allows_explicit_window(device):
    module, kind, values = _declarations()[2]
    query = kind(**values, exhaustive=False)
    space = module.TUNING.parameter_space(query, device)
    windows = {p["window_tiles"] for p in space.configurations()}
    assert {1, 16, 41, 64, 128, 129} <= windows
    config = replace(module._default_config(query, device), window_tiles=17)
    with pytest.raises(ValueError, match="efficiency predicates"):
        space.validate(config.to_dict())
    assert module.TUNING.lower(replace(query, exhaustive=True), device, config.to_dict()) == config
    assert module.TUNING.configure(query, device=device, override=config).pinned == config


@pytest.mark.parametrize("device", DEVICES)
def test_mla_keeps_quotients_and_explicit_split_caps(device):
    module, kind, values = _declarations()[4]
    for capacity, selected in ((1024, 171), (2048, 187)):
        query = kind(**(values | {"cache_tokens": capacity * 64}), exhaustive=False)
        space = module.TUNING.parameter_space(query, device)
        splits = {p["max_splits"] for p in space.configurations()}
        assert selected in splits
        expected = {(capacity + d - 1) // d for d in range(1, capacity + 1)}
        assert expected <= splits
        config = module.DenseMlaConfig(max_splits=next(x for x in range(1, capacity + 1) if x not in splits))
        with pytest.raises(ValueError, match="efficiency predicates"):
            space.validate(asdict(config))
        assert module.TUNING.configure(query, device=device, override=config).pinned == config
        assert module.TUNING.lower(replace(query, exhaustive=True), device, asdict(config)) == config
        with pytest.raises(ValueError, match="cannot exceed"):
            module.TUNING.configure(replace(query, exhaustive=True), device=device,
                override=module.DenseMlaConfig(max_splits=capacity + 1))


@pytest.mark.parametrize("device", DEVICES)
def test_varlen_retains_dense_n_axis_and_kernel_legality(device):
    module, kind, values = _declarations()[5]
    query = kind(**values, exhaustive=False)
    for n in (80, 96, 112, 176):
        assert module.TUNING.lower(query, device, {"tile_m": 128, "tile_n": n}).tile_n == n
    choice = {"tile_m": 256, "tile_n": 16}
    with pytest.raises(ValueError, match="efficiency predicates"):
        module.TUNING.lower(query, device, choice)
    exhaustive = replace(query, exhaustive=True)
    config = module.TUNING.lower(exhaustive, device, choice)
    assert module.TUNING.configure(query, device=device, override=config).pinned == config
    with pytest.raises(ValueError, match="unsupported by the production kernel"):
        module.TUNING.lower(exhaustive, device, {"tile_m": 768, "tile_n": 384})


@pytest.mark.parametrize("device", DEVICES)
def test_gqa_retains_residency_endpoints_and_explicit_controls(device):
    module, kind, values = _declarations()[6]
    query = kind(**values, exhaustive=False)
    space = module.TUNING.parameter_space(query, device)
    maximum = max(space.knobs[0].values)
    assignments = list(space.configurations())
    assert {1, 2, 3, 4, 6, 8, maximum} <= {p["graph_ctas_per_sm"] for p in assignments}
    assert {p["force_split_kv"] for p in assignments} == {False, True}
    with pytest.raises(ValueError, match="efficiency predicates"):
        space.validate({"graph_ctas_per_sm": 5, "force_split_kv": False})
    module.TUNING.parameter_space(replace(query, exhaustive=True), device).validate(
        {"graph_ctas_per_sm": 5, "force_split_kv": False})
    explicit = replace(query, requested_graph_ctas_per_sm=5, force_split_kv=True)
    assert [dict(p) for p in module.TUNING.parameter_space(explicit, device).configurations()] == [
        {"graph_ctas_per_sm": 5, "force_split_kv": True}]


def test_dense_mla_compile_jobs_ignore_search_mode(monkeypatch):
    import torch
    from b12x.attention import dense_mla
    from b12x.attention.dense_mla._tuning import TUNING
    caps = dense_mla.Caps(device="cpu", mode="decode", kv_dtype=torch.bfloat16,
        num_q_heads=16, page_size=64, max_total_q=1, max_batch=1,
        max_cache_tokens=1024, max_page_table_width=16, num_cache_pages=16)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    ordinary = dense_mla.plan(caps)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
    exhaustive = dense_mla.plan(caps)
    config = TUNING.default_config(ordinary.query, DEVICES[0])
    device = SimpleNamespace(ordinal=0)
    assert ordinary._compile_jobs(config, device) == exhaustive._compile_jobs(config, device)
    assert ordinary.query.exhaustive is False and exhaustive.query.exhaustive is True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode,heads,rows,expected", (
    ("prefill", 32, 63, 2),
    ("prefill", 32, 64, 1),
    ("prefill", 32, 96, 1),
    ("prefill", 32, 4096, 1),
    ("prefill", 16, 96, 2),
    ("decode", 32, 96, 2),
))
def test_indexer_prefill_score_domain_and_scalar_pin(device, mode, heads, rows, expected):
    module, query_type, values = _declarations()[7]
    query = query_type(**(values | dict(mode=mode, num_q_heads=heads, max_q_rows=rows)), exhaustive=False)
    candidates = module.TUNING.eligible_plan(query, device).candidates
    assert len(candidates) == expected
    if expected == 1:
        assert candidates[0][1].mxfp4_score_kind == "score_tensorcore"
    scalar = module.DsaIndexerConfig(backend="native", mxfp4_score_kind="score")
    assert module.TUNING.configure(query, device=device, override=scalar).pinned == scalar
    exhaustive = replace(query, exhaustive=True)
    assert len(module.TUNING.eligible_plan(exhaustive, device).candidates) == 2
    assert module.TUNING.lower(exhaustive, device, scalar.to_dict()) == scalar
    with pytest.raises(ValueError, match="does not use the FP8 fused merge"):
        module.TUNING.configure(exhaustive, device=device, override=replace(scalar, fused_merge="serial"))


@pytest.mark.parametrize("max_candidates,candidate_topk_blocks", ((0, 0), (16384, 0), (0, 2048)))
def test_indexer_compile_jobs_ignore_search_mode(monkeypatch, max_candidates, candidate_topk_blocks):
    from b12x.attention import dsa_indexer
    from b12x.attention.dsa_indexer._tuning import TUNING
    caps = dsa_indexer.Caps(device="cpu", num_q_heads=32, max_q_rows=96,
        max_page_table_width=64, topk=512, mode="prefill", page_size=256,
        cache_format="mxfp4", max_candidates=max_candidates,
        candidate_topk_blocks=candidate_topk_blocks)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    ordinary = dsa_indexer.plan(caps)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
    exhaustive = dsa_indexer.plan(caps)
    config = TUNING.default_config(ordinary.query, DEVICES[0])
    device = SimpleNamespace(ordinal=0)
    assert ordinary._compile_jobs(config, device) == exhaustive._compile_jobs(config, device)
    assert ordinary.query.exhaustive is False and exhaustive.query.exhaustive is True
