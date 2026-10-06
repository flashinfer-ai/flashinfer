"""Pool-dependent families key their selections without the pool's size.

The number of state slots or cache pages sizes the caller's pool and differs
between starts whenever the available memory differs, while the fastest
configuration does not depend on it. Two declarations that differ only in
pool size therefore share one selection-cache key.
"""
from dataclasses import replace

from b12x.preparation import DeviceIdentity

IDENTITY = DeviceIdentity("nvidia", (12, 0), 148, "synthetic SM120")
# Operand layouts of a served Qwen3.8 Flash Next QSA layer, as its startup unit declares them.
_QSA_ABI = {"operands": {'compressed_block_table': {'dtype': 'int32', 'strides': (1, 1)}, 'compressed_k_cache': {'dtype': 'bfloat16', 'strides': (820224, 128, 1)}, 'index_k_norm_weight': {'dtype': 'bfloat16', 'strides': (1,)}, 'index_q_norm_weight': {'dtype': 'bfloat16', 'strides': (1,)}, 'index_query': {'dtype': 'bfloat16', 'strides': (640, 128, 1)}, 'main_block_table': {'dtype': 'int32', 'strides': (1, 1)}, 'main_k_cache': {'dtype': 'float8_e4m3fn', 'strides': (729088, 256, 256, 1)}, 'main_v_cache': {'dtype': 'float8_e4m3fn', 'strides': (729088, 256, 256, 1)}, 'raw_index_key': {'dtype': 'bfloat16', 'strides': (640, 1)}, 'raw_interval_start_positions': {'dtype': 'int64', 'strides': (1,)}, 'raw_k_ring': {'dtype': 'bfloat16', 'strides': (1024, 128, 1)}, 'raw_logical_positions': {'dtype': 'int64', 'strides': (8, 1)}, 'raw_rope_positions': {'dtype': 'int64', 'strides': (24, 3, 1)}, 'raw_state_slot_ids': {'dtype': 'int32', 'strides': (1,)}, 'request_ids': {'dtype': 'int32', 'strides': (1,)}, 'rope_cos': {'dtype': 'bfloat16', 'strides': (64, 1)}, 'rope_positions': {'dtype': 'int64', 'strides': (3, 1)}, 'rope_sin': {'dtype': 'bfloat16', 'strides': (64, 1)}}}


def _encoded(contract, query):
    return contract.configure(query, device=IDENTITY, override=None).encoded_query


def test_gdn_prefill_key_ignores_the_state_slot_count():
    from b12x.sequence.gdn_prefill._tuning import TUNING, GdnPrefillQuery

    query = GdnPrefillQuery(
        key_heads=8, value_heads=24, head_dim=128, model_dtype="bfloat16", state_dtype="float32",
        qk_l2norm=True, checkpoint_export=True, max_tokens=16, max_seqs=1, max_state_slots=7,
        null_state_index=0, dt_bias_dtype="bfloat16",
    )
    assert _encoded(TUNING, query) == _encoded(TUNING, replace(query, max_state_slots=26_130))
    assert "max_state_slots" not in _encoded(TUNING, query)
    assert query.to_dict()["max_state_slots"] == 7


def test_gdn_decode_key_ignores_the_state_slot_count():
    from b12x.sequence.gdn_decode._tuning import TUNING, GdnQuery

    query = GdnQuery(
        gate_activation="sigmoid", qk_l2norm=True, state_dtype="float32", key_heads=8,
        value_heads=24, max_seqs=1, max_tokens=4, state_index_columns=4, max_state_slots=7,
        dt_bias_dtype="bfloat16",
    )
    assert _encoded(TUNING, query) == _encoded(TUNING, replace(query, max_state_slots=26_130))
    assert "max_state_slots" not in _encoded(TUNING, query)


def test_qsa_key_ignores_the_cache_page_counts():
    from b12x.attention.qsa._tuning import TUNING, QsaQuery

    query = QsaQuery(
        q_dtype="bfloat16", kv_dtype="float8_e4m3fn", q_heads=12, kv_heads=1, head_dim=256,
        index_heads=4, index_kv_heads=1, index_head_dim=128, index_rotary_dim=64,
        main_page_size=2848, max_batch=1, max_q_rows=128, max_seq_len=2048,
        max_speculative_tokens=3, compress_ratio=4, budget=2048, position_axes=3,
        mrope_interleaved=True, max_raw_state_slots=1, num_main_cache_pages=2,
        num_compressed_cache_pages=2, compressed_page_size=712, mrope_sections=(11, 11, 10),
        rms_norm_eps=1e-6, abi=_QSA_ABI,
    )
    larger = replace(query, num_main_cache_pages=21_537, num_compressed_cache_pages=5_385)
    assert _encoded(TUNING, query) == _encoded(TUNING, larger)
    assert not {"num_main_cache_pages", "num_compressed_cache_pages"} & set(_encoded(TUNING, query))


def test_pool_dependent_contracts_exclude_pool_size_fields():
    from b12x.attention.qsa._tuning import TUNING as QSA
    from b12x.sequence.gdn_decode._tuning import TUNING as GDN_DECODE
    from b12x.sequence.gdn_prefill._tuning import TUNING as GDN_PREFILL
    from b12x.sequence.ple._tuning import TUNING as PLE

    assert "max_state_slots" not in GDN_PREFILL.query_fields
    assert "max_state_slots" not in GDN_DECODE.query_fields and GDN_DECODE.query_schema_version == 6
    assert "max_state_slots" not in PLE.query_fields and PLE.query_schema_version == 4
    assert not {"num_main_cache_pages", "num_compressed_cache_pages"} & QSA.query_fields
    assert QSA.query_schema_version == 7
