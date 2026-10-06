"""Reviewed attention fixture contracts independent of offline tuning."""

from __future__ import annotations

from b12x.testing.attention_corpus import (
    COMMON_PREFILL_TOKEN_CAPACITIES,
    COMMON_SEQUENCE_CAPACITIES,
    GDN_GEOMETRIES,
    GQA_GEOMETRIES,
    MLA_GEOMETRIES,
    QSA_DIRECT_PREFILL_ROWS,
    QSA_GEOMETRIES,
    QSA_PAGE_SIZES,
    SPARSE_MLA_GEOMETRIES,
    gdn_cases,
    gqa_cases,
    mla_cases,
    qsa_cases,
    sparse_mla_cases,
)


def test_attention_corpora_have_stable_reviewed_cross_products() -> None:
    assert len(GDN_GEOMETRIES) == 21
    assert len(GQA_GEOMETRIES) == 18
    assert len(MLA_GEOMETRIES) == 1
    assert len(QSA_GEOMETRIES) == 3
    assert len(SPARSE_MLA_GEOMETRIES) == 24
    assert len(gdn_cases()) == 1_462
    assert len(gqa_cases()) == 14_400
    assert len(mla_cases()) == 200
    assert len(qsa_cases()) == 384
    assert len(sparse_mla_cases()) == 576

    cases = (*gdn_cases(), *gqa_cases(), *mla_cases(), *qsa_cases(), *sparse_mla_cases())
    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.query for case in gqa_cases()}) == len(gqa_cases())


def test_gdn_corpus_includes_qwen_and_glm_decay_contracts() -> None:
    cases = gdn_cases()
    glm_cases = [case for case in cases if case.metadata["decay_recipe"] == "kda"]

    assert {case.metadata["decay_recipe"] for case in cases} == {"gdn", "kda"}
    assert len(glm_cases) == 810
    assert {case.query["key_heads"] for case in glm_cases} == {4, 8, 16, 32, 64}
    assert all(case.query["key_heads"] == case.query["value_heads"] for case in glm_cases)
    assert (16, 16, 1) in {
        (case.query["max_seqs"], case.query["max_tokens"], case.query["state_index_columns"])
        for case in glm_cases
        if case.query["key_heads"] == 16
    }


def test_attention_capacity_axes_cover_serving_and_prefill_buckets() -> None:
    assert COMMON_SEQUENCE_CAPACITIES == (*range(1, 17), 32, 64, 128, 256)
    assert COMMON_PREFILL_TOKEN_CAPACITIES == (1_024, 2_048, 4_096, 8_192)
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["max_q_rows"])
        for case in qsa_cases()
        if int(case.query["max_q_rows"]) >= 1_024
    }
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["query_rows"])
        for case in mla_cases()
        if case.query["mode"] == "extend"
    }
    assert set(COMMON_PREFILL_TOKEN_CAPACITIES) <= {
        int(case.query["query_rows"]) for case in sparse_mla_cases()
    }
    assert QSA_DIRECT_PREFILL_ROWS == {
        2_048: (65, 128, 1_024, 2_048),
        32_768: (65, 128, 1_024, 4_096, 6_016, 8_192),
        262_144: (65, 128, 1_024, 4_096, 6_016, 8_192),
    }
    assert QSA_PAGE_SIZES == (16, 64, 1_504, 3_008)
