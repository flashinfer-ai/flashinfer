# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from flashinfer.jit import cake_gdn_noncp_decode as cake_gdn


def _prefill(**overrides):
    params = {
        "arch": "sm_100a",
        "io_dtype": "float16",
        "state_dtype": "float32",
        "num_seqs": 1,
        "total_seq_len": 16384,
        "max_seq_len": 16384,
        "num_q_heads": 2,
        "num_k_heads": 2,
        "num_v_heads": 8,
        "use_initial_state": True,
        "store_final_state": True,
        "checkpoint_every_n_tokens": 0,
        "use_state_indices": False,
    }
    params.update(overrides)
    return cake_gdn.select_cake_gdn_prefill_variant(**params)


def _decode(**overrides):
    params = {
        "arch": "sm_100a",
        "batch_size": 1,
        "io_dtype": "bfloat16",
        "state_dtype": "float32",
        "head_size": 128,
        "layout": "nontranspose",
        "num_k_heads": 16,
        "num_q_heads": 16,
        "num_v_heads": 32,
        "scale": 128**-0.5,
        "seq_len": 1,
        "use_qk_l2norm": True,
    }
    params.update(overrides)
    return cake_gdn.select_cake_gdn_decode_variant(**params)


def test_manifest_is_frozen_and_source_only() -> None:
    manifest = cake_gdn._manifest()
    assert manifest["generator_commit"] == (
        "bfef0844fb4069b084c11d3137dea07d9fe28f05"
    )
    assert manifest["contract_row_count"] == 1755
    assert manifest["architecture_row_count"] == 3510
    assert manifest["admitted_architecture_rows"] == 3450
    assert manifest["fail_closed_architecture_rows"] == 60
    assert manifest["variant_count"] == len(manifest["variants"]) == 71
    assert manifest["source_only"] is True
    assert manifest["binary_artifacts"] is False


def test_prefill_resolver_selects_dvsplit_full_and_single_chunk() -> None:
    dvsplit = _prefill()
    assert dvsplit.route_id == "cake.gdn_prefill.noncp.dvsplit"
    assert "dvsplit_initial_f16io" in dvsplit.variant_name

    full = _prefill(
        arch="sm_103a",
        num_seqs=16,
        total_seq_len=16 * 8192,
        max_seq_len=8192,
        num_q_heads=16,
        num_k_heads=16,
        num_v_heads=16,
    )
    assert full.route_id == "cake.gdn_prefill.noncp.full_dv"
    assert "dvsplit" not in full.variant_name

    single = _prefill(
        io_dtype="bfloat16",
        num_seqs=4,
        total_seq_len=4 * 64,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=False,
        store_final_state=False,
    )
    assert single.route_id == "cake.gdn_prefill.noncp.single_chunk.dvsplit"
    assert "single_chunk" in single.variant_name


def test_prefill_resolver_fails_closed_for_unpromoted_rows() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="checkpoint route requires FP16",
    ):
        _prefill(
            io_dtype="bfloat16",
            use_initial_state=False,
            checkpoint_every_n_tokens=64,
        )

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="low-precision state requires BF16 I/O",
    ):
        _prefill(state_dtype="float16")


def test_decode_resolver_selects_all_promoted_physical_routes() -> None:
    small = _decode()
    assert small.route_id.endswith("nontranspose_small")
    assert "nontranspose_fp32_t1_small" in small.variant_name

    large = _decode(arch="sm_103a", batch_size=32)
    assert large.route_id.endswith("nontranspose_large")
    assert "nontranspose_fp32_t1_" in large.variant_name
    assert "small" not in large.variant_name

    pretranspose = _decode(layout="pretranspose")
    assert pretranspose.route_id == "cake.gdn_decode.indexed_fp32_t1_splitv8"
    assert "pretranspose_splitv8" in pretranspose.variant_name


def test_decode_resolver_fails_closed_outside_child_contract() -> None:
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="requires BF16 I/O and FP32 state",
    ):
        _decode(state_dtype="bfloat16")

    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="requires in-kernel Q/K L2 normalization",
    ):
        _decode(use_qk_l2norm=False)


def test_architecture_mapping_is_exact() -> None:
    assert cake_gdn.arch_for_compute_capability(10, 0) == "sm_100a"
    assert cake_gdn.arch_for_compute_capability(10, 3) == "sm_103a"
    with pytest.raises(
        cake_gdn.CakeGDNUnsupportedError,
        match="supports only SM100a/SM103a",
    ):
        cake_gdn.arch_for_compute_capability(12, 0)
