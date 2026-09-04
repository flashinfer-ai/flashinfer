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

from flashinfer.jit import gdn_noncp as gdn_noncp


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
        "gates_present": False,
    }
    params.update(overrides)
    return gdn_noncp.select_gdn_noncp_prefill_variant(**params)


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
    return gdn_noncp.select_gdn_noncp_decode_variant(**params)


def test_manifest_is_frozen_and_source_only() -> None:
    manifest = gdn_noncp._manifest()
    assert manifest["contract_row_count"] == 1777
    assert manifest["architecture_row_count"] == 3554
    assert manifest["admitted_architecture_rows"] == 3500
    assert manifest["fail_closed_architecture_rows"] == 54
    assert manifest["variant_count"] == len(manifest["variants"]) == 104
    assert manifest["source_only"] is True
    assert manifest["binary_artifacts"] is False


def test_prefill_resolver_selects_dvsplit_full_and_single_chunk() -> None:
    dvsplit = _prefill()
    assert dvsplit.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
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
    assert full.route_id == "flashinfer.gdn_prefill.noncp.full_dv"
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
    assert single.route_id == "flashinfer.gdn_prefill.noncp.single_chunk.dvsplit"
    assert "single_chunk" in single.variant_name


def test_prefill_resolver_selects_frozen_dynamic_head_specializations() -> None:
    dynamic_heads = _prefill(
        num_seqs=1,
        total_seq_len=64,
        max_seq_len=64,
        num_q_heads=3,
        num_k_heads=3,
        num_v_heads=3,
        use_initial_state=False,
        store_final_state=True,
    )
    dynamic_group = _prefill(
        num_q_heads=6,
        num_k_heads=2,
        num_v_heads=2,
    )

    assert dynamic_heads.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert dynamic_group.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    heads_record = gdn_noncp._kernel_record(dynamic_heads.variant_name)
    group_record = gdn_noncp._kernel_record(dynamic_group.variant_name)
    assert heads_record["specializations"]["NUM_O_HEADS_LOG2"] == -1
    assert heads_record["specializations"]["HEAD_GROUP_LOG2"] == 0
    assert group_record["specializations"]["NUM_O_HEADS_LOG2"] == -1
    assert group_record["specializations"]["HEAD_GROUP_LOG2"] == -1


def test_prefill_resolver_selects_sglang_tp4_bf16_indexed_row() -> None:
    route = _prefill(
        arch="sm_103a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=5,
        total_seq_len=5 * 64,
        max_seq_len=64,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        use_state_indices=True,
    )

    assert route.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    record = gdn_noncp._kernel_record(route.variant_name)
    assert record["specializations"] == {
        "ENABLE_CHECKPOINTS": 0,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 3,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 1,
    }


def test_prefill_resolver_selects_exact_sglang_tp4_checkpoint_row() -> None:
    route = _prefill(
        arch="sm_103a",
        io_dtype="bfloat16",
        state_dtype="bfloat16",
        num_seqs=7,
        total_seq_len=421,
        max_seq_len=107,
        num_q_heads=4,
        num_k_heads=4,
        num_v_heads=8,
        use_initial_state=True,
        store_final_state=True,
        checkpoint_every_n_tokens=64,
        use_state_indices=True,
        seq_lens=(52, 93, 15, 107, 72, 61, 21),
    )

    assert route.route_id == "flashinfer.gdn_prefill.noncp.checkpoints.dvsplit"
    record = gdn_noncp._kernel_record(route.variant_name)
    assert record["specializations"] == {
        "ENABLE_CHECKPOINTS": 1,
        "HEAD_GROUP_LOG2": 1,
        "IS_GQA": 0,
        "NUM_O_HEADS_LOG2": 3,
        "SINGLE_CHUNK_NO_STATE": 0,
        "STORE_FINAL_STATE": 1,
        "UNIT_GATES": 0,
        "USE_INITIAL_STATE": 1,
        "USE_STATE_INDICES": 1,
    }


def test_prefill_resolver_fails_closed_for_unpromoted_rows() -> None:
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="checkpoint route requires the frozen FP16/FP32 packed contract",
    ):
        _prefill(
            io_dtype="bfloat16",
            use_initial_state=False,
            checkpoint_every_n_tokens=64,
        )

    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="exact SGLang TP4 BF16 indexed B7/T421",
    ):
        _prefill(
            io_dtype="bfloat16",
            state_dtype="bfloat16",
            num_seqs=7,
            total_seq_len=421,
            max_seq_len=107,
            num_q_heads=4,
            num_k_heads=4,
            num_v_heads=8,
            use_initial_state=True,
            store_final_state=True,
            checkpoint_every_n_tokens=64,
            use_state_indices=True,
        )

    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="low-precision state requires BF16 I/O",
    ):
        _prefill(state_dtype="float16")


def test_kernel_loader_fails_closed_for_unsupported_architecture() -> None:
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="unsupported GDN non-CP GDN architecture",
    ):
        gdn_noncp.load_gdn_noncp_kernel("unused", "sm_90a")  # type: ignore[arg-type]


def test_decode_resolver_selects_all_promoted_physical_routes() -> None:
    small = _decode()
    assert small.route_id.endswith("nontranspose_small")
    assert "nontranspose_fp32_t1_small" in small.variant_name

    large = _decode(arch="sm_103a", batch_size=32)
    assert large.route_id.endswith("nontranspose_large")
    assert "nontranspose_fp32_t1_" in large.variant_name
    assert "small" not in large.variant_name

    pretranspose = _decode(layout="pretranspose")
    assert pretranspose.route_id == "flashinfer.gdn_decode.indexed_fp32_t1_splitv8"
    assert "pretranspose_splitv8" in pretranspose.variant_name


def test_decode_resolver_selects_exact_promoted_fp32_mtp_rows() -> None:
    rows = (
        (
            dict(
                batch_size=1,
                seq_len=2,
                disable_state_update=True,
                cache_steps=2,
            ),
            "indexed_fp32_mtp_t2.inline_tile8_verify_cache",
            "mtp_t2_inline_tile8",
        ),
        (
            dict(batch_size=4, seq_len=4, cache_steps=4),
            "indexed_fp32_mtp_t4.splitv8_update_cache",
            "mtp_t4_splitv8",
        ),
        *(
            (
                dict(batch_size=batch_size, seq_len=4, cache_steps=4),
                "indexed_fp32_mtp_t4.tile64_update_cache",
                "mtp_t4_splitv2_tile64",
            )
            for batch_size in (16, 64)
        ),
    )
    for overrides, route_suffix, variant_fragment in rows:
        route = _decode(
            layout="pretranspose",
            strided_inputs=True,
            cache_intermediate_states=True,
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert variant_fragment in route.variant_name


def test_decode_resolver_fails_closed_for_unpromoted_fp32_mtp_rows() -> None:
    base = {
        "layout": "pretranspose",
        "strided_inputs": True,
        "cache_intermediate_states": True,
        "seq_len": 4,
        "cache_steps": 4,
    }
    for overrides in (
        {"batch_size": 5},
        {"batch_size": 4, "strided_inputs": False},
        {"batch_size": 4, "cache_intermediate_states": False},
        {"batch_size": 4, "cache_steps": 5},
        {"batch_size": 4, "num_v_heads": 64},
    ):
        with pytest.raises(
            gdn_noncp.GDNNonCPUnsupportedError,
            match="FP32 MTP decode is limited",
        ):
            _decode(**(base | overrides))


def test_decode_resolver_selects_exact_promoted_bf16_rows() -> None:
    rows = (
        (
            dict(batch_size=4, seq_len=1, num_v_heads=32, strided_inputs=True),
            "indexed_bf16_t1.wide32",
        ),
        (
            dict(
                batch_size=4,
                seq_len=2,
                num_v_heads=32,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t2.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=3,
                num_v_heads=64,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=3,
            ),
            "indexed_bf16_verify_t3.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=64,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=32,
                strided_inputs=True,
                disable_state_update=True,
                cache_intermediate_states=True,
                cache_steps=4,
            ),
            "indexed_bf16_verify_t4.wide32",
        ),
        (
            dict(
                batch_size=8,
                seq_len=2,
                num_v_heads=64,
                strided_inputs=True,
            ),
            "indexed_bf16_update_t2.wide64",
        ),
        (
            dict(
                batch_size=8,
                seq_len=4,
                num_v_heads=64,
                strided_inputs=True,
                cache_intermediate_states=True,
                cache_steps=5,
            ),
            "indexed_bf16_checkpoint_t4.wide64",
        ),
    )
    for overrides, route_suffix in rows:
        route = _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert "bf16state_wide128" in route.variant_name

    tp4_rows = (
        (
            dict(batch_size=4, seq_len=1),
            "indexed_bf16_t1.tile16_fullwarp",
            "t1_bf16state_tile16",
        ),
        *(
            (
                dict(
                    batch_size=batch_size,
                    seq_len=4,
                    disable_state_update=True,
                    cache_intermediate_states=True,
                    cache_steps=4,
                ),
                "indexed_bf16_verify_t4.tile16_fullwarp",
                "t4_bf16state_tile16",
            )
            for batch_size in range(1, 9)
        ),
    )
    for overrides, route_suffix, variant_fragment in tp4_rows:
        route = _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            num_k_heads=4,
            num_q_heads=4,
            num_v_heads=8,
            strided_inputs=True,
            **overrides,
        )
        assert route.route_id.endswith(route_suffix)
        assert variant_fragment in route.variant_name


def test_decode_resolver_fails_closed_for_unpromoted_bf16_shape() -> None:
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="sixteen exact promoted",
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=5,
            num_v_heads=32,
            seq_len=1,
            strided_inputs=True,
        )

    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="sixteen exact promoted",
    ):
        _decode(
            state_dtype="bfloat16",
            layout="pretranspose",
            batch_size=9,
            num_k_heads=4,
            num_q_heads=4,
            num_v_heads=8,
            seq_len=4,
            strided_inputs=True,
            disable_state_update=True,
            cache_intermediate_states=True,
            cache_steps=4,
        )


def test_decode_resolver_fails_closed_outside_child_contract() -> None:
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="requires BF16 I/O and FP32 or BF16 state",
    ):
        _decode(state_dtype="float16")

    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="requires in-kernel Q/K L2 normalization",
    ):
        _decode(use_qk_l2norm=False)


def test_architecture_mapping_is_exact() -> None:
    assert gdn_noncp.arch_for_compute_capability(10, 0) == "sm_100a"
    assert gdn_noncp.arch_for_compute_capability(10, 3) == "sm_103a"
    with pytest.raises(
        gdn_noncp.GDNNonCPUnsupportedError,
        match="supports only SM100a/SM103a",
    ):
        gdn_noncp.arch_for_compute_capability(12, 0)


def test_prefill_resolver_selects_exact_gated_physical_schedules() -> None:
    gatepipe = _prefill(gates_present=True)
    gatepipe_record = gdn_noncp._kernel_record(gatepipe.variant_name)
    assert gatepipe.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "gatepipe4" in gatepipe.variant_name
    assert gatepipe_record["tma_abi"] == "grid_constant"
    assert gatepipe_record["specializations"]["UNIT_GATES"] == 0

    fullgrid_vhold = _prefill(
        num_seqs=8,
        total_seq_len=8 * 128,
        max_seq_len=128,
        gates_present=True,
    )
    fullgrid_record = gdn_noncp._kernel_record(fullgrid_vhold.variant_name)
    assert fullgrid_vhold.route_id == "flashinfer.gdn_prefill.noncp.dvsplit"
    assert "fullgrid_vhold" in fullgrid_vhold.variant_name
    assert fullgrid_record["tma_abi"] == "grid_constant"
    assert fullgrid_record["specializations"]["UNIT_GATES"] == 0


def test_nvcc_identity_captures_resolved_path_and_exact_version_output(
    tmp_path, monkeypatch
) -> None:
    cuda_root = tmp_path / "cuda"
    nvcc = cuda_root / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("compiler", encoding="utf-8")
    version_output = "nvcc: NVIDIA (R) Cuda compiler driver\nBuild exact-output\n"
    calls = []

    class Result:
        returncode = 0
        stdout = version_output

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(gdn_noncp, "get_cuda_path", lambda: str(cuda_root))
    monkeypatch.setattr(gdn_noncp.subprocess, "run", fake_run)

    observed_nvcc, observed_version = gdn_noncp._nvcc_identity()

    assert observed_nvcc == nvcc.resolve()
    assert observed_version == version_output
    assert calls == [
        (
            [str(nvcc.resolve()), "--version"],
            {
                "stdout": gdn_noncp.subprocess.PIPE,
                "stderr": gdn_noncp.subprocess.STDOUT,
                "text": True,
                "check": False,
            },
        )
    ]


def test_compile_cache_digest_isolated_by_nvcc_identity(tmp_path) -> None:
    common = {
        "arch": "sm_100a",
        "cuda_sha256": "cuda-source",
        "header_sha256s": ("header-a", "header-b"),
        "compile_options": ("--use_fast_math",),
    }
    baseline = gdn_noncp._compile_cache_digest(
        **common,
        nvcc=tmp_path / "cuda-a" / "bin" / "nvcc",
        nvcc_version="release 12.9\nBuild A\n",
    )
    changed_path = gdn_noncp._compile_cache_digest(
        **common,
        nvcc=tmp_path / "cuda-b" / "bin" / "nvcc",
        nvcc_version="release 12.9\nBuild A\n",
    )
    changed_version = gdn_noncp._compile_cache_digest(
        **common,
        nvcc=tmp_path / "cuda-a" / "bin" / "nvcc",
        nvcc_version="release 12.9\nBuild B\n",
    )

    assert len(baseline) == 64
    assert len({baseline, changed_path, changed_version}) == 3
    assert len({baseline[:16], changed_path[:16], changed_version[:16]}) == 3
