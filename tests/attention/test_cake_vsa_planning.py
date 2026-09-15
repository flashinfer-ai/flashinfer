"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import contextlib
import hashlib
import math
import sys
import types

import pytest
import torch

from flashinfer import cake_vsa
from flashinfer.sparse import BlockSparseAttentionWrapper


class _NoRuntimeReduction:
    def max(self):
        raise AssertionError("static mask reductions must not run in run_cake_vsa")


def _plan_on_cpu(monkeypatch, **overrides):
    monkeypatch.setattr(cake_vsa, "_arch_for_device", lambda _device: "sm_100a")
    kwargs = {
        "indptr": None,
        "indices": None,
        "block_mask": None,
        "kv_block_lens": None,
        "q2k_indices": None,
        "q2k_num": None,
        "M": 128,
        "N": 256,
        "R": 64,
        "C": 64,
        "num_qo_heads": 2,
        "num_kv_heads": 2,
        "head_dim": 128,
        "q_data_type": torch.bfloat16,
        "sm_scale": None,
        "device": torch.device("cpu"),
    }
    kwargs.update(overrides)
    return cake_vsa.plan_cake_vsa(**kwargs)


def test_cake_wrapper_skips_generic_sparse_workspaces():
    workspace = torch.empty((1,), dtype=torch.uint8)
    wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")

    assert wrapper._int_workspace_buffer.numel() == 0
    assert wrapper._kv_lens_buffer.numel() == 0
    assert wrapper._pin_memory_int_workspace_buffer.numel() == 0


def test_cake_vsa_manifest_v2_inventory_and_digests():
    manifest = cake_vsa._manifest()
    source_records = cake_vsa._manifest_source_records(manifest)

    assert manifest["schema"] == "cake-vsa-block-sparse-source-export-v2"
    assert len(manifest["profiles"]) == 10
    assert len(source_records) == 30
    assert len({path for path, _ in source_records}) == 30
    identity = hashlib.sha256(
        "".join(
            f"{source_path}\0{digest}\n"
            for source_path, digest in sorted(source_records)
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["export_content_sha256"] == identity
    root = cake_vsa._source_dir()
    metadata = {
        source["path"]: source
        for profile in manifest["profiles"]
        for source in (profile["host"], *profile["device"].values())
    }
    for source_path, digest in source_records:
        source = root / source_path
        assert source.stat().st_size == metadata[source_path]["size_bytes"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_cake_vsa_public_apis_register_trace_templates():
    plan_trace = cake_vsa.plan_cake_vsa.fi_trace(
        M=128,
        N=256,
        R=64,
        C=64,
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim=128,
        q_data_type=torch.bfloat16,
        sm_scale=None,
        device=torch.device("cpu"),
    )
    q = torch.empty((128, 2, 128), dtype=torch.bfloat16)
    run_trace = cake_vsa.run_cake_vsa.fi_trace(q=q, k=q, v=q)

    assert plan_trace["op_type"] == "block_sparse_plan"
    assert run_trace["op_type"] == "block_sparse"


def test_block_mask_takes_precedence_over_bsr(monkeypatch):
    block_mask = torch.zeros((2, 2, 4), dtype=torch.bool)
    block_mask[:, 0, 1] = True
    block_mask[:, 1, 3] = True
    plan = _plan_on_cpu(
        monkeypatch,
        indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        indices=torch.tensor([0, 0], dtype=torch.int32),
        block_mask=block_mask,
        M=256,
        N=512,
        R=128,
        C=128,
    )

    torch.testing.assert_close(
        plan["indptr"], torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    torch.testing.assert_close(plan["indices"], torch.tensor([1, 3], dtype=torch.int32))


def test_noncanonical_bsr_rows_are_packed_for_fixed_stride_kernels(monkeypatch):
    plan = _plan_on_cpu(
        monkeypatch,
        indptr=torch.tensor([0, 7, 13], dtype=torch.int32),
        indices=torch.tensor(
            [0, 1, 2, 3, 4, 5, 5, 1, 2, 3, 4, 5, 6], dtype=torch.int32
        ),
        M=256,
        N=1024,
        R=128,
        C=128,
    )

    torch.testing.assert_close(
        plan["indptr"], torch.tensor([0, 6, 12], dtype=torch.int32)
    )
    torch.testing.assert_close(
        plan["indices"],
        torch.tensor([0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6], dtype=torch.int32),
    )
    assert plan["max_selected_blocks"] == 6
    assert plan["uniform_selected_blocks"]


@pytest.mark.parametrize("head_dim", [64, 96])
def test_native_small_head_routes_reject_more_than_64_blocks(monkeypatch, head_dim):
    block_mask = torch.ones((2, 1, 65), dtype=torch.bool)

    with pytest.raises(
        ValueError, match="D64/D96 routes support at most 64 selected blocks"
    ):
        _plan_on_cpu(
            monkeypatch,
            block_mask=block_mask,
            M=128,
            N=65 * 128,
            R=128,
            C=128,
            head_dim=head_dim,
        )


def test_head64_native_rejects_gqa_before_dispatch(monkeypatch):
    with pytest.raises(ValueError, match="D64/D96 routes support native-head BF16"):
        _plan_on_cpu(
            monkeypatch,
            M=128,
            N=256,
            R=128,
            C=128,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
        )


@pytest.mark.parametrize("bad_count", [0, 3])
def test_direct_q2k_rejects_invalid_counts(monkeypatch, bad_count):
    q2k_indices = torch.zeros((2, 2, 2), dtype=torch.int32)
    q2k_num = torch.full((2, 2), bad_count, dtype=torch.int32)

    with pytest.raises(ValueError, match=r"q2k_num entries must be in \[1, topk\]"):
        _plan_on_cpu(
            monkeypatch,
            q2k_indices=q2k_indices,
            q2k_num=q2k_num,
        )


@pytest.mark.parametrize("bad_index", [-1, 4])
def test_direct_q2k_rejects_invalid_active_indices(monkeypatch, bad_index):
    q2k_indices = torch.full((2, 2, 2), -1, dtype=torch.int32)
    q2k_indices[:, :, 0] = bad_index
    q2k_num = torch.ones((2, 2), dtype=torch.int32)

    with pytest.raises(
        ValueError, match=r"active q2k_indices entries must be in \[0, NB\)"
    ):
        _plan_on_cpu(
            monkeypatch,
            q2k_indices=q2k_indices,
            q2k_num=q2k_num,
        )


def test_direct_q2k_ignores_inactive_padding_indices(monkeypatch):
    q2k_indices = torch.full((2, 2, 2), -1, dtype=torch.int32)
    q2k_indices[:, :, 0] = 0
    q2k_num = torch.ones((2, 2), dtype=torch.int32)

    plan = _plan_on_cpu(
        monkeypatch,
        q2k_indices=q2k_indices,
        q2k_num=q2k_num,
    )

    assert plan["q2k_indices"] is q2k_indices
    assert plan["q2k_num"] is q2k_num


@pytest.mark.parametrize(
    "row_counts,expected_profile",
    [
        ([[24, 24], [24, 24]], "blk64_persistent"),
        ([[24, 28], [24, 28]], "blk64_persistent_ws_m64n256"),
        ([[25, 28], [28, 28]], "blk64_persistent"),
    ],
)
def test_blk64_profile_uses_actual_mixed_row_average(
    monkeypatch, row_counts, expected_profile
):
    q2k_indices = torch.arange(32, dtype=torch.int32).expand(2, 2, 32).contiguous()
    q2k_num = torch.tensor(row_counts, dtype=torch.int32)

    plan = _plan_on_cpu(
        monkeypatch,
        q2k_indices=q2k_indices,
        q2k_num=q2k_num,
        N=32 * 64,
    )

    assert plan["blk64_selected_blocks_total"] == sum(map(sum, row_counts))
    assert plan["blk64_profile"] == expected_profile


def test_blk64_partial_active_kv_block_uses_ordinary_profile(monkeypatch):
    q2k_indices = torch.arange(32, dtype=torch.int32).expand(2, 2, 32).contiguous()
    q2k_num = torch.full((2, 2), 28, dtype=torch.int32)
    kv_block_lens = torch.full((32,), 64, dtype=torch.int32)
    kv_block_lens[0] = 63

    plan = _plan_on_cpu(
        monkeypatch,
        q2k_indices=q2k_indices,
        q2k_num=q2k_num,
        kv_block_lens=kv_block_lens,
        N=32 * 64,
    )

    assert plan["blk64_profile"] == "blk64_persistent"


def test_run_blk64_loads_the_profile_cached_by_planning(monkeypatch):
    loaded = []
    launched = []
    module = types.SimpleNamespace(run=lambda *args: launched.append(args))
    monkeypatch.setattr(
        cake_vsa,
        "_load_module",
        lambda profile, arch: loaded.append((profile, arch)) or module,
    )
    monkeypatch.setattr(cake_vsa, "_arch_for_device", lambda _device: "sm_100a")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: types.SimpleNamespace(multi_processor_count=8),
    )
    monkeypatch.setitem(
        sys.modules,
        "tvm_ffi",
        types.SimpleNamespace(use_torch_stream=contextlib.nullcontext),
    )
    tensor = torch.empty((1,))
    plan = {
        "blk64_profile": "blk64_persistent_ws_m64n256",
        "mb": 2,
        "num_qo_heads": 2,
        "sm_scale": 0.5,
        "head_dim": 128,
        "q2k_indices": torch.empty((2, 2, 8), dtype=torch.int32),
        "q2k_num": torch.full((2, 2), 8, dtype=torch.int32),
        "kv_block_lens": torch.full((8,), 64, dtype=torch.int32),
        "M": 128,
    }

    cake_vsa._run_blk64(plan, tensor, tensor, tensor, tensor, tensor, False)

    assert loaded == [("blk64_persistent_ws_m64n256", "sm_100a")]
    assert len(launched) == 1


@pytest.mark.parametrize("bad_length", [0, 65])
def test_kv_block_lens_rejects_entries_outside_block(monkeypatch, bad_length):
    kv_block_lens = torch.full((4,), 64, dtype=torch.int32)
    kv_block_lens[1] = bad_length

    with pytest.raises(ValueError, match=r"kv_block_lens entries must be in \[1, C\]"):
        _plan_on_cpu(
            monkeypatch,
            q2k_indices=torch.zeros((2, 2, 1), dtype=torch.int32),
            kv_block_lens=kv_block_lens,
        )


def test_gqa_masks_must_match_within_each_kv_head_group(monkeypatch):
    block_mask = torch.zeros((4, 2, 4), dtype=torch.bool)
    block_mask[0, :, 0] = True
    block_mask[1, :, 1] = True
    block_mask[2:, :, 2] = True

    with pytest.raises(
        ValueError, match="masks must be identical within each KV-head group"
    ):
        _plan_on_cpu(
            monkeypatch,
            block_mask=block_mask,
            M=256,
            N=512,
            R=128,
            C=128,
            num_qo_heads=4,
            num_kv_heads=2,
            q_data_type=torch.float16,
        )


def test_fp16_gqa_metadata_uses_each_kv_head_group(monkeypatch):
    block_mask = torch.zeros((4, 2, 4), dtype=torch.bool)
    block_mask[:2, :, 0] = True
    block_mask[2:, :, 3] = True
    plan = _plan_on_cpu(
        monkeypatch,
        block_mask=block_mask,
        M=256,
        N=512,
        R=128,
        C=128,
        num_qo_heads=4,
        num_kv_heads=2,
        q_data_type=torch.float16,
    )

    q2k, *_ = cake_vsa._fp16_metadata(plan, torch.empty((1,)))

    assert tuple(q2k.shape) == (2, 256, 1)
    torch.testing.assert_close(q2k[0], torch.zeros_like(q2k[0]))
    torch.testing.assert_close(q2k[1], torch.full_like(q2k[1], 3))


def test_fp16_direct_call_matches_generated_ffi_abi(monkeypatch):
    launched = []
    module = types.SimpleNamespace(run=lambda *args: launched.append(args))
    monkeypatch.setattr(cake_vsa, "_load_module", lambda *_args: module)
    monkeypatch.setattr(cake_vsa, "_arch_for_device", lambda _device: "sm_100a")
    monkeypatch.setitem(
        sys.modules,
        "tvm_ffi",
        types.SimpleNamespace(use_torch_stream=contextlib.nullcontext),
    )

    q = torch.empty((256, 4, 128), dtype=torch.float16)
    k = torch.empty((512, 2, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    out = torch.empty_like(q)
    stats = torch.empty((256, 4), dtype=torch.float32)
    q2k = torch.empty((2, 256, 2), dtype=torch.int32)
    cu_q = torch.tensor([0, 256], dtype=torch.int32)
    cu_k = torch.tensor([0, 512], dtype=torch.int32)
    q_offsets = torch.zeros((1,), dtype=torch.int32)
    kv_lens = torch.tensor([512], dtype=torch.int32)
    page_table = torch.zeros((1,), dtype=torch.int32)
    scale_dummy = torch.empty((1, 1, 128, 8), dtype=torch.uint8)
    monkeypatch.setattr(
        cake_vsa,
        "_fp16_metadata",
        lambda *_args: (
            q2k,
            cu_q,
            cu_k,
            q_offsets,
            kv_lens,
            page_table,
            scale_dummy,
            2,
        ),
    )
    plan = {
        "M": 256,
        "num_qo_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 128,
        "sm_scale": 0.125,
    }

    cake_vsa._run_fp16(plan, q, k, v, out, stats, True)

    assert len(launched) == 1
    args = launched[0]
    assert len(args) == 32
    expected_tensors = (
        q,
        k,
        scale_dummy,
        v,
        scale_dummy,
        out,
        stats,
        stats,
        q2k,
        cu_q,
        cu_k,
        q_offsets,
        kv_lens,
        page_table,
    )
    assert all(
        actual is expected
        for actual, expected in zip(args[:14], expected_tensors, strict=True)
    )
    assert args[14:23] == (256, 4, 2, 2, 1, 0, 0, 0, 0)
    assert args[23] == pytest.approx(0.125 / math.log(2.0))
    assert args[24:29] == (1.0, 1.0, 1.0, 1, 0)
    assert args[29:] == (1, 4, 1)


def test_ultrasparse_route_rejects_non_six_topk_before_launch(monkeypatch):
    q = torch.empty((1,), dtype=torch.bfloat16)
    plan = {
        "head_dim": 128,
        "R": 128,
        "num_qo_heads": 8,
        "num_kv_heads": 8,
        "mb": 625,
        "N": 16384,
        "indices": torch.empty((1,), dtype=torch.int32),
        "max_selected_blocks": 8,
        "uniform_selected_blocks": True,
    }
    monkeypatch.setattr(cake_vsa, "_check_inputs", lambda *_args: None)
    monkeypatch.setattr(cake_vsa, "_outputs", lambda *_args: (q, q))
    monkeypatch.setattr(
        cake_vsa,
        "_run_standard",
        lambda *_args, **_kwargs: pytest.fail("launcher should not be reached"),
    )

    with pytest.raises(ValueError, match="requires exactly six selected blocks"):
        cake_vsa.run_cake_vsa(
            plan,
            q,
            q,
            q,
            out=None,
            lse=None,
            return_lse=False,
            backend="cake",
        )


def test_run_cake_vsa_uses_planned_mask_reductions(monkeypatch):
    q = torch.empty((1,), dtype=torch.bfloat16)
    stats = torch.empty((1,), dtype=torch.float32)
    profiles = []
    plan = {
        "head_dim": 128,
        "R": 128,
        "num_qo_heads": 8,
        "num_kv_heads": 8,
        "mb": 2,
        "N": 512,
        "max_selected_blocks": 2,
        "uniform_selected_blocks": True,
        "row_counts": _NoRuntimeReduction(),
    }

    monkeypatch.setattr(cake_vsa, "_check_inputs", lambda *_args: None)
    monkeypatch.setattr(
        cake_vsa,
        "_outputs",
        lambda *_args: (q, stats),
    )

    def record_profile(profile, *_args, **_kwargs):
        profiles.append(profile)

    monkeypatch.setattr(cake_vsa, "_run_standard", record_profile)

    result = cake_vsa.run_cake_vsa(
        plan,
        q,
        q,
        q,
        out=None,
        lse=None,
        return_lse=False,
        backend="cake",
    )

    assert result is q
    assert profiles == ["blk128_compact"]
