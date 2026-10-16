# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 input with native 16-bit output across PrimTS decode schedules."""

import pytest
import torch

from tests.attention.test_attention_ts_decode import (
    _FP8,
    _REQUIRES_PRIMTS_GPU,
    _assert_case_correct,
    _make_decode_case,
    _plan_case,
    _run_case,
    _with_reference,
)
from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    batch_decode_with_paged_kv_cache,
    get_prims_ts_batch_decode_workspace_size,
    prepare_prims_ts_batch_decode_with_kv_cache,
)
from flashinfer.attention.prims_ts import decode
from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config


@pytest.mark.parametrize("output_dtype", (torch.float16, torch.bfloat16))
def test_fp8_contiguous_16bit_profile_does_not_use_paged_row_guards(output_dtype):
    """Paged output recipes must not shadow contiguous or Sage qualification."""
    cfg = fmha_decode_config.make_decode_config(
        headdim=128,
        args=dict(use_keeps_mma_ab=True, tile_size_q=64, groups_tokens_heads_q=True),
        seq_len_q=1,
        seq_len_kv=256,
        batch_size=1,
        num_heads_q=64,
        num_heads_kv=1,
        q_dtype=fmha_decode_config.Float8E4M3FN,
        k_dtype=fmha_decode_config.Float8E4M3FN,
        v_dtype=fmha_decode_config.Float8E4M3FN,
        o_dtype=decode._cutlass_dtype(decode._dtype_key(output_dtype)),
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        auto_tuner=False,
    )
    assert cfg.supports_grouped_keeps
    assert not cfg.uses_guarded_fixed_q1_grouped_keeps
    assert not cfg.uses_guarded_grouped_keeps_output_rows


@pytest.mark.parametrize("split_kv", (False, True))
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.parametrize(
    "batch_size,heads,head_dim,seq_len_q,page_size,packed",
    (
        (1, 16, 128, 4, 128, False),
        (4, 16, 128, 4, 128, False),
        (64, 16, 128, 4, 128, False),
        (2, 33, 64, 1, 32, False),
        (2, 65, 128, 1, 32, False),
        (2, 65, 256, 1, 32, False),
        (4, 16, 64, 4, 32, True),
    ),
)
@_REQUIRES_PRIMTS_GPU
def test_fp8_bf16_output_preserves_fp16_tactics(
    batch_size, heads, head_dim, seq_len_q, page_size, packed, mask_type, split_kv
):
    """Changing only 16-bit output precision preserves the selected schedule."""
    configs = []
    for output in ("float16", "bfloat16"):
        configs.append(
            decode._resolve_decode_launch_spec(
                torch.cuda.current_device(),
                batch_size,
                heads,
                1,
                head_dim,
                page_size,
                4096,
                seq_len_q,
                "float8_e4m3fn",
                "float8_e4m3fn",
                "float8_e4m3fn",
                output,
                "HND",
                mask_type,
                packed,
                -1,
                split_kv=split_kv,
            ).config
        )
    signatures = [dict(config.compile_signature()) for config in configs]
    for signature in signatures:
        signature.pop("out_dtype")
    assert signatures[0] == signatures[1]
    if seq_len_q == 1 and head_dim in (64, 128) and heads in (33, 65):
        # The existing 16-bit policy uses Swaps head bands here. Enabling an
        # epilogue must not add slower, unqualified cost-model candidates.
        assert not configs[0].use_keeps_mma_ab
        assert not configs[1].use_keeps_mma_ab


@pytest.mark.parametrize("split_kv", (False, True))
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@_REQUIRES_PRIMTS_GPU
def test_fp8_bf16_output_public_entry_points(mask_type, split_kv):
    """Issue6569: validated APIs, caller scratch, preparation, and graph replay."""
    case = _make_decode_case(
        kv_lens=(4096,) * 4,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=128,
        seq_len_q=4,
        page_size=128,
        qkv_dtype=_FP8,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type=mask_type,
        device="cuda",
        seed=6569,
    )
    lengths = torch.full((4,), 4096, dtype=torch.int32, device="cuda")
    wrapper = BatchDecodePagedTSWrapper()
    for validate in (True, False):
        wrapper.plan(
            case.q.device,
            4,
            16,
            1,
            128,
            128,
            4096,
            max_seq_len_q=4,
            q_data_type=_FP8,
            kv_data_type=_FP8,
            o_data_type=torch.bfloat16,
            mask_type=mask_type,
            split_kv=split_kv,
            validate=validate,
            seq_lens=lengths.cpu(),
        )
        output = _run_case(wrapper, case)
        reference = _with_reference(
            case,
            splits_kv=wrapper._plan_state.config.splits_kv,
            num_insts_kv=wrapper._plan_state.config.num_insts_kv,
        )
        _assert_case_correct(output, reference)

    common = dict(
        seq_len_q=4,
        out_dtype=torch.bfloat16,
        mask_type=mask_type,
        split_kv=split_kv,
    )
    one_shot = batch_decode_with_paged_kv_cache(
        case.q,
        case.paged_kv_cache,
        case.block_tables,
        lengths,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        **common,
    )
    _assert_case_correct(one_shot, reference)

    workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        4,
        16,
        1,
        128,
        128,
        4096,
        q_dtype=_FP8,
        k_dtype=_FP8,
        v_dtype=_FP8,
        **common,
        device=case.q.device,
    )
    workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device="cuda")
    supplied_out = torch.empty_like(case.q, dtype=torch.bfloat16)
    plan = prepare_prims_ts_batch_decode_with_kv_cache(
        case.q,
        case.paged_kv_cache,
        workspace,
        case.block_tables,
        lengths,
        4096,
        out=supplied_out,
        **common,
    )
    assert (
        plan.run(
            case.q,
            out=supplied_out,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )
        is supplied_out
    )
    _assert_case_correct(supplied_out, reference)

    def run():
        return batch_decode_with_paged_kv_cache(
            case.q,
            case.paged_kv_cache,
            case.block_tables,
            lengths,
            workspace_buffer=workspace,
            max_kv_len=4096,
            out=supplied_out,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
            validate=False,
            **common,
        )

    assert run() is supplied_out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    supplied_out.fill_(torch.nan)
    graph.replay()
    _assert_case_correct(supplied_out, reference)


_VARIANTS = (
    ("swaps-direct", False, 8, 128, "disabled", 1),
    ("swaps-direct-q32", False, 32, 128, "disabled", 1),
    ("swaps-gmem", False, 16, 128, "gmem_reduction", 2),
    ("swaps-gmem-q32", False, 32, 128, "gmem_reduction", 2),
    ("swaps-cluster", False, 16, 128, "cluster_smem_reduction", 2),
    ("swaps-separate-q8", False, 8, 128, "gmem_reduction_with_separate_kernel", 2),
    ("swaps-separate-q32", False, 32, 128, "gmem_reduction_with_separate_kernel", 2),
    (
        "swaps-separate-clustered",
        False,
        16,
        128,
        "gmem_reduction_with_separate_kernel",
        16,
    ),
    ("keeps-direct", True, 64, 128, "disabled", 1),
    ("keeps-direct-q128", True, 128, 128, "disabled", 1),
    ("keeps-gmem", True, 128, 128, "gmem_reduction", 2),
    ("keeps-gmem-q64", True, 64, 128, "gmem_reduction", 2),
    ("keeps-separate", True, 64, 128, "gmem_reduction_with_separate_kernel", 2),
    ("keeps-separate-q128", True, 128, 128, "gmem_reduction_with_separate_kernel", 2),
    ("kv256-direct", True, 64, 256, "disabled", 1),
    ("kv256-gmem", True, 64, 256, "gmem_reduction", 2),
    ("kv256-separate", True, 64, 256, "gmem_reduction_with_separate_kernel", 2),
)


@pytest.mark.parametrize("output_dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "head_dim,keeps,tile_q,tile_kv,mode,splits",
    [
        pytest.param(
            head_dim, keeps, tile_q, tile_kv, mode, splits, id=f"{name}-d{head_dim}"
        )
        for name, keeps, tile_q, tile_kv, mode, splits in _VARIANTS
        for head_dim in ((128,) if tile_kv == 256 else (64, 128, 256))
    ],
)
@_REQUIRES_PRIMTS_GPU
def test_fp8_16bit_output_kernel_variants(
    monkeypatch, head_dim, keeps, tile_q, tile_kv, mode, splits, output_dtype
):
    """Execute each producer/output/reducer layout against the FP8-P oracle."""
    cfg = fmha_decode_config.make_decode_config(
        headdim=head_dim,
        args=(
            fmha_decode_config.arch_config_args(torch.cuda.get_device_capability()),
            dict(
                use_keeps_mma_ab=keeps,
                tile_size_q=tile_q,
                tile_size_kv=tile_kv,
                groups_tokens_heads_q=not (
                    keeps
                    and tile_kv == 128
                    and mode == "gmem_reduction_with_separate_kernel"
                ),
                use_persistent_scheduler=False,
            ),
        ),
        seq_len_q=1,
        seq_len_kv=4096,
        batch_size=2,
        num_heads_q=tile_q,
        num_heads_kv=1,
        q_dtype=fmha_decode_config.Float8E4M3FN,
        k_dtype=fmha_decode_config.Float8E4M3FN,
        v_dtype=fmha_decode_config.Float8E4M3FN,
        o_dtype=decode._cutlass_dtype(decode._dtype_key(output_dtype)),
        qkv_layout="pagedKv",
        num_tokens_per_page=32,
        split_kv_mode=mode,
        splits_kv=splits,
        max_splits_kv=splits,
        mask_type="dense",
        auto_tuner=False,
    )
    spec = decode._decode_launch_spec_from_config(
        cfg,
        batch_size=2,
        num_qo_heads=tile_q,
        num_kv_heads=1,
        head_dim=head_dim,
        seq_len_q=1,
        max_active_clusters=fmha_decode_config.get_max_active_clusters_for_cluster_size(
            1
        ),
    )
    monkeypatch.setattr(decode, "_resolve_decode_launch_spec", lambda *a, **kw: spec)
    case = _make_decode_case(
        kv_lens=(4096, 3583),
        num_qo_heads=tile_q,
        num_kv_heads=1,
        head_dim=head_dim,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=_FP8,
        output_dtype=output_dtype,
        cache_form="tuple",
        mask_type="dense",
        device="cuda",
        seed=6569 + head_dim,
    )
    # Uniform scores make FP8-P quantization exact, isolating output packing
    # and reduction across full/partial KV domains. Nonuniform FP8 rounding
    # is covered by the public-entry and mixed-precision cases above.
    case = _with_reference(
        case,
        q=torch.zeros_like(case.q),
        splits_kv=splits,
        num_insts_kv=cfg.num_insts_kv,
    )
    wrapper = _plan_case(case, max_kv_len=4096)
    _assert_case_correct(_run_case(wrapper, case), case)
