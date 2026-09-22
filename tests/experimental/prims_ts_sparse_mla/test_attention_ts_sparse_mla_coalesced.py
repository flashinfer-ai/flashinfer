# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from tests.experimental.prims_ts_sparse_mla.sparse_mla_test_utils import prepare_fixture

pytest.importorskip("cutlass", minversion="4.7.0")
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


def exercise_graph_routes(
    dtype,
    warps,
    fused,
    direct=False,
    balanced=False,
    compressed_slots=65,
    tile=16,
    reuse_kv=False,
    reuse_stages=0,
    splits=1,
    heads=24,
    family=None,
    device_scales=False,
    reduction="gmem_separate",
    assume_valid_prefix=False,
    auto_dispatch=False,
    uniform_offset_cache=False,
    scheduler="nonpersistent",
    repeat_batch=1,
    kv_pipeline_stages=0,
    paired_correction=False,
    single_kv_stream=False,
    defer_max_update=False,
    kv_tile_size=128,
    page_pipeline_stages=0,
):
    torch.manual_seed(92)
    q = (torch.randn(1, 3, heads, 512, device="cuda") * 0.15).to(dtype)
    s_storage = torch.randn(3, 257, 512, device="cuda") * 0.15
    c_storage = torch.randn(7, 33, 512, device="cuda") * 0.15 + 0.2
    s_storage[:, 256] = torch.nan
    c_storage[:, 32] = torch.nan
    s = s_storage.to(dtype)[:, :256]
    c = c_storage.to(dtype)[:, :32]
    si = torch.arange(128, device="cuda", dtype=torch.int32).repeat(1, 3, 1)
    ci = (
        torch.arange(compressed_slots, device="cuda", dtype=torch.int32) % 192
    ).repeat(1, 3, 1)
    if direct:
        # Keep the four-byte index alignment and affine row-stride contract.
        # Quad loads must also work when row starts are not 16-byte aligned.
        si_padded = torch.full((1, 3, 129), -1, device="cuda", dtype=torch.int32)
        ci_padded = torch.full(
            (1, 3, compressed_slots + 1), -1, device="cuda", dtype=torch.int32
        )
        si_padded[..., 1:].copy_(si)
        ci_padded[..., 1:].copy_(ci)
        si, ci = si_padded[..., 1:], ci_padded[..., 1:]
    si[0, 1] += 240  # The run crosses physical page padding inside a warp group.
    ci[0, 1] += 7
    sl = torch.tensor([[128, 119, 0]], device="cuda", dtype=torch.int32)
    cl = torch.tensor(
        [[compressed_slots, compressed_slots - 4, 0]], device="cuda", dtype=torch.int32
    )
    sinks = torch.randn(heads, device="cuda")
    sinks[0], sinks[1] = torch.inf, -torch.inf
    if repeat_batch > 1:
        q = (torch.randn(repeat_batch, 3, heads, 512, device="cuda") * 0.15).to(dtype)
        si = si.repeat(repeat_batch, 1, 1)
        ci = ci.repeat(repeat_batch, 1, 1)
        sl = sl.repeat(repeat_batch, 1)
        cl = cl.repeat(repeat_batch, 1)
        batch_offset = torch.arange(repeat_batch, device="cuda")[:, None, None] % 3
        si += batch_offset * 128
        ci += batch_offset * 32
        # Long lists already span most of the seven-page pool. Keep the
        # per-request shift inside its logical extent, including page wraps.
        ci %= c.shape[0] * c.shape[1]
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    tuning = _SparseMlaTuning(
        family=family or ("keep" if tile == 64 else "swap"),
        tile_size_q=tile,
        gather_issue_warps=(
            1 if family == "2cta" and dtype == torch.float8_e4m3fn else warps
        ),
        offset_cache="strided" if family == "2cta" else "coalesced",
        head_dim_ctas=1 if reuse_kv or family == "2cta" or single_kv_stream else 4,
        split_kv=splits,
        reduction=reduction,
        fuse_epilogue=fused,
        direct_inputs=direct,
        balanced_registers=balanced,
        reuse_kv=reuse_kv,
        uniform_offset_cache=uniform_offset_cache,
        scheduler=scheduler,
        kv_pipeline_stages=kv_pipeline_stages,
        paired_correction=paired_correction,
        single_kv_stream=single_kv_stream,
        defer_max_update=defer_max_update,
        kv_tile_size=kv_tile_size,
        page_pipeline_stages=page_pipeline_stages,
        reuse_kv_stages=(
            (reuse_stages or (10 if dtype == torch.float8_e4m3fn else 4))
            if reuse_kv or single_kv_stream
            else 0
        ),
    )
    wrapper._impl._tuning = None if auto_dispatch else tuning
    wrapper.plan(
        q.device,
        q.shape[0],
        heads,
        max_seq_len_q=3,
        q_data_type=dtype,
        max_topk=128,
        max_extra_topk=compressed_slots,
        has_sinks=True,
        return_lse=True,
        assume_valid_prefix=assume_valid_prefix,
    )
    if auto_dispatch and dtype == torch.float8_e4m3fn:
        assert wrapper._impl._state["tuning"].defer_max_update == defer_max_update
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device=q.device)
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
        sinks=sinks,
        out=out,
        lse=lse,
    )
    scale_kwargs = {}
    if device_scales:
        shared_kv_scale = torch.tensor(0.75, device="cuda")
        scale_kwargs = dict(
            q_scale=torch.tensor(1.25, device="cuda"),
            swa_kv_scale=shared_kv_scale,
            compressed_kv_scale=shared_kv_scale,
        )
        kwargs.update(scale_kwargs)

    def check():
        expected, expected_lse, bound = sparse_mla_reference(
            q,
            s,
            c,
            si,
            ci,
            swa_topk_lens=sl,
            compressed_topk_lens=cl,
            sinks=sinks,
            return_fp8_error_bound=True,
            **{k: v.item() for k, v in scale_kwargs.items()},
        )
        assert torch.isfinite(out).all()
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - expected).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), expected, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(lse.double(), expected_lse, atol=2e-4, rtol=1e-4)

    wrapper.run(**prepare_fixture(wrapper, q, s, c, **kwargs))
    if not auto_dispatch:
        # Forced profiles must reach their requested paths. Automatic dispatch
        # may consume either representation supplied by external preparation.
        assert wrapper._impl._state["last_fused_epilogue"] == fused
        assert wrapper._impl._state["last_direct_inputs"] == direct
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(**prepare_fixture(wrapper, q, s, c, validate=False, **kwargs))
    si[0, 0] = si[0, 0].flip(0)
    if not assume_valid_prefix:
        si[0, 0, 17::9] = -1
        ci[0, 0, 3::11] = -1
    sl[0, 2], cl[0, 2] = 128, 32
    sl[0, 1], cl[0, 1] = 0, 0
    if device_scales:
        scale_kwargs["q_scale"].fill_(0.5)
        scale_kwargs["swa_kv_scale"].fill_(1.5)
    graph.replay()
    check()
    assert torch.count_nonzero(out[0, 1]) == 0
    assert torch.isneginf(lse[0, 1]).all()

    if assume_valid_prefix:
        # Exercise both mask words, source-boundary moves and empty streams.
        # Poison inactive slots, so ignoring the live lengths is observable.
        for length in (0, 1, 31, 32, 33, 63, 64, 65, 127, 128):
            si.copy_(torch.arange(128, device=q.device).view(1, 1, 128).expand_as(si))
            ci.copy_(
                (torch.arange(compressed_slots, device=q.device) % 192)
                .view(1, 1, compressed_slots)
                .expand_as(ci)
            )
            sl[0] = torch.tensor([length, 128 - length, length], device=q.device)
            cl[0] = torch.tensor(
                [min(compressed_slots, length), min(compressed_slots, 129 - length), 0],
                device=q.device,
            )
            si.masked_fill_(
                torch.arange(128, device=q.device)[None, None, :] >= sl[..., None], -1
            )
            ci.masked_fill_(
                torch.arange(compressed_slots, device=q.device)[None, None, :]
                >= cl[..., None],
                -1,
            )
            graph.replay()
            check()
        si[0, 0, 0] = -1  # The final loop iteration leaves this prefix active.
        invalid = prepare_fixture(wrapper, q, s, c, **kwargs)
        invalid["metadata"].indices[0, 0] = -1
        with pytest.raises(ValueError, match="active prefix contains a hole"):
            wrapper.run(**invalid)
        if (
            dtype == torch.float8_e4m3fn
            and tile in (16, 64, 128)
            and reduction == "gmem_separate"
        ):
            # The same geometry must compile a distinct generic variant.
            # Otherwise a prior prefix plan could silently remove hole masks.
            prefix_compiled = wrapper._impl._state["compiled_fused"]
            prefix_static = wrapper._impl._state["compiled_static"]
            wrapper.plan(
                q.device,
                1,
                heads,
                max_seq_len_q=3,
                q_data_type=dtype,
                max_topk=128,
                max_extra_topk=compressed_slots,
                has_sinks=True,
                return_lse=True,
                assume_valid_prefix=False,
            )
            if family != "2cta":
                # 1CTA retains exactly the generic compiled mask path.
                assert wrapper._impl._state["compiled_fused"] is prefix_compiled
                assert wrapper._impl._state["compiled_static"] is prefix_static
            else:
                assert wrapper._impl._state["compiled_fused"] is not prefix_compiled
                assert wrapper._impl._state["compiled_static"] is not prefix_static
            si[0, 0, 1::2] = -1
            wrapper.run(**prepare_fixture(wrapper, q, s, c, **kwargs))
            check()


@pytest.mark.parametrize(
    "slots,splits,heads,warps", [(65, 1, 24, 1), (1024, 3, 128, 4), (8192, 2, 64, 4)]
)
def test_reused_kv_graph_lifetime(slots, splits, heads, warps):
    exercise_graph_routes(
        torch.float8_e4m3fn,
        warps,
        True,
        direct=True,
        compressed_slots=slots,
        tile=64,
        reuse_kv=True,
        splits=splits,
        heads=heads,
    )


@pytest.mark.parametrize(
    "slots,splits,heads,warps", [(65, 1, 24, 1), (1024, 3, 128, 4), (8192, 2, 64, 4)]
)
def test_bf16_reused_kv_graph_lifetime(slots, splits, heads, warps):
    exercise_graph_routes(
        torch.bfloat16,
        warps,
        True,
        direct=True,
        compressed_slots=slots,
        tile=64,
        reuse_kv=True,
        reuse_stages=4,
        splits=splits,
        heads=heads,
        device_scales=False,
    )


@pytest.mark.parametrize(
    "slots,splits,heads,device_scales",
    [(65, 2, 24, True), (1024, 3, 128, False), (8192, 2, 64, False)],
)
def test_direct_2cta_graph_lifetime(slots, splits, heads, device_scales):
    exercise_graph_routes(
        torch.float8_e4m3fn,
        1,
        True,
        direct=True,
        compressed_slots=slots,
        tile=128,
        splits=splits,
        heads=heads,
        family="2cta",
        device_scales=device_scales,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("compressed_slots", [65, 1024])
def test_m32_random_graph_routes(dtype, compressed_slots):
    # Reuse the padded-page, empty-row, sink and live graph mutation coverage.
    # Split reduction is additionally exercised by the random benchmark sweep.
    exercise_graph_routes(
        dtype,
        4,
        True,
        dtype == torch.float8_e4m3fn,
        compressed_slots=compressed_slots,
        tile=32,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("slots,splits", [(65, 2), (512, 4), (1024, 4)])
def test_direct_cluster_graph_routes(dtype, slots, splits):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        balanced=True,
        compressed_slots=slots,
        tile=16,
        splits=splits,
        heads=32,
        device_scales=dtype == torch.float8_e4m3fn,
        reduction="cluster",
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("tile,heads", [(8, 12), (16, 24)])
def test_short_merged_graph_routes(dtype, tile, heads):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        balanced=True,
        compressed_slots=65,
        tile=tile,
        heads=heads,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_single_tile_split_graph_routes(dtype):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        compressed_slots=1024,
        tile=64,
        splits=9,
        heads=128 if dtype == torch.float8_e4m3fn else 64,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize("tile,family", [(64, "keep"), (128, "2cta")])
def test_sparse_pdl_multiwave_graph_routes(tile, family):
    # The smallest launch here is 3 queries * 32 splits * 2 CTAs = 192
    # CTAs. It exceeds a 152-SM GB300 and includes pruned split slots.
    exercise_graph_routes(
        torch.float8_e4m3fn,
        4 if tile == 64 else 1,
        True,
        direct=True,
        compressed_slots=257,
        tile=tile,
        splits=32,
        heads=128,
        family=family,
        device_scales=True,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize("tile", [8, 16, 32])
def test_minimal_group_split_graph_routes(dtype, tile):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        balanced=tile in (8, 16),
        compressed_slots=257,
        tile=tile,
        splits=3,
        heads=tile + tile // 2,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
@pytest.mark.parametrize(
    "tile,splits,reduction",
    [
        (8, 1, "gmem_separate"),
        (16, 3, "gmem_separate"),
        (16, 4, "cluster"),
        (32, 3, "gmem_separate"),
        (64, 3, "gmem_separate"),
    ],
)
def test_valid_prefix_graph_routes(dtype, tile, splits, reduction):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        balanced=tile in (8, 16),
        compressed_slots=257,
        tile=tile,
        splits=splits,
        heads=32 if reduction == "cluster" else tile + tile // 2,
        device_scales=dtype == torch.float8_e4m3fn,
        reduction=reduction,
        assume_valid_prefix=True,
    )


@pytest.mark.parametrize("splits", [1, 3])
def test_valid_prefix_2cta_graph_routes(splits):
    exercise_graph_routes(
        torch.float8_e4m3fn,
        1,
        True,
        direct=True,
        compressed_slots=1024,
        tile=128,
        splits=splits,
        heads=128,
        family="2cta",
        device_scales=True,
        assume_valid_prefix=True,
    )


@pytest.mark.parametrize(
    "slots,heads,scheduler,repeat_batch",
    [
        (65, 24, "nonpersistent", 1),
        (1024, 128, "nonpersistent", 1),
        (8192, 64, "nonpersistent", 1),
        (65, 128, "clc", 64),
        (1024, 24, "clc", 64),
    ],
)
def test_bf16_2cta_uniform_pages_graph_lifetime(slots, heads, scheduler, repeat_batch):
    exercise_graph_routes(
        torch.bfloat16,
        4,
        True,
        direct=False,
        compressed_slots=slots,
        tile=128,
        splits=1,
        heads=heads,
        family="2cta",
        scheduler=scheduler,
        repeat_batch=repeat_batch,
        kv_pipeline_stages=8,
        uniform_offset_cache=True,
    )


@pytest.mark.parametrize(
    "dtype,heads,slots",
    [
        (torch.float8_e4m3fn, 64, 65),
        (torch.float8_e4m3fn, 96, 513),
    ],
)
def test_keep_staged_clc_reuse_graph_lifetime(dtype, heads, slots):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=False,
        compressed_slots=slots,
        tile=64,
        reuse_kv=True,
        reuse_stages=10 if dtype == torch.float8_e4m3fn else 5,
        splits=1,
        heads=heads,
        device_scales=dtype == torch.float8_e4m3fn,
        uniform_offset_cache=True,
        scheduler="clc",
        repeat_batch=64,
    )


@pytest.mark.parametrize(
    "dtype,heads,slots",
    [
        (torch.float8_e4m3fn, 64, 65),
        (torch.float8_e4m3fn, 96, 513),
        (torch.bfloat16, 32, 65),
        (torch.bfloat16, 64, 1024),
    ],
)
def test_keep_staged_static_reuse_graph_lifetime(dtype, heads, slots):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=False,
        compressed_slots=slots,
        tile=64,
        reuse_kv=True,
        reuse_stages=10 if dtype == torch.float8_e4m3fn else 5,
        splits=1,
        heads=heads,
        device_scales=dtype == torch.float8_e4m3fn,
        uniform_offset_cache=True,
        scheduler="static",
        repeat_batch=64,
    )


@pytest.mark.parametrize(
    "dtype,tile,slots,scheduler",
    [
        (torch.bfloat16, 8, 65, "clc"),
        (torch.bfloat16, 16, 1024, "static"),
        (torch.float8_e4m3fn, 8, 65, "static"),
        (torch.float8_e4m3fn, 16, 1024, "clc"),
    ],
)
def test_staged_swap_reuse_graph_lifetime(dtype, tile, slots, scheduler):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=False,
        balanced=True,
        compressed_slots=slots,
        tile=tile,
        reuse_kv=True,
        reuse_stages=6 if dtype == torch.bfloat16 else 12,
        splits=1,
        heads=24,
        scheduler=scheduler,
        repeat_batch=64,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    "dtype,slots,heads,uniform,scheduler,repeat_batch",
    [
        (torch.bfloat16, 65, 24, False, "nonpersistent", 1),
        (torch.bfloat16, 1024, 64, True, "nonpersistent", 1),
        (torch.float8_e4m3fn, 257, 32, True, "nonpersistent", 1),
        (torch.bfloat16, 65, 24, True, "static", 64),
    ],
)
def test_keep_eight_gather_warps_graph_lifetime(
    dtype, slots, heads, uniform, scheduler, repeat_batch
):
    exercise_graph_routes(
        dtype,
        8,
        True,
        direct=scheduler == "nonpersistent",
        compressed_slots=slots,
        tile=64,
        reuse_kv=True,
        reuse_stages=5 if dtype == torch.bfloat16 else 10,
        splits=1,
        heads=heads,
        uniform_offset_cache=uniform,
        scheduler=scheduler,
        repeat_batch=repeat_batch,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    "slots,heads,repeat_batch",
    [(65, 24, 64), (1024, 128, 64), (8192, 64, 1), (257, 128, 128)],
)
def test_fp8_2cta_balanced_static_graph_lifetime(slots, heads, repeat_batch):
    exercise_graph_routes(
        torch.float8_e4m3fn,
        1,
        True,
        direct=True,
        balanced=True,
        compressed_slots=slots,
        tile=128,
        splits=1,
        heads=heads,
        family="2cta",
        scheduler="static",
        repeat_batch=repeat_batch,
        device_scales=True,
    )


@pytest.mark.parametrize(
    "dtype,tile,scheduler,repeat_batch",
    [
        (torch.bfloat16, 8, "nonpersistent", 1),
        (torch.float8_e4m3fn, 16, "nonpersistent", 1),
        (torch.bfloat16, 16, "clc", 64),
    ],
)
def test_swap_eight_gather_warps_graph_lifetime(dtype, tile, scheduler, repeat_batch):
    exercise_graph_routes(
        dtype,
        8,
        True,
        direct=scheduler == "nonpersistent",
        balanced=True,
        compressed_slots=512,
        tile=tile,
        reuse_kv=True,
        reuse_stages=6 if dtype == torch.bfloat16 else 12,
        splits=1,
        heads=24,
        scheduler=scheduler,
        repeat_batch=repeat_batch,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    "dtype,tile,stages,scheduler",
    [
        (torch.float8_e4m3fn, 64, 10, "clc"),
        (torch.bfloat16, 64, 5, "clc"),
        (torch.bfloat16, 16, 6, "clc"),
        (torch.float8_e4m3fn, 16, 12, "static"),
    ],
)
def test_direct_persistent_1cta_graph_lifetime(dtype, tile, stages, scheduler):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        balanced=tile == 16,
        compressed_slots=512,
        tile=tile,
        reuse_kv=True,
        reuse_stages=stages,
        splits=1,
        heads=24,
        scheduler=scheduler,
        repeat_batch=64,
        uniform_offset_cache=tile == 64,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    "dtype,slots,heads,warps,scheduler,repeat_batch",
    [
        (torch.bfloat16, 129, 24, 4, "nonpersistent", 1),
        (torch.bfloat16, 1024, 64, 8, "nonpersistent", 1),
        (torch.bfloat16, 129, 24, 8, "static", 64),
        (torch.float8_e4m3fn, 512, 64, 4, "clc", 64),
    ],
)
def test_paired_keep_correction_graph_lifetime(
    dtype, slots, heads, warps, scheduler, repeat_batch
):
    exercise_graph_routes(
        dtype,
        warps,
        True,
        direct=scheduler == "nonpersistent",
        compressed_slots=slots,
        tile=64,
        reuse_kv=True,
        reuse_stages=5 if dtype == torch.bfloat16 else 10,
        splits=1,
        heads=heads,
        scheduler=scheduler,
        repeat_batch=repeat_batch,
        uniform_offset_cache=True,
        device_scales=dtype == torch.float8_e4m3fn,
        paired_correction=True,
    )


@pytest.mark.parametrize(
    "dtype,tile,splits,slots,reuse,stages",
    [
        (torch.bfloat16, 16, 2, 512, True, 5),
        (torch.float8_e4m3fn, 16, 2, 2048, True, 10),
        (torch.float8_e4m3fn, 8, 4, 512, True, 10),
        (torch.float8_e4m3fn, 16, 4, 129, False, 8),
    ],
)
def test_single_stream_split_graph_lifetime(dtype, tile, splits, slots, reuse, stages):
    exercise_graph_routes(
        dtype,
        4,
        True,
        direct=True,
        balanced=True,
        compressed_slots=slots,
        tile=tile,
        reuse_kv=reuse,
        reuse_stages=stages,
        splits=splits,
        heads=24,
        single_kv_stream=True,
        device_scales=dtype == torch.float8_e4m3fn,
    )


@pytest.mark.parametrize(
    "direct,splits,slots,scheduler",
    [
        (False, 1, 2048, "nonpersistent"),
        (True, 1, 129, "nonpersistent"),
        (True, 2, 512, "nonpersistent"),
        (False, 1, 512, "clc"),
    ],
)
def test_bf16_2cta_bounded_anchor_graph_lifetime(direct, splits, slots, scheduler):
    exercise_graph_routes(
        torch.bfloat16,
        4,
        True,
        direct=direct,
        family="2cta",
        tile=128,
        compressed_slots=slots,
        splits=splits,
        heads=128,
        repeat_batch=4,
        scheduler=scheduler,
        kv_pipeline_stages=8,
        uniform_offset_cache=True,
        defer_max_update=True,
    )


@pytest.mark.parametrize("heads,repeat_batch,slots", [(64, 64, 512), (128, 256, 2048)])
def test_auto_fp8_exact_max_graph_lifetime(heads, repeat_batch, slots):
    exercise_graph_routes(
        torch.float8_e4m3fn,
        4,
        True,
        direct=True,
        compressed_slots=slots,
        heads=heads,
        repeat_batch=repeat_batch,
        auto_dispatch=True,
        device_scales=True,
    )


def test_fp8_exact_max_score_jumps():
    for family, heads, tile, stages, warps in [("keep", 64, 64, 10, 8)]:
        q = torch.full(
            (1, 1, heads, 512), 0.1, device="cuda", dtype=torch.float8_e4m3fn
        )
        swa = torch.zeros((1, 256, 512), device="cuda", dtype=torch.float8_e4m3fn)
        compressed = torch.empty(
            (5, 128, 512), device="cuda", dtype=torch.float8_e4m3fn
        )
        for page, value in enumerate((0.25, 0.1, 0.4, 0.2, 4.0)):
            compressed[page].fill_(value)
        si = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, -1)
        ci = torch.arange(640, device="cuda", dtype=torch.int32).view(1, 1, -1)
        sl = torch.full((1, 1), 128, device="cuda", dtype=torch.int32)
        cl = torch.full((1, 1), 640, device="cuda", dtype=torch.int32)
        sinks = torch.zeros(heads, device="cuda")
        out = torch.empty_like(q, dtype=torch.bfloat16)
        lse = torch.empty(q.shape[:-1], device="cuda")
        w = BatchSparseMLADecodePagedTSWrapper()
        w._impl._tuning = _SparseMlaTuning(
            family=family,
            tile_size_q=tile,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=warps,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            balanced_registers=family == "swap",
            reuse_kv=True,
            reuse_kv_stages=stages,
            uniform_offset_cache=family == "keep",
            defer_max_update=False,
        )
        w.plan(
            q.device,
            1,
            heads,
            max_seq_len_q=1,
            q_data_type=torch.float8_e4m3fn,
            max_topk=128,
            max_extra_topk=640,
            has_sinks=True,
            return_lse=True,
        )
        kwargs = dict(
            swa_indices=si,
            compressed_indices=ci,
            swa_topk_lens=sl,
            compressed_topk_lens=cl,
            sinks=sinks,
            out=out,
            lse=lse,
        )

        def check():
            ref, rlse, bound = sparse_mla_reference(
                q,
                swa,
                compressed,
                si,
                ci,
                swa_topk_lens=sl,
                compressed_topk_lens=cl,
                sinks=sinks,
                return_fp8_error_bound=True,
            )
            assert ((out.double() - ref).abs() <= bound).all()
            torch.testing.assert_close(lse.double(), rlse, atol=2e-4, rtol=1e-4)

        w.run(**prepare_fixture(w, q, swa, compressed, **kwargs))
        check()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            w.run(**prepare_fixture(w, q, swa, compressed, validate=False, **kwargs))
        for value in (1.0, 0.01, 0.2, 0.001):
            q.fill_(value)
            g.replay()
            torch.cuda.synchronize()
            check()


@pytest.mark.parametrize("family,heads", [("keep", 64), ("2cta", 128)])
def test_six_log2_anchor_boundary(family, heads):
    dtype = torch.bfloat16
    # The first two BK128 tiles initialize both softmax streams at zero.
    # The following two tiles straddle the six-log2 update boundary. This
    # checks both retained-anchor and rescaling paths.
    q = torch.zeros((1, 1, heads, 512), device="cuda", dtype=dtype)
    q[..., 1] = 1
    swa = torch.zeros((1, 256, 512), device="cuda", dtype=q.dtype)
    kv = torch.zeros((384, 1, 512), device="cuda", dtype=q.dtype)
    kv[128:, :, 0] = 1
    si = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, -1)
    ci = torch.arange(384, device="cuda", dtype=torch.int32).view(1, 1, -1)
    sinks = torch.zeros(heads, device="cuda")
    sinks[0], sinks[1] = -torch.inf, torch.inf
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda")
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._tuning = _SparseMlaTuning(
        family=family,
        tile_size_q=heads,
        split_kv=1,
        fuse_epilogue=True,
        direct_inputs=True,
        gather_issue_warps=4 if family == "keep" else 1,
        offset_cache="coalesced" if family == "keep" else "strided",
        reuse_kv=family == "keep",
        reuse_kv_stages=5 if family == "keep" else 0,
        defer_max_update=True,
    )
    w.plan(
        q.device,
        1,
        heads,
        q_data_type=q.dtype,
        max_topk=128,
        max_extra_topk=384,
        has_sinks=True,
        return_lse=True,
    )
    kwargs = dict(
        swa_indices=si,
        compressed_indices=ci,
        sinks=sinks,
        softmax_scale=0.6931471805599453,
        out=out,
        lse=lse,
    )
    w.run(**prepare_fixture(w, q, swa, kv, **kwargs))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(**prepare_fixture(w, q, swa, kv, validate=False, **kwargs))
    for delta in (5.5, 6.0, 6.5, 12.0):
        kv[128:, :, 1] = delta
        graph.replay()
        ref, ref_lse = sparse_mla_reference(
            q,
            swa,
            kv,
            si,
            ci,
            sinks=sinks,
            softmax_scale=kwargs["softmax_scale"],
        )
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.double(), ref, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(lse.double(), ref_lse, atol=2e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "direct,warps,stages,page_stages,slots,scheduler,prefix",
    [
        (True, 8, 10, 2, 129, "nonpersistent", True),
        (True, 8, 10, 2, 2048, "clc", False),
        (False, 4, 8, 1, 512, "static", False),
    ],
)
def test_bf16_bk64_gather_graph_lifetime(
    direct, warps, stages, page_stages, slots, scheduler, prefix
):
    exercise_graph_routes(
        torch.bfloat16,
        warps,
        True,
        direct=direct,
        tile=64,
        compressed_slots=slots,
        reuse_kv=True,
        reuse_stages=stages,
        kv_tile_size=64,
        page_pipeline_stages=page_stages,
        uniform_offset_cache=True,
        defer_max_update=True,
        heads=32,
        repeat_batch=4,
        scheduler=scheduler,
        assume_valid_prefix=prefix,
    )


@pytest.mark.parametrize("stages", [0, 4, 7])
def test_bf16_bk64_rejects_insufficient_retained_kv(stages):
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper._impl._tuning = _SparseMlaTuning(
        family="keep",
        tile_size_q=64,
        gather_issue_warps=8,
        offset_cache="coalesced",
        direct_inputs=True,
        fuse_epilogue=True,
        reuse_kv=True,
        reuse_kv_stages=stages,
        kv_tile_size=64,
    )
    with pytest.raises(ValueError, match="two retained K tiles"):
        wrapper.plan(
            "cuda",
            1,
            32,
            q_data_type=torch.bfloat16,
            max_topk=128,
            max_extra_topk=2048,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("warps,fused", [(1, False), (4, True)])
def test_bulk_gather_transitions_and_poisoned_page_padding(dtype, warps, fused):
    exercise_graph_routes(dtype, warps, fused)
