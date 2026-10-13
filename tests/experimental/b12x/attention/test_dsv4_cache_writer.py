"""V4 payload/footer writes with padded pages and captured replay."""

import pytest
import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


def records(values):
    groups = values[:, :448].float().reshape(-1, 7, 64)
    maximum = groups.abs().amax(-1).clamp_min(1e-4)
    exponent = (maximum / 448).log2().ceil()
    scales = torch.exp2(exponent)
    packed = (
        (groups / scales[:, :, None])
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .reshape(-1, 448)
    )
    rope = values[:, 448:].contiguous().view(torch.uint8)
    footer = torch.cat(
        (
            (exponent + 127).to(torch.uint8),
            torch.zeros_like(exponent[:, :1], dtype=torch.uint8),
        ),
        1,
    )
    return torch.cat((packed, rope), 1), footer


@pytest.mark.parametrize("page_size", (1, 2, 32, 128))
@pytest.mark.parametrize(
    "kind,slot_dtype", (("swa", torch.int32), ("indexed", torch.int64))
)
@torch.inference_mode()
def test_v4_writer_preserves_rope_footer_and_unwritten_bytes(
    page_size, kind, slot_dtype
):
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    size = mla.page_nbytes(page_size, cache_kind=kind, cache_format="deepseek_v4")
    storage = torch.full((4, size + 32), 0xA5, dtype=torch.uint8, device=device)
    cache = storage[:, :size]
    source = torch.randn((5, 544), dtype=torch.bfloat16, device=device)[:, :512]
    source[1].zero_()
    slots = torch.tensor(
        [page_size, page_size - 1, -1, 3 * page_size, 4 * page_size],
        dtype=slot_dtype,
        device=device,
    )
    plan = mla.plan_cache_writer(
        mla.CacheWriterQuery(
            max_rows=5,
            page_size=page_size,
            cache_kind=kind,
            cache_format="deepseek_v4",
            slot_dtype=str(slot_dtype).removeprefix("torch."),
        ),
        device=device,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="v4-writer",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(source, cache, slots)
                    ),
                ),
            )
        )
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mla.write_cache(source, cache, slots, plan=plan)
        for iteration in range(3):
            source.neg_()
            storage.fill_(0xA5)
            graph.replay()
            payload, footer = records(source)
            expected = torch.full_like(storage, 0xA5)
            for row in (0, 1, 3):
                slot = int(slots[row])
                page, offset = divmod(slot, page_size)
                expected[page, offset * 576 : (offset + 1) * 576] = payload[row]
                start = page_size * 576 + offset * 8
                expected[page, start : start + 8] = footer[row]
            torch.testing.assert_close(storage, expected, atol=0, rtol=0)
        graph.reset()


@torch.inference_mode()
def test_v4_writer_uses_64_bit_page_offsets():
    require_b12x()
    device = torch.device("cuda", torch.cuda.current_device())
    size = mla.page_nbytes(1, cache_kind="indexed", cache_format="deepseek_v4")
    stride = (1 << 31) + 4096
    storage = torch.empty(stride + size, dtype=torch.uint8, device=device)
    cache = storage.as_strided((2, size), (stride, 1))
    cache.fill_(0xA5)
    source = torch.randn((1, 512), dtype=torch.bfloat16, device=device)
    slots = torch.ones(1, dtype=torch.int64, device=device)
    plan = mla.plan_cache_writer(
        mla.CacheWriterQuery(
            max_rows=1,
            page_size=1,
            cache_kind="indexed",
            cache_format="deepseek_v4",
            slot_dtype="int64",
        ),
        device=device,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="v4-large-page-offset",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(source, cache, slots)
                    ),
                ),
            )
        )
        session.freeze()
        mla.write_cache(source, cache, slots, plan=plan)
        payload, footer = records(source)
        assert (cache[0] == 0xA5).all()
        torch.testing.assert_close(cache[1, :576], payload[0], atol=0, rtol=0)
        torch.testing.assert_close(cache[1, 576:584], footer[0], atol=0, rtol=0)
        assert (cache[1, 584:] == 0xA5).all()
