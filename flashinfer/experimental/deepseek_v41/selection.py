# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Causal block8 candidate selection, including the newest visible block."""

from ._utils import _overlaps

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _block_max(
    SCORE,
    VISIBLE,
    OUT,
    END,
    NK: tl.constexpr,
    NB: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    batch = tl.program_id(1)
    blocks = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    token = blocks[:, None] * 8 + tl.arange(0, 8)[None, :]
    visible = tl.load(VISIBLE + batch)
    values = tl.load(
        SCORE + batch * SCORE_STRIDE + token,
        (token < NK) & (token < visible),
        -float("inf"),
    ).to(tl.float32)
    maximum = tl.max(values, 1)
    maximum = tl.where(
        (visible > 0) & (blocks == (visible - 1) // 8), float("inf"), maximum
    )
    tl.store(OUT + batch * OUT_STRIDE + blocks, maximum, blocks < NB)
    if tl.program_id(0) == 0:
        tl.store(END + batch, (visible + 7) // 8)


def _check(scores, visible):
    if scores.device.type != "cuda" or torch.cuda.get_device_capability(
        scores.device
    ) not in ((10, 0), (10, 3)):
        raise ValueError("V4.1 selection currently requires SM100/SM103")
    if (
        scores.ndim != 2
        or scores.dtype != torch.bfloat16
        or scores.stride(1) != 1
        or min(scores.shape) <= 0
    ):
        raise ValueError(
            "positive BF16 score matrix with contiguous last dimension required"
        )
    if (
        visible.shape != (scores.shape[0],)
        or visible.dtype != torch.int32
        or visible.device != scores.device
        or not visible.is_contiguous()
    ):
        raise ValueError(
            "visible must be contiguous CUDA int32 with one length per query"
        )


def candidate_workspace(scores, *, topk_blocks=2048):
    import deep_select

    if (
        scores.ndim != 2
        or min(scores.shape) <= 0
        or scores.dtype != torch.bfloat16
        or scores.device.type != "cuda"
        or not 1 <= topk_blocks <= 4096
    ):
        raise ValueError("positive scores and topk_blocks in [1,4096] required")
    nq, nk = scores.shape
    blocks = tr.cdiv(nk, 8)
    count = min(blocks, topk_blocks)
    input_alignment, output_alignment = deep_select.get_stride_requirement()
    stride = tr.cdiv(blocks * 2, input_alignment) * input_alignment // 2
    out_stride = tr.cdiv(count * 4, output_alignment) * output_alignment // 4
    return dict(
        block_logits=torch.empty_strided(
            (nq, blocks), (stride, 1), dtype=torch.bfloat16, device=scores.device
        ),
        block_ends=torch.empty((nq,), dtype=torch.int32, device=scores.device),
        out=torch.empty_strided(
            (nq, count), (out_stride, 1), dtype=torch.int32, device=scores.device
        ),
    )


def candidate_blocks(scores, visible, *, topk_blocks=2048, workspace=None):
    import deep_select

    _check(scores, visible)
    if not 1 <= topk_blocks <= 4096:
        raise ValueError("topk_blocks must be in [1,4096]")
    if workspace is None:
        workspace = candidate_workspace(scores, topk_blocks=topk_blocks)
    nq, nk = scores.shape
    nb = tr.cdiv(nk, 8)
    count = min(nb, topk_blocks)
    logits, ends, out = (
        workspace[name] for name in ("block_logits", "block_ends", "out")
    )
    ia, oa = deep_select.get_stride_requirement()
    for value, shape, dtype, alignment in (
        (logits, (nq, nb), torch.bfloat16, ia),
        (out, (nq, count), torch.int32, oa),
    ):
        if (
            value.shape != shape
            or value.dtype != dtype
            or value.device != scores.device
            or value.stride(1) != 1
            or value.stride(0) < shape[1]
            or value.stride(0) * value.element_size() % alignment
        ):
            raise ValueError("invalid candidate workspace matrix layout")
    if (
        ends.shape != (nq,)
        or ends.dtype != torch.int32
        or ends.device != scores.device
        or not ends.is_contiguous()
    ):
        raise ValueError("invalid candidate workspace lengths")

    for index, value in enumerate((logits, ends, out)):
        if any(
            _overlaps(value, other)
            for other in (scores, visible, *(logits, ends, out)[:index])
        ):
            raise ValueError(
                "candidate workspace storage must not overlap inputs or other workspace tensors"
            )
    with torch.cuda.device(scores.device):
        _block_max[(tr.cdiv(nb, 128), nq)](
            scores,
            visible,
            logits,
            ends,
            nk,
            nb,
            scores.stride(0),
            logits.stride(0),
            128,
            num_warps=4,
        )
        deep_select.topk(
            logits,
            count,
            end=ends,
            sorted_index=True,
            indices_type=torch.int32,
            output_idx=out,
            return_value=False,
            idx_oob_fill_value=-1,
        )
    return out


@tr.jit
def _prepare_tokens(
    SCORE,
    VISIBLE,
    CANDIDATES,
    LOGITS,
    ENDS,
    WIDTH: tl.constexpr,
    NC: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    CANDIDATE_STRIDE: tl.constexpr,
    LOGIT_STRIDE: tl.constexpr,
    SPARSE: tl.constexpr,
    COPY: tl.constexpr,
    BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    visible = tl.load(VISIBLE + row)
    if SPARSE:
        block = tl.load(
            CANDIDATES + row * CANDIDATE_STRIDE + column // 8,
            column < WIDTH,
            -1,
        )
        position = block * 8 + column % 8
        valid = (block >= 0) & (position < visible)
    else:
        valid = column < visible
    if COPY:
        value = tl.load(SCORE + row * SCORE_STRIDE + column, column < WIDTH, 0)
        tl.store(
            LOGITS + row * LOGIT_STRIDE + column,
            tl.where(valid, value, -float("inf")),
            column < WIDTH,
        )
    if tl.program_id(0) == 0:
        if SPARSE:
            ci = tl.arange(0, CBLOCK)
            blocks = tl.load(CANDIDATES + row * CANDIDATE_STRIDE + ci, ci < NC, -1)
            sizes = tl.where(
                (ci < NC) & (blocks >= 0),
                tl.minimum(tl.maximum(visible - blocks * 8, 0), 8),
                0,
            )
            end = tl.sum(sizes)
        else:
            end = visible
        tl.store(ENDS + row, end)


@tr.jit
def _token_ids(
    LOCAL,
    CANDIDATES,
    OUT,
    COUNT: tl.constexpr,
    LOCAL_STRIDE: tl.constexpr,
    CANDIDATE_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    local = tl.load(LOCAL + row * LOCAL_STRIDE + col, col < COUNT, -1)
    block = tl.load(
        CANDIDATES + row * CANDIDATE_STRIDE + local // 8,
        (col < COUNT) & (local >= 0),
        -1,
    )
    position = tl.where(local >= 0, block * 8 + local % 8, -1)
    tl.store(OUT + row * OUT_STRIDE + col, position, col < COUNT)


def token_workspace(scores, *, topk_tokens=512):
    """Aligned score copy, dynamic visible lengths and local/global ID outputs."""
    import deep_select

    if (
        scores.ndim != 2
        or min(scores.shape) <= 0
        or scores.dtype != torch.bfloat16
        or scores.device.type != "cuda"
        or not 1 <= topk_tokens <= 4096
    ):
        raise ValueError("positive BF16 scores and topk_tokens in [1,4096] required")
    nq, width = scores.shape
    count = min(width, topk_tokens)
    ia, oa = deep_select.get_stride_requirement()
    stride = tr.cdiv(width * 2, ia) * ia // 2
    out_stride = tr.cdiv(count * 4, oa) * oa // 4
    return dict(
        logits=torch.empty_strided(
            (nq, width), (stride, 1), device=scores.device, dtype=torch.bfloat16
        ),
        ends=torch.empty(nq, device=scores.device, dtype=torch.int32),
        local_ids=torch.empty_strided(
            (nq, count), (out_stride, 1), device=scores.device, dtype=torch.int32
        ),
        out=torch.empty_strided(
            (nq, count), (out_stride, 1), device=scores.device, dtype=torch.int32
        ),
    )


def token_topk(scores, visible, *, candidates=None, topk_tokens=512, workspace=None):
    import deep_select

    _check(scores, visible)
    if not 1 <= topk_tokens <= 4096:
        raise ValueError("topk_tokens must be in [1,4096]")
    nq, width = scores.shape
    sparse = candidates is not None
    if sparse and (
        candidates.ndim != 2
        or candidates.shape != (nq, width // 8)
        or width % 8
        or candidates.dtype != torch.int32
        or candidates.device != scores.device
        or candidates.stride(1) != 1
        or candidates.stride(0) < candidates.shape[1]
    ):
        raise ValueError(
            "candidate int32 [queries,score_width/8] with contiguous last dimension required"
        )
    if workspace is None:
        workspace = token_workspace(scores, topk_tokens=topk_tokens)
    logits, ends, local_ids, out = (
        workspace[name] for name in ("logits", "ends", "local_ids", "out")
    )
    count = min(width, topk_tokens)
    ia, oa = deep_select.get_stride_requirement()
    for value, shape, dtype, alignment in (
        (logits, (nq, width), torch.bfloat16, ia),
        (local_ids, (nq, count), torch.int32, oa),
        (out, (nq, count), torch.int32, oa),
    ):
        if (
            value.shape != shape
            or value.dtype != dtype
            or value.device != scores.device
            or value.stride(1) != 1
            or value.stride(0) < shape[1]
            or value.stride(0) * value.element_size() % alignment
            # Row stride alignment (1024B for logits) is not a requirement
            # on the allocation base. CUDA TMA requires a 16B global base.
            or value.data_ptr() % 16
        ):
            raise ValueError("invalid token workspace matrix layout/alignment")
    if (
        ends.shape != (nq,)
        or ends.dtype != torch.int32
        or ends.device != scores.device
        or not ends.is_contiguous()
    ):
        raise ValueError("invalid token workspace lengths")
    buffers = (logits, ends, local_ids, out)
    inputs = (scores, visible) + ((candidates,) if sparse else ())
    for index, value in enumerate(buffers):
        if any(_overlaps(value, other) for other in (*inputs, *buffers[:index])):
            raise ValueError(
                "token workspace must not overlap inputs or other workspace buffers"
            )
    nc = width // 8 if sparse else 1
    direct = (
        scores.stride(0) >= width
        and scores.stride(0) * 2 % ia == 0
        and scores.data_ptr() % 16 == 0
    )
    with torch.cuda.device(scores.device):
        if sparse or not direct:
            _prepare_tokens[(1 if direct else tr.cdiv(width, 512), nq)](
                scores,
                visible,
                candidates,
                logits,
                ends,
                width,
                nc,
                scores.stride(0),
                candidates.stride(0) if sparse else 0,
                logits.stride(0),
                sparse,
                not direct,
                512,
                tr.next_power_of_2(nc),
                num_warps=4,
            )
        deep_select.topk(
            scores if direct else logits,
            count,
            end=visible if direct and not sparse else ends,
            sorted_index=True,
            indices_type=torch.int32,
            output_idx=local_ids if sparse else out,
            return_value=False,
            idx_oob_fill_value=-1,
        )
        if sparse:
            _token_ids[(nq,)](
                local_ids,
                candidates,
                out,
                count,
                local_ids.stride(0),
                candidates.stride(0),
                out.stride(0),
                tr.next_power_of_2(count),
                num_warps=4,
            )
    return out
