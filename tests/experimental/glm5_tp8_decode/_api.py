# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers over the public operator API, shared by all tests."""

from flashinfer import glm5_tp8_decode as api


def workspace(control, *, capacity=4096, operators=("moe", "attention")):
    # Creation compiles and initializes every requested kernel family.
    return api.create_glm5_tp8_decode_workspace(
        rank=control.rank,
        group=control.group,
        device=control.device,
        max_seq_len=capacity,
        operators=operators,
    )


def pack_moe(layer, rank, device=None):
    return api.pack_glm5_tp8_moe_weights(layer, rank=rank, device=device)


def prepare_moe(x, packed, out, ws, *, enable_pdl=True):
    return api.prepare_glm5_tp8_moe(
        x, packed, out=out, workspace=ws, enable_pdl=enable_pdl
    )


def pack_attention(layer, rank, device=None):
    return api.pack_glm5_tp8_attention_weights(layer, rank=rank, device=device)


def prepare_attention(
    x,
    packed,
    caches,
    rope,
    position,
    out,
    ws,
    *,
    indexer,
    selection=None,
    enable_pdl=True,
):
    return api.prepare_glm5_tp8_sparse_attention(
        x,
        packed,
        caches=caches,
        rope=rope,
        cur_pos=position,
        out=out,
        workspace=ws,
        indexer=indexer,
        selection=selection,
        enable_pdl=enable_pdl,
    )
