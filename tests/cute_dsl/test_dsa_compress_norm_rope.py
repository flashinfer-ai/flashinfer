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

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import get_compute_capability


HEAD_SIZE = 512
ROPE_HEAD_DIM = 64
NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM
QUANT_BLOCK = 64
TOKEN_STRIDE = NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
SCALE_DIM = NOPE_HEAD_DIM // QUANT_BLOCK + 1
KV_CACHE_TOKEN_BYTES = TOKEN_STRIDE + SCALE_DIM


def _fp8_dtype():
    if torch.version.hip and hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _check_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL is not available")
    if get_compute_capability(torch.device("cuda"))[0] < 10:
        pytest.skip("DSA CuTe DSL kernels require SM100+")


def _make_inputs(
    *,
    compress_ratio: int,
    overlap: bool,
    state_width: int,
    position: int,
    num_tokens: int = 2,
    block_size: int = 4,
):
    device = torch.device("cuda")
    window = (1 + int(overlap)) * compress_ratio
    first_pos = position - window + 1
    min_live_pos = max(first_pos, 0)
    max_live_pos = max(position, 0)
    last_block = max_live_pos // block_size
    block_table_width = max(last_block + 1, 1)
    num_state_blocks = num_tokens * block_table_width

    generator = torch.Generator(device=device).manual_seed(2026)
    state_cache = torch.randn(
        (num_state_blocks, block_size, 2 * state_width),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    token_to_req_indices = torch.arange(num_tokens, device=device, dtype=torch.int32)
    positions = torch.full((num_tokens,), position, device=device, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)

    block_table = torch.empty(
        (num_tokens, block_table_width), device=device, dtype=torch.int32
    )
    for req_idx in range(num_tokens):
        base = req_idx * block_table_width
        block_table[req_idx] = torch.arange(
            base, base + block_table_width, device=device, dtype=torch.int32
        )

    compressed_kv = torch.empty(
        (num_tokens, HEAD_SIZE), device=device, dtype=torch.float32
    )
    rms_norm_weight = torch.ones(HEAD_SIZE, device=device, dtype=torch.float32)
    rms_norm_eps = 1e-6

    compressed_pos = (position // compress_ratio) * compress_ratio
    cos_sin_cache = torch.empty(
        (compressed_pos + 1, ROPE_HEAD_DIM), device=device, dtype=torch.float32
    )
    cos_sin_cache[:, : ROPE_HEAD_DIM // 2] = 1.0
    cos_sin_cache[:, ROPE_HEAD_DIM // 2 :] = 0.0

    num_kv_blocks = math.ceil(num_tokens / block_size)
    k_cache = torch.zeros(
        (num_kv_blocks, block_size, KV_CACHE_TOKEN_BYTES),
        device=device,
        dtype=torch.uint8,
    )
    kv_slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)

    return {
        "state_cache": state_cache,
        "token_to_req_indices": token_to_req_indices,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "block_table": block_table,
        "block_size": block_size,
        "compressed_kv": compressed_kv,
        "rms_norm_weight": rms_norm_weight,
        "rms_norm_eps": rms_norm_eps,
        "cos_sin_cache": cos_sin_cache,
        "k_cache": k_cache,
        "kv_slot_mapping": kv_slot_mapping,
        "kv_cache_block_size": block_size,
        "kv_block_stride": k_cache.stride(0),
        "compress_ratio": compress_ratio,
        "overlap": overlap,
        "state_width": state_width,
        "window": window,
    }


def _reference_compress(data):
    state_cache = data["state_cache"]
    positions = data["positions"]
    token_to_req_indices = data["token_to_req_indices"]
    block_table = data["block_table"]
    block_size = data["block_size"]
    state_width = data["state_width"]
    compress_ratio = data["compress_ratio"]
    window = data["window"]

    refs = []
    for token_idx in range(positions.numel()):
        position = int(positions[token_idx].item())
        req_idx = int(token_to_req_indices[token_idx].item())
        start = position - window + 1
        kv_rows = []
        score_rows = []
        for row in range(window):
            pos = start + row
            block_index = pos // block_size
            block_offset = pos - block_index * block_size
            block_number = int(block_table[req_idx, block_index].item())
            head_offset = (row // compress_ratio) * HEAD_SIZE
            cache_row = state_cache[block_number, block_offset]
            kv_rows.append(cache_row[head_offset : head_offset + HEAD_SIZE])
            score_rows.append(
                cache_row[
                    state_width
                    + head_offset : state_width
                    + head_offset
                    + HEAD_SIZE
                ]
            )
        kv = torch.stack(kv_rows, dim=0)
        scores = torch.stack(score_rows, dim=0)
        refs.append((kv * torch.softmax(scores, dim=0)).sum(dim=0))
    return torch.stack(refs, dim=0)


def _extract_cache(cache, num_tokens: int, block_size: int):
    nope = []
    rope = []
    scales = []
    for slot in range(num_tokens):
        block_idx = slot // block_size
        offset = slot % block_size
        page = cache[block_idx].reshape(-1)
        value_base = offset * TOKEN_STRIDE
        scale_base = block_size * TOKEN_STRIDE + offset * SCALE_DIM
        nope.append(page[value_base : value_base + NOPE_HEAD_DIM])
        rope_bytes = page[value_base + NOPE_HEAD_DIM : value_base + TOKEN_STRIDE]
        rope.append(rope_bytes.view(torch.bfloat16))
        scales.append(page[scale_base : scale_base + SCALE_DIM])
    return (
        torch.stack(nope).contiguous(),
        torch.stack(rope).contiguous(),
        torch.stack(scales).contiguous(),
    )


def _dequant_nope(raw_uint8: torch.Tensor, scales_uint8: torch.Tensor):
    fp8 = raw_uint8.contiguous().view(_fp8_dtype()).to(torch.float32)
    scales = scales_uint8[:, : NOPE_HEAD_DIM // QUANT_BLOCK].to(torch.int32)
    scale_fp32 = torch.pow(
        torch.full_like(scales, 2, dtype=torch.float32),
        (scales - 127).to(torch.float32),
    )
    return fp8.reshape(
        raw_uint8.shape[0], NOPE_HEAD_DIM // QUANT_BLOCK, QUANT_BLOCK
    ) * scale_fp32[:, :, None]


@pytest.mark.parametrize(
    (
        "compress_ratio",
        "overlap",
        "state_width",
        "position",
        "block_size",
    ),
    [
        pytest.param(4, True, 1024, 7, 4, id="c4-overlap-boundary"),
        pytest.param(128, False, 512, 127, 128, id="c128-boundary"),
    ],
)
def test_dsa_compress_norm_rope_matches_reference(
    compress_ratio: int,
    overlap: bool,
    state_width: int,
    position: int,
    block_size: int,
):
    _check_device()
    from flashinfer.cute_dsl import dsa_compress_kv, dsa_norm_rope_store

    data = _make_inputs(
        compress_ratio=compress_ratio,
        overlap=overlap,
        state_width=state_width,
        position=position,
        block_size=block_size,
    )
    dsa_compress_kv(
        data["state_cache"],
        data["token_to_req_indices"],
        data["positions"],
        data["slot_mapping"],
        data["block_table"],
        data["block_size"],
        data["compressed_kv"],
        head_size=HEAD_SIZE,
        state_width=data["state_width"],
        compress_ratio=data["compress_ratio"],
        overlap=data["overlap"],
    )
    torch.cuda.synchronize()

    ref_compressed = _reference_compress(data)
    torch.testing.assert_close(
        data["compressed_kv"], ref_compressed, atol=5e-4, rtol=5e-4
    )

    dsa_norm_rope_store(
        data["compressed_kv"],
        data["positions"],
        data["slot_mapping"],
        data["rms_norm_weight"],
        data["rms_norm_eps"],
        data["cos_sin_cache"],
        data["k_cache"],
        data["kv_slot_mapping"],
        data["kv_cache_block_size"],
        data["kv_block_stride"],
        head_size=HEAD_SIZE,
        rope_head_dim=ROPE_HEAD_DIM,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
        compress_ratio=data["compress_ratio"],
    )
    torch.cuda.synchronize()

    actual_nope, actual_rope, actual_scales = _extract_cache(
        data["k_cache"], data["positions"].numel(), data["block_size"]
    )
    rrms = torch.rsqrt(
        (ref_compressed * ref_compressed).sum(dim=-1, keepdim=True) / HEAD_SIZE
        + data["rms_norm_eps"]
    )
    ref_normed = ref_compressed * rrms * data["rms_norm_weight"]
    ref_nope = ref_normed[:, :NOPE_HEAD_DIM].to(torch.bfloat16).to(torch.float32)
    ref_rope = ref_normed[:, NOPE_HEAD_DIM:].to(torch.bfloat16).to(torch.float32)
    actual_dequant = _dequant_nope(actual_nope, actual_scales).reshape(
        data["positions"].numel(), NOPE_HEAD_DIM
    )

    torch.testing.assert_close(actual_dequant, ref_nope, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(actual_rope.float(), ref_rope, atol=2e-2, rtol=2e-2)


def test_dsa_compress_norm_rope_c128_non_boundary_noop():
    _check_device()
    from flashinfer.cute_dsl import dsa_compress_kv, dsa_norm_rope_store

    data = _make_inputs(
        compress_ratio=128,
        overlap=False,
        state_width=512,
        position=126,
        block_size=128,
    )
    data["compressed_kv"].fill_(-123.0)
    expected_compressed = data["compressed_kv"].clone()
    expected_cache = data["k_cache"].clone()

    dsa_compress_kv(
        data["state_cache"],
        data["token_to_req_indices"],
        data["positions"],
        data["slot_mapping"],
        data["block_table"],
        data["block_size"],
        data["compressed_kv"],
        head_size=HEAD_SIZE,
        state_width=data["state_width"],
        compress_ratio=data["compress_ratio"],
        overlap=data["overlap"],
    )
    dsa_norm_rope_store(
        data["compressed_kv"],
        data["positions"],
        data["slot_mapping"],
        data["rms_norm_weight"],
        data["rms_norm_eps"],
        data["cos_sin_cache"],
        data["k_cache"],
        data["kv_slot_mapping"],
        data["kv_cache_block_size"],
        data["kv_block_stride"],
        head_size=HEAD_SIZE,
        rope_head_dim=ROPE_HEAD_DIM,
        quant_block=QUANT_BLOCK,
        token_stride=TOKEN_STRIDE,
        scale_dim=SCALE_DIM,
        compress_ratio=data["compress_ratio"],
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(data["compressed_kv"], expected_compressed)
    assert torch.equal(data["k_cache"], expected_cache)
