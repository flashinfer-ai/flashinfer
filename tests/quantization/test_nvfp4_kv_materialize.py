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

import pytest
import torch

from flashinfer.quantization.kernels import nvfp4_kv_materialize_fp8
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] < 10,
    reason="requires SM100+",
)

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _swizzle_v_scales(linear: torch.Tensor) -> torch.Tensor:
    pages, heads, tokens, scale_dim = linear.shape
    return (
        linear.reshape(pages, heads, tokens // 4, 4, 4, scale_dim // 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape(pages, heads, tokens, scale_dim)
        .contiguous()
    )


def _add_page_padding(tensor: torch.Tensor, padding: int = 64) -> torch.Tensor:
    """Copy a compact tensor into a view with padding between physical pages."""
    compact_page_stride = tensor.stride(0)
    padded_page_stride = compact_page_stride + padding
    storage_size = padded_page_stride * (tensor.shape[0] - 1) + compact_page_stride
    storage = torch.empty(storage_size, dtype=tensor.dtype, device=tensor.device)
    padded = torch.as_strided(
        storage,
        size=tensor.shape,
        stride=(padded_page_stride, *tensor.stride()[1:]),
    )
    padded.copy_(tensor)
    return padded


def _dequantize_reference(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        _E2M1_VALUES + [-v for v in _E2M1_VALUES],
        dtype=torch.float16,
        device=data.device,
    )
    values = torch.stack(
        (lut[(data & 0x0F).long()], lut[(data >> 4).long()]), dim=-1
    ).flatten(-2)
    expanded_scales = scales.to(torch.float16).repeat_interleave(16, dim=-1)
    # E2M1 * E4M3 is exact in FP16, so one rounding to E4M3 matches the kernel.
    return (values * expanded_scales).to(torch.float8_e4m3fn)


def _random_cache(num_pages, num_heads, page_size, head_dim, generator):
    data = torch.randint(
        0,
        256,
        (num_pages, num_heads, page_size, head_dim // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scales = (
        torch.rand(
            (num_pages, num_heads, page_size, head_dim // 16),
            device="cuda",
            generator=generator,
        )
        * 1.5
        + 0.01
    ).to(torch.float8_e4m3fn)
    return data, scales


@pytest.mark.parametrize("head_dim", [128, 256, 512])
@pytest.mark.parametrize("page_size", [16, 64, 128])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
@pytest.mark.parametrize("page_strided", [False, True])
@pytest.mark.parametrize("provide_out", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_nvfp4_kv_materialize_fp8_matches_reference(
    head_dim: int,
    page_size: int,
    uses_shared_paged_kv_idx: bool,
    page_strided: bool,
    provide_out: bool,
    compact: bool,
) -> None:
    num_physical_pages = 13
    num_heads = 2
    generator = torch.Generator(device="cuda").manual_seed(head_dim * 1000 + page_size)
    k_data, k_scales = _random_cache(
        num_physical_pages, num_heads, page_size, head_dim, generator
    )
    v_data, v_scales = _random_cache(
        num_physical_pages, num_heads, page_size, head_dim, generator
    )

    # Partial last pages, a single-token request, and a full-width request.
    seq_lens_host = [2 * page_size + 3, 1, 4 * page_size]
    max_kv_len = max(seq_lens_host)
    max_pages = (max_kv_len + page_size - 1) // page_size
    batch_size = len(seq_lens_host)
    seq_lens = torch.tensor(seq_lens_host, dtype=torch.int32, device="cuda")
    # Out-of-range values in padding columns must never be dereferenced.
    k_table = torch.full((batch_size, max_pages + 2), 10**6, dtype=torch.int32)
    v_table = torch.full_like(k_table, -1)
    k_rows = [[3, 1, 12], [0], [6, 2, 3, 9]]  # repeated and non-monotonic ids
    v_rows = [[4, 8, 11], [7], [5, 10, 0, 1]]
    for b, (k_row, v_row) in enumerate(zip(k_rows, v_rows, strict=True)):
        k_table[b, : len(k_row)] = torch.tensor(k_row)
        v_table[b, : len(v_row)] = torch.tensor(v_row)
    if uses_shared_paged_kv_idx:
        v_table = k_table
        block_tables = k_table.cuda()
    else:
        block_tables = torch.stack((k_table, v_table), dim=1).cuda()

    k_cache, v_cache = k_data, v_data
    k_sf = k_scales.view(torch.uint8)
    v_sf = _swizzle_v_scales(v_scales).view(torch.uint8)
    if page_strided:
        k_cache, v_cache, k_sf, v_sf = map(
            _add_page_padding, (k_cache, v_cache, k_sf, v_sf)
        )

    # Compact mode reserves exactly each request's active pages.
    active_pages = [(n + page_size - 1) // page_size for n in seq_lens_host]
    page_starts = [sum(active_pages[:b]) for b in range(batch_size + 1)]
    kv_page_indptr = num_pages = None
    if compact:
        kv_page_indptr = torch.tensor(page_starts, dtype=torch.int32, device="cuda")
        num_pages = page_starts[-1]
    else:
        page_starts = [b * max_pages for b in range(batch_size + 1)]
    out_shape = (page_starts[-1] + 1, 2, num_heads, page_size, head_dim)
    out = None
    if provide_out:
        out = torch.full(out_shape, 1.0, dtype=torch.float8_e4m3fn, device="cuda")

    kv_fp8, fp8_tables = nvfp4_kv_materialize_fp8(
        k_cache,
        v_cache,
        k_sf,
        v_sf,
        block_tables,
        seq_lens,
        max_kv_len,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        out=out,
        kv_page_indptr=kv_page_indptr,
        num_pages=num_pages,
    )
    torch.cuda.synchronize()

    assert tuple(kv_fp8.shape) == out_shape
    if out is not None:
        assert kv_fp8.data_ptr() == out.data_ptr()
    assert fp8_tables.dtype == torch.int32
    assert tuple(fp8_tables.shape) == (batch_size, max_pages)
    assert torch.all(kv_fp8[0].view(torch.uint8) == 0)

    expected_k = _dequantize_reference(k_data, k_scales)
    expected_v = _dequantize_reference(v_data, v_scales)
    fp8_tables = fp8_tables.cpu()
    for b, seq_len in enumerate(seq_lens_host):
        num_pages = (seq_len + page_size - 1) // page_size
        for j in range(max_pages):
            page = int(fp8_tables[b, j])
            if j >= num_pages:
                assert page == 0
                continue
            assert page == page_starts[b] + j + 1
            for plane, table, expected in (
                (0, k_table, expected_k),
                (1, v_table, expected_v),
            ):
                torch.testing.assert_close(
                    kv_fp8[page, plane].view(torch.uint8),
                    expected[int(table[b, j])].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )


def test_nvfp4_kv_materialize_fp8_rejects_misaligned_page_stride() -> None:
    num_pages, num_heads, page_size, head_dim = 3, 1, 16, 128
    generator = torch.Generator(device="cuda").manual_seed(0)
    data, scales = _random_cache(num_pages, num_heads, page_size, head_dim, generator)
    misaligned = _add_page_padding(data, padding=4)
    with pytest.raises(ValueError, match="page stride"):
        nvfp4_kv_materialize_fp8(
            misaligned,
            misaligned,
            scales.view(torch.uint8),
            scales.view(torch.uint8),
            torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
            torch.ones(1, dtype=torch.int32, device="cuda"),
            1,
        )
