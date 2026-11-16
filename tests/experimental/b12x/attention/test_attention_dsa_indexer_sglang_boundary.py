from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch

from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference
from b12x.attention.dsa_indexer._impl import (
    IndexerContiguousMetadata,
    IndexerPagedDecodeMetadata,
    build_paged_mqa_schedule_metadata,
    clear_indexer_caches,
    contiguous_logits,
    paged_decode_logits,
)


_SGLANG_PYTHON_ROOT = Path("/home/luke/projects/sglang/python")


def _import_sglang_dsa_indexer():
    if not _SGLANG_PYTHON_ROOT.exists():
        pytest.skip(f"sglang sources not found at {_SGLANG_PYTHON_ROOT}")
    root = str(_SGLANG_PYTHON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("sglang.srt.layers.attention.nsa.nsa_indexer")
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        pytest.skip(f"unable to import sglang NSA indexer: {exc}")
    if not (
        hasattr(module.Indexer, "_use_b12x_indexer")
        or hasattr(module.Indexer, "_use_b12x_mla_indexer")
    ):
        pytest.skip(
            "sglang NSA/DSA indexer no longer exposes the legacy b12x boundary hooks"
        )
    return module


class _FakePool:
    page_size = 64

    def __init__(self, index_k_cache: torch.Tensor):
        self._index_k_cache = index_k_cache

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        del layer_id
        return self._index_k_cache

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
        k_out: torch.Tensor | None = None,
        s_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del layer_id, max_seq_len
        page_size = self.page_size
        data_bytes = page_size * 128
        k_bytes = (
            k_out[:seq_len_sum]
            if k_out is not None
            else torch.empty(
                (seq_len_sum, 128), dtype=torch.uint8, device=self._index_k_cache.device
            )
        )
        scale_bytes = (
            s_out[:seq_len_sum]
            if s_out is not None
            else torch.empty(
                (seq_len_sum, 4), dtype=torch.uint8, device=self._index_k_cache.device
            )
        )
        write_row = 0
        for batch_row in range(page_indices.shape[0]):
            seq_len = int(seq_len_tensor[batch_row].item())
            for token_pos in range(seq_len):
                page_col = token_pos // page_size
                slot = token_pos % page_size
                page_id = int(page_indices[batch_row, page_col].item())
                k_bytes[write_row] = self._index_k_cache[
                    page_id,
                    slot * 128 : (slot + 1) * 128,
                ]
                scale_bytes[write_row] = self._index_k_cache[
                    page_id,
                    data_bytes + slot * 4 : data_bytes + (slot + 1) * 4,
                ]
                write_row += 1
        assert write_row == seq_len_sum
        return k_bytes, scale_bytes


class _FakeDecodeMode:
    def is_decode_or_idle(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        del include_v2
        return False


class _FakeExtendMode:
    def is_decode_or_idle(self) -> bool:
        return False

    def is_extend_without_speculative(self) -> bool:
        return True

    def is_target_verify(self) -> bool:
        return False

    def is_draft_extend(self, include_v2: bool = False) -> bool:
        del include_v2
        return False


class _FakeAttnBackend:
    def __init__(
        self,
        *,
        impl_name: str,
        device: torch.device,
        topk: int,
        num_heads: int,
        v_head_dim: int = 512,
    ) -> None:
        self.nsa_decode_impl = impl_name
        self.nsa_prefill_impl = impl_name
        self.device = device
        self.topk = int(topk)
        self.num_heads = int(num_heads)
        self.model_v_head_dim = int(v_head_dim)
        self._workspaces: dict[tuple[str, torch.device], object] = {}

    def _get_b12x_workspace(self, *, mode: str, v_head_dim: int):
        from b12x.attention._shared.workspace import B12XAttentionWorkspace

        normalized_mode = "verify" if mode == "target_verify" else mode
        key = (normalized_mode, self.device)
        workspace = self._workspaces.get(key)
        if workspace is not None:
            return workspace
        max_total_q = 64 if normalized_mode != "decode" else 32
        workspace = B12XAttentionWorkspace.for_fixed_capacity(
            mode=normalized_mode,
            device=self.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            num_q_heads=self.num_heads,
            indexer_num_q_heads=self.num_heads,
            head_dim=576,
            v_head_dim=int(v_head_dim),
            topk=self.topk,
            max_page_table_width=max(self.topk, 1),
            max_total_q=max_total_q,
            max_batch=32,
            max_paged_q_rows=32 if normalized_mode == "decode" else max_total_q,
            max_kv_rows=4096 if normalized_mode != "decode" else 0,
            page_size=64,
            use_cuda_graph=True,
        )
        self._workspaces[key] = workspace
        return workspace

    def get_b12x_indexer_paged_workspace(self, *, forward_batch):
        del forward_batch
        return self._get_b12x_workspace(mode="decode", v_head_dim=self.model_v_head_dim)


class _FakePagedMetadata:
    def __init__(
        self,
        *,
        page_table_1: torch.Tensor,
        real_page_table: torch.Tensor,
        seqlens_int32: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        extend_lens: list[int],
    ) -> None:
        self._page_table_1 = page_table_1
        self._real_page_table = real_page_table
        self._seqlens_int32 = seqlens_int32
        self._seqlens_expanded = seqlens_expanded
        self._extend_lens = extend_lens
        self.paged_mqa_schedule_metadata = None

    def get_page_table_1(self) -> torch.Tensor:
        return self._page_table_1

    def get_page_table_64(self) -> torch.Tensor:
        return self._real_page_table

    def get_seqlens_int32(self) -> torch.Tensor:
        return self._seqlens_int32

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self._seqlens_expanded

    def get_nsa_extend_len_cpu(self) -> list[int]:
        return self._extend_lens

    def topk_transform(self, logits: torch.Tensor, topk: int, **kwargs) -> torch.Tensor:
        del kwargs
        rows = logits.shape[0]
        output = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
        gather_k = min(topk, logits.shape[1], self._page_table_1.shape[1])
        if gather_k == 0:
            return output
        topk_pos = torch.argsort(logits, dim=1, descending=True, stable=True)[
            :, :gather_k
        ]
        topk_values = torch.gather(logits, 1, topk_pos)
        gathered = torch.gather(self._page_table_1[:rows], 1, topk_pos.to(torch.long))
        output[:, :gather_k] = torch.where(
            torch.isfinite(topk_values),
            gathered,
            torch.full_like(gathered, -1),
        )
        return output


class _FakeRaggedMetadata:
    def __init__(
        self,
        *,
        page_table_1: torch.Tensor,
        real_page_table: torch.Tensor,
        seqlens_expanded: torch.Tensor,
        extend_lens: list[int],
        indexer_seq_lens: torch.Tensor,
        k_start: torch.Tensor,
        k_end: torch.Tensor,
        token_to_batch_idx: torch.Tensor,
    ) -> None:
        self._page_table_1 = page_table_1
        self._real_page_table = real_page_table
        self._seqlens_expanded = seqlens_expanded
        self._extend_lens = extend_lens
        self._indexer_seq_lens = indexer_seq_lens
        self._k_start = k_start
        self._k_end = k_end
        self._token_to_batch_idx = token_to_batch_idx
        self.attn_metadata = type(
            "_FakeAttnMetadata", (), {"topk_indices_offset": None}
        )()

    def get_page_table_1(self) -> torch.Tensor:
        return self._page_table_1

    def get_page_table_64(self) -> torch.Tensor:
        return self._real_page_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self._seqlens_expanded

    def get_nsa_extend_len_cpu(self) -> list[int]:
        return self._extend_lens

    def get_indexer_kvcache_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._k_start, self._k_end

    def get_indexer_seq_len(self) -> torch.Tensor:
        return self._indexer_seq_lens

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        return self._indexer_seq_lens.cpu()

    def get_indexer_seq_lens_sum(self) -> int:
        return int(self._indexer_seq_lens.sum().item())

    def get_indexer_max_seq_len(self) -> int:
        return int(self._indexer_seq_lens.max().item())

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self._token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        ks: torch.Tensor | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        ke_offset: torch.Tensor | None = None,
        batch_idx_list: list[int] | torch.Tensor | None = None,
        topk_indices_offset_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_q, batch_idx_list, topk_indices_offset_override
        if ks is None:
            raise RuntimeError("ragged topk_transform requires ks")
        lengths = (
            ke_offset
            if ke_offset is not None
            else self._seqlens_expanded[: logits.shape[0]]
        )
        output = torch.full(
            (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
        )
        gather_k = min(topk, logits.shape[1])
        if gather_k == 0:
            return output
        positions = torch.arange(
            logits.shape[1], device=logits.device, dtype=torch.int32
        ).unsqueeze(0)
        row_start = ks.unsqueeze(1)
        row_end = row_start + lengths.unsqueeze(1)
        valid = (positions >= row_start) & (positions < row_end)
        masked_logits = torch.where(
            valid, logits, torch.full_like(logits, float("-inf"))
        )
        topk_pos = torch.argsort(masked_logits, dim=1, descending=True, stable=True)[
            :, :gather_k
        ]
        topk_values = torch.gather(masked_logits, 1, topk_pos)
        output[:, :gather_k] = torch.where(
            torch.isfinite(topk_values),
            topk_pos.to(torch.int32),
            torch.full_like(topk_pos, -1, dtype=torch.int32),
        )
        return output


def _get_b12x_impl_name(module) -> str:
    return "b12x" if hasattr(module.Indexer, "_use_b12x_indexer") else "b12x_mla"


def _get_use_b12x_indexer_method(module):
    method = getattr(module.Indexer, "_use_b12x_indexer", None)
    if method is None:
        method = getattr(module.Indexer, "_use_b12x_mla_indexer")
    return method


def _make_fake_indexer(module, *, topk: int, num_heads: int):
    class _FakeIndexer:
        index_topk = topk
        layer_id = 0
        n_heads = num_heads
        _b12x_indexer_phantoms = None
        _use_b12x_indexer = staticmethod(_get_use_b12x_indexer_method(module))
        _get_b12x_paged_topk = module.Indexer._get_b12x_paged_topk
        _get_b12x_ragged_topk = module.Indexer._get_b12x_ragged_topk
        _get_b12x_indexer_phantoms = module.Indexer._get_b12x_indexer_phantoms

        @staticmethod
        def _should_chunk_mqa_logits(num_q: int, num_k: int, device: torch.device):
            del num_q, num_k, device
            return False, 0

    return _FakeIndexer()


def _make_paged_candidate_tables(
    *,
    page_starts: list[int],
    seqlens: list[int],
    width_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = len(seqlens)
    page_table_1 = torch.full(
        (rows, width_blocks * 64), -1, dtype=torch.int32, device=device
    )
    real_page_table = torch.full(
        (rows, width_blocks), -1, dtype=torch.int32, device=device
    )
    for row_idx, (page_start, seq_len) in enumerate(
        zip(page_starts, seqlens, strict=True)
    ):
        block_count = (int(seq_len) + 63) // 64
        if block_count == 0:
            continue
        real_page_table[row_idx, :block_count] = torch.arange(
            page_start,
            page_start + block_count,
            dtype=torch.int32,
            device=device,
        )
        token_ids = torch.arange(
            page_start * 64,
            (page_start + block_count) * 64,
            dtype=torch.int32,
            device=device,
        )
        page_table_1[row_idx, :seq_len] = token_ids[:seq_len]
    return page_table_1, real_page_table


def test_sglang_b12x_indexer_paged_boundary_matches_b12x_reference() -> None:
    module = _import_sglang_dsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_100)

    page_starts = [4, 8, 12]
    seqlens = torch.tensor([65, 96, 127], dtype=torch.int32)
    width_blocks = 2
    num_tokens = (max(page_starts) + width_blocks) * 64
    num_heads = 3
    topk = 4
    q_rows = len(page_starts)

    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=torch.device("cpu"),
    )

    fake_indexer = _make_fake_indexer(module, topk=topk, num_heads=num_heads)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(
                impl_name=_get_b12x_impl_name(module),
                device=torch.device("cpu"),
                topk=topk,
                num_heads=num_heads,
            ),
            "seq_lens": seqlens,
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1],
    )

    actual = module.Indexer._get_topk_paged(
        fake_indexer,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected_logits = paged_decode_logits(
        q_fp8=q_fp8,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=IndexerPagedDecodeMetadata(
            real_page_table=real_page_table,
            cache_seqlens_int32=seqlens,
        ),
    )
    expected = metadata.topk_transform(expected_logits, topk)

    assert torch.equal(actual, expected)


def test_sglang_b12x_indexer_paged_boundary_respects_active_decode_rows() -> None:
    module = _import_sglang_dsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_102)

    page_starts = [2, 6, 10, 14]
    seqlens = torch.tensor([70, 129, 66, 95], dtype=torch.int32)
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    num_heads = 4
    topk = 5
    q_rows = len(page_starts)
    active_rows = 3

    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((q_rows, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=torch.device("cpu"),
    )

    fake_indexer = _make_fake_indexer(module, topk=topk, num_heads=num_heads)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(
                impl_name=_get_b12x_impl_name(module),
                device=torch.device("cpu"),
                topk=topk,
                num_heads=num_heads,
            ),
            "seq_lens": seqlens,
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1, 0],
    )

    actual = module.Indexer._get_topk_paged(
        fake_indexer,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    expected_logits = paged_decode_logits(
        q_fp8=q_fp8[:active_rows],
        weights=weights[:active_rows],
        index_k_cache=index_k_cache,
        metadata=IndexerPagedDecodeMetadata(
            real_page_table=real_page_table[:active_rows],
            cache_seqlens_int32=seqlens[:active_rows],
        ),
    )
    expected = metadata.topk_transform(expected_logits, topk)
    padding = torch.full((1, topk), -1, dtype=torch.int32)

    assert torch.equal(actual, torch.cat([expected, padding], dim=0))


def test_sglang_b12x_indexer_ragged_boundary_matches_b12x_reference() -> None:
    module = _import_sglang_dsa_indexer()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_101)

    topk = 5
    seq_lens = torch.tensor([70, 130], dtype=torch.int32)
    extend_lens = [2, 3]
    seqlens_expanded = torch.tensor([70, 70, 130, 130, 130], dtype=torch.int32)
    token_to_batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
    k_start = torch.tensor([0, 0, 70, 70, 70], dtype=torch.int32)
    k_end = torch.tensor([70, 70, 200, 200, 200], dtype=torch.int32)
    num_tokens = (2 + 3) * 64
    num_heads = 2

    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32) / 3
    )
    q_fp8 = (
        torch.randn((6, num_heads, 128), generator=gen, dtype=torch.float32) / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn((6, num_heads, 1), generator=gen, dtype=torch.float32)
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=[1, 2],
        seqlens=seq_lens.tolist(),
        width_blocks=3,
        device=torch.device("cpu"),
    )

    fake_indexer = _make_fake_indexer(module, topk=topk, num_heads=num_heads)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeExtendMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(
                impl_name=_get_b12x_impl_name(module),
                device=torch.device("cpu"),
                topk=topk,
                num_heads=num_heads,
            ),
            "seq_lens": seq_lens,
        },
    )()
    metadata = _FakeRaggedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_expanded=seqlens_expanded,
        extend_lens=extend_lens,
        indexer_seq_lens=seq_lens,
        k_start=k_start,
        k_end=k_end,
        token_to_batch_idx=token_to_batch_idx,
    )

    actual = module.Indexer._get_topk_ragged(
        fake_indexer,
        False,
        fake_forward_batch,
        0,
        q_fp8,
        weights,
        metadata,
    )
    k_fp8_bytes, k_scale_bytes = (
        fake_forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
            0,
            seq_lens,
            real_page_table,
            int(seq_lens.sum().item()),
            int(seq_lens.max().item()),
        )
    )
    kv_fp8 = (
        k_fp8_bytes.view(torch.float8_e4m3fn),
        k_scale_bytes.view(torch.float32).squeeze(-1),
    )
    expected_logits = contiguous_logits(
        q_fp8=q_fp8[: k_start.numel()],
        weights=weights[: k_start.numel()],
        kv_fp8=kv_fp8,
        metadata=IndexerContiguousMetadata(
            k_start=k_start,
            k_end=k_end,
        ),
    )
    expected = metadata.topk_transform(expected_logits, topk, ks=k_start)
    padded_expected = torch.full((q_fp8.shape[0], topk), -1, dtype=torch.int32)
    padded_expected[: expected.shape[0]] = expected

    assert torch.equal(actual, padded_expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for graph capture coverage"
)
def test_sglang_b12x_indexer_paged_boundary_cuda_graph_capture() -> None:
    module = _import_sglang_dsa_indexer()
    device = torch.device("cuda")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(73_103)

    page_starts = [4, 8, 12, 16]
    seqlens = torch.tensor([96, 130, 71, 80], dtype=torch.int32, device=device)
    width_blocks = 3
    num_tokens = (max(page_starts) + width_blocks) * 64
    num_heads = 8
    topk = 6
    q_rows = len(page_starts)
    active_rows = 3

    index_k_cache = pack_index_k_cache_reference(
        torch.randn((num_tokens, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 3
    )
    q_fp8 = (
        torch.randn((q_rows, num_heads, 128), generator=gen, dtype=torch.float32).to(
            device=device
        )
        / 2
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (q_rows, num_heads, 1), generator=gen, dtype=torch.float32
    ).to(device=device)
    page_table_1, real_page_table = _make_paged_candidate_tables(
        page_starts=page_starts,
        seqlens=seqlens.tolist(),
        width_blocks=width_blocks,
        device=device,
    )

    fake_indexer = _make_fake_indexer(module, topk=topk, num_heads=num_heads)
    fake_forward_batch = type(
        "_FakeForwardBatch",
        (),
        {
            "forward_mode": _FakeDecodeMode(),
            "token_to_kv_pool": _FakePool(index_k_cache),
            "attn_backend": _FakeAttnBackend(
                impl_name=_get_b12x_impl_name(module),
                device=device,
                topk=topk,
                num_heads=num_heads,
            ),
            "seq_lens": seqlens,
        },
    )()
    metadata = _FakePagedMetadata(
        page_table_1=page_table_1,
        real_page_table=real_page_table,
        seqlens_int32=seqlens,
        seqlens_expanded=seqlens,
        extend_lens=[1, 1, 1, 0],
    )
    metadata.paged_mqa_schedule_metadata = build_paged_mqa_schedule_metadata(
        seqlens[:active_rows].contiguous(),
        64,
    )

    clear_indexer_caches()
    captured_out = None

    def run() -> None:
        nonlocal captured_out
        captured_out = module.Indexer._get_topk_paged(
            fake_indexer,
            fake_forward_batch,
            0,
            q_fp8,
            weights,
            metadata,
        )

    run()
    torch.cuda.synchronize(device)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize(device)

    expected_logits = paged_decode_logits(
        q_fp8=q_fp8[:active_rows],
        weights=weights[:active_rows],
        index_k_cache=index_k_cache,
        metadata=IndexerPagedDecodeMetadata(
            real_page_table=real_page_table[:active_rows],
            cache_seqlens_int32=seqlens[:active_rows],
            paged_mqa_schedule_metadata=metadata.paged_mqa_schedule_metadata,
        ),
    )
    expected = metadata.topk_transform(expected_logits, topk)
    padded_expected = torch.full((q_rows, topk), -1, dtype=torch.int32, device=device)
    padded_expected[:active_rows] = expected

    assert captured_out is not None
    assert torch.equal(captured_out, padded_expected)
