from __future__ import annotations

import math

import pytest
import torch

from b12x.attention._shared.mla.api import MLASparseDecodeMetadata, MLASparseExtendMetadata, sparse_mla_decode_forward, sparse_mla_extend_forward
from b12x.attention._shared.mla.packed import extract_packed_kv_runtime_views


class _FakeMLAWorkspace:
    def __init__(
        self,
        *,
        mode: str,
        topk: int = 4,
        max_total_q: int = 8,
        max_batch: int = 4,
        max_kv_rows: int = 0,
        use_cuda_graph: bool = False,
    ) -> None:
        self.mode = mode
        self.device = torch.device("cpu")
        self.dtype = torch.bfloat16
        self.kv_dtype = torch.uint8
        self.num_q_heads = 8
        self.head_dim = 256
        self.v_head_dim = 256
        self.topk = int(topk)
        self.max_total_q = int(max_total_q)
        self.max_batch = int(max_batch)
        self.max_kv_rows = int(max_kv_rows)
        self.fixed_capacity = True
        self.use_cuda_graph = bool(use_cuda_graph)
        self.max_chunks_per_row = 64
        self.sm_scale_tensor = None
        self.sm_scale_value = None
        self.kv_chunk_size_value = 1
        self.num_chunks_value = 1
        self.kv_chunk_size_ptr = torch.empty((1,), dtype=torch.int32)
        self.num_chunks_ptr = torch.empty((1,), dtype=torch.int32)
        tmp_shape = (
            self.max_total_q,
            self.num_q_heads,
            self.max_chunks_per_row,
            self.v_head_dim,
        )
        tmp_storage = torch.empty(math.prod(tmp_shape), dtype=self.dtype)
        self.tmp_output = tmp_storage.as_strided(
            tmp_shape,
            (
                self.num_q_heads * self.v_head_dim,
                self.v_head_dim,
                self.max_total_q * self.num_q_heads * self.v_head_dim,
                1,
            ),
        )
        self.tmp_lse = torch.empty(
            (self.max_total_q, self.num_q_heads, self.max_chunks_per_row),
            dtype=torch.float32,
        )
        self.output_buffer = self.tmp_output[:, :, 0, :]
        self.final_lse = torch.empty(
            (self.max_total_q, self.num_q_heads),
            dtype=torch.float32,
        )
        self.ragged_kv_cache = None
        self._contract_kv_rows = None
        self._contract_kv_scales = None

    def set_split_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        if num_chunks <= 0 or num_chunks > self.max_chunks_per_row:
            raise ValueError(
                f"num_chunks must be in [1, {self.max_chunks_per_row}], got {num_chunks}"
            )
        if kv_chunk_size <= 0:
            raise ValueError(f"kv_chunk_size must be positive, got {kv_chunk_size}")
        self.kv_chunk_size_ptr[0] = int(kv_chunk_size)
        self.num_chunks_ptr[0] = int(num_chunks)
        self.kv_chunk_size_value = int(kv_chunk_size)
        self.num_chunks_value = int(num_chunks)

    def set_decode_chunk_config(self, *, kv_chunk_size: int, num_chunks: int) -> None:
        self.set_split_chunk_config(
            kv_chunk_size=kv_chunk_size,
            num_chunks=num_chunks,
        )

    def bind(
        self,
        *,
        q: torch.Tensor,
        selected_indices: torch.Tensor,
        cache_seqlens_int32: torch.Tensor,
        nsa_cache_seqlens_int32: torch.Tensor,
    ):
        from b12x.attention.sparse_mla._scratch import build_sparse_mla_binding

        return build_sparse_mla_binding(
            scratch=self,
            q=q,
            selected_indices=selected_indices,
            cache_seqlens_int32=cache_seqlens_int32,
            nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
        )

    def gather_ragged_kv_rows(
        self,
        *,
        kv_cache: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> torch.Tensor:
        row_count = int(row_ids.shape[0])
        capacity = max(int(self.max_kv_rows), row_count, 1)
        if self.ragged_kv_cache is None:
            self.ragged_kv_cache = torch.empty(
                (capacity, *kv_cache.shape[1:]),
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            )
            self.max_kv_rows = capacity
            self._refresh_ragged_kv_contracts()
        if row_count:
            self.ragged_kv_cache[:row_count].copy_(kv_cache[row_ids.to(torch.long)])
        return self.ragged_kv_cache

    def contract_kv_tensors_for(
        self,
        kv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.ragged_kv_cache is None:
            return None, None
        if kv_cache.data_ptr() != self.ragged_kv_cache.data_ptr():
            return None, None
        return self._contract_kv_rows, self._contract_kv_scales

    def _refresh_ragged_kv_contracts(self) -> None:
        assert self.ragged_kv_cache is not None
        self._contract_kv_rows, self._contract_kv_scales = (
            extract_packed_kv_runtime_views(self.ragged_kv_cache)
        )


def _make_workspace(
    *,
    mode: str,
    topk: int = 4,
    max_total_q: int = 8,
    max_batch: int = 4,
    max_kv_rows: int = 0,
) -> _FakeMLAWorkspace:
    return _FakeMLAWorkspace(
        mode=mode,
        topk=topk,
        max_total_q=max_total_q,
        max_batch=max_batch,
        max_kv_rows=max_kv_rows,
    )


def _decode_binding(
    workspace: _FakeMLAWorkspace,
    q_all: torch.Tensor,
    metadata: MLASparseDecodeMetadata,
):
    return workspace.bind(
        q=q_all,
        selected_indices=metadata.page_table_1,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
    )


def _extend_binding(
    workspace: _FakeMLAWorkspace,
    q_all: torch.Tensor,
    metadata: MLASparseExtendMetadata,
):
    return workspace.bind(
        q=q_all,
        selected_indices=metadata.selected_token_offsets,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
    )


def test_sparse_mla_decode_keeps_query_head_shape(monkeypatch) -> None:
    workspace = _make_workspace(mode="decode")
    captured: dict[str, torch.Tensor | float | int] = {}

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        captured["q"] = q_all
        captured["page_table_1"] = page_table_1
        captured["active_token_counts"] = active_token_counts
        captured["kv_cache"] = kv_cache
        captured["sm_scale"] = sm_scale
        captured["d_v"] = v_head_dim
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((2, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((16, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=8,
    )

    output = sparse_mla_decode_forward(
        kv_cache=kv_cache,
        binding=_decode_binding(workspace, q_all, metadata),
        sm_scale=0.5,
        v_head_dim=256,
    )

    assert output.shape == (2, 8, 256)
    assert captured["q"].shape == (2, 8, 256)
    assert captured["page_table_1"].shape == (2, 4)
    assert captured["sm_scale"] == 0.5
    assert captured["d_v"] == 256


def test_sparse_mla_decode_with_lse_uses_reference_lse_base2(monkeypatch) -> None:
    workspace = _make_workspace(mode="decode")

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
        return_lse=False,
    ):
        del kv_cache, page_table_1, active_token_counts, sm_scale
        assert return_lse is True
        output = q_all[:, :, :v_head_dim].clone()
        lse = torch.empty((q_all.shape[0], q_all.shape[1]), dtype=torch.float32)
        lse[0].fill_(math.log2(3.0))
        lse[1].fill_(2.0)
        return output, lse

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((2, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((16, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=8,
    )

    output, lse_base2 = sparse_mla_decode_forward(
        kv_cache=kv_cache,
        binding=_decode_binding(workspace, q_all, metadata),
        sm_scale=0.5,
        v_head_dim=256,
        return_lse=True,
    )

    assert output.shape == (2, 8, 256)
    assert lse_base2.shape == (2, 8)
    expected_row0 = math.log2(3.0)
    assert torch.allclose(lse_base2[0], torch.full((8,), expected_row0))
    assert torch.allclose(lse_base2[1], torch.full((8,), 2.0))


def test_sparse_mla_decode_with_lse_natural_scales_reference_lse(
    monkeypatch,
) -> None:
    workspace = _make_workspace(mode="decode")

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
        return_lse=False,
    ):
        del kv_cache, page_table_1, active_token_counts, sm_scale
        assert return_lse is True
        output = q_all[:, :, :v_head_dim].clone()
        lse = torch.empty((q_all.shape[0], q_all.shape[1]), dtype=torch.float32)
        lse[0].fill_(math.log2(3.0))
        lse[1].fill_(2.0)
        return output, lse

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((2, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((16, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=8,
    )

    output, lse_natural = sparse_mla_decode_forward(
        kv_cache=kv_cache,
        binding=_decode_binding(workspace, q_all, metadata),
        sm_scale=0.5,
        v_head_dim=256,
        return_lse=True,
        lse_scale="natural",
    )

    assert output.shape == (2, 8, 256)
    assert lse_natural.shape == (2, 8)
    expected_row0 = math.log(3.0)
    assert torch.allclose(lse_natural[0], torch.full((8,), expected_row0))
    assert torch.allclose(lse_natural[1], torch.full((8,), 2.0 * math.log(2.0)))


def test_sparse_mla_extend_passes_runtime_metadata(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=6)

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        del kv_cache, page_table_1, active_token_counts, sm_scale, v_head_dim
        return q_all[:, :8, :].clone()

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((3, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([12, 12, 12], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=1,
        max_seq_len_k=12,
        mode="draft_extend",
    )

    output = sparse_mla_extend_forward(
        kv_cache=kv_cache,
        binding=_extend_binding(workspace, q_all, metadata),
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (3, 8, 256)


def test_mla_verify_workspace_allocates_split_buffers() -> None:
    workspace = _make_workspace(mode="verify", topk=6)

    assert workspace.mode == "verify"
    assert workspace.tmp_output is not None
    assert workspace.tmp_lse is not None


def test_workspace_ragged_kv_gather_reuses_fixed_capacity_buffer() -> None:
    workspace = _make_workspace(
        mode="extend",
        topk=6,
        max_total_q=8,
        max_batch=4,
        max_kv_rows=16,
    )
    kv_cache = torch.arange(24 * 656, dtype=torch.uint8).reshape(24, 1, 656)

    gathered = workspace.gather_ragged_kv_rows(
        kv_cache=kv_cache,
        row_ids=torch.tensor([2, 5, 7, 11], dtype=torch.int32),
    )

    assert gathered.shape == (16, 1, 656)
    assert torch.equal(
        gathered[:4], kv_cache[torch.tensor([2, 5, 7, 11], dtype=torch.long)]
    )
    assert workspace._contract_kv_rows is not None
    assert workspace._contract_kv_scales is not None

    data_ptr = gathered.data_ptr()
    gathered_again = workspace.gather_ragged_kv_rows(
        kv_cache=kv_cache,
        row_ids=torch.tensor([1, 3], dtype=torch.int32),
    )

    assert gathered_again.data_ptr() == data_ptr
    assert torch.equal(
        gathered_again[:2], kv_cache[torch.tensor([1, 3], dtype=torch.long)]
    )



def test_sparse_mla_verify_uses_reference_when_sm120_unavailable(monkeypatch) -> None:
    workspace = _make_workspace(mode="verify", topk=2048)
    captured: dict[str, object] = {}

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        del kv_cache, sm_scale
        captured["page_table_1"] = page_table_1
        captured["active_token_counts"] = active_token_counts
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((5, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.zeros((5, 2048), dtype=torch.int32)
    cache_seqlens = torch.full((1,), 12, dtype=torch.int32)
    nsa_cache_seqlens = torch.full((5,), 12, dtype=torch.int32)
    nsa_cu = torch.tensor([0, 5], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=5,
        max_seq_len_k=12,
        mode="target_verify",
    )

    output = sparse_mla_extend_forward(
        kv_cache=kv_cache,
        binding=_extend_binding(workspace, q_all, metadata),
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (5, 8, 256)
    assert captured["page_table_1"] is page_table_1
    assert torch.equal(captured["active_token_counts"], nsa_cache_seqlens)


def test_sparse_mla_extend_uses_reference_when_sm120_unavailable(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=2048)
    captured: dict[str, object] = {}

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        del kv_cache, page_table_1, sm_scale
        captured["active_token_counts"] = active_token_counts.clone()
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((5, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.zeros((5, 2048), dtype=torch.int32)
    cache_seqlens = torch.full((1,), 12, dtype=torch.int32)
    nsa_cache_seqlens = torch.tensor([1537, 1024, 257, 64, 0], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 5], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=5,
        max_seq_len_k=12,
        mode="extend",
    )

    output = sparse_mla_extend_forward(
        kv_cache=kv_cache,
        binding=_extend_binding(workspace, q_all, metadata),
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (5, 8, 256)
    assert torch.equal(captured["active_token_counts"], nsa_cache_seqlens)


def test_sparse_mla_decode_rejects_legacy_backend() -> None:
    workspace = _make_workspace(mode="decode")
    q_all = torch.ones((2, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((16, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.zeros((2, 4), dtype=torch.int32)
    cache_seqlens = torch.full((2,), 8, dtype=torch.int32)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=8,
    )

    with pytest.raises(ValueError, match="legacy sparse MLA kernels have been retired"):
        sparse_mla_decode_forward(
            kv_cache=kv_cache,
            binding=_decode_binding(workspace, q_all, metadata),
            sm_scale=1.0,
            v_head_dim=256,
            backend="legacy",
        )


def test_sparse_mla_extend_passes_active_token_counts_to_reference(monkeypatch) -> None:
    workspace = _make_workspace(mode="extend", topk=6)
    captured: dict[str, object] = {}

    def fake_sparse_mla_reference(
        *,
        q_all,
        kv_cache,
        page_table_1,
        active_token_counts=None,
        sm_scale,
        v_head_dim,
    ):
        del kv_cache, page_table_1, sm_scale
        captured["active_token_counts"] = active_token_counts.clone()
        return q_all[:, :, :v_head_dim].clone()

    monkeypatch.setattr(
        "b12x.attention._shared.mla.api.sparse_mla_reference",
        fake_sparse_mla_reference,
    )

    q_all = torch.ones((3, 8, 256), dtype=torch.bfloat16)
    kv_cache = torch.zeros((32, 1, 656), dtype=torch.uint8)
    page_table_1 = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
        ],
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor([12], dtype=torch.int32)
    nsa_cache_seqlens = torch.tensor([6, 4, 2], dtype=torch.int32)
    nsa_cu = torch.tensor([0, 3], dtype=torch.int32)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=nsa_cu,
        nsa_cu_seqlens_k=nsa_cu,
        max_seq_len_q=3,
        max_seq_len_k=12,
        mode="extend",
    )

    output = sparse_mla_extend_forward(
        kv_cache=kv_cache,
        binding=_extend_binding(workspace, q_all, metadata),
        sm_scale=1.0,
        v_head_dim=256,
    )

    assert output.shape == (3, 8, 256)
    assert torch.equal(captured["active_token_counts"], nsa_cache_seqlens)


def test_mla_decode_workspace_allocates_split_buffers_and_chunk_scalars() -> None:
    workspace = _make_workspace(
        mode="decode",
        topk=2048,
        max_total_q=8,
        max_batch=4,
    )

    assert workspace.tmp_output is not None
    assert workspace.tmp_output.shape == (8, 8, workspace.max_chunks_per_row, 256)
    assert workspace.tmp_lse is not None
    assert workspace.tmp_lse.shape == (8, 8, workspace.max_chunks_per_row)
    assert workspace.output_buffer is not None
    assert workspace.output_buffer.shape == (8, 8, 256)
    assert workspace.output_buffer.is_contiguous()
    assert workspace.output_buffer.data_ptr() == workspace.tmp_output[:, :, 0, :].data_ptr()
    assert workspace.final_lse is not None
    assert workspace.final_lse.shape == (8, 8)
    workspace.set_decode_chunk_config(kv_chunk_size=256, num_chunks=8)
    assert workspace.kv_chunk_size_ptr is not None
    assert workspace.num_chunks_ptr is not None
    assert int(workspace.kv_chunk_size_ptr[0].item()) == 256
    assert int(workspace.num_chunks_ptr[0].item()) == 8


def test_mla_workspace_enforces_capacity_limits() -> None:
    workspace = _make_workspace(mode="decode", topk=4)
    with torch.no_grad():
        too_wide = torch.zeros((2, 5), dtype=torch.int32)
        cache_seqlens = torch.zeros((2,), dtype=torch.int32)
        q_all = torch.zeros((2, 8, 256), dtype=torch.bfloat16)
        metadata = MLASparseDecodeMetadata(
            page_table_1=too_wide,
            cache_seqlens_int32=cache_seqlens,
            nsa_cache_seqlens_int32=cache_seqlens,
            max_seq_len_k=0,
        )
        try:
            sparse_mla_decode_forward(
                kv_cache=torch.zeros((16, 1, 656), dtype=torch.uint8),
                binding=_decode_binding(workspace, q_all, metadata),
                sm_scale=1.0,
                v_head_dim=256,
            )
        except ValueError as exc:
            assert "exceeds scratch topk" in str(exc)
        else:
            raise AssertionError("expected capacity validation to fail")
