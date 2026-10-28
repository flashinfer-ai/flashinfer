from __future__ import annotations

import importlib
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest
import torch

from b12x.attention._shared.mla.reference import (
    dense_mla_reference,
    pack_mla_kv_cache_reference,
)

from b12x.testing.reference.helpers import require_b12x
from .test_attention_mla_reference import _compare, _make_glm_case, _require_glm_weights


_SGLANG_PYTHON_ROOT = Path("/home/luke/projects/sglang/python")


def _import_sglang_nsa_backend():
    if not _SGLANG_PYTHON_ROOT.exists():
        pytest.skip(f"sglang sources not found at {_SGLANG_PYTHON_ROOT}")
    root = str(_SGLANG_PYTHON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("sglang.srt.layers.attention.nsa_backend")
    except Exception as exc:  # pragma: no cover - environment-dependent import path
        pytest.skip(f"unable to import sglang NSA backend: {exc}")
    return module


def _get_b12x_forward_method(nsa_backend_module):
    backend_cls = nsa_backend_module.NativeSparseAttnBackend
    method = getattr(backend_cls, "_forward_b12x", None)
    if method is None:
        method = getattr(backend_cls, "_forward_b12x_mla")
    return method


def _get_b12x_decode_kv_reshape_method(nsa_backend_module):
    return nsa_backend_module.NativeSparseAttnBackend._reshape_b12x_decode_kv_rows


def _full_prefix_page_table(
    *, cache_len: int, rows: int, width: int, device: torch.device
) -> torch.Tensor:
    page_table_1 = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    valid = min(cache_len, width)
    if valid > 0:
        page_table_1[:, :valid] = torch.arange(valid, dtype=torch.int32, device=device)
    return page_table_1


def _sample_sparse_page_table(
    *,
    cache_len: int,
    rows: int,
    width: int,
    valid_per_row: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    rng = random.Random(seed)
    page_table_1 = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    population = list(range(cache_len))
    for row_idx in range(rows):
        selected = sorted(rng.sample(population, valid_per_row))
        page_table_1[row_idx, :valid_per_row] = torch.tensor(
            selected, dtype=torch.int32, device=device
        )
    return page_table_1


def _make_fake_backend(
    cfg,
    *,
    device: torch.device,
    topk: int,
    nsa_backend_module,
    num_q_heads: int | None = None,
):
    backend_cls = nsa_backend_module.NativeSparseAttnBackend

    class _FakeBackend:
        _gather_b12x_extend_kv_rows = backend_cls._gather_b12x_extend_kv_rows
        _get_b12x_extend_row_ids = backend_cls._get_b12x_extend_row_ids
        _reshape_b12x_decode_kv_rows = _get_b12x_decode_kv_reshape_method(
            nsa_backend_module
        )

        def __init__(self):
            self.device = device
            self.q_dtype = torch.bfloat16
            self.kv_cache_dtype = torch.uint8
            self.num_q_heads = (
                cfg.num_heads if num_q_heads is None else int(num_q_heads)
            )
            self.kv_lora_rank = cfg.kv_lora_rank
            self.qk_rope_head_dim = cfg.qk_rope_head_dim
            self.nsa_index_topk = topk
            self.real_page_size = 64
            self.b12x_workspaces: dict[tuple[str, int, int], object] = {}
            self.b12x_mla_workspaces = self.b12x_workspaces
            self.max_running_requests = 32
            self.server_args = type(
                "_FakeServerArgs",
                (),
                {
                    "chunked_prefill_size": -1,
                    "max_prefill_tokens": 4096,
                    "prefill_max_requests": None,
                },
            )()

        def _b12x_extend_total_q_capacity(
            self, forward_batch=None, required_q: int | None = None
        ):
            del forward_batch
            if required_q is not None:
                return max(int(required_q), 1)
            return int(self.server_args.max_prefill_tokens)

        def _b12x_extend_batch_capacity(
            self, forward_batch=None, required_batch: int | None = None
        ):
            del forward_batch
            if required_batch is not None:
                return max(int(required_batch), 1)
            return int(self.max_running_requests)

        def _b12x_extend_kv_rows_capacity(
            self, forward_batch=None, required_rows: int | None = None
        ):
            base = int(required_rows or 0)
            if forward_batch is not None and hasattr(
                forward_batch, "get_max_chunk_capacity"
            ):
                base = max(base, int(forward_batch.get_max_chunk_capacity()))
            return max(base, 1)

        def _get_b12x_workspace(
            self,
            *,
            mode: str,
            v_head_dim: int,
            total_q: int | None = None,
            batch: int | None = None,
            max_kv_rows: int | None = None,
        ):
            from b12x.attention._shared.workspace import B12XAttentionWorkspace

            normalized_mode = "verify" if mode == "target_verify" else mode
            if normalized_mode not in ("decode", "extend", "verify", "draft_extend"):
                raise AssertionError(f"unexpected b12x workspace mode: {mode}")
            total_q_cap = (
                max(int(total_q), 1)
                if total_q is not None
                else (
                    int(self.max_running_requests)
                    if normalized_mode == "decode"
                    else int(self.server_args.max_prefill_tokens)
                )
            )
            batch_cap = (
                max(int(batch), 1)
                if batch is not None
                else int(self.max_running_requests)
            )
            kv_rows_cap = (
                0
                if normalized_mode == "decode"
                else max(int(max_kv_rows or total_q_cap), 1)
            )
            key = (normalized_mode, int(v_head_dim), kv_rows_cap)
            workspace = self.b12x_workspaces.get(key)
            if workspace is not None:
                return workspace
            workspace = B12XAttentionWorkspace.for_fixed_capacity(
                mode=normalized_mode,
                device=self.device,
                dtype=self.q_dtype,
                kv_dtype=self.kv_cache_dtype,
                num_q_heads=self.num_q_heads,
                indexer_num_q_heads=self.num_q_heads,
                head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
                v_head_dim=int(v_head_dim),
                topk=self.nsa_index_topk,
                max_page_table_width=self.nsa_index_topk,
                max_total_q=total_q_cap,
                max_batch=batch_cap,
                max_paged_q_rows=batch_cap
                if normalized_mode == "decode"
                else total_q_cap,
                max_kv_rows=kv_rows_cap,
                page_size=self.real_page_size,
                use_cuda_graph=True,
            )
            self.b12x_workspaces[key] = workspace
            return workspace

    return _FakeBackend()


def test_sglang_b12x_ragged_extend_kv_gather_uses_chunk_capacity() -> None:
    nsa_backend_module = _import_sglang_nsa_backend()
    cfg = SimpleNamespace(num_heads=8, kv_lora_rank=512, qk_rope_head_dim=64)
    backend = _make_fake_backend(
        cfg,
        device=torch.device("cpu"),
        topk=8,
        nsa_backend_module=nsa_backend_module,
    )
    kv_cache = torch.arange(32 * 656, dtype=torch.uint8).reshape(32, 1, 656)
    forward_batch = SimpleNamespace(
        extend_prefix_lens_cpu=[0],
        out_cache_loc=torch.tensor([2, 6, 9], dtype=torch.int32),
        get_max_chunk_capacity=lambda: 12,
    )
    workspace = backend._get_b12x_workspace(
        mode="extend",
        total_q=1,
        batch=1,
        v_head_dim=cfg.kv_lora_rank,
        max_kv_rows=backend._b12x_extend_kv_rows_capacity(
            forward_batch, required_rows=3
        ),
    )

    gathered = backend._gather_b12x_extend_kv_rows(
        kv_cache=kv_cache,
        forward_batch=forward_batch,
        metadata=SimpleNamespace(seq_lens_sum=3, page_table_1_flattened=None),
        workspace=workspace,
    )

    assert workspace.max_kv_rows == 12
    assert gathered.shape == (12, 1, 656)
    assert torch.equal(
        gathered[:3], kv_cache[torch.tensor([2, 6, 9], dtype=torch.long)]
    )

    first_ptr = gathered.data_ptr()
    forward_batch.out_cache_loc = torch.tensor([1, 4], dtype=torch.int32)
    gathered_again = backend._gather_b12x_extend_kv_rows(
        kv_cache=kv_cache,
        forward_batch=forward_batch,
        metadata=SimpleNamespace(seq_lens_sum=2, page_table_1_flattened=None),
        workspace=workspace,
    )

    assert gathered_again.data_ptr() == first_ptr
    assert torch.equal(
        gathered_again[:2], kv_cache[torch.tensor([1, 4], dtype=torch.long)]
    )


def _forward_b12x_mla(
    nsa_backend_module,
    backend,
    *,
    q_all: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table_1: torch.Tensor,
    metadata: object,
    sm_scale: float,
    v_head_dim: int,
    mode: str,
) -> torch.Tensor:
    forward_b12x = _get_b12x_forward_method(nsa_backend_module)
    workspace = backend._get_b12x_workspace(mode=mode, v_head_dim=v_head_dim)
    return forward_b12x(
        backend,
        q_all=q_all,
        kv_cache=kv_cache,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=sm_scale,
        v_head_dim=v_head_dim,
        mode=mode,
        workspace=workspace,
    )


def _make_decode_metadata(
    *, nsa_backend_module, cache_len: int, page_table_1: torch.Tensor
) -> object:
    cache_seqlens = torch.tensor(
        [cache_len], dtype=torch.int32, device=page_table_1.device
    )
    nsa_cache_seqlens = torch.tensor(
        [int((page_table_1[0] >= 0).sum().item())],
        dtype=torch.int32,
        device=page_table_1.device,
    )
    return nsa_backend_module.NSAMetadata(
        page_size=64,
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=cache_len,
        cu_seqlens_q=torch.tensor(
            [0, 1], dtype=torch.int32, device=page_table_1.device
        ),
        cu_seqlens_k=torch.tensor(
            [0, cache_len], dtype=torch.int32, device=page_table_1.device
        ),
        page_table_1=page_table_1,
        real_page_table=page_table_1,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        nsa_cu_seqlens_q=torch.tensor(
            [0, 1], dtype=torch.int32, device=page_table_1.device
        ),
        nsa_cu_seqlens_k=torch.tensor(
            [0, int(nsa_cache_seqlens[0].item())],
            dtype=torch.int32,
            device=page_table_1.device,
        ),
        nsa_extend_seq_lens_list=[1],
        nsa_seqlens_expanded=cache_seqlens,
    )


def _make_extend_metadata(
    *,
    nsa_backend_module,
    cache_len: int,
    page_table_1: torch.Tensor,
) -> object:
    rows = page_table_1.shape[0]
    cache_seqlens = torch.full(
        (rows,), cache_len, dtype=torch.int32, device=page_table_1.device
    )
    valid_per_row = (page_table_1 >= 0).sum(dim=1, dtype=torch.int32)
    nsa_cu = torch.zeros(rows + 1, dtype=torch.int32, device=page_table_1.device)
    nsa_cu[1:] = torch.cumsum(valid_per_row, dim=0)
    return nsa_backend_module.NSAMetadata(
        page_size=64,
        cache_seqlens_int32=cache_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=cache_len,
        cu_seqlens_q=torch.arange(
            0, rows + 1, dtype=torch.int32, device=page_table_1.device
        ),
        cu_seqlens_k=torch.arange(
            0,
            (rows + 1) * cache_len,
            cache_len,
            dtype=torch.int32,
            device=page_table_1.device,
        ),
        page_table_1=page_table_1,
        real_page_table=page_table_1,
        nsa_cache_seqlens_int32=valid_per_row,
        nsa_cu_seqlens_q=torch.arange(
            0, rows + 1, dtype=torch.int32, device=page_table_1.device
        ),
        nsa_cu_seqlens_k=nsa_cu,
        nsa_extend_seq_lens_list=[1] * rows,
        nsa_seqlens_expanded=cache_seqlens,
    )


def test_sglang_b12x_mla_decode_boundary_matches_dense_oracle() -> None:
    device = require_b12x()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 129
    topk = 2048
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=71_129,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _full_prefix_page_table(
        cache_len=cache_len, rows=1, width=topk, device=device
    )
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(
        cfg, device=device, topk=topk, nsa_backend_module=nsa_backend_module
    )

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="decode",
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_sglang_b12x_mla_decode_boundary_matches_dense_oracle_for_local_tp_heads() -> (
    None
):
    device = require_b12x()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 2050
    topk = 2048
    local_heads = 8
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=71_205,
        device=device,
    )
    q_local = q_all[:, :local_heads, :].contiguous()
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _full_prefix_page_table(
        cache_len=cache_len, rows=1, width=topk, device=device
    )
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(
        cfg,
        device=device,
        topk=topk,
        nsa_backend_module=nsa_backend_module,
        num_q_heads=local_heads,
    )

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_local,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="decode",
    )
    expected = dense_mla_reference(
        q_all=q_local,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_sglang_b12x_mla_decode_boundary_matches_dense_oracle_for_local_tp_heads_fp8_view_cache() -> (
    None
):
    device = require_b12x()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 2050
    topk = 2048
    local_heads = 8
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=71_206,
        device=device,
    )
    q_local = q_all[:, :local_heads, :].contiguous()
    packed = pack_mla_kv_cache_reference(k_nope, k_rope).view(torch.float8_e4m3fn)
    page_table_1 = _full_prefix_page_table(
        cache_len=cache_len, rows=1, width=topk, device=device
    )
    metadata = _make_decode_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(
        cfg,
        device=device,
        topk=topk,
        nsa_backend_module=nsa_backend_module,
        num_q_heads=local_heads,
    )
    backend.kv_cache_dtype = torch.float8_e4m3fn

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_local,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="decode",
    )
    expected = dense_mla_reference(
        q_all=q_local,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_sglang_b12x_mla_extend_boundary_matches_dense_oracle() -> None:
    device = require_b12x()
    _require_glm_weights()
    nsa_backend_module = _import_sglang_nsa_backend()

    cache_len = 2050
    q_len = 4
    topk = 2048
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=72_050,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = _sample_sparse_page_table(
        cache_len=cache_len,
        rows=q_len,
        width=topk,
        valid_per_row=topk,
        seed=72050,
        device=device,
    )
    metadata = _make_extend_metadata(
        nsa_backend_module=nsa_backend_module,
        cache_len=cache_len,
        page_table_1=page_table_1,
    )
    backend = _make_fake_backend(
        cfg, device=device, topk=topk, nsa_backend_module=nsa_backend_module
    )

    actual = _forward_b12x_mla(
        nsa_backend_module,
        backend,
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        metadata=metadata,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
        mode="extend",
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"
