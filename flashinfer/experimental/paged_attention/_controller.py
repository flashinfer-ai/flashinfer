"""Plan/run controller for the unified paged-prefill API.

Owns the public lifecycle behind ``flashinfer.prefill``:
validation, level-1 pinning against a ``Resolution``, level-2 choice within
it, derivation, and transactional publication of the planned backend. It does
not own any backend dialect (``_backends/``) or selection policy
(``_selection.py``).

Publication is transactional at this layer: ``plan()`` only swaps the
published metadata, derived forms, and active backend after the candidate
backend's own ``plan()`` returned, so a failed re-plan leaves the previous
plan runnable. The generated-FA backend re-plans its wrapper in place, so a
failure *inside* that wrapper's plan is the one path where backend-internal
state may already have moved — the same caveat the MLA design documents for
its generated backends.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from ._backends import CAPABILITIES, make_backend
from ._contracts import (
    PagedAttentionMetadata,
    PlanMetadata,
    Resolution,
    _expect,
    _expect_lse_mode,
    _expect_window_left,
    resolve_config_key,
)
from ._planning import Derived, validate_causal_envelope
from ._selection import resolve_paged_attention


class PagedAttentionController:
    def __init__(self, device: Optional[torch.device] = None):
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self._backends: Dict[Any, Any] = {}
        self._workspace: Optional[torch.Tensor] = None
        # published plan state (swapped together, only on success)
        self._planned = False
        self._backend_name: Optional[str] = None
        self._resolution: Optional[Resolution] = None
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
        self._active = None

    @property
    def backend(self) -> Optional[str]:
        """Name of the backend chosen by the last successful plan()."""
        return self._backend_name

    @property
    def resolution(self) -> Optional[Resolution]:
        return self._resolution

    def _shared_workspace(self) -> torch.Tensor:
        # One scratch workspace shared by the fa/cudnn backends (they never
        # run concurrently within one instance).  trtllm-gen keeps a private
        # zero-initialized buffer: its kernels rely on counter semantics that
        # a scribbled-on shared buffer would violate.
        if self._workspace is None:
            self._workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=self.device
            )
        return self._workspace

    # ------------------------------ plan ------------------------------

    def plan(
        self,
        metadata: PagedAttentionMetadata,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        q_dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype] = None,
        kv_layout: str = "HND",
        causal: bool = True,
        window_left: int = -1,
        lse_mode: str = "none",
        backend: Union[str, Resolution] = "auto",
    ) -> None:
        _expect(
            isinstance(metadata, PagedAttentionMetadata),
            "metadata must be a PagedAttentionMetadata (build it with "
            ".dense(...) or .csr(...))",
        )
        _expect(
            metadata.device == self.device,
            f"metadata lives on {metadata.device} but this instance is bound to "
            f"{self.device}",
        )
        head_dim_vo = head_dim_vo if head_dim_vo is not None else head_dim_qk
        kv_dtype = kv_dtype if kv_dtype is not None else q_dtype
        _expect(
            kv_layout in ("HND", "NHD"),
            f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}",
        )
        kv_input_form = metadata.kv_input_form
        _expect_window_left(window_left)
        _expect_lse_mode(lse_mode)
        need_lse = lse_mode != "none"
        if causal:
            validate_causal_envelope(metadata.qo_indptr_cpu, metadata.kv_seq_lens_cpu)

        if isinstance(backend, Resolution):
            # Level-1 pinning (proposal §5.3): verify the plan config matches
            # what the engine resolved at init, then choose within the set.
            want = resolve_config_key(
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo,
                q_dtype,
                metadata.page_size,
                kv_layout,
                causal,
                need_lse,
                window_left,
                kv_input_form,
            )
            _expect(
                backend.config == want,
                "plan() arguments do not match the pinned Resolution "
                f"(resolved {backend.config}, got {want}) — re-run "
                "resolve_paged_attention() with the new configuration",
            )
            resolution = backend
        else:
            resolution = resolve_paged_attention(
                device=self.device,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                q_dtype=q_dtype,
                page_size=metadata.page_size,
                kv_layout=kv_layout,
                causal=causal,
                need_lse=need_lse,
                window_left=window_left,
                kv_input_form=kv_input_form,
                backend=backend,
            )
        # Plan-time choice within the pinned set (level 2).  The prototype
        # takes the heuristic head; the autotune hook (proposal §5.4) would
        # consult its cache here, keyed on bucketed (total_q_tokens, max_kv_len).
        name = resolution.chosen

        derived = metadata.derived(needs_dense=CAPABILITIES[name].needs_dense)
        meta = PlanMetadata(
            qo_indptr=metadata.qo_indptr,
            kv_seq_lens=metadata.kv_seq_lens,
            # dense table given by the caller or derived (None where truly absent)
            block_tables=(
                metadata.block_tables
                if metadata.block_tables is not None
                else derived.block_tables
            ),
            kv_input_form=kv_input_form,
            page_size=metadata.page_size,
            max_q_len=metadata.max_q_len,
            max_kv_len=metadata.max_kv_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            causal=causal,
            window_left=window_left,
            kv_layout=kv_layout,
            lse_mode=lse_mode,
            batch_size=metadata.batch_size,
            qo_indptr_cpu=metadata.qo_indptr_cpu,
            kv_seq_lens_cpu=metadata.kv_seq_lens_cpu,
        )

        key = (name, kv_layout)
        candidate = self._backends.get(key)
        if candidate is None:
            candidate = make_backend(
                name, self.device, kv_layout, self._shared_workspace()
            )
        candidate.plan(meta, derived)

        # publish — nothing above mutated the published state
        self._backends[key] = candidate
        self._active = candidate
        self._backend_name = name
        self._resolution = resolution
        self._meta = meta
        self._derived = derived
        self._planned = True

    # ------------------------------ run -------------------------------

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Sequence[torch.Tensor],
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _expect(self._planned, "run() called before plan() — call plan() first")
        m = self._meta
        assert m is not None and self._active is not None
        _expect(
            isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2,
            "kv_cache must be a (k_cache, v_cache) pair of paged tensors "
            "(pages, num_kv_heads, page_size, head_dim) [HND]",
        )
        k_cache, v_cache = kv_cache
        layout = m.kv_layout
        h_pos, ps_pos = (1, 2) if layout == "HND" else (2, 1)
        shape_word = (
            "(pages, H, page_size, D)"
            if layout == "HND"
            else "(pages, page_size, H, D)"
        )
        for name, t, dim_ in (
            ("k_cache", k_cache, m.head_dim_qk),
            ("v_cache", v_cache, m.head_dim_vo),
        ):
            _expect(t.dim() == 4, f"{name} must be 4-D paged {shape_word} [{layout}]")
            swapped_hint = (
                f" — dims 1/2 look transposed: the plan declared kv_layout="
                f"{layout!r} {shape_word}; permute(0, 2, 1, 3).contiguous() or "
                "re-plan with the other kv_layout"
                if (
                    t.shape[ps_pos] == m.num_kv_heads
                    and t.shape[h_pos] == m.page_size
                    and m.num_kv_heads != m.page_size
                )
                else ""
            )
            _expect(
                t.shape[h_pos] == m.num_kv_heads
                and t.shape[ps_pos] == m.page_size
                and t.shape[3] == dim_,
                f"{name} shape {tuple(t.shape)} does not match plan "
                f"(kv_layout={layout}, H={m.num_kv_heads}, "
                f"page_size={m.page_size}, D={dim_})"
                f"{swapped_hint}",
            )
            _expect(
                t.dtype == m.kv_dtype,
                f"{name} dtype {t.dtype} != planned {m.kv_dtype}",
            )
        _expect(
            q.dim() == 3, "q must be packed (total_q_tokens, num_qo_heads, head_dim)"
        )
        _expect(
            q.shape[1] == m.num_qo_heads and q.shape[2] == m.head_dim_qk,
            f"q shape {tuple(q.shape)} does not match plan "
            f"(H={m.num_qo_heads}, D={m.head_dim_qk})",
        )
        _expect(q.dtype == m.q_dtype, f"q dtype {q.dtype} != planned {m.q_dtype}")
        total = m.total_q_tokens
        _expect(
            q.shape[0] == total,
            f"q has {q.shape[0]} tokens but qo_indptr sums to {total}",
        )
        cap = CAPABILITIES[self._backend_name]
        if cap.requires_contiguous_q and not q.is_contiguous():
            raise ValueError(
                f"backend {self._backend_name!r} requires contiguous packed q "
                "(token-unit addressing assumes packed THD); call "
                ".contiguous() or pin a strided-capable backend (fa2/fa3)"
            )
        if out is not None:
            _expect(
                tuple(out.shape) == (q.shape[0], m.num_qo_heads, m.head_dim_vo)
                and out.is_contiguous(),
                "out must be contiguous (total_q_tokens, num_qo_heads, head_dim_vo)",
            )
            _expect(
                out.dtype == m.q_dtype and out.device == q.device,
                f"out must match q dtype/device ({m.q_dtype}, {q.device}), "
                f"got ({out.dtype}, {out.device}) — allocate with "
                "torch.empty(..., dtype=q.dtype, device=q.device)",
            )
        if lse is not None:
            _expect(
                m.need_lse,
                "lse= buffer passed but the plan has lse_mode='none' — "
                "plan(lse_mode='base2' or 'basee') or drop the lse= argument",
            )
            _expect(
                tuple(lse.shape) == (q.shape[0], m.num_qo_heads)
                and lse.dtype == torch.float32
                and lse.is_contiguous()
                and lse.device == q.device,
                "lse must be contiguous fp32 (total_q_tokens, num_qo_heads) "
                f"on {q.device} — the LSE contract is packed fp32 in the planned "
                "base for every backend",
            )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(m.head_dim_qk)
        _expect(
            isinstance(sm_scale, float) and math.isfinite(sm_scale) and sm_scale > 0,
            f"sm_scale must be a positive finite host float, got {sm_scale!r}",
        )
        return self._active.run(
            q, k_cache, v_cache, out=out, lse=lse, sm_scale=sm_scale
        )

    def explain(self) -> str:
        _expect(self._planned, "explain() called before plan() — call plan() first")
        assert self._resolution is not None
        return f"chosen: {self._backend_name}\n{self._resolution.explain()}"


__all__ = ["PagedAttentionController"]
