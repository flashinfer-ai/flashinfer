"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# The TRT-LLM FMHAv2 prefill backend, self-contained: the JIT module accessor
# and the user-facing FmhaV2BatchPrefillWith{Paged,Ragged}KVCacheWrapper
# classes (sharing the _FmhaV2PrefillWrapperBase engine). This backend is
# deliberately NOT a backend string on the generic wrappers in
# ``flashinfer/prefill.py`` — use the wrapper classes here instead. Only the
# legacy free-function kernel APIs (``trtllm_fmha_v2_prefill``,
# ``fmha_v2_prefill_deepseek``) remain in ``flashinfer/prefill.py``.

import functools
from typing import Any, Optional, Tuple, Union

import torch

from .jit import gen_fmha_v2_module
from .utils import (
    _check_kv_layout,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    check_shape_dtype_device,
    get_compute_capability,
)


@functools.cache
def get_trtllm_fmha_v2_module(
    input_layout: str, input_dtype: torch.dtype, output_dtype: torch.dtype = None
):
    # Layout/dtype-specialised FMHAv2 module exposing prepare(), prepare_paged()
    # and run().
    return gen_fmha_v2_module(input_layout, input_dtype, output_dtype).build_and_load()


class _FmhaV2PrefillWrapperBase:
    """Shared engine and scaffolding for the FMHAv2 prefill wrappers.

    Owns the device buffers the fused prep kernel writes (cum-seq-lens, tile
    counter, encoded BMM scales) and all per-plan state consumed by
    :meth:`_run_impl`. A subclass plan() stores the per-plan state and issues
    the single fused prep launch; its run() is a single FMHA kernel launch via
    :meth:`_run_impl` and may be called many times per plan().

    Layouts follow :func:`flashinfer.prefill.trtllm_fmha_v2_prefill`: the same
    ``input_layout`` strings ("PACKED_QKV", "CONTIGUOUS_Q_KV",
    "Q_PAGED_KV_{NHD,HND}", "SEPARATE_Q_K_V") key the codegen module and,
    lowercased, the run FFI. The layout is a codegen specialization (one
    compiled module per layout), so it is fixed at plan() time. The non-paged
    layouts share prep entirely — the same cum-scan over per-sequence lens via
    the module's ``prepare()`` — so a future wrapper for the packed/contiguous
    layouts only needs to set ``self._input_layout``; the run-side slot
    normalization below already handles all four.
    """

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        # The layout/dtype-specialised JIT module; loaded by the subclass plan().
        self._module: Any = None
        self._o_dtype: Optional[torch.dtype] = None
        # Prep-kernel device buffers; (re)allocated lazily by _prep_args().
        self._cum_seq_lens_q: Optional[torch.Tensor] = None
        self._cum_seq_lens_kv: Optional[torch.Tensor] = None
        self._tile_id_counter: Optional[torch.Tensor] = None
        self._scale_bmm1_d: Optional[torch.Tensor] = None
        self._scale_bmm2_d: Optional[torch.Tensor] = None
        # Per-plan state consumed by _run_impl(); set by the subclass plan().
        self._input_layout: str = ""
        self._causal: bool = False
        self._window_left: int = -1
        self._has_alibi: bool = False
        self._logits_soft_cap: float = 0.0
        self._bmm1_scale: float = 1.0
        self._bmm2_scale: float = 1.0
        self._seq_lens_q: Optional[torch.Tensor] = None
        self._kv_lens: Optional[torch.Tensor] = None
        self._block_tables: Optional[torch.Tensor] = None
        self._page_size: int = 0
        self._max_q_len: int = 0
        self._max_kv_len: int = 0
        self._batch_size: int = 0

    def _check_gates(self, pos_encoding_mode: str) -> None:
        """Feature gates shared by all plan() paths."""
        cc = get_compute_capability(self.device)
        if cc[0] not in (9, 12):
            raise NotImplementedError(
                f"FMHAv2 prefill requires SM90 or SM120; got SM{cc[0]}{cc[1]}"
            )
        if pos_encoding_mode not in ("NONE", "ALIBI"):
            raise NotImplementedError(
                "FMHAv2 does not apply RoPE; pre-apply it to Q/K and pass NONE or ALIBI."
            )

    def _prep_args(
        self,
        batch_size: int,
        pos_encoding_mode: str,
        bmm1_scale: float,
        bmm2_scale: float,
        logits_soft_cap: Optional[float],
    ) -> tuple:
        """Shared tail of all plan() paths.

        (Re)allocates the prep-kernel device buffers and stores the scales and
        ALIBI flag for run(). Returns the trailing arguments common to the
        module's ``prepare()`` (non-paged) and ``prepare_paged()`` (paged)
        entry points — both instantiations of the same fused prep kernel — so
        each plan() path issues exactly one kernel launch.

        The scale-word encoding (dtype selection + warp-spec log2e fusion)
        happens in the C++ launcher (csrc/fmha_v2_prepare.cu), which mirrors
        fmha_v2_run.cu's host path; the kernel Data_type is baked into the
        module at codegen, so Python passes only the raw scales.
        """
        device = self.device
        if (
            self._cum_seq_lens_q is None
            or self._cum_seq_lens_q.numel() < batch_size + 1
        ):
            self._cum_seq_lens_q = torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._cum_seq_lens_kv = torch.empty(
                batch_size + 1, dtype=torch.int32, device=device
            )
        if self._tile_id_counter is None:
            self._tile_id_counter = torch.zeros(1, dtype=torch.uint32, device=device)
            self._scale_bmm1_d = torch.empty(1, dtype=torch.uint32, device=device)
            self._scale_bmm2_d = torch.empty(1, dtype=torch.uint32, device=device)

        self._has_alibi = pos_encoding_mode == "ALIBI"
        self._logits_soft_cap = float(logits_soft_cap or 0.0)
        self._bmm1_scale = float(bmm1_scale)
        self._bmm2_scale = float(bmm2_scale)

        return (
            batch_size,
            self._bmm1_scale,
            self._bmm2_scale,
            self._has_alibi,
            self._logits_soft_cap,
            self._cum_seq_lens_q,
            self._cum_seq_lens_kv,
            self._tile_id_counter,
            self._scale_bmm1_d,
            self._scale_bmm2_d,
        )

    def _resolve_dtypes(
        self,
        q_data_type: Union[str, torch.dtype],
        o_data_type: Optional[Union[str, torch.dtype]],
    ) -> torch.dtype:
        q_dtype = canonicalize_torch_dtype(q_data_type)
        self._o_dtype = (
            canonicalize_torch_dtype(o_data_type)
            if o_data_type is not None
            else q_dtype
        )
        return q_dtype

    def _alloc_out_lse(
        self,
        q: torch.Tensor,
        v_head_dim: int,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self._module is not None, "plan() must be called before run()"
        if return_lse and lse is None:
            # FMHAv2 softmax stats: (max, sum_exp) per token/head, ragged format.
            lse = torch.empty(
                (q.size(0), q.size(1), 2), dtype=torch.float32, device=q.device
            )
        out_shape = q.shape[:-1] + (v_head_dim,)
        out_dtype = self._o_dtype or q.dtype
        if out is None:
            out = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        else:
            check_shape_dtype_device(out, out_shape, out_dtype, q.device, "out")
        return out, lse

    def _zero_tile_counter(self) -> None:
        # Re-zero tile_id_counter before every run: atomicAdd in dma.h leaves it
        # at num_tiles after each launch, so plan-once/run-many would read a
        # stale value on run #2+. zero_() is async on the current stream — no
        # D2H sync.
        assert self._tile_id_counter is not None, "plan() must run before run()"
        self._tile_id_counter.zero_()

    def _run_impl(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        *,
        out: torch.Tensor,
        window_left: Optional[int] = None,
        lse: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        """Single FMHAv2 kernel launch; prep ran in the subclass plan().

        ``window_left`` overrides the value recorded at plan time when given;
        everything else per-plan (layout, mask mode, scales, lens) comes from
        the state stored by plan().

        The FFI has three tensor slots whose meaning follows the planned
        layout (fmha_v2_run.cu:381-410); unused slots expect dummy tensors,
        not None, and are filled in here so callers pass only what their
        layout actually has:

        - PAGED:           q [T,H,D], k/v = paged K/V pools
        - SEPARATE_Q_K_V:  q/k/v all [T,H,D]
        - PACKED_QKV:      q = packed qkv [T,3,H,D] (k/v omitted)
        - CONTIGUOUS_Q_KV: q [T,H,D], k = packed kv [T,2,H_kv,D] (v omitted)
        """
        assert self._module is not None and self._input_layout, (
            "plan() must be called before run()"
        )
        if self._input_layout == "PACKED_QKV":
            if q.dim() != 4 or q.shape[1] != 3:
                raise ValueError(
                    f"PACKED_QKV expects qkv of shape [total_tokens, 3, H, D]; got {tuple(q.shape)}"
                )
            k = v = q
        elif self._input_layout == "CONTIGUOUS_Q_KV":
            if k is None or k.dim() != 4 or k.shape[1] != 2:
                raise ValueError(
                    "CONTIGUOUS_Q_KV expects kv of shape [total_tokens, 2, H_kv, D] in the k slot"
                )
            v = k
        elif k is None or v is None:
            raise ValueError(f"layout {self._input_layout} requires k and v tensors")

        window_left = window_left if window_left is not None else self._window_left
        if self._causal:
            mask_mode = "sliding_window" if window_left >= 0 else "causal"
        else:
            mask_mode = "padding"

        self._zero_tile_counter()
        self._module.run(
            q,
            k,
            v,
            out,
            self._float_workspace_buffer,
            self._float_workspace_buffer.numel()
            * self._float_workspace_buffer.element_size(),
            self._block_tables,  # None for non-paged layouts
            self._page_size,  # 0 for non-paged layouts
            self._kv_lens,
            self._cum_seq_lens_q,
            self._cum_seq_lens_kv,
            self._input_layout.lower(),
            self._max_q_len,
            self._max_kv_len,
            self._batch_size,
            mask_mode,
            1.0,  # scale_softmax (encoded scales come from the prep kernel)
            self._bmm1_scale,
            self._bmm2_scale,
            window_left,
            0,  # chunked_attention_size
            self._has_alibi,
            self._logits_soft_cap,  # softcapping_scale (0.0 = disabled)
            0.0,  # skip_softmax_threshold_scale_factor
            lse,
            sinks,
            # Device-resident scratch from prep kernel; bypasses host set_alpha + memset.
            self._scale_bmm1_d,
            self._scale_bmm2_d,
            self._tile_id_counter,
        )


class FmhaV2BatchPrefillWithPagedKVCacheWrapper(_FmhaV2PrefillWrapperBase):
    r"""FMHAv2 batch prefill/append attention wrapper for paged KV cache.

    Standalone equivalent of
    :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` backed by
    the TRT-LLM FMHAv2 kernels (SM90 / SM120). plan() runs one fused prep
    kernel on device (paged indptr/indices -> dense block_tables + kv lens +
    cum-seq-lens scan + tile-counter reset + BMM scale encode); run() is a
    single FMHA kernel launch and may be called many times per plan().

    Following :func:`flashinfer.prefill.trtllm_fmha_v2_prefill`, the caller
    provides ``max_q_len`` / ``max_kv_len`` and the fused ``bmm1_scale`` /
    ``bmm2_scale`` (typically ``sm_scale`` and ``1.0``) — nothing is derived
    on the host from the indptrs.

    Restrictions: no in-kernel RoPE (pre-apply to Q/K; ``pos_encoding_mode``
    must be NONE or ALIBI).
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        _check_kv_layout(kv_layout)
        super().__init__(float_workspace_buffer)
        self._kv_layout = kv_layout
        self._kv_lens_buffer: Optional[torch.Tensor] = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        *,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        o_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        self._check_gates(pos_encoding_mode)
        q_data_type = self._resolve_dtypes(q_data_type, o_data_type)
        batch_size = qo_indptr.shape[0] - 1
        # ceil is monotonic, so the widest block-table row is ceil(max_kv_len / page_size).
        max_blocks_per_seq = (max_kv_len + page_size - 1) // page_size

        if self._kv_lens_buffer is None or batch_size > self._kv_lens_buffer.shape[0]:
            self._kv_lens_buffer = torch.empty(
                batch_size, dtype=torch.int32, device=self.device
            )
        block_tables = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32, device=self.device
        )

        input_layout = (
            "Q_PAGED_KV_HND" if self._kv_layout == "HND" else "Q_PAGED_KV_NHD"
        )
        self._module = get_trtllm_fmha_v2_module(
            input_layout,
            q_data_type,
            self._o_dtype if q_data_type == torch.float8_e4m3fn else None,
        )
        self._input_layout = input_layout
        self._causal = causal
        self._window_left = window_left
        self._seq_lens_q = None
        self._kv_lens = self._kv_lens_buffer[:batch_size]
        self._block_tables = block_tables
        self._page_size = page_size
        self._max_q_len = max_q_len
        self._max_kv_len = max_kv_len
        self._batch_size = batch_size

        # One fused prep launch. The paged layout has its own prep entry point
        # because its prep genuinely differs: prepare_paged derives q/kv lens
        # from the indptrs (the q lens feed the cum-scan in-register, no
        # intermediate tensor), writes kv_lens into _kv_lens_buffer for run(),
        # scatters the dense block_tables, zeroes the tile counter, and
        # encodes the BMM scales.
        self._module.prepare_paged(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            paged_kv_indices,
            self._kv_lens,
            block_tables,
            page_size,
            max_blocks_per_seq,
            *self._prep_args(
                batch_size, pos_encoding_mode, bmm1_scale, bmm2_scale, logits_soft_cap
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        window_left: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        out, lse = self._alloc_out_lse(q, v_cache.shape[-1], out, lse, return_lse)
        self._run_impl(
            q, k_cache, v_cache, out=out, window_left=window_left, lse=lse, sinks=sinks
        )
        return (out, lse) if return_lse else out


class FmhaV2BatchPrefillWithRaggedKVCacheWrapper(_FmhaV2PrefillWrapperBase):
    r"""FMHAv2 batch prefill attention wrapper for ragged (non-paged) KV.

    Standalone equivalent of
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` backed by
    the TRT-LLM FMHAv2 kernels (SM90 / SM120). Following
    :func:`flashinfer.prefill.trtllm_fmha_v2_prefill`, the caller provides the
    per-sequence lens (int32 device tensors of shape ``[batch_size]``),
    ``max_q_len`` / ``max_kv_len``, and the fused BMM scales; the fused prep
    kernel computes the cum-seq-lens on device. Same restrictions as the
    paged wrapper (no in-kernel RoPE); additionally ``logits_soft_cap`` is not
    supported for the SEPARATE_Q_K_V layout.

    All three non-paged layouts share the same prep; ``input_layout`` selects
    the codegen module and the run()-time tensor slots (see :meth:`run`):
    ``SEPARATE_Q_K_V`` (default), ``PACKED_QKV``, ``CONTIGUOUS_Q_KV`` — the
    same strings :func:`flashinfer.prefill.trtllm_fmha_v2_prefill` takes.
    """

    def plan(
        self,
        seq_lens_q: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        max_q_len: int,
        max_kv_len: int,
        bmm1_scale: float,
        bmm2_scale: float,
        *,
        input_layout: str = "SEPARATE_Q_K_V",
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        o_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        if input_layout not in ("SEPARATE_Q_K_V", "PACKED_QKV", "CONTIGUOUS_Q_KV"):
            raise ValueError(
                f"input_layout {input_layout!r} is not a ragged layout; use "
                "FmhaV2BatchPrefillWithPagedKVCacheWrapper for paged KV."
            )
        self._check_gates(pos_encoding_mode)
        if input_layout == "SEPARATE_Q_K_V" and logits_soft_cap:
            raise NotImplementedError(
                "logits_soft_cap is not supported for the SEPARATE_Q_K_V layout."
            )
        q_data_type = self._resolve_dtypes(q_data_type, o_data_type)
        batch_size = seq_lens_q.shape[0]

        self._module = get_trtllm_fmha_v2_module(
            input_layout,
            q_data_type,
            self._o_dtype if q_data_type == torch.float8_e4m3fn else None,
        )
        self._input_layout = input_layout
        self._causal = causal
        self._window_left = window_left
        # Hold references so the tensors aren't freed before the async prep
        # kernel (and run()) consume them.
        self._seq_lens_q = seq_lens_q
        self._kv_lens = seq_lens_kv
        self._block_tables = None
        self._page_size = 0
        self._max_q_len = max_q_len
        self._max_kv_len = max_kv_len
        self._batch_size = batch_size

        self._module.prepare(
            seq_lens_q,
            seq_lens_kv,
            *self._prep_args(
                batch_size, pos_encoding_mode, bmm1_scale, bmm2_scale, logits_soft_cap
            ),
        )

    def run(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        window_left: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Tensor slots follow the planned layout (see :meth:`_run_impl`):
        SEPARATE_Q_K_V → ``run(q, k, v)``; PACKED_QKV → ``run(qkv)``;
        CONTIGUOUS_Q_KV → ``run(q, kv)``.
        """
        if self._input_layout == "PACKED_QKV":
            # q is the packed qkv [T, 3, H, D]; out/lse are sized off its Q slice.
            out, lse = self._alloc_out_lse(q[:, 0], q.shape[-1], out, lse, return_lse)
        elif self._input_layout == "CONTIGUOUS_Q_KV":
            out, lse = self._alloc_out_lse(q, k.shape[-1], out, lse, return_lse)
        else:
            out, lse = self._alloc_out_lse(q, v.shape[-1], out, lse, return_lse)
        self._run_impl(q, k, v, out=out, window_left=window_left, lse=lse)
        return (out, lse) if return_lse else out
