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

# The TRT-LLM FMHAv2 prefill backend, self-contained: JIT module accessors,
# the FmhaV2PrefillBackend plan/run engine, and the user-facing
# FmhaV2BatchPrefillWith{Paged,Ragged}KVCacheWrapper classes. This backend is
# deliberately NOT a backend string on the generic wrappers in
# ``flashinfer/prefill.py`` — use the wrapper classes here instead. Only the
# legacy free-function kernel APIs (``trtllm_fmha_v2_prefill``,
# ``fmha_v2_prefill_deepseek``) remain in ``flashinfer/prefill.py``.

import enum
import functools
import math
from typing import Optional, Tuple, Union

import torch

from .jit import gen_fmha_v2_module
from .page import get_seq_lens
from .utils import (
    _check_kv_layout,
    _unpack_paged_kv_cache,
    canonicalize_torch_dtype,
    check_shape_dtype_device,
    get_compute_capability,
    log2e,
)


@functools.cache
def get_trtllm_fmha_v2_module(
    input_layout: str, input_dtype: torch.dtype, output_dtype: torch.dtype = None
):
    return gen_fmha_v2_module(input_layout, input_dtype, output_dtype).build_and_load()


@functools.cache
def get_trtllm_fmhav2_prefill_module(
    input_layout: str, q_dtype: torch.dtype, o_dtype: Optional[torch.dtype] = None
):
    # Layout/dtype-specialised FMHAv2 module exposing both prepare() and run().
    return get_trtllm_fmha_v2_module(input_layout, q_dtype, o_dtype)


# Map torch dtypes to the FmhaV2DType enum in csrc/fmha_v2_prepare.cu (matches Data_type
# in csrc/fmha_v2/fused_multihead_attention_utils.h).
_FMHAV2_DTYPE_CODE = {
    torch.float16: 0,  # FMHA_V2_DTYPE_FP16
    torch.float32: 1,  # FMHA_V2_DTYPE_FP32
    torch.int8: 3,  # FMHA_V2_DTYPE_INT8
    torch.bfloat16: 4,  # FMHA_V2_DTYPE_BF16
}
if hasattr(torch, "float8_e4m3fn"):
    _FMHAV2_DTYPE_CODE[torch.float8_e4m3fn] = 5  # FMHA_V2_DTYPE_E4M3


def _fmhav2_dtype_code(dtype: torch.dtype) -> int:
    code = _FMHAV2_DTYPE_CODE.get(dtype)
    if code is None:
        raise ValueError(f"Unsupported dtype for FMHAv2 prep kernel: {dtype}")
    return code


class FmhaV2Layout(str, enum.Enum):
    """FMHAv2 attention input layouts (Attention_input_layout in fmha_v2_run.cu).

    The layout is a codegen specialization — ``gen_fmha_v2_module`` compiles a
    separate module per layout — so it must be chosen at plan() time; it cannot
    be inferred from tensor shapes at run().
    """

    PAGED = "paged"  # q [T,H,D]; k/v paged pools + block_tables/page_size
    SEPARATE_Q_K_V = "separate_q_k_v"  # q/k/v all [T,H,D]
    PACKED_QKV = "packed_qkv"  # single qkv [T,3,H,D]
    CONTIGUOUS_Q_KV = "contiguous_q_kv"  # q [T,H,D]; kv [T,2,H_kv,D]


# Codegen module key (gen_fmha_v2_module input_layout) per non-paged layout;
# PAGED resolves to Q_PAGED_KV_{NHD,HND} from the wrapper's kv_layout instead.
_LAYOUT_MODULE_KEY = {
    FmhaV2Layout.SEPARATE_Q_K_V: "SEPARATE_Q_K_V",
    FmhaV2Layout.PACKED_QKV: "PACKED_QKV",
    FmhaV2Layout.CONTIGUOUS_Q_KV: "CONTIGUOUS_Q_KV",
}

# input_layout_str accepted by fmha_v2_run.cu's string_to_input_layout().
_LAYOUT_RUN_STR = {
    FmhaV2Layout.SEPARATE_Q_K_V: "separate_q_k_v",
    FmhaV2Layout.PACKED_QKV: "packed_qkv",
    FmhaV2Layout.CONTIGUOUS_Q_KV: "contiguous_q_kv",
}


class FmhaV2PrefillBackend:
    """State and plan/run engine behind the FMHAv2 prefill wrappers below.

    Owns the device buffers the fused prep kernel writes (cum-seq-lens, tile
    counter, encoded BMM scales) and the host-side resolved scales; the
    wrapper classes only derive inputs and delegate.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._cum_seq_lens_q: Optional[torch.Tensor] = None
        self._cum_seq_lens_kv: Optional[torch.Tensor] = None
        self._tile_id_counter: Optional[torch.Tensor] = None
        self._scale_bmm1_d: Optional[torch.Tensor] = None
        self._scale_bmm2_d: Optional[torch.Tensor] = None
        self._has_alibi: bool = False
        self._bmm1_scale: float = 1.0
        self._bmm2_scale: float = 1.0
        # Per-plan state consumed by run(); set by plan()/plan_paged().
        self._layout: Optional[FmhaV2Layout] = None
        self._run_layout_str: str = ""
        self._seq_lens_q: Optional[torch.Tensor] = None
        self._kv_lens: Optional[torch.Tensor] = None
        self._block_tables: Optional[torch.Tensor] = None
        self._page_size: int = 0
        self._max_q_len: int = 0
        self._max_kv_len: int = 0
        self._batch_size: int = 0

    def _check_gates(
        self, pos_encoding_mode: str, logits_soft_cap: Optional[float]
    ) -> Tuple[int, int]:
        """Feature gates shared by both plan() paths; returns the compute capability."""
        cc = get_compute_capability(self.device)
        if cc[0] not in (9, 12):
            raise NotImplementedError(
                f"trtllm-fmhav2 backend requires SM90 or SM120; got SM{cc[0]}{cc[1]}"
            )
        if pos_encoding_mode not in ("NONE", "ALIBI"):
            raise NotImplementedError(
                "trtllm-fmhav2 does not apply RoPE; pre-apply it to Q/K and pass NONE or ALIBI."
            )
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise NotImplementedError("trtllm-fmhav2 does not support logits_soft_cap.")
        return cc

    def _prep_args(
        self,
        batch_size: int,
        cc_major: int,
        pos_encoding_mode: str,
        logits_soft_cap: float,
        sm_scale: Optional[float],
        bmm1_scale: Optional[float],
        bmm2_scale: Optional[float],
        head_dim_qk: int,
        q_data_type: torch.dtype,
    ) -> tuple:
        """Shared tail of both plan() paths.

        (Re)allocates the prep-kernel device buffers, resolves and encodes the
        BMM scales, and stores the resolved scales and ALIBI flag for run().
        Returns the trailing arguments common to the module's ``prepare()``
        (ragged) and ``prepare_paged()`` (paged) entry points — both
        instantiations of the same fused prep kernel — so each plan() path
        issues exactly one kernel launch.
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
        self._bmm1_scale = float(
            bmm1_scale
            if bmm1_scale is not None
            else (sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim_qk))
        )
        self._bmm2_scale = float(bmm2_scale if bmm2_scale is not None else 1.0)

        # SM90 warp-spec path fuses log2e into scale_bmm1 (fmha_v2_run.cu:208)
        warp_spec = cc_major == 9 and not self._has_alibi and logits_soft_cap == 0.0
        if warp_spec:
            scale_bmm1_to_encode = self._bmm1_scale * float(log2e)
            scale_bmm1_dtype_code = _fmhav2_dtype_code(torch.float32)
        else:
            scale_bmm1_to_encode = self._bmm1_scale
            scale_bmm1_dtype_code = _fmhav2_dtype_code(
                torch.float16 if q_data_type == torch.float16 else torch.float32
            )
        # scale_bmm2 encoding matches scale_type2 in set_params (csrc/fmha_v2_run.cu):
        #   FP16 → FP16, BF16 → BF16, E4M3 → FP32 (acc), else FP32.
        if q_data_type in (torch.float16, torch.bfloat16):
            scale_bmm2_dtype_code = _fmhav2_dtype_code(q_data_type)
        else:
            scale_bmm2_dtype_code = _fmhav2_dtype_code(torch.float32)

        return (
            batch_size,
            scale_bmm1_dtype_code,
            scale_bmm2_dtype_code,
            float(scale_bmm1_to_encode),
            self._bmm2_scale,
            self._cum_seq_lens_q,
            self._cum_seq_lens_kv,
            self._tile_id_counter,
            self._scale_bmm1_d,
            self._scale_bmm2_d,
        )

    def plan_paged(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        kv_lens_out: torch.Tensor,
        block_tables_out: torch.Tensor,
        *,
        page_size: int,
        max_blocks_per_seq: int,
        batch_size: int,
        max_q_len: int,
        max_kv_len: int,
        kv_layout: str,
        q_data_type: torch.dtype,
        o_data_type: Optional[torch.dtype],
        pos_encoding_mode: str,
        logits_soft_cap: float,
        sm_scale: Optional[float],
        bmm1_scale: Optional[float],
        bmm2_scale: Optional[float],
        head_dim_qk: int,
    ):
        """Plan for :attr:`FmhaV2Layout.PAGED`; returns the loaded module.

        The paged layout has its own entry point because its prep genuinely
        differs: the single prepare_paged launch derives q/kv lens from the
        indptrs (the q lens feed the cum-scan in-register, no intermediate
        tensor), writes kv_lens into ``kv_lens_out`` for run(), scatters the
        dense ``block_tables_out``, zeroes the tile counter, and encodes the
        BMM scales.
        """
        cc = self._check_gates(pos_encoding_mode, logits_soft_cap)
        if kv_layout not in ("NHD", "HND"):
            raise ValueError("trtllm-fmhav2 requires kv_layout NHD or HND")

        module = get_trtllm_fmhav2_prefill_module(
            "Q_PAGED_KV_HND" if kv_layout == "HND" else "Q_PAGED_KV_NHD",
            q_data_type,
            o_data_type if q_data_type == torch.float8_e4m3fn else None,
        )
        self._layout = FmhaV2Layout.PAGED
        self._run_layout_str = (
            "q_paged_kv_hnd" if kv_layout == "HND" else "q_paged_kv_nhd"
        )
        self._seq_lens_q = None
        self._kv_lens = kv_lens_out
        self._block_tables = block_tables_out
        self._page_size = page_size
        self._max_q_len = max_q_len
        self._max_kv_len = max_kv_len
        self._batch_size = batch_size

        module.prepare_paged(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            paged_kv_indices,
            kv_lens_out,
            block_tables_out,
            page_size,
            max_blocks_per_seq,
            *self._prep_args(
                batch_size,
                cc[0],
                pos_encoding_mode,
                logits_soft_cap,
                sm_scale,
                bmm1_scale,
                bmm2_scale,
                head_dim_qk,
                q_data_type,
            ),
        )
        return module

    def plan(
        self,
        seq_lens_q: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
        layout: Union[FmhaV2Layout, str] = FmhaV2Layout.SEPARATE_Q_K_V,
        batch_size: int,
        max_q_len: int,
        max_kv_len: int,
        q_data_type: torch.dtype,
        o_data_type: Optional[torch.dtype],
        pos_encoding_mode: str,
        logits_soft_cap: float,
        sm_scale: Optional[float],
        bmm1_scale: Optional[float],
        bmm2_scale: Optional[float],
        head_dim_qk: int,
    ):
        """Plan for any non-paged layout; returns the loaded module.

        SEPARATE_Q_K_V, PACKED_QKV and CONTIGUOUS_Q_KV share this path
        entirely — their prep is the same cum-scan over caller-provided
        seq lens; only how the caller derived those lens (and the tensor
        forms later passed to run()) differ.
        """
        layout = FmhaV2Layout(layout)
        if layout == FmhaV2Layout.PAGED:
            raise ValueError("use plan_paged() for the paged layout")
        cc = self._check_gates(pos_encoding_mode, logits_soft_cap)

        module = get_trtllm_fmhav2_prefill_module(
            _LAYOUT_MODULE_KEY[layout],
            q_data_type,
            o_data_type if q_data_type == torch.float8_e4m3fn else None,
        )
        self._layout = layout
        self._run_layout_str = _LAYOUT_RUN_STR[layout]
        # Hold references so the tensors aren't freed before the async prep
        # kernel (and run()) consume them.
        self._seq_lens_q = seq_lens_q
        self._kv_lens = seq_lens_kv
        self._block_tables = None
        self._page_size = 0
        self._max_q_len = max_q_len
        self._max_kv_len = max_kv_len
        self._batch_size = batch_size

        module.prepare(
            seq_lens_q,
            seq_lens_kv,
            *self._prep_args(
                batch_size,
                cc[0],
                pos_encoding_mode,
                logits_soft_cap,
                sm_scale,
                bmm1_scale,
                bmm2_scale,
                head_dim_qk,
                q_data_type,
            ),
        )
        return module

    def _zero_tile_counter(self) -> None:
        # Re-zero tile_id_counter before every run: atomicAdd in dma.h leaves it
        # at num_tiles after each launch, so plan-once/run-many would read a
        # stale value on run #2+. zero_() is async on the current stream — no
        # D2H sync.
        assert self._tile_id_counter is not None, "plan() must run before run()"
        self._tile_id_counter.zero_()

    def run(
        self,
        module,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        *,
        out: torch.Tensor,
        workspace_buffer: torch.Tensor,
        workspace_size: Optional[int] = None,
        causal: bool,
        window_left: Optional[int],
        logits_soft_cap: Optional[float],
        lse: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor] = None,
    ) -> None:
        """Single FMHAv2 kernel launch; prep ran in plan()/plan_paged().

        The FFI has three tensor slots whose meaning follows the planned
        layout (fmha_v2_run.cu:381-410); unused slots expect dummy tensors,
        not None, and are filled in here so callers pass only what their
        layout actually has:

        - PAGED:           q [T,H,D], k/v = paged K/V pools
        - SEPARATE_Q_K_V:  q/k/v all [T,H,D]
        - PACKED_QKV:      q = packed qkv [T,3,H,D] (k/v omitted)
        - CONTIGUOUS_Q_KV: q [T,H,D], k = packed kv [T,2,H_kv,D] (v omitted)
        """
        assert self._layout is not None, "plan() must run before run()"
        if self._layout == FmhaV2Layout.PACKED_QKV:
            if q.dim() != 4 or q.shape[1] != 3:
                raise ValueError(
                    f"PACKED_QKV expects qkv of shape [total_tokens, 3, H, D]; got {tuple(q.shape)}"
                )
            k = v = q
        elif self._layout == FmhaV2Layout.CONTIGUOUS_Q_KV:
            if k is None or k.dim() != 4 or k.shape[1] != 2:
                raise ValueError(
                    "CONTIGUOUS_Q_KV expects kv of shape [total_tokens, 2, H_kv, D] in the k slot"
                )
            v = k
        elif k is None or v is None:
            raise ValueError(f"layout {self._layout.value} requires k and v tensors")
        if workspace_size is None:
            workspace_size = workspace_buffer.numel() * workspace_buffer.element_size()

        self._zero_tile_counter()
        module.run(
            q,
            k,
            v,
            out,
            workspace_buffer,
            workspace_size,
            self._block_tables,  # None for non-paged layouts
            self._page_size,  # 0 for non-paged layouts
            self._kv_lens,
            self._cum_seq_lens_q,
            self._cum_seq_lens_kv,
            self._run_layout_str,
            self._max_q_len,
            self._max_kv_len,
            self._batch_size,
            "causal" if causal else "padding",
            1.0,  # scale_softmax (encoded scales come from the prep kernel)
            self._bmm1_scale,
            self._bmm2_scale,
            window_left if window_left is not None else -1,
            0,  # chunked_attention_size
            self._has_alibi,
            float(logits_soft_cap or 0.0),
            0.0,  # skip_softmax_threshold_scale_factor
            lse,
            sinks,
            # Device-resident scratch from prep kernel; bypasses host set_alpha + memset.
            self._scale_bmm1_d,
            self._scale_bmm2_d,
            self._tile_id_counter,
        )


class FmhaV2BatchPrefillWithPagedKVCacheWrapper:
    r"""FMHAv2 batch prefill/append attention wrapper for paged KV cache.

    Standalone equivalent of
    :class:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` backed by
    the TRT-LLM FMHAv2 kernels (SM90 / SM120). plan() runs one fused prep
    kernel on device (paged indptr/indices -> dense block_tables + kv lens +
    cum-seq-lens scan + tile-counter reset + BMM scale encode); run() is a
    single FMHA kernel launch and may be called many times per plan().

    Restrictions: no in-kernel RoPE (pre-apply to Q/K; ``pos_encoding_mode``
    must be NONE or ALIBI) and no ``logits_soft_cap``.
    """

    def __init__(
        self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD"
    ) -> None:
        _check_kv_layout(kv_layout)
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._kv_layout = kv_layout
        self._backend = FmhaV2PrefillBackend(self.device)
        self._module = None
        self._kv_lens_buffer: Optional[torch.Tensor] = None
        self._block_tables: Optional[torch.Tensor] = None
        self._causal: bool = False
        self._window_left: int = -1
        self._logits_soft_cap: float = 0.0
        self._o_dtype: Optional[torch.dtype] = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        page_size: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: Optional[float] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
    ) -> None:
        q_data_type = canonicalize_torch_dtype(q_data_type)
        o_data_type = (
            canonicalize_torch_dtype(o_data_type)
            if o_data_type is not None
            else q_data_type
        )
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        batch_size = qo_indptr.shape[0] - 1

        if max_token_per_sequence is not None:
            max_q_len = int(max_token_per_sequence)
        else:
            qo_indptr_host = qo_indptr.to("cpu")
            max_q_len = int(max(qo_indptr_host[1:] - qo_indptr_host[:-1]))

        # Size block_tables host-side; ceil(kv_len / page_size) == pages per
        # sequence since last_page_len is in [1, page_size].
        if max_sequence_kv is not None:
            max_kv_len = int(max_sequence_kv)
            paged_kv_indptr_host = paged_kv_indptr.to("cpu")
            max_blocks_per_seq = int(
                max(
                    paged_kv_indptr_host[1 : batch_size + 1]
                    - paged_kv_indptr_host[:batch_size]
                )
            )
        else:
            if seq_lens is None:
                kv_lens_arr_host = get_seq_lens(
                    paged_kv_indptr.to("cpu"),
                    paged_kv_last_page_len.to("cpu"),
                    page_size,
                )
            else:
                kv_lens_arr_host = seq_lens.cpu().flatten()
            max_kv_len = int(max(kv_lens_arr_host).item())
            max_blocks_per_seq = max(
                (int(kv_len) + page_size - 1) // page_size
                for kv_len in kv_lens_arr_host
            )

        if self._kv_lens_buffer is None or batch_size > self._kv_lens_buffer.shape[0]:
            self._kv_lens_buffer = torch.empty(
                batch_size, dtype=torch.int32, device=self.device
            )
        self._block_tables = torch.zeros(
            (batch_size, max_blocks_per_seq), dtype=torch.int32, device=self.device
        )

        self._causal = causal
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._o_dtype = o_data_type
        self._module = self._backend.plan_paged(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_last_page_len,
            paged_kv_indices,
            self._kv_lens_buffer[:batch_size],
            self._block_tables,
            page_size=page_size,
            max_blocks_per_seq=max_blocks_per_seq,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            kv_layout=self._kv_layout,
            q_data_type=q_data_type,
            o_data_type=o_data_type,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            head_dim_qk=head_dim_qk,
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
        assert self._module is not None, "plan() must be called before run()"
        k_cache, v_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)
        if return_lse and lse is None:
            lse = torch.empty(
                (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
            )
        out_shape = q.shape[:-1] + (v_cache.shape[-1],)
        out_dtype = self._o_dtype or q.dtype
        if out is None:
            out = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        else:
            check_shape_dtype_device(out, out_shape, out_dtype, q.device, "out")
        self._backend.run(
            self._module,
            q,
            k_cache,
            v_cache,
            out=out,
            workspace_buffer=self._float_workspace_buffer,
            causal=self._causal,
            window_left=window_left if window_left is not None else self._window_left,
            logits_soft_cap=self._logits_soft_cap,
            lse=lse,
            sinks=sinks,
        )
        return (out, lse) if return_lse else out


class FmhaV2BatchPrefillWithRaggedKVCacheWrapper:
    r"""FMHAv2 batch prefill attention wrapper for ragged (SEPARATE_Q_K_V) KV.

    Standalone equivalent of
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper` backed by
    the TRT-LLM FMHAv2 kernels (SM90 / SM120). Same restrictions as the paged
    wrapper: no in-kernel RoPE and no ``logits_soft_cap``.
    """

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._backend = FmhaV2PrefillBackend(self.device)
        self._module = None
        self._causal: bool = False
        self._window_left: int = -1
        self._logits_soft_cap: float = 0.0
        self._o_dtype: Optional[torch.dtype] = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        sm_scale: Optional[float] = None,
        window_left: int = -1,
        logits_soft_cap: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Optional[Union[str, torch.dtype]] = None,
        bmm1_scale: Optional[float] = None,
        bmm2_scale: Optional[float] = None,
        seq_lens: Optional[torch.Tensor] = None,
        seq_lens_q: Optional[torch.Tensor] = None,
        max_token_per_sequence: Optional[int] = None,
        max_sequence_kv: Optional[int] = None,
    ) -> None:
        q_data_type = canonicalize_torch_dtype(q_data_type)
        o_data_type = (
            canonicalize_torch_dtype(o_data_type)
            if o_data_type is not None
            else q_data_type
        )
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        batch_size = qo_indptr.shape[0] - 1

        if max_token_per_sequence is not None:
            max_q_len = int(max_token_per_sequence)
        else:
            qo_indptr_host = qo_indptr.to("cpu")
            max_q_len = int(max(qo_indptr_host[1:] - qo_indptr_host[:-1]))
        if max_sequence_kv is not None:
            max_kv_len = int(max_sequence_kv)
        else:
            kv_indptr_host = kv_indptr.to("cpu")
            max_kv_len = int(max(kv_indptr_host[1:] - kv_indptr_host[:-1]))

        # Derive per-sequence lens on device unless the caller provided them.
        if seq_lens_q is not None:
            fmhav2_seq_lens_q = seq_lens_q.to(torch.int32)
        else:
            fmhav2_seq_lens_q = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
        if seq_lens is not None:
            fmhav2_seq_lens_kv = seq_lens.to(torch.int32)
        else:
            fmhav2_seq_lens_kv = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        self._causal = causal
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._o_dtype = o_data_type
        self._module = self._backend.plan(
            fmhav2_seq_lens_q,
            fmhav2_seq_lens_kv,
            layout=FmhaV2Layout.SEPARATE_Q_K_V,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            q_data_type=q_data_type,
            o_data_type=o_data_type,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            head_dim_qk=head_dim_qk,
        )

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        window_left: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert self._module is not None, "plan() must be called before run()"
        if return_lse and lse is None:
            lse = torch.empty(
                (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
            )
        out_shape = q.shape[:-1] + (v.shape[-1],)
        out_dtype = self._o_dtype or q.dtype
        if out is None:
            out = torch.empty(out_shape, dtype=out_dtype, device=q.device)
        else:
            check_shape_dtype_device(out, out_shape, out_dtype, q.device, "out")
        self._backend.run(
            self._module,
            q,
            k,
            v,
            out=out,
            workspace_buffer=self._float_workspace_buffer,
            causal=self._causal,
            window_left=window_left if window_left is not None else self._window_left,
            logits_soft_cap=self._logits_soft_cap,
            lse=lse,
        )
        return (out, lse) if return_lse else out
