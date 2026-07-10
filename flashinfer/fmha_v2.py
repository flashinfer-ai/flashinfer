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

# Internals of the ``backend="trtllm-fmhav2"`` prefill path: the JIT module
# accessors and the per-wrapper backend state/logic used by
# ``BatchPrefillWith{Paged,Ragged}KVCacheWrapper`` in ``flashinfer/prefill.py``.
# The public kernel APIs (``trtllm_fmha_v2_prefill``,
# ``fmha_v2_prefill_deepseek``) remain in ``flashinfer/prefill.py``.

import functools
import math
from typing import Optional, Tuple

import torch

from .jit import gen_fmha_v2_module
from .utils import get_compute_capability, log2e


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


class FmhaV2PrefillBackend:
    """State and plan/run logic for one wrapper's trtllm-fmhav2 backend.

    Owns the device buffers the fused prep kernel writes (cum-seq-lens, tile
    counter, encoded BMM scales) and the host-side resolved scales, so the
    wrapper classes in ``prefill.py`` only delegate. Lazily created by the
    wrapper's plan() the first time the backend is used.
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
        # Ragged-only state (the paged wrapper keeps the equivalents itself,
        # shared with its other backends).
        self._seq_lens_q: Optional[torch.Tensor] = None
        self._seq_lens_kv: Optional[torch.Tensor] = None
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
        """Load the paged module and issue the fused prep launch; returns the module.

        The single prepare_paged launch derives q/kv lens from the indptrs (the
        q lens feed the cum-scan in-register, no intermediate tensor), writes
        kv_lens into ``kv_lens_out`` for run(), scatters the dense
        ``block_tables_out``, zeroes the tile counter, and encodes the BMM
        scales.
        """
        cc = self._check_gates(pos_encoding_mode, logits_soft_cap)
        if kv_layout not in ("NHD", "HND"):
            raise ValueError("trtllm-fmhav2 requires kv_layout NHD or HND")

        module = get_trtllm_fmhav2_prefill_module(
            "Q_PAGED_KV_HND" if kv_layout == "HND" else "Q_PAGED_KV_NHD",
            q_data_type,
            o_data_type if q_data_type == torch.float8_e4m3fn else None,
        )
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

    def plan_ragged(
        self,
        seq_lens_q: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        *,
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
        """Load the SEPARATE_Q_K_V module and issue the fused prep launch; returns the module."""
        cc = self._check_gates(pos_encoding_mode, logits_soft_cap)

        module = get_trtllm_fmhav2_prefill_module(
            "SEPARATE_Q_K_V",
            q_data_type,
            o_data_type if q_data_type == torch.float8_e4m3fn else None,
        )
        # Hold references so the tensors aren't freed before run() consumes them.
        self._seq_lens_q = seq_lens_q
        self._seq_lens_kv = seq_lens_kv
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

    def run_paged(
        self,
        module,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: torch.Tensor,
        *,
        workspace_buffer: torch.Tensor,
        workspace_size: int,
        block_tables: torch.Tensor,
        page_size: int,
        kv_lens: torch.Tensor,
        kv_layout: str,
        max_q_len: int,
        max_kv_len: int,
        batch_size: int,
        causal: bool,
        window_left: Optional[int],
        logits_soft_cap: Optional[float],
        lse: Optional[torch.Tensor],
        sinks: Optional[torch.Tensor],
    ) -> None:
        """Single FMHAv2 kernel launch consuming the device-resident prep outputs."""
        self._zero_tile_counter()
        module.run(
            q,
            k_cache,
            v_cache,
            out,
            workspace_buffer,
            workspace_size,
            block_tables,
            page_size,
            kv_lens,
            self._cum_seq_lens_q,
            self._cum_seq_lens_kv,
            "q_paged_kv_hnd" if kv_layout == "HND" else "q_paged_kv_nhd",
            max_q_len,
            max_kv_len,
            batch_size,
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

    def run_ragged(
        self,
        module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        *,
        workspace_buffer: torch.Tensor,
        causal: bool,
        window_left: Optional[int],
        logits_soft_cap: Optional[float],
        lse: Optional[torch.Tensor],
    ) -> None:
        """Single SEPARATE_Q_K_V FMHAv2 kernel launch; prep ran in plan()."""
        self._zero_tile_counter()
        module.run(
            q,
            k,
            v,
            out,
            workspace_buffer,
            workspace_buffer.numel() * workspace_buffer.element_size(),
            None,  # block_tables (ragged has none)
            0,  # page_size
            self._seq_lens_kv,
            self._cum_seq_lens_q,
            self._cum_seq_lens_kv,
            "separate_q_k_v",
            self._max_q_len,
            self._max_kv_len,
            self._batch_size,
            "causal" if causal else "padding",
            1.0,  # scale_softmax
            self._bmm1_scale,
            self._bmm2_scale,
            window_left if window_left is not None else -1,
            0,  # chunked_attention_size
            self._has_alibi,
            float(logits_soft_cap or 0.0),
            0.0,  # skip_softmax_threshold_scale_factor
            lse,
            None,  # sinks
            self._scale_bmm1_d,
            self._scale_bmm2_d,
            self._tile_id_counter,
        )
