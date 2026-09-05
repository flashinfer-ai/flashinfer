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

VibeCUDA SSDCombined backend
============================

Hand-written CUDA implementation of the Mamba2 SSD combined forward pass
(mma.sync m16n8k16 bf16/fp16 with fp32 accumulation, no cuBLAS and no
CuTe-DSL).  At most two kernels per call:

* ``k_segstate`` — fused chunk-state accumulation and inter-chunk state
  passing for multi-chunk segments (skipped when the layout is host-known
  single-chunk);
* ``k_out`` — masked-decay intra-chunk matmuls, inter-chunk state
  contribution, D-skip, optional SiLU z gate, output store, and the fused
  final-state MMA for single-chunk segments.

Compiled through the regular FlashInfer nvcc JIT path; see
``flashinfer/jit/mamba/vibecuda_ssd.py``.
"""

import functools
from typing import Optional, Tuple

import torch

from ..jit.mamba.seq_chunk_cumsum import gen_seq_chunk_cumsum_module
from ..jit.mamba.vibecuda_ssd import gen_vibecuda_ssd_combined_module

_CHUNK = 128
_HEADDIM = 64
_DSTATE = 128


@functools.cache
def _get_vibecuda_module():
    """Get the cached VibeCUDA SSD combined JIT module."""
    return gen_vibecuda_ssd_combined_module().build_and_load()


@functools.cache
def _get_seq_chunk_cumsum_module():
    """Get the cached seq_chunk_cumsum JIT module."""
    return gen_seq_chunk_cumsum_module().build_and_load()


class VibeCUDASSDCombined:
    """Mamba2 SSD combined forward pass backed by the VibeCUDA kernels.

    Mirrors :class:`CakeSSDCombined`'s constructor and ``run`` surface so it
    can be selected with ``SSDCombined(..., backend="vibecuda")``.
    """

    def __init__(
        self,
        chunk_size: int,
        nheads: int,
        headdim: int,
        dstate: int,
        ngroups: int,
        io_dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.bfloat16,
        has_d: bool = True,
        d_has_hdim: bool = False,
        has_initial_states: bool = False,
        has_varlen: bool = False,
        has_z: bool = False,
        seq_idx_dtype=torch.int64,
    ):
        if chunk_size != _CHUNK or headdim != _HEADDIM or dstate != _DSTATE:
            raise ValueError(
                "VibeCUDA SSDCombined requires chunk_size=128, headdim=64, "
                f"dstate=128; got chunk_size={chunk_size}, headdim={headdim}, "
                f"dstate={dstate}"
            )
        if io_dtype != torch.bfloat16:
            raise ValueError(
                f"VibeCUDA SSDCombined requires io_dtype=bfloat16, got {io_dtype}"
            )
        if state_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "VibeCUDA SSDCombined requires state_dtype bfloat16 or float16, "
                f"got {state_dtype}"
            )
        self.chunk_size = chunk_size
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate
        self.ngroups = ngroups
        self._has_d = has_d
        self._d_has_hdim = d_has_hdim
        self._has_init_states = has_initial_states
        self._has_varlen = has_varlen
        self._has_z = has_z
        self._io_torch_dtype = io_dtype
        self._state_torch_dtype = state_dtype
        self._seq_idx_dtype = seq_idx_dtype

        self._module = _get_vibecuda_module()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _contiguous(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is not None and not t.is_contiguous():
            return t.contiguous()
        return t

    # -- main entry point ------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        initial_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
        seq_chunk_cumsum: Optional[torch.Tensor] = None,
        update_seq_chunk_cumsum: bool = False,
        checkpoint_token_indices: Optional[torch.Tensor] = None,
        checkpoint_state_slots: Optional[torch.Tensor] = None,
        checkpoint_states: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run SSD combined forward pass; see ``SSDCombined.run``."""
        batch, seqlen, nheads, headdim = x.shape
        nchunks = seqlen // _CHUNK

        has_varlen = seq_idx is not None
        if has_varlen and not self._has_varlen:
            raise ValueError(
                "seq_idx provided but VibeCUDASSDCombined was constructed with "
                "has_varlen=False"
            )
        if self._has_init_states and initial_states is None:
            raise ValueError(
                "initial_states must be provided when has_initial_states=True"
            )
        if initial_states is not None and not self._has_init_states:
            raise ValueError(
                "initial_states provided but VibeCUDASSDCombined was constructed "
                "with has_initial_states=False"
            )
        if has_varlen and initial_states is None:
            raise ValueError(
                "initial_states must be provided in varlen mode to determine num_seqs"
            )
        num_seqs = initial_states.shape[0] if initial_states is not None else batch

        state_dtype = self._state_torch_dtype
        final_states = torch.empty(
            num_seqs,
            nheads,
            _HEADDIM,
            _DSTATE,
            dtype=state_dtype,
            device=x.device,
        )

        # The host can only rule out multi-chunk segments without scanning
        # seq_idx; multi-chunk layouts need the state_in scratch.
        all_single_host = (not has_varlen) and seqlen <= _CHUNK
        if all_single_host:
            state_in = torch.empty(0, dtype=torch.bfloat16, device=x.device)
        else:
            n_lc_max = nchunks + num_seqs
            state_in = torch.empty(
                n_lc_max * nheads * _HEADDIM * _DSTATE,
                dtype=torch.bfloat16,
                device=x.device,
            )

        x_c = self._contiguous(x)
        dt_c = self._contiguous(dt)
        B_c = self._contiguous(B)
        C_c = self._contiguous(C)
        z_c = self._contiguous(z) if self._has_z else None
        d_c = self._contiguous(D) if self._has_d else None
        dt_bias_c = self._contiguous(dt_bias)
        initial_c = self._contiguous(initial_states) if self._has_init_states else None
        seq_idx_c = self._contiguous(seq_idx)

        # Match the CuTe backend's public-API D-shape coercion: a 1D D with
        # d_has_hdim broadcasts to (nheads, headdim); a 2D D with a scalar
        # parameter reduces to its first column.
        if d_c is not None:
            if self._d_has_hdim and d_c.dim() == 1:
                d_c = d_c.unsqueeze(1).expand(-1, _HEADDIM).contiguous()
            elif not self._d_has_hdim and d_c.dim() == 2:
                d_c = d_c[:, 0].contiguous()

        if out is None:
            out = torch.empty(
                batch,
                nheads,
                _HEADDIM,
                nchunks,
                _CHUNK,
                dtype=x.dtype,
                device=x.device,
            )

        self._module.vibecuda_ssd_combined_fwd(
            x_c,
            dt_c,
            dt_bias_c,
            A,
            B_c,
            C_c,
            d_c,
            z_c,
            initial_c,
            seq_idx_c if has_varlen else None,
            state_in,
            out,
            final_states,
            1 if dt_softplus else 0,
            float(dt_limit[0]),
            float(dt_limit[1]),
            1 if self._d_has_hdim else 0,
            1 if has_varlen else 0,
            1 if all_single_host else 0,
        )

        if (
            has_varlen
            and chunk_indices is not None
            and chunk_offsets is not None
            and (seq_chunk_cumsum is None or update_seq_chunk_cumsum)
        ):
            if seq_chunk_cumsum is None:
                seq_chunk_cumsum = torch.zeros(
                    num_seqs + 1, dtype=torch.int32, device=x.device
                )
            module = _get_seq_chunk_cumsum_module()
            tile_state_bytes = module.seq_chunk_cumsum_tile_state_size(num_seqs)
            tile_state = (
                torch.empty(tile_state_bytes, dtype=torch.uint8, device=x.device)
                if tile_state_bytes > 0
                else None
            )
            module.seq_chunk_cumsum(
                seq_idx_c,
                chunk_indices,
                chunk_offsets,
                seq_chunk_cumsum,
                tile_state,
                _CHUNK,
                len(chunk_indices),
                num_seqs,
            )

        out_view = out.permute(0, 3, 4, 1, 2).reshape(batch, seqlen, nheads, headdim)
        return out_view, final_states if return_final_states else None
