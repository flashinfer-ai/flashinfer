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

"""Experimental MLA value projection fused with the sigmoid output gate (SM100 / SM103)."""

from typing import Any, Literal

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry point; the weight and launch checks, the runner, the
# registry and the generated programs live in
# flashinfer.experimental.cake_mla_value_gate.
_FEATURE = "MLA value projection + sigmoid output gate"


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_mla_value_gate(
    w_uv: torch.Tensor,
    *,
    max_rows: int,
    sigmoid_precision: Literal["fp32"],
    rows_per_request: int = 1,
    enable_pdl: bool = True,
    backend: str = "cake",
) -> Any:
    r"""Bind the absorbed MLA value weight for repeated fused value-projection + gate launches.

    For the decode token rows ``t < B`` of one rank with ``H`` heads, one launch of
    the returned runner computes, for ``h < H`` and ``v < 128``::

        S[t, h, v]        = sum_k latent[t, h, k] * w_uv[h, k, v]       (k < 512, FP32 accumulation)
        out[t, h*128 + v] = bf16(f32(bf16(S[t, h, v])) * sigmoid(f32(gate[t, h*128 + v])))

    for every valid row, and ``out[t, :] = +0.0`` for every row whose request has
    ``valid_lengths[t // rows_per_request] <= 0`` (whatever the row's latent or
    gate holds, NaN included).  The projection is rounded to BF16 once before the
    gate; the sigmoid is evaluated in FP32 and is not rounded to BF16.  Rows
    ``t >= B`` of the caller's buffers are neither read nor written.  The kernel
    uses no atomics and no split-K: results are bitwise reproducible and each row
    does not depend on ``B``.

    An architecture may have two programs that differ only in the token rows one
    CTA owns (16 or 8); they give identical bits, so the choice affects time
    only.  For every ``B`` in ``[1, max_rows]`` prepare decides once which one a
    launch of ``B`` rows runs: the 8-row program when ``enable_pdl`` is true or
    when its grid of ``4 * H * ceil(B / 8)`` CTAs fits one wave (at most the
    device's SM count; ``B <= 24`` at ``H = 12`` on B200, B300 and GB300), else
    the 16-row program.  ``runner.program_for(B)`` and ``runner.tile_rows_for(B)``
    report the choice.

    Parameters
    ----------
    w_uv : torch.Tensor
        BF16 ``[H, 512, 128]`` with strides ``(65536, 1, 512)``: the
        ``[H, 512, 128]`` view of a contiguous ``[H, 128, 512]`` tensor (the
        latent axis contiguous; e.g. ``w_vc.contiguous().transpose(1, 2)``), on
        an exact SM100a / SM103a device, 16-byte aligned.  Bound by the runner.
    max_rows : int
        Largest ``B`` the runner accepts, in ``[1, 64]``.
    sigmoid_precision : {"fp32"}
        Required.  ``"fp32"``: the sigmoid is evaluated in FP32 and multiplied
        unrounded with the BF16-rounded projection.  It is the only mode; any
        other value raises ``ValueError``.
    rows_per_request : int
        ``q``: row ``t`` belongs to request ``t // q`` (``B % q == 0``).
    enable_pdl : bool
        Launch with programmatic dependent launch (and run the 8-row program
        for every ``B``).  The kernel then reads
        ``w_uv`` before it waits on the preceding kernel of the stream, so that
        kernel must not write ``w_uv``; ``latent``, ``gate`` and
        ``valid_lengths`` are read and ``out`` is written only after the wait.
        A following kernel launched with programmatic dependent launch must wait
        (``griddepcontrol.wait``) before it reads ``out``.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    MlaValueGateRunner
        ``runner.launch(latent, gate, valid_lengths, out)`` takes:

        * ``latent``: contiguous BF16 ``[B, H, 512]``, 16-byte aligned, ``1 <= B <= max_rows``;
        * ``gate``: BF16 ``[B, H * 128]`` gate logits with unit column stride and an even
          row pitch ``gate.stride(0) >= H * 128`` (e.g. a column slice of a fused
          projection output), rows 4-byte aligned;
        * ``valid_lengths``: contiguous int32 ``[B // q]``; a zero or negative
          length marks a padded request;
        * ``out``: caller-owned contiguous BF16 ``[B, H * 128]``, 16-byte aligned, not
          overlapping any input;

        checks host metadata only, launches one kernel on the current CUDA stream
        without allocating or synchronizing, and returns ``out``.  Launches may be
        captured in a CUDA graph; replays re-read the captured tensors' contents
        (keep their storage alive and in place).  In-place updates of ``w_uv`` are
        seen by later launches; a new or re-aliased weight tensor needs a new
        ``prepare_mla_value_gate`` (and a new capture).

    Raises
    ------
    RuntimeError
        When called during CUDA graph capture (prepare outside capture).
    ValueError, TypeError
        On an invalid weight or argument (at prepare) or tensor (at launch).
    NotImplementedError
        When the device is not an exact SM100a / SM103a part or the checkout
        registers no program for it.

    Notes
    -----
    The computation follows SGLang's ``MlaValue`` (Kimi-K3 decode); see
    ``flashinfer/experimental/cake_mla_value_gate/README.md`` for the numerics,
    the qualified shapes and the attribution.
    """
    if backend == "cake":
        from .experimental.cake_mla_value_gate.cake_backend import (
            prepare_mla_value_gate as cake_prepare_mla_value_gate,
        )

        return cake_prepare_mla_value_gate(
            w_uv,
            max_rows=max_rows,
            sigmoid_precision=sigmoid_precision,
            rows_per_request=rows_per_request,
            enable_pdl=enable_pdl,
        )
    raise ValueError("prepare_mla_value_gate supports backend='cake'")


__all__ = ["prepare_mla_value_gate"]
