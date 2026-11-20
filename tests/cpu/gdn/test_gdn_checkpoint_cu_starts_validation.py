# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Host-only tests of the opt-in ``checkpoint_cu_starts`` check of
``chunk_gated_delta_rule`` (``FLASHINFER_VALIDATE_INPUTS=1``).

``checkpoint_cu_starts`` is in checkpoint units (rows of
``state_checkpoints``); token offsets make the kernels write outside the
buffer without an error (issue #9559)."""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from flashinfer import gdn_prefill

pytestmark = pytest.mark.cpu_only

HV = 4
EVERY = 64


@pytest.fixture
def validate(monkeypatch):
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")


def _i64(values):
    return torch.tensor(values, dtype=torch.int64)


def _check(
    cu,
    starts,
    rows,
    *,
    every=EVERY,
    indices=None,
    prefix=None,
):
    gdn_prefill._validate_checkpoint_cu_starts(
        cu_seqlens=_i64(cu),
        checkpoint_cu_starts=_i64(starts),
        checkpoint_every_n_tokens=every,
        state_checkpoints=torch.zeros(rows, HV, 1, 1),
        checkpoint_state_indices=None if indices is None else _i64(indices),
        prefix_lens=None if prefix is None else _i64(prefix),
    )


def test_token_offsets_are_rejected_through_the_public_api(validate):
    # The case of record in #9559: 2 x 256 tokens, a checkpoint every 64
    # tokens, so 8 rows; token offsets [0, 256, 512] would write rows 256..259.
    t = 512
    q = torch.zeros(t, 2, 128, dtype=torch.bfloat16)
    v = torch.zeros(t, HV, 128, dtype=torch.bfloat16)
    cu = _i64([0, 256, t])
    with pytest.raises(ValueError, match="looks like token offsets"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            torch.zeros_like(q),
            v,
            cu_seqlens=cu,
            checkpoint_every_n_tokens=EVERY,
            state_checkpoints=torch.zeros(8, HV, 128, 128),
            checkpoint_cu_starts=cu.clone(),
        )


def test_message_names_the_first_bad_sequence(validate):
    with pytest.raises(
        ValueError,
        match=r"sequence 0 has 256 tokens, so 4 checkpoint\(s\) with "
        r"checkpoint_every_n_tokens=64, but checkpoint_cu_starts\[1\] - "
        r"checkpoint_cu_starts\[0\] = 256 - 0 = 256\. It looks like token offsets",
    ):
        _check([0, 256, 384], [0, 256, 384], rows=6)
    with pytest.raises(ValueError, match="after a prefix of 32"):
        _check([0, 96], [0, 1], rows=3, prefix=[32])


@pytest.mark.parametrize(
    "cu,starts,rows,prefix",
    [
        ([0, 256, 384], [0, 4, 6], 6, None),
        ([0, 256, 512], [0, 4, 8], 8, None),
        # Shorter than the interval and zero-length sequences take no rows.
        ([0, 63, 63, 191], [0, 0, 0, 2], 2, None),
        # prefix_lens: (32 + 96) // 64 - 32 // 64 = 2 and (60 + 10) // 64 - 0 = 1.
        ([0, 96, 106], [0, 2, 3], 3, [32, 60]),
        # An offset into a larger buffer is fine while it stays in bounds.
        ([0, 128], [3, 5], 5, None),
    ],
)
def test_consistent_offsets_pass(validate, cu, starts, rows, prefix):
    _check(cu, starts, rows, prefix=prefix)


def test_wrong_counts_are_rejected(validate):
    with pytest.raises(ValueError, match=r"sequence 1 has 128 tokens, so 2 .* = 3\.$"):
        _check([0, 256, 384], [0, 4, 7], rows=8)


def test_wrong_length_is_rejected(validate):
    with pytest.raises(ValueError, match="num_seqs \\+ 1 = 3 entries, got 2"):
        _check([0, 256, 384], [0, 4], rows=6)


def test_rows_past_the_buffer_are_rejected(validate):
    with pytest.raises(
        ValueError, match=r"rows \[4, 6\) of state_checkpoints, which has 5 rows"
    ):
        _check([0, 256, 384], [0, 4, 6], rows=5)


def test_negative_start_is_rejected(validate):
    with pytest.raises(ValueError, match=r"rows \[-1, 1\)"):
        _check([0, 128], [-1, 1], rows=4)


def test_stale_host_offsets_do_not_hide_token_offsets(validate):
    # The kernels read the device cu_seqlens, so the check does too: a host
    # copy that disagrees with it must not mask an out-of-bounds write.
    t = 512
    q = torch.zeros(t, 2, 128, dtype=torch.bfloat16)
    v = torch.zeros(t, HV, 128, dtype=torch.bfloat16)
    cu = _i64([0, 256, t])
    with pytest.raises(ValueError, match="looks like token offsets"):
        gdn_prefill.chunk_gated_delta_rule(
            q,
            torch.zeros_like(q),
            v,
            cu_seqlens=cu,
            cu_seqlens_cpu=[0, 0, 0],
            checkpoint_every_n_tokens=EVERY,
            state_checkpoints=torch.zeros(8, HV, 128, 128),
            checkpoint_cu_starts=cu.clone(),
        )


def test_checkpoint_state_indices_bounds(validate):
    # With checkpoint_state_indices the starts index that tensor, and
    # [num_seqs] entries are accepted.
    _check([0, 256, 384], [0, 4], rows=10, indices=[9, 8, 7, 6, 5, -1])
    _check([0, 256, 384], [0, 4, 6], rows=10, indices=[9, 8, 7, 6, 5, -1])
    with pytest.raises(
        ValueError, match=r"rows \[4, 6\) of checkpoint_state_indices, which has 5"
    ):
        _check([0, 256, 384], [0, 4], rows=10, indices=[0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match=r"-1 or rows in \[0, 10\)"):
        _check([0, 256, 384], [0, 4], rows=10, indices=[0, 1, 2, 3, 4, 10])
    with pytest.raises(ValueError, match="num_seqs = 2 or num_seqs \\+ 1"):
        _check([0, 256, 384], [0], rows=10, indices=[0, 1, 2, 3, 4, 5])


def test_malformed_shapes_are_left_to_the_backends(validate):
    # Not 1-D: the backends raise their own shape errors.
    gdn_prefill._validate_checkpoint_cu_starts(
        cu_seqlens=_i64([0, 256, 384]),
        checkpoint_cu_starts=_i64([[0, 256, 384]]),
        checkpoint_every_n_tokens=EVERY,
        state_checkpoints=torch.zeros(6, HV, 1, 1),
        checkpoint_state_indices=None,
        prefix_lens=None,
    )


@pytest.mark.parametrize("value", [None, "", "0"])
def test_off_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FLASHINFER_VALIDATE_INPUTS", raising=False)
    else:
        monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", value)
    _check([0, 256, 384], [0, 256, 384], rows=6)


def test_skipped_during_graph_capture(validate, monkeypatch):
    # Reading the values would be a device-to-host copy inside the capture.
    seen = []
    monkeypatch.setattr(
        gdn_prefill.torch.cuda,
        "device",
        lambda device: seen.append(device) or contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        gdn_prefill.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    # Reading this fake's values would fail, so the check must return first.
    cu = types.SimpleNamespace(is_cuda=True, device="cuda:1", ndim=1)
    gdn_prefill._validate_checkpoint_cu_starts(
        cu_seqlens=cu,
        checkpoint_cu_starts=_i64([0, 256, 384]),
        checkpoint_every_n_tokens=EVERY,
        state_checkpoints=None,
        checkpoint_state_indices=None,
        prefix_lens=None,
    )
    # The capture state is read on the device the call runs on.
    assert seen == ["cuda:1"]
