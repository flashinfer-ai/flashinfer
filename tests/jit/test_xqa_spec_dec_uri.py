"""XQA speculative-decode module identity.

``SPEC_Q_SEQ_LEN`` (SWAP_AB) is the only compile-time use of the draft length,
and ``csrc/xqa/mha_sm90.cu`` is its only consumer. Every other draft length
reaches the kernel as a runtime argument, so one module serves all of them and
the module name must not split them apart. These tests are CPU-only: they
inspect JitSpecs and never build or load a module.
"""

import itertools

import pytest
import torch

from flashinfer.jit import core
from flashinfer.jit import xqa as jit_xqa
from flashinfer.jit.xqa import (
    gen_xqa_module,
    ragged_q_changes_build,
    swap_ab_eligible,
)

HEAD_GROUP_RATIOS = (1, 2, 4, 8, 16, 32, 64)
# Ratios that leave room for at least one SWAP_AB-eligible draft length
# (``q_seq_len * head_group_ratio <= 32`` with ``q_seq_len >= 2``).
SWAP_AB_HEAD_GROUP_RATIOS = (1, 2, 4, 8, 16)
DRAFT_LENS = tuple(range(2, 41))


@pytest.fixture
def sm90_target(monkeypatch):
    """Pin the build targets so module identity does not depend on the host GPU."""
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "9.0a 10.0a")
    jit_xqa._has_sm90_target.cache_clear()
    yield
    jit_xqa._has_sm90_target.cache_clear()


def _spec(head_group_ratio, q_seq_len, use_ragged_q):
    """The JitSpec for one request, with ragged Q normalized as the caller does."""
    use_ragged_q = use_ragged_q and ragged_q_changes_build(q_seq_len, head_group_ratio)
    return gen_xqa_module(
        input_dtype=torch.float16,
        kv_cache_dtype=torch.float8_e4m3fn,
        page_size=32,
        head_dim=128,
        head_group_ratio=head_group_ratio,
        use_sliding_window=False,
        output_dtype=torch.float16,
        q_seq_len=q_seq_len,
        use_ragged_q=use_ragged_q,
    )


def _build(spec):
    return (tuple(sorted(spec.extra_cuda_cflags)), tuple(map(str, spec.sources)))


def test_module_name_and_build_are_one_to_one(sm90_target):
    """A name must identify a build, and a build must have one name.

    The second half fails on an unpatched tree: the name carries the requested
    draft length even where nvcc never sees it, so identical builds get a module
    each.
    """
    by_name = {}
    by_build = {}
    for head_group_ratio, q_seq_len, use_ragged_q in itertools.product(
        HEAD_GROUP_RATIOS, DRAFT_LENS, (False, True)
    ):
        spec = _spec(head_group_ratio, q_seq_len, use_ragged_q)
        by_name.setdefault(spec.name, set()).add(_build(spec))
        by_build.setdefault((head_group_ratio, _build(spec)), set()).add(spec.name)

    collisions = {name: b for name, b in by_name.items() if len(b) > 1}
    assert not collisions, f"one name, several builds: {sorted(collisions)}"

    duplicates = {key: n for key, n in by_build.items() if len(n) > 1}
    assert not duplicates, (
        "identical builds compiled under several module names: "
        f"{sorted(next(iter(duplicates.values())))}"
    )


@pytest.mark.parametrize("head_group_ratio", HEAD_GROUP_RATIOS)
def test_generic_draft_lengths_share_one_module(head_group_ratio, sm90_target):
    """Lengths that compile no SWAP_AB specialization share a single module."""
    generic = [q for q in DRAFT_LENS if not swap_ab_eligible(q, head_group_ratio)]
    assert generic, "pick a grid that exercises the generic build"
    names = {_spec(head_group_ratio, q, False).name for q in generic}
    assert len(names) == 1, f"{len(names)} modules for {len(generic)} draft lengths"


@pytest.mark.parametrize("head_group_ratio", HEAD_GROUP_RATIOS)
def test_ragged_q_shares_one_module_across_draft_lengths(head_group_ratio, sm90_target):
    """Ragged Q always suppresses SWAP_AB, so every draft length is one module."""
    names = {_spec(head_group_ratio, q, True).name for q in DRAFT_LENS}
    assert len(names) == 1, f"ragged Q split into {len(names)} modules"


@pytest.mark.parametrize("head_group_ratio", SWAP_AB_HEAD_GROUP_RATIOS)
def test_swap_ab_lengths_keep_a_module_each(head_group_ratio, sm90_target):
    """The specialization is per-length, so those modules must stay distinct."""
    specialized = [q for q in DRAFT_LENS if swap_ab_eligible(q, head_group_ratio)]
    assert specialized, "pick a grid that exercises the specialization"
    names = {_spec(head_group_ratio, q, False).name for q in specialized}
    assert len(names) == len(specialized)
    for q in specialized:
        spec = _spec(head_group_ratio, q, False)
        assert f"-DSPEC_Q_SEQ_LEN={q}" in spec.extra_cuda_cflags


@pytest.mark.parametrize("head_group_ratio", HEAD_GROUP_RATIOS)
def test_flags_unchanged_for_every_draft_length(head_group_ratio, sm90_target):
    """Sharing a module must not change what nvcc is asked to compile."""
    for q_seq_len, use_ragged_q in itertools.product(DRAFT_LENS, (False, True)):
        shared = _spec(head_group_ratio, q_seq_len, use_ragged_q)
        direct = gen_xqa_module(
            input_dtype=torch.float16,
            kv_cache_dtype=torch.float8_e4m3fn,
            page_size=32,
            head_dim=128,
            head_group_ratio=head_group_ratio,
            use_sliding_window=False,
            output_dtype=torch.float16,
            q_seq_len=q_seq_len,
            use_ragged_q=use_ragged_q,
        )
        assert _build(shared) == _build(direct), (
            f"head_group_ratio={head_group_ratio} q_seq_len={q_seq_len} "
            f"use_ragged_q={use_ragged_q}"
        )


def test_non_spec_dec_module_is_unaffected(sm90_target):
    """q_seq_len == 1 keeps its name and its SPEC_DEC=0 build."""
    spec = _spec(8, 1, False)
    assert spec.name.endswith("use_spec_dec_False_spec_q_seq_len_1")
    assert "-DSPEC_DEC=0" in spec.extra_cuda_cflags


def test_module_key_collapses_identical_requests(sm90_target):
    """The cache key must collapse too, or an identical module is built twice
    under the same name and the same torch op is registered again."""
    from flashinfer.jit.xqa import xqa_module_key

    for head_group_ratio in HEAD_GROUP_RATIOS:
        keys = set()
        for q_seq_len, use_ragged_q in itertools.product(DRAFT_LENS, (False, True)):
            keys.add(xqa_module_key(q_seq_len, head_group_ratio, use_ragged_q))
        names = {
            _spec(head_group_ratio, q, r).name
            for q, r in itertools.product(DRAFT_LENS, (False, True))
        }
        assert len(keys) == len(names), (
            f"head_group_ratio={head_group_ratio}: {len(keys)} keys for "
            f"{len(names)} modules"
        )


@pytest.mark.parametrize("head_group_ratio", (33, 64, 128))
def test_group_ratio_past_the_swap_ab_budget_still_spec_dec(head_group_ratio):
    """No draft length is SWAP_AB-eligible once the group ratio alone exceeds
    the row budget; the shared module must still be a spec-dec build."""
    from flashinfer.jit.xqa import generic_spec_dec_q_seq_len, xqa_module_key

    q_seq_len = generic_spec_dec_q_seq_len(head_group_ratio)
    assert q_seq_len > 1, "a spec-dec module requires q_seq_len > 1"
    assert not swap_ab_eligible(q_seq_len, head_group_ratio)
    for requested in DRAFT_LENS:
        assert xqa_module_key(requested, head_group_ratio, False) == (
            q_seq_len,
            False,
        )
