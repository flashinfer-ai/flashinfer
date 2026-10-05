# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``moe_a2a_dispatch``'s optional receive-view cache.

The cache is a pure host-side optimization: it skips rebuilding the workspace-backed
views on every dispatch. These tests pin the three properties that make that safe, and
they run without a GPU or an MPI harness by stubbing the FFI module -- the real
``tests/comm/test_trtllm_moe_alltoall.py`` path needs multi-rank NVLink hardware.
"""

import torch

import flashinfer.comm.trtllm_moe_alltoall as a2a


_RECV_OFFSETS = [0, 512]
_RECV_SIZES = [256, 128]


class _FakeModule:
    """Stands in for the compiled all-to-all module."""

    @staticmethod
    def moe_a2a_dispatch(*args, **kwargs):
        # (recv_offsets, recv_sizes, combine_payload_offset,
        #  eplb_gathered_stats_offset, eplb_stats_num_experts)
        return _RECV_OFFSETS, _RECV_SIZES, 1024, -1, 0


def _dispatch(workspace, cache, wrap_calls):
    payloads = [torch.zeros(2, 4), torch.zeros(2, 2)]
    return a2a.moe_a2a_dispatch(
        torch.zeros(2, 1, dtype=torch.int32),
        payloads,
        workspace,
        torch.zeros(8, dtype=torch.int32),
        runtime_max_tokens_per_rank=2,
        ep_rank=0,
        ep_size=2,
        top_k=1,
        num_experts=4,
        enable_pdl=False,
        recv_view_cache=cache,
    )[0]


def _patch(monkeypatch, wrap_calls):
    monkeypatch.setattr(a2a, "get_moe_alltoall_module", lambda *a, **k: _FakeModule)

    def fake_wrap(workspace, leading_shape, slice_start, slice_end, dtype):
        wrap_calls.append((int(slice_start), int(slice_end), dtype))
        return torch.zeros(1)

    monkeypatch.setattr(a2a, "moe_a2a_wrap_payload_tensor_in_workspace", fake_wrap)


def test_cache_none_rebuilds_every_call(monkeypatch):
    """``None`` (the default) must preserve the pre-cache behaviour exactly."""
    calls = []
    _patch(monkeypatch, calls)
    ws = torch.zeros(4)
    _dispatch(ws, None, calls)
    _dispatch(ws, None, calls)
    assert len(calls) == 4, "two payloads x two calls should rebuild every view"


def test_cache_hits_on_second_dispatch(monkeypatch):
    """A populated cache must serve identical keys without rebuilding.

    This is the property that actually delivers the speedup, and it is the one that
    fails silently: a key that hashes by identity still "works", it just never hits.
    """
    calls = []
    _patch(monkeypatch, calls)
    ws = torch.zeros(4)
    cache = {}
    first = _dispatch(ws, cache, calls)
    assert len(calls) == 2
    second = _dispatch(ws, cache, calls)
    assert len(calls) == 2, f"expected cache hits, got {len(calls) - 2} rebuilds"
    for a, b in zip(first, second, strict=True):
        assert a is b, "a cache hit must hand back the same view object"


def test_cache_rebinds_when_workspace_changes(monkeypatch):
    """A new workspace must invalidate every cached view.

    Reusing a view across workspaces would alias a possibly freed allocation.
    """
    calls = []
    _patch(monkeypatch, calls)
    cache = {}
    _dispatch(torch.zeros(4), cache, calls)
    assert len(calls) == 2
    other = torch.zeros(4)
    _dispatch(other, cache, calls)
    assert len(calls) == 4, "a different workspace must force a rebuild"
    assert cache["_workspace"] is other


class _IdentityInt(int):
    """An ``int`` that hashes and compares by identity.

    Stands in for a hypothetical FFI integer wrapper.  ``tvm_ffi`` does not produce
    one today -- ``Array<int64_t>`` elements arrive as plain Python ints -- so this
    is the only way to exercise the guard the cache key rests on.
    """

    __hash__ = object.__hash__
    __eq__ = object.__eq__


def test_cache_key_survives_an_identity_hashing_ffi_int(monkeypatch):
    """The key must hash by value even if the FFI hands back a wrapper.

    ``moe_a2a_dispatch`` returns ``tvm::ffi::Array<int64_t>``, and each read boxes a
    fresh object (``a[0] is a[0]`` is False outside CPython's small-int intern
    range).  Value-based hashing is what makes that harmless.  Were the elements
    ever boxed in something that hashed by identity, every lookup would miss:
    results stay correct, the cache silently never hits, and the dict grows by one
    entry per payload per dispatch with no eviction.  The ``int()`` in the key is
    what rules that out, and this test is what keeps it there.
    """
    calls = []
    _patch(monkeypatch, calls)

    class _WrapperModule:
        @staticmethod
        def moe_a2a_dispatch(*args, **kwargs):
            # Fresh objects per call, mirroring the per-read boxing above.
            return (
                [_IdentityInt(o) for o in _RECV_OFFSETS],
                [_IdentityInt(s) for s in _RECV_SIZES],
                1024,
                -1,
                0,
            )

    monkeypatch.setattr(a2a, "get_moe_alltoall_module", lambda *a, **k: _WrapperModule)

    ws = torch.zeros(4)
    cache = {}
    _dispatch(ws, cache, calls)
    assert len(calls) == 2
    _dispatch(ws, cache, calls)
    assert len(calls) == 2, "the key must not depend on FFI integer identity"
    assert len(cache) == 3, "two payload entries plus the _workspace binding"
