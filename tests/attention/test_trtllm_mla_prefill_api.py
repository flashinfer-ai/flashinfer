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

import importlib
import inspect

import pytest

import flashinfer
import flashinfer.decode
import flashinfer.mla
import flashinfer.prefill


_TRTLLM_MLA_PARAMETERS = (
    "query",
    "kv_cache",
    "workspace_buffer",
    "qk_nope_head_dim",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "block_tables",
    "seq_lens",
    "max_seq_len",
    "sparse_mla_top_k",
    "out",
    "bmm1_scale",
    "bmm2_scale",
    "sinks",
    "skip_softmax_threshold_scale_factor",
    "enable_pdl",
    "backend",
    "is_var_seq",
    "uses_shared_paged_kv_idx",
    "lse",
    "return_lse",
    "cute_dsl_impl",
    "kv_scale_format",
    "cum_seq_lens_q",
    "max_q_len",
    "multi_ctas_kv_counter_buffer",
    "sparse_mla_top_k_lens",
    "enable_dcp",
    "cp_world",
    "cp_rank",
    "causal_seqlens_kv_global",
    "use_fp16_softmax",
)
_NUM_REQUIRED_PARAMETERS = 9


def test_trtllm_mla_prefill_has_exact_approved_exports():
    prefill_api = flashinfer.mla.trtllm_prefill_with_kv_cache_mla
    decode_api = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla

    assert flashinfer.prefill.trtllm_prefill_with_kv_cache_mla is prefill_api
    assert flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla is decode_api
    assert "trtllm_prefill_with_kv_cache_mla" not in vars(flashinfer)
    assert "trtllm_prefill_with_kv_cache_mla" not in vars(flashinfer.decode)

    assert prefill_api is not decode_api
    assert prefill_api.__name__ == "trtllm_prefill_with_kv_cache_mla"
    assert decode_api.__name__ == "trtllm_batch_decode_with_kv_cache_mla"


def test_trtllm_mla_prefill_signature_annotations_and_defaults_match_decode():
    prefill_api = flashinfer.mla.trtllm_prefill_with_kv_cache_mla
    decode_api = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla
    prefill_signature = inspect.signature(prefill_api)
    decode_signature = inspect.signature(decode_api)

    assert tuple(prefill_signature.parameters) == _TRTLLM_MLA_PARAMETERS
    assert prefill_signature == decode_signature


def _patch_shared_impl_and_forbid_decode_routing(monkeypatch):
    core = importlib.import_module("flashinfer.mla._core")
    calls = []
    result = object()

    def capture_impl(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("prefill must not route through the public decode wrapper")

    monkeypatch.setattr(
        core, "_trtllm_batch_decode_with_kv_cache_mla_impl", capture_impl
    )
    monkeypatch.setattr(core, "trtllm_batch_decode_with_kv_cache_mla", forbidden_decode)
    monkeypatch.setattr(
        flashinfer.mla, "trtllm_batch_decode_with_kv_cache_mla", forbidden_decode
    )
    monkeypatch.setattr(
        flashinfer.decode, "trtllm_batch_decode_with_kv_cache_mla", forbidden_decode
    )
    return calls, result


@pytest.mark.parametrize(
    "api_name",
    ["trtllm_batch_decode_with_kv_cache_mla", "trtllm_prefill_with_kv_cache_mla"],
    ids=["decode", "prefill"],
)
def test_trtllm_mla_forwards_every_argument_to_shared_impl(monkeypatch, api_name):
    api = getattr(flashinfer.mla, api_name)
    signature = inspect.signature(api)
    values = tuple(object() for _ in _TRTLLM_MLA_PARAMETERS)
    calls, expected_result = _patch_shared_impl_and_forbid_decode_routing(monkeypatch)

    actual_result = api(*values)

    assert actual_result is expected_result
    assert len(calls) == 1
    forwarded = signature.bind(*calls[0][0], **calls[0][1])
    assert tuple(forwarded.arguments) == _TRTLLM_MLA_PARAMETERS
    for name, value in zip(_TRTLLM_MLA_PARAMETERS, values, strict=True):
        assert forwarded.arguments[name] is value


def test_trtllm_mla_prefill_forwards_required_only_call_with_decode_defaults(
    monkeypatch,
):
    prefill_api = flashinfer.mla.trtllm_prefill_with_kv_cache_mla
    signature = inspect.signature(prefill_api)
    required_values = tuple(object() for _ in range(_NUM_REQUIRED_PARAMETERS))
    expected = signature.bind(*required_values)
    expected.apply_defaults()
    calls, expected_result = _patch_shared_impl_and_forbid_decode_routing(monkeypatch)

    actual_result = prefill_api(*required_values)

    assert actual_result is expected_result
    assert len(calls) == 1
    forwarded = signature.bind(*calls[0][0], **calls[0][1])
    assert tuple(forwarded.arguments) == _TRTLLM_MLA_PARAMETERS
    for name in _TRTLLM_MLA_PARAMETERS[:_NUM_REQUIRED_PARAMETERS]:
        assert forwarded.arguments[name] is expected.arguments[name]
    for name in _TRTLLM_MLA_PARAMETERS[_NUM_REQUIRED_PARAMETERS:]:
        assert forwarded.arguments[name] == expected.arguments[name]
