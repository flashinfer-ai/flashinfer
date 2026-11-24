# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepared, allocation-free sample/filter/select/recover launch chains."""

import torch
import cutlass

DTYPES = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.uint8: cutlass.Uint8,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


class ThrowawayTopK:
    """Four-stage prefill with exact TopK recovery. Ranking uses the requested logits
    precision. Indices tied at the cutoff may vary.

    Raw callers supply monotone causal ends, 0 <= starts <= ends <= N, score ends
    in [0, N] and FP8 scales for all N keys on each replay. With candidate blocks,
    active rows start at zero. Each block addresses eight keys. Candidate valid
    lengths are writable K1 scratch. Returned indices address the source keys.
    """

    def __init__(
        self,
        q,
        k,
        weights,
        starts,
        ends,
        *,
        k_scale=None,
        q_scale=None,
        top_k=2048,
        logits_dtype=torch.float32,
        phase="prefill",
        capacity=None,
        _filter_config=None,
        model_config=None,
        pdl=True,
        math_mode="relu_strict",
        out=None,
        _score_ends=None,
        compression_ratio=None,
        _validate_numerics=True,
        candidate_k_block_ids=None,
        candidate_valid_lens=None,
        _candidate_error=None,
    ):
        self.requested_logits_dtype = logits_dtype
        self.pdl = bool(pdl)
        if phase != "prefill":
            raise ValueError("only prefill is supported")
        if top_k not in (512, 1024, 2048):
            raise ValueError("top_k must be 512/1024/2048")
        if logits_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("unsupported logits dtype")
        if q.dtype not in (torch.float8_e4m3fn, torch.int8, torch.uint8):
            raise ValueError("Q must be E4M3 or packed MXFP4 bytes")
        self.qkv_dtype = (
            ("mxfp8" if q_scale is not None else "fp8")
            if q.dtype == torch.float8_e4m3fn
            else "mxfp4"
        )
        if self.qkv_dtype in ("fp8", "mxfp8") and k.dtype != q.dtype:
            raise ValueError("FP8 Q/K dtypes must match")
        if self.qkv_dtype == "mxfp4" and k.dtype not in (torch.int8, torch.uint8):
            raise ValueError("MXFP4 K must contain packed bytes")
        width = 64 if self.qkv_dtype == "mxfp4" else 128
        if (
            q.ndim != 3
            or q.shape[1] not in (4, 32, 64)
            or q.shape[2] != width
            or k.ndim != 2
            or k.shape[1] != width
        ):
            raise ValueError("requires H4/H32/H64 prefill, D128 inputs")
        max_keys = 1048576 + (q.shape[0] if candidate_k_block_ids is not None else 0)
        if q.shape[0] == 0 or not 0 < k.shape[0] <= max_keys:
            raise ValueError(f"requires nonempty Q and 1..{max_keys} K entries")
        if weights.shape != q.shape[:2] or weights.dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("FP32 or BF16 weights [M,H] required")
        for bounds in (starts, ends):
            if bounds.shape != (q.shape[0],) or bounds.dtype != torch.int32:
                raise ValueError("INT32 [M] bounds required")
        tensors = [q, k, weights, starts, ends] + [
            x for x in (q_scale, k_scale) if x is not None
        ]
        if any(
            not x.is_cuda or x.device != q.device or not x.is_contiguous()
            for x in tensors
        ):
            raise ValueError("all inputs must be contiguous CUDA tensors on one device")
        # Infer policy from compression_ratio; raw callers may select a model explicitly.
        if compression_ratio is not None:
            from ...indexer_config import resolve_model_config

            model_config = resolve_model_config(
                q.shape[1], 128, top_k, compression_ratio, model_config
            )
        candidate_role = candidate_k_block_ids is not None or (_filter_config or {}).get(
            "publish_blocks"
        )
        if candidate_role and model_config == "DS-V4.1-Flash":
            if compression_ratio not in (None, 1):
                raise ValueError(
                    "DS-V4.1-Flash candidate layers require compression_ratio=1"
                )
            compression_ratio = 1
        if candidate_role and _validate_numerics:
            if bool(((starts != 0) & (starts < ends)).any().item()):
                raise ValueError(
                    "candidate publication and consumption require zero-based single-request bounds"
                )
        # Preparation and replay validate model-specific numerical contracts.
        if model_config is not None:
            if model_config not in (
                "DS-V3.2",
                "Qwen3.8-Flash-Next",
                "GLM-5.3-Flash",
                "DS-V4-Flash",
                "DS-V4-Pro",
                "GLM-5.3",
                "Kimi-K3.1-Pro",
                "DS-V4.1-Flash",
            ):
                raise ValueError("unknown indexer model_config")
            if (
                model_config == "Kimi-K3.1-Pro"
                and self.qkv_dtype == "fp8"
                and math_mode == "relu_strict"
            ):
                # Only indices are returned. Narrowing adds ties but preserves order.
                logits_dtype = torch.float32
                _filter_config = dict(_filter_config or {}, kimi=True)
            if model_config == "DS-V4.1-Flash":
                if q.shape[1] != 32 or top_k != 512:
                    raise ValueError("DS-V4.1-Flash requires prefill H32/K512")
                policy = dict(_filter_config or {})
                # MXFP4 native BF16 scores are exact in FP32; FP16 conversion
                # is monotone. Ranking in BF16 preserves valid TopK ties.
                if (
                    self.qkv_dtype == "mxfp4"
                    and math_mode == "native_bf16"
                    and candidate_k_block_ids is None
                ):
                    logits_dtype = torch.bfloat16
                if (
                    self.qkv_dtype == "mxfp4"
                    and math_mode == "relu_strict"
                    and policy.get("publish_blocks")
                ):
                    policy.setdefault("candidate_max_factor", 24)
                policy["ds41"] = True
                if self.qkv_dtype == "mxfp4":
                    policy.setdefault("prepare_kv_scales", True)
                    if (
                        math_mode == "native_bf16"
                        and logits_dtype == torch.bfloat16
                        and (
                            k.shape[0] >= 262144
                            or (q.shape[0] >= 8192 and k.shape[0] >= 65536)
                        )
                        and not policy.get("publish_blocks")
                    ):
                        policy.setdefault("shared_kv_queries", True)
                        policy.setdefault("kv_pairing", True)
                _filter_config = policy
            if model_config == "DS-V3.2" and q.shape[1] != 64:
                raise ValueError("DS-V3.2 requires H64")
            if model_config in ("GLM-5.3", "Kimi-K3.1-Pro") and (
                q.shape[1] != 32 or top_k != 2048
            ):
                raise ValueError(f"{model_config} requires prefill H32/K2048")
            if model_config in ("DS-V4-Flash", "DS-V4-Pro"):
                expected_k = 1024 if model_config == "DS-V4-Pro" else 512
                if phase != "prefill" or q.shape[1] != 64 or top_k != expected_k:
                    raise ValueError(
                        f"{model_config} requires prefill H64/K{expected_k}"
                    )
            if model_config in ("DS-V4-Flash", "DS-V4-Pro"):
                policy = dict(_filter_config or {})
                if "dsv4" in policy and policy["dsv4"] is not True:
                    raise ValueError("model_config conflicts with DS-V4 policy")
                policy["dsv4"] = True
                _filter_config = policy
            if model_config == "GLM-5.3-Flash" and (
                phase != "prefill" or q.shape[1] != 32 or top_k != 512
            ):
                raise ValueError("GLM-5.3-Flash requires prefill H32/K512")
            if model_config == "Qwen3.8-Flash-Next":
                if phase != "prefill" or q.shape[1] != 4 or top_k != 512:
                    raise ValueError("Qwen3.8-Flash-Next requires prefill H4/K512")
                policy = dict(_filter_config or {})
                required = dict(qsa_weights=2, qsa_unit_scale=self.qkv_dtype == "fp8")
                if any(
                    key in policy and policy[key] != value
                    for key, value in required.items()
                ):
                    raise ValueError("model_config conflicts with filter policy")
                policy.update(required)
                _filter_config = policy
        if compression_ratio is None:
            from ...indexer_config import MODEL_CONFIGS

            compression_ratio = (
                MODEL_CONFIGS[model_config][3] if model_config is not None else 1
            )
        from .prefill.api import ThrowawayTopK as PrefillTopK

        backend = PrefillTopK(
            q,
            k,
            weights,
            starts,
            ends,
            k_scale=k_scale,
            q_scale=q_scale,
            top_k=top_k,
            logits_dtype=logits_dtype,
            phase=phase,
            capacity=capacity,
            _filter_config=_filter_config,
            pdl=self.pdl,
            math_mode=math_mode,
            out=out,
            _score_ends=_score_ends,
            _validate_numerics=_validate_numerics,
            compression_ratio=compression_ratio,
            candidate_k_block_ids=candidate_k_block_ids,
            candidate_valid_lens=candidate_valid_lens,
            _candidate_error=_candidate_error,
        )
        self.__dict__.update(backend.__dict__)
        self.effective_logits_dtype = logits_dtype
        self._backend = backend

    def prepare_live_bounds(self, *args):
        from .prefill.api import ThrowawayTopK as PrefillTopK

        PrefillTopK.prepare_live_bounds(self, *args)

    def run(self, candidate_k_block_ids=None):
        from .prefill.api import ThrowawayTopK as PrefillTopK

        return PrefillTopK.run(self, candidate_k_block_ids)
