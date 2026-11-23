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

"""Prepare four allocation-free CuTe DSL prefill stages.

Stratified sampling proposes thresholds; exact recovery handles failed rows."""

import torch
from cutlass import testing
from ..api import DTYPES
from ..mqa_logits_util import KernelLaunch
from .upper_compile import compile_sampler
from .safe_filter_compile import compile_filter
from .crossing_precounted import compile_selector as compile_precounted_selector
from .queue_recovery_compile import _compile_gemm
from .crossing_precounted_mxfp4 import compile_selector as compile_mxfp4_selector


class ThrowawayTopK:
    """Prepare prefill stages over device bounds supplied by the caller.

    Every run requires 0 <= starts <= ends <= N and score ends in [0, N].
    Bounds are not read back or validated here; FP8 scales must cover N keys.
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
        chunk_rows=256,
        pdl=True,
        math_mode="relu_strict",
        out=None,
        _score_ends=None,
        _validate_numerics=True,
        compression_ratio=1,
        candidate_k_block_ids=None,
        candidate_valid_lens=None,
        _candidate_error=None,
    ):
        self.pdl = bool(pdl)
        if phase != "prefill":
            raise ValueError("this backend implements prefill")
        if top_k not in (512, 1024, 2048):
            raise ValueError("top_k must be 512, 1024, or 2048")
        if logits_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("unsupported logits dtype")
        self.qkv_dtype = (
            ("mxfp8" if q_scale is not None else "fp8")
            if q.dtype == torch.float8_e4m3fn
            else "mxfp4"
        )
        width = 64 if self.qkv_dtype == "mxfp4" else 128
        if q.dtype not in (torch.float8_e4m3fn, torch.uint8, torch.int8):
            raise ValueError("Q must contain FP8 or packed MXFP4")
        if (
            q.ndim != 3
            or q.shape[1] not in (4, 32, 64)
            or q.shape[2] != width
            or k.ndim != 2
            or k.shape[1] != width
        ):
            raise ValueError("requires H4/H32/H64, D128 inputs")
        if (self.qkv_dtype in ("fp8", "mxfp8") and k.dtype != q.dtype) or (
            self.qkv_dtype == "mxfp4" and k.dtype not in (torch.uint8, torch.int8)
        ):
            raise ValueError("Q and K encodings must match")
        m, n = q.shape[0], k.shape[0]
        heads = q.shape[1]
        from ...._selective_math_policy import validate_math_mode

        math_mode = validate_math_mode(
            math_mode,
            qkv_dtype=self.qkv_dtype,
            num_heads=heads,
            logits_dtype=logits_dtype,
        )
        native_bf16 = math_mode == "native_bf16"
        native_fp16 = math_mode == "native_fp16"
        # Use exhaustive candidates only when the query span covers physical K.
        # Earlier context still requires sampled thresholds.
        exhaustive = (
            not (heads in (4, 32) and (native_bf16 or native_fp16))
            and self.qkv_dtype != "mxfp8"
            and n <= 4 * top_k
            and n * compression_ratio <= m
        )
        max_keys = 1048576 + (m if candidate_k_block_ids is not None else 0)
        if m == 0 or not 0 < n <= max_keys:
            raise ValueError(f"requires nonempty Q and 1..{max_keys} keys")
        if weights.shape != q.shape[:2] or weights.dtype not in (
            torch.float32,
            torch.bfloat16,
        ):
            raise ValueError("weights must be FP32 or BF16 [M,H]")
        if any(t.shape != (m,) or t.dtype != torch.int32 for t in (starts, ends)):
            raise ValueError("bounds must be INT32 [M]")
        if _score_ends is not None and (
            _score_ends.shape != (m,) or _score_ends.dtype != torch.int32
        ):
            raise ValueError("score ends must be INT32 [M]")
        if self.qkv_dtype == "fp8" and k_scale is None:
            k_scale = torch.ones(n, dtype=torch.float32, device=q.device)
        if k_scale is None or (
            self.qkv_dtype in ("mxfp4", "mxfp8") and q_scale is None
        ):
            raise ValueError("missing quantization scales")
        if self.qkv_dtype == "fp8" and (
            k_scale.shape != (n,) or k_scale.dtype != torch.float32
        ):
            raise ValueError("FP8 key scales must be FP32 [N]")
        tensors = (
            [q, k, weights, starts, ends, k_scale]
            + ([] if q_scale is None else [q_scale])
            + ([] if _score_ends is None else [_score_ends])
        )
        if any(
            not t.is_cuda or t.device != q.device or not t.is_contiguous()
            for t in tensors
        ):
            raise ValueError("inputs must be contiguous CUDA tensors on one device")
        if out is not None and (
            out.shape != (m, top_k)
            or out.dtype != torch.int32
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be contiguous INT32 [M,top_k] on the input device"
            )
        self.output = (
            torch.empty((m, top_k), device=q.device, dtype=torch.int32)
            if out is None
            else out
        )
        # Zero: a step that supplies fewer rows than m launches K1-K3 for those rows only, and the rest must read as empty.
        self.counts = torch.zeros((m,), device=q.device, dtype=torch.int32)
        if (
            candidate_k_block_ids is not None
            or (_filter_config or {}).get("publish_blocks")
        ) and math_mode == "relu_fast":
            raise ValueError(
                "candidate publication and consumption require strict or native head reduction"
            )
        self.candidate_k_block_ids = candidate_k_block_ids
        self.candidate_valid_lens = candidate_valid_lens
        if candidate_k_block_ids is not None:
            if (
                self.qkv_dtype != "mxfp4"
                or heads != 32
                or top_k != 512
                or compression_ratio not in (None, 1)
                or (_filter_config or {}).get("publish_blocks")
            ):
                raise ValueError(
                    "candidate blocks require MXFP4 H32/K512, ratio 1, without publication"
                )
            if (
                candidate_k_block_ids.shape != (m, 2048)
                or candidate_k_block_ids.dtype != torch.int32
                or candidate_valid_lens is None
                or candidate_valid_lens.shape != (m,)
                or candidate_valid_lens.dtype != torch.int32
            ):
                raise ValueError(
                    "candidate blocks require INT32 [M,2048] IDs and [M] valid token counts"
                )
            for tensor in (candidate_k_block_ids, candidate_valid_lens):
                if (
                    tensor.device != q.device
                    or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16
                ):
                    raise ValueError(
                        "candidate metadata must be aligned contiguous storage on the input device"
                    )
            if (
                q_scale.dtype not in (torch.int32, torch.uint8)
                or q_scale.shape
                != ((m, heads) if q_scale.dtype == torch.int32 else (m, heads, 4))
                or k_scale.dtype != torch.uint8
                or k_scale.shape != (n, 4)
                or n % 256
                or m % 8
            ):
                raise ValueError(
                    "candidate scoring requires aligned MXFP4 Q/K scales, M divisible by 8 and N by 256"
                )
            from .sparse.prepare import prepare_stages

            prepare_stages(
                self,
                dict(
                    q=q,
                    k=k,
                    weights=weights,
                    starts=starts,
                    ends=ends,
                    q_scale=q_scale.view(torch.int32).reshape(m, heads),
                    k_scale=k_scale,
                    candidate_blocks=candidate_k_block_ids,
                    candidate_valid_lens=candidate_valid_lens,
                    error=_candidate_error
                    if _candidate_error is not None
                    else torch.zeros(1, device=q.device, dtype=torch.int32),
                ),
                math_mode,
                logits_dtype,
            )
            return
        if candidate_valid_lens is not None:
            raise ValueError("candidate_valid_lens requires candidate_k_block_ids")
        policy = _filter_config or {}
        if policy.get("publish_blocks") and m > 65535:
            raise ValueError("candidate publication supports at most 65535 query rows")
        if "query_tile" in policy or "selection_cache_twelve" in policy:
            raise ValueError("unsupported private filter configuration")
        short_publisher = bool(
            policy.get("ds41")
            and policy.get("publish_blocks")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and self.qkv_dtype == "mxfp4"
            and heads == 32
            and top_k == 512
            and compression_ratio == 1
            and m >= 8192
            and m % 4 == 0
            and 65536 < n <= 131072
            and policy.get("kv_pairing", True)
            and not policy.get("shared_kv_queries", False)
        )
        self.bank_local_candidates = short_publisher

        dsv4 = policy.get("dsv4", False)
        if dsv4 and (heads != 64 or top_k not in (512, 1024)):
            raise ValueError("DS-V4 policy requires H64/K512 or K1024")
        # K1 refreshes UTCCP scales per run; K2 reuses them to avoid
        # repeated shared-memory transposes within these shape ranges.
        glm_prepare_sf = (
            heads == 32
            and self.qkv_dtype in ("mxfp4", "mxfp8")
            and (
                (top_k == 512 and 32 * top_k <= n < 256 * top_k)
                or (top_k == 2048 and 32 * top_k <= n <= 128 * top_k)
            )
        )
        glm_prepare_sf = policy.get("prepare_kv_scales", glm_prepare_sf)
        dsv32_prepare_sf = (
            heads == 64
            and top_k == 2048
            and self.qkv_dtype in ("mxfp4", "mxfp8")
            and n >= 32 * top_k
        )
        prepare_kv_sf = glm_prepare_sf or dsv32_prepare_sf
        qsa_weights = policy.get("qsa_weights", 0)
        qsa_unit_scale = policy.get("qsa_unit_scale", False)
        if qsa_weights and _validate_numerics:
            if heads != 4:
                raise ValueError("QSA specialization requires H4")
            if qsa_weights == 1:
                valid = (weights == weights[:, :1]).all()
            else:
                valid = (
                    weights
                    == torch.tensor(128**-0.5, device=q.device, dtype=weights.dtype)
                ).all()
            if not valid.item():
                raise ValueError("QSA weight contract violated")
        if (
            _validate_numerics
            and qsa_unit_scale
            and (self.qkv_dtype != "fp8" or not (k_scale == 1).all().item())
        ):
            raise ValueError("QSA unit scale contract violated")

        if weights.dtype == torch.bfloat16 and not (native_bf16 or native_fp16):
            # FP32 head math must read the BF16-rounded constant. Native modes
            # already round to 181/2048 (45.25 after the FP8/FP16 rebalance).
            qsa_weights = 0
        self.phase, self.top_k = phase, top_k
        self.capacity = capacity or 24 * top_k
        self.packed, self.sample_size, self.recovery_qtile = False, 512, 128 // heads
        qtile = self.recovery_qtile
        rows, cols = (m + qtile - 1) // qtile * qtile, (n + 255) // 256 * 256

        def empty(shape, dtype):
            return torch.empty(shape, device=q.device, dtype=dtype)

        weight_offset = (m + 31) // 32 * 32
        filter_rows = (m + 7) // 8 * 8 if heads == 32 and self.qkv_dtype == "fp8" else m
        metadata_words = (
            weight_offset
            + (
                rows
                if heads in (4, 32) and self.qkv_dtype in ("mxfp4", "mxfp8")
                else filter_rows
            )
            * heads
        )
        if (dsv4 or dsv32_prepare_sf) and self.qkv_dtype in ("mxfp4", "mxfp8"):
            metadata_words = max(metadata_words, weight_offset + cols)
        weights_end = metadata_words
        if glm_prepare_sf:
            metadata_words += cols
        self._threshold_storage = empty((metadata_words,), torch.float32)
        self.thresholds = self._threshold_storage[:m]
        prepared_weights = self._threshold_storage[weight_offset:weights_end]
        if (
            heads in (4, 32) and self.qkv_dtype in ("mxfp4", "mxfp8")
        ) or filter_rows > m:
            prepared_weights.zero_()
        # K2 candidates die after K3. K4 reuses the same per-row arena,
        # with dense scores and recovery candidates disjoint during recovery.
        score_bytes = torch.empty((), dtype=logits_dtype).element_size()
        # Size candidates by template and physical K extent. Large-query
        # block-scaled K512 calls trade extra workspace for fewer overflow rows.
        candidate_budget = (
            cols // 4
            if m >= 8192
            and heads in (32, 64)
            and top_k == 512
            and self.qkv_dtype in ("mxfp4", "mxfp8")
            else 3 * cols // 16
        )
        candidate_cols = min(
            cols,
            policy.get("candidate_max_factor", 18 if heads in (4, 32) else 16) * top_k,
            max(
                top_k,
                min(
                    (8 if heads in (4, 32) else 4) * top_k,
                    ((cols // 2 + 255) // 256) * 256,
                ),
                ((candidate_budget + 255) // 256) * 256,
            ),
        )
        if heads == 32 and top_k != 2048 and 32 * top_k <= cols <= 64 * top_k:
            candidate_cols = max(candidate_cols, min(cols, 10 * top_k))
        if dsv4 and 32 * top_k <= n <= 128 * top_k:
            candidate_cols = max(
                candidate_cols, min(cols, (8 if top_k == 1024 else 10) * top_k)
            )
        if dsv4 and top_k == 512 and 8 * top_k < n < 32 * top_k:
            candidate_cols = max(candidate_cols, min(cols, 6 * top_k))
        if dsv4 and top_k == 1024:
            if n <= 8 * top_k:
                candidate_cols = max(candidate_cols, min(cols, 4 * top_k))
            elif 16 * top_k <= n < 32 * top_k:
                candidate_cols = max(candidate_cols, min(cols, 9 * top_k // 2))
        if heads in (32, 64) and top_k == 2048 and n <= 8 * top_k:
            candidate_cols = max(candidate_cols, min(cols, 3 * top_k))
        # For FP8 top_k=2048 rows past 64*k, K1 keeps candidate tails narrow (H64 samples grow with length; H32
        # grows past 128*k in sampler_upper_fp8): max ~5.5*k (H64) / ~6.6*k (H32) where the slab was 15-18*k
        # (1.9-2.4 GiB at Q8192). Cap the per-row slab at 10*k; a row above it takes the existing exact recovery.
        if (
            self.qkv_dtype == "fp8"
            and top_k == 2048
            and heads in (32, 64)
            and not dsv4
            and n > 64 * top_k
        ):
            candidate_cols = min(candidate_cols, 10 * top_k)
        # A short physical domain fits in the candidate arena in full. Avoid
        # overflowing conservative thresholds when K is a large fraction of N.
        if n <= 4 * top_k:
            candidate_cols = cols
        recovery_words = 32 * top_k + 8
        words_per_recovery_row = cols * score_bytes // 8 + recovery_words
        sample_target = max(256, min(2048, (n * 4 + top_k - 1) // top_k))
        if (
            m >= 8192
            and heads == 4
            and self.qkv_dtype in ("mxfp4", "mxfp8")
            and n >= 256 * top_k
        ):
            candidate_cols = max(candidate_cols, min(cols, 24 * top_k))
        if (
            m >= 8192
            and dsv4
            and top_k == 512
            and self.qkv_dtype == "fp8"
            and logits_dtype == torch.float32
            and 32 * top_k <= n < 64 * top_k
        ):
            candidate_cols = min(candidate_cols, 19 * top_k // 2)
        sample_extent = (1 << (sample_target - 1).bit_length()) * 2
        # Public indices remain int32. Only compact K2/K3 payloads use index16.
        self.packed = (
            (
                heads == 32
                or (heads == 4 and self.qkv_dtype == "fp8" and n <= 32 * top_k)
                or (
                    dsv4
                    and (
                        top_k == 1024
                        or (self.qkv_dtype in ("mxfp4", "mxfp8") and n > 8 * top_k)
                    )
                )
                or (heads == 64 and top_k == 2048)
            )
            and logits_dtype in (torch.float16, torch.bfloat16)
            and cols <= 65536
        )
        if heads == 4 and self.qkv_dtype == "fp8" and (native_bf16 or native_fp16):
            self.packed = False
        candidate_words = rows * (candidate_cols + 8) // (2 if self.packed else 1)
        arena_words = max(
            candidate_words,
            qtile * words_per_recovery_row,
            rows * (sample_extent + 32) * score_bytes // 8,
        )
        self._candidate_arena = empty((arena_words,), torch.int64)
        self.candidates = (
            self._candidate_arena[:candidate_words]
            .view(torch.int32 if self.packed else torch.int64)
            .view(rows, candidate_cols + 8)
        )
        if short_publisher:
            # Preserve arena capacity for K4; compact only the K2/K3 view.
            self.candidates = self._candidate_arena.view(torch.int32)[
                : rows * (candidate_cols + 8)
            ].view(rows, candidate_cols + 8)
        self.recovery_rows = min(
            rows, arena_words // words_per_recovery_row // qtile * qtile
        )
        dense_words = self.recovery_rows * cols * score_bytes // 8
        dense = (
            self._candidate_arena[:dense_words]
            .view(logits_dtype)
            .view(self.recovery_rows, cols)
        )
        self.recovery_candidates = self._candidate_arena[
            dense_words : dense_words + self.recovery_rows * recovery_words
        ].view(self.recovery_rows, recovery_words)
        sample_logits = self._candidate_arena.view(logits_dtype)[
            : rows * (sample_extent + 32)
        ].view(rows, sample_extent + 32)
        self.upper_samples = empty((rows, 288), logits_dtype)
        self.workspace = {
            "arena": self._candidate_arena,
            "samples": self.upper_samples,
            "threshold_storage": self._threshold_storage,
        }
        self.recovery_counters = torch.zeros(
            (rows // qtile,), device=q.device, dtype=torch.int32
        )
        self.recovery_queue = torch.zeros(
            (rows // qtile + 2,), device=q.device, dtype=torch.int32
        )
        kb, qb = k.view(torch.uint8), q.view(torch.uint8)
        safe = math_mode == "relu_fast" and self.qkv_dtype != "mxfp8"
        # MXFP8 QK values include per-32-element scales. FP4 integer-ReLU
        # proofs do not apply; use the general FP32 ReLU path.
        if self.qkv_dtype == "fp8":
            gemm_inputs = (kb, k_scale, qb.reshape(-1, 128), weights.reshape(-1))
            filter_q = gemm_inputs[2]
            if heads == 32 and filter_rows > m:
                filter_q = torch.cat(
                    (
                        filter_q,
                        torch.zeros(
                            ((filter_rows - m) * heads, 128),
                            dtype=filter_q.dtype,
                            device=q.device,
                        ),
                    )
                )
            if m % 2 and heads != 32:
                filter_q = torch.cat((filter_q, torch.zeros_like(filter_q[:heads])))
            filter_inputs = (
                kb,
                k_scale,
                filter_q,
                prepared_weights if safe else weights.reshape(-1),
            )
            # K1 publishes retained samples directly through the unused candidate argument.
            sample_candidates = self.upper_samples
        else:
            # The public adapter validates live fast-path scale bounds per replay.
            sfk = k_scale.view(torch.uint8).reshape(n, 4)
            if n != cols:
                kb = torch.cat(
                    (
                        kb,
                        torch.zeros(
                            (cols - n, width), dtype=torch.uint8, device=q.device
                        ),
                    )
                )
                sfk = torch.cat(
                    (
                        sfk,
                        torch.zeros((cols - n, 4), dtype=torch.uint8, device=q.device),
                    )
                )
            sfq, wp = q_scale.view(torch.int32).reshape(m, heads), weights
            if m != rows:
                qb = torch.cat(
                    (
                        qb,
                        torch.zeros(
                            (rows - m, heads, width), dtype=qb.dtype, device=qb.device
                        ),
                    )
                )
                sfq = torch.cat(
                    (
                        sfq,
                        torch.zeros(
                            (rows - m, heads), dtype=sfq.dtype, device=sfq.device
                        ),
                    )
                )
                wp = torch.cat(
                    (
                        wp,
                        torch.zeros(
                            (rows - m, heads), dtype=wp.dtype, device=wp.device
                        ),
                    )
                )
            gemm_inputs = (
                kb,
                sfk.view(torch.int32).reshape(-1),
                qb.reshape(-1, 128, width).permute(1, 2, 0),
                sfq.reshape(-1, 128).t(),
                wp.reshape(-1, 128).t(),
            )
            filter_inputs, sample_candidates = gemm_inputs, self.upper_samples
            if dsv4 or dsv32_prepare_sf:
                prepared_sf = self._threshold_storage[
                    weight_offset : weight_offset + cols
                ].view(torch.int32)
                filter_inputs = (gemm_inputs[0], prepared_sf, *gemm_inputs[2:])
            if heads in (4, 32) and safe:
                filter_inputs = (
                    *gemm_inputs[:4],
                    prepared_weights.reshape(-1, 128).t(),
                )
            if glm_prepare_sf:
                prepared_sf = self._threshold_storage[
                    weights_end : weights_end + cols
                ].view(torch.int32)
                filter_inputs = (filter_inputs[0], prepared_sf, *filter_inputs[2:])
            if heads == 4:
                filter_inputs = (
                    *gemm_inputs[:2],
                    qb.reshape(-1, 64, width).permute(1, 2, 0),
                    sfq.reshape(-1, 64).t(),
                    (prepared_weights if safe else wp).reshape(-1, 64).t(),
                )

        dtype = DTYPES[logits_dtype]
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        tail = (starts, ends, self._threshold_storage, self.counts)
        scoring_tail = (
            starts,
            ends if _score_ends is None else _score_ends,
            self._threshold_storage,
            self.counts,
        )
        sample_inputs = gemm_inputs
        if self.qkv_dtype in ("mxfp4", "mxfp8"):
            sample_n = 64 if heads == 4 else 256
            sample_qtile = sample_n // heads
            extra = max(
                0, ((m + sample_qtile - 1) // sample_qtile) * sample_qtile - rows
            )
            sq, ssf, sw = qb, sfq, wp
            if extra:
                sq = torch.cat(
                    (
                        qb,
                        torch.zeros(
                            (extra, *qb.shape[1:]), dtype=qb.dtype, device=qb.device
                        ),
                    )
                )
                ssf = torch.cat(
                    (
                        sfq,
                        torch.zeros((extra, heads), dtype=sfq.dtype, device=sfq.device),
                    )
                )
                sw = torch.cat(
                    (wp, torch.zeros((extra, heads), dtype=wp.dtype, device=wp.device))
                )
            sample_inputs = (
                *gemm_inputs[:2],
                sq.reshape(-1, sample_n, width).permute(1, 2, 0),
                ssf.reshape(-1, sample_n).t(),
                sw.reshape(-1, sample_n).t(),
            )
        sample_args = (
            *sample_inputs,
            sample_logits,
            *scoring_tail,
            sample_candidates,
            self.output,
            self.recovery_queue,
            m,
            kb.shape[0],
        )
        self.block_scores = (
            empty((m, cols // 8), logits_dtype)
            if policy.get("publish_blocks")
            else None
        )
        filter_storage = self.block_scores if self.block_scores is not None else dense
        filter_args = (
            *filter_inputs,
            filter_storage,
            *scoring_tail,
            self.candidates,
            self.output,
            m,
            kb.shape[0],
        )
        select_args = (
            self.candidates,
            self.counts,
            starts,
            ends,
            self.output,
            self.thresholds,
            self.upper_samples,
            self.recovery_queue,
        )
        batch_tasks = self.recovery_rows // qtile
        self.single_row_recovery = short_publisher or (
            policy.get("ds41")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and heads == 32
            and top_k == 512
            and self.qkv_dtype == "mxfp4"
            and compression_ratio == 2
            and m >= 8192
            and m % 4 == 0
            and (n <= 65536 or n >= 524288)
            and not policy.get("publish_blocks")
        )
        recovery_inputs = gemm_inputs
        if self.single_row_recovery:
            rq = gemm_inputs[2].permute(2, 0, 1).reshape(-1, 32, width).permute(1, 2, 0)
            rsf = gemm_inputs[3].t().reshape(-1, 32).t()
            rw = gemm_inputs[4].t().reshape(-1, 32).t()
            recovery_inputs = (*gemm_inputs[:2], rq, rsf, rw)
        recovery_args = (
            *recovery_inputs,
            dense,
            *tail,
            self.recovery_candidates,
            self.output,
            self.recovery_counters,
            self.recovery_queue,
            self.upper_samples,
            m,
            kb.shape[0],
        )

        def signature(args, dynamic):
            result = []
            for i, t in enumerate(args[:-2]):
                shape = (-1, *t.shape[1:])
                if self.qkv_dtype in ("mxfp4", "mxfp8") and i in (2, 3, 4):
                    shape = (*t.shape[:-1], -1)
                if i in dynamic:
                    shape = (-1, -1)
                result.append((t.dtype, shape))
            return tuple(result)

        ds41_sample_count = (
            512 if n <= 65536 else 2048 if n <= 131072 or not native_bf16 else 4096
        )
        ds41_sample_sigma = (
            5.0 if policy.get("publish_blocks") and n >= 1048576 else 4.5
        )
        if (
            policy.get("ds41")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and m >= 8192
            and policy.get("shared_kv_queries")
            and not policy.get("publish_blocks")
        ):
            if n > 131072:
                ds41_sample_count, ds41_sample_sigma = 2048, 6.0
            elif n > 65536:
                ds41_sample_count, ds41_sample_sigma = 768, 5.0
        if (
            policy.get("ds41")
            and policy.get("publish_blocks")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and self.qkv_dtype == "mxfp4"
            and heads == 32
            and top_k == 512
            and compression_ratio == 1
            and m >= 8192
            and 256 * top_k < n <= 512 * top_k
        ):
            ds41_sample_count = 2048
        if short_publisher:
            ds41_sample_count, ds41_sample_sigma = 1024, 5.0
        separate_span_loops = short_publisher or bool(
            policy.get("ds41")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and heads == 32
            and top_k == 512
            and self.qkv_dtype == "mxfp4"
            and compression_ratio == 2
            and m >= 8192
            and m % 4 == 0
            and 32768 < n <= 65536
            and not policy.get("publish_blocks")
        )
        short_native_ordinary = separate_span_loops and not short_publisher
        if short_native_ordinary:
            ds41_sample_sigma = 4.75
        short_fp8_samples = (
            1024
            if (
                policy.get("kimi")
                and self.qkv_dtype == "fp8"
                and heads == 32
                and top_k == 2048
                and logits_dtype == torch.float32
                and not (native_bf16 or native_fp16 or safe)
                and m >= 8192
                and 65536 < n <= 131072
            )
            else 0
        )
        async_sample = bool(
            policy.get("ds41")
            and policy.get("shared_kv_queries")
            and not policy.get("publish_blocks")
            and native_bf16
            and self.qkv_dtype == "mxfp4"
            and heads == 32
            and top_k == 512
            and compression_ratio == 2
            and m >= 8192
            and n >= 32768
        )
        sampler = compile_sampler(
            self.qkv_dtype,
            top_k,
            self.capacity,
            dtype,
            sms,
            6 if heads in (4, 32) or dsv4 else 4,
            signature(sample_args, ()),
            safe,
            2,
            rank_sigma=(
                ds41_sample_sigma
                if policy.get("ds41") and self.qkv_dtype == "mxfp4"
                else 5.0
                if m >= 8192
                and heads == 32
                and top_k == 2048
                and self.qkv_dtype == "fp8"
                and logits_dtype == torch.float32
                and 4 * top_k < n <= 8 * top_k
                else 6.0
                if m >= 8192
                and heads == 32
                and top_k == 512
                and self.qkv_dtype == "fp8"
                and logits_dtype == torch.float32
                and 256 * top_k <= n < 512 * top_k
                else 0.0
            ),
            count_override=(
                ds41_sample_count
                if policy.get("ds41") and self.qkv_dtype == "mxfp4"
                else short_fp8_samples
                if short_fp8_samples
                else 512
                if m >= 8192
                and heads == 32
                and top_k == 512
                and self.qkv_dtype == "fp8"
                and logits_dtype == torch.float32
                and 256 * top_k <= n < 512 * top_k
                else 256
                if m >= 8192
                and heads == 32
                and top_k == 512
                and self.qkv_dtype == "fp8"
                and logits_dtype == torch.float32
                and 64 * top_k <= n < 128 * top_k
                else 0
            ),
            glm_short_margin=(
                heads == 32
                and top_k == 512
                and self.qkv_dtype in ("mxfp4", "mxfp8")
                and n < 32 * top_k
            ),
            glm_prepare_sf=prepare_kv_sf,
            fixed_samples=(
                256
                if heads == 4 and n <= 32 * top_k
                else 256
                if dsv4
                and self.qkv_dtype == "fp8"
                and top_k == 512
                and 16 * top_k < n < 32 * top_k
                else 128
                if top_k == 512 and (heads == 32 or dsv4) and n <= 64 * top_k
                else 256
                if dsv4 and top_k == 1024 and n < 32 * top_k
                else 0
            ),
            glm_short_quantile=(
                heads == 32
                and top_k == 2048
                and logits_dtype in (torch.float16, torch.bfloat16)
                and n <= 8 * top_k
            ),
            heads=heads,
            qsa_weights=qsa_weights,
            qsa_unit_scale=qsa_unit_scale,
            short_domain=(
                (heads == 32 and n <= 64 * top_k)
                or (dsv4 and 32 * top_k <= n <= 128 * top_k)
            ),
            dsv4=dsv4,
            pro_common_domain=(
                dsv4 and top_k == 1024 and 16 * top_k <= n <= 128 * top_k
            ),
            pdl=self.pdl,
            exhaustive=exhaustive,
            native_bf16=native_bf16,
            native_fp16=native_fp16,
            cache_native_weights=short_publisher,
            separate_span_loops=separate_span_loops,
            pair_native_queries=(short_native_ordinary or (async_sample and n > 65536)),
            async_gather=async_sample,
            inline_extrema=bool(
                policy.get("ds41")
                and native_bf16
                and self.qkv_dtype == "mxfp4"
                and heads == 32
                and top_k == 512
                and m >= 8192
                and ds41_sample_count >= 256
            ),
        )
        # Shape-only specialization: amortize the extra CTA and eight-bank
        # selector for sufficiently long H32 domains. Fixed during graph capture.
        q8192_glm_dual_kv = (
            m >= 8192
            and heads == 32
            and top_k == 512
            and self.qkv_dtype in ("mxfp4", "mxfp8")
            and 32 * top_k <= n < 64 * top_k
        )
        glm_dual_kv = (
            heads == 32 and top_k == 512 and n >= 64 * top_k
        ) or q8192_glm_dual_kv
        glm_dual_kv = policy.get("kv_pairing", glm_dual_kv)
        glm_kv_parts = (
            2 if heads == 32 and self.qkv_dtype == "fp8" and n >= 32 * top_k else 1
        )
        self.glm_kv_parts = glm_kv_parts
        # Sharing a KV tile across two query pairs amortizes TMA work for
        # at least 512 KV tiles. Require a full persistent query wave and prepared
        # block scales; the small-query and default FP32 paths keep their layout.
        shared_kv_queries = (
            (native_bf16 or native_fp16)
            and heads == 64
            and self.qkv_dtype == "mxfp4"
            and (dsv4 or prepare_kv_sf)
            and m >= 4 * sms
            and n >= 512 * 128
            and torch.cuda.get_device_capability(q.device) == (10, 0)
        )
        shared_kv_queries = policy.get("shared_kv_queries", shared_kv_queries)
        self.shared_kv_queries = shared_kv_queries
        row_readiness = bool(
            self.pdl
            and policy.get("ds41")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and shared_kv_queries
            and heads == 32
            and top_k == 512
            and m >= 8192
            and n <= 1048576
            and not policy.get("publish_blocks")
        )
        # Constant SF avoids the FP16 QK rescaling epilogue for FP8. Complete
        # four-query and 128-key tiles keep this TMA view within live storage.
        shared_fp8_queries = (
            native_fp16
            and heads == 64
            and self.qkv_dtype == "fp8"
            and m >= 4 * sms
            and m % 4 == 0
            and n >= 128 * 128
            and n % 128 == 0
            and torch.cuda.get_device_capability(q.device) == (10, 0)
        )
        self.shared_fp8_queries = shared_fp8_queries
        # Validated ends fit N; aligned N contains every complete 128-key tile.
        # GLM H32 FP8 with FP32 logits: at Q >= 8192 the pair-query epilogue (two queries per thread, four TMEM
        # sub-tiles) serves every n > 4*top_k, aligned or not (unaligned n clamps the K-scale index). 8*top_k < n <
        # 32*top_k (16K-64K actual-length rows) used the default/short-tile epilogue: the pair path is 1.06-1.08x there
        # (B200 GLM-5.3, same TopK output), and capacity-planned callers (n = capacity) already took it. The former
        # long-row variant (default epilogue + integer-ReLU period from 256*top_k) spilled to local memory; B200 GLM-5.3
        # Q8192 per layer: 1M 1.08x, 512K 1.08x, unaligned 1M 1.07x. Without the pair path (Q < 8192) the integer period
        # is dropped for aligned n (no spill: Q4096/Q2048 1.03-1.06x) but kept for unaligned n, where the clamped scale
        # load otherwise lands on the critical path (Q4096 1M: +11% cycles, long-scoreboard stalls on the scale multiply).
        # All epilogues compute identical scores (same per-head FMA chains and head sum); only the publish order differs.
        glm_pair = (
            m >= 8192
            and heads == 32
            and top_k == 2048
            and self.qkv_dtype == "fp8"
            and logits_dtype == torch.float32
            and n > 4 * top_k
        )
        filt = compile_filter(
            self.qkv_dtype,
            top_k,
            self.capacity,
            dtype,
            sms,
            signature(
                filter_args,
                (len(filter_args) - 9,)
                if heads in (4, 32)
                else (len(filter_args) - 9, len(filter_args) - 4),
            ),
            safe,
            8,
            q8192_glm_dual_kv=q8192_glm_dual_kv,
            q8192_pair_queries=glm_pair,
            pair_unaligned=glm_pair and n % 128 != 0,
            q8192_flash_epi=(
                m >= 8192
                and heads == 32
                and top_k == 512
                and self.qkv_dtype == "fp8"
                and logits_dtype == torch.float32
                and n >= 64 * top_k
            ),
            glm_mx_tiny=(
                heads == 32
                and top_k == 2048
                and self.qkv_dtype in ("mxfp4", "mxfp8")
                and n <= 4 * top_k
            ),
            packed_store_enabled=dsv4 and n > 8 * top_k,
            glm_prepare_sf=prepare_kv_sf,
            full_q_blocks=(
                (self.qkv_dtype == "fp8" or heads in (4, 32))
                and m
                % (
                    256 // heads
                    if heads == 64 or (heads == 32 and self.qkv_dtype == "fp8")
                    else 128 // heads
                )
                == 0
            ),
            heads=heads,
            small_head_large_kv=(heads in (4, 32) and n >= 128 * top_k),
            dense_candidates=(heads in (4, 32) and n < 64 * top_k),
            qsa_weights=qsa_weights,
            qsa_unit_scale=qsa_unit_scale,
            offsets32=(heads in (4, 32) and self.candidates.numel() < 2**31),
            glm_kv_parts=glm_kv_parts,
            glm_dual_kv=glm_dual_kv,
            glm_short_tile=(
                heads == 32 and (n < 32 * top_k if top_k == 512 else n <= 16 * top_k)
            ),
            glm_mid_kv=(
                heads == 32
                and self.qkv_dtype in ("mxfp4", "mxfp8")
                and 32 * top_k <= n < 256 * top_k
                and not policy.get("ds41")
            ),
            dsv4=dsv4,
            aligned_k=(
                (
                    dsv4
                    or (
                        heads == 32
                        and top_k == 512
                        and (n < 32 * top_k or n >= 64 * top_k)
                    )
                    or (
                        heads == 32
                        and top_k == 2048
                        and (
                            (
                                logits_dtype in (torch.float16, torch.bfloat16)
                                and n >= 8 * top_k
                            )
                            or (
                                logits_dtype == torch.float32
                                and (n <= 16 * top_k or n >= 64 * top_k)
                            )
                        )
                    )
                )
                and n % 128 == 0
            ),
            dsv32_short=(heads == 64 and top_k == 2048 and n <= 32 * top_k),
            glm_integer_period=32
            if heads == 32
            and top_k == 2048
            and logits_dtype == torch.float32
            and n >= 256 * top_k
            and not glm_pair
            and n % 128 != 0
            else 0,
            pdl=self.pdl,
            native_bf16=native_bf16,
            native_fp16=native_fp16,
            shared_kv_queries=shared_kv_queries,
            shared_fp8_queries=shared_fp8_queries,
            publish_blocks=policy.get("publish_blocks", False),
            row_readiness=row_readiness,
            bank_local_candidates=short_publisher,
            enable_strict_interleave=bool(policy.get("kimi")),
            persistent_grid_multiplier=1 if short_native_ordinary else 0,
            native_broadcast_sum=(short_native_ordinary and shared_kv_queries),
            native_loop_unroll=16
            if (
                policy.get("ds41")
                and not policy.get("publish_blocks")
                and shared_kv_queries
                and native_bf16
                and logits_dtype == torch.bfloat16
                and self.qkv_dtype == "mxfp4"
                and heads == 32
                and top_k == 512
                and compression_ratio == 2
                and m >= 8192
                and n >= 256 * top_k
            )
            else 8,
        )
        # Dtype-specific epilogues; every KV length uses the same device code.
        compile_selection = (
            compile_precounted_selector
            if shared_kv_queries
            or (self.qkv_dtype == "fp8" and glm_kv_parts == 1)
            or (
                heads == 32
                and top_k != 2048
                and self.qkv_dtype in ("mxfp4", "mxfp8")
                and not glm_dual_kv
            )
            else compile_mxfp4_selector
        )
        if (
            heads == 32
            and top_k == 2048
            and self.qkv_dtype in ("mxfp4", "mxfp8")
            and n <= 4 * top_k
        ):
            from .crossing_glmfull_short import compile_selector as compile_selection
        if (
            heads == 32
            and top_k == 512
            and self.qkv_dtype == "fp8"
            and 32 * top_k <= n < 64 * top_k
        ):
            from .crossing_glmflash_mid import compile_selector as compile_selection
        # Fix the selector geometry from allocated capacity, never live candidate counts.
        publisher_narrow = bool(
            policy.get("ds41")
            and policy.get("publish_blocks")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and heads == 32
            and top_k == 512
            and self.qkv_dtype == "mxfp4"
            and m >= 8192
            and 65536 <= n <= 524288
        )
        parallel_blocks = publisher_narrow and n <= 262144 and self.pdl
        capacity_policy = (
            dict(candidate_cols=self.candidates.shape[1])
            if compile_selection is compile_precounted_selector
            else {}
        )
        if (
            compile_selection is compile_precounted_selector
            and policy.get("ds41")
            and not policy.get("publish_blocks")
            and native_bf16
            and logits_dtype == torch.bfloat16
            and self.qkv_dtype == "mxfp4"
            and heads == 32
            and top_k == 512
            and compression_ratio == 2
            and m >= 8192
            and 256 * top_k <= n < 512 * top_k
        ):
            capacity_policy["narrow_cache"] = True
        if row_readiness:
            capacity_policy["row_readiness"] = True
        if short_native_ordinary:
            capacity_policy["copy_full_cache"] = True
        if compile_selection is compile_mxfp4_selector:
            capacity_policy["native_bf16"] = native_bf16
            capacity_policy["bank_local"] = short_publisher
            capacity_policy["narrow_cache"] = publisher_narrow
            capacity_policy["producer_complete"] = parallel_blocks
        # DS-V4-Flash (k=512) keeps the 256-thread K3 at every n: capacity-planned callers (n = 1M) otherwise run the
        # 512-thread wide-cache K3 on short live rows (B200, K buffer = 1M tokens, MXFP4: 64K 1.022x, 128K 1.006x, 1M 1.000x; same output).
        selector = compile_selection(
            top_k,
            dtype,
            15,
            tuple((t.dtype, t.ndim) for t in select_args),
            qtile,
            True,
            **capacity_policy,
            exhaustive=exhaustive,
            pro_short=(dsv4 and top_k == 1024 and n < 32 * top_k),
            compact=(heads == 32 or dsv4),
            glm_short=(
                heads == 32
                and top_k == 2048
                and self.qkv_dtype == "fp8"
                and logits_dtype in (torch.float16, torch.bfloat16)
                and n <= 64 * top_k
            ),
            exact_ties=(
                heads == 32
                and top_k == 2048
                and logits_dtype in (torch.float16, torch.bfloat16)
                and n <= 64 * top_k
            ),
            wide_cache=(
                ((heads in (4, 32) or (dsv4 and top_k != 512)) and n >= 256 * top_k)
                or (
                    dsv4
                    and top_k == 1024
                    and self.qkv_dtype in ("mxfp4", "mxfp8")
                    and 32 * top_k <= n < 128 * top_k
                )
            ),
            **(
                dict(glm_mid_bins=16 * top_k <= n < 32 * top_k)
                if heads == 32
                and top_k == 2048
                and self.qkv_dtype == "fp8"
                and glm_kv_parts == 1
                else {}
            ),
            **(
                dict(
                    cache_twelve=logits_dtype in (torch.float16, torch.bfloat16)
                    and 8 * top_k < n < 32 * top_k
                )
                if dsv4
                and top_k == 1024
                and self.qkv_dtype in ("mxfp4", "mxfp8")
                and not shared_kv_queries
                else {}
            ),
            **(
                dict(boundary_extremes=n >= 64 * top_k)
                if heads == 4 and self.qkv_dtype in ("mxfp4", "mxfp8")
                else {}
            ),
            **(
                dict(
                    candidate_cols=self.candidates.shape[1],
                    offsets32=self.candidates.numel() < 2**31,
                )
                if not shared_kv_queries
                and (
                    (
                        self.qkv_dtype in ("mxfp4", "mxfp8")
                        and (
                            heads == 4
                            or (heads == 32 and (top_k == 2048 or glm_dual_kv))
                        )
                    )
                    or glm_kv_parts == 2
                )
                else {}
            ),
            pdl=self.pdl,
        )
        tuning = dict(
            max_w_in_reg=16, num_epi_subtiles=4, math_registers=208, store_registers=88
        )
        if heads in (4, 32):
            tuning.update(max_w_in_reg=heads, num_epi_subtiles=1)
        if self.qkv_dtype == "fp8":
            tuning["max_umma_pipeline"] = True
        recovery_splits = (
            16
            if self.qkv_dtype == "fp8" or (self.single_row_recovery and n >= 524288)
            else 8
        )
        recovery = _compile_gemm(
            self.qkv_dtype,
            "recover",
            top_k,
            self.capacity,
            False,
            dtype,
            min(
                (
                    2
                    if heads == 32
                    and top_k == 512
                    and self.qkv_dtype == "fp8"
                    and 32 * top_k <= n < 64 * top_k
                    else sms // recovery_splits
                ),
                batch_tasks,
            )
            * recovery_splits,
            recovery_splits,
            512,
            signature(recovery_args, (len(recovery_args) - 12, len(recovery_args) - 7)),
            safe,
            tuple(sorted(tuning.items())),
            heads=heads,
            glm_mid_idle=(
                heads == 32
                and top_k == 512
                and self.qkv_dtype == "fp8"
                and 32 * top_k <= n < 64 * top_k
            ),
            glm_short_recovery=(
                heads == 32
                and top_k == 512
                and self.qkv_dtype in ("mxfp4", "mxfp8")
                and n < 32 * top_k
            ),
            pdl=self.pdl,
            native_bf16=native_bf16,
            native_fp16=native_fp16,
            single_row_tasks=self.single_row_recovery,
        )
        self.stages = [
            (name, KernelLaunch(fn, testing.JitArguments(*args), out))
            for name, fn, args, out in (
                ("K1", sampler, sample_args, sample_logits),
                ("K2", filt, filter_args, dense),
                ("K3", selector, select_args, self.output),
            )
        ]
        if self.block_scores is not None:
            from .candidate_blocks import compile_block_selector

            self.candidate_blocks = empty((m, 2048), torch.int32)
            self.candidate_valid_lens = empty((m,), torch.int32)
            block_args = (
                self.block_scores,
                ends,
                self.candidate_blocks,
                self.candidate_valid_lens,
                starts,
            )
            block_stage = (
                "K3.blocks",
                KernelLaunch(
                    compile_block_selector(
                        dtype, cols // 8, 6 * sms if parallel_blocks else 0
                    ),
                    testing.JitArguments(*block_args),
                    self.candidate_blocks,
                ),
            )
            if parallel_blocks:
                self.stages.insert(2, block_stage)
            else:
                self.stages.append(block_stage)
        self.stages.append(
            ("K4", KernelLaunch(recovery, testing.JitArguments(*recovery_args), dense))
        )
        self._owners = (tensors, gemm_inputs, filter_inputs, sample_candidates, dense)

    def prepare_live_bounds(self, *args):
        if self.candidate_k_block_ids is not None:
            from .sparse.prepare import prepare_live_bounds

            prepare_live_bounds(self, *args)

    def run(self, candidate_k_block_ids=None):
        if candidate_k_block_ids is None:
            for _, stage in self.stages:
                stage.run()
        else:
            bound = self.candidate_k_block_ids
            if bound is None:
                raise ValueError("candidate inputs must be supplied during preparation")
            if (
                candidate_k_block_ids.shape != bound.shape
                or candidate_k_block_ids.dtype != bound.dtype
                or candidate_k_block_ids.device != bound.device
                or not candidate_k_block_ids.is_contiguous()
                or candidate_k_block_ids.data_ptr() % 16
            ):
                raise ValueError(
                    "candidate table must match the prepared shape, dtype, device and alignment"
                )
            for _, stage in self.stages:
                stage.run(candidate_k_block_ids)
        return self.output
