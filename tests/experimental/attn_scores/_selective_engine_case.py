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
"""Build fixed kernel cases through the production engine allocator and launches.

Public capacity planning is tested separately. This helper only translates test
prefixes into allocation geometry and warms the resulting production chain;
it has no in-place replan, option switches or alternate selection algorithm.
"""

from contextlib import contextmanager

import torch

from flashinfer.experimental.attn_scores import selective_logits as sl


@contextmanager
def _capture_cuda_graph(graph):
    """Caller-owned graph used only by the kernel test driver."""
    with torch.cuda.graph(graph):
        yield


class SelectiveEngineCase(sl._FP8PagedMQASelectiveTopKEngine):
    """A fixed input case, including an optional bounded persistent CTA count."""

    def plan(
        self,
        q,
        kv_fused,
        weights,
        cu_q,
        cu_kv,
        block_table,
        *,
        top_k=512,
        num_heads=8,
        head_dim=128,
        num_sms=None,
        compression_ratio=1,
    ):
        self._check_math_mode()
        sl._validate_selective_top_k(top_k)
        sl._validate_prefixes(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        query, original = cu_q.tolist(), cu_kv.tolist()
        physical, ends = [0], []
        for qb, qe, kb, ke in zip(
            query[:-1], query[1:], original[:-1], original[1:], strict=True
        ):
            physical.append(physical[-1] + (ke - kb) // compression_ratio)
            ends.extend(
                (ke - kb - (qe - qb) + row + 1) // compression_ratio
                for row in range(qe - qb)
            )
        self._allocate_selective_plan(
            q,
            kv_fused,
            weights,
            cu_q,
            cu_kv,
            block_table,
            q_prefix=query,
            kv_prefix=physical,
            logical_ends_host=ends,
            top_k=top_k,
            num_heads=num_heads,
            head_dim=head_dim,
            num_sms=num_sms,
            compression_ratio=compression_ratio,
        )
        self._prepare_repair_if(q.device)
        self._run_impl(q, kv_fused, weights, block_table)
        self._planned = True

    def run(self, q, kv_fused, weights, block_table):
        self._check_math_mode()
        if not getattr(self, "_planned", False):
            raise RuntimeError("plan must be called before run")
        return self._run_impl(q, kv_fused, weights, block_table)
