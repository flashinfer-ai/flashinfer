# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

import flashinfer
from flashinfer.jit import env as jit_env
from flashinfer.jit.core import gen_jit_spec
from flashinfer.jit.utils import write_if_different
from flashinfer.utils import get_compute_capability, is_sm90a_supported


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] < 8,
    reason="MLA requires SM80 or newer",
)


# Exercise the production state updaters in isolation: loop optimization in a
# complete attention kernel can hide FMA contraction of the history rescale.
# The Hopper updater itself uses only scalar arithmetic and warp shuffles, so
# this unit test also runs on Ampere; full FA3 calls below still require Hopper.
_UPDATER_SOURCE = r"""
#include <flashinfer/attention/mla_hopper.cuh>
#include "tvm_ffi_utils.h"

struct TestAttention { float sm_scale_log2; };
template <bool Sharded> struct TestTraits {
  using DTypeQKAccum = float;
  using AttentionVariant = TestAttention;
  struct SharedStorage { float m_wg[2][64]; };
  static constexpr bool QK_SHARD = Sharded;
  static constexpr int NUM_MMA_KV = 4;
  static constexpr int NUM_MMA_D_CKV = 2;
  static constexpr int NUM_REGS_S_FRAG = 8;
};

template <int Path>
__global__ void UpdateHistory(const float* inputs, float* scales, float* output) {
  using Traits = TestTraits<Path == 1>;
  __shared__ typename Traits::SharedStorage storage;
  TestAttention variant{inputs[2]};
  float m[2] = {inputs[0], inputs[0]}, d[2] = {1.f, 1.f};
  const int tid = threadIdx.x + 32 * (threadIdx.y + 4 * threadIdx.z);
  if constexpr (Path < 2) {
    float scores[Traits::QK_SHARD ? 2 : 4][8];
    for (auto& frag : scores) for (float& score : frag) score = inputs[1];
    float out[1][8];
    for (float& value : out[0]) value = 1.f;
    flashinfer::mla::update_mdo_states_<Traits>(&storage, 0, variant, scores, out, m, d);
    for (int j = 0; j < 2; ++j) scales[(Path * 256 + tid) * 2 + j] = d[j];
    for (int j = 0; j < 8; ++j) output[(Path * 256 + tid) * 8 + j] = out[0][j];
  } else {
    float scores[8], scale[2];
    for (float& score : scores) score = inputs[1];
    flashinfer::mla::hopper::update_md_<Traits>(&storage, variant, scores, m, d, scale);
    for (int j = 0; j < 2; ++j) scales[(Path * 256 + tid) * 2 + j] = scale[j];
  }
}

void RunUpdaters(TensorView inputs, TensorView scales, TensorView output) {
  ffi::CUDADeviceGuard guard(inputs.device().device_id);
  auto stream = get_stream(inputs.device());
  auto in = static_cast<const float*>(inputs.data_ptr());
  auto scale = static_cast<float*>(scales.data_ptr());
  auto out = static_cast<float*>(output.data_ptr());
  UpdateHistory<0><<<1, dim3(32, 4, 2), 0, stream>>>(in, scale, out);
  UpdateHistory<1><<<1, dim3(32, 4, 2), 0, stream>>>(in, scale, out);
  UpdateHistory<2><<<1, dim3(32, 4, 2), 0, stream>>>(in, scale, out);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, RunUpdaters);
"""


@pytest.fixture(scope="module")
def updaters():
    name = "test_mla_softmax_rescale"
    source = jit_env.FLASHINFER_GEN_SRC_DIR / name / "updaters.cu"
    write_if_different(source, _UPDATER_SOURCE)
    return gen_jit_spec(name, [source]).build_and_load()


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "previous,current,scale,expected",
    [
        (335544320.0, 335544320.0, 0.10411754250526428, 1.0),
        (-335544320.0, -335544320.0, 0.10411754250526428, 1.0),
        (-math.inf, -math.inf, 0.25, 0.0),
        (-math.inf, 1.0, 0.25, 0.0),
        (1.0, -math.inf, 0.25, 1.0),
        (0.25, 0.75, 2.0, 0.5),
    ],
    ids=[
        "equal-positive",
        "equal-negative",
        "empty",
        "first-valid",
        "masked-tile",
        "increasing",
    ],
)
def test_mla_history_rescale(updaters, previous, current, scale, expected, graph):
    inputs = torch.tensor(
        [previous, current, scale], dtype=torch.float32, device="cuda"
    )
    scales = torch.empty(3, 256, 2, dtype=torch.float32, device="cuda")
    output = torch.empty(2, 256, 8, dtype=torch.float32, device="cuda")
    updaters.run(inputs, scales, output)
    if graph:
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            updaters.run(inputs, scales, output)
        for _ in range(3):
            scales.fill_(math.nan)
            output.fill_(math.nan)
            captured.replay()
    torch.testing.assert_close(
        scales, torch.full_like(scales, expected), rtol=0, atol=1e-7
    )
    torch.testing.assert_close(
        output, torch.full_like(output, expected), rtol=0, atol=1e-7
    )


def _run_mla(q_nope, q_pe, ckv, kpe, kv_len, backend, causal, graph):
    if backend == "fa3" and not is_sm90a_supported(q_nope.device):
        pytest.skip("FA3 requires Hopper")
    pages = math.ceil(kv_len / ckv.shape[1])
    qi = torch.tensor([0, len(q_nope)], dtype=torch.int32, device=q_nope.device)
    ki = torch.tensor([0, pages], dtype=torch.int32, device=q_nope.device)
    indices = torch.arange(pages, dtype=torch.int32, device=q_nope.device)
    lengths = torch.tensor([kv_len], dtype=torch.int32, device=q_nope.device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q_nope.device),
        backend=backend,
        use_cuda_graph=graph,
        qo_indptr=qi,
        kv_indptr=ki,
        kv_indices=indices,
        kv_len_arr=lengths,
    )
    wrapper.plan(
        metadata=flashinfer.mla.MLAPlanMetadata.csr(qi, ki, indices, lengths),
        num_heads=q_nope.shape[1],
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=ckv.shape[1],
        causal=causal,
        sm_scale=192**-0.5,
        q_data_type=q_nope.dtype,
        kv_data_type=ckv.dtype,
        query_layout="split",
        kv_cache_layout="split",
        lse_mode="base2",
    )
    out, lse = wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv, kpe), return_lse=True)
    if graph:
        torch.cuda.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            out, lse = wrapper.run(
                query=(q_nope, q_pe), kv_cache=(ckv, kpe), return_lse=True
            )
        for _ in range(3):
            captured.replay()
    torch.cuda.synchronize()
    return out, lse


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["dominant", "random", "causal", "empty"])
def test_mla_rescale_public_api(backend, dtype, case):
    if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FA3 requires Hopper")
    torch.manual_seed(8790)
    qlen, kvlen, heads, page = (
        (64, 4096, 16, 16) if case == "dominant" else (17, 257, 16, 16)
    )
    if case == "empty":
        kvlen = 0
    qn = torch.randn(qlen, heads, 512, dtype=dtype, device="cuda") / 3
    qp = torch.randn(qlen, heads, 64, dtype=dtype, device="cuda") / 3
    pages = max(1, math.ceil(kvlen / page))
    ckv = torch.randn(pages, page, 512, dtype=dtype, device="cuda") / 3
    kpe = torch.randn(pages, page, 64, dtype=dtype, device="cuda") / 3
    if case == "dominant":
        qn.zero_()
        qp.fill_(32768)
        ckv.zero_()
        kpe.zero_()
        ckv[0, 0] = 1
        kpe[0, 0] = 160
        reference = torch.ones_like(qn, dtype=torch.float64)
        # All query rows are identical and the first key dominates exactly.
        scores = qp[0, 0].double() @ kpe.flatten(0, 1).double().T * 192**-0.5
        reference_lse = (torch.logsumexp(scores, -1) / math.log(2)).expand(qlen, heads)
    elif kvlen:
        keys = ckv.flatten(0, 1)[:kvlen].double()
        rope_keys = kpe.flatten(0, 1)[:kvlen].double()
        scores = (
            torch.einsum("qhd,kd->hqk", qn.double(), keys)
            + torch.einsum("qhd,kd->hqk", qp.double(), rope_keys)
        ) * 192**-0.5
        if case == "causal":
            mask = torch.arange(kvlen, device="cuda")[None, :] <= (
                kvlen - qlen + torch.arange(qlen, device="cuda")[:, None]
            )
            scores.masked_fill_(~mask, -math.inf)
        reference = torch.einsum("hqk,kd->qhd", torch.softmax(scores, -1), keys)
        reference_lse = (torch.logsumexp(scores, -1) / math.log(2)).T
    for graph in [False, True]:
        out, lse = _run_mla(qn, qp, ckv, kpe, kvlen, backend, case == "causal", graph)
        assert torch.isfinite(out).all()
        if not kvlen:
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            assert torch.isneginf(lse).all()
        else:
            assert torch.isfinite(lse).all()
            torch.testing.assert_close(out.double(), reference, rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(
                lse.double(),
                reference_lse,
                rtol=2e-7 if case == "dominant" else 2e-4,
                atol=2e-4,
            )
