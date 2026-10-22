"""SM120 materialized FP8 attention: masks, block scales, and graph replay.

Local research integration depends on the matching SGLang CuTe checkout. This
test imports only the experimental source directory, leaving installed
FlashInfer untouched. It can also run directly via unittest.
"""

import pathlib
import sys
import unittest

import torch

SOURCE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "flashinfer/experimental/mla_fp8_sm120"
)
sys.path.insert(0, str(SOURCE))

import materialized_fp8_interface as candidate
from materialized_fp8_quantize import prepare


def reference(q, k, v, qlens, klens):
    outputs = []
    qb = kb = 0
    for nq, nk in zip(qlens, klens):
        a, b, c = (
            q[qb : qb + nq].float().transpose(0, 1),
            k[kb : kb + nk].float().transpose(0, 1),
            v[kb : kb + nk].float().transpose(0, 1),
        )
        scores = a @ b.transpose(1, 2) / 16
        qi = torch.arange(nq, device=q.device)[:, None]
        ki = torch.arange(nk, device=q.device)[None, :]
        allowed = ki <= qi + nk - nq
        scores.masked_fill_(~allowed[None], -torch.inf)
        probs = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
        outputs.append((probs @ c).transpose(0, 1))
        qb += nq
        kb += nk
    return torch.cat(outputs)


def inputs(qlens, klens):
    torch.manual_seed(119)
    q = torch.randn(sum(qlens), 20, 256, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(sum(klens), 20, 448, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(sum(klens), 1, 64, device="cuda", dtype=torch.bfloat16)
    v = kv[..., 192:]
    # Adjoining KV blocks have deliberately different scales.
    v[:64] *= 0.001
    v[64:128] *= 30
    v[257:321] *= 0.07
    cq = torch.tensor(
        [0] + list(torch.tensor(qlens).cumsum(0)), device="cuda", dtype=torch.int32
    )
    ck = torch.tensor(
        [0] + list(torch.tensor(klens).cumsum(0)), device="cuda", dtype=torch.int32
    )
    k = torch.cat((kv[..., :192], rope.expand(-1, 20, -1)), dim=-1)
    packed = prepare(q, kv[..., :192], rope, v, cq, ck, max(qlens), max(klens), 64, 64)
    kwargs = dict(
        cu_seqlens_q=cq,
        cu_seqlens_k=ck,
        max_seqlen_q=max(qlens),
        max_seqlen_k=max(klens),
        softmax_scale=1 / 16,
        causal=True,
        tile_mn=(64, 64),
    )
    return q, k, v, packed, kwargs


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMaterializedPrefill(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("Validated on SM120 only")

    def test_ragged_prefixes_and_empty_causal_rows(self):
        for qlens, klens in [([129, 77, 64], [257, 211, 64]), ([97, 1], [33, 65])]:
            with self.subTest(qlens=qlens, klens=klens):
                q, k, v, packed, kw = inputs(qlens, klens)
                q8, k8, v8, scales = packed
                out = candidate._flash_attn_fwd(q8, k8, v8, aux_tensors=scales, **kw)[0]
                ref = reference(q, k, v, qlens, klens)
                self.assertTrue(torch.isfinite(out).all())
                self.assertLess(float((out.float() - ref).norm() / ref.norm()), 0.08)
                if qlens[0] > klens[0]:
                    self.assertEqual(
                        torch.count_nonzero(out[: qlens[0] - klens[0]]).item(), 0
                    )

    def test_graph_replay_reads_current_tensors_and_scales(self):
        _, _, _, packed, kw = inputs([129, 77, 64], [257, 211, 64])
        q8, k8, v8, scales = packed

        def run():
            return candidate._flash_attn_fwd(q8, k8, v8, aux_tensors=scales, **kw)[0]

        initial = run()
        self.assertTrue(torch.equal(initial, run()))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        q8.zero_()
        scales[2].mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(captured, run()))
        self.assertFalse(torch.equal(captured, initial))
        v8.zero_()
        self.assertEqual(torch.count_nonzero(run()).item(), 0)


if __name__ == "__main__":
    unittest.main()
