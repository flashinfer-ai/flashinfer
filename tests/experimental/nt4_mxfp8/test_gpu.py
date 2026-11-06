"""GB300 numerical, poisoned-slot and changed-route graph regression checks."""

import unittest

import torch


def quantized_values(x):
    """Independent per-32 E4M3/E8M0 reference, returning dequantized FP32."""
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448)).clamp(-127, 127)
    scale = torch.pow(2.0, exponent)
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


@unittest.skipUnless(torch.cuda.is_available(), "requires GB300 / SM103")
class NativeNt4Test(unittest.TestCase):
    def setUp(self):
        self.original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.original_tf32

    def test_reference_poison_and_changed_route_graph(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("requires GB300 / SM103")
        from flashinfer.fused_moe.nt4_mxfp8 import (
            pack_nt4_mxfp8_weights,
            prepare_nt4_mxfp8_local_experts,
        )

        torch.manual_seed(17)
        # Two live experts; all 128 slots retain the production weight layout.
        w1 = torch.zeros((128, 4096, 768), dtype=torch.float8_e4m3fn, device="cuda")
        w2 = torch.zeros((128, 768, 4096), dtype=torch.float8_e4m3fn, device="cuda")
        w1[:2].copy_((torch.randn((2, 4096, 768), device="cuda") / 32).to(w1.dtype))
        w2[:2].copy_((torch.randn((2, 768, 4096), device="cuda") / 64).to(w2.dtype))
        s1 = torch.full((128, 4096, 24), 127, dtype=torch.uint8, device="cuda")
        s2 = torch.full((128, 768, 128), 127, dtype=torch.uint8, device="cuda")
        # Exercise blocked-scale packing with nonuniform scales and expert gaps.
        w1[127].copy_(w1[1])
        w1[1].zero_()
        w2[127].copy_(w2[1])
        w2[1].zero_()
        s1.copy_((126 + torch.arange(24, device="cuda") % 3).to(torch.uint8))
        s2.copy_((126 + torch.arange(128, device="cuda") % 3).to(torch.uint8))
        weights = pack_nt4_mxfp8_weights(w1, s1, w2, s2)
        x = torch.randn((9, 768), device="cuda", dtype=torch.bfloat16)
        ids = torch.full((9, 10), -1, device="cuda", dtype=torch.int64)
        ids[:, 0], ids[:, 1] = 256, 383
        probs = torch.rand((9, 10), device="cuda")
        out = torch.empty((9, 768), device="cuda")
        plan = prepare_nt4_mxfp8_local_experts(
            weights, max_tokens=9, local_expert_offset=256
        )
        expected = torch.zeros_like(out)
        reference_terms = []

        def reference(x, expert):
            a = w1[expert].float() * torch.pow(
                2.0, s1[expert].float() - 127
            ).repeat_interleave(32, -1)
            b = w2[expert].float() * torch.pow(
                2.0, s2[expert].float() - 127
            ).repeat_interleave(32, -1)
            z = quantized_values(x) @ a.T
            activation = (16 * torch.tanh(z.clamp_min(0) / 16)).square()
            return (quantized_values(activation) @ b.T).bfloat16().float()

        for slot, expert in enumerate((0, 127)):
            terms = reference(x, expert)
            reference_terms.append(terms)
            expected = expected + terms * probs[:, slot, None]
        plan.run(x, ids, probs, out=out)
        error = (out - expected).norm() / expected.norm()
        row_error = (out - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(
            1e-6
        )
        self.assertLessEqual(error.item(), 0.015)
        self.assertLessEqual(row_error.max().item(), 0.015)
        for slot, terms in enumerate(reference_terms):
            local_error = (
                plan.contribution_view()[:, slot].float() - terms
            ).norm() / terms.norm()
            self.assertLessEqual(local_error.item(), 0.01)
        eager = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run(x, ids, probs, out=out)
        graph.replay()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
        other = prepare_nt4_mxfp8_local_experts(
            weights, max_tokens=1, local_expert_offset=256
        )
        other.run(x[:1], ids[:1], probs[:1], out=torch.empty((1, 768), device="cuda"))
        graph.replay()
        torch.testing.assert_close(out, eager, rtol=0, atol=0)
        plan.contribution_view().fill_(float("nan"))
        ids.fill_(-1)
        probs.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
        probs.uniform_()
        ids[:, 0], ids[:, 1] = 383, 256
        x.mul_(0.5)
        plan.run(x, ids, probs, out=out)
        changed = out.clone()
        for _ in range(5):
            graph.replay()
            torch.testing.assert_close(out, changed, rtol=0, atol=0)
        # Multiple lookup groups, >128 tokens in one expert, all local experts,
        # and remote routes. Poisoning catches missing contribution writes.
        dense = prepare_nt4_mxfp8_local_experts(
            weights, max_tokens=129, local_expert_offset=256
        )
        dx = torch.randn((129, 768), device="cuda", dtype=torch.bfloat16)
        di = torch.arange(1290, device="cuda").reshape(129, 10) % 128 + 256
        di[:, 0] = 256
        di[:, -1] = 511  # Remote expert must not contribute.
        dp = torch.rand((129, 10), device="cuda")
        do = torch.empty((129, 768), device="cuda")
        ref = torch.zeros_like(do)
        live = {expert: reference(dx, expert) for expert in (0, 127)}
        for slot in range(10):
            for expert, terms in live.items():
                ref += (
                    terms
                    * torch.where(di[:, slot] == 256 + expert, dp[:, slot], 0)[:, None]
                )
        dense.contribution_view().fill_(float("nan"))
        dense.run(dx, di, dp, out=do)
        self.assertTrue(torch.isfinite(do).all().item())
        self.assertLessEqual(((do - ref).norm() / ref.norm()).item(), 0.015)
        torch.cuda.synchronize()
        dense.close()
        graph.reset()
        plan.close()
        other.close()


if __name__ == "__main__":
    unittest.main()
