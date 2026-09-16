"""Benchmark the S5000 Simple-STP provider against generic MUSA Triton."""
import argparse
import json
import torch

from flashinfer.mamba.musa_ssu_simple import ssu_one_token_musa_simple
from flashinfer.mamba.musa_ssu_triton import ssu_one_token_musa_triton


def make_inputs(device):
    g = torch.Generator(device=device).manual_seed(17)
    state = torch.randn((2, 64, 64, 128), device=device, dtype=torch.float16, generator=g)
    x = torch.randn((1, 64, 64), device=device, dtype=torch.bfloat16, generator=g)
    dt = torch.randn((1, 64, 1), device=device, dtype=torch.bfloat16, generator=g).expand(1, 64, 64)
    a = -torch.rand((64, 1, 1), device=device, dtype=torch.bfloat16, generator=g).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device=device, dtype=torch.bfloat16, generator=g)
    c = torch.randn_like(b)
    d = torch.randn((64,), device=device, dtype=torch.bfloat16, generator=g)
    slot = torch.zeros((1,), device=device, dtype=torch.int32)
    return state, x, dt, a, b, c, d, slot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    assert torch.version.musa is not None, "run this benchmark on MUSA"
    torch.musa.set_device(0)
    results = {}
    for name, fn in (("simple", ssu_one_token_musa_simple), ("generic", ssu_one_token_musa_triton)):
        state, x, dt, a, b, c, d, slot = make_inputs("musa")
        out = torch.empty_like(x)
        for _ in range(args.warmup):
            fn(state, x, dt, a, b, c, d, slot, dt_softplus=True, out=out)
        torch.musa.synchronize()
        events = []
        for _ in range(args.iters):
            state.zero_()
            torch.musa.synchronize()
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            fn(state, x, dt, a, b, c, d, slot, dt_softplus=True, out=out)
            end.record()
            end.synchronize()
            events.append(start.elapsed_time(end))
        results[name] = {"median_ms": sorted(events)[len(events) // 2], "samples_ms": events}
    print(json.dumps({"shape": [1, 64, 64, 128, 8], "results": results}, indent=2))


if __name__ == "__main__":
    main()
