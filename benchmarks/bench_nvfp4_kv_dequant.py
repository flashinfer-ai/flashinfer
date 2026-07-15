"""Benchmark NVFP4 KV-cache dequantization (dense + paged).

Reports effective DRAM bandwidth (GB/s) on large, bandwidth-bound shapes for both the dense
(`nvfp4_kv_dequantize`) and paged (`nvfp4_kv_dequantize_paged`) kernels. Runs on any CUDA GPU.

    python benchmarks/bench_nvfp4_kv_dequant.py
"""

import torch

import flashinfer

NVFP4_BLOCK_SIZE = 16


def _time_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def bench_dense():
    print("\n== dense nvfp4_kv_dequantize ==")
    print(f"{'shape (M x K)':>18}  {'dtype':>8}  {'time (ms)':>10}  {'GB/s':>8}")
    for (M, K) in [(65536, 2048), (131072, 2048), (65536, 4096), (262144, 1024)]:
        for dtype in (torch.bfloat16, torch.float16):
            fp4 = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
            scales = torch.randint(1, 120, (M, K // NVFP4_BLOCK_SIZE), dtype=torch.uint8, device="cuda")
            gscale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
            fn = lambda: flashinfer.nvfp4_kv_dequantize(fp4, scales, gscale, output_dtype=dtype)
            ms = _time_ms(fn)
            # read fp4 (K/2) + scales (K/16) + write out (2*K) bytes per row
            gbps = M * (K // 2 + K // NVFP4_BLOCK_SIZE + 2 * K) / (ms * 1e-3) / 1e9
            print(f"{f'{M} x {K}':>18}  {str(dtype).split('.')[-1]:>8}  {ms:>10.4f}  {gbps:>8.1f}")


def bench_paged():
    print("\n== paged nvfp4_kv_dequantize_paged (K + V) ==")
    print(f"{'B x S x H x d':>20}  {'dtype':>8}  {'time (ms)':>10}  {'GB/s':>8}")
    for (B, S, NH, HD, PS) in [(256, 512, 8, 128, 16), (128, 1024, 8, 128, 16), (256, 512, 16, 128, 16)]:
        for dtype in (torch.bfloat16, torch.float16):
            pages_per_seq = (S + PS - 1) // PS
            NP = B * pages_per_seq
            k = torch.randint(0, 256, (NP, PS, NH, HD // 2), dtype=torch.uint8, device="cuda")
            v = torch.randint(0, 256, (NP, PS, NH, HD // 2), dtype=torch.uint8, device="cuda")
            ks = torch.randint(1, 120, (NP, PS, NH, HD // NVFP4_BLOCK_SIZE), dtype=torch.uint8,
                               device="cuda").view(torch.float8_e4m3fn)
            vs = torch.randint(1, 120, (NP, PS, NH, HD // NVFP4_BLOCK_SIZE), dtype=torch.uint8,
                               device="cuda").view(torch.float8_e4m3fn)
            bt = torch.arange(NP, dtype=torch.int32, device="cuda").reshape(B, pages_per_seq)
            sl = torch.full((B,), S, dtype=torch.int32, device="cuda")
            ks_v = torch.tensor([0.5], dtype=torch.float32, device="cuda")
            vs_v = torch.tensor([0.25], dtype=torch.float32, device="cuda")
            ok = torch.empty((B, S, NH, HD), dtype=dtype, device="cuda")
            ov = torch.empty((B, S, NH, HD), dtype=dtype, device="cuda")
            fn = lambda: flashinfer.nvfp4_kv_dequantize_paged(
                (k, v), (ks, vs), bt, sl, ks_v, vs_v, ok, ov, kv_layout="NHD")
            ms = _time_ms(fn)
            # K + V: 2 * [ read (HD/2 + HD/16) + write 2*HD ] per (valid row)
            per_row = 2 * (HD // 2 + HD // NVFP4_BLOCK_SIZE + 2 * HD)
            gbps = B * S * NH * per_row / (ms * 1e-3) / 1e9
            print(f"{f'{B}x{S}x{NH}x{HD}':>20}  {str(dtype).split('.')[-1]:>8}  {ms:>10.4f}  {gbps:>8.1f}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA GPU required"
    print(f"device: {torch.cuda.get_device_name(0)}  torch: {torch.__version__}")
    bench_dense()
    bench_paged()
