import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def normal_distribution(std):
    def normal_noise(shape, device):
        return torch.randn(shape, device=device) * std

    normal_noise.__name__ = f"normal_distribution(std={std})"
    return normal_noise


def gumbel_distribution(beta):
    def gumbel_noise(shape, device):
        U = torch.rand(shape, device=device)
        eps = 1e-20
        return torch.log(-torch.log(U + eps) + eps) / beta

    gumbel_noise.__name__ = f"gumbel_distribution(beta={beta})"
    return gumbel_noise


@torch.inference_mode()
def main():
    torch.manual_seed(42)
    print("---")
    print("top-p renorm")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for p in [0.1, 0.5, 0.9]:
                    logits = distrib((batch_size, vocab_size), device="cuda")
                    probs = torch.softmax(logits, dim=-1)
                    measurements = bench_gpu_time(
                        lambda: flashinfer.sampling.top_p_renorm_probs(probs, p),
                        dry_run_time_ms=100,
                        repeat_time_ms=1000,
                    )
                    ms = np.median(measurements)

                    io = (probs.numel() * probs.element_size()) * 2
                    bandwidth = io * 1e-6 / ms
                    print(
                        f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, p: {p}, duration: {ms*1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                    )

    print("---")
    print("top-k renorm")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for k in [10, 100, 1000, 5000]:
                    logits = distrib((batch_size, vocab_size), device="cuda")
                    probs = torch.softmax(logits, dim=-1)
                    measurements = bench_gpu_time(
                        lambda: flashinfer.sampling.top_k_renorm_probs(probs, k),
                        dry_run_time_ms=100,
                        repeat_time_ms=1000,
                    )
                    ms = np.median(measurements)

                    io = (probs.numel() * probs.element_size()) * 2
                    bandwidth = io * 1e-6 / ms
                    print(
                        f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, k: {k}, duration: {ms*1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                    )

    print("---")
    print("top-k mask logits")
    for vocab_size in [128512]:
        for batch_size in [1, 16, 32, 64, 128, 256, 512]:
            for distrib in [
                normal_distribution(1),
                normal_distribution(5),
                gumbel_distribution(0.1),
                gumbel_distribution(1),
            ]:
                for k in [10, 100, 1000, 5000]:
                    logits = distrib((batch_size, vocab_size), device="cuda")
                    measurements = bench_gpu_time(
                        lambda: flashinfer.sampling.top_k_mask_logits(logits, k),
                        dry_run_time_ms=100,
                        repeat_time_ms=1000,
                    )
                    ms = np.median(measurements)

                    io = (logits.numel() * logits.element_size()) * 2
                    bandwidth = io * 1e-6 / ms
                    print(
                        f"vocab_size: {vocab_size}, batch_size: {batch_size}, distrib: {distrib.__name__}, k: {k}, duration: {ms*1e3:.2f} us, effective bandwidth: {bandwidth:.2f} GB/s"
                    )


if __name__ == "__main__":
    main()
