"""Prepare fused routing once, then submit it on the current PyTorch stream."""
import torch
from flashinfer.mega_gate import prepare_mega_gate

x = torch.randn((16, 5120), dtype=torch.bfloat16, device="cuda")
weight = torch.randn((384, 5120), dtype=torch.bfloat16, device="cuda") * 5120**-0.5
bias = torch.randn(384, dtype=torch.float32, device="cuda")
mapping = torch.arange(384, dtype=torch.int32, device="cuda")[:, None]
counts = torch.ones(384, dtype=torch.int32, device="cuda")
plan = prepare_mega_gate(x, weight, bias=bias, to_physical_map=mapping, logical_count=counts)
indices, weights = plan.run()
print(indices.shape, weights.shape)
