"""Run the explicit experimental fused mHC API on a supported SM100a/SM103a GPU."""

import torch
from flashinfer.mega_mhc import prepare_mega_mhc


def main():
    tokens, hidden = 64, 5120
    values = dict(
        x=torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda"),
        residual=torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device="cuda"),
        post_mix=torch.full((tokens, 4, 1), 0.5, device="cuda"),
        comb_res_mix=torch.eye(4, device="cuda").expand(tokens, 4, 4).contiguous(),
        shifted_prev_mix=torch.full((tokens, 4, 1), 0.25, device="cuda"),
        fn=torch.randn(24, 4 * hidden, device="cuda") * 0.01,
        mix_scales=torch.full((3,), 0.1, device="cuda"),
        mix_bases=torch.zeros(24, device="cuda"),
        rmsnorm_weight=torch.ones(hidden, dtype=torch.bfloat16, device="cuda"),
    )
    plan = prepare_mega_mhc(**values, sf_layout="col")
    outputs = plan.run()
    torch.cuda.synchronize()
    print(
        {
            name: (tuple(value.shape), str(value.dtype))
            for name, value in outputs.items()
            if isinstance(value, torch.Tensor)
        }
    )


if __name__ == "__main__":
    main()
