"""Run one prepared MoE example on a catalogued 384-expert model route (SM100a 148 SMs or SM103a 152 SMs; about 48 GiB free device memory)."""

import argparse
import torch
from mega_moe_inputs import make_model, make_grouped_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=("source", "v3", "grouped-l2"), default="source"
    )
    parser.add_argument("--precision", choices=("fp4", "fp8"), default="fp4")
    parser.add_argument("--tokens", type=int, default=16, help="catalogued model token count")
    args = parser.parse_args()
    if args.family == "grouped-l2":
        if args.precision != "fp4":
            parser.error("Grouped L2 uses packed FP4 weights")
        plan = make_grouped_l2()[0]
    else:
        plan = make_model(args.family, args.precision, args.tokens)[0]
    output = plan.run()
    torch.cuda.synchronize()
    print(f"Output: {tuple(output.shape)}, {output.dtype}")


if __name__ == "__main__":
    main()
