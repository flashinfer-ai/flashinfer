"""Run one prepared MoE example; minimum hardware SM103a with 152 SMs."""
import argparse
import torch
from mega_moe_inputs import make_smoke, make_grouped_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("source", "v3", "grouped-l2"), default="source")
    parser.add_argument("--precision", choices=("fp4", "fp8"), default="fp4")
    args = parser.parse_args()
    if args.family == "grouped-l2":
        if args.precision != "fp4":
            parser.error("Grouped L2 uses packed FP4 weights")
        plan = make_grouped_l2()[0]
    else:
        plan = make_smoke(args.family, args.precision)[0]
    output = plan.run()
    torch.cuda.synchronize()
    print(f"Output: {tuple(output.shape)}, {output.dtype}")


if __name__ == "__main__":
    main()
