#!/usr/bin/env python3
"""Run a JIT-disabled GPU smoke test against an installed provider wheel."""

from __future__ import annotations

import argparse
import json
import os
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    return parser.parse_args()


def provider_capability(provider: str) -> tuple[int, int]:
    match = re.fullmatch(r"sm(\d{1,2})(\d)[af]?", provider.lower())
    if match is None:
        raise ValueError(f"Invalid provider ID: {provider}")
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    args = parse_args()
    os.environ.setdefault("FLASHINFER_DISABLE_JIT", "1")

    import torch

    capability = torch.cuda.get_device_capability()
    expected_capability = provider_capability(args.provider)
    if capability != expected_capability:
        raise RuntimeError(
            f"Visible GPU capability {capability} does not match "
            f"{args.provider} ({expected_capability})"
        )

    import flashinfer
    from flashinfer.jit import env as jit_env

    provider_path = jit_env.get_aot_path("silu_and_mul")
    expected_path_part = f"providers/{args.provider}/jit_cache"
    if expected_path_part not in provider_path.as_posix():
        raise RuntimeError(
            f"silu_and_mul resolved outside {args.provider}: {provider_path}"
        )

    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.float16)
    output = flashinfer.silu_and_mul(input_tensor)
    expected = torch.nn.functional.silu(input_tensor[:, :256].float())
    expected *= input_tensor[:, 256:].float()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "capability": capability,
                "provider_path": str(provider_path),
                "output_shape": list(output.shape),
                "max_abs_error": float((output.float() - expected).abs().max()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
