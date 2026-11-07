#!/usr/bin/env python3
"""Run a JIT-disabled GPU smoke test against installed provider wheels."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("FLASHINFER_DISABLE_JIT", "1")

    import torch

    import flashinfer
    from flashinfer.jit import env as jit_env

    installed_providers = {
        provider.provider_id: provider for provider in jit_env.FLASHINFER_AOT_PROVIDERS
    }
    artifacts = {
        artifact.provider_id: artifact
        for artifact in jit_env.get_aot_artifacts("silu_and_mul")
    }

    for provider_id in args.provider:
        installed_provider = installed_providers.get(provider_id)
        if installed_provider is None:
            raise RuntimeError(f"Provider {provider_id} is not installed")
        compatible_devices = [
            (device_index, jit_env._cuda_architecture_for_device(device_index))
            for device_index in range(torch.cuda.device_count())
            if jit_env._provider_compatibility_score(
                installed_provider.cuda_architectures,
                frozenset({jit_env._cuda_architecture_for_device(device_index)}),
            )
            is not None
        ]
        if not compatible_devices:
            raise RuntimeError(
                f"Provider targets {sorted(installed_provider.cuda_architectures)} "
                "are not compatible with any visible CUDA device"
            )
        device_index, device_architecture = compatible_devices[0]
        torch.cuda.set_device(device_index)
        capability = torch.cuda.get_device_capability(device_index)

        artifact = artifacts.get(provider_id)
        if artifact is None:
            raise RuntimeError(f"silu_and_mul did not resolve from {provider_id}")
        provider_path = artifact.path
        expected_path_part = f"providers/{provider_id}/jit_cache"
        if expected_path_part not in provider_path.as_posix():
            raise RuntimeError(
                f"silu_and_mul resolved outside {provider_id}: {provider_path}"
            )

        input_tensor = torch.randn(
            4, 512, device=f"cuda:{device_index}", dtype=torch.float16
        )
        output = flashinfer.silu_and_mul(input_tensor)
        expected = torch.nn.functional.silu(input_tensor[:, :256].float())
        expected *= input_tensor[:, 256:].float()
        torch.cuda.synchronize(device_index)
        torch.testing.assert_close(output.float(), expected, rtol=1e-2, atol=1e-2)

        print(
            json.dumps(
                {
                    "provider": provider_id,
                    "device": torch.cuda.get_device_name(device_index),
                    "device_index": device_index,
                    "device_architecture": device_architecture,
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
