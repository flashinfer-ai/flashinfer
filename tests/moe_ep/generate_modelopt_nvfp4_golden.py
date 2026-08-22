"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Generate the bundled public-exporter W4A16 NVFP4 checkpoint fixture.

The output is ModelOpt's unmodified ``model.safetensors`` file for a
deterministic tiny Llama. Reproduce in an isolated CPU environment with::

    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0
    python -m pip install nvidia-modelopt[hf]==0.45.0 \
        transformers==5.2.0 safetensors==0.7.0
    python tests/moe_ep/generate_modelopt_nvfp4_golden.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import shutil
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.qtensor import NVFP4QTensor


VERSIONS = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "nvidia-modelopt": "0.45.0",
    "transformers": "5.2.0",
    "safetensors": "0.7.0",
}
MODELOPT_SOURCE_COMMIT = "ec87a82927d003986d44fb7f4fa8b3d10c31b095"
SEED = 3704
TARGET_PREFIX = "model.layers.0.self_attn.q_proj"
EXPECTED_MODEL_SHA256 = (
    "532857e12aa4d70279dcd1bdd2219d184d549844849d45c2222fd7b2ed05f513"
)
EXPECTED_DEQUANT_SHA256 = (
    "255c4393f1ff9a228bef639a018353e2d531459bc1a37694922f0c26318d39d5"
)
DEFAULT_OUTPUT = (
    Path(__file__).with_name("data") / "modelopt_w4a16_nvfp4_v1.safetensors"
)
MAX_FIXTURE_BYTES = 16 * 1024


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().reshape(-1).contiguous().view(torch.uint8)
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def _check_environment() -> None:
    for distribution, expected in VERSIONS.items():
        actual = importlib.metadata.version(distribution)
        if _base_version(actual) != expected:
            raise RuntimeError(f"expected {distribution}=={expected}, got {actual}")
    if torch.__version__ != "2.8.0+cpu":
        raise RuntimeError(f"expected torch==2.8.0+cpu, got {torch.__version__}")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("generation must use the official CPU-only PyTorch wheel")


def _build_model() -> LlamaForCausalLM:
    torch.manual_seed(SEED)
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        architectures=["LlamaForCausalLM"],
    )
    config.dtype = torch.bfloat16
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16, device="cpu")
    with torch.no_grad():
        for index, (_, parameter) in enumerate(model.named_parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.int64)
            values = ((values + 7 * index) % 43 - 21).to(torch.float32) / 16.0
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))
    return model.eval()


def _export_fixture(output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="flashinfer-modelopt-w4a16-") as temp:
        export_dir = Path(temp)
        model = _build_model()
        mtq.quantize(model, mtq.W4A16_NVFP4_CFG)
        export_hf_checkpoint(model, dtype=torch.bfloat16, export_dir=export_dir)

        model_path = export_dir / "model.safetensors"
        if _sha256_file(model_path) != EXPECTED_MODEL_SHA256:
            raise RuntimeError("public ModelOpt export differs from the pinned fixture")
        with safe_open(model_path, framework="pt", device="cpu") as checkpoint:
            packed = checkpoint.get_tensor(f"{TARGET_PREFIX}.weight").contiguous()
            scales = checkpoint.get_tensor(f"{TARGET_PREFIX}.weight_scale").contiguous()
            alpha = checkpoint.get_tensor(
                f"{TARGET_PREFIX}.weight_scale_2"
            ).contiguous()
        if packed.dtype != torch.uint8 or tuple(packed.shape) != (16, 8):
            raise RuntimeError("unexpected exported W4A16 NVFP4 payload contract")
        if scales.dtype != torch.float8_e4m3fn or tuple(scales.shape) != (16, 1):
            raise RuntimeError("unexpected exported W4A16 NVFP4 scale contract")
        if alpha.dtype != torch.float32 or tuple(alpha.shape) != ():
            raise RuntimeError("unexpected exported W4A16 NVFP4 global scale contract")

        qtensor = NVFP4QTensor((16, 16), torch.float32, packed)
        expected = qtensor.dequantize(
            dtype=torch.float32,
            scale=scales,
            double_scale=alpha,
            block_sizes={-1: 16},
        ).contiguous()
        if _sha256_tensor(expected) != EXPECTED_DEQUANT_SHA256:
            raise RuntimeError(
                "official ModelOpt dequantization differs from the golden"
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model_path, output)
    if output.stat().st_size > MAX_FIXTURE_BYTES:
        raise RuntimeError(
            f"fixture is {output.stat().st_size} bytes; maximum is {MAX_FIXTURE_BYTES}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    _check_environment()
    _export_fixture(args.output)
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes, "
        f"sha256={_sha256_file(args.output)})"
    )
    print(f"modelopt_source_commit={MODELOPT_SOURCE_COMMIT}")
    print(f"official_dequant_sha256={EXPECTED_DEQUANT_SHA256}")


if __name__ == "__main__":
    main()
