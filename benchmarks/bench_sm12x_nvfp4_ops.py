# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare the native SM12x NVFP4 op chain with the pinned W4A4 baseline."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


def prepare_native_dir():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--native-dir", type=Path)
    args, _ = early.parse_known_args()
    if args.native_dir is None:
        return
    if any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules):
        raise RuntimeError(
            "native capture must be configured before importing Cutlass DSL"
        )
    root = args.native_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("--native-dir must be fresh and empty")
    for name in ("dsl", "triton", "native"):
        (root / name).mkdir()
    os.environ.update(
        CUTE_DSL_KEEP="cubin",
        CUTE_DSL_NO_CACHE="1",
        CUTE_DSL_DUMP_DIR=str(root / "dsl"),
        TRITON_CACHE_DIR=str(root / "triton"),
    )


if __name__ == "__main__":
    prepare_native_dir()

import torch
import torch.nn.functional as F

from flashinfer.tllm_enums import (
    DEFAULT_SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA,
    ActivationType,
)


CASES = {
    "dsv4": (4096, 2048, 256, 6, True),
    "flash_next": (2560, 640, 512, 10, True),
    "qwen30b": (2048, 768, 128, 8, True),
    "nemotron": (2688, 1856, 128, 6, False),
}
TOKEN_COUNTS = {
    "dsv4": (64, 4096),
    "flash_next": (1, 16, 64, 256, 1024, 2048, 4096, 8192),
    "qwen30b": (1, 16, 64, 256, 1024, 2048, 4096, 8192),
    "nemotron": (16, 1024, 8192),
}
BASELINE_SHA256 = "b91b043293176c7cb1e8833c1a86a0197f2d81416071fcbbb34b6b019812127f"
NATIVE_PACKAGE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    return hashlib.sha256(
        value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def loaded_source_identity():
    paths = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if (
            (name == "flashinfer" or name.startswith("flashinfer."))
            and source
            and Path(source).suffix == ".py"
        ):
            paths.add(Path(source).resolve())
    return {str(path): digest(path) for path in sorted(paths)}


def cubin_bytes(compiled):
    binary = compiled if isinstance(compiled, bytes) else compiled.__cubin__
    if isinstance(binary, str):
        binary = Path(binary).read_bytes()
    assert isinstance(binary, bytes) and binary.startswith(b"\x7fELF")
    return binary


def export_native(compiled, destination, *, warp_specialized):
    binary = cubin_bytes(compiled)
    with destination.open("xb") as stream:
        stream.write(binary)
    resources = subprocess.check_output(
        ["cuobjdump", "-res-usage", str(destination)], text=True
    )
    sass = subprocess.check_output(["cuobjdump", "-sass", str(destination)], text=True)
    for suffix, value in ((".resources", resources), (".sass", sass)):
        with destination.with_suffix(suffix).open("x") as stream:
            stream.write(value)
    fields = {
        name: [int(value) for value in re.findall(name + r":(\d+)", resources)]
        for name in ("REG", "STACK", "LOCAL", "SHARED")
    }
    assert fields["REG"] and all(
        len(fields[name]) == len(fields["REG"]) for name in ("STACK", "LOCAL")
    ), resources
    opcodes = {
        name: len(re.findall(pattern, sass))
        for name, pattern in (
            ("setmaxreg", r"U?SETMAXREG"),
            ("local_ld", r"U?LDL[.\s]"),
            ("local_st", r"U?STL[.\s]"),
        )
    }
    assert all(value == 0 for name in ("STACK", "LOCAL") for value in fields[name]), (
        fields
    )
    assert opcodes["local_ld"] == opcodes["local_st"] == 0, opcodes
    if warp_specialized:
        assert opcodes["setmaxreg"] > 0, opcodes
    return dict(
        cubin=str(destination),
        sha256=digest(destination),
        fields=fields,
        opcodes=opcodes,
        warp_specialized=warp_specialized,
        resources_sha256=digest(destination.with_suffix(".resources")),
        sass_sha256=digest(destination.with_suffix(".sass")),
    )


def auxiliary_binary_identity():
    mapped = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if "fp4_quant" in Path(path).name and path.endswith(".so"):
            mapped.add(path)
    quant = {path: digest(path) for path in sorted(mapped)}
    cache = os.environ.get("TRITON_CACHE_DIR")
    triton = (
        {}
        if cache is None
        else {str(path): digest(path) for path in sorted(Path(cache).rglob("*.cubin"))}
    )
    return dict(quantization_shared_objects=quant, triton_cubins=triton)


def native_modules():
    return tuple(
        importlib.import_module(f"{NATIVE_PACKAGE}.{name}")
        for name in (
            "moe_fp4_fc1_act_q1",
            "moe_fp4_fc2_finalize",
            "moe_fp4_q0_route_triton",
        )
    )


def calc_diff(actual, expected):
    actual, expected = actual.double(), expected.double()
    denominator = (actual * actual + expected * expected).sum().item()
    return (
        0.0
        if denominator == 0
        else 1.0 - 2.0 * (actual * expected).sum().item() / denominator
    )


def check_accuracy(actual, expected):
    actual, expected = actual.float(), expected.float()
    atol = max(0.05, 1.5 * max(expected.std().item(), 0.01))
    absolute = (actual - expected).abs()
    relative = absolute / (expected.abs() + 1e-8)
    percent = ((absolute < atol) | (relative < 0.5)).float().mean().item()
    diff = calc_diff(actual, expected)
    return percent >= 0.97 and diff <= 3e-2, percent, diff


def decode_codes(codes):
    values = codes.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    magnitude = values[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def rne_codes(value):
    absolute = value.abs().clamp_max(6.0)
    boundaries = absolute.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    code = torch.bucketize(absolute, boundaries)
    for tie in (0.75, 1.75, 3.5):
        code += (absolute == tie).to(code.dtype)
    return code.to(torch.uint8) | ((value < 0).to(torch.uint8) << 3)


def fake_quant(value, scale):
    blocks = value.float().reshape(value.shape[0], -1, 16)
    sf = (
        (blocks.abs().amax(dim=2) * (scale / 6.0))
        .clamp(max=448.0)
        .to(torch.float8_e4m3fn)
    )
    encode = torch.where(
        sf.float() == 0, torch.zeros_like(sf.float()), scale / sf.float()
    )
    codes = rne_codes(blocks * encode[:, :, None]).reshape(value.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    decoded = decode_codes(codes).reshape(blocks.shape)
    decoded = (decoded * sf.float()[:, :, None] / scale).reshape(value.shape)
    return packed, sf.view(torch.uint8), decoded


def pack_sf(sf):
    rows, columns = sf.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    row = torch.arange(rows, device=sf.device, dtype=torch.int64)[:, None]
    column = torch.arange(columns, device=sf.device, dtype=torch.int64)[None, :]
    index = (
        ((row >> 7) * (padded_columns >> 2) + (column >> 2)) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (column & 3)
    )
    packed = torch.zeros(
        padded_rows * padded_columns, device=sf.device, dtype=torch.uint8
    )
    packed[index] = sf
    return packed.view(padded_rows, padded_columns)


def activate(value, activation, limit=None):
    if activation is ActivationType.Relu2:
        return value.relu().square()
    linear, gate = value.chunk(2, dim=-1)
    if activation is ActivationType.Situ:
        return (
            DEFAULT_SITU_BETA
            * torch.tanh(gate / DEFAULT_SITU_BETA)
            * torch.sigmoid(gate)
            * DEFAULT_SITU_LINEAR_BETA
            * torch.tanh(linear / DEFAULT_SITU_LINEAR_BETA)
        )
    if activation is not ActivationType.Swiglu:
        raise ValueError("the native chain supports SwiGLU, SiTU and ReLU2")
    if limit is not None:
        gate = gate.clamp(max=limit)
        linear = linear.clamp(-limit, limit)
    return F.silu(gate) * linear


def limit10_baseline(baseline, cache_dir):
    """Derive exactly two activation sites outside the installed package tree."""
    source = Path(baseline.__file__).resolve()
    if digest(source) != BASELINE_SHA256:
        raise ValueError("limit10 adapter requires the pinned PR5863 source")
    text = source.read_text()
    sites = (
        (
            "                    x = v[j]\n"
            "                    v[j] = ((x / (cute.exp(-x) + F32(1.0))) * ru[j]).to(BF16).to(F32)",
            "                    x = cute.math.min(v[j], F32(10.0))\n"
            "                    u = cute.math.max(F32(-10.0), cute.math.min(ru[j], F32(10.0)))\n"
            "                    v[j] = ((x / (cute.exp(-x) + F32(1.0))) * u).to(BF16).to(F32)",
        ),
        (
            "                        gv = acc[i] * scale\n"
            "                        uv = acc[i + 2 * VM] * scale\n"
            "                        av = (gv / (cute.exp(-gv) + F32(1.0))) * uv",
            "                        gv = acc[i] * scale\n"
            "                        uv = acc[i + 2 * VM] * scale\n"
            "                        gv = cute.math.min(gv, F32(10.0))\n"
            "                        uv = cute.math.max(F32(-10.0), cute.math.min(uv, F32(10.0)))\n"
            "                        av = (gv / (cute.exp(-gv) + F32(1.0))) * uv",
        ),
    )
    for old, new in sites:
        if text.count(old) != 1:
            raise ValueError("limit10 adapter no longer matches exactly two sites")
        text = text.replace(old, new, 1)
    cache_dir = Path(cache_dir).resolve()
    package_root = Path(importlib.import_module("flashinfer").__file__).resolve().parent
    if cache_dir.is_relative_to(package_root):
        raise ValueError("adapter cache must be outside the installed package")
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "baseline_limit10.py"
    derived = text.encode()
    if output.exists():
        if output.read_bytes() != derived:
            raise ValueError("adapter cache contains a different derived baseline")
    else:
        with output.open("xb") as stream:
            stream.write(derived)
    spec = importlib.util.spec_from_file_location("sm12x_benchmark_limit10", output)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, dict(
        label="baseline+limit10 adapter",
        raw_path=str(source),
        raw_sha256=BASELINE_SHA256,
        derived_path=str(output),
        derived_sha256=digest(output),
        changed_sites=2,
        swiglu_limit=10.0,
    )


class Fixture:
    """Pure-Torch oracle with shared or per-expert weight encoding scales."""

    def __init__(
        self,
        model,
        tokens,
        *,
        shape_override=None,
        activation=None,
        padded_routes=False,
        input_scale=1.0,
        baseline=None,
        per_expert_scales=False,
    ):
        h, intermediate, experts, topk, gated = CASES[model]
        if shape_override is not None:
            h, intermediate, experts, topk = shape_override
        self.activation = activation or (
            ActivationType.Swiglu if gated else ActivationType.Relu2
        )
        self.gated = self.activation is not ActivationType.Relu2
        if (
            min(h, intermediate, experts, topk, tokens) <= 0
            or h % 128
            or intermediate % (128 if self.gated else 64)
            or topk > experts
        ):
            raise ValueError("shape violates native alignment or top-k bounds")
        self.limit = 10.0 if model == "dsv4" else None
        if self.limit is not None and self.activation is not ActivationType.Swiglu:
            raise ValueError("DSv4 is the SwiGLU limit10 case")
        self.h, self.intermediate, self.experts, self.topk = (
            h,
            intermediate,
            experts,
            topk,
        )
        self.fc1, self.fc2, self.route = native_modules()
        self.baseline = baseline
        torch.manual_seed(42 + tokens)
        self.x = (
            torch.randn(tokens, h, device="cuda", dtype=torch.bfloat16) * input_scale
        )
        self.ids = (
            torch.rand(tokens, experts, device="cuda").topk(topk, dim=1).indices.int()
        )
        if padded_routes:
            self.ids[-1] = -1
            if tokens > 1:
                self.ids[0, -1] = -1
        self.router_weights = torch.randn(tokens, topk, device="cuda").softmax(dim=1)
        weight_scale = torch.full((experts,), 512.0, device="cuda")
        if per_expert_scales:
            weight_scale *= torch.arange(1, experts + 1, device="cuda")
        self.weight_dequant_scale = weight_scale.reciprocal()
        self.g0 = (2688.0 / self.x.float().abs().amax()).reshape(1)
        _, _, x_decoded = fake_quant(self.x, self.g0)
        weights1, scales1, weights2, scales2, decoded1, decoded2 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for expert in range(experts):
            w1 = (
                torch.randn(
                    intermediate * (2 if self.gated else 1),
                    h,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / h**0.5
            )
            w2 = (
                torch.randn(h, intermediate, device="cuda", dtype=torch.bfloat16)
                / intermediate**0.5
            )
            q1, sf1, d1 = fake_quant(w1, weight_scale[expert : expert + 1])
            q2, sf2, d2 = fake_quant(w2, weight_scale[expert : expert + 1])
            weights1.append(q1)
            scales1.append(pack_sf(sf1))
            weights2.append(q2)
            scales2.append(pack_sf(sf2))
            decoded1.append(d1)
            decoded2.append(d2)
        self.w1, self.sf1 = torch.stack(weights1), torch.stack(scales1)
        self.w2, self.sf2 = torch.stack(weights2), torch.stack(scales2)
        acts = torch.zeros(
            tokens * topk, intermediate, device="cuda", dtype=torch.bfloat16
        )
        bf16_reference = torch.zeros(tokens, h, device="cuda", dtype=torch.float32)
        positions = []
        for expert in range(experts):
            token, slot = (self.ids == expert).nonzero(as_tuple=True)
            positions.append((token, slot))
            if not token.numel():
                continue
            pre = x_decoded[token] @ decoded1[expert].t()
            bf16_pre = self.x[token].float() @ decoded1[expert].t()
            acts[token * topk + slot] = activate(
                pre, self.activation, self.limit
            ).bfloat16()
            bf16_act = activate(bf16_pre, self.activation, self.limit).bfloat16()
            bf16_reference.index_add_(
                0,
                token,
                (bf16_act.float() @ decoded2[expert].t())
                * self.router_weights[token, slot, None],
            )
        self.g1 = (2688.0 / acts.float().abs().amax()).reshape(1)
        _, _, act_decoded = fake_quant(acts, self.g1)
        reference = torch.zeros_like(bf16_reference)
        for expert, (token, slot) in enumerate(positions):
            if token.numel():
                reference.index_add_(
                    0,
                    token,
                    (act_decoded[token * topk + slot] @ decoded2[expert].t())
                    * self.router_weights[token, slot, None],
                )
        self.reference, self.bf16_reference = reference.bfloat16(), bf16_reference
        self.out = torch.empty_like(self.x)
        self.q1 = torch.empty(
            tokens * topk, intermediate // 2, device="cuda", dtype=torch.uint8
        )
        self.q1_sf = torch.zeros(
            self.fc1.out_sf_shape(tokens * topk, intermediate, experts),
            device="cuda",
            dtype=torch.uint8,
        )
        sms = torch.cuda.get_device_properties(self.x.device).multi_processor_count
        self.fc1_tile, self.gather = self.fc1.select_tactic(
            total_rows=self.ids.numel(),
            n=intermediate,
            k=h,
            num_experts=experts,
            num_sms=sms,
            activation=self.activation,
        )
        self.fc2_tile = self.fc2.select_tile(
            total_rows=self.ids.numel(),
            n=h,
            k=intermediate,
            num_experts=experts,
            num_sms=sms,
        )
        if self.fc1.is_swapab(self.fc1_tile):
            self.workspace = self.route.make_nvfp4_route_meta_workspace(
                self.ids, experts, tile_m=self.fc1_tile[0]
            )
        elif self.gather:
            self.workspace = self.route.make_nvfp4_route_meta_workspace(
                self.ids, experts
            )
        else:
            self.workspace = self.route.make_nvfp4_q0_route_workspace(
                self.x, self.ids, experts
            )
        self.fc2_work = None
        if self.fc2.is_swapab(self.fc2_tile):
            if (
                self.fc1.is_swapab(self.fc1_tile)
                and self.fc1_tile[0] == self.fc2_tile[0]
            ):
                self.fc2_work = (self.workspace.worklist, self.workspace.workcount)
            else:
                self.fc2_work = self.route.make_nvfp4_worklist_workspace(
                    self.ids.numel(), experts, self.fc2_tile[0], self.x.device
                )
        self.main_out = torch.empty_like(self.x)
        self.inputs = dict(
            seed=42 + tokens,
            weight_encode_scale=512.0,
            g0=self.g0.item(),
            g1=self.g1.item(),
            activation=self.activation.name,
            swiglu_limit=self.limit,
        )
        if self.gated:
            self.main_w1 = torch.cat(
                (self.w1[:, intermediate:], self.w1[:, :intermediate]), dim=1
            ).contiguous()
            self.main_sf1 = torch.cat(
                (self.sf1[:, intermediate:], self.sf1[:, :intermediate]), dim=1
            ).contiguous()
        else:
            self.main_w1, self.main_sf1 = self.w1, self.sf1
        if baseline is not None:
            if self.activation not in (ActivationType.Swiglu, ActivationType.Relu2):
                raise ValueError("the baseline does not support SiTU")
            self.main_workspace = baseline.allocate_workspace(
                tokens, h, intermediate, experts, topk, self.gated, self.x.device
            )

    def native(self, enable_pdl=False):
        self.out.zero_()
        offsets, token_map, weights, q, sf = (
            self.fc1.cute_dsl_sm12x_route_fc1_act_q1_nvfp4(
                self.x,
                self.ids,
                self.router_weights,
                self.main_w1,
                self.main_sf1.view(torch.float8_e4m3fn),
                self.g0,
                self.weight_dequant_scale,
                self.g1,
                activation=self.activation,
                workspace=self.workspace,
                enable_pdl=enable_pdl,
                swiglu_limit=self.limit,
                out_q=self.q1,
                out_sf=self.q1_sf,
            )
        )
        work = {}
        if self.fc2_work is not None:
            worklist, workcount = self.fc2_work
            if worklist is not getattr(self.workspace, "worklist", None):
                self.route.build_nvfp4_worklist(
                    offsets,
                    worklist,
                    workcount,
                    self.fc2_tile[0],
                    total_rows=self.ids.numel(),
                )
            work = dict(worklist=worklist, workcount=workcount)
        return self.fc2.cute_dsl_sm12x_fc2_finalize_nvfp4(
            q,
            sf,
            self.w2,
            self.sf2.view(torch.float8_e4m3fn),
            offsets,
            token_map,
            weights,
            self.weight_dequant_scale,
            self.g1,
            self.x.shape[0],
            tune=False,
            enable_pdl=enable_pdl,
            out=self.out,
            **work,
        )

    def main(self):
        return self.baseline.run_moe_w4a4(
            self.x,
            self.ids,
            self.router_weights,
            self.main_w1,
            self.main_sf1,
            self.weight_dequant_scale,
            self.w2,
            self.sf2,
            self.weight_dequant_scale,
            self.g0,
            self.g1,
            self.main_out,
            self.main_workspace,
        )

    def check(self, output):
        quant_diff = calc_diff(output, self.reference)
        passed, percent, bf16_diff = check_accuracy(output, self.bf16_reference)
        record = dict(quant_diff=quant_diff, bf16_diff=bf16_diff, percent=percent)
        assert torch.isfinite(output).all() and quant_diff < 2e-3 and passed, record
        return record

    def input_hashes(self):
        names = (
            "x",
            "ids",
            "router_weights",
            "w1",
            "sf1",
            "w2",
            "sf2",
            "g0",
            "g1",
            "weight_dequant_scale",
            "main_w1",
            "main_sf1",
        )
        return {
            name: tensor_digest(getattr(self, name))
            for name in names
            if hasattr(self, name)
        }


def telemetry():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    processes = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu,
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    assert len(processes) == 1, processes
    status = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu,
            "--query-gpu=name,uuid,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    performance = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "-q", "-d", "PERFORMANCE"], text=True
    )
    return dict(gpu=status, performance=performance, compute_app_pids=processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--model", choices=CASES)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--native-dir",
        type=Path,
        help="fresh artifact directory; required for performance-qualified records",
    )
    parser.add_argument(
        "--shape-override",
        nargs=4,
        type=int,
        metavar=("H", "I", "E", "TOPK"),
        help="correctness-only shape; requires --check-only",
    )
    parser.add_argument(
        "--adapter-cache",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".cache/sm12x_nvfp4_benchmark",
        help="benchmark-only baseline+limit10 derived source cache",
    )
    args = parser.parse_args()
    if args.list_cases:
        for model, counts in TOKEN_COUNTS.items():
            for tokens in counts:
                print(f"{model}\t{tokens}")
        return
    if args.model is None or args.tokens is None or args.output is None:
        parser.error("--model, --tokens and --output are required")
    if args.output.exists():
        parser.error("output already exists")
    if args.native_dir is not None:
        args.native_dir = args.native_dir.resolve()
        if not (args.native_dir / "native").is_dir():
            parser.error(
                "native capture must be configured by the standalone CLI before imports"
            )
    if args.shape_override is not None and not args.check_only:
        parser.error("shape overrides require --check-only")
    if args.shape_override is None and args.tokens not in TOKEN_COUNTS[args.model]:
        parser.error("case is outside the 21-case performance matrix")
    props = torch.cuda.get_device_properties(0)
    if (props.major, props.minor) != (12, 0):
        parser.error("requires SM120")
    if not args.check_only and (
        props.multi_processor_count != 188
        or "RTX PRO 6000" not in props.name
        or "Server Edition" not in props.name
        or not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-")
    ):
        parser.error(
            "performance requires a UUID-bound 188-SM RTX PRO 6000 Server Edition"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = importlib.import_module(f"{NATIVE_PACKAGE}.moe_nvfp4_w4a4")
    if digest(baseline.__file__) != BASELINE_SHA256:
        raise ValueError("baseline source is not the pinned PR5863 implementation")
    adapter = None
    if args.model == "dsv4":
        baseline, adapter = limit10_baseline(baseline, args.adapter_cache)
    data = Fixture(
        args.model, args.tokens, shape_override=args.shape_override, baseline=baseline
    )
    immutable = data.input_hashes()
    native_root = Path(data.fc1.__file__).resolve().parent
    identity = {str(path): digest(path) for path in native_root.rglob("*.py")}
    quant_module = importlib.import_module("flashinfer.quantization.fp4_quantization")
    identity[str(Path(quant_module.__file__).resolve())] = digest(quant_module.__file__)
    identity[str(Path(__file__).resolve())] = digest(__file__)
    identity[str(Path(baseline.__file__).resolve())] = digest(baseline.__file__)
    data.fc1._COMPILED.clear()
    data.fc2._COMPILED.clear()
    outputs, graphs, events, accuracy = {}, {}, {}, {}
    calls = {"native": data.native, "baseline": data.main}
    for arm, call in calls.items():
        outputs[arm] = call()
        torch.cuda.synchronize()
        accuracy[arm + "_direct"] = data.check(outputs[arm].clone())
    before = telemetry()
    assert len(data.fc1._COMPILED) == len(data.fc2._COMPILED) == 1
    fc1_keys, fc2_keys = list(data.fc1._COMPILED), list(data.fc2._COMPILED)
    assert fc1_keys[0][0][:2] == tuple(data.fc1_tile[:2])
    assert fc1_keys[0][4] == data.gather
    assert fc2_keys[0][0][:2] == tuple(data.fc2_tile[:2])
    dispatch = dict(
        fc1_tile=data.fc1_tile,
        gather=data.gather,
        fc2_tile=data.fc2_tile,
        fc1_keys=[str(key) for key in fc1_keys],
        fc2_keys=[str(key) for key in fc2_keys],
        baseline_keys=[str(key) for key in baseline._COMPILED],
    )
    print("DISPATCH", json.dumps(dispatch), flush=True)
    for arm, call in calls.items():
        pair = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pair[0].record()
            outputs[arm] = call()
            pair[1].record()
        graphs[arm], events[arm] = graph, pair
        for repeat in range(2):
            outputs[arm].fill_(float("nan"))
            if arm == "native":
                data.q1.fill_(255)
                data.q1_sf.fill_(255)
            graph.replay()
            pair[1].synchronize()
            accuracy[f"{arm}_poison_{repeat}"] = data.check(outputs[arm].clone())
    for path, expected in loaded_source_identity().items():
        if path in identity:
            assert identity[path] == expected, path
        identity[path] = expected
    print("SOURCE_IDENTITY", json.dumps(identity), flush=True)
    compiled_objects = []
    for label, module, tile in (
        ("fc1", data.fc1, data.fc1_tile),
        ("fc2", data.fc2, data.fc2_tile),
    ):
        for index, (key, compiled) in enumerate(module._COMPILED.items()):
            compiled_objects.append(
                (f"{label}_{index}", str(key), compiled, not module.is_swapab(tile))
            )
    for index, (key, entry) in enumerate(baseline._COMPILED.items()):
        assert len(entry) == 3 and entry[2] is not None
        compiled_objects.append((f"baseline_{index}", str(key), entry[2], False))
    artifacts = {}
    if args.native_dir is not None:
        for label, key, compiled, ws in compiled_objects:
            artifacts[label] = export_native(
                compiled,
                args.native_dir / "native" / f"{label}.cubin",
                warp_specialized=ws,
            )
            artifacts[label]["cache_key"] = key
    auxiliary = auxiliary_binary_identity()
    triton_artifacts = {}
    if args.native_dir is not None:
        assert auxiliary["triton_cubins"], "no attributable Triton cubins"
        assert auxiliary["quantization_shared_objects"] or not data.gather
        for index, (path, expected) in enumerate(auxiliary["triton_cubins"].items()):
            item = export_native(
                Path(path).read_bytes(),
                args.native_dir / "native" / f"triton_{index}.cubin",
                warp_specialized=False,
            )
            assert item["sha256"] == expected, path
            item.update(cache_path=path, cache_sha256=expected)
            triton_artifacts[path] = item
    rows = []
    if not args.check_only:
        flush = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
        for order, arms in enumerate((("native", "baseline"), ("baseline", "native"))):
            for arm in arms:
                inventory_before = telemetry()
                for _ in range(5):
                    flush.zero_()
                    graphs[arm].replay()
                events[arm][1].synchronize()
                samples = []
                for _ in range(100):
                    flush.zero_()
                    graphs[arm].replay()
                    events[arm][1].synchronize()
                    samples.append(events[arm][0].elapsed_time(events[arm][1]) * 1000)
                assert min(samples) > 0
                accuracy[f"{arm}_order_{order}"] = data.check(outputs[arm].clone())
                rows.append(
                    dict(
                        arm=arm,
                        order=order,
                        samples_us=samples,
                        median_us=statistics.median(samples),
                        before=inventory_before,
                        after=telemetry(),
                    )
                )
                print("TIMING", json.dumps(rows[-1]), flush=True)
    for arm, graph in graphs.items():
        graph.replay()
        events[arm][1].synchronize()
        accuracy[arm + "_after"] = data.check(outputs[arm].clone())
    assert data.input_hashes() == immutable
    for path, expected in identity.items():
        assert digest(path) == expected, path
    for label, _, compiled, _ in compiled_objects:
        if artifacts:
            item = artifacts[label]
            assert hashlib.sha256(cubin_bytes(compiled)).hexdigest() == item["sha256"]
            assert digest(item["cubin"]) == item["sha256"]
    for files in auxiliary.values():
        for path, expected in files.items():
            assert digest(path) == expected, path
    for item in triton_artifacts.values():
        assert digest(item["cubin"]) == item["sha256"]
    assert auxiliary_binary_identity() == auxiliary
    speedups = {}
    for order in range(2):
        medians = {
            row["arm"]: row["median_us"] for row in rows if row["order"] == order
        }
        if medians:
            speedups[str(order)] = medians["baseline"] / medians["native"]
    record = dict(
        model=args.model,
        tokens=args.tokens,
        shape=args.shape_override or CASES[args.model][:4],
        performance_eligible=not args.check_only and bool(artifacts),
        baseline_adapter=adapter,
        identity=identity,
        inputs=data.inputs,
        identity_scope="native_source_inventory_and_loaded_flashinfer_python_sources",
        artifacts=artifacts,
        auxiliary_binaries=auxiliary,
        triton_artifacts=triton_artifacts,
        immutable_inputs=immutable,
        dispatch=dispatch,
        runtime=dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cutlass=importlib.import_module("cutlass").__version__,
            gpu=props.name,
            gpu_uuid=str(props.uuid),
            sms=props.multi_processor_count,
        ),
        before=before,
        after=telemetry(),
        accuracy=accuracy,
        rows=rows,
        speedups=speedups,
        protocol=dict(
            tp=1,
            pdl=False,
            warmup=0 if args.check_only else 5,
            samples=0 if args.check_only else 100,
            l2_flush_mib=256,
            events_per_graph=2,
            boundary="raw_BF16_to_MoE_output",
            output_init="included_once_at_chain_start",
            interstage_clear=False,
            interstage_flush=False,
            clock_policy="unchanged_by_benchmark",
            power_policy="unchanged_by_benchmark",
            weight_encode_scale=512.0,
            exclusive_gpu_preflight="executor_must_require_selected_UUID_compute_apps_empty_before_start",
            compute_apps_at_sampling_boundaries=1,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print("PASS", args.output, digest(args.output), flush=True)


if __name__ == "__main__":
    main()
