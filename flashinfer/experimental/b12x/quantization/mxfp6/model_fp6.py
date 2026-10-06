"""Point-at-a-model offline MX-FP6 (W6A6) converters for HF checkpoints.

Walks a HuggingFace model directory (``config.json`` + ``model.safetensors`` /
``model.safetensors.index.json``), discovers the weights the b12x FP6
kernels can consume, quantizes them with the offline primitives, and writes
per-layer artifacts + a ``manifest.json``.

* :func:`discover_moe_experts` / :func:`convert_moe_model_to_fp6` - routed-expert
  FFNs -> packed MX-FP6 in the layout the fused FP6 MoE kernel expects. FC1 rows
  are stacked ``[up; gate]`` (the kernel's gated-silu convention, verified by
  ``test_moe_fp6_numeric_vs_reference``).
* :func:`discover_dense_linears` / :func:`convert_dense_model_to_fp6` - dense MLP
  (+ optional attention) linears -> packed MX-FP6 for ``dense_fp6_linear``.

Discovery works off the safetensors **index keys** (regex, not hardcoded module
paths) so it adapts to naming variations. Linear-attention/SSM projections, the
vision tower, norms, embeddings, router gates and non-128 matrices are left in
BF16 and reported as skipped.

Use ``dry_run=True`` first: it prints the plan from the index alone, writing
nothing (the output directory is not even created).
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch

_TILE = 128

# ...layers.{L}.<module>.experts.{E}.<proj>.weight  (one Linear weight per expert)
_EXPERT_RE = re.compile(
    r"^(?P<pre>.*\.layers\.)(?P<layer>\d+)\.(?P<module>.*?)\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>[A-Za-z0-9_]+)\.weight$"
)
# ...layers.{L}.<module>.experts.<proj>  (all experts packed in one 3-D tensor;
# note: no per-expert index and no ``.weight`` suffix)
_PACKED_EXPERT_RE = re.compile(
    r"^(?P<pre>.*\.layers\.)(?P<layer>\d+)\.(?P<module>.*?)\.experts\."
    r"(?P<proj>[A-Za-z0-9_]+)$"
)
# ...layers.{L}.<relative.module.path>.weight  (relative path may be multi-segment)
_LAYER_LINEAR_RE = re.compile(
    r"^(?P<pre>.*\.layers\.)(?P<layer>\d+)\.(?P<rel>.+)\.weight$"
)
# Relative module paths the dense walker must never quantize.
_DENSE_SKIP_RE = re.compile(
    r"(experts\.|linear_attn|linear_attention|mamba|\bconv\b|in_proj|ssm|rotary|norm)",
    re.IGNORECASE,
)

_SEP_GATE = ("gate_proj", "w1")
_SEP_UP = ("up_proj", "w3")
_SEP_DOWN = ("down_proj", "w2")
_FUSED_GATE_UP = ("gate_up_proj", "w13")
_ATTN_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj")


class SafetensorsModel:
    """Lazy reader for a HF safetensors checkpoint directory."""

    def __init__(self, model_path: str | pathlib.Path):
        self.path = pathlib.Path(model_path)
        cfg = self.path / "config.json"
        self.config: dict = json.loads(cfg.read_text()) if cfg.exists() else {}
        self.text_config: dict = self.config.get("text_config", self.config)
        self.weight_map = self._build_weight_map()
        self._handles: dict[str, object] = {}

    def _build_weight_map(self) -> dict[str, str]:
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            return json.loads(index.read_text())["weight_map"]
        single = self.path / "model.safetensors"
        if single.exists():
            from safetensors import safe_open

            with safe_open(str(single), framework="pt") as f:  # type: ignore[no-untyped-call]
                return {k: "model.safetensors" for k in f.keys()}
        raise FileNotFoundError(f"no model.safetensors(.index.json) under {self.path}")

    def keys(self) -> Iterable[str]:
        return self.weight_map.keys()

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def _handle(self, key: str):
        from safetensors import safe_open

        shard = self.weight_map[key]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.path / shard), framework="pt")  # type: ignore[no-untyped-call]
            self._handles[shard] = handle
        return handle

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def shape_of(self, key: str) -> tuple[int, ...]:
        return tuple(self._handle(key).get_slice(key).get_shape())


@dataclass
class MoEExpertScheme:
    """Where/how routed-expert weights live in a checkpoint."""

    num_experts: int
    layers: list[int]
    prefix_template: str  # e.g. "model.language_model.layers.{L}.mlp"
    gate_name: str
    up_name: str
    down_name: str
    is_fused: bool  # True when gate+up share one fused tensor (gate_name == up_name)
    packed: bool = False  # True when all experts live in one stacked 3-D tensor

    def expert_key(self, layer: int, expert: int, proj: str) -> str:
        return f"{self.prefix_template.format(L=layer)}.experts.{expert}.{proj}.weight"

    def packed_key(self, layer: int, proj: str) -> str:
        return f"{self.prefix_template.format(L=layer)}.experts.{proj}"


@dataclass
class DenseLinearScheme:
    """Relative module names for dense linears, per layer."""

    layers: list[int]
    mlp_gate: Optional[str]
    mlp_up: Optional[str]
    mlp_down: Optional[str]
    mlp_fused_gate_up: Optional[str]
    attn_projs: list[str]
    prefix_template: str = ""  # e.g. "model.language_model.layers.{L}"

    def linear_key(self, layer: int, rel: str) -> str:
        return f"{self.prefix_template.format(L=layer)}.{rel}.weight"


@dataclass
class ConvertReport:
    arch: str
    layers: list[int]
    out_dir: str
    tensors_written: int = 0
    artifacts: list = field(default_factory=list)
    skipped: list = field(default_factory=list)


def _match_proj(names: set[str], candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in names:
            return c
    return None


def _skip_aux(key: str) -> bool:
    """Ignore auxiliary heads (e.g. the MTP / multi-token-prediction stack)."""
    return key.startswith("mtp.") or ".mtp." in key


def discover_moe_experts(model: SafetensorsModel) -> Optional[MoEExpertScheme]:
    """Find routed-expert weights, per-expert *or* packed, or ``None``."""
    scheme = _discover_per_expert(model)
    if scheme is not None:
        return scheme
    return _discover_packed_experts(model)


def _discover_per_expert(model: SafetensorsModel) -> Optional[MoEExpertScheme]:
    # layer -> (module, {proj: set(expert ids)})
    found: dict[int, tuple[str, dict[str, set[int]]]] = {}
    template_module: Optional[str] = None
    template_pre: Optional[str] = None
    for key in model.keys():
        if _skip_aux(key):
            continue
        m = _EXPERT_RE.match(key)
        if not m:
            continue
        if template_pre is None:
            template_pre, template_module = m["pre"], m["module"]
        slot = found.setdefault(int(m["layer"]), (m["module"], {}))
        slot[1].setdefault(m["proj"], set()).add(int(m["expert"]))

    if not found:
        return None

    sample_projs: set[str] = set()
    for _, (_, projs) in found.items():
        sample_projs |= set(projs)

    fused = _match_proj(sample_projs, _FUSED_GATE_UP)
    gate = _match_proj(sample_projs, _SEP_GATE)
    up = _match_proj(sample_projs, _SEP_UP)
    down = _match_proj(sample_projs, _SEP_DOWN)
    if down is None or (fused is None and (gate is None or up is None)):
        return None

    is_fused = fused is not None
    gate_name = fused if is_fused else gate  # type: ignore[assignment]
    up_name = fused if is_fused else up  # type: ignore[assignment]
    need = {down} | ({fused} if is_fused else {gate, up})  # type: ignore[arg-type]

    complete_layers = []
    expert_counts: set[int] = set()
    for layer, (_, projs) in found.items():
        if not need.issubset(set(projs)):
            continue
        counts = {len(projs[p]) for p in need}
        if len(counts) != 1:
            continue
        complete_layers.append(layer)
        expert_counts |= counts
    if not complete_layers or len(expert_counts) != 1:
        return None

    return MoEExpertScheme(
        num_experts=next(iter(expert_counts)),
        layers=sorted(complete_layers),
        prefix_template=f"{template_pre}{{L}}.{template_module}",
        gate_name=gate_name,  # type: ignore[arg-type]
        up_name=up_name,  # type: ignore[arg-type]
        down_name=down,  # type: ignore[arg-type]
        is_fused=is_fused,
        packed=False,
    )


def _discover_packed_experts(model: SafetensorsModel) -> Optional[MoEExpertScheme]:
    """Experts stacked in a single 3-D tensor (``experts.gate_up_proj`` etc.)."""
    layers: dict[int, set[str]] = {}
    template_module: Optional[str] = None
    template_pre: Optional[str] = None
    for key in model.keys():
        if _skip_aux(key):
            continue
        m = _PACKED_EXPERT_RE.match(key)
        if not m:
            continue
        if template_pre is None:
            template_pre, template_module = m["pre"], m["module"]
        layers.setdefault(int(m["layer"]), set()).add(m["proj"])

    if not layers:
        return None

    sample = set().union(*layers.values())
    fused = _match_proj(sample, _FUSED_GATE_UP)
    gate = _match_proj(sample, _SEP_GATE)
    up = _match_proj(sample, _SEP_UP)
    down = _match_proj(sample, _SEP_DOWN)
    if down is None or (fused is None and (gate is None or up is None)):
        return None

    is_fused = fused is not None
    gate_name = fused if is_fused else gate  # type: ignore[assignment]
    up_name = fused if is_fused else up  # type: ignore[assignment]
    need = {down} | ({fused} if is_fused else {gate, up})  # type: ignore[arg-type]
    complete = sorted(L for L, projs in layers.items() if need.issubset(projs))
    if not complete:
        return None

    prefix_template = f"{template_pre}{{L}}.{template_module}"
    # Expert count = leading dim of the packed down tensor on the first layer.
    e_count = model.shape_of(
        f"{prefix_template.format(L=complete[0])}.experts.{down}"
    )[0]
    return MoEExpertScheme(
        num_experts=int(e_count),
        layers=complete,
        prefix_template=prefix_template,
        gate_name=gate_name,  # type: ignore[arg-type]
        up_name=up_name,  # type: ignore[arg-type]
        down_name=down,  # type: ignore[arg-type]
        is_fused=is_fused,
        packed=True,
    )


def discover_dense_linears(model: SafetensorsModel) -> Optional[DenseLinearScheme]:
    """Find dense MLP + attention linears (relative module names), or ``None``."""
    rels_by_layer: dict[int, set[str]] = {}
    template_pre: Optional[str] = None
    for key in model.keys():
        m = _LAYER_LINEAR_RE.match(key)
        if not m:
            continue
        rel = m["rel"]
        if _DENSE_SKIP_RE.search(rel):
            continue
        if template_pre is None:
            template_pre = m["pre"]
        rels_by_layer.setdefault(int(m["layer"]), set()).add(rel)

    if not rels_by_layer:
        return None

    all_rels: set[str] = set()
    for rels in rels_by_layer.values():
        all_rels |= rels

    def _find(suffixes: tuple[str, ...]) -> Optional[str]:
        for rel in sorted(all_rels):
            for suf in suffixes:
                if rel == f"mlp.{suf}":
                    return rel
        return None

    mlp_gate = _find(_SEP_GATE)
    mlp_up = _find(_SEP_UP)
    mlp_down = _find(_SEP_DOWN)
    mlp_fused = _find(_FUSED_GATE_UP)

    attn_projs = [
        rel
        for proj in _ATTN_ORDER
        for rel in [f"self_attn.{proj}"]
        if rel in all_rels
    ]

    if mlp_down is None and not attn_projs:
        return None

    # Layers that actually carry a dense MLP (or attention) we can quantize.
    mlp_names = {n for n in (mlp_gate, mlp_up, mlp_down, mlp_fused) if n}
    target = mlp_names | set(attn_projs)
    used_layers = sorted(
        L for L, rels in rels_by_layer.items() if rels & target
    )
    return DenseLinearScheme(
        layers=used_layers,
        mlp_gate=mlp_gate,
        mlp_up=mlp_up,
        mlp_down=mlp_down,
        mlp_fused_gate_up=mlp_fused,
        attn_projs=attn_projs,
        prefix_template=f"{template_pre}{{L}}",
    )


def _write_manifest(out: pathlib.Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(payload, indent=2))


def _split_gate_up(gate_up: torch.Tensor, dim: int, order: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a fused gate_up tensor along ``dim`` into ``(gate, up)``."""
    half = gate_up.shape[dim] // 2
    a, b = gate_up.narrow(dim, 0, half), gate_up.narrow(dim, half, half)
    return (a, b) if order == "gate_up" else (b, a)


def _build_layer_w1_w2(
    model: SafetensorsModel,
    scheme: MoEExpertScheme,
    layer: int,
    device: str,
    is_gated: bool,
    gate_up_order: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``(w1, w2)`` for one MoE layer in the kernel's ``[up; gate]`` order."""
    if scheme.packed:
        down = model.get_tensor(scheme.packed_key(layer, scheme.down_name)).to(
            device, torch.bfloat16
        )  # (E, K, N)
        if scheme.is_fused:
            gate_up = model.get_tensor(scheme.packed_key(layer, scheme.gate_name)).to(
                device, torch.bfloat16
            )  # (E, 2N, K)
            gate, up = _split_gate_up(gate_up, dim=1, order=gate_up_order)
        else:
            gate = model.get_tensor(scheme.packed_key(layer, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            up = model.get_tensor(scheme.packed_key(layer, scheme.up_name)).to(
                device, torch.bfloat16
            )
        # Kernel gated-silu convention: FC1 rows are [up; gate] (along the row dim).
        w1 = torch.cat([up, gate], dim=1) if is_gated else gate
        return w1.contiguous(), down.contiguous()

    # Per-expert: load + stack each expert.
    w1_list, w2_list = [], []
    for e in range(scheme.num_experts):
        down = model.get_tensor(scheme.expert_key(layer, e, scheme.down_name)).to(
            device, torch.bfloat16
        )
        if scheme.is_fused:
            gate_up = model.get_tensor(scheme.expert_key(layer, e, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            gate, up = _split_gate_up(gate_up, dim=0, order=gate_up_order)
        else:
            gate = model.get_tensor(scheme.expert_key(layer, e, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            up = model.get_tensor(scheme.expert_key(layer, e, scheme.up_name)).to(
                device, torch.bfloat16
            )
        w1_list.append(torch.cat([up, gate], dim=0) if is_gated else gate)
        w2_list.append(down)
    return torch.stack(w1_list, dim=0), torch.stack(w2_list, dim=0)


def convert_moe_model_to_fp6(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    source_format: str = "mxfp6_w6a8",
    activation: str = "silu",
    gate_up_order: str = "gate_up",
    limit_layers: Optional[int] = None,
    device: str = "cuda",
    use_gpu: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> ConvertReport:
    """Quantize routed-expert FFNs to MX-FP6, one ``layer_{L}.moe_fp6.safetensors`` each.

    ``gate_up_order`` describes how the source fused FC1 rows are laid out:
    ``"gate_up"`` (HF default: ``[gate; up]``) or ``"up_gate"``. The kernel needs
    ``[up; gate]``, so the converter reorders accordingly.
    """
    from b12x.quantization.mxfp6 import (
        quantize_moe_weights_to_fp6,
        save_fp6_moe_weights,
    )

    model = SafetensorsModel(model_path)
    scheme = discover_moe_experts(model)
    if scheme is None:
        raise ValueError(f"no MoE experts discovered under {model_path}")

    layers = scheme.layers if limit_layers is None else scheme.layers[:limit_layers]
    report = ConvertReport(arch="moe", layers=list(layers), out_dir=str(out_dir))
    if verbose:
        print(
            f"[moe] experts={scheme.num_experts} layers={len(layers)} "
            f"packed={scheme.packed} fused={scheme.is_fused} "
            f"gate_up_order={gate_up_order} prefix={scheme.prefix_template}"
        )
    if dry_run:
        return report

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_gated = activation == "silu"
    for layer in layers:
        w1, w2 = _build_layer_w1_w2(
            model, scheme, layer, device, is_gated, gate_up_order
        )
        weights = quantize_moe_weights_to_fp6(
            w1, w2, source_format=source_format, activation=activation, use_gpu=use_gpu
        )
        fname = f"layer_{layer}.moe_fp6.safetensors"
        save_fp6_moe_weights(weights, str(out / fname))
        report.tensors_written += 2  # w1 + w2
        report.artifacts.append(
            {"layer": layer, "file": fname, "experts": scheme.num_experts,
             "k": weights.k, "n": weights.n}
        )
        if verbose:
            print(f"  layer {layer}: w1={tuple(w1.shape)} w2={tuple(w2.shape)} -> {fname}")
        del w1, w2, weights
        if device == "cuda":
            torch.cuda.empty_cache()

    _write_manifest(out, {
        # historical on-disk format id; do not rename
        "format": "b12x_fp6_model_v1", "arch": "moe",
        "model_type": model.config.get("model_type"),
        "num_experts": scheme.num_experts, "activation": activation,
        "source_format": source_format, "layers": list(layers),
        "artifacts": report.artifacts,
    })
    return report


def convert_dense_model_to_fp6(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    source_format: str = "mxfp6_w6a8",
    include_attention: bool = True,
    limit_layers: Optional[int] = None,
    device: str = "cuda",
    dry_run: bool = False,
    verbose: bool = True,
) -> ConvertReport:
    """Quantize dense MLP (+ optional attention) linears, one file per layer.

    Writes ``layer_{L}.dense_fp6.safetensors``: tensor fields are stored flat
    as ``{full_key}::{field}`` entries, non-tensor fields as JSON metadata
    under ``{full_key}`` (safetensors only; pickle persistence is not
    supported by project policy). Real quantization requires CUDA (the dense
    quantizer is GPU-only); ``dry_run`` works anywhere.
    """
    import json
    from dataclasses import fields

    from safetensors.torch import save_file

    from b12x.quantization.mxfp6 import quantize_dense_weight_to_fp6

    model = SafetensorsModel(model_path)
    scheme = discover_dense_linears(model)
    if scheme is None:
        raise ValueError(f"no dense linears discovered under {model_path}")

    rels: list[str] = [
        r for r in (scheme.mlp_fused_gate_up, scheme.mlp_gate, scheme.mlp_up, scheme.mlp_down)
        if r
    ]
    if include_attention:
        rels += scheme.attn_projs

    layers = scheme.layers if limit_layers is None else scheme.layers[:limit_layers]
    report = ConvertReport(arch="dense", layers=list(layers), out_dir=str(out_dir))
    if verbose:
        print(f"[dense] layers={len(layers)} rels={rels} attn={include_attention}")
    if dry_run:
        return report
    if device != "cuda":
        raise RuntimeError("dense conversion requires device='cuda' (GPU-only quantizer)")

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        tensors: dict[str, torch.Tensor] = {}
        # historical on-disk format id; do not rename
        metadata: dict[str, str] = {"__format__": "b12x_fp6_dense_layer_v1"}
        linear_keys: list[str] = []
        for rel in rels:
            key = scheme.linear_key(layer, rel)
            if not model.has(key):
                continue
            shape = model.shape_of(key)
            if len(shape) != 2 or shape[0] % _TILE != 0 or shape[1] % _TILE != 0:
                report.skipped.append({"key": key, "reason": f"shape {shape} not %128"})
                continue
            w = model.get_tensor(key).to(device, torch.bfloat16)
            qw = quantize_dense_weight_to_fp6(w, source_format=source_format)
            scalars: dict[str, object] = {}
            # fields()+getattr instead of asdict(): asdict deep-copies every
            # field, cloning the on-GPU packed tensors before the .cpu() move.
            for f2 in fields(qw):
                v = getattr(qw, f2.name)
                if isinstance(v, torch.Tensor):
                    tensors[f"{key}::{f2.name}"] = v.detach().cpu().contiguous()
                else:
                    scalars[f2.name] = v
            metadata[key] = json.dumps(scalars)
            linear_keys.append(key)
            report.tensors_written += 1
            del w, qw
            torch.cuda.empty_cache()
        if tensors:
            fname = f"layer_{layer}.dense_fp6.safetensors"
            save_file(tensors, str(out / fname), metadata=metadata)
            report.artifacts.append(
                {"layer": layer, "file": fname, "tensors": linear_keys}
            )
            if verbose:
                print(f"  layer {layer}: {len(linear_keys)} linears -> {fname}")

    _write_manifest(out, {
        # historical on-disk format id; do not rename
        "format": "b12x_fp6_model_v1", "arch": "dense",
        "model_type": model.config.get("model_type"),
        "source_format": source_format, "include_attention": include_attention,
        "layers": list(layers), "artifacts": report.artifacts,
        "skipped": report.skipped,
    })
    return report
