"""Emit a full HF safetensors checkpoint quantized to b12x MX-FP6 (W6A6).

Unlike the per-layer ``.pt`` converters in
:mod:`b12x.quantization.mxfp6.model_fp6` (useful for kernel validation),
this writes a *complete, loadable* model directory: sharded
``model-*.safetensors`` + ``model.safetensors.index.json`` + a ``config.json``
patched with a ModelOpt-mirror ``quantization_config``, plus all the original
auxiliary files (tokenizer, generation config, ...).

Quantized linears follow the :mod:`b12x.quantization.mxfp6.fp6_checkpoint`
schema (``.weight`` / ``.weight_scale`` / ``.weight_scale_2`` / ``.input_scale``).
Everything else (norms, embeddings, router gates, shared experts, attention for
MoE, SSM/linear-attention projections, lm_head, MTP heads) is copied through in
its original dtype and recorded in ``quantization_config.exclude_modules``.

* MoE: routed experts are written as **per-expert** keys
  (``...experts.{e}.gate_proj.weight`` + scales), mirroring ModelOpt NVFP4 and the
  b12x FP4 loader, regardless of whether the source stored them packed.
* Dense: MLP (+ optional attention) linears are quantized in place.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch

from .fp6_checkpoint import (
    QUANT_ALGO,
    build_quantization_config,
    dequantize_linear_from_fp6,
    quantize_linear_to_fp6,
)
from .model_fp6 import (
    SafetensorsModel,
    _split_gate_up,
    discover_moe_experts,
)

_TILE = 128
_GATE_PROJ = "gate_proj"
_UP_PROJ = "up_proj"
_DOWN_PROJ = "down_proj"

_LAYER_IDX_RE = re.compile(r"\.layers\.\d+\.")
_EXPERT_IDX_RE = re.compile(r"\.experts\.\d+\.")

# Dense quantization follows the LLM-Compressor "golden rule": quantize every 2-D
# Linear weight EXCEPT this ignore list (mirrors targets="Linear", ignore=[...]).
# ``lm_head`` is always skipped; ``embed``/norms/conv/rotary are not Linear matmuls;
# ``linear_attn`` / ``visual`` are skipped by default (opt back in via flags).
_DENSE_IGNORE_ALWAYS = (
    r"(?:^|\.)lm_head\.",
    r"(?:^|\.)(?:embed_tokens|embeddings?|word_embeddings|wte)\.",
    # Positional / patch embedding *tables* are 2-D but are NOT Linear matmuls
    # (vLLM builds them as plain Parameters / nn.Embedding, with no input_scale),
    # so they must stay un-quantized even when --include-vision strips ``.visual.``.
    r"(?:^|\.)pos_embed",
    r"(?:^|\.)position_embeddings?\.",
    r"(?:^|\.)patch_embed\.",
    r"(?:^|\.)mtp(?:\.|_)",
    r"\.mlp\.gate\.weight$",            # MoE router gate (NOT mlp.gate_proj)
    r"\.shared_expert_gate\.",
    r"\.(?:input_layernorm|post_attention_layernorm|q_norm|k_norm|norm|layernorm|ln_?\w*)\.",
    r"\.rotary",
    r"\.conv\d?d?\.",
)
_RE_LINEAR_ATTN = r"\.linear_attn\."
_RE_VISION = r"\.visual\."
_RE_ATTENTION = r"\.self_attn\."


def _dense_ignore_re(
    *,
    include_attention: bool,
    include_linear_attn: bool,
    include_vision: bool,
    extra_ignores: tuple[str, ...] = (),
) -> re.Pattern:
    """Build the dense ignore regex for the requested scope (golden-rule mirror)."""
    parts = list(_DENSE_IGNORE_ALWAYS) + list(extra_ignores)
    if not include_linear_attn:
        parts.append(_RE_LINEAR_ATTN)
    if not include_vision:
        parts.append(_RE_VISION)
    if not include_attention:
        parts.append(_RE_ATTENTION)
    return re.compile("(" + "|".join(parts) + ")", re.IGNORECASE)


def _module_pattern(key: str) -> str:
    """Collapse a tensor key to a layer/expert-agnostic module glob pattern."""
    k = key
    for suffix in (".weight", ".bias", ".weight_scale", ".weight_scale_2", ".input_scale"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
            break
    k = _LAYER_IDX_RE.sub(".layers.*.", k)
    k = _EXPERT_IDX_RE.sub(".experts.*.", k)
    return k


@dataclass
class ExportReport:
    arch: str
    out_dir: str
    layers: list[int] = field(default_factory=list)
    quantized_tensors: int = 0
    copied_tensors: int = 0
    shards: int = 0
    total_bytes: int = 0
    skipped: list = field(default_factory=list)
    error_groups: dict = field(default_factory=dict)


def _error_group(name: str) -> str:
    """Collapse a module name to a sensitivity group for error aggregation."""
    proj = name.rsplit(".", 1)[-1]
    if _EXPERT_IDX_RE.search(name + "."):
        return f"experts.{proj}"
    for tag in ("shared_expert", "self_attn", "linear_attn", "visual", "mlp"):
        if f".{tag}." in f".{name}.":
            return f"{tag}.{proj}"
    return proj


class _ErrorStats:
    """Per-group quantization error accumulator (relative Frobenius RMSE).

    Group error is ``sqrt(sum ||W - W_hat||^2 / sum ||W||^2)`` over the group's
    tensors (energy-weighted, so large tensors dominate the way they dominate
    the forward pass), plus the single worst tensor by its own relative error.
    """

    def __init__(self) -> None:
        self._acc: dict[str, dict] = {}

    def add(self, name: str, w: torch.Tensor, w_hat: torch.Tensor) -> None:
        err_sq = (w_hat - w.float()).pow(2).sum().item()
        ref_sq = w.float().pow(2).sum().item()
        rel = (err_sq / ref_sq) ** 0.5 if ref_sq > 0 else 0.0
        g = self._acc.setdefault(
            _error_group(name),
            {"tensors": 0, "err_sq": 0.0, "ref_sq": 0.0, "worst": 0.0, "worst_key": ""},
        )
        g["tensors"] += 1
        g["err_sq"] += err_sq
        g["ref_sq"] += ref_sq
        if rel > g["worst"]:
            g["worst"] = rel
            g["worst_key"] = name

    def summary(self) -> dict[str, dict]:
        out = {}
        for name, g in self._acc.items():
            rel = (g["err_sq"] / g["ref_sq"]) ** 0.5 if g["ref_sq"] > 0 else 0.0
            out[name] = {
                "tensors": g["tensors"],
                "rel_rmse": rel,
                "worst": g["worst"],
                "worst_key": g["worst_key"],
            }
        return out

    def print_table(self) -> None:
        rows = sorted(
            self.summary().items(), key=lambda kv: kv[1]["rel_rmse"], reverse=True
        )
        print(
            f"[fp6-error] {'group':<28}{'tensors':>8}  "
            f"{'rel-RMSE':>9}  {'worst':>9}  worst tensor"
        )
        for name, s in rows:
            print(
                f"[fp6-error] {name:<28}{s['tensors']:>8}  "
                f"{s['rel_rmse']:>9.5f}  {s['worst']:>9.5f}  {s['worst_key']}"
            )


class _ShardWriter:
    """Stream tensors to size-capped safetensors shards, then finalize the index."""

    def __init__(self, out_dir: pathlib.Path, *, max_shard_bytes: int):
        from safetensors.torch import save_file

        self._save_file = save_file
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self._buf: dict[str, torch.Tensor] = {}
        self._buf_bytes = 0
        self._shard_keys: list[list[str]] = []  # provisional shard idx -> keys
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        t = tensor.detach().cpu().contiguous()
        nbytes = t.numel() * t.element_size()
        if self._buf_bytes and self._buf_bytes + nbytes > self.max_shard_bytes:
            self._flush()
        self._buf[key] = t
        self._buf_bytes += nbytes
        self.total_bytes += nbytes

    def add_many(self, tensors: dict[str, torch.Tensor]) -> None:
        for key, tensor in tensors.items():
            self.add(key, tensor)

    def _flush(self) -> None:
        if not self._buf:
            return
        idx = len(self._shard_keys)
        name = f"model-{idx:05d}.safetensors"
        self._save_file(self._buf, str(self.out_dir / name), metadata={"format": "pt"})
        self._shard_keys.append(list(self._buf.keys()))
        self._buf = {}
        self._buf_bytes = 0

    def finalize(self) -> int:
        """Flush, rename shards to HF ``-of-N`` form, write the index. Returns shard count."""
        self._flush()
        n = len(self._shard_keys)
        for idx, keys in enumerate(self._shard_keys):
            old = self.out_dir / f"model-{idx:05d}.safetensors"
            final = (
                "model.safetensors"
                if n == 1
                else f"model-{idx + 1:05d}-of-{n:05d}.safetensors"
            )
            (self.out_dir / final).unlink(missing_ok=True)
            old.rename(self.out_dir / final)
            for key in keys:
                self.weight_map[key] = final
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": self.weight_map,
        }
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2)
        )
        return n


def _copy_aux_files(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Copy tokenizer / generation / misc files so the output dir is fully loadable."""
    skip_suffixes = (".safetensors",)
    skip_names = {"model.safetensors.index.json", "config.json"}
    for item in src.iterdir():
        if not item.is_file():
            continue
        if item.name in skip_names or item.suffix in skip_suffixes:
            continue
        if item.name.endswith(".safetensors.index.json"):
            continue
        shutil.copy2(item, dst / item.name)


def _write_config(
    src_config: dict, out_dir: pathlib.Path, quant_config: dict
) -> None:
    cfg = dict(src_config)
    cfg["quantization_config"] = quant_config
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def _emit_quantized_linear(
    writer: _ShardWriter,
    report: ExportReport,
    name: str,
    weight_bf16: torch.Tensor,
    *,
    source_format: str,
    use_gpu: bool,
    block_scale_rule: str = "mse",
    error_stats: Optional[_ErrorStats] = None,
) -> bool:
    """Quantize one ``(out, in)`` weight and stream its four FP6 keys.

    Returns ``True`` if the weight was quantized, ``False`` if it was copied
    through as BF16 (non-2-D, or a dimension the quantizer can't handle). The
    caller must record copied weights in ``exclude_modules``.
    """
    # Safety net for opt-in extra scope: anything that isn't a 2-D matmul
    # (conv1d, embeddings, ...) is copied through untouched.
    if weight_bf16.ndim != 2:
        report.skipped.append(
            {"key": name + ".weight", "reason": f"ndim {weight_bf16.ndim} not 2-D"}
        )
        writer.add(name + ".weight", weight_bf16)
        report.copied_tensors += 1
        return False
    out_f, in_f = weight_bf16.shape
    # MSE joint encode is torch-only (no TMA tile constraint). Ceil+GPU needs
    # out/in multiples of 128; ceil torch / mse only need in % 32 and in % 4.
    needs_tma_tile = use_gpu and block_scale_rule == "ceil"
    quantizable = (
        (out_f % _TILE == 0 and in_f % _TILE == 0)
        if needs_tma_tile
        else (in_f % 32 == 0 and in_f % 4 == 0)
    )
    if not quantizable:
        report.skipped.append(
            {"key": name + ".weight", "reason": f"shape {(out_f, in_f)} unquantizable"}
        )
        writer.add(name + ".weight", weight_bf16)
        report.copied_tensors += 1
        return False
    qt = quantize_linear_to_fp6(
        weight_bf16,
        source_format=source_format,
        use_gpu=use_gpu,
        block_scale_rule=block_scale_rule,
    )
    if error_stats is not None:
        dev = weight_bf16.device
        w_hat = dequantize_linear_from_fp6(
            qt.weight.to(dev),
            qt.weight_scale.to(dev),
            fmt=qt.fmt,
            weight_scale_2=qt.weight_scale_2,
        )
        error_stats.add(name, weight_bf16, w_hat)
        del w_hat
    writer.add_many(qt.to_state_dict(name))
    report.quantized_tensors += 1
    return True


def export_moe_model_to_fp6_safetensors(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    source_format: str = "mxfp6_w6a8",
    activation: str = "silu",
    gate_up_order: str = "gate_up",
    include_attention: bool = True,
    include_linear_attn: bool = False,
    include_vision: bool = False,
    skip_experts: bool = False,
    skip_shared_expert: bool = False,
    report_error: bool = False,
    limit_layers: Optional[int] = None,
    device: str = "cuda",
    use_gpu: bool = True,
    block_scale_rule: str = "mse",
    max_shard_bytes: int = 4 * 1024**3,
    dry_run: bool = False,
    verbose: bool = True,
) -> ExportReport:
    """Quantize routed experts to per-expert FP6 keys + the non-expert Linears.

    Routed experts are always quantized. In addition, the same golden-rule walk
    used by the dense exporter is applied to every *non-expert* 2-D Linear weight
    (attention, shared experts, ...): quantize unless it matches the ignore set.
    ``include_linear_attn`` / ``include_vision`` opt the linear-attention / vision
    projections into FP6 too (off by default). Leaving the large ``linear_attn``
    block in BF16 keeps it at 16-bit -- heavier than the FP8 build's 8-bit -- which
    is why an experts-only FP6 MoE does not shrink below FP8; enable the flag to
    cross under it. Quality trade-off: validate downstream.

    ``skip_experts`` / ``skip_shared_expert`` are sensitivity-sweep knobs: they
    copy that group through in its original dtype (and record it in
    ``exclude_modules`` so serving BF16-falls-back) so a KLD run isolates the
    contribution of the remaining quantized groups. Not for production exports.
    ``report_error`` dequantizes every emitted tensor and prints a per-group
    relative-RMSE table (also returned on ``report.error_groups``).
    """
    model = SafetensorsModel(model_path)
    scheme = discover_moe_experts(model)
    if scheme is None:
        raise ValueError(f"no MoE experts discovered under {model_path}")
    is_gated = activation == "silu"

    layers = scheme.layers if limit_layers is None else scheme.layers[:limit_layers]
    report = ExportReport(arch="moe", out_dir=str(out_dir), layers=list(layers))
    if verbose:
        print(
            f"[moe-export] experts={scheme.num_experts} layers={len(layers)} "
            f"packed={scheme.packed} fused={scheme.is_fused} "
            f"gate_up_order={gate_up_order} src={source_format}"
        )

    # Source keys the per-expert FP6 emission replaces (dropped from the copy
    # pass). With skip_experts the source expert tensors are copied through
    # unmodified instead (the copy pass records their exclude patterns).
    drop_keys: set[str] = set()
    for layer in layers if not skip_experts else []:
        if scheme.packed:
            drop_keys.add(scheme.packed_key(layer, scheme.gate_name))
            if not scheme.is_fused:
                drop_keys.add(scheme.packed_key(layer, scheme.up_name))
            drop_keys.add(scheme.packed_key(layer, scheme.down_name))
        else:
            for e in range(scheme.num_experts):
                drop_keys.add(scheme.expert_key(layer, e, scheme.gate_name))
                if not scheme.is_fused:
                    drop_keys.add(scheme.expert_key(layer, e, scheme.up_name))
                drop_keys.add(scheme.expert_key(layer, e, scheme.down_name))

    # Golden-rule selection for the NON-expert weights (attention, shared experts,
    # optionally linear-attn / vision). Routed experts are handled by the dedicated
    # per-expert pass below, so they are excluded here.
    ignore_re = _dense_ignore_re(
        include_attention=include_attention,
        include_linear_attn=include_linear_attn,
        include_vision=include_vision,
        extra_ignores=(r"\.shared_expert\.",) if skip_shared_expert else (),
    )
    quant_keys: set[str] = set()
    for key in model.keys():
        if key in drop_keys or _EXPERT_IDX_RE.search(key):
            continue
        if not key.endswith(".weight") or ignore_re.search(key):
            continue
        shape = tuple(model.shape_of(key))
        if len(shape) != 2:
            continue
        out_f, in_f = shape
        needs_tma_tile = use_gpu and block_scale_rule == "ceil"
        ok = (
            (out_f % _TILE == 0 and in_f % _TILE == 0)
            if needs_tma_tile
            else (in_f % 32 == 0 and in_f % 4 == 0)
        )
        if ok:
            quant_keys.add(key)

    expert_quant = (
        0 if skip_experts
        else len(layers) * scheme.num_experts * (3 if is_gated else 2)
    )
    if verbose:
        print(
            f"[moe-export] non-expert quantizable Linear weights={len(quant_keys)} "
            f"(attention={include_attention} linear_attn={include_linear_attn} "
            f"vision={include_vision} skip_experts={skip_experts} "
            f"skip_shared_expert={skip_shared_expert} "
            f"block_scale_rule={block_scale_rule})"
        )

    if dry_run:
        report.quantized_tensors = expert_quant + len(quant_keys)
        return report

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(out, max_shard_bytes=max_shard_bytes)
    exclude_patterns: set[str] = set()
    error_stats = _ErrorStats() if report_error else None

    # 1) Walk non-expert tensors: quantize golden-rule Linears, copy the rest BF16.
    for key in model.keys():
        if key in drop_keys:
            continue
        if key in quant_keys:
            name = key[: -len(".weight")]
            w = model.get_tensor(key).to(device, torch.bfloat16)
            quantized = _emit_quantized_linear(
                writer, report, name, w, source_format=source_format,
                use_gpu=use_gpu, block_scale_rule=block_scale_rule,
                error_stats=error_stats,
            )
            if not quantized:
                exclude_patterns.add(_module_pattern(key))
            del w
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            writer.add(key, model.get_tensor(key))
            report.copied_tensors += 1
            exclude_patterns.add(_module_pattern(key))

    # 2) Emit per-expert FP6 keys for the quantized layers.
    for layer in layers if not skip_experts else []:
        gate_e, up_e, down_e = _layer_expert_matrices(
            model, scheme, layer, device, is_gated, gate_up_order
        )
        prefix = scheme.prefix_template.format(L=layer)
        for e in range(scheme.num_experts):
            _emit_quantized_linear(
                writer, report, f"{prefix}.experts.{e}.{_DOWN_PROJ}",
                down_e[e], source_format=source_format, use_gpu=use_gpu,
                block_scale_rule=block_scale_rule, error_stats=error_stats,
            )
            _emit_quantized_linear(
                writer, report, f"{prefix}.experts.{e}.{_GATE_PROJ}",
                gate_e[e], source_format=source_format, use_gpu=use_gpu,
                block_scale_rule=block_scale_rule, error_stats=error_stats,
            )
            if is_gated:
                _emit_quantized_linear(
                    writer, report, f"{prefix}.experts.{e}.{_UP_PROJ}",
                    up_e[e], source_format=source_format, use_gpu=use_gpu,
                    block_scale_rule=block_scale_rule, error_stats=error_stats,
                )
        if verbose:
            print(f"  layer {layer}: {scheme.num_experts} experts -> FP6")
        del gate_e, up_e, down_e
        if device == "cuda":
            torch.cuda.empty_cache()

    report.shards = writer.finalize()
    report.total_bytes = writer.total_bytes
    quant_config = build_quantization_config(
        source_format=source_format,
        exclude_modules=sorted(exclude_patterns),
        block_scale_rule=block_scale_rule,
    )
    _write_config(model.config, out, quant_config)
    _copy_aux_files(pathlib.Path(model_path), out)
    if error_stats is not None:
        report.error_groups = error_stats.summary()
        if verbose:
            error_stats.print_table()
    if verbose:
        print(
            f"[moe-export] done: quantized={report.quantized_tensors} "
            f"copied={report.copied_tensors} shards={report.shards} -> {out}"
        )
    return report


def _layer_expert_matrices(
    model: SafetensorsModel,
    scheme,
    layer: int,
    device: str,
    is_gated: bool,
    gate_up_order: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-expert ``(gate (E,N,K), up (E,N,K), down (E,K,N))`` bf16 matrices.

    ``gate``/``up`` are in ModelOpt linear orientation (out=N, in=K); ``down`` is
    (out=K, in=N). For non-gated activations ``up`` is returned but unused.
    """
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
        return gate.contiguous(), up.contiguous(), down.contiguous()

    gate_list, up_list, down_list = [], [], []
    for e in range(scheme.num_experts):
        down_list.append(
            model.get_tensor(scheme.expert_key(layer, e, scheme.down_name)).to(
                device, torch.bfloat16
            )
        )
        if scheme.is_fused:
            gate_up = model.get_tensor(scheme.expert_key(layer, e, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            g, u = _split_gate_up(gate_up, dim=0, order=gate_up_order)
        else:
            g = model.get_tensor(scheme.expert_key(layer, e, scheme.gate_name)).to(
                device, torch.bfloat16
            )
            u = model.get_tensor(scheme.expert_key(layer, e, scheme.up_name)).to(
                device, torch.bfloat16
            )
        gate_list.append(g)
        up_list.append(u)
    gate = torch.stack(gate_list, dim=0)
    up = torch.stack(up_list, dim=0) if is_gated else gate
    down = torch.stack(down_list, dim=0)
    return gate, up, down


def export_dense_model_to_fp6_safetensors(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    source_format: str = "mxfp6_w6a8",
    include_attention: bool = True,
    include_linear_attn: bool = False,
    include_vision: bool = False,
    report_error: bool = False,
    limit_layers: Optional[int] = None,
    device: str = "cuda",
    use_gpu: bool = True,
    block_scale_rule: str = "mse",
    max_shard_bytes: int = 4 * 1024**3,
    dry_run: bool = False,
    verbose: bool = True,
) -> ExportReport:
    """Quantize dense MLP (+ optional attention) linears; copy everything else through.

    ``include_linear_attn`` / ``include_vision`` opt the 2-D matmul projections of
    linear-attention layers / the vision tower into FP6 too (off by default). These
    are normally the dominant BF16 residue in multimodal-hybrid checkpoints, so
    enabling them is what brings the FP6 size below the FP8 equivalent. They are a
    quality trade-off (FP6 is lower precision than the FP8 those layers usually
    ship in) — validate downstream before relying on them.
    """
    model = SafetensorsModel(model_path)

    # Golden-rule selection (mirrors LLM-Compressor targets="Linear", ignore=[...]):
    # quantize every 2-D ``.weight`` whose dims the quantizer can handle and that is
    # not in the ignore set. No reliance on hardcoded module names, so it adapts to
    # any architecture ("insert any model").
    ignore_re = _dense_ignore_re(
        include_attention=include_attention,
        include_linear_attn=include_linear_attn,
        include_vision=include_vision,
    )
    layer_idx_re = re.compile(r"\.layers\.(\d+)\.")

    quant_keys: set[str] = set()
    for key in model.keys():
        if not key.endswith(".weight") or ignore_re.search(key):
            continue
        shape = tuple(model.shape_of(key))
        if len(shape) != 2:
            continue
        out_f, in_f = shape
        needs_tma_tile = use_gpu and block_scale_rule == "ceil"
        ok = (
            (out_f % _TILE == 0 and in_f % _TILE == 0)
            if needs_tma_tile
            else (in_f % 32 == 0 and in_f % 4 == 0)
        )
        if not ok:
            continue
        if limit_layers is not None:
            lm = layer_idx_re.search(key)
            if lm and int(lm.group(1)) >= limit_layers:
                continue
        quant_keys.add(key)

    layers = sorted(
        {int(m.group(1)) for k in quant_keys if (m := layer_idx_re.search(k))}
    )
    report = ExportReport(arch="dense", out_dir=str(out_dir), layers=list(layers))

    if verbose:
        print(
            f"[dense-export] quantizable Linear weights={len(quant_keys)} "
            f"layers={len(layers)} (attention={include_attention} "
            f"linear_attn={include_linear_attn} vision={include_vision}) "
            f"src={source_format} block_scale_rule={block_scale_rule}"
        )

    if dry_run:
        report.quantized_tensors = len(quant_keys)
        return report

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(out, max_shard_bytes=max_shard_bytes)
    exclude_patterns: set[str] = set()
    error_stats = _ErrorStats() if report_error else None

    for key in model.keys():
        if key in quant_keys:
            name = key[: -len(".weight")]
            w = model.get_tensor(key).to(device, torch.bfloat16)
            quantized = _emit_quantized_linear(
                writer, report, name, w, source_format=source_format,
                use_gpu=use_gpu, block_scale_rule=block_scale_rule,
                error_stats=error_stats,
            )
            if not quantized:
                # Copied through as BF16 -> must be excluded from the quant config.
                exclude_patterns.add(_module_pattern(key))
            del w
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            writer.add(key, model.get_tensor(key))
            report.copied_tensors += 1
            exclude_patterns.add(_module_pattern(key))

    report.shards = writer.finalize()
    report.total_bytes = writer.total_bytes
    quant_config = build_quantization_config(
        source_format=source_format,
        exclude_modules=sorted(exclude_patterns),
        block_scale_rule=block_scale_rule,
    )
    _write_config(model.config, out, quant_config)
    _copy_aux_files(pathlib.Path(model_path), out)
    if error_stats is not None:
        report.error_groups = error_stats.summary()
        if verbose:
            error_stats.print_table()
    if verbose:
        print(
            f"[dense-export] done: quantized={report.quantized_tensors} "
            f"copied={report.copied_tensors} shards={report.shards} -> {out}"
        )
    return report


def dequantize_fp6_checkpoint_to_bf16(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    device: str = "cuda",
    max_shard_bytes: int = 4 * 1024**3,
    verbose: bool = True,
) -> ExportReport:
    """Decode an FP6 checkpoint back to a plain BF16 HF checkpoint.

    Inverse of the FP6 exporters: every quantized linear (the four-key
    ``.weight``/``.weight_scale``/``.weight_scale_2``/``.input_scale`` set) is
    dequantized to a single BF16 ``.weight``; everything else is copied through.
    The output runs on stock vLLM with no b12x involvement, so a KLD
    against the original BF16 model isolates *weight* quantization error from
    the runtime W6A6 *activation* quantization error.
    """
    model = SafetensorsModel(model_path)
    qcfg = model.config.get("quantization_config") or {}
    if qcfg.get("quant_algo") != QUANT_ALGO:
        raise ValueError(
            f"{model_path} is not a b12x FP6 checkpoint "
            f"(quant_algo={qcfg.get('quant_algo')!r})"
        )
    fmt = qcfg.get("weight_format", "e2m3")

    quantized = {
        k[: -len(".weight_scale")]
        for k in model.keys()
        if k.endswith(".weight_scale") and not k.endswith(".weight_scale_2")
    }
    report = ExportReport(arch="dequant-bf16", out_dir=str(out_dir))
    if verbose:
        print(
            f"[fp6-dequant] quantized linears={len(quantized)} fmt={fmt} "
            f"-> BF16 {out_dir}"
        )

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = _ShardWriter(out, max_shard_bytes=max_shard_bytes)
    drop_suffixes = (".weight_scale", ".weight_scale_2", ".input_scale")

    for key in model.keys():
        base = key
        for suffix in drop_suffixes:
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                break
        if base != key and base in quantized:
            continue  # consumed by the dequantized .weight emission
        if key.endswith(".weight") and key[: -len(".weight")] in quantized:
            name = key[: -len(".weight")]
            packed = model.get_tensor(key).to(device)
            scale = model.get_tensor(name + ".weight_scale").to(device)
            ws2_key = name + ".weight_scale_2"
            ws2 = model.get_tensor(ws2_key).to(device) if model.has(ws2_key) else None
            w = dequantize_linear_from_fp6(
                packed, scale, fmt=fmt, weight_scale_2=ws2
            )
            writer.add(key, w.to(torch.bfloat16))
            report.quantized_tensors += 1
            del packed, scale, w
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            writer.add(key, model.get_tensor(key))
            report.copied_tensors += 1

    report.shards = writer.finalize()
    report.total_bytes = writer.total_bytes
    cfg = dict(model.config)
    cfg.pop("quantization_config", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    _copy_aux_files(pathlib.Path(model_path), out)
    if verbose:
        print(
            f"[fp6-dequant] done: dequantized={report.quantized_tensors} "
            f"copied={report.copied_tensors} shards={report.shards} -> {out}"
        )
    return report


def export_model_to_fp6_safetensors(
    model_path: str | pathlib.Path,
    out_dir: str | pathlib.Path,
    *,
    arch: str = "auto",
    **kwargs,
) -> ExportReport:
    """Dispatch to the MoE or dense exporter (``arch`` ``"auto"``/``"moe"``/``"dense"``)."""
    if arch == "auto":
        model = SafetensorsModel(model_path)
        arch = "moe" if discover_moe_experts(model) is not None else "dense"
    if arch == "moe":
        return export_moe_model_to_fp6_safetensors(model_path, out_dir, **kwargs)
    if arch == "dense":
        return export_dense_model_to_fp6_safetensors(model_path, out_dir, **kwargs)
    raise ValueError(f"unknown arch {arch!r} (expected auto/moe/dense)")
