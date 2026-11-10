# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Generate kernel descriptors from shared tile, activation and store axes.

Schema v4 stores producer-verified geometry recipes once per topology/store,
then takes their product with the family's activations. Device-dependent values
are resolved during preparation, independently of these logical identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

DEVICE_CONSTANTS = frozenset(
    ("grid_num_clusters", "swizzle_l2_budget_bytes", "ab_stages")
)
GATED = ("swiglu", "geglu", "geglu_tanh", "swiglu_step", "situ")
NON_GATED = ("gelu", "identity", "relu", "relu2", "silu")


def logical_id(raw, dtype):
    """Stable axis identity; code/compiler fingerprints qualify persisted tactics."""
    numerical = {
        k: v
        for k, v in raw["contract"].items()
        if k not in ("s", "n", "k", "experts", "groups")
    }
    if numerical.get("activation") == "silu(gate) * up":
        numerical["activation"] = "swiglu"
    axes = (dtype, raw["arch"], raw["op"], raw["abi"], raw["tactic"], numerical)
    digest = hashlib.sha256(
        json.dumps(axes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"{raw['arch'].replace('_', '').removesuffix('a')}_cudnn_frost_{raw['op']}_{dtype}_{digest}"


def expand_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand the constrained Cartesian product without querying a GPU."""
    for key in (
        "tiles",
        "parameter_bases",
        "store_policies",
        "fragments",
        "activation_groups",
    ):
        if not isinstance(payload.get(key), dict):
            raise RuntimeError(f"Frost dimension catalog requires a '{key}' table")
    if not isinstance(payload.get("families"), list):
        raise RuntimeError("Frost dimension catalog requires families")
    dtype = payload["dtype"]
    records = []
    seen = set()
    for family in payload["families"]:
        activations = payload["activation_groups"][family["activation_group"]]
        if family["stage"] == "fc2":
            activations = ("identity",)
        for variant in family["variants"]:
            policy = payload["store_policies"][variant["store_policy"]]
            base = payload["parameter_bases"][variant["parameter_base"]]
            for activation, tile_id in product(activations, variant["tile_parameters"]):
                if activation not in variant["activation_fragments"]:
                    raise RuntimeError("missing Frost activation definition")
                tile = deepcopy(payload["tiles"][tile_id]) | {"tile": tile_id}
                values = base["defaults"] | variant["tile_parameters"][tile_id]
                constants = [[name, values[name]] for name in base["order"]]
                fragments = [
                    payload["fragments"][r]
                    for r in variant["activation_fragments"][activation]
                ]
                suffix = tile["tile"].split("_", 2)[2]
                parameters = dict(
                    version=1,
                    constants=constants,
                    swap_ab=tile["swap_ab"],
                    kernel_symbol=variant["kernel_prefix"] + suffix,
                    activation_fragments=fragments,
                    device_dependent=True,
                )
                contract = deepcopy(variant["contract"])
                contract["activation"] = activation
                op = ("" if dtype == "bf16" else "block_scale_") + (
                    "grouped_gemm2"
                    if family["stage"] == "fc2"
                    else f"grouped_gemm1_{activation}"
                )
                quantized = contract["output_dtype"] != "bfloat16"
                abi = f"cudnn_frost_{op}{'_quantized' if quantized else ''}{'_swap_ab' if tile['swap_ab'] else ''}"
                aux = [] if family["stage"] == "fc2" else ["scale"]
                if dtype != "bf16":
                    aux = (
                        ["alpha"]
                        if family["stage"] == "fc2"
                        else ["gate_alpha"]
                        + (
                            ["up_alpha"]
                            if family["activation_group"] == "gated"
                            else []
                        )
                        + aux
                    )
                if activation == "situ":
                    aux = aux[:-1] + ["gate_scale", "linear_scale"] + aux[-1:]
                if quantized and dtype == "nvfp4":
                    aux.append("quant_scale")
                tail = (
                    (["output_scale"] if quantized else []) + aux + ["output"]
                    if policy["data"] == "tma"
                    else ["output"] + (["output_scale"] if quantized else []) + aux
                )
                tile.update(
                    store_mode=policy["data"], template=variant["producer_template"]
                )
                raw = dict(
                    op=op,
                    arch=family["arch"],
                    abi=abi,
                    source=deepcopy(variant["source"]) | {"parameters": parameters},
                    workspace_bytes=128,
                    contract=contract,
                    launch={"tail": tail},
                    tactic=tile,
                )
                # BF16 FC2 has no auxiliary scale, even though activation is identity.
                if dtype == "bf16" and family["stage"] == "fc2":
                    raw["launch"] = {"tail": ["output"]}
                raw["id"] = logical_id(raw, dtype)
                if raw["id"] in seen:
                    raise RuntimeError("duplicate Frost dimension combination")
                seen.add(raw["id"])
                records.append(raw)
    return dict(
        schema_version=2,
        producer=deepcopy(payload.get("producer", {})),
        kernels=records,
    )


def _factor_sources(sources):
    """Split frozen sources into a shared body and exact activation fragments."""
    from difflib import SequenceMatcher

    first = sources[min(sources)].splitlines(keepends=True)
    changes = {}
    intervals = []
    for activation, source in sources.items():
        lines = source.splitlines(keepends=True)
        operations = SequenceMatcher(None, first, lines, autojunk=False).get_opcodes()
        changes[activation] = (lines, operations)
        intervals.extend((a, b) for tag, a, b, _, _ in operations if tag != "equal")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    parts, offset = [], 0
    for index, (start, end) in enumerate(merged):
        parts += first[offset:start] + [f"# @@FROST_ACTIVATION_{index}@@\n"]
        offset = end
    parts += first[offset:]
    shared = "".join(parts)
    fragments = {}
    for activation, (lines, operations) in changes.items():
        output = []
        for start, end in merged:
            part = []
            for tag, a, b, c, d in operations:
                if tag == "equal":
                    left, right = max(start, a), min(end, b)
                    if left < right:
                        part += lines[c + left - a : c + right - a]
                elif tag == "insert":
                    if start <= a <= end:
                        part += lines[c:d]
                elif start <= a and b <= end:
                    part += lines[c:d]
            output.append("".join(part))
        reconstructed = shared
        for index, fragment in enumerate(output):
            reconstructed = reconstructed.replace(
                f"# @@FROST_ACTIVATION_{index}@@\n", fragment
            )
        if reconstructed != sources[activation]:
            raise RuntimeError("Frost activation factoring changed the frozen source")
        fragments[activation] = output
    return shared, fragments


def _substitute_activation(reference: str, donor: str, activation: str) -> str:
    """Reuse a frozen activation expression in a previously unexported store arm."""
    import re
    import textwrap

    pattern = r"(?m)^([ \t]*)_c_0_a = .*?^\1(_r_\d+) = [^\n]*\.to\(cutlass.BFloat16\)\n"
    old = re.search(pattern, reference, re.S)
    new = re.search(pattern, donor, re.S)
    if old is None or new is None:
        raise RuntimeError("cannot locate the frozen nongated epilogue")
    body = textwrap.indent(textwrap.dedent(new[0]), old[1])
    tail = re.sub(rf"\b{old[2]}\b", new[2], reference[old.end() :])
    result = reference[: old.start()] + body + tail
    if activation == "situ":
        # SiTU's two extra scalar inputs accompany its expression on every
        # store arm. All other activation interfaces share scale/alpha inputs.
        result = re.sub(
            r"(?m)^([ \t]*)scale: cute.Tensor,",
            r"\1gate_scale: cute.Tensor,\n\1linear_scale: cute.Tensor,\n\1scale: cute.Tensor,",
            result,
        )
        result = re.sub(
            r"(?m)^([ \t]*)scale,$", r"\1gate_scale,\n\1linear_scale,\n\1scale,", result
        )
        loads = re.search(
            r"(?m)^([ \t]*)_aux_gate_scale_ptr = .*?^\1_aux_scale_ptr", donor, re.S
        )
        if loads is None:
            raise RuntimeError("missing frozen SiTU scalar loads")
        load_block = (
            textwrap.dedent(loads[0].rsplit("_aux_scale_ptr", 1)[0].rstrip()) + "\n"
        )
        result = re.sub(
            r"(?m)^([ \t]*)_aux_scale_ptr",
            lambda m: textwrap.indent(load_block, m[1]) + m[0],
            result,
        )
        fake = re.search(
            r"(?m)^([ \t]*)fake_gate_scale = .*?^\1fake_scale", donor, re.S
        )
        if fake is None:
            raise RuntimeError("missing frozen SiTU scalar descriptors")
        fake_block = textwrap.dedent(fake[0].rsplit("fake_scale", 1)[0].rstrip()) + "\n"
        result = re.sub(
            r"(?m)^([ \t]*)fake_scale =",
            lambda m: textwrap.indent(fake_block, m[1]) + m[0],
            result,
        )
        result = result.replace(
            ", fake_scale,", ", fake_gate_scale, fake_linear_scale, fake_scale,"
        )
    return result


def write_catalog(
    path: Path, records: list[dict[str, Any]], dtype: str, *, producer=None
):
    """Factor exporter records into axes; return their logical ID migration map."""
    from collections import Counter, defaultdict
    from .manifest import write_manifest

    root = path.parent
    groups = defaultdict(list)
    tiles: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = deepcopy(raw)
        activation = record["contract"]["activation"]
        if activation == "silu(gate) * up":
            activation = record["contract"]["activation"] = "swiglu"
        stage = "fc2" if record["op"].endswith("gemm2") else "fc1"
        topology = "gated" if stage == "fc1" and activation in GATED else "non_gated"
        tactic = record["tactic"]
        tile = {
            k: deepcopy(v)
            for k, v in tactic.items()
            if k not in ("tile", "template", "store_mode")
        }
        name = tactic["tile"]
        if name in tiles and tiles[name] != tile:
            raise RuntimeError("inconsistent Frost tile geometry")
        tiles[name] = tile
        key = (
            record["arch"],
            stage,
            topology,
            tactic["swap_ab"],
            record["contract"]["output_dtype"],
            tactic["store_mode"],
        )
        groups[key].append(record)

    prepared = []
    schemas: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for key, members in sorted(groups.items()):
        arch, stage, topology, swapped, output, store = key
        sources: dict[str, str] = {}
        prefixes: set[str] = set()
        parameters: dict[str, dict[str, str]] = {}
        contract = deepcopy(members[0]["contract"])
        contract.pop("activation")
        for raw in members:
            source = raw["source"]
            p = source["parameters"]
            template = (root / source["path"]).read_text()
            for index, fragment in enumerate(p.get("activation_fragments", ())):
                template = template.replace(
                    f"# @@FROST_ACTIVATION_{index}@@\n", fragment
                )
            activation = raw["contract"]["activation"]
            if activation in sources and sources[activation] != template:
                raise RuntimeError("multiple Frost source implementations for one axis")
            sources[activation] = template
            constants = dict(p["constants"])
            constants.update({name: "1" for name in DEVICE_CONSTANTS})
            name = raw["tactic"]["tile"]
            if name in parameters and parameters[name] != constants:
                raise RuntimeError("activation-specific Frost tile geometry")
            parameters[name] = constants
            # Symbols are generated labels, independent of numerical semantics.
            # Older exports appended an STG suffix; use one canonical spelling.
            prefixes.add("frost_" + raw["tactic"]["template"].removesuffix(".py") + "_")
            if {
                k: v for k, v in raw["contract"].items() if k != "activation"
            } != contract:
                raise RuntimeError("inconsistent Frost family numerical contract")
        if len(prefixes) != 1:
            raise RuntimeError("inconsistent Frost kernel symbol prefix")
        expected = (
            ("identity",)
            if stage == "fc2"
            else GATED
            if topology == "gated"
            else NON_GATED
        )
        if set(sources) != set(expected):
            if stage != "fc1":
                raise RuntimeError(
                    f"missing activation source for {key}: {set(expected) - sources.keys()}"
                )
            # SiTU has two additional scalar arguments. Start from a common
            # activation ABI when synthesizing other activation/store arms.
            ordinary = sources.keys() - {"situ"}
            if not ordinary:
                raise RuntimeError(
                    "Frost activation expansion needs a common ABI source"
                )
            reference = sources[min(ordinary)]
            for activation in set(expected) - sources.keys():
                donors = [
                    r
                    for other_key, rows in groups.items()
                    if other_key[:3] == key[:3] and other_key[4] == output
                    for r in rows
                    if r["contract"]["activation"] == activation
                ]
                donor = min(donors, key=lambda r: r["tactic"]["swap_ab"] != swapped)
                donor_template = (root / donor["source"]["path"]).read_text()
                for index, fragment in enumerate(
                    donor["source"]["parameters"].get("activation_fragments", ())
                ):
                    donor_template = donor_template.replace(
                        f"# @@FROST_ACTIVATION_{index}@@\n", fragment
                    )
                sources[activation] = _substitute_activation(
                    reference, donor_template, activation
                )
        schema = tuple(sorted(next(iter(parameters.values()))))
        if any(tuple(sorted(p)) != schema for p in parameters.values()):
            raise RuntimeError("inconsistent Frost parameter schema")
        schemas[schema].extend(parameters.values())
        prepared.append(
            (
                key,
                sources,
                prefixes.pop(),
                parameters,
                contract,
                schema,
                members[0]["tactic"]["template"],
            )
        )

    payload: dict[str, Any] = dict(
        schema_version=4,
        producer={
            "name": "cudnn_frost",
            "revisions": sorted(
                set((producer or {}).get("revisions", ()))
                | {
                    revision
                    for r in records
                    for revision in (
                        *r.get("producer_revisions", ()),
                        *([r["producer_revision"]] if "producer_revision" in r else []),
                    )
                }
            ),
        },
        dtype=dtype,
        activation_groups={"gated": list(GATED), "non_gated": list(NON_GATED)},
        store_policies={
            "stg": {"data": "stg", "scales": "stg"},
            "tma": {"data": "tma", "scales": None},
            "tmastg": {"data": "tma", "scales": "stg"},
        },
        tiles=tiles,
        parameter_bases={},
        fragments={},
        families=[],
    )
    bases = {}
    for index, (schema, values) in enumerate(sorted(schemas.items())):
        name = f"base_{index}"
        defaults = {
            key: Counter(v[key] for v in values).most_common(1)[0][0] for key in schema
        }
        payload["parameter_bases"][name] = dict(order=list(schema), defaults=defaults)
        bases[schema] = (name, defaults)
    families = {}
    pending_sources = []
    for (
        key,
        sources,
        prefix,
        parameters,
        contract,
        schema,
        producer_template,
    ) in prepared:
        arch, stage, topology, swapped, output, store = key
        family_key = (arch, stage, topology)
        if family_key not in families:
            families[family_key] = dict(
                arch=arch, stage=stage, activation_group=topology, variants=[]
            )
        quantized = output != "bfloat16"
        policy = "tmastg" if store == "tma" and quantized else store
        tag = f"{arch.replace('_', '').removesuffix('a')}_{stage}_{topology}_{'swap' if swapped else 'normal'}_{'quantized' if quantized else 'bf16'}_{policy}"
        template, fragments = _factor_sources(sources)
        relative = f"sources/{tag}.py"
        destination = root / relative
        pending_sources.append((destination, template))
        refs: dict[str, list[str]] = {}
        for activation, chunks in fragments.items():
            refs[activation] = []
            for chunk in chunks:
                ref = hashlib.sha256(chunk.encode()).hexdigest()[:16]
                if ref in payload["fragments"] and payload["fragments"][ref] != chunk:
                    raise RuntimeError("Frost fragment reference collision")
                payload["fragments"][ref] = chunk
                refs[activation].append(ref)
        base, defaults = bases[schema]
        families[family_key]["variants"].append(
            dict(
                store_policy=policy,
                source={
                    "path": relative,
                    "sha256": hashlib.sha256(template.encode()).hexdigest(),
                },
                activation_fragments=refs,
                parameter_base=base,
                tile_parameters={
                    tile: {
                        key: value for key, value in p.items() if value != defaults[key]
                    }
                    for tile, p in sorted(parameters.items())
                },
                kernel_prefix=prefix,
                contract=contract,
                producer_template=producer_template,
            )
        )
    payload["families"] = list(families.values())
    expanded = expand_catalog(payload)
    new_ids = {logical_id(r, dtype): r["id"] for r in expanded["kernels"]}
    mapping = {r["id"]: new_ids[logical_id(r, dtype)] for r in records}
    for destination, template in pending_sources:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix=destination.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as file:
                file.write(template)
            os.chmod(
                temporary,
                destination.stat().st_mode & 0o777 if destination.exists() else 0o644,
            )
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    write_manifest(path, payload)
    return mapping
