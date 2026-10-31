"""Cache a topology-specific B12X plain all-reduce routing policy.

This controller runs :mod:`b12x.comm.pcie.plain_allreduce_probe` before a
model server initializes CUDA.  Cache identity includes the selected GPU
order, upstream PCIe target speed, collective environment, NCCL library, and
the source that implements the probe and transport.  A cache record therefore
cannot silently cross a different placement or transport implementation.

The command writes only the validated policy JSON to standard output.  Status
and diagnostics use standard error so a launcher can assign stdout directly to
``B12X_PCIE_PLAIN_ALLREDUCE_POLICY``.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any, Sequence


SCHEMA_VERSION = 1
MIN_FREE_MIB = 2048
POLICY_FIELDS = frozenset(
    {
        "version",
        "world_size",
        "hidden_size",
        "dtype",
        "measured_rows",
        "custom_rows",
    }
)
COLLECTIVE_ENV_PREFIXES = ("NCCL_", "B12X_PCIE_", "VLLM_PCIE_")
FINGERPRINT_ENV_EXCLUSIONS = frozenset(
    {
        "B12X_PCIE_PLAIN_ALLREDUCE_POLICY",
        "B12X_PCIE_PLAIN_ALLREDUCE_CALIBRATION",
        "B12X_PCIE_PLAIN_ALLREDUCE_CALIBRATION_CACHE_DIR",
        "B12X_PCIE_PLAIN_ALLREDUCE_CALIBRATION_TIMEOUT",
    }
)


class CalibrationUnavailable(Exception):
    """The host cannot run a calibration without disturbing other work."""


class CalibrationFailure(Exception):
    """The probe failed after launch and carries its complete diagnostic."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if (
        not values
        or any(item <= 0 for item in values)
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError(
            "expected unique comma-separated positive integers"
        )
    return values


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _file_identity(value: str | None) -> dict[str, Any]:
    if not value:
        return {"path": "", "exists": False}
    path = Path(value.split(":", 1)[0])
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _source_digest() -> str:
    source_names = (
        "plain_allreduce_calibration.py",
        "plain_allreduce_probe.py",
        "pcie_oneshot.py",
        "_oneshot_cute.py",
        "_cute_intrinsics.py",
    )
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    try:
        for name in source_names:
            path = root / name
            digest.update(name.encode())
            digest.update(path.read_bytes())
    except OSError:
        return "unavailable"
    return digest.hexdigest()


def _run_command(command: list[str], *, timeout: float = 30.0) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (
        FileNotFoundError,
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
    ) as exc:
        raise CalibrationUnavailable(f"cannot run {' '.join(command)}: {exc}") from exc
    return completed.stdout


def _normalize_pci_bus_id(value: str) -> str:
    fields = value.strip().lower().split(":")
    if len(fields) == 3 and len(fields[0]) > 4:
        fields[0] = fields[0][-4:]
    return ":".join(fields)


def _upstream_link_contract(pci_bus_id: str) -> dict[str, Any]:
    device_path = Path("/sys/bus/pci/devices") / _normalize_pci_bus_id(pci_bus_id)
    try:
        resolved = device_path.resolve(strict=True)
    except OSError:
        return {"upstream_bdf": "unavailable", "link_control": "unavailable"}
    upstream_bdf = resolved.parent.name
    try:
        output = _run_command(["lspci", "-s", upstream_bdf, "-vv"])
    except CalibrationUnavailable:
        return {"upstream_bdf": upstream_bdf, "link_control": "unavailable"}
    # Link status reflects an idle P-state and must not invalidate the cache.
    # Link capabilities and the configured target speed describe the contract
    # under which the warmed measurement will execute.
    lines = tuple(
        line.strip()
        for line in output.splitlines()
        if any(token in line for token in ("LnkCap:", "LnkCap2:", "LnkCtl2:"))
    )
    return {"upstream_bdf": upstream_bdf, "link_control": lines}


def _nvidia_inventory() -> list[dict[str, Any]]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,pcie.link.gen.max,"
            "pcie.link.width.max,driver_version,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    inventory: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            index = int(fields[0])
            max_generation = int(fields[3])
            max_width = int(fields[4])
            free_mib = int(fields[6])
        except ValueError:
            continue
        pci_bus_id = _normalize_pci_bus_id(fields[2])
        inventory.append(
            {
                "index": index,
                "uuid": fields[1],
                "pci_bus_id": pci_bus_id,
                "max_generation": max_generation,
                "max_width": max_width,
                "driver_version": fields[5],
                "free_mib": free_mib,
                "upstream": _upstream_link_contract(pci_bus_id),
            }
        )
    if not inventory:
        raise CalibrationUnavailable("nvidia-smi returned no usable GPU inventory")
    return inventory


def _select_gpus(
    inventory: Sequence[dict[str, Any]],
    *,
    world_size: int,
    visible_devices: str,
) -> list[dict[str, Any]]:
    by_index = {str(item["index"]): item for item in inventory}
    by_uuid = {str(item["uuid"]): item for item in inventory}
    if visible_devices.strip():
        selectors = tuple(
            item.strip() for item in visible_devices.split(",") if item.strip()
        )
    else:
        selectors = tuple(str(item["index"]) for item in inventory)
    if len(selectors) < world_size:
        raise CalibrationUnavailable(
            f"visible GPU list has {len(selectors)} entries for world size {world_size}"
        )
    selected: list[dict[str, Any]] = []
    for selector in selectors[:world_size]:
        item = by_index.get(selector) or by_uuid.get(selector)
        if item is None:
            raise CalibrationUnavailable(
                f"selected GPU {selector!r} is not in nvidia-smi"
            )
        selected.append(item)
    return selected


def _collective_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(COLLECTIVE_ENV_PREFIXES)
        and key not in FINGERPRINT_ENV_EXCLUSIONS
    }


def build_fingerprint_payload(args: argparse.Namespace) -> dict[str, Any]:
    inventory = _nvidia_inventory()
    selected = _select_gpus(
        inventory,
        world_size=args.world_size,
        visible_devices=args.visible_devices,
    )
    return {
        "schema": SCHEMA_VERSION,
        "operation": {
            "world_size": args.world_size,
            "hidden_size": args.hidden_size,
            "rows": args.rows,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "ops_per_sample": args.ops_per_sample,
            "trials": args.trials,
            "stabilization_replays": args.stabilization_replays,
            "min_absolute_margin_us": args.min_absolute_margin_us,
            "min_relative_margin": args.min_relative_margin,
        },
        "gpu_order": [
            {key: value for key, value in item.items() if key != "free_mib"}
            for item in selected
        ],
        "software": {
            "python": sys.version,
            "torch": _package_version("torch"),
            "b12x": _package_version("b12x"),
            "source_sha256": _source_digest(),
            "nccl": _file_identity(
                os.getenv("VLLM_NCCL_SO_PATH")
                or os.getenv("NCCL_LOCAL_INFERENCE_PATH")
                or os.getenv("LD_PRELOAD")
            ),
            "collective_environment": _collective_environment(),
        },
    }


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_probe_result(
    result: dict[str, Any],
    *,
    world_size: int,
    hidden_size: int,
    rows: tuple[int, ...],
    dtype: str,
) -> str:
    policy = result.get("policy")
    if not isinstance(policy, dict) or set(policy) != POLICY_FIELDS:
        raise ValueError("probe result has no policy matching the versioned schema")
    if int(policy["version"]) != 1:
        raise ValueError("probe result has an unsupported policy version")
    if int(policy["world_size"]) != world_size:
        raise ValueError("probe policy world size differs from the requested world")
    if int(policy["hidden_size"]) != hidden_size:
        raise ValueError("probe policy hidden size differs from the requested tensor")
    if str(policy["dtype"]) != dtype:
        raise ValueError("probe policy dtype differs from the requested tensor")
    measured_rows = tuple(int(value) for value in policy["measured_rows"])
    custom_rows = frozenset(int(value) for value in policy["custom_rows"])
    if frozenset(measured_rows) != frozenset(rows) or len(measured_rows) != len(rows):
        raise ValueError("probe policy does not cover every requested row exactly once")
    if not custom_rows.issubset(measured_rows):
        raise ValueError("probe policy selects an unmeasured row")
    return json.dumps(policy, sort_keys=True, separators=(",", ":"))


def _load_record(
    path: Path,
    *,
    expected_fingerprint: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("schema") != SCHEMA_VERSION
        or record.get("fingerprint") != expected_fingerprint
    ):
        return None
    try:
        policy = validate_probe_result(
            record["probe"],
            world_size=args.world_size,
            hidden_size=args.hidden_size,
            rows=args.rows,
            dtype=args.dtype,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return record, policy


def _terminate_process_group(process: subprocess.Popen[str]) -> str:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        stdout, _ = process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
    return stdout or ""


def _run_probe(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.world_size}",
        "-m",
        "b12x.comm.pcie.plain_allreduce_probe",
        "--hidden-size",
        str(args.hidden_size),
        "--rows",
        ",".join(str(value) for value in args.rows),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--ops-per-sample",
        str(args.ops_per_sample),
        "--trials",
        str(args.trials),
        "--stabilization-replays",
        str(args.stabilization_replays),
        "--min-absolute-margin-us",
        str(args.min_absolute_margin_us),
        "--min-relative-margin",
        str(args.min_relative_margin),
        "--json",
        str(output),
    ]
    environment = os.environ.copy()
    environment.pop("B12X_PCIE_PLAIN_ALLREDUCE_POLICY", None)
    if args.visible_devices.strip():
        environment["CUDA_VISIBLE_DEVICES"] = args.visible_devices
    try:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise CalibrationFailure(f"failed to launch probe: {exc}") from exc
    try:
        stdout, _ = process.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        detail = _terminate_process_group(process)
        raise CalibrationFailure(
            f"probe timed out after {args.timeout:g}s", detail
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    stdout = stdout or ""
    if process.returncode != 0:
        raise CalibrationFailure(f"probe exited with {process.returncode}", stdout)
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationFailure(
            f"probe did not write valid JSON: {exc}", stdout
        ) from exc
    try:
        validate_probe_result(
            result,
            world_size=args.world_size,
            hidden_size=args.hidden_size,
            rows=args.rows,
            dtype=args.dtype,
        )
    except ValueError as exc:
        raise CalibrationFailure(f"probe result rejected: {exc}", stdout) from exc
    return result


def _preflight(selected: Sequence[dict[str, Any]]) -> None:
    busy = [item for item in selected if int(item["free_mib"]) < MIN_FREE_MIB]
    if busy:
        detail = ", ".join(f"GPU{item['index']}:{item['free_mib']}MiB" for item in busy)
        raise CalibrationUnavailable(
            f"selected GPUs do not have {MIN_FREE_MIB} MiB free ({detail})"
        )


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def calibrate(args: argparse.Namespace) -> tuple[str, str, Path]:
    fingerprint_payload = build_fingerprint_payload(args)
    digest = fingerprint(fingerprint_payload)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_dir / f"{digest}.json"
    lock_path = args.cache_dir / f"{digest}.lock"

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not args.force:
            cached = _load_record(
                cache_path,
                expected_fingerprint=digest,
                args=args,
            )
            if cached is not None:
                return "cache-hit", cached[1], cache_path

        inventory = _nvidia_inventory()
        selected = _select_gpus(
            inventory,
            world_size=args.world_size,
            visible_devices=args.visible_devices,
        )
        _preflight(selected)
        with tempfile.TemporaryDirectory(dir=args.cache_dir) as temporary_dir:
            probe = _run_probe(args, Path(temporary_dir) / "probe.json")
        policy = validate_probe_result(
            probe,
            world_size=args.world_size,
            hidden_size=args.hidden_size,
            rows=args.rows,
            dtype=args.dtype,
        )
        record = {
            "schema": SCHEMA_VERSION,
            "fingerprint": digest,
            "fingerprint_payload": fingerprint_payload,
            "probe": probe,
        }
        _atomic_write(cache_path, record)
        return "measured", policy, cache_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--rows", type=_positive_ints, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--visible-devices", default=os.getenv("CUDA_VISIBLE_DEVICES", "")
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=41)
    parser.add_argument("--ops-per-sample", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--stabilization-replays", type=int, default=2048)
    parser.add_argument("--min-absolute-margin-us", type=float, default=0.25)
    parser.add_argument("--min-relative-margin", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.world_size < 2 or args.hidden_size <= 0:
        parser.error("world-size must be at least 2 and hidden-size must be positive")
    if (
        args.timeout <= 0
        or args.warmup < 0
        or args.repeat <= 0
        or args.ops_per_sample <= 0
        or args.trials <= 0
        or args.trials % 2 == 0
        or args.stabilization_replays <= 0
    ):
        parser.error(
            "timeout, repeat, operations, odd trials, and stabilization replays "
            "must be positive; warmup must be non-negative"
        )
    try:
        status, policy, cache_path = calibrate(args)
    except CalibrationUnavailable as exc:
        print(f"B12X plain all-reduce calibration unavailable: {exc}", file=sys.stderr)
        return 3
    except CalibrationFailure as exc:
        print(
            f"B12X plain all-reduce calibration failed: {exc.reason}", file=sys.stderr
        )
        if exc.detail:
            failure_path = args.cache_dir / "last-failure.log"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(exc.detail, encoding="utf-8")
            print(f"Complete probe output: {failure_path}", file=sys.stderr)
        return 4
    print(
        f"B12X plain all-reduce calibration: {status} cache={cache_path}",
        file=sys.stderr,
    )
    print(policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
