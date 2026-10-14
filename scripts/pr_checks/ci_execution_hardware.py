"""Bounded pre-test inventory inside the actual CI container; no usage claims.

The workflow mounts this trusted helper read-only. The parent writer accepts its
first context-bound frame and streams subsequent test output only as opaque logs.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile

MAGIC = b"WATCHER_CI_HARDWARE_V1:"
LIMIT = 65536


def unavailable(reason):
    return {
        "status": "unavailable",
        "source": "container-cuda-driver",
        "reason": reason,
        "enumeration": "cuda-visible",
        "devices": [],
        "gpuCount": None,
        "gpuUseProven": False,
        "rankParticipation": "not_recorded",
    }


def checkout_sha():
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    value = (
        subprocess.check_output(
            ["git", "-c", "safe.directory=" + os.getcwd(), "rev-parse", "HEAD"],
            env=env,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        .decode()
        .strip()
    )
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("Actual container checkout unavailable")
    return value


def inventory(kind, driver=None):
    result = {
        "status": "observed",
        "source": "container-platform",
        "devices": [],
        "gpuCount": None,
        "cpuArchitecture": platform.machine(),
        "gpuUseProven": False,
        "rankParticipation": "not_recorded",
    }
    if kind == "cpu":
        return result
    if kind != "gpu":
        raise ValueError("Unknown inventory kind")
    driver = driver or ctypes.CDLL("libcuda.so.1")
    signatures = {
        "cuInit": [ctypes.c_uint],
        "cuDeviceGetCount": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        "cuDeviceGetName": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
        "cuDeviceGetAttribute": [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ],
    }
    for name, arguments in signatures.items():
        function = getattr(driver, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
    if driver.cuInit(0) != 0:
        raise ValueError("CUDA initialization failed")
    count = ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or not 1 <= count.value <= 16:
        raise ValueError("CUDA visible device count unavailable or out of bounds")
    devices = []
    for ordinal in range(count.value):
        device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        if (
            driver.cuDeviceGet(ctypes.byref(device), ordinal) != 0
            or driver.cuDeviceGetName(name, len(name), device) != 0
            or driver.cuDeviceGetAttribute(ctypes.byref(major), 75, device) != 0
            or driver.cuDeviceGetAttribute(ctypes.byref(minor), 76, device) != 0
            or not 1 <= major.value <= 99
            or not 0 <= minor.value <= 9
        ):
            raise ValueError("CUDA device property query failed")
        label = name.value.decode("utf-8", errors="replace")
        if not label or len(label) > 200 or any(ord(c) < 32 for c in label):
            raise ValueError("CUDA device name invalid")
        devices.append(
            {
                "ordinal": ordinal,
                "name": label,
                "computeCapability": f"{major.value}.{minor.value}",
                "hardwareTarget": f"sm{major.value}{minor.value}",
            }
        )
    result.update(
        source="container-cuda-driver",
        enumeration="cuda-visible",
        devices=devices,
        gpuCount=len(devices),
    )
    return result


def probe_result(kind):
    try:
        actual = checkout_sha()
    except (OSError, ValueError, subprocess.SubprocessError):
        return {
            "testedCheckoutSha": None,
            "hardware": unavailable("Actual container checkout could not be read"),
        }
    try:
        observed = inventory(kind)
    except OSError:
        observed = unavailable(
            "CUDA driver library is unavailable inside the test container"
        )
    except AttributeError:
        observed = unavailable("Required CUDA driver inventory API is unavailable")
    except ValueError as error:
        # inventory() raises only these controller-owned diagnostics.
        observed = unavailable(str(error))
    return {"testedCheckoutSha": actual, "hardware": observed}


def capture(kind, context):
    hardware = unavailable("Actual container hardware probe failed")
    reason = "Actual container hardware probe failed"
    try:
        with tempfile.TemporaryDirectory(prefix="watcher-hardware-") as directory:
            output = Path(directory) / "inventory.json"

            def limits():
                import resource

                resource.setrlimit(resource.RLIMIT_FSIZE, (LIMIT, LIMIT))

            with output.open("wb") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-I", "-S", __file__, "probe", "--kind", kind],
                    stdout=stream,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    preexec_fn=limits,
                )
                try:
                    code = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    reason = "Actual container hardware probe exceeded the 20-second deadline"
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                    raise TimeoutError("Hardware probe timeout") from None
            data = output.read_bytes()
            if code != 0 or len(data) >= LIMIT:
                reason = "Actual container hardware probe exited or exceeded its output bound"
                raise ValueError("Hardware probe incomplete")
            observed = json.loads(data)
            if observed.get("testedCheckoutSha") != context["testedCheckoutSha"]:
                reason = "Actual container checkout did not match the tested checkout"
                raise ValueError("Actual container checkout mismatch")
            hardware = observed["hardware"]
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        TimeoutError,
        subprocess.SubprocessError,
        AttributeError,
    ):
        hardware = unavailable(reason)
    return {
        "schemaVersion": 1,
        "kind": "watcher-ci-hardware-observation",
        "context": context,
        "capturedAt": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "run"))
    parser.add_argument("--kind", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--context", type=Path)
    arguments = sys.argv[1:]
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    command = arguments[separator + 1 :]
    args = parser.parse_args(arguments[:separator])
    if args.mode == "probe":
        print(json.dumps(probe_result(args.kind)))
        return 0
    if not command:
        parser.error("Original lane command required")
    try:
        if args.context.is_symlink() or args.context.stat().st_size > LIMIT:
            raise ValueError("Invalid hardware context")
        context = json.loads(args.context.read_bytes())
        if (
            context.get("probeSha256")
            != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ):
            raise ValueError("Trusted hardware probe changed")
        document = capture(args.kind, context)
        data = json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if len(data) > LIMIT:
            raise ValueError("Hardware frame exceeds bound")
        sys.stdout.buffer.write(MAGIC + base64.b64encode(data) + b"\n")
        sys.stdout.buffer.flush()
    except (OSError, ValueError, TypeError, AttributeError):
        # No raw exception paths or inherited environment values in public logs.
        # Even a malformed context ends the trusted prelude before PR code can
        # emit output. The parent locks this first frame as unavailable.
        sys.stdout.buffer.write(
            MAGIC + base64.b64encode(b'{"unavailable":true}') + b"\n"
        )
        sys.stdout.buffer.flush()
    os.execvp(command[0], command)


if __name__ == "__main__":
    raise SystemExit(main())
