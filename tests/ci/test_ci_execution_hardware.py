"""Actual-container inventory protocol tests with CPU/fake-driver fixtures only."""

import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_ci_execution_manifest as fixtures

manifest = fixtures.manifest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_execution_hardware", ROOT / "scripts/pr_checks/ci_execution_hardware.py"
)
hardware = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hardware)


class Call:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class FakeDriver:
    def __init__(self, capabilities, failure=False):
        self.cuInit = Call(lambda _: int(failure))
        self.cuDeviceGetCount = Call(
            lambda pointer: self.set_value(pointer, len(capabilities))
        )
        self.cuDeviceGet = Call(
            lambda pointer, ordinal: self.set_value(pointer, ordinal)
        )

        def name(buffer, limit, device):
            buffer.value = ("Fixture GPU " + str(device.value)).encode()
            return 0

        self.cuDeviceGetName = Call(name)
        self.cuDeviceGetAttribute = Call(
            lambda pointer, attribute, device: self.set_value(
                pointer, capabilities[device.value][0 if attribute == 75 else 1]
            )
        )

    @staticmethod
    def set_value(pointer, value):
        pointer._obj.value = value
        return 0


def observation():
    return hardware.inventory("gpu", FakeDriver([(9, 0)]))


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "testedCheckoutSha": "a" * 40,
            "runId": 10,
            "runAttempt": 1,
            "laneKey": "b" * 64,
            "captureNonce": "c" * 48,
        }

    def document(self, value=None):
        return {
            "schemaVersion": 1,
            "kind": "watcher-ci-hardware-observation",
            "context": self.context,
            "capturedAt": "2026-10-14T00:00:00+00:00",
            "hardware": observation() if value is None else value,
        }

    def frame(self, document):
        return hardware.MAGIC + base64.b64encode(json.dumps(document).encode()) + b"\n"

    def test_actual_driver_values_not_configured_labels(self):
        value = hardware.inventory("gpu", FakeDriver([(9, 0), (10, 0)]))
        self.assertEqual(
            [d["hardwareTarget"] for d in value["devices"]], ["sm90", "sm100"]
        )
        self.assertEqual(value["gpuCount"], 2)
        self.assertEqual(value["enumeration"], "cuda-visible")
        self.assertIs(value["gpuUseProven"], False)
        self.assertEqual(value["rankParticipation"], "not_recorded")
        self.assertNotIn("uuid", json.dumps(value))
        manifest.validate_hardware(value)

    def test_missing_driver_no_devices_and_unbounded_inventory_fail(self):
        for driver in [
            FakeDriver([]),
            FakeDriver([(9, 0)] * 17),
            FakeDriver([(9, 0)], failure=True),
        ]:
            with self.assertRaises(ValueError):
                hardware.inventory("gpu", driver)
        value = hardware.inventory("cpu")
        self.assertEqual(value["source"], "container-platform")
        self.assertEqual(value["devices"], [])
        self.assertIsNone(value["gpuCount"])

    def test_probe_unavailability_has_specific_safe_reason(self):
        with (
            patch.object(hardware, "checkout_sha", return_value="a" * 40),
            patch.object(hardware, "inventory", side_effect=OSError("secret=fixture")),
        ):
            value = hardware.probe_result("gpu")
        self.assertEqual(value["testedCheckoutSha"], "a" * 40)
        self.assertIn("driver library", value["hardware"]["reason"])
        self.assertNotIn("secret", json.dumps(value))

    def test_first_bound_frame_wins_and_later_pr_output_is_opaque(self):
        reader = manifest.HardwareFrame(self.context)
        first = self.frame(self.document())
        for chunk in [b"container startup logs\n", first[:19], first[19:]]:
            reader.feed(chunk)
        forged = self.document(hardware.inventory("gpu", FakeDriver([(10, 0)])))
        reader.feed(self.frame(forged))
        self.assertEqual(
            reader.document["hardware"]["devices"][0]["hardwareTarget"], "sm90"
        )
        self.assertTrue(reader.seen)

    def test_malformed_or_wrong_context_first_frame_cannot_be_replaced(self):
        for data in [
            hardware.MAGIC + b"not base64\n",
            self.frame({**self.document(), "context": {}}),
            hardware.MAGIC + b"x" * 150000 + b"\n",
        ]:
            reader = manifest.HardwareFrame(self.context)
            for offset in range(0, len(data), 65536):
                reader.feed(data[offset : offset + 65536])
            reader.feed(self.frame(self.document()))
            self.assertTrue(reader.seen)
            self.assertIsNone(reader.document)
            self.assertLessEqual(len(reader.pending), reader.MAX_LINE)

    def test_wrong_property_or_rank_assertion_is_rejected(self):
        value = observation()
        value["devices"][0]["hardwareTarget"] = "sm100"
        with self.assertRaises(ValueError):
            manifest.validate_hardware(value)
        value = observation()
        value["gpuUseProven"] = True
        with self.assertRaises(ValueError):
            manifest.validate_hardware(value)

    def test_capture_subprocess_bounded_and_checkout_bound(self):
        worker = Mock()
        worker.wait.return_value = 0

        def start(argv, **kwargs):
            self.assertIn("-I", argv)
            self.assertIn("-S", argv)
            self.assertEqual(argv[-2:], ["--kind", "gpu"])
            self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
            kwargs["stdout"].write(
                json.dumps(
                    {
                        "testedCheckoutSha": self.context["testedCheckoutSha"],
                        "hardware": observation(),
                    }
                ).encode()
            )
            return worker

        with patch.object(hardware.subprocess, "Popen", side_effect=start):
            document = hardware.capture("gpu", self.context)
        worker.wait.assert_called_once_with(timeout=20)
        self.assertEqual(document["hardware"]["status"], "observed")
        with patch.object(hardware.subprocess, "Popen", side_effect=start):
            document = hardware.capture(
                "gpu", {**self.context, "testedCheckoutSha": "d" * 40}
            )
        self.assertEqual(document["hardware"]["status"], "unavailable")

    def test_probe_timeout_and_raw_errors_never_disclose_secrets(self):
        worker = Mock(pid=123)
        worker.wait.side_effect = [subprocess.TimeoutExpired("probe", 20), 0]
        with (
            patch.object(hardware.subprocess, "Popen", return_value=worker),
            patch.object(hardware.os, "killpg") as kill,
        ):
            document = hardware.capture("gpu", self.context)
        kill.assert_called_once()
        self.assertEqual(document["hardware"]["status"], "unavailable")
        with patch.object(
            hardware.subprocess,
            "Popen",
            side_effect=OSError("secret=fixture credential path"),
        ):
            document = hardware.capture("gpu", self.context)
        self.assertNotIn("secret", json.dumps(document))

    def test_failed_context_emits_a_terminal_unavailable_frame_before_exec(self):
        output = Mock()
        output.buffer = io.BytesIO()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text("{}")
            argv = [
                "helper",
                "run",
                "--kind",
                "gpu",
                "--context",
                str(path),
                "--",
                "/fixture/script",
                "arg",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "stdout", output),
                patch.object(os, "execvp") as execute,
            ):
                hardware.main()
        reader = manifest.HardwareFrame(self.context)
        reader.feed(output.buffer.getvalue())
        self.assertTrue(reader.seen)
        self.assertIsNone(reader.document)
        execute.assert_called_once_with("/fixture/script", ["/fixture/script", "arg"])

    def test_real_cpu_wrapper_and_host_capture_preserve_original_exit(self):
        fixture = fixtures.ManifestTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        tool = fixture.root / "tools"
        tool.mkdir()
        probe = tool / "hardware.py"
        probe.write_bytes(
            (ROOT / "scripts/pr_checks/ci_execution_hardware.py").read_bytes()
        )
        context = tool / "context.json"
        manifest.initialize(
            fixture.plan,
            fixture.lane,
            fixture.workspace,
            fixture.output,
            hardware_context=context,
        )
        command = [
            sys.executable,
            "-I",
            "-S",
            str(probe),
            "run",
            "--kind",
            "cpu",
            "--context",
            str(context),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(11)",
        ]
        output = Mock()
        output.buffer = io.BytesIO()
        with patch.object(sys, "stdout", output):
            self.assertEqual(
                manifest.run_command(fixture.output, fixture.workspace, command), 11
            )
        result = manifest.finalize(fixture.output, fixture.workspace)
        self.assertEqual(result["exitCode"], 11)
        self.assertEqual(result["hardware"]["source"], "container-platform")
        self.assertEqual(result["hardware"]["status"], "observed")
        artifact = next(
            a for a in result["evidence"] if a["kind"] == "hardware-observation"
        )
        payload = (fixture.output / artifact["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), artifact["sha256"])
        saved = json.loads(payload)
        self.assertEqual(saved["context"], result["hardwareContext"])
        self.assertEqual(saved["context"]["testedCheckoutSha"], fixture.merge)
        self.assertEqual(result["ranks"]["status"], "not_recorded")


if __name__ == "__main__":
    unittest.main()
