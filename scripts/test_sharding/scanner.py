from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .io import atomic_write_json, atomic_write_text
from .junit import element_properties, test_suites, testcase_outcome
from .models import (
    Batch,
    CollectedNode,
    Plan,
    Unit,
    base_function_for_nodeid,
    source_file_for_nodeid,
)
from .observations import (
    ObservedCase,
    ObservedOverhead,
    adjust_first_case_warmup,
)
from .state import manifest_path
from .summary import batch_xml_path


def _run_identity(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def _load_run(run_dir: Path) -> tuple[dict[str, Any], Plan, str]:
    path = manifest_path(run_dir)
    if not path.exists():
        raise ValueError(f"no sharding manifest below {run_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest, Plan.from_dict(manifest["plan"]), _run_identity(path)


@dataclass
class _RunScanContext:
    profile: str
    node_metadata: dict[str, CollectedNode]
    run_id: str
    seen: set[tuple[str, str, str]]
    diagnostics: list[str]


def _parse_batch_suite(
    suite: ET.Element,
    expected_batch: Batch,
    context: _RunScanContext,
) -> list[ObservedCase]:
    found: list[str] = []
    observations: list[ObservedCase] = []
    for index, testcase in enumerate(suite.findall("./testcase")):
        properties = element_properties(testcase)
        nodeid = properties.get("pytest_nodeid")
        if nodeid is None:
            context.diagnostics.append(
                f"{expected_batch.id}: testcase lacks pytest_nodeid"
            )
            continue
        found.append(nodeid)
        if nodeid not in context.node_metadata:
            context.diagnostics.append(f"{expected_batch.id}: unknown node {nodeid}")
            continue
        key = (context.run_id, expected_batch.id, nodeid)
        if key in context.seen:
            context.diagnostics.append(f"{expected_batch.id}: duplicate node {nodeid}")
            continue
        context.seen.add(key)
        try:
            seconds = float(testcase.attrib.get("time", "0") or 0)
        except ValueError:
            context.diagnostics.append(
                f"{expected_batch.id}: invalid time for {nodeid}"
            )
            continue
        metadata = context.node_metadata[nodeid]
        synthetic = properties.get("synthetic") == "true"
        if synthetic:
            continue
        observations.append(
            ObservedCase(
                profile=context.profile,
                nodeid=nodeid,
                source_file=metadata.source_file,
                base_function=metadata.base_function,
                outcome=testcase_outcome(testcase),
                seconds=seconds,
                adjusted_seconds=seconds,
                synthetic=False,
                run_id=context.run_id,
                batch_id=expected_batch.id,
                first_in_batch=index == 0,
            )
        )
    expected = set(expected_batch.nodeids)
    actual = set(found)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        context.diagnostics.append(
            f"{expected_batch.id}: coverage mismatch missing={missing[:5]} extra={extra[:5]}"
        )
    return observations


def _read_suites(path: Path, diagnostics: list[str]) -> list[ET.Element]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        diagnostics.append(f"{path}: malformed XML: {error}")
        return []
    suites = test_suites(root)
    if suites:
        return suites
    diagnostics.append(f"{path}: unexpected root {root.tag!r}")
    return []


def _overhead_for_batch(
    run_dir: Path, run_id: str, plan: Plan, unit: Unit, batch: Batch
) -> ObservedOverhead | None:
    xml_path = batch_xml_path(run_dir, unit, batch)
    telemetry_path = xml_path.with_name(f"{batch.id}.telemetry.json")
    try:
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        launch = float(telemetry["process_launch"])
        collection = float(telemetry["collection_complete"])
        first_case = float(telemetry.get("first_case_start") or collection)
        report = float(telemetry.get("report_complete") or telemetry["process_exit"])
        process_exit = float(telemetry["process_exit"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    startup = max(0.0, collection - launch) + max(0.0, process_exit - report)
    warmup = max(0.0, first_case - collection)
    return ObservedOverhead(
        profile=plan.options.profile,
        source_file=batch.source_file,
        process_startup_seconds=startup,
        source_warmup_seconds=warmup,
        run_id=run_id,
        batch_id=batch.id,
    )


def _scan_run(
    run_dir: Path,
    selected_xml: set[Path] | None = None,
) -> tuple[list[ObservedCase], list[ObservedOverhead], list[str]]:
    _, plan, run_id = _load_run(run_dir)
    observations: list[ObservedCase] = []
    overheads: list[ObservedOverhead] = []
    diagnostics: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    node_metadata = {node.nodeid: node for node in plan.nodes}
    context = _RunScanContext(
        profile=plan.options.profile,
        node_metadata=node_metadata,
        run_id=run_id,
        seen=seen,
        diagnostics=diagnostics,
    )
    for unit in plan.units:
        for batch in unit.batches:
            raw = batch_xml_path(run_dir, unit, batch)
            if not raw.exists() or (
                selected_xml is not None and raw.resolve() not in selected_xml
            ):
                continue
            suites = _read_suites(raw, diagnostics)
            for suite in suites:
                observations.extend(
                    _parse_batch_suite(
                        suite,
                        batch,
                        context,
                    )
                )
            overhead = _overhead_for_batch(run_dir, run_id, plan, unit, batch)
            if overhead is not None:
                overheads.append(overhead)
    adjusted, warmup_excess = adjust_first_case_warmup(observations)
    overheads = [
        ObservedOverhead(
            profile=item.profile,
            source_file=item.source_file,
            process_startup_seconds=item.process_startup_seconds,
            source_warmup_seconds=item.source_warmup_seconds
            + warmup_excess.get((item.run_id, item.batch_id), 0.0),
            run_id=item.run_id,
            batch_id=item.batch_id,
        )
        for item in overheads
    ]
    adjusted.sort(
        key=lambda item: (item.profile.encode(), item.nodeid.encode(), item.run_id)
    )
    overheads.sort(
        key=lambda item: (
            item.profile.encode(),
            item.source_file.encode(),
            item.run_id,
            item.batch_id,
        )
    )
    return adjusted, overheads, diagnostics


def _find_run_dir(path: Path) -> Path | None:
    candidates = [path] if path.is_dir() else [path.parent, *path.parents]
    for candidate in candidates:
        if manifest_path(candidate).exists():
            return candidate
    return None


def _register_observation_input(
    run_dirs: dict[Path, set[Path] | None],
    supplied: Path,
    diagnostics: list[str],
) -> None:
    path = supplied.resolve()
    run_dir = _find_run_dir(path)
    if run_dir is None:
        diagnostics.append(f"{path}: no containing sharding manifest")
    elif path.is_dir():
        run_dirs[run_dir] = None
    elif run_dir not in run_dirs or run_dirs[run_dir] is not None:
        selected = run_dirs.setdefault(run_dir, set())
        assert selected is not None
        selected.add(path)


def scan_observation_inputs(
    inputs: Iterable[Path],
) -> tuple[list[ObservedCase], list[ObservedOverhead], list[str]]:
    run_dirs: dict[Path, set[Path] | None] = {}
    diagnostics: list[str] = []
    for supplied in inputs:
        _register_observation_input(run_dirs, supplied, diagnostics)
    observations: list[ObservedCase] = []
    overheads: list[ObservedOverhead] = []
    seen_cases: set[tuple[str, str, str]] = set()
    seen_overheads: set[tuple[str, str]] = set()
    for run_dir in sorted(run_dirs, key=lambda value: str(value).encode("utf-8")):
        try:
            cases, batch_overheads, run_diagnostics = _scan_run(
                run_dir, run_dirs[run_dir]
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as error:
            diagnostics.append(f"{run_dir}: {error}")
            continue
        for case in cases:
            case_key = (case.run_id, case.batch_id, case.nodeid)
            if case_key in seen_cases:
                diagnostics.append(
                    f"{run_dir}: duplicate supplied observation {case.batch_id}:{case.nodeid}"
                )
            else:
                seen_cases.add(case_key)
                observations.append(case)
        for overhead in batch_overheads:
            overhead_key = (overhead.run_id, overhead.batch_id)
            if overhead_key not in seen_overheads:
                seen_overheads.add(overhead_key)
                overheads.append(overhead)
        diagnostics.extend(run_diagnostics)
    return observations, overheads, diagnostics


_PYTEST_RESULT = re.compile(
    r"PYTEST RESULT worker=\d+ batch=(?P<batch>\S+) "
    r"outcome=(?P<outcome>\S+) duration=(?P<seconds>[0-9.]+)s "
    r"node=(?P<nodeid>.+?)\s*$"
)


@dataclass(frozen=True)
class _ArtifactObservationKey:
    run_id: str
    profile: str
    batch_id: str
    nodeid: str


def _artifact_run_identity(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_suites(
    content: bytes,
    *,
    label: str,
    diagnostics: list[str],
) -> list[ET.Element]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        diagnostics.append(f"{label}: malformed XML: {error}")
        return []
    suites = test_suites(root)
    if suites:
        return suites
    diagnostics.append(f"{label}: unexpected root {root.tag!r}")
    return []


def _artifact_case(
    testcase: ET.Element,
    key: _ArtifactObservationKey,
    first_in_batch: bool,
) -> ObservedCase | None:
    properties = element_properties(testcase)
    if properties.get("synthetic") == "true":
        return None
    try:
        seconds = float(testcase.attrib.get("time", "0") or 0)
    except ValueError:
        return None
    return ObservedCase(
        profile=key.profile,
        nodeid=key.nodeid,
        source_file=source_file_for_nodeid(key.nodeid),
        base_function=base_function_for_nodeid(key.nodeid),
        outcome=testcase_outcome(testcase),
        seconds=seconds,
        adjusted_seconds=seconds,
        synthetic=False,
        run_id=key.run_id,
        batch_id=key.batch_id,
        first_in_batch=first_in_batch,
    )


@dataclass(frozen=True)
class _ArtifactIndex:
    sorted_nodeids: tuple[str, ...]
    batches: dict[str, Batch]

    @classmethod
    def from_manifest(cls, path: Path) -> _ArtifactIndex:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        plan = Plan.from_dict(manifest["plan"])
        return cls(
            sorted_nodeids=tuple(
                sorted(
                    (node.nodeid for node in plan.nodes),
                    key=lambda value: value.encode("utf-8"),
                )
            ),
            batches={batch.id: batch for unit in plan.units for batch in unit.batches},
        )


@dataclass
class _ArchiveOmissions:
    ambiguous: int = 0
    unknown: int = 0
    invalid: int = 0


@dataclass
class _ArchiveContext:
    path: Path
    run_id: str
    profiles: set[str] = field(default_factory=set)
    omissions: _ArchiveOmissions = field(default_factory=_ArchiveOmissions)


@dataclass
class _CleanedArtifactScan:
    index: _ArtifactIndex
    observations: list[ObservedCase] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    seen: set[_ArtifactObservationKey] = field(default_factory=set)
    xml_resolution: dict[_ArtifactObservationKey, str] = field(default_factory=dict)
    prefix_conflicts: set[_ArtifactObservationKey] = field(default_factory=set)
    job_metadata: dict[Path, tuple[str, str]] = field(default_factory=dict)

    def scan_archive(self, archive_path: Path) -> None:
        context = _ArchiveContext(
            path=archive_path,
            run_id=_artifact_run_identity(archive_path),
        )
        try:
            archive = zipfile.ZipFile(archive_path)
        except (OSError, zipfile.BadZipFile) as error:
            self.diagnostics.append(f"{archive_path}: unreadable ZIP: {error}")
            return
        with archive:
            members = sorted(
                (name for name in archive.namelist() if name.endswith(".xml")),
                key=lambda value: value.encode("utf-8"),
            )
            for member in members:
                self._scan_archive_member(
                    archive,
                    member,
                    context,
                )
        self._record_archive_metadata(
            archive_path,
            context.run_id,
            context.profiles,
            context.omissions,
        )

    def _scan_archive_member(
        self,
        archive: zipfile.ZipFile,
        member: str,
        context: _ArchiveContext,
    ) -> None:
        label = f"{context.path}:{member}"
        try:
            content = archive.read(member)
        except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
            self.diagnostics.append(f"{label}: unreadable ZIP member: {error}")
            return
        for suite in _artifact_suites(
            content,
            label=label,
            diagnostics=self.diagnostics,
        ):
            self._scan_suite(
                suite,
                label,
                context.run_id,
                context.profiles,
                context.omissions,
            )

    def _scan_suite(
        self,
        suite: ET.Element,
        label: str,
        run_id: str,
        profiles: set[str],
        omissions: _ArchiveOmissions,
    ) -> None:
        properties = element_properties(suite)
        profile = properties.get("timing_profile", "")
        batch_id = properties.get("batch_id", "")
        if not profile or not batch_id:
            self.diagnostics.append(f"{label}: suite lacks timing_profile or batch_id")
            return
        profiles.add(profile)
        if properties.get("synthetic") == "true":
            return
        testcases = list(suite.findall("./testcase"))
        exact_nodeids = self._exact_batch_nodeids(batch_id, testcases)
        for position, testcase in enumerate(testcases):
            nodeid, resolution = self._resolve_nodeid(
                testcase,
                exact_nodeids[position] if exact_nodeids is not None else None,
            )
            if nodeid is None:
                if resolution == "ambiguous":
                    omissions.ambiguous += 1
                else:
                    omissions.unknown += 1
                continue
            key = _ArtifactObservationKey(
                run_id=run_id,
                profile=profile,
                batch_id=batch_id,
                nodeid=nodeid,
            )
            observation = _artifact_case(
                testcase,
                key,
                position == 0,
            )
            if observation is None:
                omissions.invalid += 1
                continue
            self._add_observation(observation, resolution)

    def _exact_batch_nodeids(
        self,
        batch_id: str,
        testcases: list[ET.Element],
    ) -> tuple[str, ...] | None:
        expected = self.index.batches.get(batch_id)
        if expected is None or len(expected.nodeids) != len(testcases):
            return None
        if all(
            (prefix := element_properties(testcase).get("pytest_nodeid", ""))
            and nodeid.startswith(prefix)
            for testcase, nodeid in zip(testcases, expected.nodeids, strict=True)
        ):
            return expected.nodeids
        return None

    def _resolve_nodeid(
        self,
        testcase: ET.Element,
        exact_nodeid: str | None,
    ) -> tuple[str | None, str]:
        if exact_nodeid is not None:
            return exact_nodeid, "batch"
        prefix = element_properties(testcase).get("pytest_nodeid", "")
        if not prefix:
            return None, "unknown"
        lower = bisect.bisect_left(self.index.sorted_nodeids, prefix)
        upper = bisect.bisect_left(self.index.sorted_nodeids, prefix + chr(0x10FFFF))
        testcase_name = testcase.attrib.get("name", "")
        candidates = [
            candidate
            for candidate in self.index.sorted_nodeids[lower:upper]
            if candidate.rsplit("::", 1)[-1].startswith(testcase_name)
        ]
        if len(candidates) == 1:
            return candidates[0], "prefix"
        return None, "ambiguous" if candidates else "unknown"

    def _add_observation(
        self,
        observation: ObservedCase,
        resolution: str,
    ) -> None:
        key = _ArtifactObservationKey(
            run_id=observation.run_id,
            profile=observation.profile,
            batch_id=observation.batch_id,
            nodeid=observation.nodeid,
        )
        if key not in self.seen:
            self.seen.add(key)
            self.xml_resolution[key] = resolution
            self.observations.append(observation)
        elif self.xml_resolution.get(key) == "prefix" or resolution == "prefix":
            self.prefix_conflicts.add(key)

    def _record_archive_metadata(
        self,
        archive_path: Path,
        run_id: str,
        profiles: set[str],
        omissions: _ArchiveOmissions,
    ) -> None:
        if len(profiles) == 1:
            self.job_metadata[archive_path.with_suffix("")] = (
                next(iter(profiles)),
                run_id,
            )
        elif profiles:
            self.diagnostics.append(
                f"{archive_path}: inconsistent timing profiles {sorted(profiles)}; "
                "sibling log omitted"
            )
        labels = (
            (
                omissions.ambiguous,
                "ambiguous testcase node IDs after attribute truncation",
            ),
            (
                omissions.unknown,
                "testcase node IDs absent from the reconstruction manifest",
            ),
            (
                omissions.invalid,
                "testcases with invalid or synthetic timing data",
            ),
        )
        for count, label in labels:
            if count:
                self.diagnostics.append(f"{archive_path}: omitted {count} {label}")

    def remove_prefix_conflicts(self) -> None:
        by_profile_node: dict[tuple[str, str], list[_ArtifactObservationKey]] = {}
        for key in self.xml_resolution:
            by_profile_node.setdefault((key.profile, key.nodeid), []).append(key)
        for keys in by_profile_node.values():
            if len({key.batch_id for key in keys}) > 1:
                batch_verified = {
                    key for key in keys if self.xml_resolution[key] == "batch"
                }
                self.prefix_conflicts.update(set(keys) - batch_verified)
        if not self.prefix_conflicts:
            return
        conflicts = self.prefix_conflicts
        self.observations = [
            observation
            for observation in self.observations
            if _ArtifactObservationKey(
                run_id=observation.run_id,
                profile=observation.profile,
                batch_id=observation.batch_id,
                nodeid=observation.nodeid,
            )
            not in conflicts
        ]
        self.seen.difference_update(conflicts)
        self.diagnostics.append(
            "cleaned artifacts: omitted "
            f"{len(conflicts)} prefix-resolved observations that collided within "
            "a batch or mapped one node ID into different batches"
        )

    def scan_log(self, log_path: Path) -> None:
        metadata = self.job_metadata.get(log_path.with_suffix(""))
        if metadata is None:
            self.diagnostics.append(
                f"{log_path}: no sibling ZIP with one timing profile; log omitted"
            )
            return
        profile, run_id = metadata
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            self.diagnostics.append(f"{log_path}: unreadable log: {error}")
            return
        for line in lines:
            self._add_log_line(line, profile, run_id)

    def _add_log_line(self, line: str, profile: str, run_id: str) -> None:
        match = _PYTEST_RESULT.search(line)
        if match is None:
            return
        key = _ArtifactObservationKey(
            run_id=run_id,
            profile=profile,
            batch_id=match.group("batch"),
            nodeid=match.group("nodeid"),
        )
        if key in self.seen:
            return
        try:
            seconds = float(match.group("seconds"))
        except ValueError:
            return
        self.seen.add(key)
        self.observations.append(
            ObservedCase(
                profile=profile,
                nodeid=key.nodeid,
                source_file=source_file_for_nodeid(key.nodeid),
                base_function=base_function_for_nodeid(key.nodeid),
                outcome=match.group("outcome"),
                seconds=seconds,
                adjusted_seconds=seconds,
                synthetic=False,
                run_id=run_id,
                batch_id=key.batch_id,
                first_in_batch=False,
            )
        )


def _discover_cleaned_artifacts(
    artifact_dirs: Iterable[Path],
) -> tuple[list[Path], list[Path], list[str]]:
    directories = sorted(
        {path.resolve() for path in artifact_dirs},
        key=lambda value: str(value).encode("utf-8"),
    )
    zip_paths: list[Path] = []
    log_paths: list[Path] = []
    diagnostics: list[str] = []
    for directory in directories:
        if not directory.is_dir():
            diagnostics.append(f"{directory}: cleaned artifact path is not a directory")
            continue
        zip_paths.extend(directory.glob("*.zip"))
        log_paths.extend(directory.glob("*.log"))
    return (
        sorted(set(zip_paths), key=lambda value: str(value).encode("utf-8")),
        sorted(set(log_paths), key=lambda value: str(value).encode("utf-8")),
        diagnostics,
    )


def scan_cleaned_artifact_dirs(
    artifact_dirs: Iterable[Path],
    reconstruction_manifest: Path,
) -> tuple[list[ObservedCase], list[ObservedOverhead], list[str]]:
    """Recover observations from GitLab-cleaned ZIPs and sibling job logs.

    The GitLab cleaner truncates XML attributes, including long pytest node
    IDs. A compatible reconstruction manifest restores exact IDs when the
    deterministic batch hash and testcase order agree. For batches from a
    different source revision, only uniquely resolvable combinations of the
    node-ID and testcase-name prefixes are accepted. Sibling logs supply exact
    failed/skipped node IDs, including results from batches that never
    finalized their XML.
    """

    index = _ArtifactIndex.from_manifest(reconstruction_manifest)
    zip_paths, log_paths, diagnostics = _discover_cleaned_artifacts(artifact_dirs)
    scan = _CleanedArtifactScan(index=index, diagnostics=diagnostics)
    for archive_path in zip_paths:
        scan.scan_archive(archive_path)
    scan.remove_prefix_conflicts()
    for log_path in log_paths:
        scan.scan_log(log_path)
    scan.observations.sort(
        key=lambda item: (
            item.profile.encode(),
            item.nodeid.encode(),
            item.run_id,
            item.batch_id,
        )
    )
    return scan.observations, [], scan.diagnostics


def write_scan_outputs(
    output_dir: Path,
    observations: Iterable[ObservedCase],
    overheads: Iterable[ObservedOverhead],
    diagnostics: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    duration_stream = io.StringIO(newline="")
    writer = csv.writer(duration_stream, lineterminator="\n")
    writer.writerow(
        [
            "profile",
            "nodeid",
            "source_file",
            "base_function",
            "outcome",
            "wall_clock_seconds",
            "adjusted_seconds",
            "run_id",
            "batch_id",
            "first_in_batch",
        ]
    )
    observation_list = list(observations)
    for observation in observation_list:
        writer.writerow(
            [
                observation.profile,
                observation.nodeid,
                observation.source_file,
                observation.base_function,
                observation.outcome,
                f"{observation.seconds:.6f}",
                f"{observation.adjusted_seconds:.6f}",
                observation.run_id,
                observation.batch_id,
                str(observation.first_in_batch).lower(),
            ]
        )
    atomic_write_text(
        output_dir / "observed_test_durations.csv", duration_stream.getvalue()
    )

    overhead_stream = io.StringIO(newline="")
    overhead_writer = csv.writer(overhead_stream, lineterminator="\n")
    overhead_writer.writerow(
        [
            "profile",
            "source_file",
            "process_startup_seconds",
            "source_warmup_seconds",
            "run_id",
            "batch_id",
        ]
    )
    overhead_list = list(overheads)
    for overhead in overhead_list:
        overhead_writer.writerow(
            [
                overhead.profile,
                overhead.source_file,
                f"{overhead.process_startup_seconds:.6f}",
                f"{overhead.source_warmup_seconds:.6f}",
                overhead.run_id,
                overhead.batch_id,
            ]
        )
    atomic_write_text(
        output_dir / "observed_batch_overheads.csv", overhead_stream.getvalue()
    )
    atomic_write_json(
        output_dir / "duration_refresh_summary.json",
        {
            "schema_version": 1,
            "observations": len(observation_list),
            "overhead_observations": len(overhead_list),
            "diagnostics": diagnostics,
        },
    )
