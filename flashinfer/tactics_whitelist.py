"""Offline-generated blacklist of known-invalid CUTLASS tactics.

This module provides the ``TacticsWhitelist`` class that loads a JSON file
mapping (custom_op, runner_class) pairs to sets of tactics known to fail on
a specific GPU.  The autotuner calls ``filter()`` before profiling to skip
these tactics, saving GPU time and avoiding noisy error logs.

When no whitelist is loaded or no entry exists for a given operation,
all tactics pass through unchanged (zero regression risk).

See ``scripts/generate_tactics_whitelist.py`` for the offline probe tool
that produces the JSON files consumed here.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .autotuner import (
    _METADATA_KEY,
    _collect_metadata,
    _tactic_to_json,
    _tactic_to_json_hashable,
)
from .jit.core import logger


class TacticsWhitelist:
    """Blacklist-based tactic filter backed by an offline-generated JSON file.

    Despite the name "whitelist", the implementation is a *blacklist*: it
    stores known-invalid tactics and removes them from the candidate list.
    Unknown or new tactics are allowed through, which is safer than a strict
    whitelist that would block anything not explicitly approved.
    """

    def __init__(self) -> None:
        self._invalid: Dict[str, Set[Any]] = {}
        self._loaded_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, path: str) -> bool:
        """Load invalid-tactics data from a JSON file.

        Args:
            path: Path to the whitelist JSON file produced by
                ``generate_tactics_whitelist.py``.

        Returns:
            True if loaded successfully, False if the file's GPU metadata
            does not match the current device (the file is silently skipped).
        """
        with open(path, "r") as f:
            data = json.load(f)

        saved_meta = data.get(_METADATA_KEY)
        if saved_meta is not None:
            current_meta = _collect_metadata()
            saved_gpu = saved_meta.get("gpu", "")
            current_gpu = current_meta.get("gpu", "")
            if saved_gpu != current_gpu and saved_gpu != "*":
                logger.warning(
                    f"[TacticsWhitelist]: File {path} was generated for "
                    f"'{saved_gpu}' but current GPU is '{current_gpu}'. "
                    f"Skipping."
                )
                return False

        invalid_tactics = data.get("invalid_tactics", {})
        for key, tactics_list in invalid_tactics.items():
            self._invalid[key] = {_tactic_to_json_hashable(t) for t in tactics_list}

        self._loaded_path = path
        total = sum(len(v) for v in self._invalid.values())
        logger.info(
            f"[TacticsWhitelist]: Loaded {total} invalid tactic(s) "
            f"across {len(self._invalid)} operation(s) from {path}"
        )
        return True

    # ------------------------------------------------------------------
    # Runtime filtering
    # ------------------------------------------------------------------

    def filter(
        self,
        custom_op: str,
        runner: Any,
        tactics: List,
    ) -> List:
        """Remove known-invalid tactics from *tactics*.

        Args:
            custom_op: Custom operation name
                (e.g. ``"flashinfer::trtllm_fp4_block_scale_moe"``).
            runner: A ``TunableRunner`` instance.
            tactics: Candidate tactics returned by ``get_valid_tactics()``.

        Returns:
            Filtered list with known-bad tactics removed.  If no whitelist is
            loaded or no entry matches, returns *tactics* unchanged.
        """
        if not self._invalid:
            return tactics

        key = f"{custom_op}::{runner.__class__.__name__}"
        invalid_set = self._invalid.get(key)
        if invalid_set is None:
            return tactics

        filtered = [
            t for t in tactics if _tactic_to_json_hashable(t) not in invalid_set
        ]
        removed = len(tactics) - len(filtered)
        if removed > 0:
            logger.debug(
                f"[TacticsWhitelist]: Filtered {removed} invalid tactic(s) "
                f"for {key} ({len(filtered)} remaining)"
            )
        return filtered

    # ------------------------------------------------------------------
    # Saving (used by the probe script)
    # ------------------------------------------------------------------

    @staticmethod
    def save(
        path: str,
        invalid_tactics: Dict[str, List],
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Write an invalid-tactics file to *path*.

        Args:
            path: Output file path.
            invalid_tactics: Mapping of ``"custom_op::RunnerClass"`` to lists
                of tactics that failed probing.
            metadata: Optional metadata dict.  If ``None``, environment
                metadata is auto-collected via ``_collect_metadata()``.
        """
        meta = dict(metadata) if metadata else _collect_metadata()
        meta["generated_at"] = datetime.now(timezone.utc).isoformat()
        meta["generator_version"] = "1.0"

        serialized: Dict[str, list] = {}
        for key, tacs in invalid_tactics.items():
            serialized[key] = [_tactic_to_json(t) for t in tacs]

        output = {
            _METADATA_KEY: meta,
            "invalid_tactics": serialized,
        }

        abs_path = os.path.abspath(path)
        dir_name = os.path.dirname(abs_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        import tempfile

        fd, tmp_path = tempfile.mkstemp(
            dir=dir_name, suffix=".tmp", prefix=".whitelist_"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(output, f, indent=2)
            os.replace(tmp_path, abs_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        total = sum(len(v) for v in serialized.values())
        logger.info(
            f"[TacticsWhitelist]: Saved {total} invalid tactic(s) "
            f"across {len(serialized)} operation(s) to {path}"
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True when at least one whitelist file has been loaded."""
        return bool(self._invalid)

    @property
    def loaded_path(self) -> Optional[str]:
        return self._loaded_path

    def summary(self) -> Dict[str, int]:
        """Return a {op_key: count} mapping of invalid tactics per operation."""
        return {k: len(v) for k, v in self._invalid.items()}
