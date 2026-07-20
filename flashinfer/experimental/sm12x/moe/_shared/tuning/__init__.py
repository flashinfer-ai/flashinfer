# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/tuning/__init__.py @ b49787f5 (2026-07-08) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import importlib.util
import pathlib
import sys

from .registry import (
    MAX_ACTIVE_CLUSTERS_POLICY,
    Ladder,
    MaxActiveClustersPolicy,
    get_max_active_clusters_policy,
    lookup_max_active_clusters,
    register_max_active_clusters_policy,
)


_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
for _policy_path in sorted(_PACKAGE_DIR.glob("*.py")):
    if _policy_path.name in {"__init__.py", "registry.py"}:
        continue
    _module_name = f"{__name__}._generated_{_policy_path.stem.replace('.', '_')}"
    if _module_name in sys.modules:
        continue
    _spec = importlib.util.spec_from_file_location(_module_name, _policy_path)
    if _spec is None or _spec.loader is None:
        continue
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = _module
    _spec.loader.exec_module(_module)

__all__ = [
    "MAX_ACTIVE_CLUSTERS_POLICY",
    "Ladder",
    "MaxActiveClustersPolicy",
    "get_max_active_clusters_policy",
    "lookup_max_active_clusters",
    "register_max_active_clusters_policy",
]
