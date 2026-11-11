"""Source and output locations for the PR documentation checks.

``FLASHINFER_SRC`` and ``DOC_CHECK_OUT`` override the defaults.  The PR
adapter sets both variables while comparing archived base and head revisions;
the defaults make individual check modules usable from a checkout as well.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

CHECKS_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = CHECKS_DIR.parents[1]


def find_agent_guide(root: Path) -> Path:
    """Read the canonical guide, including the old AGENTS.md forwarding shim."""
    agents = root / "AGENTS.md"
    claude = root / "CLAUDE.md"
    if claude.is_file() and (
        not agents.is_file()
        or "instructions live in [CLAUDE.md](./CLAUDE.md)"
        in agents.read_text("utf-8", "replace")
    ):
        return claude
    return agents


FLASHINFER_ROOT: Path = Path(os.environ.get("FLASHINFER_SRC", REPO_ROOT))
FLASHINFER_PKG: Path = FLASHINFER_ROOT / "flashinfer"
DOCS_DIR: Path = FLASHINFER_ROOT / "docs"
DOCS_API_DIR: Path = DOCS_DIR / "api"
CSRC_DIR: Path = FLASHINFER_ROOT / "csrc"
AGENT_GUIDE_MD: Path = find_agent_guide(FLASHINFER_ROOT)
SKILLS_DIR: Path = FLASHINFER_ROOT / ".claude" / "skills"

OUTPUT_DIR: Path = Path(
    os.environ.get(
        "DOC_CHECK_OUT", Path(tempfile.gettempdir()) / "flashinfer-pr-checks"
    )
)

__all__ = (
    "CHECKS_DIR",
    "REPO_ROOT",
    "FLASHINFER_ROOT",
    "FLASHINFER_PKG",
    "DOCS_DIR",
    "DOCS_API_DIR",
    "CSRC_DIR",
    "AGENT_GUIDE_MD",
    "SKILLS_DIR",
    "OUTPUT_DIR",
)
