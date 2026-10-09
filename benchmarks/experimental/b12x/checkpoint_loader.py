from __future__ import annotations

import json
import pathlib

from safetensors import safe_open


class IndexedSafetensorLoader:
    """Load checkpoint tensors by key through the safetensors index."""

    def __init__(self, model_path: pathlib.Path):
        self.model_path = pathlib.Path(model_path)
        index_path = self.model_path / "model.safetensors.index.json"
        self.weight_map = json.loads(index_path.read_text())["weight_map"]
        self._open_files: dict[str, object] = {}

    def get_tensor(self, key: str):
        shard = self.weight_map[key]
        handle = self._open_files.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_path / shard), framework="pt")
            self._open_files[shard] = handle
        return handle.get_tensor(key)
