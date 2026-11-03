# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Capture I/O and graph timing for the standalone real-input benchmark."""

import importlib
import json
import sys
import zipfile


def capture(call):
    import torch

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    with torch.cuda.graph(graph):
        begin.record()
        call()
        end.record()
    return graph, begin, end


class CaptureLoader:
    """Read tensor dictionaries or .igemm archives with their portable loader.

    Archives retain ZIP CRC and per-tensor SHA checks in indexer_inputs. Torch
    dictionaries use weights_only=True. Inputs are never requantized. Choosing
    a case uses tensor shapes, so no model weights or serving runtime is needed.
    """

    def __init__(self, query_rows):
        self.query_rows = query_rows
        self._catalog = {}

    def choose(self, root, n, encoding, count, *, model, length_axis):
        import torch

        key = (str(root), model, encoding)
        if key not in self._catalog:
            folder = root / model / encoding
            if not folder.exists():
                folder = root
            cases = []
            for path in sorted(folder.rglob("*")):
                if path.suffix == ".igemm":
                    with zipfile.ZipFile(path) as archive:
                        header = json.loads(archive.read("index.json"))
                    tree = header["tree"]
                    q = header["tensors"][tree["q"]["tensor"]]
                    k = header["tensors"][tree["k"]["tensor"]]
                    q_shape, k_shape, q_dtype = q["shape"], k["shape"], q["dtype"]
                elif path.suffix == ".pt":
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    if (
                        not isinstance(payload, dict)
                        or not {"q", "k", "weights", "starts", "ends"} <= payload.keys()
                    ):
                        continue
                    meta = payload.get("metadata", {})
                    if meta.get("model", model) != model:
                        continue
                    q_shape, k_shape = payload["q"].shape, payload["k"].shape
                    q_dtype = str(payload["q"].dtype)
                else:
                    continue
                actual = "fp8" if "float8" in q_dtype else "mxfp4"
                if actual != encoding or q_shape[0] < self.query_rows:
                    continue
                cases.append(
                    dict(
                        file=str(path.relative_to(root)),
                        qkv_dtype=actual,
                        source_key_tokens=k_shape[0],
                        query_tokens=q_shape[0],
                        model=model,
                    )
                )
            self._catalog[key] = cases
        # Historical compressed lengths count original history; stored keys
        # additionally contain this query chunk. Other lengths count keys.
        requested = n // 4 + self.query_rows // 4 if length_axis == "history" else n
        result = [
            dict(c, gemm_key_tokens=requested)
            for c in self._catalog[key]
            if c["source_key_tokens"] >= requested
        ]
        return result[:count]

    def load(self, root, case, n, query_rows):
        import torch

        path = root / case["file"]
        if path.suffix == ".igemm":
            sys.path.insert(0, str(root))
            module = importlib.import_module("indexer_inputs")
            payload = module.load_snapshot(path)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        names = ("q", "k", "weights", "q_scale", "k_scale", "starts", "ends")
        inputs = {name: payload.get(name) for name in names}
        for name in ("q", "weights", "q_scale", "starts", "ends"):
            if inputs[name] is not None:
                inputs[name] = inputs[name][-query_rows:]
        inputs["k"] = inputs["k"][:n]
        if inputs["k_scale"] is not None:
            inputs["k_scale"] = inputs["k_scale"][:n]
        inputs["starts"] = inputs["starts"].clamp(0, n)
        inputs["ends"] = inputs["ends"].clamp(0, n)
        for name, value in inputs.items():
            if value is not None:
                inputs[name] = value.contiguous().cuda()
        return inputs, payload.get("metadata", {})
