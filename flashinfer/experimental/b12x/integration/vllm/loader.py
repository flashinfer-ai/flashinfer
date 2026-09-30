"""Checkpoint loading into ordinary PyTorch CUDA allocations."""

from __future__ import annotations

import dataclasses
from contextlib import ExitStack
import heapq
import json
import math
import sys
import time
from pathlib import Path

import torch
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.weight_utils import enable_tqdm

from b12x.loader._checkpoint import DirectWeightSession
from b12x.loader._progress import CheckpointDisplay

logger = init_logger("vllm.model_executor.model_loader.b12x")


class B12xModelLoader(DefaultModelLoader):
    """Route checkpoint views through an asynchronous read ring or GDS."""

    def __init__(self, load_config):
        missing = [name for name in ("file_source_tensor", "safetensors_file_sources")
                   if not hasattr(weight_utils, name)]
        if missing:
            raise RuntimeError("b12x checkpoint loader requires vLLM file-source hooks: "
                               + ", ".join(missing))
        options = dict(load_config.model_loader_extra_config)
        self.io_threads = options.pop("io_threads", 8)
        self.read_mode = options.pop("read_mode", "auto")
        if self.read_mode not in ("auto", "bounce", "gds"):
            raise ValueError("read_mode must be auto, bounce or gds")
        if options.get("enable_multithread_load"):
            raise ValueError("b12x currently uses synchronous checkpoint routing")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "b12x O_DIRECT input does not use safetensors read strategies"
            )
        super().__init__(
            dataclasses.replace(
                load_config,
                load_format="safetensors",
                model_loader_extra_config=options,
            )
        )
        self._session = None
        self._progress = None

    def load_model(self, vllm_config, model_config, prefix=""):
        from vllm.model_executor.weight_transfer import weight_transfer

        device = torch.device(
            self.load_config.device or vllm_config.device_config.device
        )
        if device.type != "cuda":
            raise ValueError("the b12x loader requires a CUDA device")
        index = torch.cuda.current_device() if device.index is None else device.index
        shared_read_group = None
        read_mode = self.read_mode
        if read_mode == "auto":
            from b12x.loader import capabilities

            read_mode = "bounce" if capabilities(index)["host_page_tables"] else "gds"
        if read_mode == "gds":
            from vllm.distributed.parallel_state import get_tp_group
            from b12x.loader._shared_checkpoint import SharedReadGroup

            shared_read_group = SharedReadGroup(get_tp_group().cpu_group, index)
        with (
            DirectWeightSession(
                index, io_threads=self.io_threads,
                shared_read_group=shared_read_group, read_mode=read_mode,
            ) as session,
            weight_transfer(session),
        ):
            self._session = session
            try:
                model = super().load_model(vllm_config, model_config, prefix)
                io_stats = session.stats()
            finally:
                self._session = None
        model._b12x_loader_storage = {
            "parameter_bytes": sum(p.nbytes for p in model.parameters()),
            "io": io_stats,
        }
        logger.debug("b12x O_DIRECT I/O counters: %s", io_stats)
        load_seconds = (
            self.counter_after_loading_weights - self.counter_before_loading_weights
        )
        payload_gb = io_stats["payload_bytes"] / 1e9
        logger.debug(
            "b12x effective weight loading: %.2f GB in %.2f s = %.2f GB/s "
            "(selected checkpoint payload, per rank)",
            payload_gb,
            load_seconds,
            payload_gb / load_seconds,
        )
        return model

    def load_weights(self, model, model_config):
        from vllm.utils.system_utils import undecorated_log_stream
        from vllm.v1.executor._b12x_output import PreparationOutput

        session = self._session
        group = session.shared_read_group if session is not None else None
        if group is not None:
            group.start(session)
            # Construction-time rank output must precede the live panel.
            group._gather(None)
        progress = CheckpointDisplay(
            enabled=enable_tqdm(self.load_config.use_tqdm_on_load),
            stream=undecorated_log_stream(sys.stderr),
        )
        with ExitStack() as stack:
            output = stack.enter_context(PreparationOutput(
                progress.write_output, enabled=progress._enabled and progress._stream.isatty(),
            ))
            if output.stream is not None:
                progress._stream = output.stream
            stack.enter_context(progress)
            stack.callback(output.stop)
            self._progress = progress
            if session is not None:
                session.progress = progress.phase
            if group is not None:
                group.progress = progress.phase
            try:
                super().load_weights(model, model_config)
            finally:
                self._progress = None
                if session is not None:
                    session.progress = None
                if group is not None:
                    group.progress = None

    def _log_loading_time(self):
        session, progress = self._session, self._progress
        if session is not None and progress is not None:
            group = session.shared_read_group
            if group is not None and not group.epoch:
                raise RuntimeError("device checkpoint loading requires vLLM's finish_weight_transfers routing hook")
            session.flush()
            stats = session.stats(flush=False)
            summary = dict(group.summary) if group is not None else dict(
                ranks=1, payload_bytes=stats["payload_bytes"],
                physical_bytes=stats["physical_bytes"],
            )
            summary["load_seconds"] = self.counter_after_loading_weights - self.counter_before_loading_weights
            progress.complete(summary)
            progress.stop()
        super()._log_loading_time()

    @staticmethod
    def _needs_values(entry):
        return math.prod(entry.shape) == 1 or entry.name.rsplit(".", 1)[-1] in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        }

    def _file_backed_weights_iterator(self, files, source, index_path):
        from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight

        weight_map = None
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        indexed_paths = {}
        for path in files:
            sources = weight_utils.safetensors_file_sources(path)
            file_names = {name for name in sources if source.file_weight_filter(name)}
            resolved_path = Path(path).resolve()
            selected = []
            for name in sorted(sources):
                if source.weight_name_prefixes and not name.startswith(
                    source.weight_name_prefixes
                ):
                    continue
                if weight_map is not None:
                    indexed_file = weight_map.get(name)
                    if indexed_file is None:
                        continue
                    if indexed_file not in indexed_paths:
                        indexed_paths[indexed_file] = (
                            index_path.parent / indexed_file
                        ).resolve()
                    if indexed_paths[indexed_file] != resolved_path:
                        continue
                if not should_skip_weight(name, self.local_expert_ids):
                    selected.append(name)
            file_backed = (
                (source.prefix + name, weight_utils.file_source_tensor(sources[name]))
                for name in selected
                if name in file_names
            )
            # File-backed entries never enter the checkpoint direct reader, even
            # for mixed files or before metadata-value reads and routing.
            ordinary = (
                self._session.weights(
                    [path],
                    prefixes=source.weight_name_prefixes,
                    prefix=source.prefix,
                    index_path=index_path,
                    needs_values=self._needs_values,
                    skip=lambda name: (
                        name in file_names
                        or should_skip_weight(name, self.local_expert_ids)
                    ),
                )
                if any(name not in file_names for name in selected)
                else ()
            )
            yield from heapq.merge(file_backed, ordinary, key=lambda item: item[0])

    def _get_weights_iterator(self, source):
        from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight

        if self._session is None:
            raise RuntimeError("b12x requires an active initial-load session")
        folder, files, _ = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            False,
            source.allow_patterns_overrides,
            source.weight_name_prefixes,
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        if self._progress is not None:
            self._progress.source(len(files))

        def progress_files():
            for path in files:
                if self._progress is not None:
                    self._progress.file(Path(path).name)
                yield path
                if self._progress is not None:
                    self._progress.advance(self._session.payload_bytes)

        if source.file_weight_filter is not None:
            yield from self._file_backed_weights_iterator(
                progress_files(),
                source,
                Path(folder) / "model.safetensors.index.json",
            )
            return
        yield from self._session.weights(
            progress_files(),
            prefixes=source.weight_name_prefixes,
            prefix=source.prefix,
            index_path=Path(folder) / "model.safetensors.index.json",
            needs_values=self._needs_values,
            skip=lambda name: should_skip_weight(name, self.local_expert_ids),
        )


def register_b12x_loader():
    from vllm.model_executor.model_loader import register_model_loader

    register_model_loader("b12x")(B12xModelLoader)
