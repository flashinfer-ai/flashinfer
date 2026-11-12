# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-local code-generation options; no process environment overrides."""

from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType

_OPTIONS = ContextVar("kimi_k3_codegen_options", default=MappingProxyType({}))


class _Options:
    def get(self, key, default=None):
        return _OPTIONS.get().get(key, default)

    def __getitem__(self, key):
        return _OPTIONS.get()[key]


environ = _Options()


@contextmanager
def option_scope(options):
    """Freeze the caller's options for construction and compilation."""
    token = _OPTIONS.set(MappingProxyType(dict(options)))
    try:
        yield
    finally:
        _OPTIONS.reset(token)
