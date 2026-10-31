# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Compatibility import for tests.test_helpers.cake_kda_det_train_reference (the same module object)."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("tests.test_helpers.cake_kda_det_train_reference")
