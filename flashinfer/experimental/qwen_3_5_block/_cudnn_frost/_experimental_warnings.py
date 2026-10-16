# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

_experimental_api_warnings_emitted = set()


def warn_experimental_api_once(logger: logging.Logger, api_name: str) -> None:
    """Emit the experimental API warning once per API class per process."""
    if api_name in _experimental_api_warnings_emitted:
        return

    _experimental_api_warnings_emitted.add(api_name)

    logger.warning("%s is an experimental API", api_name)


def _reset_experimental_api_warning_registry() -> None:
    """Reset experimental API warning state for tests."""
    _experimental_api_warnings_emitted.clear()
