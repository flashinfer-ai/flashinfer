# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from flashinfer.utils import _ensure_user_env


_USER_ENV_VARS = ("LOGNAME", "USER", "LNAME", "USERNAME")


def _clear_user_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _USER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("identity_var", _USER_ENV_VARS)
def test_ensure_user_env_preserves_existing_identity(monkeypatch, identity_var):
    _clear_user_env(monkeypatch)
    monkeypatch.setenv(identity_var, "mapped-user")

    _ensure_user_env()

    assert os.environ[identity_var] == "mapped-user"
    assert {
        name: os.environ[name] for name in _USER_ENV_VARS if name in os.environ
    } == {identity_var: "mapped-user"}


def test_ensure_user_env_falls_back_to_numeric_uid(monkeypatch):
    _clear_user_env(monkeypatch)
    monkeypatch.setattr(os, "getuid", lambda: 28686)

    _ensure_user_env()

    assert os.environ["USER"] == "28686"
    assert all(name not in os.environ for name in ("LOGNAME", "LNAME", "USERNAME"))
