import ctypes
import importlib.util
import pathlib
import sys

import pytest

import flashinfer.jit._windows_acl as acl_source

_USER = "S-1-5-21-1000"
_OTHER = "S-1-5-21-2000"
_DIRECTORY = 0x00000010
_PATH = pathlib.Path("C:/_fij_test")


class _FakeDll:
    def __init__(self, name, **kwargs):
        self.name = name

    def __getattr__(self, name):
        def stub(*args):
            return 0

        setattr(self, name, stub)
        return stub


@pytest.fixture
def acl(monkeypatch):
    # The module's Windows branch, loaded on any platform over stub system DLLs
    # into a module object of its own; the directory exists and the user owns it.
    spec = importlib.util.spec_from_file_location(
        "_windows_acl_under_test", acl_source.__file__
    )
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        patch.setattr(ctypes, "WinDLL", _FakeDll, raising=False)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_current_user", lambda: (_USER, False))
    monkeypatch.setattr(module, "_create_private", lambda path, sid: False)
    monkeypatch.setattr(module._kernel32, "GetFileAttributesW", lambda path: _DIRECTORY)
    monkeypatch.setattr(module, "_owner", lambda path: _USER)
    monkeypatch.setattr(module, "_foreign_writers", lambda path, trusted: [])
    return module


@pytest.mark.cpu_only
def test_windows_acl_accepts_the_directory_the_user_owns_alone(acl):
    acl.ensure_private_dir(_PATH)


@pytest.mark.cpu_only
def test_windows_acl_refuses_a_directory_another_account_owns(acl, monkeypatch):
    monkeypatch.setattr(acl, "_owner", lambda path: _OTHER)

    with pytest.raises(RuntimeError, match=f"owned by {_OTHER}, not by the current"):
        acl.ensure_private_dir(_PATH)


@pytest.mark.cpu_only
def test_windows_acl_refuses_a_junction_or_symbolic_link(acl, monkeypatch):
    reparse = _DIRECTORY | acl._FILE_ATTRIBUTE_REPARSE_POINT
    monkeypatch.setattr(acl._kernel32, "GetFileAttributesW", lambda path: reparse)

    with pytest.raises(RuntimeError, match="junction or symbolic link"):
        acl.ensure_private_dir(_PATH)


@pytest.mark.cpu_only
def test_windows_acl_refuses_a_foreign_write_entry(acl, monkeypatch):
    trusted = set()
    monkeypatch.setattr(
        acl, "_foreign_writers", lambda path, sids: trusted.update(sids) or [_OTHER]
    )

    with pytest.raises(RuntimeError, match=f"writable by {_OTHER}, not only by"):
        acl.ensure_private_dir(_PATH)
    assert trusted == {_USER, "S-1-5-18", "S-1-5-32-544", "S-1-3-0"}
