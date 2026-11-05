"""Owner-only creation of the default JIT cache root on Windows.

The default cache root is a short, predictable path at the drive root, so any
local account could create it first and plant files that a later user loads.
``ensure_private_dir`` creates that directory with a DACL that admits the
current user alone and refuses an existing directory that another account owns.
"""

import ctypes
import pathlib
import sys
from ctypes import wintypes

_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_TOKEN_ELEVATION = 20
_SDDL_REVISION_1 = 1
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x00000001
_ERROR_ALREADY_EXISTS = 183
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_ADMINISTRATORS_SID = "S-1-5-32-544"
_OVERRIDE_HINT = "set FLASHINFER_CACHE_DIR to a directory of your own"


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _TOKEN_USER_INFO(ctypes.Structure):
    _fields_ = [("User", _SID_AND_ATTRIBUTES)]


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


if sys.platform == "win32":
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetFileAttributesW.restype = wintypes.DWORD

    def _checked(result: int) -> None:
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())

    def _sid_text(sid: ctypes.c_void_p) -> str:
        text = wintypes.LPWSTR()
        _checked(_advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)))
        try:
            return text.value or ""
        finally:
            _kernel32.LocalFree(text)

    def _current_user() -> tuple[str, bool]:
        """Return the SID of the process token's user and whether the token is elevated."""
        token = wintypes.HANDLE()
        process = wintypes.HANDLE(_kernel32.GetCurrentProcess())
        _checked(_advapi32.OpenProcessToken(process, _TOKEN_QUERY, ctypes.byref(token)))
        try:
            size = wintypes.DWORD()
            _advapi32.GetTokenInformation(
                token, _TOKEN_USER, None, 0, ctypes.byref(size)
            )
            user = ctypes.create_string_buffer(size.value)
            _checked(
                _advapi32.GetTokenInformation(
                    token, _TOKEN_USER, user, size, ctypes.byref(size)
                )
            )
            sid = ctypes.cast(user, ctypes.POINTER(_TOKEN_USER_INFO)).contents.User.Sid
            elevation = wintypes.DWORD()
            _checked(
                _advapi32.GetTokenInformation(
                    token,
                    _TOKEN_ELEVATION,
                    ctypes.byref(elevation),
                    ctypes.sizeof(elevation),
                    ctypes.byref(size),
                )
            )
            return _sid_text(ctypes.c_void_p(sid)), bool(elevation.value)
        finally:
            _kernel32.CloseHandle(token)

    def _owner(path: str) -> str:
        owner = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        error = _advapi32.GetNamedSecurityInfoW(
            path,
            _SE_FILE_OBJECT,
            _OWNER_SECURITY_INFORMATION,
            ctypes.byref(owner),
            None,
            None,
            None,
            ctypes.byref(descriptor),
        )
        if error:
            raise ctypes.WinError(error)
        try:
            return _sid_text(owner)
        finally:
            _kernel32.LocalFree(descriptor)

    def _create_private(path: str, sid: str) -> bool:
        """Create ``path`` owned by ``sid`` with full control for ``sid`` alone; False if it exists."""
        descriptor = ctypes.c_void_p()
        sddl = f"O:{sid}D:P(A;OICI;FA;;;{sid})"
        _checked(
            _advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl, _SDDL_REVISION_1, ctypes.byref(descriptor), None
            )
        )
        try:
            attributes = _SECURITY_ATTRIBUTES(
                ctypes.sizeof(_SECURITY_ATTRIBUTES), descriptor.value, False
            )
            if _kernel32.CreateDirectoryW(path, ctypes.byref(attributes)):
                return True
            error = ctypes.get_last_error()
        finally:
            _kernel32.LocalFree(descriptor)
        if error != _ERROR_ALREADY_EXISTS:
            raise ctypes.WinError(error)
        return False

    def ensure_private_dir(path: pathlib.Path) -> None:
        """Create ``path`` for the current user alone, or check that the current user owns it.

        Raises ``RuntimeError`` when another account owns the directory (one owned
        by the Administrators group passes only from an elevated process), when it
        is a junction or symbolic link, or when Windows refuses one of the calls.
        """
        try:
            _ensure_private_dir_win32(str(path))
        except OSError as error:
            raise RuntimeError(
                f"Cannot secure the FlashInfer cache directory {path}: {error}; "
                f"{_OVERRIDE_HINT}."
            ) from error

    def _ensure_private_dir_win32(path: str) -> None:
        user, elevated = _current_user()
        if _create_private(path, user):
            return
        attributes = _kernel32.GetFileAttributesW(path)
        if attributes == _INVALID_FILE_ATTRIBUTES:
            raise ctypes.WinError(ctypes.get_last_error())
        if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise RuntimeError(
                f"The FlashInfer cache directory {path} is a junction or symbolic "
                f"link; remove it or {_OVERRIDE_HINT}."
            )
        owner = _owner(path)
        if owner == user or (owner == _ADMINISTRATORS_SID and elevated):
            return
        raise RuntimeError(
            f"The FlashInfer cache directory {path} is owned by {owner}, not by the "
            f"current user {user}; remove it or {_OVERRIDE_HINT}."
        )

else:

    def ensure_private_dir(path: pathlib.Path) -> None:
        """No-op: only the Windows default cache root needs an owner-only directory."""
        return
