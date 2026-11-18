"""Owner-only creation of the default JIT cache root on Windows.

The default cache root is a short, predictable path at the drive root, so any
local account could create it first and plant files that a later user loads.
``ensure_private_dir`` creates that directory with a DACL that admits the
current user alone and refuses an existing directory that another account owns
or that another account may write to.
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
_SYSTEM_SID = "S-1-5-18"
_CREATOR_OWNER_SID = "S-1-3-0"
_DACL_SECURITY_INFORMATION = 0x00000004
_ACL_SIZE_INFORMATION_CLASS = 2
_ACCESS_ALLOWED_ACE_TYPE = 0
_ACCESS_ALLOWED_CALLBACK_ACE_TYPE = 9
# Rights that let an account change the cache: FILE_WRITE_DATA, FILE_APPEND_DATA,
# FILE_WRITE_EA, FILE_DELETE_CHILD, FILE_WRITE_ATTRIBUTES, DELETE, WRITE_DAC,
# WRITE_OWNER, MAXIMUM_ALLOWED, GENERIC_ALL and GENERIC_WRITE.
_WRITE_RIGHTS = (
    0x00000002
    | 0x00000004
    | 0x00000010
    | 0x00000040
    | 0x00000100
    | 0x00010000
    | 0x00040000
    | 0x00080000
    | 0x02000000
    | 0x10000000
    | 0x40000000
)
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


class _ACL_SIZE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("AceCount", wintypes.DWORD),
        ("AclBytesInUse", wintypes.DWORD),
        ("AclBytesFree", wintypes.DWORD),
    ]


class _ACE_HEADER(ctypes.Structure):
    _fields_ = [
        ("AceType", wintypes.BYTE),
        ("AceFlags", wintypes.BYTE),
        ("AceSize", wintypes.WORD),
    ]


class _ACCESS_ALLOWED_ACE(ctypes.Structure):
    _fields_ = [
        ("Header", _ACE_HEADER),
        ("Mask", wintypes.DWORD),
        ("SidStart", wintypes.DWORD),
    ]


# Offset of the trustee SID inside an access-allowed entry.
_SID_START_OFFSET = ctypes.sizeof(_ACE_HEADER) + ctypes.sizeof(wintypes.DWORD)


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

    def current_user_sid() -> str:
        """The SID of the process token's user, or "" when Windows refuses the query."""
        try:
            return _current_user()[0]
        except OSError:
            return ""

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

    def _foreign_writers(path: str, trusted: frozenset) -> list:
        """SIDs outside ``trusted`` that the DACL of ``path`` grants a write right.

        Entries that apply to children only count as well: the files written
        into the cache inherit them. Deny entries are ignored, so the check can
        refuse a directory a deny entry would have protected, never accept one
        it should refuse.
        """
        dacl = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        error = _advapi32.GetNamedSecurityInfoW(
            path,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION,
            None,
            None,
            ctypes.byref(dacl),
            None,
            ctypes.byref(descriptor),
        )
        if error:
            raise ctypes.WinError(error)
        try:
            if not dacl.value:
                return ["everyone (the directory has no DACL)"]
            information = _ACL_SIZE_INFORMATION()
            _checked(
                _advapi32.GetAclInformation(
                    dacl,
                    ctypes.byref(information),
                    ctypes.sizeof(information),
                    _ACL_SIZE_INFORMATION_CLASS,
                )
            )
            writers = []
            for index in range(information.AceCount):
                ace = ctypes.c_void_p()
                _checked(_advapi32.GetAce(dacl, index, ctypes.byref(ace)))
                entry = ctypes.cast(ace, ctypes.POINTER(_ACCESS_ALLOWED_ACE)).contents
                if entry.Header.AceType not in (
                    _ACCESS_ALLOWED_ACE_TYPE,
                    _ACCESS_ALLOWED_CALLBACK_ACE_TYPE,
                ):
                    continue
                if not entry.Mask & _WRITE_RIGHTS:
                    continue
                sid = _sid_text(
                    ctypes.c_void_p(ctypes.addressof(entry) + _SID_START_OFFSET)
                )
                if sid not in trusted and sid not in writers:
                    writers.append(sid)
            return writers
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
        by the Administrators group passes only from an elevated process), when its
        DACL lets an account other than the user, SYSTEM, Administrators or CREATOR
        OWNER write to it or to the files under it, when it is a junction or
        symbolic link, or when Windows refuses one of the calls.
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
        if owner != user and not (owner == _ADMINISTRATORS_SID and elevated):
            raise RuntimeError(
                f"The FlashInfer cache directory {path} is owned by {owner}, not by "
                f"the current user {user}; remove it or {_OVERRIDE_HINT}."
            )
        trusted = frozenset(
            {user, _SYSTEM_SID, _ADMINISTRATORS_SID, _CREATOR_OWNER_SID}
        )
        writers = _foreign_writers(path, trusted)
        if writers:
            raise RuntimeError(
                f"The FlashInfer cache directory {path} is writable by "
                f"{', '.join(writers)}, not only by the current user {user}; remove "
                f"it or {_OVERRIDE_HINT}."
            )

else:

    def ensure_private_dir(path: pathlib.Path) -> None:
        """No-op: only the Windows default cache root needs an owner-only directory."""
        return

    def current_user_sid() -> str:
        """No-op: only the Windows default cache root is named after the account."""
        return ""
