"""Documented Windows virtual-desktop membership check.

Only the public ``IVirtualDesktopManager`` interface is used.  Dexter does not
enumerate, switch, name, or otherwise manipulate virtual desktops.
"""

from __future__ import annotations

import ctypes
import os
import uuid
from ctypes import wintypes


HRESULT = ctypes.c_int32
ULONG = ctypes.c_uint32
_CALL = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)

COINIT_APARTMENTTHREADED = 0x2
CLSCTX_INPROC_SERVER = 0x1
S_OK = 0
S_FALSE = 1
RPC_E_CHANGED_MODE = -2147417850


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def parse(cls, value: str) -> "GUID":
        parsed = uuid.UUID(value)
        return cls(
            parsed.time_low,
            parsed.time_mid,
            parsed.time_hi_version,
            (ctypes.c_ubyte * 8)(*parsed.bytes[8:]),
        )


_QueryInterface = _CALL(
    HRESULT,
    ctypes.c_void_p,
    ctypes.POINTER(GUID),
    ctypes.POINTER(ctypes.c_void_p),
)
_AddRef = _CALL(ULONG, ctypes.c_void_p)
_Release = _CALL(ULONG, ctypes.c_void_p)
_IsWindowOnCurrentVirtualDesktop = _CALL(
    HRESULT,
    ctypes.c_void_p,
    wintypes.HWND,
    ctypes.POINTER(wintypes.BOOL),
)
_GetWindowDesktopId = _CALL(
    HRESULT,
    ctypes.c_void_p,
    wintypes.HWND,
    ctypes.POINTER(GUID),
)
_MoveWindowToDesktop = _CALL(
    HRESULT,
    ctypes.c_void_p,
    wintypes.HWND,
    ctypes.POINTER(GUID),
)


class _VirtualDesktopManagerVTable(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", _QueryInterface),
        ("AddRef", _AddRef),
        ("Release", _Release),
        ("IsWindowOnCurrentVirtualDesktop", _IsWindowOnCurrentVirtualDesktop),
        ("GetWindowDesktopId", _GetWindowDesktopId),
        ("MoveWindowToDesktop", _MoveWindowToDesktop),
    ]


class _VirtualDesktopManagerInterface(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_VirtualDesktopManagerVTable))]


CLSID_VIRTUAL_DESKTOP_MANAGER = GUID.parse(
    "AA509086-5CA9-4C25-8F95-589D3C07B48A"
)
IID_IVIRTUAL_DESKTOP_MANAGER = GUID.parse(
    "A5CD92FF-29BE-454C-8D04-D82879FB3F1B"
)


class VirtualDesktopManager:
    """Apartment-local wrapper around ``IVirtualDesktopManager``."""

    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Virtual desktops are available on Windows only.")

        self._ole32 = ctypes.OleDLL("ole32")
        self._initialized_here = False
        self._interface: ctypes.POINTER(_VirtualDesktopManagerInterface) | None = None

        self._ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        self._ole32.CoInitializeEx.restype = HRESULT
        self._ole32.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._ole32.CoCreateInstance.restype = HRESULT
        self._ole32.CoUninitialize.argtypes = []
        self._ole32.CoUninitialize.restype = None

        initialized = int(self._ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED))
        if initialized in (S_OK, S_FALSE):
            self._initialized_here = True
        elif initialized != RPC_E_CHANGED_MODE:
            raise OSError(f"CoInitializeEx failed with HRESULT 0x{initialized & 0xFFFFFFFF:08X}.")

        raw_interface = ctypes.c_void_p()
        result = int(
            self._ole32.CoCreateInstance(
                ctypes.byref(CLSID_VIRTUAL_DESKTOP_MANAGER),
                None,
                CLSCTX_INPROC_SERVER,
                ctypes.byref(IID_IVIRTUAL_DESKTOP_MANAGER),
                ctypes.byref(raw_interface),
            )
        )
        if result < 0 or not raw_interface.value:
            self.close()
            raise OSError(
                "Could not create IVirtualDesktopManager "
                f"(HRESULT 0x{result & 0xFFFFFFFF:08X})."
            )
        self._interface = ctypes.cast(
            raw_interface,
            ctypes.POINTER(_VirtualDesktopManagerInterface),
        )

    def is_window_on_current_desktop(self, hwnd: int) -> bool:
        interface = self._interface
        if interface is None:
            raise RuntimeError("VirtualDesktopManager is closed.")
        on_current = wintypes.BOOL()
        result = int(
            interface.contents.lpVtbl.contents.IsWindowOnCurrentVirtualDesktop(
                ctypes.cast(interface, ctypes.c_void_p),
                wintypes.HWND(hwnd),
                ctypes.byref(on_current),
            )
        )
        if result < 0:
            raise OSError(
                "IsWindowOnCurrentVirtualDesktop failed "
                f"(HRESULT 0x{result & 0xFFFFFFFF:08X})."
            )
        return bool(on_current.value)

    def close(self) -> None:
        interface = self._interface
        self._interface = None
        if interface is not None:
            interface.contents.lpVtbl.contents.Release(
                ctypes.cast(interface, ctypes.c_void_p)
            )
        if self._initialized_here:
            self._initialized_here = False
            self._ole32.CoUninitialize()

    def __enter__(self) -> "VirtualDesktopManager":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
