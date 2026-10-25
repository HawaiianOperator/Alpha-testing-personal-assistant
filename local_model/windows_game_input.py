"""Checked scan-code input for the bounded local game routine."""
import ctypes
from ctypes import wintypes

class MouseInput(ctypes.Structure):
    _fields_=[("dx",wintypes.LONG),("dy",wintypes.LONG),("mouseData",wintypes.DWORD),("dwFlags",wintypes.DWORD),("time",wintypes.DWORD),("dwExtraInfo",ctypes.c_size_t)]
class KeyboardInput(ctypes.Structure):
    _fields_=[("wVk",wintypes.WORD),("wScan",wintypes.WORD),("dwFlags",wintypes.DWORD),("time",wintypes.DWORD),("dwExtraInfo",ctypes.c_size_t)]
class HardwareInput(ctypes.Structure):
    _fields_=[("uMsg",wintypes.DWORD),("wParamL",wintypes.WORD),("wParamH",wintypes.WORD)]
class Payload(ctypes.Union):
    _fields_=[("mi",MouseInput),("ki",KeyboardInput),("hi",HardwareInput)]
class Input(ctypes.Structure):
    _fields_=[("type",wintypes.DWORD),("payload",Payload)]

SCAN_CODES={"primary":0x02,"w":0x11,"a":0x1e,"s":0x1f,"d":0x20,"jump":0x39,"sprint":0x2a,"lower_stance":0x2e,"knife":0x04}

def send(event):
    user=ctypes.WinDLL("user32",use_last_error=True)
    user.SendInput.argtypes=[wintypes.UINT,ctypes.POINTER(Input),ctypes.c_int]
    user.SendInput.restype=wintypes.UINT
    if user.SendInput(1,ctypes.byref(event),ctypes.sizeof(Input))!=1:
        raise OSError(ctypes.get_last_error(),"Windows rejected the game input. Check that Roblox and the assistant run at the same privilege level.")

def keyboard_event(name,down):
    event=Input();event.type=1
    event.payload.ki=KeyboardInput(0,SCAN_CODES[name],0x0008 | (0 if down else 0x0002),0,0)
    return event

def set_key(name,down): send(keyboard_event(name,down))

def set_ads(down):
    event=Input();event.type=0
    event.payload.mi=MouseInput(0,0,0,0x0008 if down else 0x0010,0,0)
    send(event)


def set_fullscreen_key(down):
    event=Input();event.type=1
    event.payload.ki=KeyboardInput(0,0x57,0x0008 | (0 if down else 0x0002),0,0)
    send(event)


def set_fire(down):
    event=Input();event.type=0
    event.payload.mi=MouseInput(0,0,0,0x0002 if down else 0x0004,0,0)
    send(event)

def move_camera(dx,dy):
    event=Input();event.type=0
    event.payload.mi=MouseInput(int(dx),int(dy),0,0x0001,0,0)
    send(event)
