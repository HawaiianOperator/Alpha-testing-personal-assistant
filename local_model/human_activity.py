"""Track physical input timestamps locally, ignoring injected input. No keys or mouse positions are stored."""
import ctypes
from ctypes import wintypes
import threading
import time

class HumanActivity:
    def __init__(self):
        self.last_activity=time.monotonic()-5
        self.ready=threading.Event()
        self.error=None
        self.thread=None
        self.thread_id=None

    def start(self):
        self.thread=threading.Thread(target=self.watch,daemon=True,name="human-input-monitor")
        self.thread.start()
        if not self.ready.wait(3):
            self.stop()
            raise RuntimeError("Physical-input monitor did not start; movement was not enabled.")
        if self.error: raise RuntimeError(self.error)
        return self

    def idle_seconds(self): return max(0,time.monotonic()-self.last_activity)

    def stop(self):
        if self.thread_id:
            user=ctypes.WinDLL("user32",use_last_error=True)
            user.PostThreadMessageW.argtypes=[wintypes.DWORD,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
            user.PostThreadMessageW(self.thread_id,0x0012,0,0)
        if self.thread and self.thread is not threading.current_thread():self.thread.join(timeout=2)

    def watch(self):
        user=ctypes.WinDLL("user32",use_last_error=True)
        kernel=ctypes.WinDLL("kernel32",use_last_error=True)
        result_type=ctypes.c_ssize_t
        callback_type=ctypes.WINFUNCTYPE(result_type,ctypes.c_int,wintypes.WPARAM,wintypes.LPARAM)
        class Keyboard(ctypes.Structure):
            _fields_=[("vkCode",wintypes.DWORD),("scanCode",wintypes.DWORD),("flags",wintypes.DWORD),("time",wintypes.DWORD),("extra",ctypes.c_size_t)]
        class Mouse(ctypes.Structure):
            _fields_=[("point",wintypes.POINT),("mouseData",wintypes.DWORD),("flags",wintypes.DWORD),("time",wintypes.DWORD),("extra",ctypes.c_size_t)]
        user.SetWindowsHookExW.argtypes=[ctypes.c_int,callback_type,wintypes.HINSTANCE,wintypes.DWORD]
        user.SetWindowsHookExW.restype=wintypes.HANDLE
        user.CallNextHookEx.argtypes=[wintypes.HANDLE,ctypes.c_int,wintypes.WPARAM,wintypes.LPARAM]
        user.CallNextHookEx.restype=result_type
        user.UnhookWindowsHookEx.argtypes=[wintypes.HANDLE]
        kernel.GetModuleHandleW.argtypes=[wintypes.LPCWSTR]
        kernel.GetModuleHandleW.restype=wintypes.HMODULE
        hooks=[]
        callbacks=[]
        def make_callback(structure,injected_flag):
            def callback(code,message,data):
                if code>=0:
                    event=ctypes.cast(data,ctypes.POINTER(structure)).contents
                    if not event.flags & injected_flag:
                        self.last_activity=time.monotonic()
                return user.CallNextHookEx(None,code,message,data)
            wrapped=callback_type(callback)
            callbacks.append(wrapped)
            return wrapped
        try:
            self.thread_id=kernel.GetCurrentThreadId()
            message=wintypes.MSG()
            # Create the message queue before another thread may request shutdown.
            user.PeekMessageW(ctypes.byref(message),None,0,0,0)
            module=kernel.GetModuleHandleW(None)
            for kind,structure,flag in ((13,Keyboard,0x10),(14,Mouse,0x01)):
                hook=user.SetWindowsHookExW(kind,make_callback(structure,flag),module,0)
                if not hook:raise ctypes.WinError(ctypes.get_last_error())
                hooks.append(hook)
            self.ready.set()
            while True:
                status=user.GetMessageW(ctypes.byref(message),None,0,0)
                if status<=0:break
                user.TranslateMessage(ctypes.byref(message));user.DispatchMessageW(ctypes.byref(message))
        except Exception as error:
            self.error=f"Physical-input monitoring failed: {error}"
            self.ready.set()
        finally:
            for hook in hooks:user.UnhookWindowsHookEx(hook)
            self.thread_id=None
