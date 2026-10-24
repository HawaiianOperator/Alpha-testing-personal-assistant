"""Opt-in AFK movement with a fresh idle timer after every focus loss."""
import threading
import time
from .roblox_roam import foreground_roblox, escape_pressed, roam, ensure_roblox_fullscreen, phantom_marker_visible

class IdleGate:
    def __init__(self, idle_seconds=0):
        self.idle_seconds = idle_seconds
        self.window = None
        self.since = None
        self.input_tick = None
        self.blocked = False

    def update(self, window, input_tick, now):
        if window is None:
            self.__init__(self.idle_seconds)
            return False
        if window != self.window:
            self.window = window
            self.since = now
            self.input_tick = input_tick
            self.blocked = False
            return self.idle_seconds == 0
        if input_tick != self.input_tick:
            self.input_tick = input_tick
            self.since = now
            self.blocked = False
        return not self.blocked and now - self.since >= self.idle_seconds

    def completed(self, input_tick, now):
        self.blocked = True
        self.input_tick = input_tick
        self.since = now

class AfkRoamer:
    def __init__(self, progress=None):
        self.report = progress or (lambda message:None)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return {"status":"completed", "message":"AFK mode is already enabled."}
        try:
            self.report("Bringing Roblox to the front for the Phantom Forces test...")
            hwnd,error=ensure_roblox_fullscreen(self.report)
            if error:return {"status":"needs_attention", "message":error}
            if not phantom_marker_visible(hwnd):
                return {"status":"needs_attention", "message":"Phantom Forces marker not found. Deploy into the match, then run calibrate phantom forces before enabling AFK mode."}
        except Exception as error:
            return {"status":"needs_attention", "message":f"Could not prepare Roblox: {error}"}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True, name="local-afk-roam")
        self.thread.start()
        return {"status":"completed", "message":"AFK test mode enabled. Switch to Phantom Forces: movement starts immediately when Roblox is focused, with separate run, aim-down-sights, crouch, prone, knife-selection and jump stages for up to 30 seconds. No firing inputs. Mouse or keyboard activity pauses movement until five seconds without input. Focus loss stops the test. Escape disables AFK mode. Type afk roam off to disable it."}

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)

    def watch(self):
        gate = IdleGate()
        try:
            import win32api
            while not self.stop_event.is_set():
                if escape_pressed():
                    self.report("AFK mode disabled by Escape.")
                    break
                window = foreground_roblox()
                if gate.update(window, win32api.GetLastInputInfo(), time.monotonic()):
                    self.report("Roblox is focused. Starting the staged Phantom Forces movement test.")
                    result = roam(self.report, self.stop_event, deploy_first=False, bring_to_front=False)
                    self.report(result["message"])
                    if escape_pressed(): break
                    # Our injected inputs must not trigger another run by themselves.
                    gate.completed(win32api.GetLastInputInfo(), time.monotonic())
                self.stop_event.wait(0.1)
        except Exception as error:
            self.report(f"AFK mode stopped: {error}")
        finally:
            self.stop_event.set()
