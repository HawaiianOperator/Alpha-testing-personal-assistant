"""Bounded local Roblox movement demo. No aiming, firing, or combat inputs."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import random
import time
from .human_activity import HumanActivity

SETTINGS = Path(__file__).resolve().parent.parent / "local_settings" / "roblox"
from .windows_game_input import SCAN_CODES as ALLOWED_KEYS, set_key, set_ads, set_fullscreen_key, set_fire, move_camera

def is_roblox_window(hwnd):
    import win32gui
    import win32process
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    process = kernel.OpenProcess(0x1000, False, pid)
    if not process: return None
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return None
        if Path(buffer.value).name.casefold() not in ("robloxplayerbeta.exe", "robloxplayer.exe"):
            return None
    finally:
        kernel.CloseHandle(process)
    return hwnd

def escape_pressed():
    import win32api
    return bool(win32api.GetAsyncKeyState(0x1B) & 0x8000)

def foreground_roblox():
    import win32gui
    return is_roblox_window(win32gui.GetForegroundWindow())

def bring_roblox_to_front():
    import win32gui
    from pywinauto import Desktop
    windows=[]
    def collect(hwnd, _):
        if win32gui.IsWindowVisible(hwnd) and is_roblox_window(hwnd): windows.append(hwnd)
    win32gui.EnumWindows(collect,None)
    if len(windows)!=1:
        raise RuntimeError("Expected one visible Roblox game window. Open Phantom Forces and close extra Roblox windows first.")
    hwnd=windows[0]
    Desktop(backend="win32").window(handle=hwnd).set_focus()
    if foreground_roblox()!=hwnd:
        raise RuntimeError("Windows did not let Roblox take focus. Click its window and retry.")
    return hwnd

def release_keys():
    for name in ALLOWED_KEYS:
        try: set_key(name, False)
        except OSError: pass
    try: set_ads(False)
    except OSError: pass
    try: set_fire(False)
    except OSError: pass

def calibrate(report, game_marker=False):
    import win32gui
    from pywinauto import Desktop
    # Capture the terminal window before calibration brings Roblox forward.
    terminal_window = win32gui.GetForegroundWindow()
    try:
        return _calibrate(report, game_marker)
    finally:
        try:
            if win32gui.IsWindow(terminal_window):
                Desktop(backend="win32").window(handle=terminal_window).set_focus()
        except Exception:
            report("Calibration ended. Windows could not bring the terminal forward; press Alt+Tab to return.")

def _calibrate(report, game_marker=False):
    import win32api
    import win32gui
    from PIL import ImageGrab
    report("Bringing Roblox to the front for calibration...")
    bring_roblox_to_front()
    report("Calibration has no time limit. While deployed in Phantom Forces, hover over a distinctive static game label that stays visible while moving, then press F8 once. Escape cancels." if game_marker else "Calibration has no time limit. Leave the Deploy screen visible, hover over the center of Deploy, then press F8 once. Escape cancels.")
    while True:
        if escape_pressed(): return {"status":"cancelled", "message":"Calibration cancelled."}
        if win32api.GetAsyncKeyState(0x77) & 0x8000:
            hwnd = foreground_roblox()
            if not hwnd:
                return {"status":"needs_attention", "message":"Roblox must be the active window when you press F8."}
            left,top,right,bottom = win32gui.GetWindowRect(hwnd)
            x,y = win32api.GetCursorPos()
            if not (left+45 <= x < right-45 and top+20 <= y < bottom-20):
                return {"status":"needs_attention", "message":"Point at the center of Deploy inside the Roblox window."}
            SETTINGS.mkdir(parents=True, exist_ok=True)
            # Save a small local button patch, not a full-screen recording.
            stem = "game-marker" if game_marker else "deploy"
            ImageGrab.grab(bbox=(x-40,y-15,x+40,y+15)).convert("RGB").save(SETTINGS/(stem+".png"))
            (SETTINGS/(stem+".json")).write_text(json.dumps({"x":x-left,"y":y-top,"width":right-left,"height":bottom-top}))
            if game_marker:
                return {"status":"completed", "message":"Phantom Forces visual marker saved locally. The routine will stop if this marker disappears or the window size changes."}
            return {"status":"completed", "message":"Deploy calibrated. Keep the same game window size and display scale. Type roam phantom forces to run the 30-second demo."}
        time.sleep(0.05)

def deploy(hwnd):
    import win32api
    import win32con
    import win32gui
    from PIL import Image, ImageChops, ImageStat, ImageGrab
    if not (SETTINGS/"deploy.json").is_file():
        return "First type calibrate deploy to identify the Deploy button."
    data=json.loads((SETTINGS/"deploy.json").read_text())
    left,top,right,bottom=win32gui.GetWindowRect(hwnd)
    if (right-left,bottom-top)!=(data["width"],data["height"]):
        return "The Roblox window size changed. Run calibrate deploy again."
    x,y=left+data["x"],top+data["y"]
    current=ImageGrab.grab(bbox=(x-40,y-15,x+40,y+15)).convert("RGB")
    with Image.open(SETTINGS/"deploy.png") as saved:
        difference=ImageStat.Stat(ImageChops.difference(current,saved.convert("RGB")))
    if sum(difference.mean)/3 > 8:
        return "The calibrated Deploy button is not visible. Return to the Deploy screen or recalibrate."
    if foreground_roblox()!=hwnd or escape_pressed(): return "Stopped: Roblox lost focus or Escape was pressed."
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
    return None

def phantom_marker_visible(hwnd):
    import win32gui
    from PIL import Image,ImageGrab,ImageChops,ImageStat
    if not (SETTINGS/"game-marker.json").is_file(): return False
    data=json.loads((SETTINGS/"game-marker.json").read_text())
    left,top,right,bottom=win32gui.GetWindowRect(hwnd)
    if (right-left,bottom-top)!=(data["width"],data["height"]): return False
    x,y=left+data["x"],top+data["y"]
    current=ImageGrab.grab(bbox=(x-40,y-15,x+40,y+15)).convert("RGB")
    with Image.open(SETTINGS/"game-marker.png") as saved:
        return sum(ImageStat.Stat(ImageChops.difference(current,saved.convert("RGB"))).mean)/3 <= 8

def roam(report, stop_event=None, deploy_first=True, bring_to_front=True, random_combat=False, movement_only=False, require_marker=True):
    if bring_to_front:
        report("Step 1: Bringing the Roblox game window to the front...")
        hwnd=bring_roblox_to_front()
    else:
        hwnd=foreground_roblox()
    if not hwnd: return {"status":"needs_attention", "message":"Roblox is not active. No movement started."}
    if deploy_first:
        error=deploy(hwnd)
        if error: return {"status":"needs_attention", "message":error}
        time.sleep(1)
    if require_marker and not phantom_marker_visible(hwnd):
        return {"status":"needs_attention", "message":"Phantom Forces visual marker not found. While deployed, run calibrate phantom forces over a distinctive static game label. No movement started."}
    activity=HumanActivity().start()
    started=time.monotonic()
    active_seconds=0.0
    last_tick=started
    paused=False
    last_countdown=None
    next_random_action=0
    phase=-1
    jump_release=0
    next_jump=0
    reason="30-second test finished."
    def stopped():
        return escape_pressed() or (stop_event and stop_event.is_set()) or foreground_roblox()!=hwnd
    def tap(name):
        if stopped(): return
        set_key(name,True)
        try: time.sleep(0.08)
        finally: set_key(name,False)
    if movement_only:
        report("Roblox is focused. Starting movement and jumping only, without calibration. Escape/focus loss stops; user input pauses for five seconds.")
    else:
        report("Phantom Forces marker matched. Starting the random movement/camera/firing test; no enemy recognition or target tracking. Escape or focus loss stops it." if random_combat else "Step 2: Phantom Forces marker matched. Starting the input test. Escape or focus loss stops it. No attacks.")
    try:
        while active_seconds<30:
            if activity.error:
                reason=activity.error;break
            if stopped(): reason="Stopped by Escape, cancellation, or focus loss.";break
            if require_marker and not phantom_marker_visible(hwnd): reason="Stopped because the Phantom Forces marker disappeared.";break
            now=time.monotonic()
            idle=activity.idle_seconds()
            if idle<5:
                if not paused:
                    release_keys()
                    report("Your mouse/keyboard input was detected. Automatic movement paused.")
                    paused=True
                remaining=max(1,int(5-idle+0.999))
                if remaining!=last_countdown:
                    report(f"Resuming in {remaining} seconds without further input...")
                    last_countdown=remaining
                last_tick=now
                time.sleep(0.05)
                continue
            if paused:
                report("Five seconds without user input. Resuming automatic movement.")
                phase=-1
                paused=False
                last_countdown=None
                last_tick=now
            active_seconds+=max(0,now-last_tick)
            last_tick=now
            elapsed=active_seconds
            if movement_only:
                if now>=next_random_action:
                    release_keys()
                    if stopped() or activity.idle_seconds()<5: continue
                    moving=random.choice(("w","w","a","d",None))
                    if moving:set_key(moving,True)
                    if moving=="w" and random.random()<0.4:set_key("sprint",True)
                    if not stopped() and activity.idle_seconds()>=5 and random.random()<0.25:tap("jump")
                    next_random_action=now+random.uniform(0.7,2)
                time.sleep(0.03)
                continue
            if random_combat:
                if now>=next_random_action:
                    release_keys()
                    if stopped() or activity.idle_seconds()<5: continue
                    tap("primary")
                    if stopped() or activity.idle_seconds()<5: continue
                    moving=random.choice(("w","w","a","s","d",None))
                    if moving:set_key(moving,True)
                    sprint=moving=="w" and random.random()<0.35
                    if sprint:set_key("sprint",True)
                    elif random.random()<0.4:set_ads(True)
                    if stopped() or activity.idle_seconds()<5: continue
                    move_camera(random.randint(-25,25),random.randint(-8,8))
                    if random.random()<0.2:tap("jump")
                    if random.random()<0.1:tap("lower_stance")
                    if not stopped() and activity.idle_seconds()>=5 and random.random()<0.3:
                        set_fire(True)
                        try:time.sleep(0.04)
                        finally:set_fire(False)
                    next_random_action=now+random.uniform(0.7,2.5)
                time.sleep(0.03)
                continue
            new_phase=0 if elapsed<6 else 1 if elapsed<11 else 2 if elapsed<15 else 3 if elapsed<19 else 4 if elapsed<24 else 5
            if new_phase!=phase:
                release_keys();phase=new_phase
                if stopped(): break
                if phase==0:
                    report("Step 3: Running forward (W + Shift)...")
                    set_key("w",True);set_key("sprint",True)
                elif phase==1:
                    report("Step 4: Stopping sprint and aiming down sights (right mouse)...")
                    set_ads(True)
                elif phase==2:
                    report("Step 5: Crouching (C)...")
                    tap("lower_stance")
                elif phase==3:
                    report("Step 6: Lowering stance again to prone (C)...")
                    tap("lower_stance")
                elif phase==4:
                    report("Step 7: Standing and selecting the knife (Shift, then 3). No attack...")
                    tap("sprint");tap("knife")
                else:
                    report("Step 8: Running and jumping with the knife selected...")
                    set_key("w",True);set_key("sprint",True);next_jump=now
            if phase==5 and now>=next_jump:
                if stopped(): break
                set_key("jump",True);jump_release=now+0.1;next_jump=now+2
            if jump_release and now>=jump_release:
                set_key("jump",False);jump_release=0
            time.sleep(0.05)
    finally:
        release_keys()
        activity.stop()
    return {"status":"completed", "message":reason+" All held inputs released. In-game actions have not been visually verified."}

def is_fullscreen(hwnd):
    import win32gui
    import win32api
    left,top,right,bottom=win32gui.GetWindowRect(hwnd)
    monitor=win32api.GetMonitorInfo(win32api.MonitorFromWindow(hwnd,2))["Monitor"]
    return all(abs(actual-expected)<=2 for actual,expected in zip((left,top,right,bottom),monitor))

def ensure_roblox_fullscreen(report):
    report("Bringing Roblox to the front...")
    hwnd=bring_roblox_to_front()
    if not is_fullscreen(hwnd):
        if foreground_roblox()!=hwnd or escape_pressed():
            return hwnd, "Stopped before fullscreen because focus changed or Escape was pressed."
        report("Requesting Roblox fullscreen with F11...")
        set_fullscreen_key(True)
        try: time.sleep(0.08)
        finally: set_fullscreen_key(False)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            if foreground_roblox()!=hwnd or escape_pressed():
                return hwnd, "Stopped because Roblox lost focus or Escape was pressed."
            if is_fullscreen(hwnd): break
            time.sleep(0.1)
    if not is_fullscreen(hwnd):
        return hwnd, "Fullscreen could not be confirmed. Enable fullscreen in Roblox yourself and retry. No movement started."
    return hwnd, None

def fullscreen_and_roam(report):
    hwnd,error=ensure_roblox_fullscreen(report)
    if error:return {"status":"needs_attention", "message":error}
    report("Fullscreen confirmed. Checking the Phantom Forces calibration before movement...")
    if not phantom_marker_visible(hwnd):
        return {"status":"needs_attention", "message":"Roblox is fullscreen. The game marker needs calibration at this size: while deployed, type calibrate phantom forces, hover over the same distinctive static game label and press F8. Then type fullscreen phantom forces again. No movement started."}
    return roam(report,deploy_first=False,bring_to_front=False)

def run_roam_command(text, progress=None):
    report=progress or (lambda message:None)
    try:
        if text.strip().casefold() in ("auto move", "auto move roblox", "move roblox"):
            hwnd,error=ensure_roblox_fullscreen(report)
            if error:return {"status":"needs_attention", "message":error}
            return roam(report,deploy_first=False,bring_to_front=False,movement_only=True,require_marker=False)
        if text.strip().casefold() in ("combat test phantom forces", "random combat phantom forces"):
            hwnd,error=ensure_roblox_fullscreen(report)
            if error:return {"status":"needs_attention", "message":error}
            return roam(report,deploy_first=False,bring_to_front=False,random_combat=True)
        if text.strip().casefold() in ("fullscreen phantom forces", "full screen phantom forces"):
            return fullscreen_and_roam(report)
        if text.strip().casefold()=="calibrate deploy": return calibrate(report)
        if text.strip().casefold()=="calibrate phantom forces": return calibrate(report,game_marker=True)
        return roam(report)
    except KeyboardInterrupt:
        release_keys()
        return {"status":"cancelled", "message":"Stopped; movement keys released."}
    except Exception as error:
        release_keys()
        return {"status":"needs_attention", "message":f"Local movement stopped: {error}"}
