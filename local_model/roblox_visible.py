"""Local accessibility-driven Roblox search and Play; no direct experience launch links."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import time

TITLE="Phantom Forces"
CREATOR="stylis studios"

def process_name(pid):
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:return ""
    try:
        buffer=ctypes.create_unicode_buffer(32768);size=wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle,0,buffer,ctypes.byref(size)):return ""
        return Path(buffer.value).name.casefold()
    finally:kernel.CloseHandle(handle)

def text(control):return " ".join(control.window_text().casefold().split())

def window_for(desktop,browser=False):
    allowed={"msedge.exe","chrome.exe","firefox.exe","brave.exe"} if browser else {"robloxplayerbeta.exe","robloxplayer.exe"}
    windows=[window for window in desktop.windows(visible_only=True)
             if process_name(window.process_id()) in allowed
             and (not browser or "roblox" in text(window))]
    return windows[0] if len(windows)==1 else None

def unique(controls):
    targets=[];seen=set()
    for control in controls:
        if not control.is_visible() or not control.is_enabled():continue
        rect=control.rectangle();key=(rect.left,rect.top,rect.right,rect.bottom)
        if rect.width()>0 and rect.height()>0 and key not in seen:
            targets.append(control);seen.add(key)
    return targets[0] if len(targets)==1 else None

def find_search(window):
    return unique([control for control in window.descendants(control_type="Edit")
                   if "search" in text(control) and "address" not in text(control)])

def visual_search_point(window):
    """Match only the generic search-box patch from the supplied app screenshot."""
    from PIL import Image,ImageChops,ImageStat
    captured=window.capture_as_image().convert("RGB")
    width,height=captured.size
    if abs(width/height-1920/1080)>0.12:return None
    # Compare at the screenshot's reference scale. Screenshots remain in memory.
    scaled=captured.resize((1920,1080))
    observed=scaled.crop((752,12,1254,59))
    with Image.open(Path(__file__).parent/"assets"/"roblox-search-reference.png") as reference:
        error=sum(ImageStat.Stat(ImageChops.difference(observed,reference.convert("RGB"))).mean)/3
    if error>8:return None
    return int(1000*width/1920),int(35*height/1080)

def require_focus(window):
    import win32gui
    from .roblox_roam import escape_pressed
    if escape_pressed() or win32gui.GetForegroundWindow()!=window.handle:
        raise InterruptedError("Stopped because focus changed or Escape was pressed")

def type_literal(window,value):
    from .windows_game_input import Input,KeyboardInput,send
    encoded=value.encode("utf-16-le")
    for offset in range(0,len(encoded),2):
        require_focus(window)
        unit=int.from_bytes(encoded[offset:offset+2],"little")
        for flags in (0x0004,0x0006):
            event=Input();event.type=1;event.payload.ki=KeyboardInput(0,unit,flags,0,0)
            send(event)
        time.sleep(0.03)

class SearchResultsUnavailable(RuntimeError):pass

def verified_game_link(window,title=TITLE,creator=CREATOR):
    matches=[]
    for control in window.descendants():
        if text(control)!=title.casefold() or control.element_info.control_type not in ("Hyperlink","Button"):
            continue
        parent=control
        for _ in range(4):
            if parent is None:break
            descendants=parent.descendants()
            # Do not use a whole search-results page as proof of a tile's creator.
            title_links=[child for child in descendants if text(child)==title.casefold()
                         and child.element_info.control_type in ("Hyperlink","Button")]
            labels=[text(parent)]+[text(child) for child in descendants]
            if len(descendants)<=60 and len(title_links)<=1 and (not creator or any(creator.casefold() in label for label in labels)):
                matches.append(control)
                break
            parent=parent.parent()
    return unique(matches)

def find_play(window,title=TITLE,creator=CREATOR):
    controls=window.descendants()
    labels=[text(control) for control in controls if control.is_visible()]
    if not any(title.casefold()==label for label in labels) or not (not creator or any(creator.casefold() in label for label in labels)):return None
    return unique([control for control in controls if control.element_info.control_type in ("Button","Hyperlink")
                   and text(control) in ("play","play "+title.casefold())])

def wait_for(getter,seconds=12):
    from .roblox_roam import escape_pressed
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if escape_pressed():raise InterruptedError("Stopped by Escape")
        found=getter()
        if found is not None:return found
        time.sleep(0.25)
    raise RuntimeError("The expected accessible control did not appear.")

def search_and_play(desktop,report,browser=False,title=TITLE,creator=CREATOR):
    window=wait_for(lambda:window_for(desktop,browser))
    window.set_focus()
    report("Finding the Roblox search box...")
    try:
        search=wait_for(lambda:find_search(window),seconds=3)
    except RuntimeError:
        search=None
    if search is not None:
        search.click_input()
        report(f"Typing {title} into Roblox search...")
        require_focus(window)
        search.set_edit_text(title)
        require_focus(window)
        search.type_keys('{ENTER}',set_foreground=True)
    elif not browser:
        point=visual_search_point(window)
        if point is None:
            raise RuntimeError("The app did not expose a Search field, and its visible search area did not match the supplied screenshot.")
        report("The visible Search box matches your screenshot. Clicking it...")
        require_focus(window)
        window.click_input(coords=point)
        require_focus(window)
        window.type_keys('^a',set_foreground=True)
        report(f"Typing {title} into the visible Search box...")
        type_literal(window,title)
        require_focus(window)
        window.type_keys('{ENTER}',set_foreground=True)
    else:
        raise RuntimeError("No accessible Roblox search field was found in the browser.")
    report(f"Waiting for {title}" + (f" by {creator}" if creator else "") + " in the search results...")
    try:
        game=wait_for(lambda:verified_game_link(window,title,creator))
    except RuntimeError:
        raise SearchResultsUnavailable("The game name was typed and submitted, but the search result was not exposed as an accessible link. Send a screenshot of the search-results screen so its next step can be configured.")
    game.click_input()
    report("Opening the game page and identifying its Play button...")
    # Browser titles may change when navigation finishes; re-select a current window.
    window=wait_for(lambda:window_for(desktop,browser))
    play=wait_for(lambda:find_play(window,title,creator))
    window.set_focus()
    play=wait_for(lambda:find_play(window,title,creator))
    report(f"I am going to play {title}. Clicking Play now...")
    play.click_input()
    return {"status":"launch_requested", "command":"play_roblox_game",
            "message":f"Searched Roblox for {title}, selected the matching result, and clicked Play. Approve browser launch prompts yourself if shown. Joining the game has not been verified."}

def play_visible(find_target,report,title=TITLE,creator=CREATOR):
    try:
        from pywinauto import Desktop
        desktop=Desktop(backend="uia")
        report("Step 1: Trying to open the Roblox app...")
        try:
            target=find_target()
            if not target:raise RuntimeError("Roblox app launcher not found")
            os.startfile(target)
            return search_and_play(desktop,report,title=title,creator=creator)
        except SearchResultsUnavailable as error:
            return {"status":"needs_attention", "message":str(error)}
        except InterruptedError as error:
            return {"status":"cancelled", "message":str(error)}
        except Exception as error:
            report(f"The app flow could not finish: {error}")
        report("Step 2: Trying Roblox.com in your default browser instead...")
        os.startfile("https://www.roblox.com/")
        return search_and_play(desktop,report,browser=True,title=title,creator=creator)
    except InterruptedError:
        return {"status":"cancelled", "message":"Roblox walkthrough stopped by Escape."}
    except Exception as error:
        return {"status":"needs_attention", "message":f"The visible search-and-play flow stopped: {error}. Check Roblox's search results, sign in yourself if needed, then retry. No direct game link was used. If Play was already clicked, check the game before retrying."}


def diagnose_roblox():
    """Read-only structural diagnostics; no screenshots or typed content are saved."""
    try:
        from pywinauto import Desktop
        desktop=Desktop(backend="uia")
        apps=[];browsers=[]
        for window in desktop.windows(visible_only=True):
            name=process_name(window.process_id())
            if name in ("robloxplayerbeta.exe","robloxplayer.exe"):
                apps.append(window)
            elif name in ("msedge.exe","chrome.exe","firefox.exe","brave.exe"):
                browsers.append(window)
        lines=[f"Roblox app windows found: {len(apps)}",f"Browser windows found: {len(browsers)}"]
        for index,window in enumerate(apps,1):
            lines.append(f"App window {index}:")
            try:
                edits=window.descendants(control_type="Edit")
                visible=[control for control in edits if control.is_visible()]
                lines.append(f"  Visible text fields: {len(visible)}")
                lines.append(f"  Accessible search match: {find_search(window) is not None}")
            except Exception as error:lines.append(f"  Accessibility check failed: {type(error).__name__}")
            try:
                image=window.capture_as_image()
                lines.append(f"  Window image size: {image.width} x {image.height}")
                lines.append(f"  Visual search match: {visual_search_point(window) is not None}")
            except Exception as error:lines.append(f"  Visual check failed: {type(error).__name__}")
        if not apps:lines.append("No supported Roblox player window was found; the app process may use a different name or be inaccessible.")
        return {"status":"completed","command":"roblox_diagnostic","message":"\n".join(lines)}
    except Exception as error:
        return {"status":"needs_attention","message":f"Roblox diagnostic failed: {type(error).__name__}: {error}"}
