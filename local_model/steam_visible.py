"""Local-only visible Steam launch using Windows accessibility controls."""
from pathlib import Path
import subprocess
import time

def normalize(text):
    import re
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

def inspect_controls(window, title):
    controls = window.descendants()
    names = [(control, normalize(control.window_text())) for control in controls if control.is_visible()]
    wanted = normalize(title)
    title_found = any(wanted == name or wanted in name for _, name in names if name)
    buttons = []
    seen = set()
    diagnostics = []
    for control, name in names:
        kind = control.element_info.control_type
        if name in ("play", "launch", "play " + wanted) or name.startswith("play " + wanted + " "):
            rect = control.rectangle()
            box = (rect.left, rect.top, rect.right, rect.bottom)
            diagnostics.append({"label":control.window_text(), "type":kind, "enabled":control.is_enabled(), "bounds":box})
            # Chromium may expose Steam's Play label as Text rather than Button.
            if kind not in ("Button", "Hyperlink", "Text", "Group", "Custom") or not control.is_enabled():
                continue
            if rect.width() <= 0 or rect.height() <= 0:
                continue
            # Nested labels describe the same visible button; keep one click target.
            center = (int((rect.left + rect.right)/2), int((rect.top + rect.bottom)/2))
            if center not in seen:
                seen.add(center)
                buttons.append(control)
    # Merge nested targets whose centers fall inside the same button rectangle.
    unique = []
    for button in buttons:
        rect = button.rectangle()
        def encloses(a, b):
            return a.left <= b.left and a.top <= b.top and a.right >= b.right and a.bottom >= b.bottom
        if any(encloses(other.rectangle(), rect) or encloses(rect, other.rectangle()) for other in unique):
            continue
        unique.append(button)
    return (unique[0] if title_found and len(unique) == 1 else None), {
        "visible_controls":len(names), "game_title_found":title_found, "play_candidates":diagnostics,
        "unique_play_targets":len(unique)}

def select_play(window, title):
    return inspect_controls(window, title)[0]

def save_diagnostics(data):
    import json
    destination = Path(__file__).resolve().parent.parent / "logs" / "steam-controls.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")

def visible_launch(root, appid, title, report):
    try:
        from pywinauto import Desktop
    except ImportError:
        return {"status":"unavailable", "message":"Visible Steam control needs its local dependency. Run: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-desktop.txt"}
    try:
        report(f"Step 4/6: Opening Steam's library page for {title}...")
        subprocess.Popen([str(Path(root) / "steam.exe"), f"steam://nav/games/details/{appid}"], shell=False)
        desktop = Desktop(backend="uia")
        deadline = time.monotonic() + 35
        focused = False
        diagnostics = {"steam_windows":0}
        while time.monotonic() < deadline:
            windows = desktop.windows(title="Steam", visible_only=True)
            diagnostics["steam_windows"] = len(windows)
            if len(windows) == 1:
                window = windows[0]
                if not focused:
                    window.set_focus()
                    report("Step 5/6: Showing Steam and locating the game's Play button...")
                    focused = True
                button, observed = inspect_controls(window, title)
                diagnostics.update(observed)
                if button is not None:
                    # Re-read controls after focusing; don't reuse old coordinates.
                    window.set_focus()
                    button = select_play(window, title)
                    if button is None:
                        continue
                    report(f"Step 6/6: Moving the pointer to Play and clicking for {title}...")
                    button.move_mouse_input()
                    time.sleep(0.8)
                    button = select_play(window, title)
                    if button is None:
                        continue
                    button.click_input()
                    return {"status":"launch_requested", "command":"play_game", "game":title,
                            "message":f"Clicked Play for {title} in Steam. Steam may still show updates or launch options; the game running has not been verified."}
            time.sleep(0.5)
        save_diagnostics(diagnostics)
        if diagnostics.get("visible_controls", 0) < 5:
            detail = "Steam did not expose its game-page controls to Windows accessibility."
        elif not diagnostics.get("game_title_found"):
            detail = "The requested game's title was not found in Steam's visible controls."
        else:
            detail = "Steam did not expose one enabled Play target."
        return {"status":"needs_attention", "message":detail + " No click was made. Diagnostic details are in logs/steam-controls.json. Open the game's Library page in Steam, then retry."}
    except Exception as error:
        return {"status":"needs_attention", "message":f"Visible Steam control stopped: {error}. Check Steam before retrying; a click may already have occurred."}
