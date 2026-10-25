"""Discover local Steam libraries and request launches by installed game title."""
import os
from pathlib import Path
import re
import subprocess
import sys
from .steam_visible import visible_launch

def normalize(title):
    return " ".join(re.findall(r"[a-z0-9]+", title.casefold()))

def steam_root():
    candidates = []
    if sys.platform == "win32":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
                candidates.append(Path(winreg.QueryValueEx(key, "SteamPath")[0]))
        except OSError:
            pass
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        value = os.environ.get(variable)
        if value: candidates.append(Path(value) / "Steam")
    return next((root for root in candidates if (root / "steam.exe").is_file()), None)

def installed_games(root):
    libraries = {root}
    config = root / "steamapps" / "libraryfolders.vdf"
    if config.is_file():
        content = config.read_text(encoding="utf-8", errors="replace")
        for path in re.findall(r'"path"\s*"([^"\n]+)"', content):
            libraries.add(Path(path.replace("\\\\", "\\")))
    games = {}
    for library in libraries:
        for manifest in (library / "steamapps").glob("appmanifest_*.acf"):
            try:
                fields = dict(re.findall(r'"([^"\n]+)"\s*"([^"\n]*)"', manifest.read_text(encoding="utf-8", errors="replace")))
                appid, name, directory = fields.get("appid", ""), fields.get("name", ""), fields.get("installdir", "")
                if appid.isdigit() and name and directory and (library / "steamapps" / "common" / directory).is_dir():
                    games[appid] = {"id":appid, "name":name}
            except OSError:
                continue
    return sorted(games.values(), key=lambda game: game["name"].casefold())

def launch_game(query=None, progress=None):
    report = progress or (lambda message: None)
    if sys.platform != "win32":
        return {"status":"unavailable", "message":"Launch games on your Windows computer."}
    try:
        report("Step 1/6: Checking whether Steam is installed...")
        root = steam_root()
        if root is None:
            return {"status":"unavailable", "message":"Steam was not found. You can type install Steam to install it first."}
        report("Step 2/6: Reading your installed Steam games...")
        games = installed_games(root)
        if query is None:
            names = "\n".join(game["name"] for game in games)
            return {"status":"completed", "message":"Detected Steam games:\n" + (names or "No installed games detected.") + "\nType play followed by the game name."}
        report(f"Step 3/6: Finding {query} in your Steam libraries...")
        wanted = normalize(query)
        if wanted in ("cs2", "counter strike", "counterstrike 2", "counter strike 2"):
            matches = [game for game in games if game["id"] == "730"]
        else:
            matches = [game for game in games if normalize(game["name"]) == wanted]
            if not matches and wanted:
                matches = [game for game in games if wanted in normalize(game["name"])]
        if len(matches) > 1:
            return {"status":"needs_clarification", "question":"Several games match. Type play with the full title:\n" + "\n".join(game["name"] for game in matches)}
        if not matches:
            return {"status":"unavailable", "message":f"I could not find {query} installed in Steam. Install it through your Steam library, or type list games to see detected games."}
        game = matches[0]
        return visible_launch(root, game["id"], game["name"], report)
    except OSError as error:
        return {"status":"unavailable", "message":f"Could not launch the Steam game: {error}"}


def open_steam(progress=None):
    report=progress or (lambda message:None)
    try:
        root=steam_root()
        if root is None:return {"status":"unavailable", "message":"Steam was not found. Type install Steam to install it."}
        report("Opening the Steam library...")
        subprocess.Popen([str(root/"steam.exe"),"steam://nav/games"],shell=False)
        return {"status":"launch_requested", "command":"open_steam", "message":"Steam category opened. What game would you like to play? Type its name, list games, or back to assistant."}
    except OSError as error:
        return {"status":"unavailable", "message":f"Could not open Steam: {error}"}
