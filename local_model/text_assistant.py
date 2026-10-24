"""Terminal text assistant: learned intents, bounded tools, no third-party dependencies."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlencode
from .commands import compute, REJECTION_MESSAGE, MAX_REQUEST_BYTES
from .text_features import DIMENSIONS, LABELS, features
from .terminal_input import read_input
from .roblox_install import install_and_open
from .app_manager import AppManager
from .game_launcher import launch_game
from .memory_usage import process_memory
from .roblox_roam import run_roam_command
from .afk_roam import AfkRoamer
from .codex_handoff import prepare_request
from .manual_guide import ManualGuide
from .debug_log import create_debug_logger, record_performance

class IntentModel:
    def __init__(self,path):
        model=json.loads(Path(path).read_text(encoding="utf-8"))
        if model.get("format")!="hashed-ngram-softmax-v1" or model.get("labels")!=list(LABELS) or model.get("dimensions")!=DIMENSIONS:
            raise ValueError("Incompatible model format.")
        self.weights=model["weights"]; self.bias=model["bias"]
        if len(self.weights)!=3 or any(len(row)!=DIMENSIONS for row in self.weights) or len(self.bias)!=3:
            raise ValueError("Incorrect weight shape.")
        if not all(math.isfinite(v) for row in self.weights for v in row) or not all(math.isfinite(v) for v in self.bias):
            raise ValueError("Non-finite model parameters.")
        self.threshold=float(model["min_probability"]); self.margin=float(model["min_margin"])
        if not 0 <= self.threshold <= 1 or not 0 <= self.margin <= 1: raise ValueError("Invalid thresholds.")
    def predict(self,text):
        f=features(text)
        scores=[b+sum(row[k]*v for k,v in f.items()) for row,b in zip(self.weights,self.bias)]
        peak=max(scores); exp=[math.exp(s-peak) for s in scores]; total=sum(exp)
        probabilities=[v/total for v in exp]; order=sorted(range(3),key=lambda i:probabilities[i],reverse=True)
        best,second=order[:2]
        accepted=probabilities[best]>=self.threshold and probabilities[best]-probabilities[second]>=self.margin
        return {"intent":LABELS[best] if accepted else "reject","probability":probabilities[best],
                "margin":probabilities[best]-probabilities[second]}

def rejected():
    return {"status":"rejected","message":REJECTION_MESSAGE}

# Extract arguments only after learned intent recognition. Unsupported language is
# refused rather than interpreted as an action sequence.
PREFIX=r"(?:(?:please|can you|could you|would you)\s+)?"
SEARCH=re.compile(r"^"+PREFIX+r"(?:search(?:\s+google)?(?:\s+for)?|google|look\s+up|find\s+information\s+about)(?:\s*:\s*|\s+|$)(.*?)\s*$",re.I|re.S)
CALCULATE=re.compile(r"^"+PREFIX+r"(?:compute|calculate|evaluate|work\s+out|what\s+is|what's)(?:\s*:\s*|\s+|$)(.*?)\s*$",re.I|re.S)
MULTISTEP=re.compile(r"\b(?:then|also|and\s+(?:buy|install|download|email|send|open|delete|schedule|generate|calculate|compute|search))\b",re.I)

def handle(text,model,pending=None):
    if not isinstance(text,str): raise TypeError("Request must be text.")
    if len(text)>MAX_REQUEST_BYTES or len(text.encode("utf-8"))>MAX_REQUEST_BYTES: return rejected()
    text=text.strip()
    if not text: return {"status":"needs_clarification","command":None,"question":"What would you like me to do?"}
    if pending in ("search","compute"): text=pending+" "+text
    elif re.fullmatch(r"[0-9\s.+*/()\-]+", text) and re.search(r"[0-9]", text):
        text = "calculate " + text
    if len(text.encode("utf-8"))>MAX_REQUEST_BYTES or MULTISTEP.search(text): return rejected()
    prediction=model.predict(text)
    intent=prediction["intent"]
    if intent=="reject": return rejected()
    match=(SEARCH if intent=="search" else CALCULATE).fullmatch(text)
    if not match: return rejected()
    argument=match[1].strip()
    if intent=="search":
        argument=re.sub(r"\s+on\s+google\s*$","",argument,flags=re.I).strip()
    if not argument:
        return {"status":"needs_clarification","command":intent,"question":
            "What would you like to search for?" if intent=="search" else "What would you like me to calculate?"}
    if intent=="compute":
        try: value=compute(argument.rstrip("? "))
        except ValueError: return rejected()
        return {"status":"completed","command":"compute","value":value}
    return {"status":"ready","command":"search","query":argument,
        "url":"https://www.google.com/search?"+urlencode({"q":argument}),
        "message":"Search prepared. A connected computer/browser must open this URL; the offline board has not searched the web."}

def roblox_target():
    """Find an installed player or registered URI, avoiding the Store fallback."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        versions = Path(local) / "Roblox" / "Versions"
        if versions.is_dir():
            players = sorted(versions.glob("*/RobloxPlayerBeta.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
            if players:
                return str(players[0])
    import winreg
    for protocol in ("roblox", "roblox-player"):
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, protocol + r"\shell\open\command") as key:
                command = winreg.QueryValueEx(key, "")[0]
                if isinstance(command, str) and command.strip():
                    return protocol + ":"
        except OSError:
            continue
    return None

def open_roblox(progress=None):
    report = progress or (lambda message: None)
    """Launch Roblox only when an installed launcher can be found."""
    if sys.platform != "win32":
        return {"status":"unavailable", "message":"Open Roblox on your Windows computer; this device cannot launch it."}
    try:
        report("Step 1/2: Checking for an installed Roblox launcher...")
        target = roblox_target()
        if target is None:
            raise FileNotFoundError("Roblox launcher not found")
        report("Step 2/2: Asking Windows to open Roblox...")
        os.startfile(target)
    except OSError:
        return {"status":"needs_clarification", "command":"confirm_roblox_install", "question":"Hey, it looks like Roblox is not installed or Windows cannot open it. Would you like to install it? (yes/no)"}
    return {"status":"launch_requested", "command":"open_roblox", "message":"Asked Windows to open Roblox."}

def play_phantom_forces(progress=None):
    report = progress or (lambda message: None)
    """Request joining the official StyLiS Studios Phantom Forces experience."""
    if sys.platform != "win32":
        return {"status":"unavailable", "message":"Launch Phantom Forces on your Windows computer."}
    try:
        report("Step 1/4: Checking whether Roblox is installed...")
        if roblox_target() is None:
            return {"status":"needs_clarification", "command":"confirm_roblox_install",
                    "question":"Hey, it looks like Roblox is not installed. Would you like to install it, then open Phantom Forces? (yes/no)"}
        report("Step 2/4: Searching Roblox for Phantom Forces in your browser...")
        os.startfile("https://www.roblox.com/discover/?" + urlencode({"Keyword":"Phantom Forces"}))
        time.sleep(3)
        report("Step 3/4: Opening Phantom Forces by StyLiS Studios so you can see the game...")
        os.startfile("https://www.roblox.com/games/292439477/Phantom-Forces")
        time.sleep(3)
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"roblox\shell\open\command") as key:
                registered = bool(winreg.QueryValueEx(key, "")[0])
        except OSError:
            registered = False
        target = "roblox://placeId=292439477" if registered else "https://www.roblox.com/games/start?placeId=292439477"
        report("Step 4/4: I am going to play Phantom Forces. Asking Roblox to join now...")
        os.startfile(target)
        return {"status":"launch_requested", "command":"play_phantom_forces",
                "message":"Requested Phantom Forces in Roblox. Sign in or approve the launch if prompted; joining the game has not been verified."}
    except OSError as error:
        return {"status":"unavailable", "message":f"Could not launch Phantom Forces: {error}"}

class Session:
    def __init__(self,model,progress=None): self.model=model; self.progress=progress; self.pending=None; self.search_suggestion=None; self.install_suggestion=False; self.apps=AppManager(progress); self.pending_game=False; self.afk=AfkRoamer(progress); self.display_name="Assistant"; self.manual=ManualGuide()
    def accept(self,text):
        started = time.perf_counter_ns()
        result = self._accept(text)
        elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
        result["elapsed_ms"] = round(elapsed_seconds * 1000, 3)
        result["elapsed_seconds"] = round(elapsed_seconds, 6)
        result.update(process_memory())
        return result
    def _accept(self,text):
        if text.strip().casefold() in ("list available features", "list avaiable features", "list features", "help"):
            return {"status":"completed", "command":"list_features", "message":
                "Available features:\n"
                "- Calculate: 9+10 or calculate (9+10)*2\n"
                "- Search links: search Nvidia; a bare topic asks for confirmation\n"
                "- Tab autocomplete: nv, ra, jaz, mi (capitalization preserved)\n"
                "- Roblox: open roblox (offers installation if missing)\n"
                "- Phantom Forces: play phantom forces (also accepts phatom forces)\n"
                "- Steam: list games; play cs2; play <installed game name>\n"
                "- Apps: install <app>; uninstall <app> (selection and confirmation required)\n"
                "- Manual walkthrough: guide phantom forces; manual mode on\n"
                "- Walkthrough controls: next, repeat, back, cancel; manual mode off\n"
                "- Game calibration: calibrate deploy; calibrate phantom forces\n"
                "- Bounded game test: roam phantom forces (requires calibration)\n"
                "- Fullscreen movement test: fullscreen phantom forces\n"
                "- AFK movement test: afk roam on; afk roam off (Escape stops)\n"
                "- Codex: open codex; then type a request to submit through Codex CLI\n"
                "- Return to local commands: back to assistant\n"
                "- Debug: timing and RAM saved in logs/debug.jsonl\n"
                "- Cancel a pending request: cancel\n"
                "- Quit: exit\n"
                + ("Manual mode is currently ON: actions are instructions only.\n" if self.manual.enabled else "")
                + ("Codex mode is currently ON: requests are submitted to Codex.\n" if self.display_name == "Codex" else "")}
        if text.strip().casefold() in ("manual mode on", "slow mode on", "guide phantom forces", "manual phantom forces"):
            self.afk.stop()
            self.apps.cancel()
            self.pending = None
            self.search_suggestion = None
            self.install_suggestion = False
            self.pending_game = False
        manual_result = self.manual.accept(text)
        if manual_result is not None:
            return manual_result
        if text.strip().casefold() in ("open codex", "open this chat", "switch to codex", "please open codex"):
            try:
                os.startfile("codex://threads/01a16f40-e339-7bb1-8820-af17c111497f")
            except OSError as error:
                return {"status":"unavailable", "message":f"Could not open Codex: {error}"}
            self.display_name = "Codex"
            self.pending = None
            self.search_suggestion = None
            self.install_suggestion = False
            self.apps.cancel()
            return {"status":"launch_requested", "command":"open_codex", "message":"Asked Windows to open this Codex chat. Codex mode enabled. Requests you type here will be submitted automatically to this session through the Codex CLI, using your existing sign-in. Type back to assistant to return to local commands."}
        if text.strip().casefold() in ("back to assistant", "switch to assistant"):
            self.display_name = "Assistant"
            return {"status":"completed", "message":"Local display name changed to Assistant."}
        if self.display_name == "Codex" and text.strip().casefold() not in ("cancel", "/cancel"):
            return prepare_request(text, self.progress)
        if text.strip().casefold() == "afk roam on":
            return self.afk.start()
        if text.strip().casefold() == "afk roam off":
            self.afk.stop()
            return {"status":"completed", "message":"AFK mode disabled. Movement stopped."}
        if text.strip().casefold() in ("calibrate deploy", "calibrate phantom forces", "roam phantom forces", "roam phatom forces", "fullscreen phantom forces", "full screen phantom forces"):
            self.apps.cancel()
            self.pending = None
            self.search_suggestion = None
            self.install_suggestion = False
            self.pending_game = False
            return run_roam_command(text, self.progress)
        # A declined suggestion may include a replacement request in the same turn.
        if self.search_suggestion is not None:
            correction = re.fullmatch(r"(?:no|nope|nah)\b[\s,;:!.-]+(.+)", text.strip(), re.I | re.S)
            if correction:
                self.search_suggestion = None
                self.pending = None
                text = correction[1].strip()
        if text.strip().lower() in ("cancel","/cancel"):
            self.pending=None
            self.search_suggestion=None
            self.install_suggestion=False
            self.pending_game=False
            self.apps.cancel()
            return {"status":"cancelled","message":"Task cancelled."}
        if re.fullmatch(r"(?:please\s+)?(?:(?:play|open|launch|start)\s+)?pha(?:n)?tom\s+forces(?:\s+on\s+roblox)?", text.strip(), re.I):
            self.apps.cancel()
            self.pending = None
            self.search_suggestion = None
            result = play_phantom_forces(self.progress)
            self.install_suggestion = result.get("command") == "confirm_roblox_install"
            self.pending_game = self.install_suggestion
            return result
        game_request = re.fullmatch(r"(?:please\s+)?(?:play|run|launch|start)\s+(.+)", text.strip(), re.I)
        if text.strip().casefold() in ("list games", "show games", "my games") or (game_request and game_request[1].casefold() != "roblox"):
            self.apps.cancel()
            self.pending = None
            self.search_suggestion = None
            self.install_suggestion = False
            self.pending_game = False
            return launch_game(game_request[1] if game_request else None, self.progress)
        if re.fullmatch(r"(?:please\s+)?(?:install|uninstall)\s+.+", text.strip(), re.I) or self.apps.pending:
            self.install_suggestion = False
            self.search_suggestion = None
            self.pending = None
            app_result = self.apps.accept(text)
            if app_result is not None:
                return app_result
        if self.install_suggestion:
            self.install_suggestion = False
            answer = text.strip().casefold()
            if answer in ("yes", "y", "yeah", "yep", "sure", "ok", "okay"):
                game = self.pending_game
                self.pending_game = False
                result = install_and_open(roblox_target, self.progress)
                if game and result.get("status") == "launch_requested":
                    return play_phantom_forces(self.progress)
                return result
            if answer in ("no", "n", "nope"):
                self.pending_game = False
                return {"status":"cancelled", "message":"Okay, installation cancelled."}
        if text.strip().casefold() in ("roblox", "open roblox", "launch roblox", "start roblox", "please open roblox", "can you open roblox"):
            self.pending = None
            self.search_suggestion = None
            result = open_roblox(self.progress)
            self.install_suggestion = result.get("command") == "confirm_roblox_install"
            return result
        if self.search_suggestion is not None:
            query = self.search_suggestion
            self.search_suggestion = None
            answer = text.strip().casefold()
            if answer in ("yes", "y", "yeah", "yep", "sure", "ok", "okay"):
                return handle("google " + query, self.model)
            if answer in ("no", "n", "nope"):
                return {"status":"needs_clarification", "question":"What would you like me to do instead?"}
        # Offer a search for a bare topic; explicit commands keep their existing behavior.
        topic = text.strip()
        reserved = {"search", "google", "compute", "calculate", "evaluate", "yes", "no", "hello", "hi", "help"}
        if self.pending is None and len(topic.encode("utf-8")) <= 100 and re.fullmatch(r"[^\W\d_][\w.-]*", topic) and topic.casefold() not in reserved:
            self.search_suggestion = topic
            return {"status":"needs_clarification", "command":"confirm_search", "query":topic,
                    "question":f"Did you mean Google {topic}? (yes/no)"}
        try: result=handle(text,self.model,self.pending)
        finally: self.pending=None
        if result["status"]=="needs_clarification": self.pending=result.get("command")
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request",nargs="?")
    parser.add_argument("--model",default=str(Path(__file__).resolve().parent.parent/"artifacts_intent/intent_model.json"))
    parser.add_argument("--jsonl",action="store_true",help="One JSON result per input line for a local UI adapter.")
    parser.add_argument("--debug-log", default=str(Path(__file__).resolve().parent.parent / "logs" / "debug.jsonl"), help="Separate rotating performance log; no request text.")
    args=parser.parse_args()
    debug_logger = create_debug_logger(args.debug_log)
    session=Session(IntentModel(args.model), progress=None if args.jsonl else lambda message: print(session.display_name + ">", message, flush=True))
    if args.request is not None:
        result = session.accept(args.request)
        record_performance(debug_logger, result)
        print(json.dumps({k:v for k,v in result.items() if k not in ("elapsed_ms", "elapsed_seconds", "process_ram_mib", "peak_process_ram_mib")},ensure_ascii=False)); return
    if not args.jsonl: print("Local text assistant. Supports arithmetic and search links. Press Tab to complete words (nv -> nvidia). Type exit to quit; cancel clears clarification.")
    while True:
        try: text=input("") if args.jsonl else read_input("You> ")
        except (EOFError,KeyboardInterrupt): break
        if text.strip().lower() in ("exit","quit","/exit"): break
        result=session.accept(text)
        record_performance(debug_logger, result)
        result = {k:v for k,v in result.items() if k not in ("elapsed_ms", "elapsed_seconds", "process_ram_mib", "peak_process_ram_mib")}
        if args.jsonl: print(json.dumps(result,ensure_ascii=False),flush=True)
        elif result.get("command")=="compute" and result["status"]=="completed": print(session.display_name + ">",result["value"])
        elif result.get("command")=="search" and result["status"]=="ready": print(session.display_name + ">",result["url"],"\n"+result["message"])
        else: print(session.display_name + ">",result.get("message") or result.get("question"))
        # Only performance/status is logged; request text is excluded.
        del text,result
    session.afk.stop()
    session.pending=None

if __name__ == "__main__": main()
