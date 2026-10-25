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
from .game_launcher import launch_game, open_steam
from .memory_usage import process_memory
from .roblox_roam import run_roam_command
from .afk_roam import AfkRoamer
from .codex_handoff import prepare_request
from .manual_guide import ManualGuide
from .text_files import TextFileAssistant
from .roblox_visible import play_visible, diagnose_roblox
from .roblox_games import open_pizza_place
from .pizza_delivery import inspect_destination
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
    if sys.platform != "win32":
        return {"status":"unavailable", "message":"Launch Phantom Forces on your Windows computer."}
    try:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"roblox\shell\open\command") as key:
                registered=bool(winreg.QueryValueEx(key, "")[0])
        except OSError:
            registered=False
        target="roblox://placeId=292439477" if registered else "https://www.roblox.com/games/start?placeId=292439477"
        os.startfile(target)
        return {"status":"launch_requested", "command":"play_phantom_forces", "message":"Asked Roblox to launch Phantom Forces directly. Sign in or approve the browser launch if prompted; joining has not been verified."}
    except OSError as error:
        return {"status":"unavailable", "message":f"Could not launch Phantom Forces: {error}"}

class Session:
    def __init__(self,model,progress=None): self.model=model; self.progress=progress; self.pending=None; self.search_suggestion=None; self.install_suggestion=False; self.apps=AppManager(progress); self.pending_game=False; self.afk=AfkRoamer(progress); self.display_name="Assistant"; self.manual=ManualGuide(); self.text_files=TextFileAssistant(progress)
    def accept(self,text):
        started = time.perf_counter_ns()
        result = self._accept(text)
        elapsed_seconds = (time.perf_counter_ns() - started) / 1_000_000_000
        result["elapsed_ms"] = round(elapsed_seconds * 1000, 3)
        result["elapsed_seconds"] = round(elapsed_seconds, 6)
        result.update(process_memory())
        return result
    def _accept(self,text):
        if text.strip().casefold() in ("pizza destination", "read pizza address"):
            return inspect_destination()
        if text.strip().casefold() == "debug roblox":
            return diagnose_roblox()
        if text.strip().casefold() in ("list available features", "list avaiable features", "list features", "help"):
            return {"status":"completed", "command":"list_features", "message":
                "Available features:\n"
                "- Calculate: 9+10 or calculate (9+10)*2\n"
                "- Search links: search Nvidia; a bare topic asks for confirmation\n"
                "- Tab autocomplete: nv, ra, jaz, mi (capitalization preserved)\n"
                "- Roblox category: play roblox; then enter a game name\n"
                "- Phantom Forces: play phantom forces (also accepts phatom forces)\n"
                "- Work at a Pizza Place: play pizza place (delivery automation still needs setup)\n"
                "- Steam category: open steam; then list games or enter an installed game name\n"
                "- Apps: install <app>; uninstall <app> (selection and confirmation required)\n"
                "- Manual walkthrough: guide phantom forces; manual mode on\n"
                "- Walkthrough controls: next, repeat, back, cancel; manual mode off\n"
                "- Game calibration: calibrate deploy; calibrate phantom forces\n"
                "- Bounded game test: roam phantom forces (requires calibration)\n"
                "- Movement only without calibration: auto move (30 seconds)\n"
                "- Fullscreen movement test: fullscreen phantom forces\n"
                "- Random movement/camera/firing test: combat test phantom forces (30 seconds)\n"
                "- AFK movement test: afk roam on; afk roam off (Escape stops)\n"
                "- Text files: open text files; then open <path> or new notes.txt\n"
                "- Reopen the selected text file from this session: open last file\n"
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
        if text.strip().casefold() in ("open last file", "open last text file"):
            return self.text_files.accept("open last file")
        if text.strip().casefold() in ("open codex", "open this chat", "switch to codex", "please open codex"):
            if self.display_name == "text-file-assistant":
                warning=self.text_files.leave()
                if warning is not None:return warning
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
            if self.display_name == "text-file-assistant":
                warning = self.text_files.leave()
                if warning is not None:return warning
            self.display_name = "Assistant"
            return {"status":"completed", "message":"Local display name changed to Assistant."}
        category_entry = text.strip().casefold()
        file_entry = re.fullmatch(r"(?:open text file|edit text file)\s+(.+)",text.strip(),re.I)
        new_file_entry = re.fullmatch(r"(?:new|create text file)\s+(.+)",text.strip(),re.I) if not self.text_files.collecting else None
        if file_entry is None and not self.text_files.collecting:
            file_entry = re.fullmatch(r"open\s+(.+\.(?:txt|text|md)\"?)",text.strip(),re.I)
        if category_entry in ("open text files", "open text file", "text files", "text file mode") or file_entry or new_file_entry:
            self.afk.stop()
            self.apps.cancel()
            self.pending=None
            self.search_suggestion=None
            self.install_suggestion=False
            self.display_name="text-file-assistant"
            if file_entry:return self.text_files.accept("open "+file_entry[1])
            if new_file_entry:return self.text_files.accept("new "+new_file_entry[1])
            return {"status":"needs_clarification", "command":"text_file", "question":"Text-file category opened. Type open <path> to open a file, or new notes.txt to create one. Then type your text, review with show, and save. Use open that file to open the saved file in your editor."}
        if self.display_name == "text-file-assistant":
            return self.text_files.accept(text)
        if category_entry in ("play roblox", "open roblox", "switch to roblox", "roblox mode"):
            self.display_name = "Roblox"
            self.pending = None
            self.search_suggestion = None
            self.apps.cancel()
            result = open_roblox(self.progress)
            self.install_suggestion = result.get("command") == "confirm_roblox_install"
            if not self.install_suggestion:
                result["message"] = result.get("message", "") + " Roblox category: what game would you like to play? Type a game name, what can you do, or back to assistant."
            return result
        if category_entry in ("open steam", "play steam", "switch to steam", "steam mode"):
            self.display_name = "Steam"
            self.pending = None
            self.search_suggestion = None
            self.install_suggestion = False
            self.apps.cancel()
            return open_steam(self.progress)
        if self.display_name in ("Roblox", "Steam") and not self.install_suggestion and not self.apps.pending:
            if category_entry in ("what can you do", "what can i do", "what game", "what game to play", "what game should i play", "what do i do"):
                return {"status":"needs_clarification", "question":
                    "Roblox category: type a game name such as Phantom Forces, play <game>, auto move, or back to assistant." if self.display_name=="Roblox" else
                    "Steam category: type list games to see installed titles, a game name such as cs2, or back to assistant."}
            passthrough = re.match(r"(?:afk |calibrate |roam |fullscreen |full screen |combat test |random combat |auto move|move roblox|install |uninstall |calculate |compute |search |google |manual |slow |guide )",category_entry)
            if not passthrough and category_entry not in ("cancel", "/cancel"):
                if self.display_name=="Steam":
                    if category_entry in ("list games", "show games", "my games"):
                        return launch_game(None,self.progress)
                    title=re.sub(r"^(?:play|run|launch|start|open)\s+", "",text.strip(),flags=re.I)
                    return launch_game(title,self.progress)
                if category_entry in ("list games", "show games", "my games"):
                    return {"status":"needs_clarification", "question":"Tell me which Roblox game to search for. For example: Phantom Forces. Exact matches are required before Play is clicked."}
                title=re.sub(r"^(?:play|run|launch|start|open)\s+", "",text.strip(),flags=re.I)
                if title.casefold() in ("work at a pizza place", "work a pizza place", "pizza place"):
                    return open_pizza_place()
                if title.casefold() in ("phatom forces", "phantom forces"):
                    return play_phantom_forces(self.progress)
                if not title or len(title)>120:
                    return {"status":"needs_clarification", "question":"Which Roblox game would you like to search for?"}
                return play_visible(roblox_target,self.progress or (lambda message:None),title=title,creator=None)
        if self.display_name == "Codex" and text.strip().casefold() not in ("cancel", "/cancel"):
            return prepare_request(text, self.progress)
        if re.fullmatch(r"(?:(?:play|open|launch)\s+)?(?:work (?:at )?a pizza place|pizza place)",text.strip(),re.I):
            return open_pizza_place()
        if text.strip().casefold() in ("start pizza delivery", "pizza delivery"):
            return {"status":"needs_attention", "command":"pizza_delivery_setup", "message":"Delivery automation is not configured yet. Open Work at a Pizza Place, join Delivery, and provide a screenshot of the pickup area with a pizza destination visible. Planned behavior: wait when there are no deliveries; pick up a pizza, read its address, walk and jump to the correct house, verify delivery, return, and repeat. No delivery loop has started."}
        if text.strip().casefold() == "afk roam on":
            return self.afk.start()
        if text.strip().casefold() == "afk roam off":
            self.afk.stop()
            return {"status":"completed", "message":"AFK mode disabled. Movement stopped."}
        if text.strip().casefold() in ("calibrate deploy", "calibrate phantom forces", "roam phantom forces", "roam phatom forces", "fullscreen phantom forces", "full screen phantom forces", "combat test phantom forces", "random combat phantom forces", "auto move", "auto move roblox", "move roblox"):
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
        if text.strip().lower() in ("exit","quit","/exit"):
            warning=session.text_files.leave()
            if warning is not None:
                print(session.display_name + ">", warning["message"])
                continue
            break
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
