"""Submit local requests through the supported Codex CLI session interface."""
import json
import os
from pathlib import Path
import shutil
import subprocess

THREAD_ID = "01a16f40-e339-7bb1-8820-af17c111497f"
CHAT_URL = "codex://threads/" + THREAD_ID

def find_codex():
    executable = shutil.which("codex.exe") or shutil.which("codex")
    if executable and Path(executable).suffix.casefold() == ".exe": return executable
    roots = [Path.home()/"AppData"/"Local"/"OpenAI"/"Codex"/"bin"]
    local = os.environ.get("LOCALAPPDATA")
    if local: roots.append(Path(local)/"OpenAI"/"Codex"/"bin")
    candidates = [path for root in roots for path in root.glob("*/codex.exe")]
    if candidates: return str(max(candidates,key=lambda path:path.stat().st_mtime))
    return None

def prepare_request(text, progress=None):
    report = progress or (lambda message:None)
    executable = find_codex()
    if not executable:
        return {"status":"unavailable", "command":"codex_submit", "message":"Codex CLI was not found. Your request was not submitted. Install or update Codex and restart this terminal."}
    report("Submitting your request to Codex using your existing local sign-in...")
    command = [executable, "exec", "--sandbox", "workspace-write", "--skip-git-repo-check", "--json", "resume", THREAD_ID, "-"]
    try:
        # Stdin preserves literal request text, avoiding command-line interpretation.
        result = subprocess.run(command, input=text, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", shell=False,
                                cwd=str(Path(__file__).resolve().parent.parent), timeout=900)
    except subprocess.TimeoutExpired:
        return {"status":"needs_attention", "command":"codex_submit", "message":"The Codex CLI timed out. Open the chat to check whether the request ran before submitting it again."}
    except OSError as error:
        return {"status":"unavailable", "command":"codex_submit", "message":f"Could not start Codex: {error}"}
    replies = []
    errors = []
    turn_failed = False
    turn_completed = False
    for line in result.stdout.splitlines():
        try: event = json.loads(line)
        except json.JSONDecodeError: continue
        if event.get("type") == "turn.completed": turn_completed = True
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                replies.append(item["text"])
        if event.get("type") in ("error", "turn.failed"):
            turn_failed = True
            error = event.get("error", event)
            if isinstance(error,dict): errors.append(str(error.get("message", "Codex reported an error")))
    if result.returncode or turn_failed or not turn_completed:
        detail = "\n".join(errors) or result.stderr.strip()[-2000:] or "Check the Codex chat for details."
        return {"status":"operation_failed", "command":"codex_submit", "message":"Codex did not report successful completion. " + detail}
    try: os.startfile(CHAT_URL)
    except OSError: pass
    return {"status":"completed", "command":"codex_submit", "message":"\n".join(replies) or "Codex completed the submitted request. Open the chat to view its result."}
