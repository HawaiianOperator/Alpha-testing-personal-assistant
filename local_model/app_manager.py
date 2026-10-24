"""Confirmed app installation and removal using Windows Package Manager."""
import re
import shutil
import subprocess
import sys

class AppManager:
    def __init__(self, progress=None):
        self.pending = None
        self.progress = progress

    def cancel(self):
        self.pending = None

    def run(self, arguments, timeout=60):
        executable = shutil.which("winget")
        if not executable:
            raise OSError("Windows Package Manager (winget) is unavailable. Install or update Microsoft's App Installer first.")
        return subprocess.run([executable] + arguments, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, shell=False)

    def accept(self, text):
        clean = text.strip()
        request = re.fullmatch(r"(?:please\s+)?(install|uninstall)\s+(.+)", clean, re.I)
        if request:
            self.pending = None
            action, query = request[1].lower(), request[2].strip()
            if len(query) > 150 or query.startswith("-") or not query:
                return {"status":"rejected", "message":"Enter one app name, such as install Firefox or uninstall Spotify."}
            if sys.platform != "win32":
                return {"status":"unavailable", "message":"App installation and removal require your Windows computer."}
            try:
                if self.progress: self.progress("Step 1/4: Finding matching apps...")
                args = ["search" if action == "install" else "list", "--query", query,
                        "--accept-source-agreements", "--disable-interactivity"]
                if action == "install": args += ["--source", "winget"]
                result = self.run(args)
                if result.returncode:
                    return {"status":"unavailable", "message":"No matching apps could be retrieved. " + (result.stdout + result.stderr).strip()}
                self.pending = {"action":action, "stage":"select"}
                return {"status":"needs_clarification", "question":result.stdout.strip() +
                        "\nStep 2/4: Copy the exact ID of the app you want to " + action + " (or type cancel):"}
            except (OSError, subprocess.SubprocessError) as error:
                return {"status":"unavailable", "message":str(error)}
        if not self.pending:
            return None
        pending = self.pending
        if clean.casefold() in ("no", "n", "cancel", "/cancel"):
            self.cancel()
            return {"status":"cancelled", "message":"App operation cancelled."}
        if pending["stage"] == "select":
            # IDs are passed as one argument; never interpreted by a shell.
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+{}\-]{0,199}", clean):
                return {"status":"needs_clarification", "question":"Copy an app ID from the list, or type cancel."}
            action = pending["action"]
            args = ["show" if action == "install" else "list", "--id", clean, "--exact",
                    "--accept-source-agreements", "--disable-interactivity"]
            if action == "install": args += ["--source", "winget"]
            try:
                result = self.run(args)
            except (OSError, subprocess.SubprocessError) as error:
                self.cancel()
                return {"status":"unavailable", "message":str(error)}
            if result.returncode:
                return {"status":"needs_clarification", "question":"That exact app ID was not found. Copy an ID from the list, or type cancel."}
            self.pending = {"stage":"confirm", "action":action, "id":clean}
            return {"status":"needs_clarification", "question":result.stdout.strip() +
                    f"\nStep 3/4: Do you want to {action} {clean}? (yes/no)"}
        if clean.casefold() not in ("yes", "y"):
            return {"status":"needs_clarification", "question":f"Reply yes to {pending['action']} {pending['id']}, or no to cancel."}
        self.cancel()
        action, package = pending["action"], pending["id"]
        args = [action, "--id", package, "--exact", "--accept-source-agreements", "--disable-interactivity"]
        if action == "install": args += ["--source", "winget", "--accept-package-agreements"]
        try:
            if self.progress: self.progress(f"Step 4/4: Starting {action} for {package}. Approve any Windows or installer prompts...")
            result = self.run(args, timeout=900)
            if result.returncode:
                return {"status":"operation_failed", "message":f"Could not {action} {package}. " + (result.stdout + result.stderr).strip()}
            return {"status":"completed", "command":action, "package_id":package,
                    "message":f"Windows Package Manager reported successful {action} for {package}.\n" + result.stdout.strip()}
        except subprocess.TimeoutExpired:
            return {"status":"operation_pending", "message":"The operation is taking longer than expected. Check its installer window before trying again."}
        except OSError as error:
            return {"status":"operation_failed", "message":str(error)}
