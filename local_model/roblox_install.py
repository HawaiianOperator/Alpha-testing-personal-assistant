"""Download the Roblox Windows installer and run it after user confirmation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from urllib.parse import urlparse

INSTALLER_URL = "https://setup.rbxcdn.com/RobloxPlayerInstaller.exe"

def download_installer(destination):
    with urlopen(INSTALLER_URL, timeout=30) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or not (final.hostname == "rbxcdn.com" or (final.hostname or "").endswith(".rbxcdn.com")):
            raise ValueError("Unexpected installer download destination")
        total = 0
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk: break
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise ValueError("Unexpected installer size")
                output.write(chunk)
    with destination.open("rb") as downloaded:
        if downloaded.read(2) != b"MZ":
            raise ValueError("The download is not a Windows installer")

def verify_installer(destination):
    env = os.environ.copy()
    env["ROBLOX_INSTALLER_TO_VERIFY"] = str(destination)
    script = "$s = Get-AuthenticodeSignature -LiteralPath $env:ROBLOX_INSTALLER_TO_VERIFY; [pscustomobject]@{Status=$s.Status.ToString();Subject=$s.SignerCertificate.Subject} | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                            env=env, capture_output=True, text=True, timeout=30, check=True)
    signature = json.loads(result.stdout.lstrip("\ufeff"))
    if signature.get("Status") != "Valid" or "roblox" not in (signature.get("Subject") or "").casefold():
        raise ValueError("Installer does not have a valid Roblox publisher signature")

def install_and_open(find_target, progress=None):
    if sys.platform != "win32":
        return {"status":"unavailable", "message":"Roblox installation requires your Windows computer."}
    report = progress or (lambda message: None)
    try:
        # Keep the installer available if it starts a separate installation process.
        folder = Path(tempfile.mkdtemp(prefix="assistant-roblox-"))
        installer = folder / "RobloxPlayerInstaller.exe"
        report("Step 1/4: Downloading Roblox...")
        download_installer(installer)
        report("Step 2/4: Checking the Roblox installer signature...")
        verify_installer(installer)
        report("Step 3/4: Installing Roblox. Approve any Windows prompts that appear...")
        process = subprocess.Popen([str(installer)], shell=False)
        try:
            code = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            return {"status":"installation_pending", "message":"The installer is still running. Finish its prompts, then type open roblox."}
        if code != 0:
            return {"status":"installation_failed", "message":f"Roblox installer exited with code {code}. Check its window for details."}
        deadline = time.monotonic() + 60
        while True:
            target = find_target()
            if target:
                report("Step 4/4: Asking Windows to open Roblox...")
                os.startfile(target)
                return {"status":"launch_requested", "message":"Roblox installation completed; asked Windows to open Roblox."}
            if time.monotonic() >= deadline: break
            time.sleep(2)
        return {"status":"installation_pending", "message":"The installer finished, but Roblox is not detected yet. Check its window, then try open roblox again."}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status":"installation_failed", "message":f"Could not install Roblox: {error}"}
