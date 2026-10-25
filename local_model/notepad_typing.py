"""Visible local Notepad typing for an explicitly selected saved text file."""
from datetime import datetime
import os
from pathlib import Path
import re
import subprocess
import time


def editor_contents(editor):
    try:return editor.iface_text.DocumentRange.GetText(-1)
    except Exception:pass
    try:return editor.get_value()
    except Exception:pass
    if editor.element_info.control_type=="Edit":return editor.window_text()
    raise RuntimeError("Notepad did not expose its document contents; typing was not started.")


def find_editor(desktop,path):
    from .roblox_visible import process_name
    matches=[]
    filename=path.name.casefold();stem=path.stem.casefold()
    for window in desktop.windows(visible_only=True):
        if process_name(window.process_id())!="notepad.exe":continue
        title=window.window_text().casefold()
        if filename in title or re.match(r"^\*?\s*"+re.escape(stem)+r"(?:\s*[-–]|$)",title):
            matches.append(window)
    if len(matches)!=1:return None
    window=matches[0]
    controls=[c for c in window.descendants() if c.element_info.control_type in ("Edit","Document")
              and c.is_visible() and c.is_enabled()]
    # Prefer the document surface over a search box or filename field.
    documents=[c for c in controls if c.element_info.control_type=="Document"]
    if len(documents)==1:return window,documents[0]
    if len(controls)==1:return window,controls[0]
    return None


def ensure_focus(window):
    import win32gui
    import win32api
    if win32api.GetAsyncKeyState(0x1B)&0x8000:
        raise InterruptedError("Typing stopped by Escape. Check the unsaved text in Notepad.")
    if win32gui.GetForegroundWindow()!=window.handle:
        raise InterruptedError("Typing stopped because Notepad lost focus. Check its unsaved text before retrying.")


def type_character(character):
    from .windows_game_input import Input,KeyboardInput,send
    encoded=character.encode("utf-16-le")
    for index in range(0,len(encoded),2):
        unit=int.from_bytes(encoded[index:index+2],"little")
        for flags in (0x0004,0x0006):
            event=Input();event.type=1;event.payload.ki=KeyboardInput(0,unit,flags,0,0);send(event)


def append_slowly(path,original,encoding,value,progress=None):
    from pywinauto import Desktop
    report=progress or (lambda message:None)
    if not path.is_file():raise ValueError("Type save to create the selected file first.")
    if path.read_bytes()!=original:raise ValueError("The file changed outside the assistant. Reopen it before typing.")
    old=original.decode(encoding)
    added=("\n" if old and not old.endswith(("\n","\r")) else "")+value
    report("Opening the selected file in Notepad...")
    subprocess.Popen(["notepad.exe",str(path)],shell=False)
    desktop=Desktop(backend="uia");deadline=time.monotonic()+15;found=None
    while time.monotonic()<deadline:
        found=find_editor(desktop,path)
        if found:break
        time.sleep(0.2)
    if not found:raise RuntimeError("Could not identify one Notepad editing surface for this filename. Keep one matching Notepad window open and retry.")
    window,editor=found
    current=editor_contents(editor)
    normalize=lambda content:content.replace("\r\n","\n").replace("\r","\n")
    if normalize(current)!=normalize(old):
        raise ValueError("Notepad contains different or unsaved text. Save or close that tab yourself before retrying.")
    backup=path.with_name(path.name+'.'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.bak')
    with backup.open('xb') as output:output.write(original)
    window.set_focus();editor.click_input();ensure_focus(window)
    editor.type_keys('^{END}',set_foreground=True)
    report("Typing slowly in Notepad. Escape or switching windows stops typing...")
    for character in added:
        ensure_focus(window)
        if character=='\n':editor.type_keys('{ENTER}',set_foreground=True)
        else:type_character(character)
        time.sleep(0.065)
    ensure_focus(window)
    expected=old+added
    actual=editor_contents(editor)
    if normalize(actual)!=normalize(expected):
        raise RuntimeError("Notepad's text did not match the expected result. Automatic save stopped; review the editor.")
    # Do not overwrite a concurrent disk change while the text was being entered.
    if path.read_bytes()!=original:raise ValueError("The file changed on disk during typing. Automatic save stopped; review Notepad.")
    editor.type_keys('^s',set_foreground=True)
    deadline=time.monotonic()+4
    while time.monotonic()<deadline:
        payload=path.read_bytes()
        if normalize(payload.decode(encoding))==normalize(expected):return payload,backup
        time.sleep(0.1)
    raise RuntimeError("Typing finished, but saving could not be verified. Review Notepad and save it manually if needed.")
