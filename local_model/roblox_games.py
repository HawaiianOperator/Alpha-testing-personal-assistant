"""Direct launch of named Roblox experiences with verified place IDs."""
import os
import sys

def open_pizza_place():
    if sys.platform!="win32":
        return {"status":"unavailable","message":"Open Work at a Pizza Place on your Windows computer."}
    try:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT,r"roblox\shell\open\command") as key:
                registered=bool(winreg.QueryValueEx(key,"")[0])
        except OSError:registered=False
        target="roblox://placeId=192800" if registered else "https://www.roblox.com/games/start?placeId=192800"
        os.startfile(target)
        return {"status":"launch_requested","command":"play_pizza_place","message":"Asked Roblox to launch Work at a Pizza Place. Join the Delivery role. Automatic address reading and delivery routes are not configured yet."}
    except OSError as error:
        return {"status":"unavailable","message":f"Could not open Work at a Pizza Place: {error}"}
