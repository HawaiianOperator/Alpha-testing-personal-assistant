"""Local recognition of the demonstrated C3 delivery item. No route is assumed."""
from pathlib import Path
from PIL import Image,ImageChops,ImageStat

def destination_from_image(image):
    # Inspect only the bottom-center inventory region; don't store the screen.
    width,height=image.size
    if abs(width/height-1920/1080)>0.12:return None
    screen=image.convert("RGB").resize((1920,1080))
    with Image.open(Path(__file__).parent/"assets"/"pizza-destination-c3.png") as source:
        template=source.convert("RGB")
    best=255.0
    for x in range(650,1251,2):
        patch=screen.crop((x,1000,x+73,1075))
        error=sum(ImageStat.Stat(ImageChops.difference(patch,template)).mean)/3
        best=min(best,error)
    return "C3" if best<=8 else None

def inspect_destination():
    try:
        from pywinauto import Desktop
        from .roblox_roam import foreground_roblox
        from .roblox_visible import process_name
        windows=[window for window in Desktop(backend="win32").windows(visible_only=True)
                 if process_name(window.process_id()) in ("robloxplayerbeta.exe","robloxplayer.exe")]
        if len(windows)!=1:
            return {"status":"needs_attention","message":"Open one Roblox game window with the delivery item selected, then retry pizza destination."}
        window=windows[0]
        # Read-only inspection. No window activation, keys or mouse clicks.
        destination=destination_from_image(window.capture_as_image())
        if destination:
            return {"status":"completed","command":"pizza_destination","destination":destination,
                    "message":"Selected delivery destination detected: C3. The walking route is not configured; no movement started."}
        return {"status":"needs_attention","command":"pizza_destination",
                "message":"The demonstrated C3 label was not detected. This does not prove there are no deliveries: select a delivery item so its address is visible. No movement started."}
    except Exception as error:
        return {"status":"needs_attention","message":f"Destination check stopped: {error}"}
