"""Small optional Windows terminal editor with Tab completion; no dependencies."""
import re
import sys

WORDS = ("michael", "rayan", "jazmine", "codex", "cs2", "counter", "strike", "list", "games", "play", "phantom", "forces", "install", "uninstall", "roblox", "open", "nvidia", "amd", "arduino", "google", "search", "calculate", "compute", "cancel", "exit")

def complete(text, cursor):
    """Complete the current word; leave unknown prefixes unchanged."""
    match = re.search(r"[A-Za-z]+$", text[:cursor])
    if not match:
        return text, cursor
    prefix = match.group().casefold()
    candidates = [word for word in WORDS if word.startswith(prefix)]
    if not candidates:
        return text, cursor
    word = candidates[0]
    typed = match.group()
    if len(typed) > 1 and typed.isupper():
        word = word.upper()
    elif typed[0].isupper():
        word = word.capitalize()
    return text[:match.start()] + word + text[cursor:], match.start() + len(word)

def read_input(prompt):
    if sys.platform != "win32" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return input(prompt)
    import msvcrt
    text = ""
    cursor = 0
    sys.stdout.write(prompt)
    sys.stdout.flush()
    while True:
        key = msvcrt.getwch()
        old = text
        old_cursor = cursor
        if key == "\r":
            print()
            return text
        if key == "\x03":
            raise KeyboardInterrupt
        if key == "\x1a" and not text:
            raise EOFError
        if key in ("\x00", "\xe0"):
            key = msvcrt.getwch()
            if key == "K": cursor = max(0, cursor - 1)
            elif key == "M": cursor = min(len(text), cursor + 1)
            elif key == "G": cursor = 0
            elif key == "O": cursor = len(text)
            elif key == "S": text = text[:cursor] + text[cursor + 1:]
        elif key == "\b":
            if cursor:
                text = text[:cursor - 1] + text[cursor:]
                cursor -= 1
        elif key == "\t":
            text, cursor = complete(text, cursor)
        elif key.isprintable():
            text = text[:cursor] + key + text[cursor:]
            cursor += len(key)
        # Backspaces also work in Windows consoles without ANSI escape support.
        sys.stdout.write("\b" * old_cursor + text + " " * max(0, len(old) - len(text)))
        sys.stdout.write("\b" * (max(len(old), len(text)) - cursor))
        sys.stdout.flush()
