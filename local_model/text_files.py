"""Local plain-text draft editor. File contents are data, never commands."""
from datetime import datetime
import os
from pathlib import Path
import tempfile
import re
import subprocess
from .notepad_typing import append_slowly

DEFAULT_FOLDER=Path(__file__).resolve().parent.parent/"text_files"

class TextFileAssistant:
    def __init__(self,progress=None):
        self.progress=progress
        self.live_typing=False
        self.path=None
        self.draft=""
        self.original=None
        self.encoding="utf-8"
        self.collecting=None
        self.lines=[]

    @property
    def dirty(self):
        if self.path is None:return False
        if self.original is None:return bool(self.draft)
        return self.draft.encode(self.encoding)!=self.original

    def reply(self,message,status="completed"):
        return {"status":status,"command":"text_file","message":message}

    def leave(self):
        if self.dirty or self.collecting:
            return self.reply("You have unsaved text. Type save to keep it, or discard to discard the draft, then back to assistant.","needs_attention")
        return None

    def resolve(self,name):
        name=name.strip().strip('"')
        if not name:raise ValueError("Supply a text-file path.")
        path=Path(os.path.expandvars(name)).expanduser()
        if not path.is_absolute():path=DEFAULT_FOLDER/path
        if not path.suffix:path=path.with_suffix(".txt")
        path=path.resolve()
        if path.suffix.casefold() not in (".txt",".text",".md"):
            raise ValueError("Use a .txt, .text, or .md file.")
        return path

    def load(self,name,new=False):
        if self.dirty or self.collecting:return self.reply("Save or discard the current draft before opening another file.","needs_attention")
        path=self.resolve(name)
        if new and path.exists():raise ValueError("That file already exists. Use open instead.")
        if not new and not path.is_file():raise ValueError("File not found. Use new followed by a filename to create one.")
        raw=None if new else path.read_bytes()
        if raw is not None and len(raw)>1048576:raise ValueError("This editor supports text files up to 1 MiB.")
        encoding="utf-16" if raw and raw.startswith((b"\xff\xfe",b"\xfe\xff")) else "utf-8-sig" if raw and raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        draft="" if raw is None else raw.decode(encoding)
        self.path=path;self.original=raw;self.encoding=encoding;self.draft=draft
        return self.reply(f"{'New draft' if new else 'Opened'}: {path}\n"+(draft[:8000] or "(empty file)")+("\n[Preview truncated; show displays the first 8,000 characters.]" if len(draft)>8000 else "")+"\nType ordinary text to append it, or use write <text>, append <text>, write for multiple lines, show, save, discard, or open that file. Changes stay in memory until save.")

    def type_in_editor(self,value):
        if self.dirty:return self.reply('You have an unsaved draft. Use save or discard first, then add your text again.','needs_attention')
        if self.original is None:return self.reply('Type save to create the file first, then add your text again.','needs_attention')
        try:
            payload,backup=append_slowly(self.path,self.original,self.encoding,value,self.progress)
        except Exception as error:
            return self.reply(f'Visible Notepad typing stopped: {error}','needs_attention')
        self.original=payload;self.draft=payload.decode(self.encoding)
        return self.reply(f'Typed your text visibly in Notepad and verified the saved file: {self.path}\nBackup: {backup}')

    def append_text(self,value):
        if self.live_typing:return self.type_in_editor(value)
        self.draft+=("\n" if self.draft and not self.draft.endswith("\n") else "")+value
        return self.reply('Text added to the draft. Use show results to review, save to save, or open that file after saving to enable visible Notepad typing.')

    def accept(self,text):
        clean=text.strip();lower=clean.casefold()
        try:
            if self.collecting:
                if lower=='.cancel':
                    self.collecting=None;self.lines=[]
                    return self.reply("Multiline entry cancelled; previous draft kept.")
                if lower=='.done':
                    added="\n".join(self.lines)
                    self.draft=added if self.collecting=='write' else self.draft+("\n" if self.draft and not self.draft.endswith("\n") else "")+added
                    self.collecting=None;self.lines=[]
                    return self.reply("Draft updated. Type show to review or save to write it to disk.")
                self.lines.append(text)
                return self.reply("Line added. Continue typing; .done finishes, .cancel cancels.")
            if lower in ('help','what can you do','list features'):
                return self.reply('Text files: open <path>, new <filename.txt>, open that file to launch your default editor, type ordinary text to append it, write <text>, append <text>, write or append for multiline entry, show, save, discard, back to assistant. Relative filenames use '+str(DEFAULT_FOLDER)+'. Existing files get a backup when saved. Use type slowly <text> to append visibly in Notepad and save automatically.')
            if lower in ('open that file','open this file','open the file','open last file','open last text file','open in notepad','open in editor'):
                if self.path is None:return self.reply('No file is remembered in this session. Type open notes.txt or new notes to choose one.','needs_clarification')
                if self.dirty or self.collecting:return self.reply('Type save first so the editor opens your latest text.','needs_attention')
                if not self.path.is_file():return self.reply('Type save to create this file first.','needs_attention')
                subprocess.Popen(["notepad.exe",str(self.path)],shell=False)
                self.live_typing=True
                return self.reply(f'Asked Notepad to open {self.path}. Visible typing is enabled: add <text> or ordinary text will now be typed slowly into Notepad and saved.')
            if lower.startswith('open '):return self.load(clean[5:])
            if lower.startswith('new '):return self.load(clean[4:],new=True)
            if self.path is None:
                if lower.endswith(('.txt','.text','.md')):return self.load(clean)
                return self.reply("Which file? Type open <path> or new notes.txt.","needs_clarification")
            if lower in ('live typing on','slow typing on'):
                self.live_typing=True
                return self.reply('Visible Notepad typing enabled. Type add <text> or ordinary text. The saved file must match the editor first.')
            if lower in ('live typing off','slow typing off'):
                self.live_typing=False
                return self.reply('Visible typing disabled. Text changes stay in a draft until save.')
            if lower.startswith('type slowly '):
                return self.type_in_editor(clean[len('type slowly '):])
            if re.fullmatch(r'(?:save|discard)type slowly\s+.+',lower):
                return self.reply('Those commands were joined together. Press Enter after save or discard, then enter type slowly <text> separately. Nothing was added.','needs_clarification')
            if lower in ('clear','clear text','clear draft'):
                self.draft=''
                return self.reply('Draft cleared. Type save to empty the file on disk, or discard to restore the saved text.')
            if lower in ('show','show results','show result','review','review file','show to review','show review','type to show review'):return self.reply(str(self.path)+"\n"+(self.draft[:8000] or '(empty draft)'))
            if lower=='discard':
                self.draft='' if self.original is None else self.original.decode(self.encoding)
                return self.reply('Unsaved changes discarded.')
            if lower in ('write','append'):
                self.collecting=lower;self.lines=[]
                return self.reply('Type your text one line at a time. Type .done on its own line to finish, or .cancel to cancel. Nothing is saved until save.')
            if lower.startswith('write '):self.draft=text[text.lower().index('write ')+6:];return self.reply('Replacement draft ready. Type show to review, then save.')
            if lower.startswith(('append ','add ')):
                prefix='append ' if lower.startswith('append ') else 'add '
                added=clean[len(prefix):]
                return self.append_text(added)
            if lower=='save':
                current=self.path.read_bytes() if self.path.exists() else None
                if current!=self.original:return self.reply('The file changed outside this assistant. Save stopped to avoid overwriting those changes. Copy your draft before reopening the file.', 'needs_attention')
                payload=self.draft.encode(self.encoding)
                if len(payload)>1048576:raise ValueError('The draft exceeds the 1 MiB limit.')
                self.path.parent.mkdir(parents=True,exist_ok=True)
                backup=None
                if current is not None:
                    backup=self.path.with_name(self.path.name+'.'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.bak')
                    with backup.open('xb') as output:output.write(current)
                descriptor,temporary=tempfile.mkstemp(prefix='.text-draft-',dir=str(self.path.parent))
                try:
                    with os.fdopen(descriptor,'wb') as output:output.write(payload)
                    os.replace(temporary,self.path)
                finally:
                    if os.path.exists(temporary):os.unlink(temporary)
                self.original=payload
                return self.reply(f'Saved: {self.path}'+(f'\nBackup: {backup}' if backup else ''))
            if not text.strip():return self.reply('Type text to append, show to review, or save to save.','needs_clarification')
            return self.append_text(text)
        except (OSError,ValueError,UnicodeError,RuntimeError,InterruptedError) as error:
            return self.reply(f'Text-file operation stopped: {error}','needs_attention')
