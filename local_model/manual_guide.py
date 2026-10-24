"""User-paced instructions only. This module never controls apps or inputs."""
class ManualGuide:
    def __init__(self):
        self.enabled=False
        self.steps=[]
        self.index=0

    def response(self):
        return {"status":"needs_clarification", "command":"manual_guide",
                "question":f"Step {self.index+1}/{len(self.steps)}: {self.steps[self.index]}\nTake as long as you need. Type next when finished, repeat to see this step again, or cancel to stop."}

    def accept(self,text):
        clean=text.strip();lower=clean.casefold()
        if lower in ("manual mode on","slow mode on","guide phantom forces","manual phantom forces"):
            self.enabled=True;self.steps=[];self.index=0
            if 'phantom forces' not in lower:
                return {"status":"completed", "command":"manual_guide", "message":"Manual mode is on. I will give instructions one step at a time and wait for next. Nothing will be automated. Try play phantom forces, play cs2, install an app, or uninstall an app. Type manual mode off to return to automatic commands."}
            lower='play phantom forces'
        if lower in ("manual mode off","slow mode off"):
            self.enabled=False;self.steps=[]
            return {"status":"completed", "message":"Manual mode is off. Automatic commands are available again; AFK mode remains off."}
        if not self.enabled:return None
        if lower in ('cancel','/cancel'):
            self.steps=[]
            return {"status":"cancelled", "message":"Walkthrough cancelled. Manual mode remains on."}
        if self.steps:
            if lower in ('next','done','continue'):
                self.index+=1
                if self.index>=len(self.steps):
                    self.steps=[]
                    return {"status":"completed", "message":"Walkthrough finished. You performed all actions yourself. Manual mode remains on."}
                return self.response()
            if lower in ('repeat','back'):
                if lower=='back':self.index=max(0,self.index-1)
                return self.response()
            return {"status":"needs_clarification", "question":"I am waiting at the current step. Type next, repeat, back, or cancel. No actions were performed."}
        if lower in ('play phantom forces','play phatom forces','roam phantom forces','afk roam on'):
            self.steps=[
                'Open Roblox yourself.',
                'Click Roblox’s search box.',
                'Type Phantom Forces into the search box.',
                'Submit the search and find Phantom Forces by StyLiS Studios.',
                'Open that game’s page.',
                'Click its Play button and wait for the game to load.',
                'Click Deploy when you are ready to enter the match.',
                'Hold W briefly to walk forward, then release it.',
                'Hold W and Shift briefly to try sprinting, then release both.',
                'Hold the right mouse button briefly to aim down sights, then release it.',
                'Press your crouch key once (usually C).',
                'Press the lower-stance key again to try going prone (usually C again).',
                'Use your stand/sprint control to stand up again.',
                'Press your melee-slot key (usually 3) to select the knife. Do not click to attack.',
                'Press Space once to jump. This completes the manual movement walkthrough.'
            ]
        elif lower.startswith(('play ','run ','launch ')):
            title=clean.split(' ',1)[1]
            self.steps=['Open Steam yourself.','Select Library.',f'Find {title} in your library.',f'Select {title} to open its library page.','Click Play.','Read any launch options and continue when ready. Wait for the game to open.']
        elif lower.startswith(('install ','uninstall ')):
            action,name=clean.split(' ',1)
            if action.casefold()=='install':
                self.steps=['Open your browser yourself.',f'Find the official website for {name} and check the publisher.',f'Open the official download page for {name}.','Choose the Windows download.','Open the downloaded installer when ready.','Review each installer screen and follow its instructions.','Wait for installation to finish, then open the app yourself.']
            else:
                self.steps=['Open Windows Settings yourself.','Select Apps, then Installed apps.',f'Search for {name} and check that it is the exact app you intend to remove.',f'Open that app’s menu and choose Uninstall only when you are ready.','Review the confirmation and follow the app’s uninstaller.','Wait for removal to finish.']
        else:
            return {"status":"needs_clarification", "question":"Manual mode is on. Choose play phantom forces, play <Steam game>, install <app>, or uninstall <app>. I will wait for next after each instruction."}
        self.index=0
        return self.response()
