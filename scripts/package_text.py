"""Build a minimal standard-Python UNO Q text inference package."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def package(root,model,output):
    root,model,output=Path(root).resolve(),Path(model).resolve(),Path(output).resolve()
    names=["run_assistant.py","TEXT_MODEL.md","LICENSE","local_model/__init__.py",
        "local_model/text_assistant.py","local_model/text_features.py","local_model/commands.py"]
    files={name:root/name for name in names}
    files["artifacts_intent/intent_model.json"]=model
    for p in files.values():
        if not p.is_file(): raise FileNotFoundError(p)
    manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,p in files.items(): archive.write(p,"uno-q-text/"+name)
        archive.writestr("uno-q-text/manifest.json",json.dumps(manifest,indent=2))
    print("Text runtime package:",output)
    return output

if __name__ == "__main__":
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default=str(root/"artifacts_intent/intent_model.json"))
    parser.add_argument("--out",default=str(root/"uno-q-text.zip"))
    args=parser.parse_args(); package(root,args.model,args.out)
