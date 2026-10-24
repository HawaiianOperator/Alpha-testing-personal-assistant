"""Synthetic training corpus and separate fixed evaluation prompts; editable source."""
import json
from pathlib import Path

SEARCH_TEMPLATES = ["search {q}", "google {q}", "search for {q}", "google: {q}",
    "please search for {q}", "please google {q}", "can you search for {q}",
    "look up {q}", "look up {q} on google", "find information about {q}",
    "search google for {q}", "could you google {q}"]
COMPUTE_TEMPLATES = ["compute {q}", "calculate {q}", "what is {q}", "what's {q}",
    "please calculate {q}", "can you calculate {q}", "work out {q}",
    "evaluate {q}", "compute: {q}", "could you compute {q}"]
QUERIES = ["Nvidia", "AMD graphics cards", "the Empire State Building", "weather tomorrow",
    "best GPU", "affordable graphics cards", "Python tutorials", "meeting planning tips",
    "local restaurants", "new laptops", "images of the Empire State Building",
    "how to calculate a budget", "image generation tools", "travel to Japan",
    "RTX performance", "Arduino UNO Q", "privacy tools", "research and development"]
EXPRESSIONS = ["2 + 2", "10 / 2", "(5 + 3) * 2", "0.1 + 0.2", "9 - 4", "-3 * 7",
    "123 + 456", "20 / (2 + 3)", "100 * 0.2", "8 / 3", "5 + 6 - 2", "12"]
REJECT_TEMPLATES = ["generate an image of {q}", "draw {q}", "create a picture of {q}",
    "make a video about {q}", "write an essay about {q}", "buy {q}",
    "download and install {q}", "open {q}", "delete {q}", "send an email about {q}",
    "optimize my calendar for {q}", "schedule a meeting about {q}",
    "compare all options for {q} and choose the best", "summarize {q}",
    "search for {q} then buy it", "calculate {q} and email it", "search for {q} and install it",
    "tell me a joke about {q}", "type {q} into another application"]
# Fixed evaluation prompts are not added to the generated training corpus.
EVALUATION = [
 ("Search for Radeon specifications", "search"), ("Google Saturn's moons", "search"),
 ("Please look up historical maps", "search"), ("Find information about ocean currents", "search"),
 ("Could you search Google for keyboard layouts?", "search"),
 ("Calculate 17 + 29", "compute"), ("Compute (18 - 6) / 4", "compute"),
 ("What is 7 * 13?", "compute"), ("Please evaluate 0.25 + 0.75", "compute"),
 ("Can you compute 49 / 7?", "compute"),
 ("generate an image of the empire state building at sunset", "reject"),
 ("Optimize my calendar to maximize my time", "reject"),
 ("Search for a cheap GPU and buy it", "reject"),
 ("Compute 2 + 2 then email the answer", "reject"),
 ("Create an illustration of a skyscraper", "reject"),
 ("Install a graphics driver", "reject"), ("Write a poem for my friend", "reject"),
 ("Move the mouse and click the blue button", "reject"),
]

def rows():
    records=[]
    for label, templates, values in (("search", SEARCH_TEMPLATES, QUERIES),
            ("compute", COMPUTE_TEMPLATES, EXPRESSIONS), ("reject", REJECT_TEMPLATES, QUERIES)):
        for template in templates:
            for value in values:
                records.append({"text":template.format(q=value), "label":label})
    # Include missing-argument intents: clarification happens in the executor.
    for word in ("search", "google", "compute", "calculate"):
        records.append({"text":word, "label":"search" if word in ("search","google") else "compute"})
    return records

def write_dataset(path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows()), encoding="utf-8")
    return path

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/intent_training.jsonl")
    print(write_dataset(parser.parse_args().out))
