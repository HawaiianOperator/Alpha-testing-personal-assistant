"""Stable dependency-free hashed text features shared by training and deployment."""
import hashlib
import math
import re

LABELS = ("search", "compute", "reject")
DIMENSIONS = 2048

def features(text):
    # Numeric magnitudes are handled by the arithmetic executor, not memorized.
    normalized = re.sub(r"\d+(?:\.\d+)?", "#", " ".join(text.lower().split()))
    tokens = ["word:" + w for w in re.findall(r"[a-z]+|[#()+*/-]", normalized)]
    padded = "^" + normalized + "$"
    for width in (3, 4):
        tokens.extend(padded[i:i + width] for i in range(max(0, len(padded)-width+1)))
    counts = {}
    for token in tokens:
        index = int.from_bytes(hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest(), "little") % DIMENSIONS
        counts[index] = counts.get(index, 0) + 1
    length = math.sqrt(sum(v*v for v in counts.values())) or 1.0
    return {k: v/length for k,v in counts.items()}
