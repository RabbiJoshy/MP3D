"""Embed every cleaned passage locally. Writes data/embed/passages.npy (float32, L2-normalised)."""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "embed"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "BAAI/bge-base-en-v1.5"

df = pd.read_parquet(ROOT / "data" / "clean" / "passages.parquet", columns=["passage_id", "text"])
limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
texts = df.text.tolist()[:limit]
model = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
model.max_seq_length = 256
dim = model.get_sentence_embedding_dimension()

# Longest first keeps batches evenly padded, and memory problems show up at the start, not hours in.
order = np.argsort([-len(t) for t in texts], kind="stable")
path = OUT / ("passages.npy" if limit is None else "sample.npy")
progress = path.with_suffix(".progress")

# Resumable: this takes about an hour and a restart shouldn't lose the work.
first = int(progress.read_text()) if path.exists() and progress.exists() else 0
out = np.lib.format.open_memmap(path, mode="r+" if first else "w+", dtype="float32", shape=(len(texts), dim))
print(f"starting at {first}/{len(texts)}", flush=True)
start, step = time.time(), 2048
for i in range(first, len(texts), step):
    idx = order[i:i + step]
    out[idx] = model.encode([texts[j] for j in idx], batch_size=48, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
    out.flush()
    progress.write_text(str(i + len(idx)))
    if ((i - first) // step) % 10 == 0:
        rate = (i - first + len(idx)) / (time.time() - start)
        print(f"{i + len(idx)}/{len(texts)} {rate:.0f}/s eta {(len(texts) - i) / rate / 60:.0f} min", flush=True)
    torch.mps.empty_cache()
out.flush()
print("DONE", path, out.shape, flush=True)
