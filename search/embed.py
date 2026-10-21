"""PatentSBERTa embeddings with an on-disk cache keyed by text hash."""
import hashlib
import json
import os
import time

import numpy as np

MODEL = "AI-Growth-Lab/PatentSBERTa"
CACHE = "data/index/embeddings.npz"


class Embedder:
    def __init__(self, cache_path: str = CACHE, model_name: str = MODEL):
        self.cache_path, self.model_name = cache_path, model_name
        self._model = None
        self.cache: dict[str, np.ndarray] = {}
        self.meta = {"embed_seconds": 0.0, "n_embedded": 0}  # cumulative cost of fresh (uncached) embedding
        if os.path.exists(cache_path):
            z = np.load(cache_path)
            if str(z["model"]) == model_name:
                self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
                self.meta = json.loads(str(z["meta"]))

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str], progress: bool = False) -> np.ndarray:
        """L2-normalised embeddings, shape (len(texts), dim)."""
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self.cache}
        if missing:
            t0 = time.perf_counter()
            vecs = self.model.encode(list(missing.values()), batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=progress)
            self.meta["embed_seconds"] += time.perf_counter() - t0
            self.meta["n_embedded"] += len(missing)
            self.cache.update(zip(missing, vecs))
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            np.savez(self.cache_path, keys=np.array(list(self.cache)), vecs=np.stack(list(self.cache.values())),
                     model=self.model_name, meta=json.dumps(self.meta))
        return np.stack([self.cache[k] for k in keys])

    def truncation(self, texts: list[str]) -> tuple[int, int, int]:
        """-> (number of texts longer than the model's max_seq_length, max_seq_length, longest text in tokens)."""
        lengths = [len(ids) for ids in self.model.tokenizer(texts)["input_ids"]]
        return sum(n > self.model.max_seq_length for n in lengths), self.model.max_seq_length, max(lengths)
