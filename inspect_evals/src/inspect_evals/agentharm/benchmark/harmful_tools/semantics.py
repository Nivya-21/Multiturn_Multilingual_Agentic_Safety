# semantics.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os

# Singletons (lazy)
_SEM_MODEL = None
_KEY_EMBS = None
_KEYS: List[str] = []

# Default model id if FORCE_ST_MODEL not provided
_MODEL_NAME = "intfloat/multilingual-e5-small"

def _prep(texts: List[str], *, as_query: bool) -> List[str]:
    prefix = "query: " if as_query else "passage: "
    return [prefix + (t or "") for t in texts]

def _maybe_load_encoder(keys: List[str]) -> bool:
    """
    Try to load a sentence-transformers encoder once.
    Priority:
      1) FORCE_ST_MODEL (local dir) for offline use
      2) Hub id (cache/hub)
    """
    global _SEM_MODEL, _KEY_EMBS, _KEYS
    if _SEM_MODEL is not None and _KEY_EMBS is not None and _KEYS == keys:
        return True

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        prefer = os.getenv("FORCE_ST_MODEL", "").strip()

        if prefer and os.path.isabs(prefer) and os.path.isdir(prefer):
            print(f"[harmful_tools.semantics] Loading ST (LOCAL): {prefer}", flush=True)
            _SEM_MODEL = SentenceTransformer(prefer, device="cpu", local_files_only=True)
        else:
            print(f"[harmful_tools.semantics] Loading ST (HUB): {_MODEL_NAME}", flush=True)
            _SEM_MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")

        _KEYS = list(keys)
        _KEY_EMBS = _SEM_MODEL.encode(_prep(_KEYS, as_query=False), normalize_embeddings=True)
        print("[harmful_tools.semantics] Encoder ready.", flush=True)
        return True

    except Exception as e:
        print("[harmful_tools.semantics] FAILED to load ST:", repr(e), flush=True)
        _SEM_MODEL = None
        _KEY_EMBS = None
        _KEYS = []
        return False

def semantic_lookup(
    query: str,
    mapping: Dict[str, Any],
    *,
    top_k: int = 1,
    score_threshold: float = 0.5,
) -> List[Tuple[str, Any, float]]:
    """
    Semantic matcher over {key -> payload}. Returns (matched_key, payload, score).
    """
    if not query or not mapping:
        return []

    keys = list(mapping.keys())
    if not _maybe_load_encoder(keys):
        return []

    from sentence_transformers import util  # local import

    q = _SEM_MODEL.encode(_prep([query], as_query=True), normalize_embeddings=True)
    sims = util.dot_score(q, _KEY_EMBS)[0].cpu().numpy()
    order = sims.argsort()[::-1][:top_k]

    out: List[Tuple[str, Any, float]] = []
    for i in order:
        score = float(sims[i])
        if score >= score_threshold:
            k = keys[i]
            out.append((k, mapping[k], score))
    return out
