# --- helpers.py (or top of generate_attack_plans.py) ---
import json, os, re, tempfile, hashlib
from typing import Dict, Any, Iterable

BRACE = re.compile(r'\{|\}')

def _first_balanced_obj(text: str, start_idx: int) -> tuple[int, int] | None:
    """Return (start,end_exclusive) of first balanced {...} starting at/after start_idx."""
    s = text.find("{", start_idx)
    if s == -1: 
        return None
    depth = 0
    for i in range(s, len(text)):
        ch = text[i]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (s, i + 1)
    return None

def extract_all_json_objects(text: str) -> Iterable[Dict[str, Any]]:
    """Pull every balanced top-level JSON object out of messy model outputs."""
    # strip code fences if present
    if text.strip().startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text)
    out = []
    i = 0
    while True:
        span = _first_balanced_obj(text, i)
        if not span: break
        s, e = span
        chunk = text[s:e]
        i = e
        try:
            obj = json.loads(chunk)
            out.append(obj)
        except Exception:
            # ignore malformed chunk, continue scanning
            continue
    return out

REQUIRED_KEYS = {"persona","context","approach","conversation_plan"}

def is_valid_strategy(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict): return False
    if not REQUIRED_KEYS.issubset(obj.keys()): return False
    cp = obj.get("conversation_plan") or {}
    return isinstance(cp, dict) and "final_turn" in cp

def normalize_strategies(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Keep only entries named strategy_* and valid by schema."""
    picks = {}
    for k, v in raw.items():
        if isinstance(k, str) and k.lower().startswith("strategy_") and is_valid_strategy(v):
            picks[k] = v
    return picks

def strategy_signature(s: Dict[str, Any]) -> str:
    """Hash a few fields so we can dedupe across runs."""
    m = hashlib.sha1()
    m.update((s.get("persona","") + "||" + s.get("context","") + "||" + s.get("approach","")).encode("utf-8"))
    # include a stable subset of conversation_plan
    cp = s.get("conversation_plan") or {}
    m.update(("||".join(sorted(cp.keys())) + "||" + cp.get("final_turn","")).encode("utf-8"))
    return m.hexdigest()

def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    finally:
        try: os.unlink(tmp)
        except OSError: pass

def merge_and_save(progress_path: str, behavior_id: str, new_strats: Dict[str, Dict[str, Any]]) -> int:
    """Merge new strategies into a per-behavior store and save immediately. Returns #added."""
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            store = json.load(f)
    else:
        store = {}

    bucket = store.setdefault(behavior_id, {"items": [], "sigs": []})
    seen = set(bucket.get("sigs", []))
    added = 0
    for v in new_strats.values():
        sig = strategy_signature(v)
        if sig in seen: 
            continue
        bucket["items"].append(v)
        bucket["sigs"].append(sig)
        seen.add(sig)
        added += 1

    store[behavior_id] = bucket
    atomic_write_json(progress_path, store)
    return added
