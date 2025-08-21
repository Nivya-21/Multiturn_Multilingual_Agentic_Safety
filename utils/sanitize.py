# utils/sanitize.py
import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)

# Optional: catch other scratchpad styles that some models emit
_OTHER_BLOCKS = re.compile(
    r"<(reasoning|scratchpad|analysis)>.*?</\1>\s*",
    flags=re.DOTALL | re.IGNORECASE
)

# Optional: simple “Reasoning:” headers
_REASONING_HDR = re.compile(
    r"(?is)^(?:\s*(?:reasoning|chain[- ]?of[- ]?thought|analysis)\s*:.*?)(?:\n{2,}|$)"
)

def strip_reasoning(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = _THINK_BLOCK.sub("", text)
    text = _OTHER_BLOCKS.sub("", text)
    text = _REASONING_HDR.sub("", text)
    return text.strip()
