import os
from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    raise RuntimeError("HF_TOKEN not set; export it or put it in your env")
# ... rest of your logic ...
