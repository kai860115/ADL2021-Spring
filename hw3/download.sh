wget https://www.dropbox.com/s/qkjn0vbee90mxy1/pytorch_model.bin?dl=1 -O ./mt5-summarization/pytorch_model.bin

python tw_rouge/twrouge.py

python -c "
import nltk
from filelock import FileLock
from transformers.file_utils import is_offline_mode
try:
    nltk.data.find(\"tokenizers/punkt\")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(\"Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files\")
    with FileLock(\".lock\") as lock:
        nltk.download(\"punkt\", quiet=True)
"