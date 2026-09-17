"""Portable configuration facade; no repository or runtime discovery."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / 'models.json').read_text())['models']
GEMMA, QWEN = MODELS['gemma'], MODELS['qwen']
RANK = 16
KIND = {'INVARIANT_SET': 'invariant', 'FREE_OVERWRITE': 'free'}
INSTRUCTION = '\nReply with only that answer on the first line.'

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
