"""Exact historical FIT demonstrations, materialized to avoid old dataset trees."""
import json
from pathlib import Path

def demo_block(procedure):
    return json.loads((Path(__file__).resolve().parents[1] / 'prompts.json').read_text())[procedure]
