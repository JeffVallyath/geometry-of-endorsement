"""Deterministic trusted renderer and answer-format parsing (no neural component).

The requested output encoding is parsed from the visible question text by a fixed rule:
  'Use X for yes and Y for no.'  -> {True: X, False: Y}
  'Answer with exactly one of No, Yes.' (no Use-line) -> {True: 'Yes', False: 'No'}
Entity/enum answers are output as the computed canonical string.
"""
from __future__ import annotations

import re

USE = re.compile(r'Use (\S+) for yes and (\S+) for no\.')
ONE_OF = re.compile(r'Answer with exactly one of ([^\n]+?)\.\s*$', re.M)


def answer_mapping(question_text):
    m = USE.search(question_text)
    if m:
        return {True: m.group(1), False: m.group(2)}
    m = ONE_OF.search(question_text)
    if m:
        labels = [x.strip() for x in m.group(1).split(',')]
        if sorted(labels) == ['No', 'Yes']:
            return {True: 'Yes', False: 'No'}
    return None


def render(result, question_text):
    """Return (canonical answer string or None, error). Never consults history or metadata."""
    if result.get('status') != 'OK':
        return None, 'EXECUTION_' + result.get('error', 'ERROR')
    v = result['value']
    if result['type'] == 'bool':
        mapping = answer_mapping(question_text)
        if mapping is None:
            return None, 'RENDER_NO_BOOLEAN_FORMAT'
        return mapping[bool(v)], None
    if not isinstance(v, str):
        return None, 'RENDER_TYPE'
    return v, None
