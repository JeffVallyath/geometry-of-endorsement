"""The original SCALE normalization/parser functions, extracted verbatim.
Only these two pure functions are imported; no historical driver is run.
"""
import unicodedata

def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def parse_response(row, case):
    """Frozen first-line exact-alias rule; never use the gold answer to parse."""
    if row['truncated'] or not row['control_tokens_valid']:
        return None, None
    lines = row['text'].strip().splitlines()
    if not lines:
        return None, None
    line = lines[0].strip()
    forms = {normalize(line)}
    prefixes = list(forms)
    if normalize(line).startswith('Answer:'):
        prefixes.append(normalize(line)[len('Answer:'):].strip())
    for text in prefixes:
        forms.add(text)
        unperiod = text[:-1].rstrip() if text.endswith('.') else text
        forms.add(unperiod)
        for left, right in [('"', '"'), ("'", "'"), ('\u201c', '\u201d'), ('\u2018', '\u2019')]:
            for candidate in (text, unperiod):
                if len(candidate) >= 2 and candidate.startswith(left) and candidate.endswith(right):
                    unquoted = candidate[1:-1].strip()
                    forms.add(unquoted)
                    if unquoted.endswith('.'):
                        forms.add(unquoted[:-1].rstrip())
    entities = {case['registry'][normalize(s)] for s in forms
                if normalize(s) and normalize(s) in case['registry']}
    canonical = dict(case['canonical_labels'], **{case['decoy_id']: case['decoy_label']})
    strict = {entity for entity, label in canonical.items() if label == line}
    return (next(iter(entities)) if len(entities) == 1 else None,
            next(iter(strict)) if len(strict) == 1 else None)


