"""Frozen scorers for ICMH1-RIPPLE-v2.

Two rules only, both fixed before any model output (AMENDMENT.md section 3):

1. strict_line + public label/alias equality: the primary C and D correctness rule.
   First generated line, outer whitespace removed, at most one final punctuation
   character from '.,;:!?' removed; case-sensitive equality against a public answer
   or alias processed by the identical rule. No line skipping, no substring rescue.

2. typed full-string entity extraction: the primary A rule and the D validity/
   distinctness rule. Apply the same line/punctuation rule, then NFKC + casefold +
   whitespace collapse, then exact full-string match against the source entity alias
   registry restricted to the record's declared object_type. Exactly one entity ID
   must match; otherwise abstain. The gold target is never consulted.
"""
import unicodedata

PUNCTUATION = '.,;:!?'


def strict_line(text):
    """First generated line, outer whitespace removed, at most one final punctuation removed."""
    if text is None:
        return None
    line = text.lstrip().split('\n', 1)[0].strip()
    if line and line[-1] in PUNCTUATION:
        line = line[:-1]
    return line


def answer_forms(answers):
    """Every public answer value and alias, processed by the same strict rule."""
    forms = []
    for answer in answers:
        for raw in [answer['value']] + list(answer.get('aliases') or []):
            processed = strict_line(raw)
            if processed:
                forms.append(processed)
    return forms


def strict_correct(text, answers):
    line = strict_line(text)
    if not line:
        return False
    return line in answer_forms(answers)


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def registry_key(text):
    processed = strict_line(text)
    return norm(processed) if processed else ''


def build_registry(entities, object_types=None):
    """{object_type: {normalized full string: [entity ids]}} from the source entity manifest."""
    registry = {}
    for entity_id, entity in entities.items():
        for object_type in entity.get('types') or []:
            if object_types is not None and object_type not in object_types:
                continue
            bucket = registry.setdefault(object_type, {})
            for raw in list(entity.get('labels') or []) + list(entity.get('aliases') or []):
                key = registry_key(raw)
                if not key:
                    continue
                ids = bucket.setdefault(key, [])
                if entity_id not in ids:
                    ids.append(entity_id)
    for bucket in registry.values():
        for key in bucket:
            bucket[key] = sorted(bucket[key])
    return registry


def extract(text, registry, object_type):
    """Typed full-string entity extraction; abstain on empty, unmatched or ambiguous."""
    line = strict_line(text)
    if not line:
        return {'line': line, 'key': '', 'status': 'empty', 'entity_id': None}
    key = norm(line)
    ids = (registry.get(object_type) or {}).get(key) or []
    if len(ids) == 1:
        return {'line': line, 'key': key, 'status': 'unique', 'entity_id': ids[0]}
    if len(ids) > 1:
        return {'line': line, 'key': key, 'status': 'ambiguous', 'entity_id': None, 'candidates': ids}
    return {'line': line, 'key': key, 'status': 'unmatched', 'entity_id': None}


def unresolved(status):
    """G4 counts abstentions: empty, unmatched or ambiguous. A wrong unique entity is resolved."""
    return status in ('empty', 'unmatched', 'ambiguous')
