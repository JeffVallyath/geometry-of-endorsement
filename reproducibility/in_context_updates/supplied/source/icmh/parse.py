"""Scorers for the prospective ICMH1-RIPPLE-v2-LP1 amendment.

Strict scorer unchanged; entity registry precedence amended before any model output:

1. strict_line + public label/alias equality: the primary C and D correctness rule.
   First generated line, outer whitespace removed, at most one final punctuation
   character from '.,;:!?' removed; case-sensitive equality against a public answer
   or alias processed by the identical rule. No line skipping, no substring rescue.

2. typed full-string entity extraction: the primary A rule and the D validity/
   distinctness rule. Apply the same line/punctuation rule, then NFKC + casefold +
   whitespace collapse, then exact full-string match against the source entity alias
   registry restricted to the record's declared object_type. A source-label match
   takes precedence over alias-only matches. Multiple source-label matches remain
   ambiguous; if there is no label match, aliases must identify exactly one ID.
   This is a declared interpretation convention, not a claim about model intent.
   No target answer or per-question gold is used. build_registry_union retains v2
   for a same-response sensitivity analysis.
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


def build_registry_union(entities, object_types=None):
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


def build_registry(entities, object_types=None):
    """Source labels take priority; otherwise use aliases. Never consult gold.

    The returned shape remains {object_type: {normalized string: [candidate IDs]}}.
    A multi-label collision is retained as multiple IDs, so extract() abstains.
    No entity, label, alias or type is removed from the source data.
    """
    labels, aliases = {}, {}
    for entity_id, entity in entities.items():
        for object_type in entity.get('types') or []:
            if object_types is not None and object_type not in object_types:
                continue
            label_bucket = labels.setdefault(object_type, {})
            alias_bucket = aliases.setdefault(object_type, {})
            for field, bucket in (('labels', label_bucket), ('aliases', alias_bucket)):
                for raw in entity.get(field) or []:
                    key = registry_key(raw)
                    if key:
                        bucket.setdefault(key, set()).add(entity_id)
    result = {}
    for object_type in sorted(set(labels) | set(aliases)):
        label_bucket, alias_bucket = labels.get(object_type, {}), aliases.get(object_type, {})
        result[object_type] = {
            key: sorted(label_bucket.get(key) or alias_bucket.get(key) or ())
            for key in sorted(set(label_bucket) | set(alias_bucket))
        }
    return result


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
