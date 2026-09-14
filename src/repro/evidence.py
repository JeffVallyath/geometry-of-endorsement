"""Offline checks of scientific claims, evidence bindings, and document safety.

Pattern checks guard explicitly recorded scientific qualifications, not writing
quality. They cannot establish arbitrary natural-language entailment; unfamiliar
scientific assertions still require review against the cited evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
NAME_MASKS = (r'Llama-3\.1-8B(?:-Instruct)?', r'Llama 3\.1',
              r'Gemma-2-9B(?:-it)?', r'MiniLM-L6-v2', r'SHA-256',
              r'AMPERE\+\+', r'US2016', r'GPT-2', r'\b1/2\b')


@dataclass
class Report:
    findings: list[dict] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)

    def add(self, rule, message, document='', unit='', level='error'):
        self.findings.append(dict(rule=rule, message=message, document=document,
                                  unit=unit, level=level))

    @property
    def errors(self):
        return [f for f in self.findings if f['level'] == 'error']

    @property
    def pending(self):
        return [f for f in self.findings if f['level'] == 'pending']

    def ok(self):
        return not self.errors

    def result(self):
        return dict(ok=self.ok(), errors=self.errors, pending=self.pending,
                    coverage=self.coverage,
                    scope='Declared scientific evidence and bounded qualification checks; not semantic or model evaluation.')


def rooted(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or p.as_posix() != name:
        raise ValueError('Noncanonical or escaping evidence path')
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Evidence path resolves outside repository')
    return resolved


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def resolve(data, dotted):
    """Resolve a JSON field; '*' explicitly quantifies every list/dict member."""
    parts = list(dotted) if isinstance(dotted, (list, tuple)) else dotted.split('.') if dotted else []
    def visit(value, remaining):
        if not remaining:
            return [value]
        key, *rest = remaining
        if key == '*':
            children = list(value.values()) if isinstance(value, dict) else value
            if not isinstance(children, list) or not children:
                raise ValueError('Wildcard evidence set must be nonempty')
            return [item for child in children for item in visit(child, rest)]
        return visit(value[int(key)] if isinstance(value, list) else value[key], rest)
    values = visit(data, parts)
    return values if '*' in parts else values[0]


def normalize(token):
    return str(token).replace('{,}', '').replace(',', '').rstrip('.')


def number_tokens(text):
    for pattern in NAME_MASKS:
        text = re.sub(pattern, ' ', text, flags=re.I)
    text = re.sub(r'```.*?```|~~~.*?~~~', ' ', text, flags=re.S)
    text = re.sub(r'`[^`]*`|\]\([^)]*\)', ' ', text)
    text = re.sub(r'[_^]\{[^}]*\}|[_^]\d+', ' ', text)
    return [normalize(t) for t in re.findall(r'(?<![\w.])(-?\d[\d,]*(?:\.\d+)?)', text.replace('{,}', ','))]


def render(value, entry):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Numerical binding must resolve to a finite number')
    if entry.get('percent', False):
        value *= 100
    decimals = entry.get('decimals')
    comma = ',' if entry.get('thousands') else ''
    return format(value, comma + (f'.{decimals}f' if decimals is not None else ''))


def sections(text):
    """Keep duplicate headings visible and check numerical introductory prose."""
    out, heading, lines = [], '', []
    for line in text.splitlines():
        hit = re.match(r'^#{2,3}\s+(.+)$', line)
        if hit:
            out.append((heading, '\n'.join(lines)))
            heading, lines = hit[1], []
        else:
            lines.append(line)
    out.append((heading, '\n'.join(lines)))
    return out


def matches(rule, text):
    flags = re.I | re.S
    if rule.get('alternatives'):
        return any(matches(option, text) for option in rule['alternatives'])
    return (all(re.search(p, text, flags) for p in rule.get('all_of', []))
            and (not rule.get('any_of') or any(re.search(p, text, flags) for p in rule['any_of'])))


def asserted_match(pattern, text):
    """Do not reject an explicit denial of an unsupported proposition."""
    for hit in re.finditer(pattern, text, re.I):
        prefix = re.split(r'[.;!?]|\bbut\b', text[:hit.start()], flags=re.I)[-1][-100:]
        if not re.search(r'\b(?:not|no|never|neither|cannot|unproven)\b', prefix, re.I):
            return True
    return False


def check_source(assertion, artifacts):
    value = resolve(artifacts, assertion['path'])
    parts = assertion['path'] if isinstance(assertion['path'], (list, tuple)) else assertion['path'].split('.')
    values = value if '*' in parts else [value]
    if not values:
        raise ValueError('No supporting evidence values')
    operations = [key for key in ('equals', 'contains', 'lt', 'le', 'gt', 'ge') if key in assertion]
    if len(operations) != 1:
        raise ValueError('Exactly one evidence predicate is required')
    key, target = operations[0], assertion[operations[0]]
    def valid(v):
        if key == 'equals':
            return type(v) is type(target) and v == target
        if key == 'contains':
            return target in v
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
        return {'lt': v < target, 'le': v <= target, 'gt': v > target, 'ge': v >= target}[key]
    return all(valid(v) for v in values)


def check_unit(body, unit, artifacts, report, document='', strict=True, all_sections=None):
    uid = unit['id']
    allowed, tokens = set(), number_tokens(body)
    counts = dict(document=document, unit=uid, bound=0, pending=0,
                  headline_bound=0, headline_pending=0)
    for item in unit.get('numbers', []):
        status = item.get('status', 'verified')
        try:
            if status == 'verified':
                expected = render(resolve(artifacts, item['path']), item)
                if 'text' in item and normalize(item['text']) != normalize(expected):
                    report.add('number-contract-mismatch', item['path'], document, uid)
                counts['bound'] += 1
                counts['headline_bound'] += int(item.get('headline', False))
            elif status == 'pending':
                expected = str(item['value'])
                counts['pending'] += 1
                counts['headline_pending'] += int(item.get('headline', False))
                report.add('artifact-pending', f"{expected}: {item.get('claimed_source_family', 'unspecified')} / {item.get('reason', 'No source resolved')}", document, uid, 'error' if strict else 'pending')
            else:
                raise ValueError('Unknown binding status')
            allowed.add(normalize(expected))
            if item.get('required', True) and normalize(expected) not in tokens:
                report.add('number-missing', expected, document, uid)
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as error:
            report.add('binding-malformed', f"{item.get('path', '')}: {error}", document, uid)
    allowed.update(normalize(x['text']) for x in unit.get('literals', []))
    for token in set(tokens) - allowed:
        report.add('number-undeclared', token, document, uid)
    flat = re.sub(r'\s+', ' ', body)
    for fact in unit.get('facts', []):
        target = flat
        if fact.get('in_section'):
            destinations = [b for h, b in (all_sections or []) if re.search(fact['in_section'], h, re.I)]
            if len(destinations) != 1:
                report.add('fact-destination-missing', fact['id'], document, uid)
                continue
            target = re.sub(r'\s+', ' ', destinations[0])
        if not matches(fact, target):
            report.add('material-qualification-missing', f"{fact['id']}: {fact['description']}", document, uid)
        for assertion in fact.get('source_evidence', []):
            try:
                if not check_source(assertion, artifacts):
                    report.add('scientific-evidence-mismatch', f"{fact['id']}: {assertion['path']}", document, uid)
            except (KeyError, IndexError, TypeError, ValueError) as error:
                report.add('scientific-evidence-missing', f"{fact['id']}: {error}", document, uid)
    for rule in unit.get('unsupported', []):
        if asserted_match(rule['pattern'], flat):
            report.add('unsupported-scientific-assertion', f"{rule['id']}: {rule['reason']}", document, uid)
    counts['total'] = counts['bound'] + counts['pending']
    report.coverage.append(counts)


def check_document_safety(text, path, root, report, document):
    fence, display = None, False
    plain = []
    for line_number, line in enumerate(text.splitlines(), 1):
        mark = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
        if mark:
            if fence is None:
                fence = mark[1]
            elif mark[1][0] == fence[0] and len(mark[1]) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        plain.append(line)
        mathline = re.sub(r'`[^`]*`|\\\$', '', line)
        pairs = mathline.count('$$')
        if pairs % 2:
            display = not display
        if not display and mathline.replace('$$', '').count('$') % 2:
            report.add('unbalanced-math', f'line {line_number}', document)
    if fence:
        report.add('unclosed-fence', 'Code/math fence is not closed', document)
    if display:
        report.add('unbalanced-math', 'Display math is not closed', document)
    # Ordinary local Markdown links: no network checks or account access.
    for target in re.findall(r'\]\(([^)]+)\)', '\n'.join(plain)):
        target = target.strip().strip('<>')
        if ' "' in target:
            target = target.split(' "', 1)[0]
        try:
            parsed = urlsplit(target)
        except ValueError:
            report.add('unsafe-link', 'Malformed link target', document)
            continue
        if parsed.scheme:
            if parsed.scheme.lower() not in ('https', 'http', 'mailto'):
                report.add('unsafe-link', 'Unsupported link scheme', document)
            continue
        if not parsed.path:  # Anchors are resolved by the ordinary Markdown renderer.
            continue
        candidate = (path.parent / unquote(parsed.path)).resolve()
        if not candidate.is_relative_to(root.resolve()):
            report.add('unsafe-link', 'Local link escapes repository', document)
        elif not candidate.exists():
            report.add('broken-link', target, document)


def check(root=ROOT, registry_path=None, *, document=None, strict=True, overrides=None):
    root = Path(root)
    registry = load(Path(registry_path) if registry_path else root / 'reproducibility/evidence.json')
    if registry.get('schema_version') != 1:
        raise ValueError('Unsupported evidence registry')
    report, artifacts = Report(), {}
    for key, name in registry['sources'].items():
        try:
            artifacts[key] = load(rooted(root, name))
        except (OSError, ValueError) as error:
            report.add('artifact-missing', f'{key}: {error}')
    for assertion in registry.get('source_checks', []):
        try:
            if not check_source(assertion, artifacts):
                report.add('scientific-evidence-mismatch', assertion['id'])
        except (KeyError, IndexError, TypeError, ValueError) as error:
            report.add('binding-malformed', f"{assertion['id']}: {error}")
    documents = [d for d in registry['documents'] if document is None or d['id'] == document]
    if not documents:
        report.add('document-missing', 'No matching scientific document')
    for doc in documents:
        path = rooted(root, doc['path'])
        try:
            text = (overrides or {}).get(doc['id'], path.read_text(encoding='utf8'))
        except OSError:
            report.add('document-missing', doc['path'], doc['id'])
            continue
        units, covered = sections(text), set()
        for contract in doc['units']:
            found = [(i, body) for i, (heading, body) in enumerate(units)
                     if re.search(contract['heading_pattern'], heading, re.I)]
            if len(found) != 1:
                report.add('unit-missing-or-ambiguous', contract['id'], doc['id'])
                continue
            index, body = found[0]
            covered.add(index)
            check_unit(body, contract, artifacts, report, doc['id'], strict, units)
        for i, (heading, body) in enumerate(units):
            if i not in covered and number_tokens(body):
                report.add('uncontracted-numbers', heading or 'Introduction', doc['id'])
        check_document_safety(text, path, root, report, doc['id'])
    return report
