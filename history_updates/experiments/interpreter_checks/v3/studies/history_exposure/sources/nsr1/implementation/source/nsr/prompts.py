"""NSR1 single development prompt set (frozen before protected inference; see records/EVAL_FREEZE.json).

One example world, eight fixed examples (hand-written; names and wordings disjoint from every final
root and from every final question template). Every arm sees the same schema and example world,
rendered in that arm's own output format.

Revision 2 (development roots only, before any protected outcome): operator and relation glosses, nested-path
guidance, program-first flat output, sentence-style canonical records for the table reader.
Revision 3 (development roots only): explicit BOTH/EITHER/SAME rules and the topic-name rule.
"""
from __future__ import annotations

import json

PROMPT_SET_ID = 'NSR1-PROMPTS-3'

MEMBER_SCHEMA = '''Member records describe one person and one proposal, with three fields:
  stance: "favor" or "oppose"   (in favor of / supports / backs = "favor"; opposed to / against = "oppose")
  status: "pending", "approved", "rejected" or "deferred"
  day (the shipment day): "Monday", "Tuesday", "Wednesday", "Thursday" or "Friday"'''
FIXED_RELATIONS = '''Relation records map one entity to exactly one entity:
  team: person -> team                   (the team the person is on)
  hub: team -> hub                       (the hub where the team is based)
  region: hub -> region                  (the region the hub is in)
  organization: person -> organization   (the organization the person works for; the person's employer)
  founder: organization -> person        (the person who founded the organization)
  location: person -> city               (the city where the person lives)'''
SNAPSHOTS = '''Snapshots: "CURRENT" = the state after every correction so far; "ORIGINAL" = the original records before any
correction; "EVENT:k" = the state right after correction k (k = 1, 2, ...).'''
PUBLIC_RELATIONS = ''
PUBLIC_WORDING = {}


def use_public_schema(vocab):
    """Declared input schema of the public bridge: a source-level relation vocabulary shared by every arm
    (relation name and its record wording); never a per-question path."""
    global PUBLIC_RELATIONS, PUBLIC_WORDING
    rows = '\n'.join(f'  {r["relation"]}: {r["record_wording"]}' for r in vocab)
    PUBLIC_WORDING = {r['relation']: r['record_wording'] for r in vocab}
    PUBLIC_RELATIONS = ('Relation records map one entity to exactly one entity. In this dataset every record is a plain sentence in '
                        'one of the wordings below ({S} is the entity, {O} is its value). Use the relation name exactly as listed '
                        '(relation name: record wording):\n' + rows +
                        '\nA correction line "SET <record sentence>" writes that relation value for that entity. Records and '
                        'corrections are cited by position: "Record 1" is the first record line, "Correction 1" the first '
                        'correction line.\n(The examples below additionally use two example-only relations, team: person -> team '
                        'and hub: team -> hub.)')


def use_fixed_schema():
    global PUBLIC_RELATIONS, PUBLIC_WORDING
    PUBLIC_RELATIONS, PUBLIC_WORDING = '', {}


def schema():
    return 'SCHEMA (fixed)\n' + MEMBER_SCHEMA + '\n' + (PUBLIC_RELATIONS or FIXED_RELATIONS) + '\n' + SNAPSHOTS


DSL = '''QUERY PROGRAM (one JSON object; these are the only allowed forms)
{"op":"READ","entity":<person>,"field":"stance"|"status"|"day","topic":<proposal>,"snapshot":S}   the value of a member field
{"op":"READ","entity":<name>,"field":<relation>,"topic":null,"snapshot":S}                      one relation step
{"op":"FOLLOW","entity":<name>,"path":[<relation>, ...],"snapshot":S}                             1 to 4 relation steps
{"op":"TEST","arg":<READ or FOLLOW>,"value":<allowed value>}        true when the value read equals the given value
{"op":"SAME","left":<READ or FOLLOW>,"right":<READ or FOLLOW>}      true when the two values read are equal
{"op":"BOTH","left":<true/false program>,"right":<true/false program>}     logical AND: true only when both parts are true
{"op":"EITHER","left":<true/false program>,"right":<true/false program>}   logical OR: true when at least one part is true
{"op":"NOT","arg":<true/false program>}                                    negation
S is "CURRENT", "ORIGINAL" or "EVENT:k".
Rules:
- A yes/no question compiles to a true/false program; a question asking for a name or a field value compiles to a
  READ or a FOLLOW.
- "X supports / favors / is in favor of P" is TEST(READ stance, "favor"); "X opposes / is against P" is
  TEST(READ stance, "oppose").
- BOTH asks whether each of two statements holds: "do X and Y both support P" / "P is supported by both X and Y"
  is BOTH of two TESTs (X supports P AND Y supports P). It is never SAME.
- EITHER asks whether at least one holds: "X or Y (or both) supports P" / "at least one of X and Y supports P" is
  EITHER of two TESTs.
- SAME compares two values and is true when they are equal, whatever they are: use it only when the question asks
  whether two things are the same / match / agree ("are X and Y on the same side of P").
- The topic of a member field is the proposal name alone: for "the Beacon proposal" the topic is "Beacon".
- FOLLOW starts from the entity named in the question and lists one relation per step, innermost first: "the R2 of
  the R1 of X" is {"op":"FOLLOW","entity":"X","path":["R1","R2"],...}. Use only relation names from the schema and do
  not skip a step.
- Copy every name exactly as written in the question: only the proper name itself, without "the" or describing words.
  Keep the operands in the order the question mentions them.
- The answer-format lines of a question ("Answer with exactly one of ...", "Use A for yes ...") are handled
  elsewhere: never encode them. No code, no constants, no comments.
- If the question cannot be expressed with these forms, output {"op":"UNSUPPORTED","reason":"<short reason>"}.'''

EX_RECORDS = [
    'Record 01: Ansel Okafor on the Beacon proposal: in favor; status approved; shipment day Tuesday.',
    'Record 02: Bianca Voss on the Beacon proposal: opposed; status pending; shipment day Friday.',
    'Record 03: Cyrus Lindgren on the Beacon proposal: opposed; status rejected; shipment day Monday.',
    'Record 04: Dalia Petrov on the Beacon proposal: in favor; status deferred; shipment day Tuesday.',
    'Record 05: Emeka Sorensen on the Beacon proposal: opposed; status pending; shipment day Wednesday.',
    'Record 06: Dalia Petrov is on the Lynx team.',
    'Record 07: The Lynx team is based at the Vega hub.',
]
EX_UPDATES = [
    'Correction 1: Bianca Voss now favors the Beacon proposal.',
    'Correction 2: Cyrus Lindgren now opposes the Beacon proposal.',
]
EX_CURRENT = [EX_RECORDS[0], 'Record 02: Bianca Voss on the Beacon proposal: in favor; status pending; shipment day Friday.'] + EX_RECORDS[2:]


def _r(e, snap='CURRENT'):
    return {'op': 'READ', 'entity': e, 'field': 'stance', 'topic': 'Beacon', 'snapshot': snap}


def _fav(e, snap='CURRENT'):
    return {'op': 'TEST', 'arg': _r(e, snap), 'value': 'favor'}


def _fact(src, e, v, snap='CURRENT'):
    return {'source': src, 'entity': e, 'field': 'stance', 'topic': 'Beacon', 'value': v, 'snapshot': snap}


# (question, program, answer, flat facts)
EXAMPLES = [
    ('Right now, does Ansel Okafor support the Beacon proposal?\nAnswer with exactly one of No, Yes.',
     _fav('Ansel Okafor'), 'Yes', [_fact('Record 01', 'Ansel Okafor', 'favor')]),
    ('Is Dalia Petrov presently opposed to the Beacon proposal?\nUse B for yes and A for no.\nAnswer with exactly one of A, B.',
     {'op': 'TEST', 'arg': _r('Dalia Petrov'), 'value': 'oppose'}, 'A', [_fact('Record 04', 'Dalia Petrov', 'favor')]),
    ('Do Ansel Okafor and Cyrus Lindgren hold the same position on the Beacon proposal?\nAnswer with exactly one of No, Yes.',
     {'op': 'SAME', 'left': _r('Ansel Okafor'), 'right': _r('Cyrus Lindgren')}, 'No',
     [_fact('Record 01', 'Ansel Okafor', 'favor'), _fact('Correction 2', 'Cyrus Lindgren', 'oppose')]),
    ('Is the Beacon proposal backed by both Bianca Voss and Dalia Petrov?\nUse A for yes and B for no.\nAnswer with exactly one of A, B.',
     {'op': 'BOTH', 'left': _fav('Bianca Voss'), 'right': _fav('Dalia Petrov')}, 'A',
     [_fact('Correction 1', 'Bianca Voss', 'favor'), _fact('Record 04', 'Dalia Petrov', 'favor')]),
    ('Is at least one of Cyrus Lindgren and Emeka Sorensen in favor of the Beacon proposal?\nAnswer with exactly one of No, Yes.',
     {'op': 'EITHER', 'left': _fav('Cyrus Lindgren'), 'right': _fav('Emeka Sorensen')}, 'No',
     [_fact('Correction 2', 'Cyrus Lindgren', 'oppose'), _fact('Record 05', 'Emeka Sorensen', 'oppose')]),
    ('According to the original records (before corrections), did Bianca Voss support the Beacon proposal?\nAnswer with exactly one of No, Yes.',
     _fav('Bianca Voss', 'ORIGINAL'), 'No', [_fact('Record 02', 'Bianca Voss', 'oppose', 'ORIGINAL')]),
    ('At which hub is the team of Dalia Petrov based?',
     {'op': 'FOLLOW', 'entity': 'Dalia Petrov', 'path': ['team', 'hub'], 'snapshot': 'CURRENT'}, 'Vega',
     [{'source': 'Record 06', 'entity': 'Dalia Petrov', 'field': 'team', 'topic': None, 'value': 'Lynx', 'snapshot': 'CURRENT'},
      {'source': 'Record 07', 'entity': 'Lynx', 'field': 'hub', 'topic': None, 'value': 'Vega', 'snapshot': 'CURRENT'}]),
    ('Is at least one of Cyrus Lindgren and Bianca Voss in favor of the Beacon proposal?\nUse B for yes and A for no.\nAnswer with exactly one of A, B.',
     {'op': 'EITHER', 'left': _fav('Cyrus Lindgren'), 'right': _fav('Bianca Voss')}, 'B',
     [_fact('Correction 2', 'Cyrus Lindgren', 'oppose'), _fact('Correction 1', 'Bianca Voss', 'favor')]),
]


def J(x):
    return json.dumps(x, separators=(',', ':'))


def _world():
    return 'Example records:\n' + '\n'.join(EX_RECORDS) + '\nExample corrections:\n' + '\n'.join(EX_UPDATES)


ANSWER_FORMAT = '''ANSWER FORMAT
For a yes/no question output exactly one of the labels the question asks for. For a question asking for a name or a
field value output only that name or value. No explanation and no other text.'''


# ----------------------------- system prompts -----------------------------
def compiler_system():
    ex = '\n\n'.join(f'Question: {q}\nProgram: {J(p)}' for q, p, _, _ in EXAMPLES)
    return ('You translate one question into a query program. You are NOT shown any records, corrections or values: '
            'translate the wording only.\n\n' + schema() + '\n\n' + DSL +
            '\n\nOutput exactly one JSON object and nothing else.\n\nEXAMPLES\n' + ex)


def record_system():
    return ('You convert ONE record line into JSON. Copy names exactly as written (a name never includes the final '
            'full stop of the sentence). Output exactly one JSON object and nothing else.\n\n' + schema() + '''

OUTPUT FORMS
Member record: {"kind":"member","person":<name>,"project":<proposal>,"stance":...,"status":...,"day":...}
  (omit a field that the line does not state)
Relation record: {"kind":"relation","entity":<name>,"relation":<relation name from the schema>,"value":<name>}
A line that states no record (a heading, an instruction, an end marker): {"kind":"none"}

EXAMPLES
Line: Record 02: Bianca Voss on the Beacon proposal: opposed; status pending; shipment day Friday.
JSON: {"kind":"member","person":"Bianca Voss","project":"Beacon","stance":"oppose","status":"pending","day":"Friday"}

Line: Record 06: Dalia Petrov is on the Lynx team.
JSON: {"kind":"relation","entity":"Dalia Petrov","relation":"team","value":"Lynx"}

Line: Record 07: The Lynx team is based at the Vega hub.
JSON: {"kind":"relation","entity":"Lynx","relation":"hub","value":"Vega"}

Line: Tobias is in favor of the Lighthouse proposal.
JSON: {"kind":"member","person":"Tobias","project":"Lighthouse","stance":"favor"}

Line: End of records.
JSON: {"kind":"none"}''')


def update_system():
    return ('You convert ONE authoritative correction statement into a JSON transaction. Copy names exactly as written (a '
            'name never includes the final full stop of the sentence). Output exactly one JSON object and nothing else.\n\n'
            + schema() + '''

OUTPUT FORM
{"writes":[{"entity":<name>,"field":<field or relation>,"topic":<proposal or null>,"value":<new value>}, ...],
 "restores":[{"entity":<name>,"field":<field or relation>,"topic":<proposal or null>,"snapshot":"ORIGINAL" or "EVENT:k"}, ...]}
Use "writes" for new values and "restores" when the statement restores an earlier value; omit an empty list.
For member fields the topic is the proposal; for relations the topic is null. A statement that changes several
things at once is ONE transaction with several entries.

EXAMPLES
Statement: Correction 1: Bianca Voss now favors the Beacon proposal.
JSON: {"writes":[{"entity":"Bianca Voss","field":"stance","topic":"Beacon","value":"favor"}]}

Statement: Correction 4: The status of Ansel Okafor's Beacon record is now deferred.
JSON: {"writes":[{"entity":"Ansel Okafor","field":"status","topic":"Beacon","value":"deferred"}]}

Statement: Correction 5 (applied together): Cyrus Lindgren now favors the Beacon proposal; Emeka Sorensen's Beacon shipment day is now Thursday.
JSON: {"writes":[{"entity":"Cyrus Lindgren","field":"stance","topic":"Beacon","value":"favor"},{"entity":"Emeka Sorensen","field":"day","topic":"Beacon","value":"Thursday"}]}

Statement: Correction 6: Bianca Voss's Beacon stance is restored to its original value.
JSON: {"restores":[{"entity":"Bianca Voss","field":"stance","topic":"Beacon","snapshot":"ORIGINAL"}]}

Statement: Correction 7: Dalia Petrov has moved to the Puma team.
JSON: {"writes":[{"entity":"Dalia Petrov","field":"team","topic":null,"value":"Puma"}]}''')


def flat_system():
    ex = '\n\n'.join(f'Question: {q}\nJSON: {J({"program": p, "facts": f})}' for q, p, _, f in EXAMPLES)
    return ('You are given records, authoritative corrections (applied in order) and one question. Translate the problem '
            'into a small symbolic problem: a query program and the facts that the program reads. A solver will compute '
            'the answer from your program and facts.\n\n' + schema() + '\n\n' + DSL + '''

OUTPUT FORM
{"program":<query program>,
 "facts":[{"source":<"Record NN" or "Correction K" that determines the value>,"entity":<name>,"field":<field or relation>,
           "topic":<proposal or null>,"value":<value in that snapshot, in the schema spelling>,"snapshot":S}, ...]}
Write the program first. Then list ONLY the facts that the program reads: one fact per READ (or per FOLLOW step), with
the value it has in the snapshot the program reads it from (for "CURRENT": the value after all corrections), citing
the record or correction that determines that value. Do not list any other record. Values use the schema spelling
(for example "favor" / "oppose", never "in favor" / "opposed"). Output exactly one JSON object and nothing else.

''' + _world() + '\n\nEXAMPLES (for the example records and corrections above)\n' + ex)


def native_system():
    ex = '\n\n'.join(f'Question: {q}\nAnswer: {a}' for q, _, a, _ in EXAMPLES)
    return ('You are given records, authoritative corrections (applied in order) and one question. Corrections override '
            'the records. Answer the question.\n\n' + schema() + '\n\n' + ANSWER_FORMAT + '\n\n' + _world() +
            '\n\nEXAMPLES (for the example records and corrections above)\n' + ex)


def current_system():
    ex = '\n\n'.join(f'Question: {q}\nAnswer: {a}' for q, _, a, _ in EXAMPLES)
    return ('You are given the CURRENT records (the state after every correction), the ORIGINAL records (the state before '
            'any correction) and one question. Answer the question from these records.\n\n' + schema() + '\n\n' + ANSWER_FORMAT +
            '\n\nExample CURRENT records:\n' + '\n'.join(EX_CURRENT) + '\nExample ORIGINAL records:\n' + '\n'.join(EX_RECORDS) +
            '\n\nEXAMPLES (for the example records above)\n' + ex)


def renderer_system():
    """Diagnostic only (never in the trusted path): a neural renderer that sees the answer and the history."""
    ex = '\n\n'.join(f'Question: {q}\nComputed answer: {a}\nFinal answer: {a}' for q, _, a, _ in EXAMPLES[:4])
    return ('You are given records, authoritative corrections, one question and the answer computed by a solver. '
            'State the final answer to the question.\n\n' + schema() + '\n\n' + ANSWER_FORMAT + '\n\n' + _world() +
            '\n\nEXAMPLES (for the example records and corrections above)\n' + ex)


REPAIR = ('Your output was rejected by the format validator: {error}\n'
          'Output a corrected JSON object only.')


# ----------------------------- user turns -----------------------------
def history_text(records, updates):
    s = 'Records:\n' + '\n'.join(records)
    s += '\nCorrections:\n' + ('\n'.join(updates) if updates else '(none)')
    return s


def compiler_user(question):
    return 'Question: ' + question + '\nProgram:'


def exposed_compiler_user(records, updates, question):
    """Diagnostic: the exact compiler prompt augmented with the (obsolete) history."""
    return history_text(records, updates) + '\n\nQuestion: ' + question + '\nProgram:'


def record_user(line):
    return 'Line: ' + line + '\nJSON:'


def update_user(statement):
    return 'Statement: ' + statement + '\nJSON:'


def flat_user(records, updates, question):
    return history_text(records, updates) + '\n\nQuestion: ' + question + '\nJSON:'


def native_user(records, updates, question):
    return history_text(records, updates) + '\n\nQuestion: ' + question + '\nAnswer:'


def current_user(cur_table, orig_table, question):
    return ('CURRENT records:\n' + '\n'.join(cur_table) + '\nORIGINAL records:\n' + '\n'.join(orig_table) +
            '\n\nQuestion: ' + question + '\nAnswer:')


def renderer_user(records, updates, question, answer):
    return history_text(records, updates) + '\n\nQuestion: ' + question + '\nComputed answer: ' + answer + '\nFinal answer:'
