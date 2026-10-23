"""Independent SQLite reference for the NSR1 contract (docs/FORMAL_SPEC.md); imports nothing from nsr/.

Facts are versioned rows in SQLite: an address's value at snapshot i is its newest row with
version <= i. Every READ, FOLLOW hop and restore source is a SQL lookup; validation and Boolean
combination are plain Python (standard library only).

Choices made where the spec is silent or ambiguous:
- execute(): MALFORMED is checked over the whole AST first, then HOP_LIMIT / TYPE_ERROR in
  left-to-right post-order; both happen before any lookup (empty read set). Evaluation is then
  strict left-to-right post-order. Runtime errors are UNSUPPORTED_SNAPSHOT (raised when its
  READ/FOLLOW is evaluated, before any lookup of that node), MISSING and AMBIGUOUS.
- "Every subexpression is evaluated" is taken literally: an operand right of a failing one is
  still evaluated, the FIRST error in post-order is returned, and the read set holds every
  lookup actually made (a FOLLOW stops at its own failing hop). The other reading - stop at
  the first error, read set ends at the failing lookup - is `SQLStore.evaluate_all = False`.
  Both readings return the same status / value / error; only error-case read sets differ.
- Operand-kind violations (TEST of a bool, BOTH/EITHER/NOT of a value, SAME of bools) are
  TYPE_ERROR; wrong ops / key sets / non-list paths are MALFORMED.
- READ/FOLLOW may omit "snapshot": snapshot_default applies (CURRENT when it is None).
- Snapshots: exact "CURRENT", "ORIGINAL" or canonical "EVENT:k" (1 <= k <= current index);
  anything else (non-strings, "EVENT:01", ...) is UNSUPPORTED_SNAPSHOT. Restores reject CURRENT.
- Names (entities, member topics, relation values, entity TEST literals) must be str of 1..80
  characters, else TYPE_ERROR, in records and in the DSL.
- Entity types form one monotone registry (initial state + accepted events) over one shared
  namespace; `entity`-typed (public-bridge) positions assign no type.
- Check order, initial state: MALFORMED (shapes) -> TYPE_ERROR -> TYPE_CONFLICT.
- Check order, event: MALFORMED (not a dict) -> UNSUPPORTED_OPERATION (unknown top-level key)
  -> MALFORMED (shapes) -> TYPE_ERROR -> TYPE_CONFLICT -> restore sources in order
  (UNSUPPORTED_SNAPSHOT, MISSING, AMBIGUOUS) -> CONFLICTING_WRITES. "writes"/"restores" are
  each optional and an event with no operations is accepted (it still publishes a snapshot).
- Public-bridge relation names are not listed in the spec: pass them as `public_relations`.
"""
import re
import sqlite3

ENUMS = {"stance": ("favor", "oppose"),
         "status": ("pending", "approved", "rejected", "deferred"),
         "day": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")}
RELATIONS = {"team": ("person", "team"), "hub": ("team", "hub"), "region": ("hub", "region"),
             "organization": ("person", "organization"), "founder": ("organization", "person"),
             "location": ("person", "city")}  # field -> (domain type, range type)
ANY = "entity"            # domain/range of public-bridge relations; chains with any type
PUBLIC_SCHEMA = ()        # default public-bridge relation names (not enumerated in the spec)
AMBIGUOUS_VALUE = "<AMBIGUOUS>"
NODE_KEYS = {"READ": {"entity", "field", "topic", "snapshot"},
             "FOLLOW": {"entity", "path", "snapshot"}, "TEST": {"arg", "value"},
             "SAME": {"left", "right"}, "BOTH": {"left", "right"}, "EITHER": {"left", "right"},
             "NOT": {"arg"}}
CHILDREN = {"TEST": ("arg",), "NOT": ("arg",), "SAME": ("left", "right"),
            "BOTH": ("left", "right"), "EITHER": ("left", "right")}
WRITE_KEYS = {"entity", "field", "topic", "value"}
RESTORE_KEYS = {"entity", "field", "topic", "snapshot"}
EVENT_SNAPSHOT = re.compile(r"EVENT:[1-9][0-9]{0,17}")

DDL = """
CREATE TABLE facts(version INTEGER NOT NULL, entity TEXT NOT NULL, field TEXT NOT NULL,
                   topic TEXT, value TEXT, ambiguous INTEGER NOT NULL);
CREATE INDEX facts_by_address ON facts(entity, field, topic, version);
CREATE TABLE entity_types(name TEXT PRIMARY KEY, type TEXT NOT NULL);
CREATE TABLE event_log(seq INTEGER PRIMARY KEY, event_index INTEGER, status TEXT NOT NULL,
                       error TEXT);
CREATE TABLE staged_values(entity TEXT, field TEXT, topic TEXT, value TEXT);
CREATE TABLE staged_types(name TEXT, type TEXT);
CREATE INDEX staged_types_by_name ON staged_types(name);
"""
SQL_LOOKUP = ("SELECT value, ambiguous FROM facts WHERE entity = ? AND field = ? AND topic IS ? "
              "AND version <= ? ORDER BY version DESC LIMIT 1")
SQL_SNAPSHOT = ("SELECT entity, field, topic, value, ambiguous FROM (SELECT *, ROW_NUMBER() OVER "
                "(PARTITION BY entity, field, topic ORDER BY version DESC) AS rn FROM facts "
                "WHERE version <= ?) WHERE rn = 1")
SQL_INITIAL = ("INSERT INTO facts SELECT 0, entity, field, topic, CASE WHEN COUNT(*) > 1 THEN NULL "
               "ELSE MIN(value) END, COUNT(*) > 1 FROM staged_values GROUP BY entity, field, topic")
SQL_PUBLISH = ("INSERT INTO facts SELECT ?, entity, field, topic, MIN(value), 0 FROM staged_values "
               "GROUP BY entity, field, topic")
SQL_TYPE_CONFLICT = ("SELECT 1 FROM staged_types s WHERE EXISTS (SELECT 1 FROM staged_types o "
                     "WHERE o.name = s.name AND o.type <> s.type) OR EXISTS (SELECT 1 FROM "
                     "entity_types t WHERE t.name = s.name AND t.type <> s.type) LIMIT 1")
SQL_WRITE_CONFLICT = ("SELECT 1 FROM staged_values GROUP BY entity, field, topic "
                      "HAVING COUNT(DISTINCT value) > 1 LIMIT 1")
SQL_REGISTER = "INSERT OR IGNORE INTO entity_types SELECT DISTINCT name, type FROM staged_types"


class SQLRefError(Exception):
    """A spec error; `code` is one of the FORMAL_SPEC.md section 6 codes."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _is_name(x):
    if not (isinstance(x, str) and 0 < len(x) <= 80):
        return False
    try:
        x.encode("utf-8")  # a lone surrogate is not a character and cannot be SQLite TEXT
    except UnicodeEncodeError:
        return False
    return True


def _sorted_reads(reads):
    ordered = sorted(reads, key=lambda r: (r[0], r[1], r[2], r[3] is not None, r[3] or ""))
    return [list(r) for r in ordered]


class SQLStore:
    evaluate_all = True  # False: stop evaluating at the first runtime error (see module docstring)

    def __init__(self, initial_records, public_relations=None):
        public = PUBLIC_SCHEMA if public_relations is None else public_relations
        self.relations = {**{n: (ANY, ANY) for n in public if n not in ENUMS}, **RELATIONS}
        self.k = 0  # index of the current (last published) snapshot
        self.db = sqlite3.connect(":memory:", isolation_level=None)  # explicit BEGIN/COMMIT
        self.db.executescript(DDL)
        if not isinstance(initial_records, list):
            raise SQLRefError("MALFORMED")
        for rec in initial_records:
            self._require_shape(rec, WRITE_KEYS)
        for rec in initial_records:
            self._check_write(rec)
        self._check_entity_types(initial_records, [])
        self._stage_values([self._row(rec, rec["value"]) for rec in initial_records])
        self.db.execute(SQL_INITIAL)   # duplicated addresses become one ambiguous row
        self.db.execute(SQL_REGISTER)

    def current_index(self):
        return self.k

    def snapshot_map(self, snapshot):
        index = self._snapshot_index(snapshot)
        return {(e, f, t): AMBIGUOUS_VALUE if ambiguous else value
                for e, f, t, value, ambiguous in self.db.execute(SQL_SNAPSHOT, (index,))}

    # ---------------------------------------------------------------- shared helpers
    @staticmethod
    def _require_shape(op, keys):
        if not isinstance(op, dict) or set(op) != keys:
            raise SQLRefError("MALFORMED")

    @staticmethod
    def _row(op, value):
        return (op["entity"], op["field"], op["topic"], value)

    def _address_ok(self, entity, field, topic):
        if not (_is_name(entity) and isinstance(field, str)
                and (field in ENUMS or field in self.relations)):
            return False
        return _is_name(topic) if field in ENUMS else topic is None

    def _check_write(self, w):
        field, value = w["field"], w["value"]
        if not self._address_ok(w["entity"], field, w["topic"]):
            raise SQLRefError("TYPE_ERROR")
        if not (isinstance(value, str) and value in ENUMS[field] if field in ENUMS
                else _is_name(value)):
            raise SQLRefError("TYPE_ERROR")

    def _type_pairs(self, op, with_value):
        if op["field"] in ENUMS:
            pairs = [(op["entity"], "person"), (op["topic"], "project")]
        else:
            domain, range_ = self.relations[op["field"]]
            pairs = [(op["entity"], domain)] + ([(op["value"], range_)] if with_value else [])
        return [(name, typ) for name, typ in pairs if typ != ANY]

    def _check_entity_types(self, writes, restores):
        # A restored value was typed when it was first written and the registry only grows,
        # so only the restore's own entity/topic can introduce a conflict.
        pairs = [p for op in writes for p in self._type_pairs(op, True)]
        pairs += [p for op in restores for p in self._type_pairs(op, False)]
        self.db.execute("DELETE FROM staged_types")
        self.db.executemany("INSERT INTO staged_types VALUES (?, ?)", pairs)
        if self.db.execute(SQL_TYPE_CONFLICT).fetchone():
            raise SQLRefError("TYPE_CONFLICT")

    def _stage_values(self, rows):
        self.db.execute("DELETE FROM staged_values")
        self.db.executemany("INSERT INTO staged_values VALUES (?, ?, ?, ?)", rows)

    def _snapshot_index(self, snapshot, allow_current=True):
        if snapshot == "ORIGINAL":
            return 0
        if snapshot == "CURRENT" and allow_current:
            return self.k
        if (isinstance(snapshot, str) and EVENT_SNAPSHOT.fullmatch(snapshot)
                and int(snapshot[6:]) <= self.k):
            return int(snapshot[6:])
        raise SQLRefError("UNSUPPORTED_SNAPSHOT")

    def _lookup(self, index, entity, field, topic, reads=None):
        if reads is not None:
            reads.add((index, entity, field, topic))  # recorded before it can fail
        row = self.db.execute(SQL_LOOKUP, (entity, field, topic, index)).fetchone()
        if row is None:
            raise SQLRefError("MISSING")
        if row[1]:
            raise SQLRefError("AMBIGUOUS")
        return row[0]

    # ---------------------------------------------------------------- events
    def apply(self, event):
        self.db.execute("BEGIN")
        try:
            k = self._validate_and_publish(event)
        except SQLRefError as err:
            self.db.execute("ROLLBACK")
            self.db.execute("INSERT INTO event_log(status, error) VALUES ('REJECTED', ?)",
                            (err.code,))
            return {"status": "REJECTED", "error": err.code}
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("INSERT INTO event_log(event_index, status) VALUES (?, 'ACCEPTED')", (k,))
        self.db.execute("COMMIT")
        self.k = k
        return {"status": "ACCEPTED", "event_index": k}

    def _validate_and_publish(self, event):
        if not isinstance(event, dict):
            raise SQLRefError("MALFORMED")
        if set(event) - {"writes", "restores"}:
            raise SQLRefError("UNSUPPORTED_OPERATION")
        writes, restores = event.get("writes", []), event.get("restores", [])
        if not isinstance(writes, list) or not isinstance(restores, list):
            raise SQLRefError("MALFORMED")
        for w in writes:
            self._require_shape(w, WRITE_KEYS)
        for r in restores:
            self._require_shape(r, RESTORE_KEYS)
        for w in writes:
            self._check_write(w)
        for r in restores:
            if not self._address_ok(r["entity"], r["field"], r["topic"]):
                raise SQLRefError("TYPE_ERROR")
        self._check_entity_types(writes, restores)
        rows = [self._row(w, w["value"]) for w in writes]
        for r in restores:
            source = self._snapshot_index(r["snapshot"], allow_current=False)
            rows.append(self._row(r, self._lookup(source, r["entity"], r["field"], r["topic"])))
        self._stage_values(rows)
        if self.db.execute(SQL_WRITE_CONFLICT).fetchone():
            raise SQLRefError("CONFLICTING_WRITES")
        self.db.execute(SQL_PUBLISH, (self.k + 1,))
        self.db.execute(SQL_REGISTER)
        return self.k + 1

    # ---------------------------------------------------------------- DSL
    def execute(self, program, snapshot_default=None):
        default = "CURRENT" if snapshot_default is None else snapshot_default
        reads = set()
        try:
            self._parse(program)
            typ = self._type(program)
            value = self._eval(program, default, reads)
        except SQLRefError as err:
            return {"status": "ERROR", "error": err.code, "read_set": _sorted_reads(reads)}
        return {"status": "OK", "value": value, "type": typ, "read_set": _sorted_reads(reads)}

    def _parse(self, node):
        op = node.get("op") if isinstance(node, dict) else None
        if not isinstance(op, str) or op not in NODE_KEYS:
            raise SQLRefError("MALFORMED")
        keys, need = set(node) - {"op"}, NODE_KEYS[op]
        if keys != need and not (op in ("READ", "FOLLOW") and keys == need - {"snapshot"}):
            raise SQLRefError("MALFORMED")
        if op == "FOLLOW" and not isinstance(node["path"], list):
            raise SQLRefError("MALFORMED")
        for child in CHILDREN.get(op, ()):
            self._parse(node[child])

    def _type(self, node):
        op = node["op"]
        if op == "READ":
            if not self._address_ok(node["entity"], node["field"], node["topic"]):
                raise SQLRefError("TYPE_ERROR")
            return node["field"] if node["field"] in ENUMS else ANY
        if op == "FOLLOW":
            path = node["path"]
            if not 1 <= len(path) <= 4:
                raise SQLRefError("HOP_LIMIT")
            if not _is_name(node["entity"]) or not all(
                    isinstance(r, str) and r in self.relations for r in path):
                raise SQLRefError("TYPE_ERROR")
            for a, b in zip(path, path[1:]):
                range_, domain = self.relations[a][1], self.relations[b][0]
                if ANY not in (range_, domain) and range_ != domain:
                    raise SQLRefError("TYPE_ERROR")
            return ANY
        kinds = [self._type(node[child]) for child in CHILDREN[op]]  # post-order, left to right
        if op == "TEST":
            kind, literal = kinds[0], node["value"]
            ok = kind != "bool" and (isinstance(literal, str) and literal in ENUMS[kind]
                                     if kind in ENUMS else _is_name(literal))
        elif op == "SAME":
            ok = kinds[0] == kinds[1] != "bool"
        else:  # BOTH / EITHER / NOT
            ok = all(kind == "bool" for kind in kinds)
        if not ok:
            raise SQLRefError("TYPE_ERROR")
        return "bool"

    def _eval(self, node, default, reads):
        op = node["op"]
        if op in ("READ", "FOLLOW"):
            index = self._snapshot_index(node.get("snapshot", default))
            if op == "READ":
                return self._lookup(index, node["entity"], node["field"], node["topic"], reads)
            current = node["entity"]
            for relation in node["path"]:
                current = self._lookup(index, current, relation, None, reads)
            return current
        values, first_error = [], None
        for child in CHILDREN[op]:  # strict: every operand, left to right, even after an error
            try:
                values.append(self._eval(node[child], default, reads))
            except SQLRefError as err:
                if not self.evaluate_all:
                    raise
                if first_error is None:
                    first_error = err
        if first_error is not None:
            raise first_error
        if op == "TEST":
            return values[0] == node["value"]
        if op == "NOT":
            return not values[0]
        if op == "SAME":
            return values[0] == values[1]
        return (values[0] and values[1]) if op == "BOTH" else (values[0] or values[1])
