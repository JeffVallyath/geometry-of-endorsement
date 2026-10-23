"""Plain-assert tests for sqlref.py against docs/FORMAL_SPEC.md. Run: python test_sqlref.py"""
import random
import sys

from sqlref import SQLRefError, SQLStore

P = "Apollo"


def W(entity, field, topic, value):
    return {"entity": entity, "field": field, "topic": topic, "value": value}


def RS(entity, field, topic, snapshot):
    return {"entity": entity, "field": field, "topic": topic, "snapshot": snapshot}


def READ(entity, field, topic=None, snapshot="CURRENT"):
    return {"op": "READ", "entity": entity, "field": field, "topic": topic, "snapshot": snapshot}


def FOLLOW(entity, path, snapshot="CURRENT"):
    return {"op": "FOLLOW", "entity": entity, "path": list(path), "snapshot": snapshot}


def TEST(arg, value):
    return {"op": "TEST", "arg": arg, "value": value}


def SAME(left, right):
    return {"op": "SAME", "left": left, "right": right}


def BOTH(left, right):
    return {"op": "BOTH", "left": left, "right": right}


def EITHER(left, right):
    return {"op": "EITHER", "left": left, "right": right}


def NOT(arg):
    return {"op": "NOT", "arg": arg}


INIT = [
    W("alice", "stance", P, "favor"), W("bob", "stance", P, "oppose"),
    W("erin", "stance", P, "favor"),
    W("carol", "stance", P, "favor"), W("carol", "stance", P, "oppose"),     # duplicate, differing
    W("frank", "status", P, "pending"), W("frank", "status", P, "pending"),  # duplicate, identical
    W("alice", "status", P, "approved"),
    W("alice", "team", None, "Red"), W("Red", "hub", None, "North"),
    W("North", "region", None, "Europe"),
    W("alice", "organization", None, "Acme"), W("Acme", "founder", None, "erin"),
    W("erin", "team", None, "Blue"), W("Blue", "hub", None, "South"),
    W("bob", "team", None, "Green"), W("bob", "location", None, "Paris"),    # Green has no hub
]
N_ADDRESSES = 15


def store():
    return SQLStore([dict(r) for r in INIT])


def ok(res, value, typ=None):
    assert res["status"] == "OK" and res["value"] == value, res
    assert type(res["value"]) is type(value), res
    if typ is not None:
        assert res["type"] == typ, res
    return res


def err(res, code, reads=None):
    assert res["status"] == "ERROR" and res["error"] == code, res
    assert "value" not in res and "type" not in res, res
    if reads is not None:
        assert res["read_set"] == reads, res
    return res


def accepted(res, k):
    assert res == {"status": "ACCEPTED", "event_index": k}, res


def rejected(res, code):
    assert res == {"status": "REJECTED", "error": code}, res


def raises(code, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except SQLRefError as e:
        assert e.code == code, (e.code, code)
        return
    raise AssertionError("expected SQLRefError(%s)" % code)


# Four Boolean states: stance on Apollo is favor (T), oppose (F), absent (M), duplicated (A).
PERSON = {"T": "alice", "F": "bob", "M": "dave", "A": "carol"}
ERROR = {"M": "MISSING", "A": "AMBIGUOUS"}


def operand(state):
    return READ(PERSON[state], "stance", P)


def boolean(state):
    return TEST(operand(state), "favor")


def lookup(state):
    return [0, PERSON[state], "stance", P]


def check_binary(s, node, x, y, combine):
    res = s.execute(node)
    both = [list(t) for t in sorted({tuple(lookup(x)), tuple(lookup(y))})]
    # Both operands are looked up; with evaluate_all off, nothing right of a failing operand is.
    assert res["read_set"] == (both if s.evaluate_all or x not in ERROR else [lookup(x)]), res
    if x in ERROR or y in ERROR:
        err(res, ERROR[x] if x in ERROR else ERROR[y])  # the first error in operand order
    else:
        ok(res, combine(x, y), "bool")


def test_four_valued_boolean_matrix():
    for evaluate_all in (True, False):  # the error and the value are the same under both readings
        s = store()
        s.evaluate_all = evaluate_all
        for x in "TFMA":
            res = s.execute(NOT(boolean(x)))
            assert res["read_set"] == [lookup(x)], res
            if x in ERROR:
                err(res, ERROR[x])
            else:
                ok(res, x == "F", "bool")
            for y in "TFMA":  # all ordered pairs, so both operand orders
                check_binary(s, BOTH(boolean(x), boolean(y)), x, y, lambda a, b: a == "T" and b == "T")
                check_binary(s, EITHER(boolean(x), boolean(y)), x, y, lambda a, b: a == "T" or b == "T")
                check_binary(s, SAME(operand(x), operand(y)), x, y, lambda a, b: a == b)
    s = store()
    # Hand-written spot checks: an error is never folded into False/True.
    err(s.execute(BOTH(boolean("F"), boolean("M"))), "MISSING")
    err(s.execute(BOTH(boolean("M"), boolean("F"))), "MISSING")
    err(s.execute(EITHER(boolean("T"), boolean("A"))), "AMBIGUOUS")
    err(s.execute(EITHER(boolean("A"), boolean("T"))), "AMBIGUOUS")
    err(s.execute(BOTH(boolean("M"), boolean("A"))), "MISSING", [lookup("A"), lookup("M")])
    err(s.execute(BOTH(boolean("A"), boolean("M"))), "AMBIGUOUS", [lookup("A"), lookup("M")])
    err(s.execute(SAME(operand("A"), operand("M"))), "AMBIGUOUS")
    err(s.execute(NOT(boolean("A"))), "AMBIGUOUS")
    ok(s.execute(BOTH(boolean("T"), boolean("T"))), True, "bool")
    ok(s.execute(BOTH(boolean("T"), boolean("F"))), False, "bool")
    ok(s.execute(EITHER(boolean("F"), boolean("T"))), True, "bool")
    ok(s.execute(EITHER(boolean("F"), boolean("F"))), False, "bool")
    ok(s.execute(SAME(operand("T"), READ("erin", "stance", P))), True, "bool")
    ok(s.execute(SAME(operand("T"), operand("F"))), False, "bool")
    ok(s.execute(NOT(NOT(boolean("T")))), True, "bool")
    ok(s.execute(operand("T")), "favor", "stance")
    ok(s.execute(READ("alice", "status", P)), "approved", "status")
    res = ok(s.execute(BOTH(boolean("F"), EITHER(boolean("T"), NOT(boolean("F"))))), False, "bool")
    assert res["read_set"] == [lookup("T"), lookup("F")], res  # a set, sorted, deduplicated


def test_follow_hops_missing_and_inserted_edge():
    s = store()
    res = ok(s.execute(FOLLOW("alice", ["team", "hub"])), "North", "entity")
    assert res["read_set"] == [[0, "Red", "hub", None], [0, "alice", "team", None]], res
    res = ok(s.execute(FOLLOW("alice", ["team", "hub", "region"])), "Europe", "entity")
    assert len(res["read_set"]) == 3, res
    res = ok(s.execute(FOLLOW("alice", ["organization", "founder", "team", "hub"])), "South", "entity")
    assert res["read_set"] == [[0, "Acme", "founder", None], [0, "Blue", "hub", None],
                               [0, "alice", "organization", None], [0, "erin", "team", None]], res
    ok(s.execute(FOLLOW("bob", ["location"])), "Paris", "entity")
    ok(s.execute(READ("alice", "team")), "Red", "entity")
    ok(s.execute(TEST(FOLLOW("alice", ["organization", "founder"]), "erin")), True, "bool")
    ok(s.execute(TEST(FOLLOW("alice", ["organization", "founder"]), "Erin")), False, "bool")
    # Missing intermediate edge: the failing lookup is in the read set, the next hop is not.
    missing = [[0, "Green", "hub", None], [0, "bob", "team", None]]
    err(s.execute(FOLLOW("bob", ["team", "hub", "region"])), "MISSING", missing)
    err(s.execute(TEST(FOLLOW("bob", ["team", "hub", "region"]), "Asia")), "MISSING", missing)
    err(s.execute(FOLLOW("dave", ["team", "hub"])), "MISSING", [[0, "dave", "team", None]])
    # Insert the edge: CURRENT / EVENT:1 now resolve, ORIGINAL still does not.
    accepted(s.apply({"writes": [W("Green", "hub", None, "West"),
                                 W("West", "region", None, "Asia")]}), 1)
    res = ok(s.execute(FOLLOW("bob", ["team", "hub", "region"])), "Asia", "entity")
    assert res["read_set"] == [[1, "Green", "hub", None], [1, "West", "region", None],
                               [1, "bob", "team", None]], res
    ok(s.execute(FOLLOW("bob", ["team", "hub", "region"], "EVENT:1")), "Asia", "entity")
    err(s.execute(FOLLOW("bob", ["team", "hub", "region"], "ORIGINAL")), "MISSING", missing)
    ok(s.execute(SAME(FOLLOW("bob", ["team", "hub"]), FOLLOW("alice", ["team", "hub"]))), False)
    # Hop limit: k < 1 or k > 4 (a 5-hop path that chains correctly).
    five = ["organization", "founder", "organization", "founder", "organization"]
    err(s.execute(FOLLOW("alice", five)), "HOP_LIMIT", [])
    err(s.execute(FOLLOW("alice", [])), "HOP_LIMIT", [])
    err(s.execute(FOLLOW("alice", five[:4])), "MISSING",   # 4 hops are legal; hop 3 has no edge
        [[1, "Acme", "founder", None], [1, "alice", "organization", None],
         [1, "erin", "organization", None]])
    # An ambiguous intermediate edge stops the walk at that lookup.
    s2 = SQLStore([W("x", "team", None, "T1"), W("x", "team", None, "T2"), W("T1", "hub", None, "H")])
    err(s2.execute(FOLLOW("x", ["team", "hub"])), "AMBIGUOUS", [[0, "x", "team", None]])


def test_atomic_rejection_leaves_state_unchanged():
    s = store()
    before = s.snapshot_map("CURRENT")
    assert len(before) == N_ADDRESSES
    rejected(s.apply({"writes": [W("alice", "stance", P, "oppose"),
                                 W("alice", "team", None, "Yan"),
                                 W("bob", "status", P, "approved"),
                                 W("bob", "status", P, "rejected")]}), "CONFLICTING_WRITES")
    assert s.current_index() == 0 and s.snapshot_map("CURRENT") == before
    ok(s.execute(operand("T")), "favor")
    err(s.execute(READ("bob", "status", P)), "MISSING")
    err(s.execute(operand("T") | {"snapshot": "EVENT:1"}), "UNSUPPORTED_SNAPSHOT")
    # A write and a restore of one address with different resulting values conflict too.
    rejected(s.apply({"writes": [W("alice", "stance", P, "oppose")],
                      "restores": [RS("alice", "stance", P, "ORIGINAL")]}), "CONFLICTING_WRITES")
    # A rejected event registers no entity types, whichever check rejected it: Yan (conflict
    # above), Zed and Wyn would each be a team, yet all three are still untyped afterwards.
    rejected(s.apply({"writes": [W("alice", "team", None, "Zed"),
                                 W("Zed", "stance", P, "favor")]}), "TYPE_CONFLICT")
    rejected(s.apply({"writes": [W("bob", "team", None, "Wyn")],
                      "restores": [RS("dave", "stance", P, "ORIGINAL")]}), "MISSING")
    assert s.current_index() == 0 and s.snapshot_map("CURRENT") == before
    accepted(s.apply({"writes": [W(name, "stance", P, "favor") for name in ("Yan", "Zed", "Wyn")]}),
             1)  # no index was consumed by the rejections
    # Identical duplicate writes, and a write equal to a restore's value, are allowed.
    accepted(s.apply({"writes": [W("bob", "day", P, "Monday"), W("bob", "day", P, "Monday"),
                                 W("alice", "stance", P, "favor")],
                      "restores": [RS("alice", "stance", P, "ORIGINAL")]}), 2)
    ok(s.execute(READ("bob", "day", P)), "Monday", "day")
    # A write equal to the current value changes no content but still publishes an event.
    current = s.snapshot_map("CURRENT")
    accepted(s.apply({"writes": [W("alice", "stance", P, "favor")]}), 3)
    assert s.current_index() == 3 and s.snapshot_map("CURRENT") == current
    assert s.snapshot_map("EVENT:3") == s.snapshot_map("EVENT:2") != s.snapshot_map("EVENT:1")
    assert s.snapshot_map("ORIGINAL") == before
    accepted(s.apply({}), 4)  # an event with no operations still publishes


def test_restore_to_original_and_event():
    s = store()
    accepted(s.apply({"writes": [W("alice", "stance", P, "oppose")]}), 1)
    accepted(s.apply({"writes": [W("alice", "status", P, "rejected")]}), 2)
    accepted(s.apply({"restores": [RS("alice", "stance", P, "ORIGINAL")]}), 3)
    ok(s.execute(operand("T")), "favor", "stance")
    accepted(s.apply({"restores": [RS("alice", "stance", P, "EVENT:1"),
                                   RS("alice", "status", P, "EVENT:1")]}), 4)
    ok(s.execute(operand("T")), "oppose")                       # value published by event 1
    ok(s.execute(READ("alice", "status", P)), "approved")       # unchanged until event 2
    ok(s.execute(READ("alice", "status", P, "EVENT:2")), "rejected")
    ok(s.execute(READ("alice", "stance", P, "EVENT:3")), "favor")
    ok(s.execute(READ("alice", "stance", P, "ORIGINAL")), "favor")
    for snapshot in ("CURRENT", "EVENT:5", "EVENT:6", "EVENT:0", "YESTERDAY", None):
        rejected(s.apply({"restores": [RS("alice", "stance", P, snapshot)]}), "UNSUPPORTED_SNAPSHOT")
    rejected(s.apply({"restores": [RS("dave", "stance", P, "ORIGINAL")]}), "MISSING")
    rejected(s.apply({"restores": [RS("carol", "stance", P, "ORIGINAL")]}), "AMBIGUOUS")
    rejected(s.apply({"restores": [RS("Red", "stance", P, "ORIGINAL")]}), "TYPE_CONFLICT")
    rejected(s.apply({"writes": [W("bob", "stance", P, "favor")],
                      "restores": [RS("dave", "stance", P, "EVENT:4")]}), "MISSING")
    assert s.current_index() == 4
    ok(s.execute(operand("F")), "oppose")  # the write in the rejected event was not applied
    # Restoring a relation edge re-creates the old link.
    accepted(s.apply({"writes": [W("alice", "team", None, "Blue")]}), 5)
    ok(s.execute(FOLLOW("alice", ["team", "hub"])), "South")
    accepted(s.apply({"restores": [RS("alice", "team", None, "EVENT:4")]}), 6)
    ok(s.execute(FOLLOW("alice", ["team", "hub"])), "North")
    ok(s.execute(FOLLOW("alice", ["team", "hub"], "EVENT:5")), "South")


def test_ambiguous_duplicate_initial_addresses():
    s = store()
    err(s.execute(boolean("A")), "AMBIGUOUS", [[0, "carol", "stance", P]])
    err(s.execute(READ("frank", "status", P)), "AMBIGUOUS")  # identical duplicates too
    original = s.snapshot_map("ORIGINAL")
    assert len(original) == N_ADDRESSES
    assert original[("carol", "stance", P)] == "<AMBIGUOUS>"
    assert original[("frank", "status", P)] == "<AMBIGUOUS>"
    assert original[("alice", "stance", P)] == "favor"
    assert original[("alice", "team", None)] == "Red"
    assert ("dave", "stance", P) not in original
    # Writing an ambiguous address resolves it; the original snapshot stays ambiguous.
    accepted(s.apply({"writes": [W("carol", "stance", P, "oppose")]}), 1)
    ok(s.execute(operand("A")), "oppose", "stance")
    ok(s.execute(boolean("A")), False, "bool")
    err(s.execute(READ("carol", "stance", P, "ORIGINAL")), "AMBIGUOUS")
    assert s.snapshot_map("CURRENT")[("carol", "stance", P)] == "oppose"
    assert s.snapshot_map("ORIGINAL") == original
    rejected(s.apply({"restores": [RS("carol", "stance", P, "ORIGINAL")]}), "AMBIGUOUS")
    accepted(s.apply({"restores": [RS("carol", "stance", P, "EVENT:1")]}), 2)


def test_unsupported_snapshot():
    s = store()
    bad = ("EVENT:0", "EVENT:99", "YESTERDAY", "EVENT:1", "event:1", "EVENT:01", "EVENT:-1",
           "EVENT:", "EVENT:1 ", "current", "", None, 0, 1, ["CURRENT"])
    for snapshot in bad:
        err(s.execute(TEST(READ("alice", "stance", P, snapshot), "favor")), "UNSUPPORTED_SNAPSHOT", [])
        err(s.execute(FOLLOW("alice", ["team", "hub"], snapshot)), "UNSUPPORTED_SNAPSHOT", [])
        raises("UNSUPPORTED_SNAPSHOT", s.snapshot_map, snapshot)
    accepted(s.apply({"writes": [W("alice", "stance", P, "oppose")]}), 1)
    ok(s.execute(READ("alice", "stance", P, "EVENT:1")), "oppose")
    err(s.execute(READ("alice", "stance", P, "EVENT:2")), "UNSUPPORTED_SNAPSHOT", [])
    # A runtime error in left-to-right post-order: the first one wins, and a node with an
    # unsupported snapshot looks nothing up (its sibling still does).
    late = TEST(READ("alice", "stance", P, "EVENT:99"), "favor")
    err(s.execute(BOTH(boolean("M"), late)), "MISSING", [[1, "dave", "stance", P]])
    err(s.execute(BOTH(late, boolean("M"))), "UNSUPPORTED_SNAPSHOT", [[1, "dave", "stance", P]])
    err(s.execute(BOTH(boolean("F"), late)), "UNSUPPORTED_SNAPSHOT", [[1, "bob", "stance", P]])
    s.evaluate_all = False
    err(s.execute(BOTH(late, boolean("M"))), "UNSUPPORTED_SNAPSHOT", [])


def test_reads_after_an_error():
    s = store()
    walk = TEST(FOLLOW("alice", ["team", "hub"]), "North")
    walk_reads = [[0, "Red", "hub", None], [0, "alice", "team", None]]
    nested = EITHER(NOT(BOTH(boolean("T"), boolean("A"))), SAME(operand("M"), operand("F")))
    assert SQLStore.evaluate_all is True
    # Default (spec section 5 read literally): every subexpression is evaluated, so operands
    # right of a failing one are still looked up; the first error in post-order is returned.
    err(s.execute(BOTH(boolean("M"), walk)), "MISSING", walk_reads + [lookup("M")])
    err(s.execute(BOTH(walk, boolean("M"))), "MISSING", walk_reads + [lookup("M")])
    err(s.execute(nested), "AMBIGUOUS", [lookup("T"), lookup("F"), lookup("A"), lookup("M")])
    # A failing FOLLOW still stops at its own failing hop (the next entity is unknown).
    err(s.execute(SAME(FOLLOW("bob", ["team", "hub", "region"]), READ("alice", "team"))), "MISSING",
        [[0, "Green", "hub", None], [0, "alice", "team", None], [0, "bob", "team", None]])
    # The other reading: stop at the first error; the read set ends at the failing lookup.
    s.evaluate_all = False
    err(s.execute(BOTH(boolean("M"), walk)), "MISSING", [lookup("M")])
    err(s.execute(BOTH(walk, boolean("M"))), "MISSING", walk_reads + [lookup("M")])
    err(s.execute(nested), "AMBIGUOUS", [lookup("T"), lookup("A")])
    ok(s.execute(BOTH(walk, boolean("T"))), True, "bool")


def test_type_errors():
    s = store()
    favor = boolean("T")
    for program in (
        TEST(operand("T"), "maybe"),                        # literal outside the enum
        TEST(operand("T"), "Favor"),                        # names compare case-sensitively
        TEST(READ("alice", "status", P), "favor"),          # literal of another enum
        TEST(operand("T"), None),
        TEST(READ("alice", "team"), ""),                    # entity literal must be a name
        TEST(READ("alice", "team"), 7),
        FOLLOW("alice", ["team", "region"]),                # range team != domain hub
        FOLLOW("bob", ["location", "hub"]),                 # range city != domain team
        FOLLOW("alice", ["team", "hub", "region", "team"]),
        FOLLOW("alice", ["stance"]),                        # not a relation field
        FOLLOW("alice", ["team", "nope"]),
        SAME(operand("T"), READ("alice", "status", P)),     # stance vs status
        SAME(READ("alice", "status", P), operand("T")),
        SAME(operand("T"), READ("alice", "team")),          # enum vs entity
        SAME(favor, favor),                                 # SAME takes READ/FOLLOW operands
        READ("alice", "team", P),                           # relation field needs topic null
        READ("alice", "stance"),                            # member field needs a topic
        READ("alice", "colour", P),                         # undeclared field
        READ("", "stance", P), READ("a" * 81, "stance", P), READ(5, "stance", P),
        BOTH(operand("T"), favor), EITHER(favor, FOLLOW("alice", ["team"])), NOT(operand("T")),
        TEST(favor, "favor"),                               # TEST of a bool
    ):
        err(s.execute(program), "TYPE_ERROR", [])
    # Static checks run before evaluation: no lookup happens, even left of the bad node.
    err(s.execute(BOTH(boolean("M"), TEST(operand("T"), "maybe"))), "TYPE_ERROR", [])
    err(s.execute(BOTH(boolean("M"), TEST(FOLLOW("alice", []), "Red"))), "HOP_LIMIT", [])
    # Entity-valued operands of different relations are comparable (both entity-valued).
    ok(s.execute(SAME(READ("alice", "team"),
                      FOLLOW("alice", ["organization", "founder", "team"]))), False, "bool")
    ok(s.execute(SAME(READ("alice", "team"), READ("bob", "location"))), False, "bool")
    err(s.execute(READ("a" * 80, "stance", P)), "MISSING", [[0, "a" * 80, "stance", P]])  # 80 is legal
    # Initial state: a type-invalid record or a doubly-typed name rejects the whole state.
    raises("TYPE_ERROR", SQLStore, [W("x", "stance", P, "maybe")])
    raises("TYPE_ERROR", SQLStore, [W("x", "stance", None, "favor")])
    raises("TYPE_ERROR", SQLStore, [W("x", "team", P, "Red")])
    raises("TYPE_ERROR", SQLStore, [W("x", "colour", None, "red")])
    raises("TYPE_ERROR", SQLStore, [W("x", "team", None, "R" * 81)])
    raises("TYPE_ERROR", SQLStore, [W("x", "team", None, "")])
    raises("TYPE_ERROR", SQLStore, [W("", "team", None, "Red")])
    raises("TYPE_ERROR", SQLStore, [W("alice", "stance", P, "favor"), W("x", "day", P, "Sunday")])
    raises("TYPE_CONFLICT", SQLStore, [W("alice", "team", None, "Red"), W("Red", "stance", P, "favor")])
    raises("TYPE_CONFLICT", SQLStore, [W("alice", "stance", P, "favor"), W(P, "team", None, "Red")])
    raises("TYPE_CONFLICT", SQLStore, [W("x", "team", None, "x")])
    raises("MALFORMED", SQLStore, [{"entity": "x", "field": "team", "topic": None}])
    SQLStore([W("x", "team", None, "R" * 80)])
    # Events: type errors and type conflicts reject and publish nothing.
    rejected(s.apply({"writes": [W("alice", "stance", P, "maybe")]}), "TYPE_ERROR")
    rejected(s.apply({"writes": [W("alice", "team", P, "Red")]}), "TYPE_ERROR")
    rejected(s.apply({"restores": [RS("alice", "colour", P, "ORIGINAL")]}), "TYPE_ERROR")
    rejected(s.apply({"writes": [W("Red", "stance", P, "favor")]}), "TYPE_CONFLICT")
    rejected(s.apply({"writes": [W("alice", "hub", None, "North")]}), "TYPE_CONFLICT")
    rejected(s.apply({"writes": [W("bob", "team", None, "North")]}), "TYPE_CONFLICT")
    assert s.current_index() == 0


def test_malformed_and_unsupported_operation():
    s = store()
    before = s.snapshot_map("CURRENT")
    extra = dict(READ("alice", "team"), extra=1)
    for program in (
        {"op": "DELETE", "entity": "alice", "field": "team", "topic": None},
        {"op": "OR", "left": boolean("T"), "right": boolean("F")},
        {"entity": "alice", "field": "team", "topic": None, "snapshot": "CURRENT"},
        extra, {"op": "TEST", "arg": operand("T")}, {"op": "NOT", "arg": "x"},
        {"op": "READ", "entity": "alice", "field": "team", "snapshot": "CURRENT"},
        {"op": "FOLLOW", "entity": "alice", "path": "team", "snapshot": "CURRENT"},
        dict(boolean("T"), snapshot="CURRENT"), "READ", None, [boolean("T")],
        BOTH(boolean("T"), TEST(extra, "Red")),
        BOTH(TEST(operand("T"), "maybe"), {"op": "OR"}),   # MALFORMED anywhere beats TYPE_ERROR
    ):
        err(s.execute(program), "MALFORMED", [])
    # delete is unsupported: any event key other than writes/restores rejects before mutation.
    address = {"entity": "alice", "field": "stance", "topic": P}
    rejected(s.apply({"deletes": [address]}), "UNSUPPORTED_OPERATION")
    rejected(s.apply({"delete": address}), "UNSUPPORTED_OPERATION")
    rejected(s.apply({"writes": [W("alice", "stance", P, "oppose")], "deletes": [address]}),
             "UNSUPPORTED_OPERATION")
    rejected(s.apply({"writes": [address]}), "MALFORMED")
    rejected(s.apply({"writes": [dict(W("alice", "stance", P, "oppose"), op="delete")]}), "MALFORMED")
    rejected(s.apply({"writes": W("alice", "stance", P, "oppose")}), "MALFORMED")
    rejected(s.apply({"restores": [W("alice", "stance", P, "oppose")]}), "MALFORMED")
    rejected(s.apply(["writes"]), "MALFORMED")
    assert s.current_index() == 0 and s.snapshot_map("CURRENT") == before
    ok(s.execute(operand("T")), "favor")


def test_snapshot_default_and_read_set_index():
    s = store()
    accepted(s.apply({"writes": [W("alice", "stance", P, "oppose")]}), 1)
    accepted(s.apply({"writes": [W("bob", "stance", P, "favor")]}), 2)
    bare = TEST({"op": "READ", "entity": "alice", "field": "stance", "topic": P}, "favor")
    assert ok(s.execute(bare), False)["read_set"] == [[2, "alice", "stance", P]]
    assert ok(s.execute(bare, snapshot_default="ORIGINAL"), True)["read_set"] == [[0, "alice", "stance", P]]
    assert ok(s.execute(bare, "EVENT:1"), False)["read_set"] == [[1, "alice", "stance", P]]
    err(s.execute(bare, "EVENT:3"), "UNSUPPORTED_SNAPSHOT", [])
    ok(s.execute(boolean("T"), snapshot_default="ORIGINAL"), False)  # explicit CURRENT wins
    walk = {"op": "FOLLOW", "entity": "alice", "path": ["team"]}
    assert ok(s.execute(walk, "ORIGINAL"), "Red", "entity")["read_set"] == [[0, "alice", "team", None]]
    # One program reading three snapshots: indices are resolved integers, sorted.
    res = ok(s.execute(SAME(READ("alice", "stance", P, "ORIGINAL"),
                            READ("bob", "stance", P, "CURRENT"))), True, "bool")
    assert res["read_set"] == [[0, "alice", "stance", P], [2, "bob", "stance", P]], res
    res = ok(s.execute(SAME(READ("bob", "stance", P, "EVENT:2"),
                            READ("bob", "stance", P, "EVENT:1"))), False, "bool")
    assert res["read_set"] == [[1, "bob", "stance", P], [2, "bob", "stance", P]], res


def test_public_bridge_relations_are_configurable():
    records = [W("alice", "team", None, "Red"), W("Red", "partner", None, "North"),
               W("North", "region", None, "Europe"), W("Europe", "partner", None, "alice")]
    raises("TYPE_ERROR", SQLStore, records)  # undeclared unless listed as public
    s = SQLStore(records, public_relations=["partner"])
    ok(s.execute(FOLLOW("alice", ["team", "partner", "region"])), "Europe", "entity")
    ok(s.execute(FOLLOW("alice", ["team", "partner", "region", "partner"])), "alice", "entity")
    ok(s.execute(READ("Red", "partner")), "North", "entity")
    err(s.execute(FOLLOW("alice", ["team", "region"])), "TYPE_ERROR", [])


def test_random_event_sequences_match_python_model():
    rng = random.Random(20261001)
    enums = {"stance": ["favor", "oppose"], "status": ["pending", "approved", "rejected", "deferred"]}
    addresses = [(p, f, t) for p in ("p1", "p2", "p3") for f in enums for t in ("X", "Y")]
    amb = "<AMBIGUOUS>"
    for _ in range(25):
        initial, snaps = [], [{}]
        for a in addresses:
            copies = rng.choice([0, 1, 1, 2])
            initial += [W(*a, rng.choice(enums[a[1]])) for _ in range(copies)]
            if copies:
                snaps[0][a] = amb if copies > 1 else initial[-1]["value"]
        rng.shuffle(initial)
        s = SQLStore(initial)
        for _ in range(40):
            writes = [W(*a, rng.choice(enums[a[1]])) for a in rng.sample(addresses, rng.randint(0, 3))]
            sources = ["ORIGINAL", "ORIGINAL", "CURRENT"] + ["EVENT:%d" % j for j in range(len(snaps) + 2)]
            restores = [RS(*a, rng.choice(sources)) for a in rng.sample(addresses, rng.randint(0, 2))]
            staged, error = [(w["entity"], w["field"], w["topic"], w["value"]) for w in writes], None
            for r in restores:  # plain-Python model of spec section 3
                a, name = (r["entity"], r["field"], r["topic"]), r["snapshot"]
                j = 0 if name == "ORIGINAL" else -1 if name in ("CURRENT", "EVENT:0") else int(name[6:])
                if not 0 <= j < len(snaps):
                    error = "UNSUPPORTED_SNAPSHOT"
                elif a not in snaps[j]:
                    error = "MISSING"
                elif snaps[j][a] == amb:
                    error = "AMBIGUOUS"
                if error:
                    break
                staged.append(a + (snaps[j][a],))
            new = {}
            for *a, value in staged:
                if error is None and new.setdefault(tuple(a), value) != value:
                    error = "CONFLICTING_WRITES"
            result = s.apply({"writes": writes, "restores": restores})
            if error:
                rejected(result, error)
            else:
                snaps.append({**snaps[-1], **new})
                accepted(result, len(snaps) - 1)
            assert s.current_index() == len(snaps) - 1
            assert s.snapshot_map("CURRENT") == snaps[-1]
        for i, snap in enumerate(snaps):
            name = "ORIGINAL" if i == 0 else "EVENT:%d" % i
            assert s.snapshot_map(name) == snap
            for a in addresses:
                res = s.execute(READ(*a, name))
                assert res["read_set"] == [[i, *a]], res
                if a not in snap:
                    err(res, "MISSING")
                elif snap[a] == amb:
                    err(res, "AMBIGUOUS")
                else:
                    ok(res, snap[a], a[1])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("ok   " + name)
    print("PASSED %d/%d tests (python %s)" % (len(tests), len(tests), sys.version.split()[0]))
