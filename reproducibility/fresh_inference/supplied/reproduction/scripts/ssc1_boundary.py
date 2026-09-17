"""Audited parse-order guard for the real entrypoint, not a security sandbox."""
import builtins
import io
import json
import os
from pathlib import Path
import time
import hashlib

ACTIVE = None

class ReadBoundary:
    def __init__(self, journal):
        self.journal = Path(journal)
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.current = None
        self.sealed = False
        self.sequence = 0

    def event(self, event, **fields):
        self.sequence += 1
        with self.original_io(self.journal, 'a', encoding='utf8') as handle:
            handle.write(json.dumps(dict(sequence=self.sequence, event=event,
                monotonic_ns=time.monotonic_ns(), **fields), allow_nan=False)+'\n')
            handle.flush(); os.fsync(handle.fileno())

    def begin(self, actor, root_id):
        self.current = (actor, root_id); self.sealed = False
        self.event('ROOT_COMPILE_BEGIN', actor=actor, root_id=root_id)

    def seal(self, actor, root_id, seal_path):
        if self.current != (actor,root_id): raise ValueError('Seal root differs')
        if not Path(seal_path).is_file(): raise ValueError('Durable seal missing')
        self.sealed = True
        self.event('ROOT_CONTEXTS_SEALED', actor=actor,root_id=root_id,seal_file=str(seal_path))

    def check(self, file, mode):
        if isinstance(file, int) or 'r' not in mode and '+' not in mode: return
        path = Path(file)
        if path.name not in ('questions.json','questions.jsonl','labels.json','labels.jsonl','root.json','roots.jsonl'): return
        allowed = self.current and path.parent.name == self.current[1] and self.sealed
        self.event('EVALUATION_FILE_OPEN' if allowed else 'FORBIDDEN_EVALUATION_FILE_OPEN',
            path=str(path),actor=self.current[0] if self.current else None,
            root_id=self.current[1] if self.current else None,sealed=self.sealed)
        if not allowed: raise RuntimeError('Evaluation content parsed before matching root seal: '+str(path))

    def __enter__(self):
        global ACTIVE
        if ACTIVE is not None: raise RuntimeError('Concurrent model boundary')
        self.original_open, self.original_io = builtins.open, io.open
        def opened(file, mode='r', *args, **kwargs):
            self.check(file, mode); return self.original_open(file, mode, *args, **kwargs)
        def io_opened(file, mode='r', *args, **kwargs):
            self.check(file, mode); return self.original_io(file, mode, *args, **kwargs)
        builtins.open, io.open, ACTIVE = opened, io_opened, self
        self.event('PROCESS_BOUNDARY_INSTALLED')
        return self

    def __exit__(self, *args):
        global ACTIVE
        self.event('PROCESS_BOUNDARY_REMOVED')
        builtins.open, io.open, ACTIVE = self.original_open, self.original_io, None

def begin(actor, root_id):
    if ACTIVE is not None: ACTIVE.begin(actor, root_id)

def seal(actor, root_id, path):
    if ACTIVE is not None: ACTIVE.seal(actor, root_id, path)

def hash_file(path):
    """Narrow hash-only bypass: bytes never escape to a parser or caller."""
    opener=ACTIVE.original_io if ACTIVE is not None else io.open
    with opener(path,'rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()
