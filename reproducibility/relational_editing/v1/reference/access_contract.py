"""A minimal typed boundary for the pre-query compiler.
This cannot police arbitrary agent code: runtime tracing and artifact hashes are
also required. It makes accidental use of future-query fields detectable.
"""
from dataclasses import dataclass,asdict
import hashlib,json

@dataclass(frozen=True)
class CompileRequest:
    source_prefix: str
    addressed_actor: str
    addressed_project: str|None
    task: str
    new_value: int
    clause_char_span: tuple[int,int]
    model_revision: str
    method_digest: str
    def validate(self):
        a,b=self.clause_char_span
        if not (0<=a<b<=len(self.source_prefix)):raise ValueError('invalid source address')
        if self.addressed_actor not in self.source_prefix[a:b]:raise ValueError('address mismatch')
        if self.new_value not in (0,1):raise ValueError('invalid semantic value')
    def key(self):
        self.validate()
        return hashlib.sha256(json.dumps(asdict(self),sort_keys=True).encode()).hexdigest()
    @classmethod
    def from_dict(cls,d):
        return cls(**d)  # unknown question/answer fields are a TypeError
