"""Frozen-contract Claim 2 failure diagnostics.

Llama diagnostics in this package are exploratory and post-outcome. The
pre-existing frozen M2/M3 result is never rewritten by this package.
"""

from .input_audit import MODES, load_contract, run_audit_and_freeze

__all__ = ["MODES", "load_contract", "run_audit_and_freeze"]
