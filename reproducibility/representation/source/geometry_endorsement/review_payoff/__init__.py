"""Existing-human-review payoff study.

The package is deliberately split into audit, frozen-score analysis, fresh
verdict scoring, causal intervention, and type-transfer phases so that no GPU
phase can silently alter the study contract written by the CPU audit phase.
"""

from .input_audit import run_input_audit

__all__ = ["run_input_audit"]
