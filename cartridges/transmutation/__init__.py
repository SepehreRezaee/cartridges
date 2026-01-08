"""Utilities for turning synthetic conversations into weight updates.

Implements the "Transmuting Prompts into Weights" style pipeline in two
phases: (1) token-level patch extraction and (2) aggregation into a
single low-rank "thought" update.
"""
