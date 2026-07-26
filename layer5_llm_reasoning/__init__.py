"""
DeepReality — Layer 5: LLM Reasoning Engine

Every upstream pin produces an independent, narrowly scoped signal.
Layer 5 is the adjudication stage: it compresses the full evidence set
into a token-efficient digest, submits it to a large language model
under a forensic reasoning protocol, and returns a calibrated verdict
accompanied by a human-readable justification.

PIN-E1: LLM Reasoning Engine  -> final verdict, confidence, narrative report
"""
