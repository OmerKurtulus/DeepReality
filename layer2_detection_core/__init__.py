"""
DeepReality — Layer 2: Detection Core Pins

Four detectors spanning deliberately different paradigms, so that the
blind spot of any one architecture is covered by another. Their
disagreement is itself informative and is surfaced to the adjudication
stage rather than averaged away.

PIN-B1: CLIP ViT-L/14 (frozen + LN-tune)  -> generalisation specialist
PIN-B2: SigLIP2-base-512 (fine-tuned)     -> high-resolution micro-anomalies
PIN-B3: Frequency Analysis (DCT/DWT+CNN)  -> spectral traces, disjoint domain
PIN-B4: Independent Core (3-class)        -> AI vs Deepfake vs Real taxonomy
"""
