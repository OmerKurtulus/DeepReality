"""
DeepReality — Layer 4: Explainability Pins (XAI)

Renders the decisions of the Layer 2 detectors as spatial evidence and
localises candidate manipulation regions, converting opaque model
outputs into findings an analyst can inspect and contest.

PIN-D1: Grad-CAM Heatmap      -> visualises each model's decision focus
PIN-D2: Anomaly Localisation  -> fuses ELA and Grad-CAM evidence to mark
                                 candidate manipulation regions
"""
