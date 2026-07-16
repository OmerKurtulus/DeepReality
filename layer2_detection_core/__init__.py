"""
DeepReality — Layer 2: Detection Core Pins
Ana tespit motoru — farklı derin öğrenme mimarileri paralel çalışır.

PIN-B1: CLIP ViT-L/14 (Frozen + LN-tune)  → Generalist deepfake/AI detection
PIN-B2: SigLIP2-base-512 (Fine-tuned)     → High-resolution micro-anomaly detection
PIN-B3: Frekans Analizi (DCT/DWT + CNN)   → Frequency domain artifact detection
PIN-B4: Independent Core (3-sınıf)        → AI vs Deepfake vs Real classification
"""