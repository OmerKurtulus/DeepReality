"""
DeepReality — Layer 1: Preprocessing Pins

Lightweight but high-value signals extracted before any model is
invoked. These pins read documentary and physical evidence — metadata,
cryptographic provenance, compression behaviour — which is independent
of, and in the evidence hierarchy superior to, statistical inference.

PIN-A1: EXIF/Metadata Analysis     -> generator signatures, capture telemetry
PIN-A2: C2PA Provenance Analysis   -> signed provenance, source verification
PIN-A3: Error Level Analysis       -> localised manipulation, compression traces
PIN-A4: Face Detection & Cropping  -> detection, alignment, normalisation
"""
