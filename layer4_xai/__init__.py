"""
DeepReality — Layer 4: Açıklanabilirlik Pinleri (XAI Pins)
Katman 2 modellerinin kararlarını görselleştirir ve manipülasyon
bölgelerini lokalize eder. Kara kutu modelleri şeffaflaştırır.

PIN-D1: Grad-CAM Heatmap        → Her modelin karar odağını ısı haritası olarak görselleştirir
PIN-D2: Anomaly Localization    → ELA + Grad-CAM anomali haritalarını birleştirip manipülasyon bölgelerini işaretler
"""
