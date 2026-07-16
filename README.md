# DeepReality

**Çok Katmanlı PIN Mimarisi ile Deepfake ve Yapay Zekâ Üretimi Görsel Tespiti**

DeepReality, bir görselin yapay zekâ tarafından üretilip üretilmediğini veya manipüle edilip edilmediğini tespit etmek için birden fazla bağımsız analiz yaklaşımını tek bir platformda birleştiren çok katmanlı bir tespit sistemidir. Proje, mevcut tespit araçlarının en temel zayıflığı olan **genelleme açığını** — tek bir model mimarisine dayanan sistemlerin yalnızca eğitildikleri sahtecilik türlerini tanıyabilmesini — birbirini tamamlayan bağımsız analiz katmanlarının füzyonu ile aşmayı hedefler.

## PIN Architecture

Sistemin temel tasarım fikri, bir CPU'daki pinlerden esinlenen **PIN Architecture** yaklaşımıdır: işlemcide her pin farklı bir işlevden sorumluyken hepsi birlikte tek bir bütün oluşturur. DeepReality'de de her analiz modülü bağımsız çalışan bir "pin"dir; her pin görseli kendi perspektifinden inceler ve 0.0 (gerçek) ile 1.0 (sahte) arasında standart bir skor üretir.

Bu modüler yapının iki önemli sonucu vardır:

1. **Paralellik** — Birbirine bağımlı olmayan tüm pinler, DAG (bağımlılık grafiği) tabanlı orkestratör (`core/pipeline.py`) üzerinde eşzamanlı çalışır. Yalnızca başka bir pinin çıktısına ihtiyaç duyan pinler (ör. açıklanabilirlik katmanı) bağımlılıkları tamamlandığı anda devreye girer.
2. **Genişletilebilirlik** — Yeni bir pin eklemek veya mevcut bir pini güncellemek sistemin geri kalanını etkilemez; sürekli evrilen üretim tekniklerine hızlı adaptasyon sağlanır.

### Katmanlar ve Pinler

| Katman | Pin | İşlev | Teknoloji |
|---|---|---|---|
| **1 — Ön İşlem** | PIN-A1 | EXIF/Metadata analizi, AI aracı imza tespiti | Pillow, binary parsing |
| | PIN-A2 | C2PA Content Credentials köken doğrulama | c2pa-python |
| | PIN-A3 | ELA (Error Level Analysis) manipülasyon tespiti | OpenCV, PIL |
| | PIN-A4 | Yüz tespiti, hizalama ve normalizasyon | MediaPipe |
| **2 — Tespit Çekirdeği** | PIN-B1 | CLIP ViT-L/14 — frozen backbone + LayerNorm tuning | PyTorch, Transformers |
| | PIN-B2 | SigLIP2-base-512 — yüksek çözünürlüklü mikro-anomali tespiti | PyTorch, Transformers |
| | PIN-B3 | Frekans analizi — DCT/DWT dönüşümü + özel CNN | SciPy, PyWavelets |
| | PIN-B4 | Independent Core — 3 sınıflı ayrım (AI / Deepfake / Real) | SigLIP2 |
| **4 — Açıklanabilirlik (XAI)** | PIN-D1 | Grad-CAM ısı haritaları — modellerin karar odağı | PyTorch (hook tabanlı) |
| | PIN-D2 | Anomali lokalizasyonu — ELA + Grad-CAM kanıt füzyonu | OpenCV |

> Katman 3 (video temporal analiz), Katman 5 (LLM muhakeme motoru) ve Katman 6 (ensemble karar motoru) yol haritasında olup geliştirme sırası aşağıdaki "Yol Haritası" bölümünde verilmiştir.

### Tespit Çekirdeği Tasarım Gerekçesi

Katman 2'deki dört model bilinçli olarak **farklı paradigmalardan** seçilmiştir:

- **PIN-B1 (CLIP, frozen + LN-tune):** Backbone dondurulup yalnızca LayerNorm parametreleri eğitilerek (~365K / 427M parametre) modelin genel görsel temsili korunur — eğitimde hiç görülmemiş üretim tekniklerine karşı maksimum genelleme.
- **PIN-B2 (SigLIP2, full fine-tune):** 512×512 girişle CLIP'in kaçırabileceği mikro-anomalileri yakalayan hassasiyet odaklı model.
- **PIN-B3 (Frekans CNN):** Görseli DCT/DWT ile frekans domain'ine çevirir; GAN upsampling ve diffusion izleri gibi spatial modellerin ilkesel olarak göremeyeceği artefaktları yakalar.
- **PIN-B4 (Independent Core):** Sistemdeki tek 3 sınıflı model — görselin sıfırdan AI üretimi mi yoksa gerçek içeriğin manipülasyonu (deepfake) mı olduğunu ayırt eder.

Spatial + frekans + köken doğrulama sinyallerinin tek pipeline'da birleşmesi, hem GAN hem diffusion tabanlı üretimlerin tespitini mümkün kılar.

### Açıklanabilirlik Katmanı (XAI)

Ticari tespit araçlarının çoğu kara kutudur; kullanıcıya *neden* sahte kararı verildiği söylenmez. DeepReality'de:

- **PIN-D1**, Grad-CAM (Selvaraju et al., 2017) tekniğini ViT tabanlı modellere uyarlayarak her modelin karar verirken görüntünün hangi bölgelerine odaklandığını ısı haritası olarak üretir. Harici XAI paketi kullanılmaz; gradyanlar `torch.autograd.grad` ile yalnızca hedef katmana kadar hesaplanarak yüksek performans elde edilir. Modeller arası uzamsal tutarlılık IoU metriğiyle raporlanır.
- **PIN-D2**, iki bağımsız kanıt kaynağını — sıkıştırma fiziğine dayanan ELA bölgeleri ile model kararına dayanan Grad-CAM odakları — tek haritada birleştirir. İki yöntemin aynı bölgeyi işaretlemesi "füzyon-doğrulamalı" güçlü kanıt olarak sınıflandırılır ve manipülasyon bölgeleri renk kodlu olarak orijinal görsel üzerinde işaretlenir.

## Kurulum

```bash
git clone <repo-url>
cd DeepReality

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r Requirements.txt
```

**Model ağırlıkları** boyutları nedeniyle (toplam ~3.4 GB) bu repoda yer almaz. İndirme bağlantıları ve kurulum adımları için [models/README.md](models/README.md) dosyasına bakınız.

## Kullanım

```bash
# 1. Analiz edilecek görselleri input/ klasörüne koyun
#    (desteklenen formatlar: jpg, png, webp, bmp, tiff, gif, heic/heif)

# 2. Sistemi çalıştırın
python3 main.py
```

Sistem her görsel için tüm pinleri paralel çalıştırır, terminale pin bazlı canlı ilerleme ve katman özetleri yazar. Cihaz seçimi otomatiktir (CUDA → Apple MPS → CPU).

### Çıktılar

Her görsel için `outputs/` klasörüne yazılır:

| Çıktı | İçerik |
|---|---|
| `{görsel}_PIN-XX.json` | Her pin için standart sonuç (skor, verdict, detaylı bulgular) — 10 adet |
| `{görsel}_ELA_heatmap.png` | ELA ısı haritası |
| `{görsel}_face_N.png` | Tespit edilen ve normalize edilen yüz kırpmaları |
| `{görsel}_XAI_D1_{model}.png` | Model başına Grad-CAM overlay (clip, siglip, freq, combined) |
| `{görsel}_XAI_D2_anomaly.png` | İşaretli anomali haritası (ELA + CAM füzyonu) |

Tüm pinler ortak bir JSON şeması kullanır:

```json
{
    "pin_id": "PIN-B1",
    "layer": 2,
    "input_hash": "sha256...",
    "score": 0.9676,
    "verdict": "high_risk",
    "results": { "...pin'e özel bulgular..." },
    "details": "Türkçe doğal dil açıklama"
}
```

## Proje Yapısı

```
DeepReality/
├── main.py                      # Ana çalıştırıcı — paralel pipeline
├── config/settings.py           # Tüm eşikler, ağırlıklar ve yollar (tek merkez)
├── core/
│   ├── base_pin.py              # Standart pin sözleşmesi (BasePin)
│   └── pipeline.py              # DAG tabanlı paralel orkestratör
├── layer1_preprocessing/        # PIN-A1 ... PIN-A4
├── layer2_detection_core/       # PIN-B1 ... PIN-B4
├── layer3_video_temporal/       # (yol haritasında)
├── layer4_xai/                  # PIN-D1, PIN-D2
├── layer5_llm_reasoning/        # (yol haritasında)
├── layer6_ensemble/             # (yol haritasında)
├── models/                      # Model ağırlıkları (bkz. models/README.md)
├── input/                       # Analiz edilecek görseller
└── outputs/                     # Analiz sonuçları
```

Ayrıntılı mimari dokümantasyon için: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## Yol Haritası

- [x] **Katman 1** — Ön işlem pinleri (metadata, C2PA, ELA, yüz tespiti)
- [x] **Katman 2** — Çok paradigmalı tespit çekirdeği (4 model)
- [x] **Paralel orkestratör** — DAG tabanlı eşzamanlı pin çalıştırma
- [x] **Katman 4** — XAI: Grad-CAM ve anomali lokalizasyonu
- [ ] **Katman 5** — LLM muhakeme motoru: tüm pin çıktılarını sentezleyen Türkçe doğal dil raporu
- [ ] **Katman 6** — Ensemble karar motoru: XGBoost meta-learner ile ağırlıklı füzyon
- [ ] **Katman 3** — Video temporal pinleri (frame tutarlılığı, dudak-ses senkronu, biyolojik sinyal)
- [ ] Web arayüzü, REST API ve tarayıcı eklentisi
- [ ] ONNX dönüşümü ve kuantalama ile üretim optimizasyonu

## Ekip

- **Ömer Faruk Kurtuluş** — ML / backend geliştirme, model eğitimi
- **Burcu Bayrak** — Frontend geliştirme, test süreçleri

## Kaynakça

- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
- Selvaraju, R.R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
- Rössler, A. et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019.
- C2PA — Coalition for Content Provenance and Authenticity. *Technical Specification.* https://c2pa.org
