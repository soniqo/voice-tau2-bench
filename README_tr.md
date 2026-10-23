# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo), Apple Silicon üzerinde **yerel sesli ajan performansını** ölçmek için bu bağımsız τ-Voice kıyaslamasını sürdürüyor. Ajanı, simüle edilen arayanı, konuşma modellerini ve anlamsal değerlendiriciyi [speech-swift](https://github.com/soniqo/speech-swift) ile çalıştırıyoruz.

## Belgeler

- [Kurulum ve model dosyaları](docs/runtime-setup.md)
- [Değerlendirme ve puanlama](docs/evaluation.md)
- [Uygulama](docs/local-voice-extraction.md)
- [Model çalıştırma kaydı](docs/results/gemma4-e4b-local-600.json)
- [Gemma 4 12B çalıştırması ve doğrulanan parametreler](docs/validation/gemma4-12b-voice-e2e.md)

## Neyi ölçüyoruz

Sesli bir ajanın konuşma ve araç kullanımı yoluyla müşteri hizmetleri görevlerini tamamlayıp tamamlamadığını ölçüyoruz. Görev ödülleri, gerekli son veritabanı durumunu ve iletilen bilgileri denetler.

Sabit İngilizce `voice-v1` çerçevesi **278 görev** içerir: havayolu **50**, perakende **114**, telekom **114**. Görevler, politikalar, araçlar, veritabanları ve ödül ölçütleri özgün içerik bağlarını korur.

## Test edilen modeller ve sonuçlar

| Rol | Model | Hassasiyet |
| --- | --- | --- |
| Ajan | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| Arayan, dinleme yardımcıları, anlamsal hakem | Gemma 4 E4B IT | INT4 |
| Ajan ve arayan konuşması | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Konuşma tanıma | Whisper Large-v3 (full, CoreML) | FP16 |
| Konuşma algılama | Silero VAD (CoreML) | Native |

Geliştirme örneklemi: her alanda sabit sıradaki ilk görev, toplam **3 görev**, **1 deneme**, tohum **42**, **600 saniye görüşme sınırı**, temiz `control` telefon sesi. Donanım: **Apple M5 Pro, 48 GB birleşik bellek**. Başarı, yukarıdaki tüm model yığınını ölçer.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| Ajan modeli | Havayolu | Perakende | Telekom | Toplam pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

Bu küçük ve belirlenimci örneklem yerel arayan ve hakem kullanır. Tüm 278 görevin puanını veya özgün sıralamayı temsil etmez. Zaman aşımı sıfır puandır; çıkarım istisnaları puanlanmaz.

12B örneklemi tamamlanmadı: telekom denemesi yerel çıkarım hatası verdi ve puanlanmadı. Kontrollü 12B ses testi: doğru veritabanı yazımı için **1.0**, yanlış yazımın yeniden değerlendirilmesi için **0.0**. [Doğrulanan parametreler](docs/validation/gemma4-12b-voice-e2e.md). Arayanın girdisi sabittir; bu test kıyaslama puanına katılmaz. Alan denemeleri üretken arayanı kullanır ve bağlantılı raporlar her çalışma zamanı sürümünü kaydeder.

## Yerel çalıştırma

Python **3.12/3.13**, Apple Silicon ve **macOS 15+** kullanın. Görevlerden önce [yerel model servisini](docs/runtime-setup.md#serve) başlatın.

```bash
git clone https://github.com/soniqo/voice-tau2-bench.git
cd voice-tau2-bench
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

```bash
uv run --frozen tau-voice batch \
  --domain all --num-tasks 1 --trials 1 --seed 42 \
  --max-seconds 600 --complexity control \
  --base-url http://127.0.0.1:8787 \
  --output data/simulations/local-baseline
```

Her görev araç/konuşma akışını, ödül bileşenlerini, iki WAV dosyasını, süreleri, ayarları ve kaynak/model parmak izlerini saklar. Aynı grubu `--resume` ile sürdürün. Tüm 278 görev için `--num-tasks` seçeneğini kaldırın.

Kontroller: `make test` · `make lint` · `make verify`.

Kaynak: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice makalesi](https://arxiv.org/abs/2603.13686) · [MIT lisansı](LICENSE).
