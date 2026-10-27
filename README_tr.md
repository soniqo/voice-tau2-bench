# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo), sesli ajanların konuşma ve araç kullanımıyla görevleri tamamlayıp tamamlamadığını ölçmek için bu kıyaslamayı sürdürür. Sabit İngilizce küme **278 görev** içerir: havayolu **50**, perakende **114**, telekom **114**.

[Kurulum](docs/runtime-setup.md) · [Puanlama](docs/evaluation.md) · [Model karşılaştırması ve sınırlar](docs/model-comparison.md)

## Modeller ve sistemler

| Ajan veya sistem | Mevcut destek |
| --- | --- |
| Gemma 4 12B IT, INT4 | Test edilmiş yerel v2 profili |
| Gemma 4 E4B IT, INT4 | Yerel çalışma zamanı destekliyor; eş koşullu v2 karşılaştırması bekliyor |
| OpenAI GPT-Live + adı belirtilmiş bir arka uç ajanı | Adaptör gerekiyor |
| OpenAI Realtime | Adaptör gerekiyor |
| Gemini Live | Adaptör gerekiyor |
| ElevenLabs Agents | Adaptör gerekiyor |

Yerel yapı [speech-swift](https://github.com/soniqo/speech-swift) kullanır: tam Whisper Large-v3 FP16, VoxCPM2 ana ağırlıkları FP16 / AudioVAE FP32, Silero VAD ve arayan/dinleme yardımcıları/değerlendirici için Gemma E4B. Bulut sistemleri burada henüz entegre edilmedi veya kıyaslanmadı.

## Karşılaştırma metrikleri

| Metrik | Neyi ölçer |
| --- | --- |
| Görev başarısı, pass^1 | Gerekli son veritabanı durumu ve iletişim |
| Kapsam ve hatalar | Puanlanan/planlanan denemeler, normal bitişler ve puanlanmayan çıkarım hataları |
| Gecikme, p50 / p95 | Ajan LLM ve ASR yürütmesi; TTS ilk ses süresi |
| Sesli yanıt gecikmesi | Arayanın konuşmasının bitişinden ajanın ilk sesine kadar; standart özet bekliyor |
| Araç hata oranı | Başarısız veya reddedilmiş ajan çağrıları / denenen çağrılar |
| Tamamlanma süresi | Simülasyon çağrı süresi ve ölçülen gerçek süre ayrı ayrı |
| Tepe bellek | Ölçüm kapsamıyla birlikte yerel sürecin fiziksel bellek kullanımı |
| API maliyeti | Deneme ve başarılı deneme başına USD; başarısız denemeler dahil |

Aynı görevleri, seed değerlerini, süre sınırlarını, akustik koşulları ve arayan/değerlendirici ayarlarını karşılaştırın. Yerel LLM karşılaştırmalarında konuşma yapısını sabit tutun. Yerel ses işleyen ve yönetilen ajanlar bütün yapıyı belirtmelidir. Bileşen süreleri, uçtan uca sesli yanıt gecikmesinden ayrıdır.

## Kaydedilmiş yerel tanılama

| Ajan | Puanlanan / planlanan | Pass^1 | Araç hataları | LLM p50 / p95 | TTS ilk ses p50 / p95 | API maliyeti / deneme |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

Havayolu görevi 12, bir deneme, seed 70, v2; en fazla **600 simülasyon saniyesi / 1.200 gerçek saniye**. Çağrı **312,9 simülasyon saniyesi / 554,4 ölçülen gerçek saniye** sonra normal bitti. Donanım: Apple M5 Pro, 48 GB, paylaşılan iş istasyonu. Bu tek bir geliştirme denemesidir; eş koşullu çok modelli sıralama henüz yoktur. [Veriler ve ölçüm tanımları](docs/results/gemma4-12b-local-v2-airline.json).

## Yerel çalıştırma

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Yerel hizmeti başlatmak, arayan/değerlendiriciyi doğrulamak ve görevleri çalıştırmak için [kurulum](docs/runtime-setup.md) ve [v2 çalıştırma](docs/local-v2.md) belgelerini izleyin.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

Çalıştırmayı `--num-tasks`, `--trials`, `--max-seconds` ve `--wall-timeout` ile sınırlayın. Bulut adaptörleri ve USD harcama sınırı **uygulanmış değildir**. Mevcut çıkarım yereldir ve bulut API anahtarı gerektirmez.

Kontroller: `make test` · `make lint` · `make verify`.

Kaynak: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
