# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo), sesli ajanların konuşma ve araç kullanımıyla görevleri tamamlayıp tamamlamadığını ölçmek için bu kıyaslamayı sürdürür. Sabit İngilizce küme **278 görev** içerir: havayolu **50**, perakende **114**, telekom **114**.

[Kurulum](docs/runtime-setup.md) · [Puanlama](docs/evaluation.md) · [Model karşılaştırması ve sınırlar](docs/model-comparison.md)

## Modeller ve sistemler

| Ajan veya sistem | Mevcut destek |
| --- | --- |
| Gemma 4 12B IT, INT4 | Test edilmiş yerel v2 profili |
| Gemma 4 E4B IT, INT4 | Yerel çalışma zamanı destekliyor; eş koşullu v2 karşılaştırması bekliyor |
| Amazon Nova 2.5 Sonic | Doğrudan ses işleme doğrulandı; Retail pass^1 **1/1** (bir v6 geliştirme denemesi, 420 sn sınırı; [kayıt](docs/results/nova25-v6-retail-development.json)) |
| OpenAI GPT-Live + adı belirtilmiş bir arka uç ajanı | [Deneysel adaptör çevrimdışı test edildi; canlı doğrulama bekliyor](docs/providers.md) |
| OpenAI Realtime | [Deneysel adaptör çevrimdışı test edildi; canlı doğrulama bekliyor](docs/providers.md) |
| Gemini Live | [Deneysel adaptör çevrimdışı test edildi; canlı doğrulama bekliyor](docs/providers.md) |
| ElevenLabs Agents | [Deneysel adaptör çevrimdışı test edildi; canlı doğrulama bekliyor](docs/providers.md) |

Yerel yapı [speech-swift](https://github.com/soniqo/speech-swift) kullanır: tam Whisper Large-v3 FP16, VoxCPM2 ana ağırlıkları FP16 / AudioVAE FP32, Silero VAD ve arayan/dinleme yardımcıları/değerlendirici için Gemma E4B. Nova 2.5 Sonic için isteğe bağlı bir [doğal ses çalıştırıcısı](docs/nova.md) vardır; arayan ve değerlendirici yerel kalır.

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

Aynı görevleri, seed değerlerini, süre sınırlarını, akustik koşulları ve arayan/değerlendirici ayarlarını karşılaştırın. Yerel LLM karşılaştırmalarında konuşma yapısını sabit tutun. Yerel ses işleyen ve yönetilen ajanlar bütün yapıyı belirtmelidir. Bileşen süreleri, uçtan uca sesli yanıt gecikmesinden ayrıdır.

## Kaydedilmiş geliştirme sonuçları

Her satır bir geliştirme denemesidir. Görevler ve zaman ayarları farklıdır; eş koşullu model sıralaması henüz yoktur.

| Ajan | Koşul / görev | Puanlanan / planlanan | Pass^1 | Araç hataları | Gerçek süre |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | v2 · Havayolu 12 | 1/1 | 0/1 | 1/6 | 554.4 s |
| Amazon Nova 2.5 Sonic | v6 · Perakende 0 | 1/1 | **1/1** | **0/5** | **296.3 s** |

Havayolu görevi 12, bir deneme, seed 70, v2; en fazla **600 simülasyon saniyesi / 1.200 gerçek saniye**. Çağrı **312,9 simülasyon saniyesi / 554,4 ölçülen gerçek saniye** sonra normal bitti. Donanım: Apple M5 Pro, 48 GB, paylaşılan iş istasyonu. Bu tek bir geliştirme denemesidir; eş koşullu çok modelli sıralama henüz yoktur. [Veriler ve ölçüm tanımları](docs/results/gemma4-12b-local-v2-airline.json).

Gemma LLM p50 / p95: **7.00 / 11.13 s**; TTS p50 / p95: **1.22 / 1.70 s**.

Nova: bir deneme, seed 70, control akustiği, **420 sn gerçek zaman sınırı**. İstenen iki ürün değiştirildi; **5/5** referans eylem ve son veritabanı durumu eşleşti. Çağrı **259,0 sn ses / 296,3 sn gerçek süre** sonra normal sona erdi. [Kayıt ve sınırlar](docs/results/nova25-v6-retail-development.json).

## Yerel çalıştırma

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Yerel hizmeti başlatmak, arayan/değerlendiriciyi doğrulamak ve görevleri çalıştırmak için [kurulum](docs/runtime-setup.md) ve [v2 çalıştırma](docs/local-v2.md) belgelerini izleyin.

```bash
uv run --frozen --extra dev python scripts/run_local_voice.py \
  --output data/simulations/first-local-run
```

Bu komut sabitlenmiş yerel çalışma zamanını derler, modelleri indirir, arayanı ve değerlendiriciyi kalibre eder ve kontrollü bir ses testi çalıştırır. Alan karşılaştırması puanı üretmez. Yeni bir çıktı dizini kullanın. [Kurulum ve görev çalıştırma](docs/runtime-setup.md#one-command-local-run).

Yürütmeyi `--num-tasks`, `--trials`, `--max-seconds` ve `--wall-timeout` ile sınırlayın. Varsayılan çalışma yereldir. İsteğe bağlı çalıştırıcılar: [Nova](docs/nova.md) · [OpenAI, Gemini ve ElevenLabs](docs/providers.md).

Kontroller: `make test` · `make lint` · `make verify`.

Kaynak: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
