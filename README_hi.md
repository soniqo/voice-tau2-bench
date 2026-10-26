# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) Apple Silicon पर **स्थानीय वॉइस एजेंट के प्रदर्शन** को मापने के लिए इस स्वतंत्र τ-Voice बेंचमार्क का रखरखाव करता है। हम एजेंट, सिम्युलेटेड कॉलर, स्पीच मॉडल और अर्थ-आधारित मूल्यांकन मॉडल को [speech-swift](https://github.com/soniqo/speech-swift) से चलाते हैं।

## दस्तावेज़

- [सेटअप और मॉडल फ़ाइलें](docs/runtime-setup.md)
- [मूल्यांकन और स्कोरिंग](docs/evaluation.md)
- [कार्यान्वयन](docs/local-voice-extraction.md)
- [मॉडल रन का रिकॉर्ड](docs/results/gemma4-e4b-local-600.json)
- [Gemma 4 12B रन और सत्यापित पैरामीटर](docs/validation/gemma4-12b-voice-e2e.md)

## हम क्या मापते हैं

हम मापते हैं कि वॉइस एजेंट बातचीत और टूल के उपयोग से ग्राहक सेवा के कार्य पूरे कर पाता है या नहीं। कार्य के पुरस्कार आवश्यक अंतिम डेटाबेस स्थिति और दी गई जानकारी की जाँच करते हैं।

स्थिर अंग्रेज़ी `voice-v1` में **278 कार्य** हैं: एयरलाइन **50**, रिटेल **114**, टेलीकॉम **114**। कार्य, नीतियाँ, टूल, डेटाबेस और पुरस्कार मानदंड मूल पिन बरकरार रखते हैं।

## परीक्षण किए गए मॉडल और परिणाम

| भूमिका | मॉडल | प्रेसिजन |
| --- | --- | --- |
| एजेंट | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| कॉलर, सुनने के निर्णय, अर्थ संबंधी जज | Gemma 4 E4B IT | INT4 |
| एजेंट और कॉलर की आवाज़ | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| वाक् पहचान | Whisper Large-v3 (full, CoreML) | FP16 |
| वाणी का पता लगाना | Silero VAD (CoreML) | Native |

विकास नमूना: हर डोमेन के स्थिर क्रम का पहला कार्य, कुल **3 कार्य**, **1 प्रयास**, सीड **42**, बातचीत सीमा **600 सेकंड**, साफ़ `control` टेलीफोनी। हार्डवेयर: **Apple M5 Pro, 48 GB यूनिफाइड मेमोरी**। सफलता ऊपर दिए पूरे स्टैक को मापती है।

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| एजेंट मॉडल | एयरलाइन | रिटेल | टेलीकॉम | कुल pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

इस छोटे निश्चित नमूने में स्थानीय कॉलर और जज हैं। यह सभी 278 कार्यों का स्कोर या मूल लीडरबोर्ड परिणाम नहीं है। टाइमआउट का स्कोर शून्य है; अनुमान अपवाद बिना स्कोर रहते हैं।

12B नमूना अधूरा है: टेलीकॉम कार्य में नेटिव इन्फरेंस त्रुटि आई और उसे स्कोर नहीं दिया गया। नियंत्रित 12B वॉइस जाँच में सही डेटाबेस लेखन को **1.0** और गलत लेखन के दोबारा मूल्यांकन को **0.0** मिला। [सत्यापित पैरामीटर](docs/validation/gemma4-12b-voice-e2e.md)। कॉलर का इनपुट तय है; यह जाँच बेंचमार्क स्कोर में शामिल नहीं है। डोमेन रन जनरेटिव कॉलर का उपयोग करते हैं और लिंक की गई रिपोर्ट हर रनटाइम संशोधन को दर्ज करती हैं।

लोकल v2 अब वास्तविक फ़ोन टूल के अवलोकनों के आधार पर टेलीकॉम डिवाइस के प्रश्नों का जवाब देता है। निदान और होल्डआउट पायलट की प्रगति ऐतिहासिक v1 स्कोर से अलग दर्ज की जाती है। [वर्तमान सत्यापन और पैरामीटर](docs/validation/grounded-caller.md)।

`soniqo-local-gemma12b-v2` · `--max-seconds 600 --wall-timeout 1200`

## स्थानीय रूप से चलाएँ

Python **3.12/3.13**, Apple Silicon और **macOS 15+** चाहिए। कार्य चलाने से पहले [स्थानीय मॉडल सेवा](docs/runtime-setup.md#serve) शुरू करें।

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

हर कार्य में टूल/बातचीत क्रम, पुरस्कार घटक, दोनों WAV, समय, सेटिंग और स्रोत/मॉडल फिंगरप्रिंट सहेजते हैं। उसी बैच को `--resume` से जारी रखें। सभी 278 कार्यों के लिए `--num-tasks` हटाएँ।

जाँच: `make test` · `make lint` · `make verify`.

स्रोत: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice शोधपत्र](https://arxiv.org/abs/2603.13686) · [MIT लाइसेंस](LICENSE).
