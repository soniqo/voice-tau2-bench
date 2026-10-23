# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[speech-swift](https://github.com/soniqo/speech-swift) पर स्थानीय अनुमान वाला स्वतंत्र **τ-Voice वॉइस एजेंट बेंचमार्क**। Apple Silicon पर ग्राहक सेवा वार्तालाप, टूल उपयोग और कार्य पूरा करने का मूल्यांकन करता है।

स्थिर अंग्रेज़ी `voice-v1` में **278 कार्य** हैं: एयरलाइन **50**, रिटेल **114**, टेलीकॉम **114**। कार्य, नीतियाँ, टूल, डेटाबेस और पुरस्कार मानदंड मूल पिन बरकरार रखते हैं।

## स्थानीय मॉडल

| भूमिका | मॉडल | प्रेसिजन |
| --- | --- | --- |
| एजेंट, कॉलर, सुनने के निर्णय, अर्थ संबंधी जज | Gemma 4 E4B IT | INT4 |
| एजेंट और कॉलर की आवाज़ | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| वाक् पहचान | Whisper Large-v3 (full, CoreML) | FP16 |
| वाणी का पता लगाना | Silero VAD (CoreML) | Native |

मॉडल तैयारी के समय डाउनलोड होते हैं; अनुमान में क्लाउड प्रदाता या API कुंजी नहीं लगती। उपयुक्त speech-swift बिल्ड और मॉडल फाइलों के लिए [रनटाइम सेटअप](docs/runtime-setup.md) देखें।

## मापे गए परिणाम

विकास नमूना: हर डोमेन के स्थिर क्रम का पहला कार्य, कुल **3 कार्य**, **1 प्रयास**, सीड **42**, बातचीत सीमा **600 सेकंड**, साफ़ `control` टेलीफोनी। हार्डवेयर: **Apple M5 Pro, 48 GB यूनिफाइड मेमोरी**। सफलता ऊपर दिए पूरे स्टैक को मापती है।

`soniqo-local-v1` · `2026-10-23`

| एजेंट मॉडल | एयरलाइन | रिटेल | टेलीकॉम | कुल pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

इस छोटे निश्चित नमूने में स्थानीय कॉलर और जज हैं। यह सभी 278 कार्यों का स्कोर या मूल लीडरबोर्ड परिणाम नहीं है। टाइमआउट का स्कोर शून्य है; अनुमान अपवाद बिना स्कोर रहते हैं।

## जल्दी शुरू करें

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

## जाँच और मूल्यांकन

```bash
make test
make lint
make verify
```

[मूल्यांकन नियम](docs/evaluation.md), [आर्किटेक्चर](docs/local-voice-extraction.md) और [मॉडल रन रिकॉर्ड](docs/results/gemma4-e4b-local-600.json) देखें। Mock और घटक जाँच विकास के लिए हैं, कार्य स्कोर का भाग नहीं।

## स्रोत

[Sierra Research τ²-Bench](https://github.com/sierra-research/tau2-bench) के कमिट `05736c27219b769d0be507574dd3135010916644` से निकाला गया। मूल [MIT लाइसेंस](LICENSE) बना रहता है। उपयोग पर [τ-Voice शोधपत्र](https://arxiv.org/abs/2603.13686) का उल्लेख करें।
