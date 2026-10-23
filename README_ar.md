# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

تطوّر [Soniqo](https://github.com/soniqo) هذه النسخة المتفرعة من τ-Voice لقياس **أداء الوكلاء الصوتيين محليًا** على Apple Silicon. نشغّل الوكيل والمتصل المحاكى ونماذج الكلام والمقيّم الدلالي عبر [speech-swift](https://github.com/soniqo/speech-swift).

## التوثيق

- [الإعداد وملفات النماذج](docs/runtime-setup.md)
- [التقييم واحتساب الدرجات](docs/evaluation.md)
- [التنفيذ](docs/local-voice-extraction.md)
- [سجل تشغيل النماذج](docs/results/gemma4-e4b-local-600.json)

## ما الذي نقيسه

نقيس ما إذا كان الوكيل الصوتي ينجز مهام خدمة العملاء من خلال المحادثة واستخدام الأدوات. تتحقق مكافآت المهام من الحالة النهائية المطلوبة لقاعدة البيانات والمعلومات التي نُقلت إلى المتصل.

يحتوي إطار `voice-v1` الإنجليزي المثبت على **278 مهمة**: الطيران **50**، التجزئة **114**، والاتصالات **114**. تحتفظ المهام والسياسات والأدوات وقواعد البيانات ومعايير المكافأة ببصماتها الأصلية.

## النماذج المختبرة والنتائج

| الدور | النموذج | الدقة |
| --- | --- | --- |
| الوكيل والمتصل وقرارات الاستماع والحكم الدلالي | Gemma 4 E4B IT | INT4 |
| توليد صوت الوكيل والمتصل | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| التعرف على الكلام | Whisper Large-v3 (full, CoreML) | FP16 |
| كشف الكلام | Silero VAD (CoreML) | Native |

عينة تطوير: أول مهمة بالترتيب المثبت لكل مجال، **3 مهام** إجمالاً، **محاولة واحدة**، البذرة **42**، مهلة محادثة **600 ثانية**، وصوت هاتف نظيف `control`. الجهاز: **Apple M5 Pro بذاكرة موحدة 48 GB**. يقيس نجاح المهمة المنظومة الكاملة أعلاه.

`soniqo-local-v1` · `2026-10-23`

| نموذج الوكيل | الطيران | التجزئة | الاتصالات | إجمالي pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

تستخدم هذه العينة الصغيرة الحتمية متصلاً وحكماً محليين. لا تثبت نتيجة المهام الـ278 كلها أو ترتيباً رسمياً. انتهاء المهلة يعطي صفراً؛ استثناءات الاستدلال تبقى دون تقييم.

## التشغيل محليًا

يلزم Python **3.12/3.13** وApple Silicon و**macOS 15+**. شغّل [خدمة النماذج المحلية](docs/runtime-setup.md#serve) قبل المهام.

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

تحفظ كل مهمة مسار الأدوات والمحادثة ومكونات المكافأة وملفي WAV والأزمنة والإعدادات وبصمات المصدر والنماذج. استأنف الدفعة نفسها باستخدام `--resume`. احذف `--num-tasks` لتقييم المهام الـ278 كلها.

الفحوصات: `make test` · `make lint` · `make verify`.

المصدر: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [ورقة τ-Voice](https://arxiv.org/abs/2603.13686) · [ترخيص MIT](LICENSE).
