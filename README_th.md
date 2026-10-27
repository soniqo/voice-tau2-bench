# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) ดูแลเบนช์มาร์ก τ-Voice แบบอิสระนี้เพื่อวัด**ประสิทธิภาพเอเจนต์เสียงที่ทำงานในเครื่อง**บน Apple Silicon เรารันเอเจนต์ ผู้โทรจำลอง โมเดลเสียง และโมเดลประเมินเชิงความหมายผ่าน [speech-swift](https://github.com/soniqo/speech-swift)

## เอกสาร

- [การตั้งค่าและไฟล์โมเดล](docs/runtime-setup.md)
- [การประเมินและให้คะแนน](docs/evaluation.md)
- [การทำงานภายใน](docs/local-voice-extraction.md)
- [บันทึกการรันโมเดล](docs/results/gemma4-e4b-local-600.json)
- [การรัน Gemma 4 12B และพารามิเตอร์ที่ทดสอบแล้ว](docs/validation/gemma4-12b-voice-e2e.md)

## เราวัดอะไร

เราวัดว่าเอเจนต์เสียงทำงานบริการลูกค้าจนสำเร็จผ่านการสนทนาและการใช้เครื่องมือได้หรือไม่ คะแนนงานตรวจสอบสถานะฐานข้อมูลสุดท้ายและข้อมูลที่สื่อสารตามข้อกำหนด

ชุดภาษาอังกฤษ `voice-v1` ที่ตรึงไว้มี **278 งาน**: สายการบิน **50** ค้าปลีก **114** โทรคมนาคม **114** โดยคงงาน นโยบาย เครื่องมือ ฐานข้อมูล และเกณฑ์รางวัลตามต้นฉบับ

## โมเดลที่ทดสอบและผลลัพธ์

| หน้าที่ | โมเดล | ความแม่นยำเชิงตัวเลข |
| --- | --- | --- |
| เอเจนต์ | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| ผู้โทร การตัดสินใจฟัง และผู้ประเมินความหมาย | Gemma 4 E4B IT | INT4 |
| สร้างเสียงเอเจนต์และผู้โทร | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| รู้จำเสียงพูด | Whisper Large-v3 (full, CoreML) | FP16 |
| ตรวจจับเสียงพูด | Silero VAD (CoreML) | Native |

ตัวอย่างสำหรับพัฒนา: งานแรกตามลำดับคงที่ของแต่ละโดเมน รวม **3 งาน** งานละ **1 ครั้ง** seed **42** จำกัดบทสนทนา **600 วินาที** ใช้เสียงโทรศัพท์สะอาด `control` ฮาร์ดแวร์ **Apple M5 Pro หน่วยความจำรวม 48 GB** ความสำเร็จวัดทั้งชุดโมเดลข้างต้น

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| โมเดลเอเจนต์ | สายการบิน | ค้าปลีก | โทรคมนาคม | pass^1 รวม |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

ตัวอย่างขนาดเล็กที่เลือกแน่นอนนี้ใช้ผู้โทรและผู้ประเมินภายในเครื่อง ไม่ใช่คะแนนครบ 278 งานหรืออันดับต้นฉบับ หมดเวลาได้ศูนย์ ส่วนข้อยกเว้นการอนุมานไม่มีคะแนน

ตัวอย่าง 12B ยังไม่เสร็จสมบูรณ์: งานโทรคมนาคมเกิดข้อผิดพลาดในการอนุมานแบบเนทีฟจึงยังไม่มีคะแนน การตรวจสอบเสียง 12B แบบควบคุมได้คะแนน **1.0** เมื่อเขียนฐานข้อมูลถูกต้อง และ **0.0** เมื่อประเมินการเขียนที่ผิดอีกครั้ง [พารามิเตอร์ที่ทดสอบแล้ว](docs/validation/gemma4-12b-voice-e2e.md) ข้อความของผู้โทรกำหนดไว้ตายตัวและการตรวจสอบนี้ไม่นับเป็นคะแนนเบนช์มาร์ก การรันแต่ละโดเมนใช้ผู้โทรแบบสร้างข้อความ ส่วนรายงานที่ลิงก์ไว้บันทึกเวอร์ชันรันไทม์ของแต่ละครั้ง

ผลตรวจสอบการพัฒนา local v2 ล่าสุด: **Gemma 4 12B IT (INT4), ผลดิบ: สายการบิน 0/1, ค้าปลีก 0/1, โทรคมนาคม 0/1** ทดสอบสามงานที่ต้องเปลี่ยนสถานะ งานละหนึ่งครั้ง โดยใช้ seed **70** ทุกสายถึง **ขีดจำกัดเวลาจำลอง 300 วินาที** (**จำกัดเวลาจริง 900 วินาที**) ผลการรันทดสอบแบบย่อนี้ครอบคลุมเฉพาะสามงานที่เลือก [ผลและพารามิเตอร์ที่ทดสอบแล้ว](docs/validation/development-smoke.md)

`soniqo-local-gemma12b-v2` · `2026-10-27` · `--max-seconds 300 --wall-timeout 900`

## รันในเครื่อง

ใช้ Python **3.12/3.13**, Apple Silicon และ **macOS 15+** เริ่ม[บริการโมเดลภายในเครื่อง](docs/runtime-setup.md#serve)ก่อนรันงาน

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

แต่ละงานบันทึกลำดับบทสนทนาและเครื่องมือ ส่วนประกอบรางวัล WAV สองฝั่ง เวลา การตั้งค่า และลายนิ้วมือซอร์ส/โมเดล ใช้ `--resume` ต่อแบตช์เดิม ตัด `--num-tasks` ออกเพื่อประเมินครบ 278 งาน

การตรวจสอบ: `make test` · `make lint` · `make verify`.

ที่มา: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [บทความ τ-Voice](https://arxiv.org/abs/2603.13686) · [สัญญาอนุญาต MIT](LICENSE).
