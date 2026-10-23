# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

เบนช์มาร์ก **เอเจนต์เสียง τ-Voice** แบบอิสระ อนุมานภายในเครื่องด้วย [speech-swift](https://github.com/soniqo/speech-swift) ประเมินบทสนทนาบริการลูกค้า การใช้เครื่องมือ และการทำงานสำเร็จบน Apple Silicon

ชุดภาษาอังกฤษ `voice-v1` ที่ตรึงไว้มี **278 งาน**: สายการบิน **50** ค้าปลีก **114** โทรคมนาคม **114** โดยคงงาน นโยบาย เครื่องมือ ฐานข้อมูล และเกณฑ์รางวัลตามต้นฉบับ

## โมเดลภายในเครื่อง

| หน้าที่ | โมเดล | ความแม่นยำเชิงตัวเลข |
| --- | --- | --- |
| เอเจนต์ ผู้โทร การตัดสินใจฟัง และผู้ประเมินความหมาย | Gemma 4 E4B IT | INT4 |
| สร้างเสียงเอเจนต์และผู้โทร | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| รู้จำเสียงพูด | Whisper Large-v3 (full, CoreML) | FP16 |
| ตรวจจับเสียงพูด | Silero VAD (CoreML) | Native |

ดาวน์โหลดโมเดลตอนเตรียมระบบ การอนุมานไม่ใช้ผู้ให้บริการคลาวด์หรือ API key ดูบิลด์ speech-swift และไฟล์ที่ต้องใช้ใน[การตั้งค่ารันไทม์](docs/runtime-setup.md)

## ผลที่วัดได้

ตัวอย่างสำหรับพัฒนา: งานแรกตามลำดับคงที่ของแต่ละโดเมน รวม **3 งาน** งานละ **1 ครั้ง** seed **42** จำกัดบทสนทนา **600 วินาที** ใช้เสียงโทรศัพท์สะอาด `control` ฮาร์ดแวร์ **Apple M5 Pro หน่วยความจำรวม 48 GB** ความสำเร็จวัดทั้งชุดโมเดลข้างต้น

`soniqo-local-v1` · `2026-10-23`

| โมเดลเอเจนต์ | สายการบิน | ค้าปลีก | โทรคมนาคม | pass^1 รวม |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

ตัวอย่างขนาดเล็กที่เลือกแน่นอนนี้ใช้ผู้โทรและผู้ประเมินภายในเครื่อง ไม่ใช่คะแนนครบ 278 งานหรืออันดับต้นฉบับ หมดเวลาได้ศูนย์ ส่วนข้อยกเว้นการอนุมานไม่มีคะแนน

## เริ่มต้นใช้งาน

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

## การตรวจสอบและประเมิน

```bash
make test
make lint
make verify
```

ดู[เกณฑ์ประเมิน](docs/evaluation.md) [สถาปัตยกรรม](docs/local-voice-extraction.md) และ[บันทึกการรันโมเดล](docs/results/gemma4-e4b-local-600.json) การทดสอบ mock และส่วนประกอบใช้พัฒนา ไม่รวมในคะแนนงาน

## แหล่งที่มา

พัฒนาจาก [τ²-Bench ของ Sierra Research](https://github.com/sierra-research/tau2-bench) คอมมิต `05736c27219b769d0be507574dd3135010916644` คง[ใบอนุญาต MIT](LICENSE) และโปรดอ้างอิง[บทความ τ-Voice](https://arxiv.org/abs/2603.13686)เมื่อใช้เบนช์มาร์ก
