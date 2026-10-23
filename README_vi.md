# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Bộ đánh giá **tác nhân giọng nói τ-Voice** độc lập, suy luận cục bộ qua [speech-swift](https://github.com/soniqo/speech-swift). Đánh giá hội thoại chăm sóc khách hàng, sử dụng công cụ và hoàn thành tác vụ trên Apple Silicon.

Khung tiếng Anh `voice-v1` cố định gồm **278 tác vụ**: hàng không **50**, bán lẻ **114**, viễn thông **114**. Tác vụ, chính sách, công cụ, cơ sở dữ liệu và tiêu chí thưởng giữ nguyên dấu vân tay của bản gốc.

## Mô hình cục bộ

| Vai trò | Mô hình | Độ chính xác số |
| --- | --- | --- |
| Tác nhân, người gọi, quyết định lắng nghe, giám khảo ngữ nghĩa | Gemma 4 E4B IT | INT4 |
| Giọng của tác nhân và người gọi | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Nhận dạng tiếng nói | Whisper Large-v3 (full, CoreML) | FP16 |
| Phát hiện lời nói | Silero VAD (CoreML) | Native |

Tải mô hình trong bước chuẩn bị; suy luận không dùng nhà cung cấp đám mây hay khóa API. Xem [thiết lập môi trường](docs/runtime-setup.md) để dựng speech-swift và chuẩn bị tệp mô hình phù hợp.

## Kết quả đo

Mẫu phát triển: tác vụ đầu theo thứ tự cố định của mỗi miền, tổng **3 tác vụ**, **1 lần thử**, seed **42**, hạn hội thoại **600 giây**, âm thanh điện thoại sạch `control`. Phần cứng: **Apple M5 Pro, bộ nhớ hợp nhất 48 GB**. Thành công đánh giá toàn bộ cấu hình trên.

`soniqo-local-v1` · `2026-10-23`

| Mô hình tác nhân | Hàng không | Bán lẻ | Viễn thông | pass^1 tổng |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Mẫu nhỏ xác định này dùng người gọi và giám khảo cục bộ. Nó không xác lập điểm toàn bộ 278 tác vụ hay thứ hạng gốc. Hết giờ nhận điểm không; ngoại lệ suy luận không được chấm.

## Bắt đầu nhanh

Dùng Python **3.12/3.13**, Apple Silicon và **macOS 15+**. Khởi động [dịch vụ mô hình cục bộ](docs/runtime-setup.md#serve) trước khi chạy tác vụ.

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

Mỗi tác vụ lưu diễn tiến công cụ/hội thoại, các thành phần thưởng, WAV hai phía, thời gian, cấu hình và dấu vân tay nguồn/mô hình. Tiếp tục cùng đợt bằng `--resume`. Bỏ `--num-tasks` để đánh giá đủ 278 tác vụ.

## Kiểm tra và đánh giá

```bash
make test
make lint
make verify
```

Xem [quy tắc đánh giá](docs/evaluation.md), [kiến trúc](docs/local-voice-extraction.md) và [bản ghi mô hình](docs/results/gemma4-e4b-local-600.json). Mock và kiểm tra thành phần phục vụ phát triển, không tính vào điểm tác vụ.

## Nguồn

Dựa trên [τ²-Bench của Sierra Research](https://github.com/sierra-research/tau2-bench), commit `05736c27219b769d0be507574dd3135010916644`. Giữ [giấy phép MIT](LICENSE). Hãy trích dẫn [bài τ-Voice](https://arxiv.org/abs/2603.13686) khi sử dụng.
