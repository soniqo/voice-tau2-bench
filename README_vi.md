# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) duy trì bộ đánh giá τ-Voice độc lập này để đo **hiệu suất tác tử giọng nói cục bộ** trên Apple Silicon. Chúng tôi chạy tác tử, người gọi mô phỏng, các mô hình tiếng nói và bộ chấm ngữ nghĩa qua [speech-swift](https://github.com/soniqo/speech-swift).

## Tài liệu

- [Thiết lập và tệp mô hình](docs/runtime-setup.md)
- [Đánh giá và chấm điểm](docs/evaluation.md)
- [Triển khai](docs/local-voice-extraction.md)
- [Bản ghi lần chạy mô hình](docs/results/gemma4-e4b-local-600.json)
- [Lần chạy Gemma 4 12B và tham số đã kiểm chứng](docs/validation/gemma4-12b-voice-e2e.md)

## Chúng tôi đo gì

Chúng tôi đo xem tác tử giọng nói có hoàn thành tác vụ chăm sóc khách hàng bằng hội thoại và công cụ hay không. Điểm thưởng kiểm tra trạng thái cơ sở dữ liệu cuối cùng theo yêu cầu và thông tin đã truyền đạt.

Khung tiếng Anh `voice-v1` cố định gồm **278 tác vụ**: hàng không **50**, bán lẻ **114**, viễn thông **114**. Tác vụ, chính sách, công cụ, cơ sở dữ liệu và tiêu chí thưởng giữ nguyên dấu vân tay của bản gốc.

## Mô hình đã thử và kết quả

| Vai trò | Mô hình | Độ chính xác số |
| --- | --- | --- |
| Tác nhân | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| Người gọi, quyết định lắng nghe, giám khảo ngữ nghĩa | Gemma 4 E4B IT | INT4 |
| Giọng của tác nhân và người gọi | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Nhận dạng tiếng nói | Whisper Large-v3 (full, CoreML) | FP16 |
| Phát hiện lời nói | Silero VAD (CoreML) | Native |

Mẫu phát triển: tác vụ đầu theo thứ tự cố định của mỗi miền, tổng **3 tác vụ**, **1 lần thử**, seed **42**, hạn hội thoại **600 giây**, âm thanh điện thoại sạch `control`. Phần cứng: **Apple M5 Pro, bộ nhớ hợp nhất 48 GB**. Thành công đánh giá toàn bộ cấu hình trên.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| Mô hình tác nhân | Hàng không | Bán lẻ | Viễn thông | pass^1 tổng |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

Mẫu nhỏ xác định này dùng người gọi và giám khảo cục bộ. Nó không xác lập điểm toàn bộ 278 tác vụ hay thứ hạng gốc. Hết giờ nhận điểm không; ngoại lệ suy luận không được chấm.

Mẫu 12B chưa hoàn tất: tác vụ viễn thông gặp lỗi suy luận native nên chưa được chấm điểm. Kiểm tra giọng nói 12B có kiểm soát: **1.0** cho lần ghi cơ sở dữ liệu đúng và **0.0** khi chấm lại một lần ghi sai. [Tham số đã kiểm chứng](docs/validation/gemma4-12b-voice-e2e.md). Đầu vào người gọi được cố định; kiểm tra này không tính vào điểm đánh giá. Các lần chạy theo miền dùng người gọi sinh văn bản và báo cáo liên kết ghi lại từng phiên bản runtime.

Các lần chạy lại V2 vẫn mang tính chẩn đoán: một ca từ chối hàng không đạt **1.0** vì cơ sở dữ liệu không đổi; bán lẻ hết thời gian và viễn thông phát hiện lỗi điều khiển người gọi. **Đợt thử nghiệm giữ riêng gồm 18 cuộc gọi chưa hoàn tất**. Các lần chạy này chưa xác định được tỷ lệ vượt qua tổng thể của mô hình. [Kết quả và tham số](docs/validation/caller-rerun.md).

`soniqo-local-gemma12b-v2` · `--max-seconds 600 --wall-timeout 1200`

## Chạy cục bộ

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

Kiểm tra: `make test` · `make lint` · `make verify`.

Nguồn: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [Bài báo τ-Voice](https://arxiv.org/abs/2603.13686) · [Giấy phép MIT](LICENSE).
