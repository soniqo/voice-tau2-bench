# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) duy trì benchmark này để đo liệu tác nhân giọng nói có hoàn thành nhiệm vụ qua hội thoại và công cụ hay không. Bộ tiếng Anh cố định gồm **278 tác vụ**: hàng không **50**, bán lẻ **114**, viễn thông **114**.

[Thiết lập](docs/runtime-setup.md) · [Chấm điểm](docs/evaluation.md) · [So sánh mô hình và giới hạn](docs/model-comparison.md)

## Mô hình và hệ thống

| Tác nhân hoặc hệ thống | Hỗ trợ hiện tại |
| --- | --- |
| Gemma 4 12B IT, INT4 | Hồ sơ v2 cục bộ đã kiểm thử |
| Gemma 4 E4B IT, INT4 | Runtime cục bộ hỗ trợ; chưa so sánh v2 trong điều kiện tương đương |
| OpenAI GPT-Live + tác nhân backend được chỉ định | Cần bộ điều hợp |
| OpenAI Realtime | Cần bộ điều hợp |
| Gemini Live | Cần bộ điều hợp |
| ElevenLabs Agents | Cần bộ điều hợp |

Hệ thống cục bộ dùng [speech-swift](https://github.com/soniqo/speech-swift): Whisper Large-v3 đầy đủ FP16, trọng số chính VoxCPM2 FP16 / AudioVAE FP32, Silero VAD và Gemma E4B cho người gọi, trợ lý nghe và bộ chấm. Các hệ thống đám mây chưa được tích hợp hay đo trong benchmark này.

## Chỉ số so sánh

| Chỉ số | Nội dung đo |
| --- | --- |
| Thành công tác vụ, pass^1 | Trạng thái cơ sở dữ liệu cuối cùng và thông tin giao tiếp bắt buộc |
| Độ bao phủ và lỗi | Lượt được chấm/dự kiến, kết thúc bình thường và lỗi suy luận không chấm điểm |
| Độ trễ, p50 / p95 | Thời gian thực thi LLM, ASR của tác nhân; TTS đến âm thanh đầu tiên |
| Độ trễ trả lời bằng giọng nói | Từ khi người gọi nói xong đến âm thanh đầu tiên của tác nhân; chưa có tổng hợp chuẩn |
| Tỷ lệ lỗi công cụ | Lệnh gọi tác nhân thất bại hoặc bị từ chối / tổng lần thử gọi |
| Thời gian hoàn thành | Thời gian cuộc gọi mô phỏng và thời gian thực đo được, tách riêng |
| Bộ nhớ đỉnh | Bộ nhớ vật lý của tiến trình native, kèm phạm vi đo |
| Chi phí API | USD mỗi lượt và mỗi lượt thành công; gồm cả lượt thất bại |

So sánh cùng tác vụ, seed, giới hạn, điều kiện âm thanh và cấu hình người gọi/bộ chấm. Giữ nguyên hệ thống giọng nói khi so sánh LLM cục bộ. Tác nhân âm thanh native và dịch vụ được quản lý phải nêu đầy đủ hệ thống. Thời gian từng thành phần khác với độ trễ trả lời bằng giọng nói từ đầu đến cuối.

## Chẩn đoán cục bộ đã ghi nhận

| Tác nhân | Đã chấm / dự kiến | Pass^1 | Lỗi công cụ | LLM p50 / p95 | Âm thanh TTS đầu p50 / p95 | Chi phí API / lượt |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

Tác vụ hàng không 12, một lượt, seed 70, v2; tối đa **600 giây mô phỏng / 1.200 giây thực**. Cuộc gọi kết thúc bình thường sau **312,9 giây mô phỏng / 554,4 giây thực đo được**. Phần cứng: Apple M5 Pro, 48 GB, máy dùng chung. Đây là một lượt phát triển; chưa có xếp hạng nhiều mô hình trong điều kiện tương đương. [Dữ liệu và định nghĩa phép đo](docs/results/gemma4-12b-local-v2-airline.json).

## Chạy cục bộ

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Theo [thiết lập](docs/runtime-setup.md) và [chạy v2](docs/local-v2.md) để khởi động dịch vụ cục bộ, xác nhận người gọi/bộ chấm rồi chạy tác vụ.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

Giới hạn thực thi bằng `--num-tasks`, `--trials`, `--max-seconds` và `--wall-timeout`. Bộ điều hợp đám mây và trần chi phí USD **chưa được triển khai**. Suy luận hiện chạy cục bộ, không cần khóa API đám mây.

Kiểm tra: `make test` · `make lint` · `make verify`.

Nguồn: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
