# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo)는 음성 에이전트가 대화와 도구 사용으로 작업을 완료하는지 측정하는 이 벤치마크를 유지합니다. 고정된 영어 데이터에는 **278개 작업**이 있습니다: 항공 **50**, 소매 **114**, 통신 **114**.

[설정](docs/runtime-setup.md) · [평가](docs/evaluation.md) · [모델 비교 및 제한](docs/model-comparison.md)

## 모델 및 시스템

| 에이전트 또는 시스템 | 현재 지원 |
| --- | --- |
| Gemma 4 12B IT, INT4 | 검증된 로컬 v2 프로필 |
| Gemma 4 E4B IT, INT4 | 로컬 런타임 지원; 동일 조건의 v2 비교는 대기 중 |
| OpenAI GPT-Live + 명시한 백엔드 에이전트 | 어댑터 필요 |
| OpenAI Realtime | 어댑터 필요 |
| Gemini Live | 어댑터 필요 |
| ElevenLabs Agents | 어댑터 필요 |

로컬 구성은 [speech-swift](https://github.com/soniqo/speech-swift)를 사용합니다: 전체 Whisper Large-v3 FP16, VoxCPM2 주 가중치 FP16 / AudioVAE FP32, Silero VAD 및 발신자·청취 보조·평가자용 Gemma E4B. 클라우드 시스템은 아직 통합되거나 이 벤치마크에서 평가되지 않았습니다.

## 비교 지표

| 지표 | 측정 내용 |
| --- | --- |
| 작업 성공률, pass^1 | 요구되는 최종 데이터베이스 상태와 정보 전달 |
| 범위 및 실패 | 평가 완료/계획된 시도, 정상 종료, 평가하지 않은 추론 오류 |
| 지연, p50 / p95 | 에이전트 LLM·ASR 실행 시간 및 TTS 첫 오디오 시간 |
| 음성 응답 지연 | 발신자 발화 종료부터 에이전트 첫 오디오까지; 표준 집계는 대기 중 |
| 도구 오류율 | 실패하거나 거부된 에이전트 호출 / 시도한 호출 |
| 완료 시간 | 시뮬레이션 통화 시간과 측정된 실제 시간을 별도로 기록 |
| 최대 메모리 | 측정 범위를 명시한 네이티브 프로세스의 물리 메모리 사용량 |
| API 비용 | 시도 및 성공한 시도당 USD; 실패 시도 포함 |

동일한 작업, 시드, 통화 제한, 음향 조건, 발신자·평가자 설정으로 비교합니다. 로컬 LLM 비교에서는 음성 구성을 고정합니다. 오디오 네이티브 및 관리형 에이전트는 전체 구성을 명시해야 합니다. 구성 요소별 시간과 전체 음성 응답 지연은 별도 지표입니다.

## 기록된 로컬 진단

| 에이전트 | 평가 / 계획 | Pass^1 | 도구 오류 | LLM p50 / p95 | TTS 첫 오디오 p50 / p95 | API 비용 / 시도 |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

항공 작업 12, 1회 시도, 시드 70, v2; 최대 **시뮬레이션 600초 / 실제 1,200초**. 통화는 **시뮬레이션 312.9초 / 측정된 실제 554.4초** 후 정상 종료되었습니다. 하드웨어: Apple M5 Pro, 48 GB, 공유 워크스테이션. 개발용 단일 시도이며 동일 조건의 여러 모델 순위는 아직 없습니다. [데이터 및 측정 정의](docs/results/gemma4-12b-local-v2-airline.json).

## 로컬 실행

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. [설정](docs/runtime-setup.md) 및 [v2 실행](docs/local-v2.md)에 따라 로컬 서비스를 시작하고 발신자·평가자를 검증한 뒤 작업을 실행합니다.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

`--num-tasks`, `--trials`, `--max-seconds`, `--wall-timeout`으로 실행을 제한합니다. 클라우드 어댑터와 USD 지출 한도는 **구현되지 않았습니다**. 현재 추론은 로컬에서 실행되며 클라우드 API 키가 필요 없습니다.

검사: `make test` · `make lint` · `make verify`.

출처: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
