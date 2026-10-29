# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo)는 음성 에이전트가 대화와 도구 사용으로 작업을 완료하는지 측정하는 이 벤치마크를 유지합니다. 고정된 영어 데이터에는 **278개 작업**이 있습니다: 항공 **50**, 소매 **114**, 통신 **114**.

[설정](docs/runtime-setup.md) · [평가](docs/evaluation.md) · [모델 비교 및 제한](docs/model-comparison.md)

## 모델 및 시스템

| 에이전트 또는 시스템 | 현재 지원 |
| --- | --- |
| Gemma 4 12B IT, INT4 | 검증된 로컬 v2 프로필 |
| Gemma 4 E4B IT, INT4 | 로컬 런타임 지원; 동일 조건의 v2 비교는 대기 중 |
| Amazon Nova 2.5 Sonic | 네이티브 오디오 검증 완료; Retail pass^1 **1/1** (v6 개발 시험 1회, 제한 420초; [기록](docs/results/nova25-v6-retail-development.json)) |
| OpenAI GPT-Live + 명시한 백엔드 에이전트 | [실험용 어댑터 오프라인 테스트 완료; 실제 서비스 검증 대기 중](docs/providers.md) |
| OpenAI Realtime | [실험용 어댑터 오프라인 테스트 완료; 실제 서비스 검증 대기 중](docs/providers.md) |
| Gemini Live | [실험용 어댑터 오프라인 테스트 완료; 실제 서비스 검증 대기 중](docs/providers.md) |
| ElevenLabs Agents | [실험용 어댑터 오프라인 테스트 완료; 실제 서비스 검증 대기 중](docs/providers.md) |

로컬 구성은 [speech-swift](https://github.com/soniqo/speech-swift)를 사용합니다: 전체 Whisper Large-v3 FP16, VoxCPM2 주 가중치 FP16 / AudioVAE FP32, Silero VAD 및 발신자·청취 보조·평가자용 Gemma E4B. Nova 2.5 Sonic은 선택적 [네이티브 오디오 실행기](docs/nova.md)를 지원하며 발신자와 평가자는 로컬에서 실행됩니다.

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

동일한 작업, 시드, 통화 제한, 음향 조건, 발신자·평가자 설정으로 비교합니다. 로컬 LLM 비교에서는 음성 구성을 고정합니다. 오디오 네이티브 및 관리형 에이전트는 전체 구성을 명시해야 합니다. 구성 요소별 시간과 전체 음성 응답 지연은 별도 지표입니다.

## 기록된 개발 시험 결과

각 행은 개발 시험 1회입니다. 작업과 시간 설정이 다르며 동일 조건의 모델 순위는 아직 없습니다.

| 에이전트 | 조건 / 작업 | 평가 / 계획 | Pass^1 | 도구 오류 | 실제 시간 |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | v2 · 항공 12 | 1/1 | 0/1 | 1/6 | 554.4 s |
| Amazon Nova 2.5 Sonic | v6 · 소매 0 | 1/1 | **1/1** | **0/5** | **296.3 s** |

항공 작업 12, 1회 시도, 시드 70, v2; 최대 **시뮬레이션 600초 / 실제 1,200초**. 통화는 **시뮬레이션 312.9초 / 측정된 실제 554.4초** 후 정상 종료되었습니다. 하드웨어: Apple M5 Pro, 48 GB, 공유 워크스테이션. 개발용 단일 시도이며 동일 조건의 여러 모델 순위는 아직 없습니다. [데이터 및 측정 정의](docs/results/gemma4-12b-local-v2-airline.json).

Gemma LLM p50 / p95: **7.00 / 11.13 s**; TTS p50 / p95: **1.22 / 1.70 s**.

Nova: 1회 시험, 시드 70, control 음향 조건, **실시간 / 실제 시간 제한 420초**. 요청한 두 상품을 교환했고 **5/5** 참조 작업과 최종 데이터베이스 상태가 일치했습니다. 통화는 **오디오 259.0초 / 실제 296.3초** 후 정상 종료되었습니다. [실행 기록 및 한계](docs/results/nova25-v6-retail-development.json).

## 로컬 실행

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. [설정](docs/runtime-setup.md) 및 [v2 실행](docs/local-v2.md)에 따라 로컬 서비스를 시작하고 발신자·평가자를 검증한 뒤 작업을 실행합니다.

```bash
uv run --frozen --extra dev python scripts/run_local_voice.py \
  --output data/simulations/first-local-run
```

이 명령은 고정된 네이티브 런타임을 빌드하고 모델을 다운로드하며 발신자와 평가기를 보정한 뒤 제어된 음성 스모크 테스트를 실행합니다. 도메인 벤치마크 점수는 생성하지 않습니다. 새로운 출력 디렉터리를 사용하세요. [설정 및 작업 실행](docs/runtime-setup.md#one-command-local-run).

`--num-tasks`, `--trials`, `--max-seconds`, `--wall-timeout`으로 실행을 제한합니다. 기본 실행은 로컬입니다. 선택적 실행기: [Nova](docs/nova.md) · [OpenAI, Gemini, ElevenLabs](docs/providers.md).

검사: `make test` · `make lint` · `make verify`.

출처: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
