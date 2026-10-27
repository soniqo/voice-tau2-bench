# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo)는 Apple Silicon에서 **로컬 음성 에이전트 성능**을 측정하기 위해 이 독립형 τ-Voice 벤치마크를 관리합니다. 에이전트, 모의 발신자, 음성 모델, 의미 평가 모델을 [speech-swift](https://github.com/soniqo/speech-swift)로 실행합니다.

## 문서

- [설정 및 모델 파일](docs/runtime-setup.md)
- [평가와 채점](docs/evaluation.md)
- [구현](docs/local-voice-extraction.md)
- [모델 실행 기록](docs/results/gemma4-e4b-local-600.json)
- [Gemma 4 12B 실행 기록과 동작 매개변수](docs/validation/gemma4-12b-voice-e2e.md)

## 측정 대상

음성 에이전트가 대화와 도구 사용으로 고객 서비스 작업을 완료하는지 측정합니다. 작업 보상은 요구된 최종 데이터베이스 상태와 전달 내용을 확인합니다.

고정 영어 `voice-v1`은 총 **278개 작업**입니다. 항공 **50**, 소매 **114**, 통신 **114**개입니다. 작업, 정책, 도구, 데이터베이스, 보상 기준은 원본의 고정 내용을 유지합니다.

## 테스트한 모델과 결과

| 역할 | 모델 | 정밀도 |
| --- | --- | --- |
| 에이전트 | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| 발신자, 청취 판단, 의미 평가 | Gemma 4 E4B IT | INT4 |
| 양측 음성 합성 | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| 음성 인식 | Whisper Large-v3 (full, CoreML) | FP16 |
| 발화 감지 | Silero VAD (CoreML) | Native |

개발용 표본: 각 분야 고정 순서의 첫 작업을 선택하여 총 **3개**, 각 **1회**, 시드 **42**, 대화 제한 **600초**, 깨끗한 `control` 전화 음성을 사용합니다. 하드웨어는 **Apple M5 Pro, 통합 메모리 48 GB**입니다. 성공률은 위 구성 전체를 평가합니다.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| 에이전트 모델 | 항공 | 소매 | 통신 | 전체 pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

이 작은 결정적 표본은 로컬 발신자와 평가 모델을 쓰는 사용자 설정입니다. 전체 278개 점수나 원본 순위표 결과를 뜻하지 않습니다. 시간 초과는 0점이며 추론 예외는 점수를 부여하지 않습니다.

12B 표본은 미완료입니다. 통신 작업에서 네이티브 추론 오류가 발생하여 점수를 매기지 않았습니다. 통제된 12B 음성 검사에서 올바른 데이터베이스 쓰기는 **1.0**, 잘못된 쓰기의 재평가는 **0.0**을 받았습니다. [동작 매개변수](docs/validation/gemma4-12b-voice-e2e.md). 발신자 입력은 고정되어 있으며 벤치마크 점수에 포함되지 않습니다. 분야별 실행은 생성형 발신자를 사용하며, 각 런타임 버전은 연결된 보고서에 기록되어 있습니다.

최신 로컬 v2 개발 진단: **Gemma 4 12B IT (INT4), 원시 결과: 항공 0/1, 소매 0/1, 통신 0/1**. 상태 변경이 필요한 작업 3개를 시드 **70**으로 각각 한 번 실행했으며, 모든 통화가 **시뮬레이션 300초 제한**에 도달했습니다(**실제 시간 제한 900초**). 이 단축 실행 결과는 선택한 세 작업에만 해당합니다. [결과와 검증된 매개변수](docs/validation/development-smoke.md).

`soniqo-local-gemma12b-v2` · `2026-10-27` · `--max-seconds 300 --wall-timeout 900`

## 로컬 실행

Python **3.12/3.13**, Apple Silicon, **macOS 15+**가 필요합니다. 작업 실행 전에 [로컬 모델 서비스](docs/runtime-setup.md#serve)를 시작하세요.

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

각 작업은 도구·대화 기록, 보상 항목, 양측 WAV, 시간, 설정, 소스·모델 지문을 저장합니다. `--resume`으로 동일 배치를 이어갑니다. `--num-tasks`를 생략하면 전체 278개를 평가합니다.

검사: `make test` · `make lint` · `make verify`.

출처: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice 논문](https://arxiv.org/abs/2603.13686) · [MIT 라이선스](LICENSE).
