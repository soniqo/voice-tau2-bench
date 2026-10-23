# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[speech-swift](https://github.com/soniqo/speech-swift)로 로컬 추론하는 독립 **τ-Voice 음성 에이전트 벤치마크**입니다. Apple Silicon에서 고객 서비스 대화, 도구 사용, 작업 완료를 평가합니다.

고정 영어 `voice-v1`은 총 **278개 작업**입니다. 항공 **50**, 소매 **114**, 통신 **114**개입니다. 작업, 정책, 도구, 데이터베이스, 보상 기준은 원본의 고정 내용을 유지합니다.

## 로컬 모델 구성

| 역할 | 모델 | 정밀도 |
| --- | --- | --- |
| 에이전트, 발신자, 청취 판단, 의미 평가 | Gemma 4 E4B IT | INT4 |
| 양측 음성 합성 | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| 음성 인식 | Whisper Large-v3 (full, CoreML) | FP16 |
| 발화 감지 | Silero VAD (CoreML) | Native |

모델은 준비 단계에서 내려받습니다. 추론에는 클라우드나 API 키를 사용하지 않습니다. 필요한 speech-swift 빌드와 모델 준비는 [런타임 설정](docs/runtime-setup.md)을 확인하세요.

## 측정 결과

개발용 표본: 각 분야 고정 순서의 첫 작업을 선택하여 총 **3개**, 각 **1회**, 시드 **42**, 대화 제한 **600초**, 깨끗한 `control` 전화 음성을 사용합니다. 하드웨어는 **Apple M5 Pro, 통합 메모리 48 GB**입니다. 성공률은 위 구성 전체를 평가합니다.

`soniqo-local-v1` · `2026-10-23`

| 에이전트 모델 | 항공 | 소매 | 통신 | 전체 pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

이 작은 결정적 표본은 로컬 발신자와 평가 모델을 쓰는 사용자 설정입니다. 전체 278개 점수나 원본 순위표 결과를 뜻하지 않습니다. 시간 초과는 0점이며 추론 예외는 점수를 부여하지 않습니다.

## 빠른 시작

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

## 검증과 평가

```bash
make test
make lint
make verify
```

[평가 규칙](docs/evaluation.md), [아키텍처](docs/local-voice-extraction.md), [모델 실행 기록](docs/results/gemma4-e4b-local-600.json)을 참조하세요. Mock와 구성 요소 검증은 개발용으로 작업 점수에 포함되지 않습니다.

## 출처

[Sierra Research τ²-Bench](https://github.com/sierra-research/tau2-bench)의 커밋 `05736c27219b769d0be507574dd3135010916644`에서 파생했습니다. 원본 [MIT 라이선스](LICENSE)를 유지합니다. 사용 시 [τ-Voice 논문](https://arxiv.org/abs/2603.13686)을 인용하세요.
