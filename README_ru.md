# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Самостоятельный **тест голосовых агентов τ-Voice** с локальным выводом через [speech-swift](https://github.com/soniqo/speech-swift). Оценивает диалоги поддержки, использование инструментов и решение задач на Apple Silicon.

Фиксированный английский набор `voice-v1` содержит **278 задач**: авиаперевозки **50**, торговля **114**, телеком **114**. Задачи, правила, инструменты, базы данных и критерии награды сохраняют исходные привязки.

## Локальные модели

| Роль | Модель | Точность |
| --- | --- | --- |
| Агент, клиент, решения при прослушивании, смысловая оценка | Gemma 4 E4B IT | INT4 |
| Синтез речи агента и клиента | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Распознавание речи | Whisper Large-v3 (full, CoreML) | FP16 |
| Детекция речи | Silero VAD (CoreML) | Native |

Модели загружаются при подготовке; вывод не использует облако или ключи API. Нужная сборка speech-swift и подготовка файлов описаны в [настройке среды](docs/runtime-setup.md).

## Измеренные результаты

Выборка для разработки: первая задача каждого домена в фиксированном порядке, всего **3 задачи**, **1 попытка**, seed **42**, предел разговора **600 секунд**, чистая телефония `control`. Оборудование: **Apple M5 Pro, 48 ГБ объединённой памяти**. Успех оценивает весь указанный стек.

`soniqo-local-v1` · `2026-10-23`

| Модель агента | Авиа | Торговля | Телеком | Общий pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Эта малая детерминированная выборка использует локальные модели клиента и судьи. Она не устанавливает результат всех 278 задач или официального рейтинга. Таймаут получает ноль; ошибки вывода остаются без оценки.

## Быстрый старт

Нужны Python **3.12/3.13**, Apple Silicon и **macOS 15+**. Перед задачами запустите [локальный сервис моделей](docs/runtime-setup.md#serve).

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

Каждая задача сохраняет историю и инструменты, компоненты награды, оба WAV, время, настройки и отпечатки исходников/моделей. Продолжайте тот же запуск через `--resume`. Без `--num-tasks` оцениваются все 278 задач.

## Проверки и оценка

```bash
make test
make lint
make verify
```

См. [правила оценки](docs/evaluation.md), [архитектуру](docs/local-voice-extraction.md) и [протокол моделей](docs/results/gemma4-e4b-local-600.json). Mock и проверки компонентов предназначены для разработки и не входят в итог задач.

## Источники

Основано на [τ²-Bench Sierra Research](https://github.com/sierra-research/tau2-bench), коммит `05736c27219b769d0be507574dd3135010916644`. Сохранена [лицензия MIT](LICENSE). При использовании цитируйте [статью τ-Voice](https://arxiv.org/abs/2603.13686).
