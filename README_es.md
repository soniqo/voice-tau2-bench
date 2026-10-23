# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Benchmark independiente de **agentes de voz τ-Voice**, con inferencia local mediante [speech-swift](https://github.com/soniqo/speech-swift). Evalúa conversaciones de atención al cliente, uso de herramientas y cumplimiento de tareas en Apple Silicon.

El marco inglés `voice-v1` fija **278 tareas**: aerolíneas **50**, comercio **114** y telecomunicaciones **114**. Tareas, políticas, herramientas, bases de datos y recompensas conservan las referencias originales.

## Modelos locales

| Función | Modelo | Precisión |
| --- | --- | --- |
| Agente, interlocutor, escucha y juez semántico | Gemma 4 E4B IT | INT4 |
| Voces del agente y del interlocutor | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Reconocimiento de voz | Whisper Large-v3 (full, CoreML) | FP16 |
| Detección de voz | Silero VAD (CoreML) | Native |

Los modelos se descargan durante la preparación; la inferencia no usa proveedores de nube ni claves API. La [configuración del entorno](docs/runtime-setup.md) explica la compilación de speech-swift y los archivos necesarios.

## Resultados medidos

Muestra de desarrollo: primera tarea congelada de cada dominio, **3 tareas** en total, **1 intento**, semilla **42**, plazo de conversación de **600 segundos**, telefonía limpia `control`. Equipo: **Apple M5 Pro, 48 GB de memoria unificada**. El éxito mide el conjunto de modelos anterior.

`soniqo-local-v1` · `2026-10-23`

| Modelo del agente | Aerolíneas | Comercio | Telecomunicaciones | pass^1 total |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Esta muestra pequeña y determinista usa interlocutor y juez locales. No establece una puntuación de las 278 tareas ni una clasificación oficial. Los timeouts reciben cero; las excepciones de inferencia quedan sin puntuar.

## Inicio rápido

Use Python **3.12/3.13**, Apple Silicon y **macOS 15+**. Inicie el [servicio local](docs/runtime-setup.md#serve) antes de ejecutar las tareas.

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

Cada tarea guarda trayectoria, herramientas, recompensas, WAV de ambos participantes, tiempos, configuración y huellas de fuentes/modelos. Continúe el mismo lote con `--resume`. Omita `--num-tasks` para evaluar las 278 tareas.

## Verificación y evaluación

```bash
make test
make lint
make verify
```

Consulte las [reglas de evaluación](docs/evaluation.md), la [arquitectura](docs/local-voice-extraction.md) y el [registro del modelo](docs/results/gemma4-e4b-local-600.json). Los casos mock y pruebas de componentes son controles de desarrollo y no suman a la puntuación.

## Atribución

Derivado de [τ²-Bench de Sierra Research](https://github.com/sierra-research/tau2-bench), commit `05736c27219b769d0be507574dd3135010916644`. Se conserva la [licencia MIT](LICENSE). Cite el [artículo τ-Voice](https://arxiv.org/abs/2603.13686) al usar el benchmark.
