# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) mantiene este benchmark para medir si los agentes de voz completan tareas mediante conversación hablada y herramientas. El conjunto inglés congelado contiene **278 tareas**: aerolínea **50**, comercio **114**, telecomunicaciones **114**.

[Configuración](docs/runtime-setup.md) · [Puntuación](docs/evaluation.md) · [Comparación de modelos y límites](docs/model-comparison.md)

## Modelos y sistemas

| Agente o sistema | Soporte actual |
| --- | --- |
| Gemma 4 12B IT, INT4 | Perfil local v2 probado |
| Gemma 4 E4B IT, INT4 | Runtime local compatible; comparación v2 equivalente pendiente |
| OpenAI GPT-Live + un agente backend identificado | Requiere adaptador |
| OpenAI Realtime | Requiere adaptador |
| Gemini Live | Requiere adaptador |
| ElevenLabs Agents | Requiere adaptador |

La pila local utiliza [speech-swift](https://github.com/soniqo/speech-swift): Whisper Large-v3 completo FP16, pesos principales de VoxCPM2 FP16 / AudioVAE FP32, Silero VAD y Gemma E4B como interlocutor, asistente de escucha y juez. Los sistemas en la nube no están integrados ni se han evaluado aquí.

## Métricas de comparación

| Métrica | Qué mide |
| --- | --- |
| Éxito de tarea, pass^1 | Estado final requerido de la base de datos y comunicación |
| Cobertura y fallos | Ensayos puntuados/previstos, finales normales y errores de inferencia sin puntuar |
| Latencia, p50 / p95 | Ejecución del LLM y ASR del agente; tiempo de TTS hasta el primer audio |
| Retardo de respuesta de voz | Fin del habla del interlocutor hasta el primer audio del agente; resumen normalizado pendiente |
| Tasa de error de herramientas | Llamadas del agente fallidas o rechazadas / llamadas intentadas |
| Tiempo de finalización | Duración simulada de llamada y tiempo real medido, por separado |
| Memoria máxima | Memoria física del proceso nativo, indicando el alcance de medición |
| Coste de API | USD por ensayo y por ensayo exitoso; incluir intentos fallidos |

Compare las mismas tareas, semillas, límites, condiciones acústicas y configuración del interlocutor/juez. Mantenga fija la pila de voz al comparar LLM locales. Los agentes de audio nativo y gestionados deben identificar su pila completa. Los tiempos de componentes son distintos del retardo de respuesta de voz de extremo a extremo.

## Diagnóstico local registrado

| Agente | Puntuados / previstos | Pass^1 | Errores de herramientas | LLM p50 / p95 | Primer audio TTS p50 / p95 | Coste API / ensayo |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

Tarea de aerolínea 12, un ensayo, semilla 70, v2; máximo de **600 segundos simulados / 1.200 segundos reales**. La llamada terminó normalmente tras **312,9 segundos simulados / 554,4 segundos reales medidos**. Equipo: Apple M5 Pro, 48 GB, estación compartida. Es un ensayo de desarrollo; la clasificación con varios modelos bajo condiciones equivalentes sigue pendiente. [Datos y definiciones de medición](docs/results/gemma4-12b-local-v2-airline.json).

## Ejecución local

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Siga la [configuración](docs/runtime-setup.md) y la [ejecución v2](docs/local-v2.md) para iniciar el servicio local, validar al interlocutor/juez y ejecutar tareas.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

Limite la ejecución con `--num-tasks`, `--trials`, `--max-seconds` y `--wall-timeout`. Los adaptadores de nube y un límite de gasto en USD **no están implementados**. La inferencia actual es local y no necesita una clave de API de nube.

Comprobaciones: `make test` · `make lint` · `make verify`.

Origen: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
