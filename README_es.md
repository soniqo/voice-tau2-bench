# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) mantiene este benchmark para medir si los agentes de voz completan tareas mediante conversación hablada y herramientas. El conjunto inglés congelado contiene **278 tareas**: aerolínea **50**, comercio **114**, telecomunicaciones **114**.

[Configuración](docs/runtime-setup.md) · [Puntuación](docs/evaluation.md) · [Comparación de modelos y límites](docs/model-comparison.md)

## Modelos y sistemas

| Agente o sistema | Soporte actual |
| --- | --- |
| Gemma 4 12B IT, INT4 | Perfil local v2 probado |
| Gemma 4 E4B IT, INT4 | Runtime local compatible; comparación v2 equivalente pendiente |
| Amazon Nova 2.5 Sonic | Audio nativo validado; Retail pass^1 **1/1** (un ensayo de desarrollo v6, límite de 420 s; [registro](docs/results/nova25-v6-retail-development.json)) |
| OpenAI GPT-Live + un agente backend identificado | [Adaptador experimental probado sin conexión; validación en vivo pendiente](docs/providers.md) |
| OpenAI Realtime | [Adaptador experimental probado sin conexión; validación en vivo pendiente](docs/providers.md) |
| Gemini Live | [Adaptador experimental probado sin conexión; validación en vivo pendiente](docs/providers.md) |
| ElevenLabs Agents | [Adaptador experimental probado sin conexión; validación en vivo pendiente](docs/providers.md) |

La pila local utiliza [speech-swift](https://github.com/soniqo/speech-swift): Whisper Large-v3 completo FP16, pesos principales de VoxCPM2 FP16 / AudioVAE FP32, Silero VAD y Gemma E4B como interlocutor, asistente de escucha y juez. Nova 2.5 Sonic dispone de un [ejecutor de audio nativo](docs/nova.md) opcional; el interlocutor y el juez siguen siendo locales.

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

Compare las mismas tareas, semillas, límites, condiciones acústicas y configuración del interlocutor/juez. Mantenga fija la pila de voz al comparar LLM locales. Los agentes de audio nativo y gestionados deben identificar su pila completa. Los tiempos de componentes son distintos del retardo de respuesta de voz de extremo a extremo.

## Resultados de desarrollo registrados

Un ensayo de desarrollo por fila. Las tareas y los relojes difieren; la clasificación con condiciones equivalentes sigue pendiente.

| Agente | Condición / tarea | Puntuados / previstos | Pass^1 | Errores de herramientas | Tiempo real |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | v2 · Aerolínea 12 | 1/1 | 0/1 | 1/6 | 554.4 s |
| Amazon Nova 2.5 Sonic | v6 · Comercio 0 | 1/1 | **1/1** | **0/5** | **296.3 s** |

Tarea de aerolínea 12, un ensayo, semilla 70, v2; máximo de **600 segundos simulados / 1.200 segundos reales**. La llamada terminó normalmente tras **312,9 segundos simulados / 554,4 segundos reales medidos**. Equipo: Apple M5 Pro, 48 GB, estación compartida. Es un ensayo de desarrollo; la clasificación con varios modelos bajo condiciones equivalentes sigue pendiente. [Datos y definiciones de medición](docs/results/gemma4-12b-local-v2-airline.json).

Gemma LLM p50 / p95: **7.00 / 11.13 s**; TTS p50 / p95: **1.22 / 1.70 s**.

Nova: un ensayo, semilla 70, acústica control, **límite de 420 s en tiempo real**. Se intercambiaron ambos artículos solicitados; coincidieron las **5/5** acciones de referencia y el estado final de la base de datos. La llamada terminó normalmente tras **259,0 s de audio / 296,3 s reales**. [Registro y límites](docs/results/nova25-v6-retail-development.json).

## Ejecución local

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Siga la [configuración](docs/runtime-setup.md) y la [ejecución v2](docs/local-v2.md) para iniciar el servicio local, validar al interlocutor/juez y ejecutar tareas.

```bash
uv run --frozen --extra dev python scripts/run_local_voice.py \
  --output data/simulations/first-local-run
```

Este comando compila el entorno nativo fijado, descarga los modelos, calibra al interlocutor y al evaluador y ejecuta una prueba de voz controlada. No genera una puntuación del benchmark por dominio. Use un directorio de salida nuevo. [Configuración y ejecución de tareas](docs/runtime-setup.md#one-command-local-run).

Limite la ejecución con `--num-tasks`, `--trials`, `--max-seconds` y `--wall-timeout`. La ejecución predeterminada sigue siendo local. Ejecutores opcionales: [Nova](docs/nova.md) · [OpenAI, Gemini y ElevenLabs](docs/providers.md).

Comprobaciones: `make test` · `make lint` · `make verify`.

Origen: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
