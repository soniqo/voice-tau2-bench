# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Eigenständiger **τ-Voice-Benchmark für Sprachagenten** mit lokaler Inferenz durch [speech-swift](https://github.com/soniqo/speech-swift). Bewertet Kundengespräche, Werkzeugnutzung und Aufgabenerfüllung auf Apple Silicon.

Der fixierte englische Rahmen `voice-v1` enthält **278 Aufgaben**: Flugreisen **50**, Handel **114**, Telekommunikation **114**. Aufgaben, Richtlinien, Werkzeuge, Datenbanken und Bewertung behalten ihre ursprünglichen Inhaltsbindungen.

## Lokale Modelle

| Rolle | Modell | Präzision |
| --- | --- | --- |
| Agent, Anrufer, Zuhörhilfen und semantische Bewertung | Gemma 4 E4B IT | INT4 |
| Agenten- und Anruferstimme | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Spracherkennung | Whisper Large-v3 (full, CoreML) | FP16 |
| Sprachaktivitätserkennung | Silero VAD (CoreML) | Native |

Modelle werden bei der Einrichtung heruntergeladen; die Inferenz braucht keine Cloud-Anbieter oder API-Schlüssel. Der [Laufzeitleitfaden](docs/runtime-setup.md) beschreibt den passenden speech-swift-Build und die Modelldateien.

## Gemessene Ergebnisse

Entwicklungsstichprobe: erste fixierte Aufgabe je Domäne, insgesamt **3 Aufgaben**, **1 Versuch**, Seed **42**, Gesprächsfrist **600 Sekunden**, saubere `control`-Telefonie. Hardware: **Apple M5 Pro, 48 GB gemeinsamer Speicher**. Aufgabenerfolg bewertet den gesamten Modellstapel.

`soniqo-local-v1` · `2026-10-23`

| Agentenmodell | Flugreisen | Handel | Telekom | Gesamt pass^1 |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Diese kleine deterministische Stichprobe nutzt lokale Anrufer- und Bewertungsmodelle. Sie belegt weder den Wert für alle 278 Aufgaben noch ein offizielles Ranglistenergebnis. Zeitüberschreitungen zählen null; Inferenzfehler bleiben unbewertet.

## Schnellstart

Benötigt Python **3.12/3.13**, Apple Silicon und **macOS 15+**. Vor den Aufgaben den [lokalen Modelldienst](docs/runtime-setup.md#serve) starten.

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

Jede Aufgabe speichert Verlauf, Werkzeuge, Bewertungskomponenten, beide WAV-Dateien, Zeiten, Einstellungen und Quell-/Modellfingerabdrücke. Mit `--resume` denselben Lauf fortsetzen. Ohne `--num-tasks` werden alle 278 Aufgaben geprüft.

## Prüfung und Bewertung

```bash
make test
make lint
make verify
```

Siehe [Bewertungsregeln](docs/evaluation.md), [Architektur](docs/local-voice-extraction.md) und [Modellprotokoll](docs/results/gemma4-e4b-local-600.json). Mock-Fälle und Komponententests sind Entwicklungsprüfungen und zählen nicht zum Aufgabenergebnis.

## Herkunft

Basiert auf [Sierra Researchs τ²-Bench](https://github.com/sierra-research/tau2-bench), Commit `05736c27219b769d0be507574dd3135010916644`. Die [MIT-Lizenz](LICENSE) bleibt erhalten. Bei Nutzung das [τ-Voice-Paper](https://arxiv.org/abs/2603.13686) zitieren.
