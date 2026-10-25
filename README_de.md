# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) pflegt diesen eigenständigen τ-Voice-Benchmark, um die **Leistung lokaler Sprachagenten** auf Apple Silicon zu messen. Wir führen Agent, simulierten Anrufer, Sprachmodelle und semantische Bewertung mit [speech-swift](https://github.com/soniqo/speech-swift) aus.

## Dokumentation

- [Einrichtung und Modelldateien](docs/runtime-setup.md)
- [Auswertung und Bewertung](docs/evaluation.md)
- [Implementierung](docs/local-voice-extraction.md)
- [Protokoll des Modelllaufs](docs/results/gemma4-e4b-local-600.json)
- [Gemma-4-12B-Lauf und geprüfte Parameter](docs/validation/gemma4-12b-voice-e2e.md)

## Was wir messen

Wir messen, ob ein Sprachagent Kundenserviceaufgaben durch Gespräche und Werkzeugnutzung abschließt. Die Aufgabenbewertung prüft den geforderten Datenbankendzustand und die übermittelten Informationen.

Der fixierte englische Rahmen `voice-v1` enthält **278 Aufgaben**: Flugreisen **50**, Handel **114**, Telekommunikation **114**. Aufgaben, Richtlinien, Werkzeuge, Datenbanken und Bewertung behalten ihre ursprünglichen Inhaltsbindungen.

## Getestete Modelle und Ergebnisse

| Rolle | Modell | Präzision |
| --- | --- | --- |
| Agent | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| Anrufer, Zuhörhilfen und semantische Bewertung | Gemma 4 E4B IT | INT4 |
| Agenten- und Anruferstimme | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Spracherkennung | Whisper Large-v3 (full, CoreML) | FP16 |
| Sprachaktivitätserkennung | Silero VAD (CoreML) | Native |

Entwicklungsstichprobe: erste fixierte Aufgabe je Domäne, insgesamt **3 Aufgaben**, **1 Versuch**, Seed **42**, Gesprächsfrist **600 Sekunden**, saubere `control`-Telefonie. Hardware: **Apple M5 Pro, 48 GB gemeinsamer Speicher**. Aufgabenerfolg bewertet den gesamten Modellstapel.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| Agentenmodell | Flugreisen | Handel | Telekom | Gesamt pass^1 |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

Diese kleine deterministische Stichprobe nutzt lokale Anrufer- und Bewertungsmodelle. Sie belegt weder den Wert für alle 278 Aufgaben noch ein offizielles Ranglistenergebnis. Zeitüberschreitungen zählen null; Inferenzfehler bleiben unbewertet.

Die 12B-Stichprobe ist unvollständig: Beim Telekommunikationstest trat ein nativer Inferenzfehler auf; er bleibt unbewertet. Kontrollierter 12B-Sprachtest: **1.0** für den korrekten Datenbankeintrag und **0.0** bei erneuter Bewertung eines falschen Eintrags. [Geprüfte Parameter](docs/validation/gemma4-12b-voice-e2e.md). Die Anrufereingabe ist fest vorgegeben; dieser Test zählt nicht zum Benchmark-Ergebnis. Domänenläufe verwenden den generativen Anrufer. Die verlinkten Berichte enthalten die jeweilige Runtime-Revision.

Lokales v2 beantwortet Fragen zum Telekommunikationsgerät jetzt anhand tatsächlicher Beobachtungen der Telefonwerkzeuge. Diagnoseergebnisse und der Fortschritt des zurückgehaltenen Piloten werden getrennt von historischen v1-Werten berichtet. [Aktuelle Validierung und Parameter](docs/validation/grounded-caller.md).

`soniqo-local-gemma12b-v2` · `--max-seconds 600 --wall-timeout 1200`

## Lokal ausführen

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

Prüfungen: `make test` · `make lint` · `make verify`.

Quelle: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice-Paper](https://arxiv.org/abs/2603.13686) · [MIT-Lizenz](LICENSE).
