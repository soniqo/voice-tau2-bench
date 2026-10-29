# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) betreut diesen Benchmark, um die Aufgabenerfüllung von Sprachagenten durch gesprochene Gespräche und Werkzeugnutzung zu messen. Der eingefrorene englische Aufgabensatz enthält **278 Aufgaben**: Flug **50**, Handel **114**, Telekom **114**.

[Einrichtung](docs/runtime-setup.md) · [Bewertung](docs/evaluation.md) · [Modellvergleich und Grenzen](docs/model-comparison.md)

## Modelle und Systeme

| Agent oder System | Aktuelle Unterstützung |
| --- | --- |
| Gemma 4 12B IT, INT4 | Getestetes lokales v2-Profil |
| Gemma 4 E4B IT, INT4 | Lokale Laufzeit unterstützt; gleichwertiger v2-Vergleich steht aus |
| Amazon Nova 2.5 Sonic | Native Audioverarbeitung validiert; Retail pass^1 **1/1** (ein v6-Entwicklungslauf, Limit 420 s; [Ergebnis](docs/results/nova25-v6-retail-development.json)) |
| OpenAI GPT-Live + ein benannter Backend-Agent | [Experimenteller Adapter offline getestet; Live-Validierung steht aus](docs/providers.md) |
| OpenAI Realtime | [Experimenteller Adapter offline getestet; Live-Validierung steht aus](docs/providers.md) |
| Gemini Live | [Experimenteller Adapter offline getestet; Live-Validierung steht aus](docs/providers.md) |
| ElevenLabs Agents | [Experimenteller Adapter offline getestet; Live-Validierung steht aus](docs/providers.md) |

Die lokale Pipeline nutzt [speech-swift](https://github.com/soniqo/speech-swift): vollständiges Whisper Large-v3 FP16, VoxCPM2-Hauptgewichte FP16 / AudioVAE FP32, Silero VAD und Gemma E4B für Anrufer, Zuhörhilfen und Bewertung. Nova 2.5 Sonic hat einen optionalen [Runner für natives Audio](docs/nova.md); Anrufer und Bewertung bleiben lokal.

## Vergleichsmetriken

| Metrik | Messgröße |
| --- | --- |
| Aufgabenerfolg, pass^1 | Erforderlicher Datenbankendzustand und Kommunikation |
| Abdeckung und Fehler | Bewertete/geplante Versuche, normale Gesprächsenden und unbewertete Inferenzfehler |
| Latenz, p50 / p95 | Ausführungszeit von Agenten-LLM und ASR; TTS-Zeit bis zum ersten Audio |
| Verzögerung der Sprachantwort | Ende der Anruferäußerung bis zum ersten Agentenaudio; standardisierte Zusammenfassung steht aus |
| Werkzeugfehlerrate | Fehlgeschlagene oder abgelehnte Agentenaufrufe / versuchte Aufrufe |
| Abschlusszeit | Simulierte Gesprächszeit und gemessene reale Zeit getrennt |
| Maximaler Speicherbedarf | Physischer Speicherbedarf des nativen Prozesses mit Messumfang |

Vergleichen Sie dieselben Aufgaben, Startwerte, Gesprächsgrenzen, Akustikbedingungen und Anrufer-/Bewertungseinstellungen. Halten Sie die Sprachpipeline beim Vergleich lokaler LLMs konstant. Audionative und verwaltete Agenten müssen ihre gesamte Pipeline angeben. Komponentenlaufzeiten sind von der gesamten Sprachantwortverzögerung zu unterscheiden.

## Aufgezeichnete Entwicklungsergebnisse

Ein Entwicklungsversuch pro Zeile. Aufgaben und Zeitmodelle unterscheiden sich; eine Modellrangfolge unter gleichen Bedingungen steht aus.

| Agent | Bedingung / Aufgabe | Bewertet / geplant | Pass^1 | Werkzeugfehler | Reale Dauer |
| --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | v2 · Flug 12 | 1/1 | 0/1 | 1/6 | 554.4 s |
| Amazon Nova 2.5 Sonic | v6 · Handel 0 | 1/1 | **1/1** | **0/5** | **296.3 s** |

Flugaufgabe 12, ein Versuch, Startwert 70, v2; höchstens **600 simulierte Sekunden / 1.200 reale Sekunden**. Das Gespräch endete normal nach **312,9 simulierten Sekunden / 554,4 gemessenen realen Sekunden**. Hardware: Apple M5 Pro, 48 GB, gemeinsam genutzter Rechner. Dies ist ein Entwicklungsversuch; eine Rangfolge mehrerer Modelle unter gleichen Bedingungen steht aus. [Daten und Messdefinitionen](docs/results/gemma4-12b-local-v2-airline.json).

Gemma LLM p50 / p95: **7.00 / 11.13 s**; TTS p50 / p95: **1.22 / 1.70 s**.

Nova: ein Versuch, Startwert 70, control-Akustik, **420 s Echtzeitlimit**. Beide gewünschten Artikel wurden umgetauscht; alle **5/5** Referenzaktionen und der endgültige Datenbankzustand stimmten überein. Das Gespräch endete normal nach **259,0 s Audio / 296,3 s realer Zeit**. [Ergebnis und Grenzen](docs/results/nova25-v6-retail-development.json).

## Lokal ausführen

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Folgen Sie der [Einrichtung](docs/runtime-setup.md) und [v2-Ausführung](docs/local-v2.md), um den lokalen Dienst zu starten, Anrufer/Bewerter zu qualifizieren und Aufgaben auszuführen.

```bash
uv run --frozen --extra dev python scripts/run_local_voice.py \
  --output data/simulations/first-local-run
```

Dieser Befehl baut die festgelegte native Laufzeit, lädt die Modelldateien herunter, kalibriert Anrufer und Bewertung und führt einen kontrollierten Sprachtest aus. Er erzeugt keinen Domänen-Benchmarkwert. Verwenden Sie ein neues Ausgabeverzeichnis. [Einrichtung und Aufgabenläufe](docs/runtime-setup.md#one-command-local-run).

Begrenzen Sie die Ausführung mit `--num-tasks`, `--trials`, `--max-seconds` und `--wall-timeout`. Der Standardlauf bleibt lokal. Optionale Runner: [Nova](docs/nova.md) · [OpenAI, Gemini und ElevenLabs](docs/providers.md).

Prüfungen: `make test` · `make lint` · `make verify`.

Quelle: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
