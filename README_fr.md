# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) maintient cette version dérivée de τ-Voice pour mesurer les **performances des agents vocaux locaux** sur Apple Silicon. Nous exécutons l’agent, l’appelant simulé, les modèles vocaux et le juge sémantique avec [speech-swift](https://github.com/soniqo/speech-swift).

## Documentation

- [Configuration et fichiers des modèles](docs/runtime-setup.md)
- [Évaluation et notation](docs/evaluation.md)
- [Implémentation](docs/local-voice-extraction.md)
- [Compte rendu de l’exécution](docs/results/gemma4-e4b-local-600.json)

## Ce que nous mesurons

Nous mesurons si un agent vocal accomplit des tâches de service client par la conversation et l’utilisation d’outils. Les récompenses vérifient l’état final attendu de la base de données et les informations communiquées.

Le cadre anglais figé `voice-v1` contient **278 tâches** : aérien **50**, commerce **114**, télécommunications **114**. Tâches, règles, outils, bases et critères de récompense conservent leurs empreintes d’origine.

## Modèles testés et résultats

| Rôle | Modèle | Précision |
| --- | --- | --- |
| Agent, appelant, écoute et juge sémantique | Gemma 4 E4B IT | INT4 |
| Voix de l’agent et de l’appelant | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Reconnaissance vocale | Whisper Large-v3 (full, CoreML) | FP16 |
| Détection de parole | Silero VAD (CoreML) | Native |

Échantillon de développement : première tâche figée par domaine, **3 tâches** au total, **1 essai**, graine **42**, délai de conversation de **600 secondes**, téléphonie propre `control`. Matériel : **Apple M5 Pro, 48 Go de mémoire unifiée**. La réussite évalue la pile complète ci-dessus.

`soniqo-local-v1` · `2026-10-23`

| Modèle de l’agent | Aérien | Commerce | Télécoms | pass^1 global |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Ce petit échantillon déterministe utilise un appelant et un juge locaux. Il ne constitue ni le score des 278 tâches ni un résultat officiel. Un dépassement de délai vaut zéro ; les exceptions d’inférence restent non notées.

## Exécution locale

Utilisez Python **3.12/3.13**, Apple Silicon et **macOS 15+**. Lancez le [service local](docs/runtime-setup.md#serve) avant les tâches.

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

Chaque tâche conserve trajectoire, outils, récompenses, deux WAV, durées, paramètres et empreintes des sources/modèles. Reprenez le même lot avec `--resume`. Omettez `--num-tasks` pour les 278 tâches.

Vérifications: `make test` · `make lint` · `make verify`.

Source: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [Article τ-Voice](https://arxiv.org/abs/2603.13686) · [Licence MIT](LICENSE).
