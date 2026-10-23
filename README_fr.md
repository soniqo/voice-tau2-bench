# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Benchmark autonome d’**agents vocaux τ-Voice**, avec inférence locale via [speech-swift](https://github.com/soniqo/speech-swift). Évalue conversations de service client, utilisation d’outils et résolution de tâches sur Apple Silicon.

Le cadre anglais figé `voice-v1` contient **278 tâches** : aérien **50**, commerce **114**, télécommunications **114**. Tâches, règles, outils, bases et critères de récompense conservent leurs empreintes d’origine.

## Modèles locaux

| Rôle | Modèle | Précision |
| --- | --- | --- |
| Agent, appelant, écoute et juge sémantique | Gemma 4 E4B IT | INT4 |
| Voix de l’agent et de l’appelant | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Reconnaissance vocale | Whisper Large-v3 (full, CoreML) | FP16 |
| Détection de parole | Silero VAD (CoreML) | Native |

Les modèles sont téléchargés à la préparation ; l’inférence ne nécessite ni fournisseur cloud ni clé API. La [configuration du service](docs/runtime-setup.md) décrit la compilation de speech-swift et les fichiers requis.

## Résultats mesurés

Échantillon de développement : première tâche figée par domaine, **3 tâches** au total, **1 essai**, graine **42**, délai de conversation de **600 secondes**, téléphonie propre `control`. Matériel : **Apple M5 Pro, 48 Go de mémoire unifiée**. La réussite évalue la pile complète ci-dessus.

`soniqo-local-v1` · `2026-10-23`

| Modèle de l’agent | Aérien | Commerce | Télécoms | pass^1 global |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Ce petit échantillon déterministe utilise un appelant et un juge locaux. Il ne constitue ni le score des 278 tâches ni un résultat officiel. Un dépassement de délai vaut zéro ; les exceptions d’inférence restent non notées.

## Démarrage rapide

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

## Vérification et évaluation

```bash
make test
make lint
make verify
```

Voir les [règles d’évaluation](docs/evaluation.md), l’[architecture](docs/local-voice-extraction.md) et le [relevé du modèle](docs/results/gemma4-e4b-local-600.json). Les cas mock et contrôles de composants sont des vérifications de développement, exclues du score.

## Attribution

Dérivé de [τ²-Bench de Sierra Research](https://github.com/sierra-research/tau2-bench), commit `05736c27219b769d0be507574dd3135010916644`. La [licence MIT](LICENSE) est conservée. Citez l’[article τ-Voice](https://arxiv.org/abs/2603.13686) lors de l’utilisation.
