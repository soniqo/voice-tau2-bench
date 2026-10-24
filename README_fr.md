# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) maintient ce benchmark autonome τ-Voice pour mesurer les **performances des agents vocaux locaux** sur Apple Silicon. Nous exécutons l’agent, l’appelant simulé, les modèles vocaux et le juge sémantique avec [speech-swift](https://github.com/soniqo/speech-swift).

## Documentation

- [Configuration et fichiers des modèles](docs/runtime-setup.md)
- [Évaluation et notation](docs/evaluation.md)
- [Implémentation](docs/local-voice-extraction.md)
- [Compte rendu de l’exécution](docs/results/gemma4-e4b-local-600.json)
- [Exécution de Gemma 4 12B et paramètres validés](docs/validation/gemma4-12b-voice-e2e.md)

## Ce que nous mesurons

Nous mesurons si un agent vocal accomplit des tâches de service client par la conversation et l’utilisation d’outils. Les récompenses vérifient l’état final attendu de la base de données et les informations communiquées.

Le cadre anglais figé `voice-v1` contient **278 tâches** : aérien **50**, commerce **114**, télécommunications **114**. Tâches, règles, outils, bases et critères de récompense conservent leurs empreintes d’origine.

## Modèles testés et résultats

| Rôle | Modèle | Précision |
| --- | --- | --- |
| Agent | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| Appelant, écoute et juge sémantique | Gemma 4 E4B IT | INT4 |
| Voix de l’agent et de l’appelant | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Reconnaissance vocale | Whisper Large-v3 (full, CoreML) | FP16 |
| Détection de parole | Silero VAD (CoreML) | Native |

Échantillon de développement : première tâche figée par domaine, **3 tâches** au total, **1 essai**, graine **42**, délai de conversation de **600 secondes**, téléphonie propre `control`. Matériel : **Apple M5 Pro, 48 Go de mémoire unifiée**. La réussite évalue la pile complète ci-dessus.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| Modèle de l’agent | Aérien | Commerce | Télécoms | pass^1 global |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

Ce petit échantillon déterministe utilise un appelant et un juge locaux. Il ne constitue ni le score des 278 tâches ni un résultat officiel. Un dépassement de délai vaut zéro ; les exceptions d’inférence restent non notées.

L’échantillon 12B est incomplet : le test télécom a rencontré une erreur d’inférence native et reste non noté. Test vocal contrôlé avec 12B : **1.0** pour l’écriture correcte en base et **0.0** lors de la réévaluation d’une écriture erronée. [Paramètres validés](docs/validation/gemma4-12b-voice-e2e.md). La demande de l’appelant est fixe et ce test ne contribue pas au score du benchmark. Les essais par domaine utilisent l’appelant génératif ; les rapports liés indiquent chaque révision du runtime.

Les reprises V2 restent diagnostiques : un refus dans le domaine aérien a obtenu **1.0** pour une base inchangée ; le commerce a dépassé le délai et les télécommunications ont révélé des défauts de contrôle de l’appelant. Le **pilote réservé de 18 appels est incomplet**. Ces essais ne permettent pas d’établir un taux de réussite global du modèle. [Résultats et paramètres](docs/validation/caller-rerun.md).

`soniqo-local-gemma12b-v2` · `--max-seconds 600 --wall-timeout 1200`

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
