# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

[Soniqo](https://github.com/soniqo) maintient ce benchmark pour mesurer si les agents vocaux accomplissent des tâches par la conversation et les outils. Le corpus anglais figé comprend **278 tâches** : aérien **50**, commerce **114**, télécoms **114**.

[Installation](docs/runtime-setup.md) · [Évaluation](docs/evaluation.md) · [Comparaison des modèles et limites](docs/model-comparison.md)

## Modèles et systèmes

| Agent ou système | Prise en charge actuelle |
| --- | --- |
| Gemma 4 12B IT, INT4 | Profil local v2 testé |
| Gemma 4 E4B IT, INT4 | Runtime local compatible ; comparaison v2 à conditions égales en attente |
| OpenAI GPT-Live + un agent backend identifié | Adaptateur nécessaire |
| OpenAI Realtime | Adaptateur nécessaire |
| Gemini Live | Adaptateur nécessaire |
| ElevenLabs Agents | Adaptateur nécessaire |

La chaîne locale utilise [speech-swift](https://github.com/soniqo/speech-swift) : Whisper Large-v3 complet FP16, poids principaux VoxCPM2 FP16 / AudioVAE FP32, Silero VAD et Gemma E4B pour l’appelant, les aides d’écoute et le juge. Les systèmes cloud ne sont ni intégrés ni évalués ici.

## Métriques de comparaison

| Métrique | Ce qu’elle mesure |
| --- | --- |
| Réussite de tâche, pass^1 | État final requis de la base et communication |
| Couverture et échecs | Essais évalués/prévus, fins normales et erreurs d’inférence non évaluées |
| Latence, p50 / p95 | Exécution LLM et ASR de l’agent ; délai TTS jusqu’au premier audio |
| Délai de réponse vocale | Fin de parole de l’appelant jusqu’au premier audio de l’agent ; agrégat standardisé en attente |
| Taux d’erreur des outils | Appels d’outils de l’agent échoués ou rejetés / appels tentés |
| Durée d’achèvement | Durée simulée de l’appel et durée réelle mesurée, séparément |
| Mémoire maximale | Empreinte physique du processus natif, avec périmètre de mesure |
| Coût API | USD par essai et par essai réussi ; inclure les échecs |

Comparez les mêmes tâches, graines, limites, conditions acoustiques et réglages appelant/juge. Gardez la chaîne vocale fixe pour comparer les LLM locaux. Les agents audio natifs et gérés doivent préciser leur chaîne complète. Les temps des composants sont distincts du délai de réponse vocale de bout en bout.

## Diagnostic local enregistré

| Agent | Évalués / prévus | Pass^1 | Erreurs d’outils | LLM p50 / p95 | Premier audio TTS p50 / p95 | Coût API / essai |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

Tâche aérienne 12, un essai, graine 70, v2 ; maximum de **600 secondes simulées / 1 200 secondes réelles**. L’appel s’est terminé normalement après **312,9 secondes simulées / 554,4 secondes réelles mesurées**. Matériel : Apple M5 Pro, 48 Go, poste partagé. Il s’agit d’un essai de développement ; un classement de plusieurs modèles à conditions égales reste à établir. [Données et définitions des mesures](docs/results/gemma4-12b-local-v2-airline.json).

## Exécution locale

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Suivez l’[installation](docs/runtime-setup.md) et l’[exécution v2](docs/local-v2.md) pour démarrer le service local, qualifier l’appelant/juge et exécuter les tâches.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

Limitez l’exécution avec `--num-tasks`, `--trials`, `--max-seconds` et `--wall-timeout`. Les adaptateurs cloud et un plafond de dépense en USD **ne sont pas implémentés**. L’inférence actuelle est locale et ne nécessite aucune clé API cloud.

Vérifications: `make test` · `make lint` · `make verify`.

Source: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
