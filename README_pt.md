# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

Benchmark independente de **agentes de voz τ-Voice**, com inferência local via [speech-swift](https://github.com/soniqo/speech-swift). Avalia conversas de atendimento, uso de ferramentas e conclusão de tarefas no Apple Silicon.

O conjunto inglês fixo `voice-v1` contém **278 tarefas**: aéreo **50**, varejo **114**, telecomunicações **114**. Tarefas, políticas, ferramentas, bancos de dados e recompensas mantêm suas referências originais.

## Modelos locais

| Função | Modelo | Precisão |
| --- | --- | --- |
| Agente, chamador, escuta e juiz semântico | Gemma 4 E4B IT | INT4 |
| Vozes do agente e do chamador | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Reconhecimento de voz | Whisper Large-v3 (full, CoreML) | FP16 |
| Detecção de fala | Silero VAD (CoreML) | Native |

Os modelos são baixados na preparação; a inferência não usa nuvem nem chaves API. A [configuração do serviço](docs/runtime-setup.md) explica a compilação de speech-swift e os arquivos necessários.

## Resultados medidos

Amostra de desenvolvimento: primeira tarefa fixa por domínio, **3 tarefas** no total, **1 tentativa**, semente **42**, prazo de conversa de **600 segundos**, telefonia limpa `control`. Hardware: **Apple M5 Pro, 48 GB de memória unificada**. O sucesso avalia toda a pilha acima.

`soniqo-local-v1` · `2026-10-23`

| Modelo do agente | Aéreo | Varejo | Telecom | pass^1 geral |
| --- | --- | --- | --- | --- |
| Gemma 4 E4B IT (INT4) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |

Esta amostra pequena e determinista usa chamador e juiz locais. Não representa a pontuação das 278 tarefas nem uma classificação oficial. Timeout vale zero; exceções de inferência ficam sem pontuação.

## Início rápido

Use Python **3.12/3.13**, Apple Silicon e **macOS 15+**. Inicie o [serviço local](docs/runtime-setup.md#serve) antes de executar tarefas.

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

Cada tarefa salva trajetória, ferramentas, recompensas, dois WAV, tempos, configurações e impressões das fontes/modelos. Continue o mesmo lote com `--resume`. Omita `--num-tasks` para avaliar as 278 tarefas.

## Verificação e avaliação

```bash
make test
make lint
make verify
```

Veja as [regras de avaliação](docs/evaluation.md), a [arquitetura](docs/local-voice-extraction.md) e o [registro do modelo](docs/results/gemma4-e4b-local-600.json). Casos mock e testes de componentes são verificações de desenvolvimento, fora da pontuação.

## Atribuição

Derivado de [τ²-Bench da Sierra Research](https://github.com/sierra-research/tau2-bench), commit `05736c27219b769d0be507574dd3135010916644`. Mantém a [licença MIT](LICENSE). Cite o [artigo τ-Voice](https://arxiv.org/abs/2603.13686) ao usar o benchmark.
