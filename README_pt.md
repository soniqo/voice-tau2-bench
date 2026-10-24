# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

A [Soniqo](https://github.com/soniqo) mantém este benchmark independente de τ-Voice para medir o **desempenho de agentes de voz locais** no Apple Silicon. Executamos o agente, o interlocutor simulado, os modelos de fala e o avaliador semântico com [speech-swift](https://github.com/soniqo/speech-swift).

## Documentação

- [Configuração e arquivos dos modelos](docs/runtime-setup.md)
- [Avaliação e pontuação](docs/evaluation.md)
- [Implementação](docs/local-voice-extraction.md)
- [Registro da execução](docs/results/gemma4-e4b-local-600.json)
- [Execução do Gemma 4 12B e parâmetros validados](docs/validation/gemma4-12b-voice-e2e.md)

## O que medimos

Medimos se um agente de voz conclui tarefas de atendimento ao cliente por meio de conversa e uso de ferramentas. As recompensas verificam o estado final exigido da base de dados e as informações comunicadas.

O conjunto inglês fixo `voice-v1` contém **278 tarefas**: aéreo **50**, varejo **114**, telecomunicações **114**. Tarefas, políticas, ferramentas, bancos de dados e recompensas mantêm suas referências originais.

## Modelos testados e resultados

| Função | Modelo | Precisão |
| --- | --- | --- |
| Agente | Gemma 4 E4B IT / Gemma 4 12B IT | INT4 |
| Chamador, escuta e juiz semântico | Gemma 4 E4B IT | INT4 |
| Vozes do agente e do chamador | VoxCPM2 | FP16 main weights; FP32 AudioVAE |
| Reconhecimento de voz | Whisper Large-v3 (full, CoreML) | FP16 |
| Detecção de fala | Silero VAD (CoreML) | Native |

Amostra de desenvolvimento: primeira tarefa fixa por domínio, **3 tarefas** no total, **1 tentativa**, semente **42**, prazo de conversa de **600 segundos**, telefonia limpa `control`. Hardware: **Apple M5 Pro, 48 GB de memória unificada**. O sucesso avalia toda a pilha acima.

`soniqo-local-v1` · `soniqo-local-gemma12b-v1` · `2026-10-23`

| Modelo do agente | Aéreo | Varejo | Telecom | pass^1 geral |
| --- | --- | --- | --- | --- |
| [Gemma 4 E4B IT (INT4)](docs/results/gemma4-e4b-local-600.json) | 0/1 | 0/1 | 0/1 | 0.0% (0/3) |
| [Gemma 4 12B IT (INT4)](docs/results/gemma4-12b-local-600.json) | 0/1 | 0/1 | — | 0.0% (0/2) |

Esta amostra pequena e determinista usa chamador e juiz locais. Não representa a pontuação das 278 tarefas nem uma classificação oficial. Timeout vale zero; exceções de inferência ficam sem pontuação.

A amostra de 12B está incompleta: telecomunicações teve um erro de inferência nativa e ficou sem pontuação. Teste de voz controlado com 12B: **1.0** para a escrita correta na base de dados e **0.0** na reavaliação de uma escrita incorreta. [Parâmetros validados](docs/validation/gemma4-12b-voice-e2e.md). A entrada do chamador é fixa; este teste não contribui para a pontuação do benchmark. As execuções por domínio usam o chamador generativo e os relatórios ligados registam cada revisão do runtime.

Verificação V2: **Gemma 4 12B IT (INT4) obteve 1.0 em um diagnóstico de recusa no domínio aéreo** (tarefa `0`, semente `42`, espera do cliente de 15 segundos). Essa tarefa verifica que o banco de dados permanece inalterado. Varejo, telecomunicações e o piloto reservado de 18 chamadas ainda não foram executados nessa condição. [Validação e parâmetros](docs/validation/native-prefill.md).

`soniqo-local-gemma12b-v2` · `--max-seconds 600 --wall-timeout 1200`

## Execução local

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

Verificações: `make test` · `make lint` · `make verify`.

Fonte: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [Artigo do τ-Voice](https://arxiv.org/abs/2603.13686) · [Licença MIT](LICENSE).
