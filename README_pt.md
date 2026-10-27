# voice-tau2-bench

[English](README.md) · [简体中文](README_zh.md) · [日本語](README_ja.md) · [한국어](README_ko.md) · [Español](README_es.md) · [Deutsch](README_de.md) · [Français](README_fr.md) · [हिन्दी](README_hi.md) · [Português](README_pt.md) · [Русский](README_ru.md) · [العربية](README_ar.md) · [ไทย](README_th.md) · [Türkçe](README_tr.md) · [Tiếng Việt](README_vi.md)

A [Soniqo](https://github.com/soniqo) mantém este benchmark para medir se agentes de voz concluem tarefas por meio de conversas faladas e ferramentas. O conjunto inglês congelado contém **278 tarefas**: aviação **50**, varejo **114**, telecomunicações **114**.

[Configuração](docs/runtime-setup.md) · [Pontuação](docs/evaluation.md) · [Comparação de modelos e limites](docs/model-comparison.md)

## Modelos e sistemas

| Agente ou sistema | Suporte atual |
| --- | --- |
| Gemma 4 12B IT, INT4 | Perfil local v2 testado |
| Gemma 4 E4B IT, INT4 | Runtime local compatível; comparação v2 em condições equivalentes pendente |
| OpenAI GPT-Live + um agente backend identificado | Requer adaptador |
| OpenAI Realtime | Requer adaptador |
| Gemini Live | Requer adaptador |
| ElevenLabs Agents | Requer adaptador |

A pilha local usa [speech-swift](https://github.com/soniqo/speech-swift): Whisper Large-v3 completo FP16, pesos principais do VoxCPM2 FP16 / AudioVAE FP32, Silero VAD e Gemma E4B como interlocutor, auxiliares de escuta e avaliador. Os sistemas em nuvem não foram integrados nem avaliados aqui.

## Métricas de comparação

| Métrica | O que mede |
| --- | --- |
| Sucesso da tarefa, pass^1 | Estado final exigido do banco de dados e comunicação |
| Cobertura e falhas | Tentativas avaliadas/planejadas, encerramentos normais e erros de inferência sem pontuação |
| Latência, p50 / p95 | Execução do LLM e ASR do agente; tempo de TTS até o primeiro áudio |
| Atraso de resposta de voz | Fim da fala do interlocutor até o primeiro áudio do agente; resumo padronizado pendente |
| Taxa de erro das ferramentas | Chamadas do agente com falha ou rejeitadas / chamadas tentadas |
| Tempo de conclusão | Tempo simulado da chamada e tempo real medido, separadamente |
| Pico de memória | Memória física do processo nativo, com escopo de medição |
| Custo de API | USD por tentativa e por tentativa bem-sucedida; incluir falhas |

Compare as mesmas tarefas, sementes, limites, condições acústicas e configuração do interlocutor/avaliador. Mantenha a pilha de voz fixa ao comparar LLMs locais. Agentes de áudio nativo e gerenciados devem identificar a pilha completa. Os tempos dos componentes são separados da latência de resposta de voz de ponta a ponta.

## Diagnóstico local registrado

| Agente | Avaliadas / planejadas | Pass^1 | Erros de ferramentas | LLM p50 / p95 | Primeiro áudio TTS p50 / p95 | Custo API / tentativa |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 12B IT, INT4 | 1/1 | 0/1 | 1/6 | 7.00 / 11.13 s | 1.22 / 1.70 s | $0 |

Tarefa de aviação 12, uma tentativa, semente 70, v2; máximo de **600 segundos simulados / 1.200 segundos reais**. A chamada terminou normalmente após **312,9 segundos simulados / 554,4 segundos reais medidos**. Hardware: Apple M5 Pro, 48 GB, estação compartilhada. É uma tentativa de desenvolvimento; a classificação de vários modelos em condições equivalentes ainda está pendente. [Dados e definições de medição](docs/results/gemma4-12b-local-v2-airline.json).

## Execução local

Python **3.12/3.13**, Apple Silicon, **macOS 15+**. Siga a [configuração](docs/runtime-setup.md) e a [execução v2](docs/local-v2.md) para iniciar o serviço local, qualificar o interlocutor/avaliador e executar tarefas.

```bash
uv sync --frozen --extra dev
uv run --frozen tau-voice verify
```

Limite a execução com `--num-tasks`, `--trials`, `--max-seconds` e `--wall-timeout`. Adaptadores de nuvem e um teto de gastos em USD **não estão implementados**. A inferência atual é local e não exige chave de API de nuvem.

Verificações: `make test` · `make lint` · `make verify`.

Origem: [τ²-Bench](https://github.com/sierra-research/tau2-bench) (`05736c27219b769d0be507574dd3135010916644`) · [τ-Voice](https://arxiv.org/abs/2603.13686) · [MIT](LICENSE).
