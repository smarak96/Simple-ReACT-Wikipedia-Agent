# ReAct vs. Chain-of-Thought

A minimal implementation of **ReAct** ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)) — *Reasoning + Acting* — compared against plain Chain-of-Thought (CoT) prompting.

The agent interleaves **Thought → Action → Observation** steps, using live Wikipedia search/lookup as tools instead of relying solely on the model's frozen knowledge.

## Why it matters

The image below runs the same question — *"Which team won IPL 2026?"* — through both approaches:

![ReAct vs CoT comparison](reactcompare.png)

- **ReAct** searches Wikipedia, reads the result, and answers correctly: *Royal Challengers Bengaluru*.
- **CoT** hits its knowledge cutoff (Dec 2023) and refuses, since IPL 2026 happened after training.

Grounding reasoning in external tools lets the model answer questions beyond its training data.

## Evaluation on HotpotQA

Evaluated on 100 [HotpotQA](https://hotpotqa.github.io/) questions:

| Metric | Score |
|--------|-------|
| Exact Match (EM) | 41.0% |
| F1 | 49.4% |

Main failure mode: giving up on hard multi-hop questions rather than hallucinating — the agent tends to emit `finish[unknown]` when it can't find a clear answer after a few search iterations.

## How it works

- `WikiEnv` — a tool environment exposing `search[entity]` and `lookup[keyword]` against live Wikipedia.
- `run_react()` — the ReAct loop: prompt the LLM, parse its `Action`, execute it, feed back the `Observation`, repeat until `finish[answer]`.
- `run_cot()` — baseline that reasons step-by-step with no tools.

---

## Reflexion + ReAct

An implementation of **Reflexion** ([Shinn et al., 2023](https://arxiv.org/abs/2303.11366)) built on top of the ReAct agent. When the agent fails a question, a reflection LLM diagnoses what went wrong and suggests a better search strategy. That critique is fed into the next attempt as context.

### Memory mechanism

Reflections are stored in a sliding window of the last 4 entries. After each failed trial the self-critique is appended and injected into the next ReAct prompt. Older reflections beyond the window are dropped to avoid bloating the context, so the agent carries forward lessons from recent failures without accumulating the entire history.

### Evaluation on HotpotQA

Same 100-question subset, same model (Llama 3.3 70B):

| Agent | EM | F1 |
|---|---|---|
| ReAct | 44.0% | 55.6% |
| Reflexion + ReAct | 52.0% | 52.0% |

**+8 percentage points EM gained purely from self-reflection.**

The F1 inversion (ReAct F1 > Reflexion F1 despite lower EM) is an artifact: ReAct's wrong answers often have partial token overlap with the gold answer, while Reflexion returns empty after exhausting all trials, giving F1 = 0 on failures. EM is the right metric here.

### Trial breakdown

| Trial | Questions solved | Notes |
|---|---|---|
| 0 | 43 | Pure ReAct, no reflection used |
| 1 | 5 | Rescued by first reflection |
| 2 | 2 | Rescued by second reflection |
| 3 | 2 | Rescued by third reflection |
| Unsolved | 48 | All 4 trials exhausted |

Diminishing returns are visible exactly as the paper predicts. Most of the gain comes from trial 1. The agent either recovers quickly from a good reflection or fails to recover at all.

### Regression analysis

Cross-referencing both result files:

| | ReAct correct | ReAct wrong |
|---|---|---|
| **Reflexion correct** | 42 | 10 |
| **Reflexion wrong** | 2 | 46 |

- **10 questions gained** by Reflexion that ReAct could not solve
- **2 regressions** where ReAct was correct but Reflexion's trial 0 returned a slight variation, failed exact match, and reflection could not recover
- **Net: +8 questions**

The 2 regressions point to the stopping criterion being slightly too strict. If trial 0 returns a near-correct answer that fails exact match, reflection is triggered unnecessarily.

---

## Setup

```bash
pip install openai groq python-dotenv requests beautifulsoup4
```

The agent supports two backends — set whichever key you have in a `.env` file:

**OpenRouter** (used for evaluation):
```
OPENROUTER_API_KEY=your_key_here
```

**Groq** (alternative):
```
GROQ_API_KEY=your_key_here
```

Get keys at [openrouter.ai](https://openrouter.ai) or [console.groq.com](https://console.groq.com).

## Run

```bash
python react.py
```

Default model: `meta-llama/llama-3.3-70b-instruct` via OpenRouter (`llama-3.3-70b-versatile` on Groq).
