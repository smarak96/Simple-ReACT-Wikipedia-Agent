# ReAct vs. Chain-of-Thought

A minimal implementation of **ReAct** ([Yao et al., 2022](https://arxiv.org/abs/2210.03629)) — *Reasoning + Acting* — compared against plain Chain-of-Thought (CoT) prompting.

The agent interleaves **Thought → Action → Observation** steps, using live Wikipedia search/lookup as tools instead of relying solely on the model's frozen knowledge.

## Why it matters

The image below runs the same question — *"Which team won IPL 2026?"* — through both approaches:

![ReAct vs CoT comparison](reactcompare.png)

- **ReAct** searches Wikipedia, reads the result, and answers correctly: *Royal Challengers Bengaluru*.
- **CoT** hits its knowledge cutoff (Dec 2023) and refuses, since IPL 2026 happened after training.

Grounding reasoning in external tools lets the model answer questions beyond its training data.

## How it works

- `WikiEnv` — a tool environment exposing `search[entity]` and `lookup[keyword]` against live Wikipedia.
- `run_react()` — the ReAct loop: prompt the LLM, parse its `Action`, execute it, feed back the `Observation`, repeat until `finish[answer]`.
- `run_cot()` — baseline that reasons step-by-step with no tools.

## Setup

```bash
pip install groq python-dotenv requests beautifulsoup4
```

Create a `.env` file with your [Groq](https://console.groq.com) key:

```
GROQ_API_KEY=your_key_here
```

## Run

```bash
python react.py
```

Model: `llama-3.3-70b-versatile` via the Groq API.
