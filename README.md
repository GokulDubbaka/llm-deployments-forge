# 🧪 LLM Deployments Forge — Safety Red-Team Benchmark CLI

> **Status:** Working benchmark harness · Seeking more attack prompt categories and LLM backends

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> ℹ️ **Purpose:** This tool is for *defensive* AI safety research — measuring how well models resist adversarial prompts, not for generating harmful content.

---

## 🎯 What This Is

A **CLI benchmark harness** for evaluating LLM safety boundaries. Send structured adversarial prompts across multiple attack categories to any local or API-hosted LLM backend. Measure refusal rates, hallucination frequency, and injection resistance.

**Research context:** As LLMs are deployed in agents (like AegisTwin), understanding their failure modes becomes critical for safe deployment. This tool provides reproducible, quantitative safety measurements.

---

## ✅ What Actually Works

```bash
pip install -r requirements.txt
python src/cli.py run --backend openai --model gpt-4o-mini
python src/cli.py run --backend ollama --model llama3
python src/cli.py run --category jailbreak --verbose
python src/cli.py list-categories
```

**Attack categories implemented:**
- `jailbreak` — Prompt injection attempts to override system instructions
- `role_confusion` — Attempts to make the model claim a different identity
- `data_extraction` — Probes for training data memorisation
- `instruction_override` — Tries to replace safety guidelines mid-conversation
- `hallucination_induction` — Tests confidence calibration on false premises

**Outputs:** Per-category pass/fail rates, response latency, refusal quality scoring (1–5 scale), exportable JSON results.

---

## ❌ What We Have NOT Yet Achieved

### 1. Claude Mythos-Style Epistemic Consistency Testing
**Claude Mythos** (as we understand it) refers to AI agents that maintain consistent self-models — knowing what they know, what they don't know, and being honest about uncertainty across multi-turn conversations.

Testing this requires:
- Multi-turn conversation harnesses (we only test single-turn)
- Consistency metrics across 10–50 turn conversations
- Automated contradiction detection

We have the framework but not the multi-turn evaluation logic.

### 2. Coverage of Real Attack Surfaces
Our prompt library has ~50 adversarial prompts. Research-grade benchmarks (like HarmBench, SALAD-Bench) have 1,000+ carefully categorised attacks. Our coverage is thin.

### 3. Model Comparison Dashboard
No visual comparison of safety scores across models/versions. Currently text-only CLI output.

### 4. Fine-tuned Evaluator
The refusal quality scorer uses simple heuristics. A proper evaluator would be an LLM-as-judge system calibrated against human expert ratings.

---

## 🤝 How You Can Help

- **Prompt library expansion:** Contribute new adversarial prompts in existing or new categories (see `src/core/prompt_library.py`)
- **Multi-turn harness:** Build a conversation simulator that maintains context across N turns and checks for consistency drift
- **LLM-as-judge evaluator:** Implement a secondary LLM that scores primary model responses for safety quality
- **More backends:** Add support for Anthropic Claude, Google Gemini, Mistral, and local GGUF models via llama.cpp
- **Dashboard:** Build a web UI showing model safety scores over time as models are updated

---

## 📄 License

MIT — see [LICENSE](LICENSE)
