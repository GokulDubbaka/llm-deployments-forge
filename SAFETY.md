# Safety & Usage Policy

## Purpose

`llm-deployments-forge` is a **research and red-team evaluation framework**.
It benchmarks how large language models respond to adversarial, ambiguous, and
potentially harmful prompts — with the explicit goal of measuring refusal
behaviour and improving safety guardrails.

## What the prompt library contains

The `prompt_library.py` module contains prompts across the following categories:

| Category | Count | Purpose |
|---|---|---|
| `web_security` | 8 | Evaluating refusals on SQLi, XSS, SSRF |
| `network` | 5 | Evaluating refusals on port scanning, MITM |
| `injection` | 3 | Prompt injection and jailbreak resistance |
| `malware` | 4 | Evaluating refusals on malware generation |
| `social_eng` | 2 | Phishing and impersonation refusal tests |
| `cryptography` | 2 | Cipher misuse and key exposure scenarios |
| `opsec` | 2 | Identity concealment and evasion scenarios |

**These prompts are benchmarks, not instructions.** They are fed to an LLM
evaluator that measures whether the model under test refuses, hedges, or
complies — and scores the response accordingly.

## Permitted use

✅ Academic research on LLM safety and alignment  
✅ Internal red-teaming of your own deployed models  
✅ Building and evaluating custom refusal classifiers  
✅ Comparing refusal behaviour across model versions  

## Prohibited use

❌ Generating real attack payloads against systems you do not own  
❌ Bypassing safety guardrails in production LLM deployments  
❌ Using benchmark outputs as operational attack instructions  
❌ Any use that violates applicable law or the terms of service of the
  underlying LLM provider  

## Output containment

All model responses are stored locally and are never transmitted to third
parties. The evaluator scores responses on a 0–1 refusal scale — it does
**not** extract or execute any code or instructions present in the output.

## Responsible disclosure

If you discover a genuine safety issue with this framework, please open a
GitHub issue or email the maintainer directly.
