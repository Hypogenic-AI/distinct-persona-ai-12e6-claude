# Resources Catalog — Distinct Persona

This document catalogs all resources gathered for the Distinct Persona research project. See
[`literature_review.md`](literature_review.md) for synthesis and experimental recommendations.

---

## Summary

| Category | Count | Location |
|---|---|---|
| Papers downloaded | 28 | `papers/` |
| Code repositories cloned | 6 | `code/` |
| Datasets locally available | 5 (+3 in code repos) | `datasets/` and `code/*/datasets/` |
| Documentation deliverables | 5 | this file + per-dir READMEs + literature_review.md |

Environment: isolated `uv` venv with `pypdf`, `requests`, `arxiv`, `httpx`, `datasets` installed
via `pyproject.toml`. Activate with `source .venv/bin/activate`.

---

## Papers

28 PDFs in `papers/`. See [`papers/README.md`](papers/README.md) for full descriptions and grouping
by research-question bucket.

### Quick index (most relevant first)

| Title | Authors | Year | File |
|---|---|---|---|
| Towards Understanding Sycophancy in Language Models | Sharma et al. (Anthropic) | 2023 | `papers/sharma2023_understanding_sycophancy.pdf` |
| Frictive Policy Optimization for LLMs | Pustejovsky & Krishnaswamy | 2026 | `papers/frictive_policy_optimization.pdf` |
| The Personality Illusion | Han et al. (Caltech) | 2025 | `papers/personality_illusion_dissociation.pdf` |
| Bias Runs Deep: Persona-Assigned LLMs | Gupta et al. (AI2) | 2023 | `papers/gupta2023_bias_runs_deep_persona.pdf` |
| Role-Play with Large Language Models | Shanahan et al. (DeepMind) | 2023 | `papers/shanahan2023_roleplay_llms.pdf` |
| Discovering LM Behaviors with Model-Written Evaluations | Perez et al. (Anthropic) | 2022 | `papers/perez2022_discovering_model_behaviors.pdf` |
| Constitutional AI | Bai et al. (Anthropic) | 2022 | `papers/bai2022_constitutional_ai.pdf` |
| TRUTH DECAY: Multi-Turn Sycophancy | Liu et al. | 2025 | `papers/truth_decay_multiturn_sycophancy.pdf` |
| SOTOPIA: Interactive Evaluation for Social Intelligence | Zhou et al. (CMU) | 2023 | `papers/sotopia_social_intelligence.pdf` |
| DELPHI: Controversial Issues | Sun et al. (Apple) | 2023 | `papers/delphi_controversial_issues.pdf` |
| Toxicity in Persona-Assigned ChatGPT | Deshpande et al. | 2023 | `papers/deshpande2023_toxicity_persona_chatgpt.pdf` |
| Simple synthetic data reduces sycophancy | Wei et al. (Google) | 2023 | `papers/wei2023_synthetic_data_reduce_sycophancy.pdf` |
| Persona-Steered Generation Bias | (anon) | 2024 | `papers/persona_steered_generation_bias.pdf` |
| RoleLLM Benchmark | Wang et al. | 2023 | `papers/wang2023_rolellm_benchmark.pdf` |
| Generative Agents (Smallville) | Park et al. | 2023 | `papers/park2023_generative_agents.pdf` |
| Personalizing Dialogue Agents (PersonaChat) | Zhang et al. (FB) | 2018 | `papers/zhang2018_personachat.pdf` |
| Multiagent Debate (Du, Li) | Du & Li et al. (MIT) | 2023 | `papers/duLi2023_multiagent_debate_factuality.pdf` |
| Can LLM Agents Really Debate? | (anon) | 2025 | `papers/can_llm_agents_really_debate.pdf` |
| Spotting Out-of-Character Behavior | (anon) | 2025 | `papers/out_of_character_persona_fidelity.pdf` |
| Linear Personality Probing (Big Five) | (anon) | 2025 | `papers/linear_personality_probing_big_five.pdf` |
| Moral Susceptibility under Persona Role-Play | (anon) | 2025 | `papers/moral_susceptibility_persona_roleplay.pdf` |
| Are Economists Always More Introverted? | (anon) | 2025 | `papers/economists_introverted_persona_consistency.pdf` |
| Whose Personae? Synthetic Persona Experiments | (anon) | 2025 | `papers/whose_personae_synthetic_experiments.pdf` |
| Multi-Turn Sycophancy (companion to TRUTH DECAY) | (anon) | 2025 | `papers/measuring_sycophancy_multiturn.pdf` |
| Not Your Typical Sycophant | (anon) | 2026 | `papers/not_your_typical_sycophant.pdf` |
| Sycophancy Claims, Missing Human-in-the-Loop | (anon) | 2025 | `papers/sycophancy_missing_human_loop.pdf` |
| Pink Elephants — Direct Principle Feedback | (anon) | 2024 | `papers/pink_elephants_direct_principle_feedback.pdf` |
| Augment Critical Thinking — Design | (anon) | 2025 | `papers/augment_critical_thinking_design.pdf` |

PDF chunks for deep-read papers (3 pages each) are in `papers/pages/`. Manifests at
`papers/pages/*_manifest.txt`.

---

## Datasets

See [`datasets/README.md`](datasets/README.md) for full download instructions. Data files are
gitignored; redownload via the snippets in that README.

| Name | Source | Local Path | Size | Task |
|---|---|---|---|---|
| **SycophancyEval** (3 splits) | Sharma et al. (Anthropic) | `code/sycophancy-eval/datasets/` | 20.6k examples | Sycophancy probes — **primary benchmark** |
| **TruthfulQA (MC)** | HF: `truthful_qa` | `datasets/truthful_qa_mc/` | 817 questions | Misconception MC |
| **MMLU (1 subject)** | HF: `cais/mmlu` | `datasets/mmlu_hs_math_sample/` | 304 questions | Reasoning (extensible) |
| **Anthropic Persuasion** | HF: `Anthropic/persuasion` | `datasets/anthropic_persuasion/` | 3,939 records | Persuasiveness ratings |
| **hh-rlhf (helpful-base sample)** | HF: `Anthropic/hh-rlhf` | `datasets/hh_rlhf_helpful_sample/` | 2.5k records (sub of 46k) | Preference pairs |
| Personality-Illusion behavioral tasks | psychology-of-AI repo | `code/Personality-Illusion/behavioral_tasks/datasets/` | 313+643+75 | Dilemmas, IAT, norms |
| DELPHI controversial questions | apple/ml-delphi | (clone separately) | ~30k | Controversy stance-taking |
| RoleBench | HF: `ZenMoore/RoleBench` | (download separately) | 168k samples × 100 roles | Role-play fidelity |
| Persona-Bias outputs (gated) | HF: `allenai/persona-bias` | (request access) | 1.5M records | Pre-computed persona × task outputs |

Sample records for inspection: `datasets/sample_records.json`.

---

## Code repositories

6 repos in `code/`. See [`code/README.md`](code/README.md) for descriptions, dependencies, and
how each fits the hypothesis.

| Name | URL | Location | Notes |
|---|---|---|---|
| **sycophancy-eval** | github.com/meg-tong/sycophancy-eval | `code/sycophancy-eval/` | **Primary measurement instrument** — ready to run with minimal deps |
| **persona-bias** | github.com/allenai/persona-bias | `code/persona-bias/` | Persona × 24-reasoning-task rig |
| **Personality-Illusion** | github.com/psychology-of-AI/Personality-Illusion | `code/Personality-Illusion/` | Self-report-vs-behavior replication code |
| **sotopia** | github.com/sotopia-lab/sotopia | `code/sotopia/` | Heavy: full multi-agent social-eval framework |
| **RoleLLM-public** | github.com/InteractiveNLP-Team/RoleLLM-public | `code/RoleLLM-public/` | Docs + RoleBench pointers |
| **DELPHI** | github.com/ZidiXiu/DELPHI | `code/DELPHI/` | Pointer to apple/ml-delphi |

---

## Resource gathering notes

### Search strategy

The paper-finder service was unavailable during this session (both `fast` and `diligent` modes
returned empty after timeout). Fell back to direct arXiv API queries via the `arxiv` Python
package. Searched four query clusters:

1. Sycophancy + RLHF + reward hacking
2. Persona + role-play + identity + steering
3. Critical thinking + epistemic humility + alignment friction
4. Multi-agent debate + dialogue + collaborative AI

Then pulled known foundational papers by arXiv ID list (Sharma et al., Perez et al., Bai et al.,
Shanahan, etc.). This produced ~150 candidates; the 28 downloaded are the curated set with
direct relevance to the hypothesis.

### Selection criteria

For each candidate paper, the inclusion criteria were:
- **Year ≥ 2022** unless foundational (PersonaChat 2018 is the only exception)
- **Direct relevance** to one of the three hypothesis claims (C1 sycophancy, C2 persona, C3
  friction-as-alignment), OR
- **Methodological relevance** (recent critiques of sycophancy measurement, persona experiment
  meta-reviews)
- **Code/data availability** preferred — 11 of 28 papers have associated open releases we can
  use.

### Challenges encountered

1. **Paper-finder service down** — fell back to direct arXiv search via the `arxiv` package.
2. **PersonaChat HF loader broken** — newer `datasets` rejects the legacy `.py` loading script.
   Documented workarounds in `datasets/README.md`.
3. **Sotopia HF dataset broken** — column-mismatch error across the multi-config setup. The
   cloned repo (`code/sotopia/`) ships the scenarios directly.
4. **allenai/persona-bias is gated** — requires HF authentication. Documented in
   `datasets/README.md`.
5. **DELPHI canonical location is apple/ml-delphi**, but the linked repo
   (github.com/ZidiXiu/DELPHI) is just a pointer. We cloned the pointer; experiment runner will
   need to manually clone apple/ml-delphi for the actual data.

### Gaps and workarounds

- **No PersonaChat locally**: workaround is to use it via raw download from ParlAI if needed.
  Not core to the hypothesis since it predates LLMs.
- **No SOTOPIA episode database**: the `cmu-lti/sotopia` HF dataset has a multi-config bug; use
  the JSON files inside the cloned repo (`code/sotopia/`).
- **No DELPHI data**: need to clone apple/ml-delphi separately. The pointer repo we cloned has
  citation/abstract only.

---

## Recommendations for experiment design

Synthesized in [`literature_review.md` §8](literature_review.md); here is the short version:

1. **Primary dataset**: SycophancyEval (all three splits) for the failure-mode metric.
2. **Supporting datasets**: TruthfulQA + DELPHI (stance-taking) + Anthropic Persuasion
   (resistance to persuasion) + Personality-Illusion behavioral tasks (cross-task validation).
3. **Baseline persona conditions**: no-persona → shallow demographic → constitutional
   principled → substantive belief-based → (optional) persona-tuned model.
4. **Metrics**: existing SycophancyEval metrics + Gupta-style abstention rate + DELPHI
   stance-taking rate + multi-turn stability + behavioral-self-report alignment.
5. **Code to adapt/reuse**: `code/sycophancy-eval/` (drop in new persona conditions), then
   `code/persona-bias/` for the reasoning-task suite, then `code/Personality-Illusion/` for the
   behavior-vs-self-report measurement.

---

## File map

```
/workspaces/distinct-persona-ai-12e6-claude/
├── pyproject.toml                       # Isolated uv environment manifest
├── .venv/                               # Local virtualenv
├── literature_review.md                 # Synthesis (deliverable)
├── resources.md                         # This file (deliverable)
├── papers/                              # 28 PDFs
│   ├── README.md                        # Per-paper relevance notes
│   ├── download_log.json                # Download manifest
│   └── pages/                           # PDF chunks for deep-read papers
├── datasets/
│   ├── .gitignore                       # Exclude data from git
│   ├── README.md                        # Download instructions
│   ├── sample_records.json              # Sample first-records per dataset
│   ├── truthful_qa_mc/
│   ├── mmlu_hs_math_sample/
│   ├── anthropic_persuasion/
│   └── hh_rlhf_helpful_sample/
├── code/
│   ├── README.md                        # Per-repo descriptions
│   ├── sycophancy-eval/                 # Sharma et al. benchmark + utils
│   ├── persona-bias/                    # Gupta et al. reasoning rig
│   ├── Personality-Illusion/            # Han et al. probes + data
│   ├── sotopia/                         # CMU social-eval framework
│   ├── RoleLLM-public/                  # Pointers to RoleBench
│   └── DELPHI/                          # Pointer to apple/ml-delphi
├── paper_search_results/                # Raw arXiv search outputs
└── .resource_finder_complete            # Pipeline marker (after final step)
```
