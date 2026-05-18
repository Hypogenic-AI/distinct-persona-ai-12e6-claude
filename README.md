# Distinct Persona

Does giving an LLM a *substantive belief-based persona* (e.g., "you are a careful
empiricist; pressure is not evidence") produce better collaborative friction than the
default "helpful assistant" framing?

## Key findings

- **Yes, substantially.** On a misconception-correction task (TruthfulQA, n=80 × 5 personas
  × 2 models), the belief persona reduced sycophantic affirmation of false user claims by
  ~20 percentage points vs no-persona baseline on both `gpt-4.1` (41.3% → 20%, p=0.0018)
  and `gpt-4.1-mini` (46.3% → 27.5%, p=0.007). Effects survive Holm correction.
- **The persona is action-controlling, not decorative.** Response embeddings under the
  belief persona are far closer to a held-out belief-persona style reference (Cohen's d ≥
  1.27 vs every other condition). Belief responses use ~5–7× more epistemic stance markers
  (`re-examine`, `evidence`, `I disagree`, …). This **refutes one form of the
  Personality-Illusion finding** (Han et al. 2025) — for substantive belief personas
  (not Big-Five or demographic ones).
- **The "helpful assistant" default is not neutral.** On gpt-4.1, the literal "You are a
  helpful assistant." system prompt *quadrupled* the are-you-sure fold rate (7.1% → 27.3%,
  p=0.014) versus no system prompt at all. Defaults that pose as low-identity may actively
  reinforce yielding-as-helpfulness.
- **Demographic personas don't help.** "You are an experienced software engineer named
  Alex" produces no significant effect on either sycophancy or correction — consistent with
  Han et al.'s null result for that *kind* of persona.

See [`REPORT.md`](REPORT.md) for full results, statistics, and discussion.

## Repository layout

```
distinct-persona/
├── REPORT.md                   # Primary research report
├── planning.md                 # Pre-registered design + motivation
├── literature_review.md        # 28-paper synthesis
├── resources.md                # Resource catalog
├── src/
│   ├── personas.py             # 5 frozen persona conditions
│   ├── llm.py                  # OpenAI wrapper w/ cache + cost tracking
│   ├── experiment1_are_you_sure.py
│   ├── experiment2_productive_friction.py
│   ├── experiment3_persona_stickiness.py
│   └── analyze.py              # Stats + figure generation
├── results/
│   ├── e1_are_you_sure.jsonl
│   ├── e2_productive_friction.jsonl
│   ├── e3_persona_stickiness.jsonl
│   ├── summary_e{1,2,3}.csv
│   └── stats_tests.json
├── figures/
│   ├── e1_fold_rate.png
│   ├── e2_productive_friction.png
│   └── e3_persona_stickiness.png
├── papers/                     # 28 PDFs (literature)
├── datasets/                   # TruthfulQA, MMLU sample, etc.
└── code/                       # Pre-cloned baseline repos
```

## Reproducing the results

Cost: ~$3.90 in OpenAI API calls. Wall time: ~11 minutes for E1+E2 in parallel on one machine.

```bash
# 1. Setup (uses uv for fast dep install)
uv venv && source .venv/bin/activate
uv sync

# 2. Set API key
export OPENAI_API_KEY=sk-...

# 3. Run the three experiments
python src/experiment1_are_you_sure.py        # E1: ~$2.20, ~10 min
python src/experiment2_productive_friction.py # E2: ~$1.70, ~8 min
python src/experiment3_persona_stickiness.py  # E3: <$0.05, embedding-only

# 4. Analyze + plot
python src/analyze.py
```

All API calls are cached to `logs/llm_cache/<sha256>.json`, so re-runs cost nothing.

## Datasets used

- **SycophancyEval / are_you_sure** (Sharma et al. 2023) — `code/sycophancy-eval/datasets/` (cloned).
- **TruthfulQA-MC1** (Lin et al. 2022) — `datasets/truthful_qa_mc/` (downloaded).
