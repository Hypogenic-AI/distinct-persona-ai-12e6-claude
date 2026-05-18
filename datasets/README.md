# Datasets

Data files are excluded from git via `.gitignore` (see that file). Download instructions are below
for each dataset; running the snippets in an activated `.venv` will reproduce the local state.

For experiment-runner reference, a single JSON of `first_record` samples is at
`sample_records.json`.

---

## 1. SycophancyEval (Sharma et al., 2023) — `code/sycophancy-eval/datasets/`

Three JSONL files shipped inside the SycophancyEval repo (no separate download needed):

| File | Rows | Probe |
|------|------|-------|
| `feedback.jsonl` | 8,500 | Feedback sycophancy: model rates arguments/poems/math with stated user preference |
| `are_you_sure.jsonl` | 4,887 | "Are you sure?" sycophancy: model challenged after correct answer |
| `answer.jsonl` | 7,267 | Answer sycophancy: user states weak (in)correct belief |

Already on disk under `code/sycophancy-eval/datasets/`. This is the **primary benchmark** for the
sycophancy side of the Distinct Persona hypothesis — it operationalizes "no pushback."

Schema (all files): `{"prompt": [{"type": "human|ai", "content": "..."}, ...], "base": {...},
"metadata": {...}}`.

---

## 2. TruthfulQA (multiple_choice) — `datasets/truthful_qa_mc/`

- **Source**: `truthful_qa` on HuggingFace
- **Size**: 817 questions (validation only)
- **Task**: MC question-answering against common misconceptions
- **Why we need it**: Used by TRUTH DECAY and several sycophancy benchmarks as the
  underlying factual ground truth that user pressure tries to corrupt.

```python
from datasets import load_dataset
ds = load_dataset("truthful_qa", "multiple_choice")
ds.save_to_disk("datasets/truthful_qa_mc")
```

---

## 3. MMLU (high_school_mathematics sample) — `datasets/mmlu_hs_math_sample/`

- **Source**: `cais/mmlu` on HuggingFace
- **Size**: 270 test / 29 val / 5 dev (one subject only)
- **Why we need it**: Same role as TruthfulQA — ground-truth questions over which a sycophancy /
  persona test can be layered. Used by Gupta et al. "Bias Runs Deep" as one of 24 reasoning
  datasets. Full MMLU is 57 subjects; we pulled one as a sample. Pull more as needed:

```python
from datasets import load_dataset
# Available subjects: see https://huggingface.co/datasets/cais/mmlu
ds = load_dataset("cais/mmlu", "high_school_mathematics")
ds.save_to_disk("datasets/mmlu_hs_math_sample")
```

---

## 4. Anthropic Persuasion — `datasets/anthropic_persuasion/`

- **Source**: `Anthropic/persuasion` on HuggingFace
- **Size**: 3,939 records
- **Schema**: `worker_id, claim, argument, source, prompt_type, rating_initial, rating_final, persuasiveness_metric`
- **Why we need it**: Pairs of (claim, argument) labeled with how much the argument moved a human's
  rating. Directly probes the *persuasion-from-the-other-side* dynamic implicated by the hypothesis
  — does an LLM with distinct persona resist persuasion differently than a "low identity" one?

```python
from datasets import load_dataset
ds = load_dataset("Anthropic/persuasion")
ds.save_to_disk("datasets/anthropic_persuasion")
```

---

## 5. Anthropic hh-rlhf (helpful-base sample) — `datasets/hh_rlhf_helpful_sample/`

- **Source**: `Anthropic/hh-rlhf` (helpful-base config) on HuggingFace
- **Size**: 2,000 train + 500 test (subsampled; full is 43,835 + 2,354)
- **Schema**: `chosen, rejected` (both raw dialogues)
- **Why we need it**: This is the dataset Sharma et al. analyzed to show "matches user's beliefs"
  is one of the strongest predictors of preference. Useful for replicating their Bayesian-logistic
  analysis or running ablations.

```python
from datasets import load_dataset
ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base")
ds.save_to_disk("datasets/hh_rlhf_helpful_sample")
```

---

## 6. DELPHI (Controversial Questions) — see `code/DELPHI/`

- **Source**: Hosted at github.com/apple/ml-delphi (the github.com/ZidiXiu/DELPHI repo points to it)
- **Size**: ~30,000 human-labeled controversial-vs-not Quora questions
- **Why we need it**: Probes whether a persona-conditioned model maintains a stance on controversial
  issues or retreats into "low identity" neutrality.

```bash
# Manual download required; see https://github.com/apple/ml-delphi
# Then place CSV/JSON under datasets/delphi/
```

---

## 7. Persona-Bias (Gupta et al., 2023) — gated HF dataset

- **Source**: `allenai/persona-bias` on HuggingFace (**gated** — requires HF token + access request)
- **Size**: ~1.5M model-generated outputs across 19 personas × 24 reasoning datasets
- **Why we need it**: Ready-made reasoning-task outputs from 4 LLMs under varied personas. Saves
  weeks of inference cost if used as-is for a replication / extension.

```python
# Request access at https://huggingface.co/datasets/allenai/persona-bias
# Then: huggingface-cli login
from datasets import load_dataset
ds = load_dataset("allenai/persona-bias")
ds.save_to_disk("datasets/persona_bias")
```

---

## 8. Personality-Illusion behavioral tasks — `code/Personality-Illusion/`

Three small JSON/CSV datasets shipped inside the Personality-Illusion repo:

| File | Rows | Task |
|------|------|------|
| `behavioral_tasks/datasets/dilemmas.json` | 313 | Moral dilemma probes |
| `behavioral_tasks/datasets/iat_stimuli.json` | 643 | Implicit Association Test stimuli |
| `behavioral_tasks/datasets/norm300_syn.csv` | 75 | Honesty norm violations |

Plus `data/baseline_responses/` and `data/persona_injection/` with the authors' captured model
outputs (useful baselines).

---

## 9. SOTOPIA (notes only — see code repo)

- **Source**: `cmu-lti/sotopia` on HuggingFace, but the multi-config setup is broken via
  `load_dataset` (column-mismatch error observed on 2026-05-18). Use the cloned repo directly:
  `code/sotopia/` ships scenario/character data files and the full evaluation pipeline.
- **Why we need it**: The most directly aligned eval environment — persona-assigned agents with
  social goals pursuing/resisting each other in scripted scenarios.

---

## What we did NOT download (and why)

- **PersonaChat** (Zhang et al. 2018): the canonical version on HF (`bavard/personachat_truecased`)
  uses a legacy `.py` loading script that newer `datasets` rejects. Available via raw download
  from github.com/facebookresearch/ParlAI or Kaggle if needed. Not core for the hypothesis since
  it predates LLMs.
- **RoleBench** (RoleLLM): hosted on HF at `ZenMoore/RoleBench`. Useful if we want to evaluate
  *quality* of persona role-play; download directly if/when that becomes an experiment.
- **Generative Agents traces** (Park et al. 2023): visualization-only release; no labeled dataset
  to ingest.

## Provenance

All downloads performed 2026-05-18 in an isolated `uv` venv. See `pyproject.toml` for exact
versions of `datasets` and dependencies.
