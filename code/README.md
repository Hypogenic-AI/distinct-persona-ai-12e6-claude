# Cloned Repositories

All cloned shallow (`--depth 1`) on 2026-05-18 in `code/`. Each entry lists URL, what it
provides, and how it relates to the Distinct Persona hypothesis.

---

## 1. `sycophancy-eval/` — Sharma et al. 2023 (Anthropic)

- **URL**: https://github.com/meg-tong/sycophancy-eval
- **Provides**: Three JSONL benchmarks (`feedback.jsonl`, `are_you_sure.jsonl`, `answer.jsonl`) plus
  a Python `utils.py` for sync/async OpenAI/Anthropic inference and an `example.ipynb` walkthrough.
- **Use for the hypothesis**: This is the **primary measurement instrument** for the failure mode
  the hypothesis targets (lack of pushback). The experiment runner can drop a model in front of these
  prompts and get a directly-comparable sycophancy metric. Key entry points:
  - `example.ipynb` shows the full eval loop.
  - `utils.inference` / `utils.async_inference` for batched generation.
- **Dependencies**: minimal — `pandas`, `anthropic`, `openai`.
- **Status**: ✅ ready to use, no setup needed beyond installing the SDK for the target provider.

---

## 2. `persona-bias/` — Gupta et al. 2023 (Allen AI; ICLR 2024)

- **URL**: https://github.com/allenai/persona-bias
- **Provides**:
  - `persona/run.py` — single entry point that runs any of 24 reasoning datasets under any of 19
    personas with any of 3 persona instructions.
  - `persona/dataset/` — wrappers around MMLU, MATH, MoCA, AQuA, ARC, etc.
  - `persona/prompts/` — the 3 persona instruction templates plus 19 socio-demographic personas.
  - `persona/evaluators/` — accuracy + abstention classifiers.
- **Use for the hypothesis**: Most useful pre-built rig for asking "does identity X change what
  the model can/will do?" Drop in new personas easily by editing the persona prompt dict.
- **Dependencies**: `requirements.txt` (OpenAI SDK + standard ML stack).
- **Setup**: see README — needs `OPENAI_API_KEY` and `PYTHONPATH=.`.
- **Status**: ✅ ready; their HF dataset `allenai/persona-bias` (gated) has 1.5M precomputed outputs.

---

## 3. `Personality-Illusion/` — Han et al. 2025 (Caltech)

- **URL**: https://github.com/psychology-of-AI/Personality-Illusion
- **Provides**:
  - `self-reports/` — BFI + SRQ questionnaire prompts and grading code.
  - `behavioral_tasks/` — four behavioral probes (`Honesty.ipynb`, `IAT.ipynb`,
    `RiskTaking.ipynb`, `Sycophancy.ipynb`) plus 3 datasets.
  - `data/baseline_responses/` and `data/persona_injection/` — captured model outputs for 12
    base + instruction-tuned LLMs, ready for re-analysis.
- **Use for the hypothesis**: Directly relevant — replicates the dissociation between persona
  self-report and behavior. The `Sycophancy.ipynb` notebook is the cleanest example of layering
  persona injection onto a behavioral probe.
- **Status**: ✅ data shipped in repo; notebooks are runnable with model API keys.

---

## 4. `sotopia/` — Zhou et al. 2023 (CMU; ICLR 2024 spotlight)

- **URL**: https://github.com/sotopia-lab/sotopia
- **Provides**: Full social-interaction simulation framework. Agents have personas + social goals,
  scenarios are scripted, evaluation uses 7-dimension rubrics (relationship, knowledge, secret,
  social rules, financial, goal completion, believability).
- **Use for the hypothesis**: The most ambitious eval environment. Directly tests whether
  persona-conditioned agents pursue goals (i.e., maintain identity) under social pressure. Useful
  for the "what kinds of conversations require pushback?" research question — many scenarios
  *require* one agent to disagree to satisfy their goal.
- **Dependencies**: pyproject.toml + `uv.lock`; requires Redis for episode storage.
- **Status**: ⚠️ heavy setup (Redis, multiple model API keys). Consider only if running interactive
  multi-agent eval.

---

## 5. `RoleLLM-public/` — Wang et al. 2023

- **URL**: https://github.com/InteractiveNLP-Team/RoleLLM-public
- **Provides**: Documentation + pointers to RoleBench (168k samples, 100 roles) hosted at
  `ZenMoore/RoleBench` on HF. Repo itself is primarily README + paper figures.
- **Use for the hypothesis**: Useful for evaluating role-play *fidelity* — i.e., how distinctly
  a model can maintain a persona — but not the friction/pushback side.
- **Status**: ✅ docs only; download RoleBench separately if needed.

---

## 6. `DELPHI/` — Sun et al. 2023 (Apple; EMNLP 2023)

- **URL**: https://github.com/ZidiXiu/DELPHI (forwarder; canonical at github.com/apple/ml-delphi)
- **Provides**: README + appendix; actual dataset is at apple/ml-delphi.
- **Use for the hypothesis**: Probes whether persona-conditioned models hold a position on
  controversial topics. Need to clone apple/ml-delphi separately for the actual ~30k question CSV.
- **Status**: ⚠️ pointer-only; manual download of apple/ml-delphi required for actual data.

---

## What we did NOT clone (and why)

- **github.com/facebookresearch/ParlAI** — multi-GB framework; we use HF directly.
- **Anthropic constitutional-AI** — no official open-source repo of the training procedure.
- **OpenAI evals / Anthropic evals** — large and not directly aligned to persona-friction.
- **Park et al. generative-agents** (joonspk-research/generative_agents) — Unity/JS heavy; not
  useful for headless experiments.

## How these fit together

For a minimum-viable Distinct Persona experiment, the experiment runner only needs:
1. `sycophancy-eval/` — measures the failure mode.
2. `persona-bias/` — provides the persona-conditioning rig and reasoning tasks.
3. `Personality-Illusion/` — provides the self-report-vs-behavior dissociation pattern to extend.

Everything else is optional depth.
