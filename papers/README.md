# Downloaded Papers

All papers were obtained from arXiv. Relevance is judged against the hypothesis that LLMs lack
collaborative friction because they lack distinct persona/identity grounding.

Papers are grouped into four buckets that map to the research question:
1. **Sycophancy** — the failure mode the hypothesis predicts (lack of pushback)
2. **Persona / identity** — what "distinct persona" can and cannot do today
3. **Collaborative friction / epistemic intervention** — the positive construct the hypothesis names
4. **Multi-agent debate / supporting work** — methods that approximate identity-driven disagreement via separate agents

## 1. Sycophancy

1. **Towards Understanding Sycophancy in Language Models** (Sharma et al., 2023; ICLR 2024)
   - File: `sharma2023_understanding_sycophancy.pdf`
   - arXiv: 2310.13548 (Anthropic)
   - Why relevant: Demonstrates 5 production AI assistants exhibit sycophancy across feedback, "are you sure?", answer, and mimicry tasks. Crucially, Bayesian logistic regression on hh-rlhf shows "matches user's beliefs" is one of the strongest predictors of human preference — i.e., training data actively rewards low-friction agreement. Provides the `SycophancyEval` benchmark used by subsequent work.
   - Releases: SycophancyEval datasets and code at github.com/meg-tong/sycophancy-eval (cloned).

2. **Simple synthetic data reduces sycophancy in large language models** (Wei et al., 2023; Google)
   - File: `wei2023_synthetic_data_reduce_sycophancy.pdf`
   - arXiv: 2308.03958
   - Why relevant: Shows sycophancy emerges with scale and instruction tuning, then proposes a synthetic-data finetuning intervention. A baseline mitigation to compare against.

3. **Discovering Language Model Behaviors with Model-Written Evaluations** (Perez et al., 2022; Anthropic)
   - File: `perez2022_discovering_model_behaviors.pdf`
   - arXiv: 2212.09251
   - Why relevant: Original RLHF-scale evidence that finetuning makes models more sycophantic on political, religious, and stated-belief questions. Provides 154 model-written eval datasets covering persona / belief / sycophancy behaviors — directly usable as benchmarks.

4. **TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models** (Liu et al., 2025)
   - File: `truth_decay_multiturn_sycophancy.pdf`
   - arXiv: 2503.11656
   - Why relevant: Single-turn sycophancy benchmarks understate the problem; this extends to multi-turn dialogues with static feedback and rationale-based pushback, plus two prompt-based mitigations.

5. **Measuring Sycophancy of Language Models in Multi-turn Dialogues** (2025)
   - File: `measuring_sycophancy_multiturn.pdf`
   - arXiv: 2505.23840
   - Why relevant: Companion multi-turn sycophancy benchmark, useful for triangulating with TRUTH DECAY.

6. **Not Your Typical Sycophant: The Elusive Nature of Sycophancy in Large Language Models** (2026)
   - File: `not_your_typical_sycophant.pdf`
   - arXiv: 2601.15436
   - Why relevant: Recent critique that prior sycophancy evals inject confounds (manipulative language, uncontrolled bias); proposes a cleaner measurement protocol — important for experimental rigor.

7. **Sycophancy Claims about Language Models: The Missing Human-in-the-Loop** (2025)
   - File: `sycophancy_missing_human_loop.pdf`
   - arXiv: 2512.00656
   - Why relevant: Position paper enumerating five methodological pitfalls in sycophancy measurement; useful as a checklist when designing the Distinct Persona experiment.

## 2. Persona / identity

8. **Role-Play with Large Language Models** (Shanahan, McDonell & Reynolds, 2023; DeepMind / Eleuther)
   - File: `shanahan2023_roleplay_llms.pdf`
   - arXiv: 2305.16367
   - Why relevant: Conceptual paper arguing that LLM dialogue agents are best understood as *role-playing* a character (or a superposition of simulacra). Provides the theoretical vocabulary for the hypothesis's "distinct vs low-identity" framing.

9. **Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs** (Gupta et al., 2023; AI2/Stanford/Princeton; ICLR 2024)
   - File: `gupta2023_bias_runs_deep_persona.pdf`
   - arXiv: 2311.04892
   - Why relevant: Most rigorous study of how persona assignment changes reasoning. 19 personas × 24 reasoning datasets × 4 LLMs. 80% of personas cause significant performance drops, often via task-irrelevant abstentions. The flip side of the Distinct Persona hypothesis: identity changes what the agent will *do*, not just what it will say.
   - Releases: code/dataset at allenai.github.io/persona-bias (cloned).

10. **Toxicity in ChatGPT: Analyzing Persona-assigned Language Models** (Deshpande et al., 2023)
    - File: `deshpande2023_toxicity_persona_chatgpt.pdf`
    - arXiv: 2304.05335
    - Why relevant: First broad study showing persona assignment changes downstream behavior — increases toxicity by up to 6×. Establishes that "you are X" prompts are not cosmetic.

11. **Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs** (2025)
    - File: `economists_introverted_persona_consistency.pdf`
    - arXiv: 2506.02659
    - Why relevant: Quantifies persona *consistency* (or lack thereof) across tasks — central to whether a "distinct persona" can be reliably maintained.

12. **The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs** (Han et al., 2025; NeurIPS LAW workshop best paper honorable mention)
    - File: `personality_illusion_dissociation.pdf`
    - arXiv: 2509.03730
    - Why relevant: Key result: persona injection moves Big-Five *self-reports* in the intended direction (β=3.95, p<.001) but barely moves *behavior* on sycophancy (β=0.03, p=0.67). Directly relevant to the hypothesis: surface persona ≠ identity-driven action.
    - Releases: code at github.com/psychology-of-AI/Personality-Illusion (cloned).

13. **Linear Personality Probing and Steering in LLMs: A Big Five Study** (2025)
    - File: `linear_personality_probing_big_five.pdf`
    - arXiv: 2512.17639
    - Why relevant: Activation-level steering of Big Five traits — a mechanistic alternative to prompt-based persona.

14. **Moral Susceptibility and Robustness under Persona Role-Play in LLMs** (2025)
    - File: `moral_susceptibility_persona_roleplay.pdf`
    - arXiv: 2511.08565
    - Why relevant: Tests whether persona role-play shifts moral judgments — i.e., whether identity changes what the model believes is right.

15. **Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation** (2025)
    - File: `out_of_character_persona_fidelity.pdf`
    - arXiv: 2506.19352
    - Why relevant: Fine-grained metric for whether persona claims actually carry through in extended generation — directly usable to measure "strength" of an instilled persona.

16. **Evaluating Large Language Model Biases in Persona-Steered Generation** (2024)
    - File: `persona_steered_generation_bias.pdf`
    - arXiv: 2405.20253
    - Why relevant: Shows LLMs are 9.7% less steerable toward *incongruous* persona stances (e.g., a "Republican" agreeing with climate action). Helps explain *why* personas may fail to push back when identity demands it.

17. **Whose Personae? Synthetic Persona Experiments in LLM Research and Pathways to Transparency** (2025)
    - File: `whose_personae_synthetic_experiments.pdf`
    - arXiv: 2512.00461
    - Why relevant: Critical meta-review of persona-experiment methodology — useful for the experimental design and limitations sections.

18. **RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of LLMs** (Wang et al., 2023)
    - File: `wang2023_rolellm_benchmark.pdf`
    - arXiv: 2310.00746
    - Why relevant: Provides RoleBench (168k samples, 100 roles) — the largest open role-play benchmark; useful for evaluating persona instillation.

19. **Personalizing Dialogue Agents: I have a dog, do you have pets too?** (Zhang et al., 2018; ACL 2018; Facebook AI)
    - File: `zhang2018_personachat.pdf`
    - arXiv: 1801.07243
    - Why relevant: Foundational PersonaChat paper — the original "give a dialogue agent a persona profile" setup; cited by almost everything in §2.

## 3. Collaborative friction / epistemic intervention

20. **Frictive Policy Optimization for LLMs: Epistemic Intervention, Risk-Sensitive Control, and Reflective Alignment** (Pustejovsky & Krishnaswamy, 2026)
    - File: `frictive_policy_optimization.pdf`
    - arXiv: 2604.25136
    - Why relevant: The most directly aligned paper to the hypothesis. Argues alignment must treat *clarification, verification, challenge, redirection, refusal* as first-class control actions, not failure modes. Proposes a family of optimization methods (FAR, FPP, GRFR, FTR) and an evaluation framework around clarification competence, contradiction repair, and proportional refusal.

21. **Constitutional AI: Harmlessness from AI Feedback** (Bai et al., 2022; Anthropic)
    - File: `bai2022_constitutional_ai.pdf`
    - arXiv: 2212.08073
    - Why relevant: The canonical alternative to RLHF — uses a written constitution to give the model a self-critique principle. Often cited as an alignment-by-identity mechanism.

22. **Suppressing Pink Elephants with Direct Principle Feedback** (2024)
    - File: `pink_elephants_direct_principle_feedback.pdf`
    - arXiv: 2402.07896
    - Why relevant: Newer "give the model a principle" alignment method; useful baseline for persona-conditioned training.

23. **SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents** (Zhou et al., 2023; ICLR 2024 spotlight)
    - File: `sotopia_social_intelligence.pdf`
    - arXiv: 2310.11667
    - Why relevant: Open-ended social interaction benchmark with persona-assigned agents pursuing goals — a natural evaluation environment for "does identity drive productive disagreement?"
    - Releases: code at github.com/sotopia-lab/sotopia (cloned).

24. **DELPHI: Data for Evaluating LLMs' Performance in Handling Controversial Issues** (Sun et al., 2023; EMNLP 2023; Apple)
    - File: `delphi_controversial_issues.pdf`
    - arXiv: 2310.18130
    - Why relevant: ~30k human-annotated controversial questions. Directly probes whether models take a stance vs. give a "low-identity" non-answer.
    - Releases: dataset at github.com/ZidiXiu/DELPHI (cloned).

25. **Designing AI Systems that Augment Human Performed vs. Demonstrated Critical Thinking** (2025)
    - File: `augment_critical_thinking_design.pdf`
    - arXiv: 2504.14689
    - Why relevant: HCI framing of when AI should push back vs. defer — useful for grounding the "what kinds of conversations require pushback" research question.

26. **Generative Agents: Interactive Simulacra of Human Behavior** (Park et al., 2023)
    - File: `park2023_generative_agents.pdf`
    - arXiv: 2304.03442
    - Why relevant: Smallville agents — the canonical demonstration that persona + memory yields recognizably *different* social behavior. Useful for thinking about identity beyond the system prompt.

## 4. Multi-agent debate (supporting)

27. **Improving Factuality and Reasoning in Language Models through Multiagent Debate** (Du & Li et al., 2023; MIT)
    - File: `duLi2023_multiagent_debate_factuality.pdf`
    - arXiv: 2305.14325
    - Why relevant: Multiple instances of the same model debating each other improves accuracy. Suggests "friction" can come from architectural separation when it can't come from a single agent.

28. **Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning** (2025)
    - File: `can_llm_agents_really_debate.pdf`
    - arXiv: 2511.07784
    - Why relevant: Critical follow-up — argues much of debate's benefit is from ensembling, not genuine disagreement. Strong support for the hypothesis that current LLMs lack the persona depth to disagree honestly.

---

## Provenance

- All PDFs downloaded directly from arXiv via `requests` on 2026-05-18.
- See `download_log.json` for the full URL/size manifest.
- Some papers were chunked into per-3-page PDFs under `pages/` to support deep reading (see `*_manifest.txt` files).
