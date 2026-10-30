# GemmaCleanCode

An empirical mechanistic interpretability study investigating whether Gemma 4 LLMs (`google/gemma-4-12B-it` and `google/gemma-4-31B-it`) maintain native, un-decomposed MLP neurons for Python Abstract Syntax Tree (AST) primitives without Sparse Autoencoders (SAEs).

## Key Findings

For detailed methodology, held-out MSR scoring, zero-ablation experiments, and cross-scale metrics, see [docs/GemmaCleanCode.md](docs/GemmaCleanCode.md).

* **Structural Monosemanticity**: In held-out evaluations across exact layer coordinates (12B: Layers 16, 24, 32; 31B: Layers 24, 30, 36), single un-decomposed MLP neurons exhibit near-unity selectivity (held-out MSR > 0.96, F1 = 1.0000) for AST primitives (`try_except`, `with_statement`, `class_def`, `async_await`, `nested_loop`), where $A$ is the raw scalar post-activation value at `blocks.{layer}.mlp.hook_post` measured at the exact AST keyword token offset. Candidate neurons remain silent on non-construct keyword controls ($A_{\text{control}} \le 0.1097$ across identifiers, comments, and string literals), indicating that candidate neurons act as structural AST primitive detectors rather than lexical token matchers.
* **Causal Necessity**: Zero-ablating candidate neurons on unseen held-out test snippets induces a >54% (12B, mean Delta Logit -3.23) and >57% (31B, mean Delta Logit -3.44) target token probability collapse while shifting near-miss controls by <1.03% (>50x causal specificity).
* **Out-of-Sample Validation & Scale Preservation**: Candidate neurons selected on a discovery split (N=30 pairs per category) generalize out-of-sample across disjoint pairs in the held-out test split (N=30 pairs per category), in contrast to an unselected random-neuron baseline (MSR = 0.5002 ± 0.0046, F1 = 0.0000). Cross-scale comparison is consistent with selectivity being preserved (not degraded) at larger scale (Mean Held-Out MSR = 0.9706 in 12B vs 0.9755 in 31B).
* **In-Situ Activation Telemetry (Proof of Concept)**: Measuring $A$ as the raw scalar post-activation value at `blocks.{layer}.mlp.hook_post` at the final token sequence position immediately prior to next-token emission recovers the target AST category on held-out code with 0.00% FPR and 0.00% FNR at confidence threshold >= 0.75, with 0.62 ms mean hook overhead (max 0.84 ms on NVIDIA A100-SXM4-80GB).
* **Scope, Reproducibility & Limitations**: Evaluations span two model scales (`google/gemma-4-12B-it` and `31B-it`), five AST categories, three sampled layers per model (12B: L16, L24, L32; 31B: L24, L30, L36), and a template-generated contrastive corpus with global PyTorch and NumPy seed initialization (`torch.manual_seed(SEED)`, `np.random.seed(SEED)`) to ensure consistent sampling and corpus generation across runs. Whether safety-critical hazard constructs (`subprocess`, `eval`, file I/O, credential strings) exhibit similar native monosemantic legibility remains an open question for future work, and in-flight dynamic steering is deferred because zero-ablating structural syntax neurons induces probability collapse and breaks code syntax rather than redirecting generation toward a safe alternative.

## Quick Start

### Environment Setup

```bash
git clone https://github.com/baddest-cmd/gemma-clean-code.git
cd gemma-clean-code
pip install -r requirements.txt
```

### Reproduce Baseline & Scaling Studies

```bash
# Run 12B baseline evaluation on held-out split
python3 scripts/analyze_monosemanticity.py

# Run 12B vs 31B comparative scaling analysis
python3 scripts/compare_12b_31b.py

# Run in-situ activation telemetry hook proof of concept
python3 scripts/agent_diagnostic_hook.py
```

## Repository Structure

```text
.
├── corpus/   # 600 contrastive snippets & evaluation logs
├── scripts/  # Caching, MSR scoring, ablation, & telemetry hooks
└── docs/     # Full manuscript writeup (docs/GemmaCleanCode.md)
```
