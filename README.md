# GemmaCleanCode

An empirical mechanistic interpretability study investigating whether Gemma 4 LLMs (`google/gemma-4-12B-it` and `google/gemma-4-31B-it`) maintain native, un-decomposed MLP neurons for Python Abstract Syntax Tree (AST) primitives without Sparse Autoencoders (SAEs).

## Key Findings

For detailed methodology, held-out MSR scoring, zero-ablation experiments, and cross-scale metrics, see [docs/GemmaCleanCode.md](docs/GemmaCleanCode.md).

* **Structural Monosemanticity**: In held-out evaluations, single un-decomposed MLP neurons exhibit near-unity selectivity (held-out MSR > 0.96, F1 = 1.0000) for AST primitives (`try_except`, `with_statement`, `class_def`, `async_await`, `nested_loop`). Candidate neurons remain silent on non-construct keyword controls (identifiers, comments, string literals), which indicates structural syntax detection rather than lexical token matching.
* **Causal Necessity**: Zero-ablating candidate neurons on unseen test snippets induces a >54% (12B) and >57% (31B) target token probability collapse with >50x causal specificity over near-miss controls.
* **Out-of-Sample Validation & Null Baseline**: Candidate neurons selected on a discovery split (N=30 pairs) generalize out-of-sample with zero score degradation on a held-out test split (N=30 pairs), standing in stark contrast to an unselected random-neuron baseline (MSR = 0.5002, F1 = 0.0000).

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

# Attach in-situ streaming telemetry hook
python3 scripts/agent_diagnostic_hook.py
```

## Repository Structure

```text
.
├── corpus/   # 600 contrastive snippets & evaluation logs
├── scripts/  # Caching, MSR scoring, ablation, & telemetry hooks
└── docs/     # Full manuscript writeup (docs/GemmaCleanCode.md)
```
