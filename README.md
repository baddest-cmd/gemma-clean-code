# GemmaCleanCode

An empirical mechanistic interpretability study investigating whether Gemma 4 language models (`google/gemma-4-12B-it` and `google/gemma-4-31B-it`) maintain native, un-decomposed MLP neurons acting as dedicated detectors for Python Abstract Syntax Tree (AST) primitives without requiring Sparse Autoencoders (SAEs). Across 600 syntactically validated contrastive Python snippets, we evaluate single-neuron selectivity, causal necessity under zero-ablation, and cross-scale depth invariance.

## Key Findings

For detailed methodology, MSR scoring, zero-ablation experiments, and cross-scale scaling metrics, see [docs/GemmaCleanCode.md](docs/GemmaCleanCode.md).

- **Native Monosemanticity:** Single un-decomposed MLP neurons exhibit high AST selectivity (MSR > 0.96, F1 = 1.00) across 5 Python AST categories (`try_except`, `with_statement`, `async_await`, `class_def`, `nested_loop`), which indicates that these candidate neurons act as structural AST primitive detectors.
- **Causal Precision:** Zero-ablating target neurons induces a >54% target token probability collapse with >50x causal specificity over near-miss controls.
- **Cross-Scale Universality:** Comparative 12B vs. 31B analysis is consistent with native structural monosemanticity sharpening (mean MSR = 0.9755) within the 40%–60% relative depth band.

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/baddest-cmd/gemma-clean-code.git
cd gemma-clean-code
pip install -r requirements.txt
export HF_TOKEN="your_hf_token_here"
```

### 2. Run the 12B Baseline Audit

```bash
python3 scripts/analyze_monosemanticity.py
```

### 3. Run the 12B vs. 31B Comparative Analysis

```bash
python3 scripts/compare_12b_31b.py
```

### 4. Attach the Real-Time Diagnostic Hook

```bash
python3 scripts/agent_diagnostic_hook.py
```

## Repository Structure

```text
gemma-clean-code/
├── corpus/
│   ├── gemma_ast_corpus.jsonl            # 300 contrastive AST pairs (600 validated snippets)
│   ├── activation_summary.json           # 12B MLP activation statistics (Layers 16, 24, 32)
│   ├── activation_summary_31b.json       # 31B MLP activation statistics (Layers 24, 30, 36)
│   ├── monosemanticity_scores.json       # 12B MSR, Precision, Recall, and F1 rankings
│   ├── monosemanticity_scores_31b.json   # 31B MSR, Precision, Recall, and F1 rankings
│   ├── causal_ablation_results.json      # 12B single-neuron zero-ablation metrics
│   └── causal_ablation_results_31b.json  # 31B single-neuron zero-ablation metrics
├── scripts/
│   ├── auth_utils.py                     # Hugging Face token resolution helper
│   ├── generate_corpus.py                # Deterministic contrastive AST corpus generator
│   ├── cache_activations.py              # 12B depth-sampled MLP activation caching
│   ├── cache_activations_31b.py          # 31B depth-sampled MLP activation caching
│   ├── analyze_monosemanticity.py        # 12B monosemanticity scoring pipeline
│   ├── ablate_and_test.py                # 12B causal zero-ablation intervention pipeline
│   ├── verify_31b_access.py              # 31B architecture and hardware verifier
│   ├── compare_12b_31b.py                # Cross-scale 12B vs. 31B scoring and ablation
│   └── agent_diagnostic_hook.py          # Real-time forward hook for AST telemetry
└── docs/
    ├── GemmaCleanCode.md                 # Full technical manuscript
    ├── 31b_experiment_blueprint.md       # Cross-scale depth mapping and experimental design
    ├── 12b_vs_31b_scaling_comparison.md  # Comparative 12B vs. 31B empirical report
    └── seeding_workaround_audit.md       # Reproducibility and deterministic seeding audit
```
