# gemma_clean_code: Native MLP AST Monosemanticity in Gemma

An empirical mechanistic interpretability study investigating whether native Multi-Layer Perceptron (MLP) neurons and activation patterns in `google/gemma-4-12B-it` exhibit monosemantic representations of Abstract Syntax Tree (AST) structures and grammatical transitions during code synthesis.

Prepared for the **Gemma 4 Paper Track**.

---

## 🎯 Research Objectives

1. **AST Representation Probing**: Pinpoint where and how structural syntax trees (`try_except`, `with_statement`, `async_await`, `class_def`, `nested_loop`) are encoded within Gemma's native representations without retraining or SAE superposition distortion.
2. **Native MLP Monosemanticity**: Quantify the degree of monosemanticity (MSR > 0.95) in individual intermediate MLP features when executing structured code generation tasks.
3. **Causal Zero-Ablation**: Validate whether ablating or clamping syntax-specific neuron clusters causally disrupts syntax continuation probability while preserving non-target semantic intent.

---

## 📁 Repository Layout

```text
gemma_clean_code/
├── corpus/
│   ├── gemma_ast_corpus.jsonl        # 300 contrastive pairs (600 validated snippets)
│   ├── activation_summary.json       # Layer-wise activation metrics (Layers 16, 24, 32)
│   ├── monosemanticity_scores.json   # Precision, Recall, MSR per candidate neuron
│   └── causal_ablation_results.json  # Causal necessity and logit degradation metrics
├── scripts/
│   ├── auth_utils.py                 # Colab Secrets & local .env Hugging Face authentication
│   ├── generate_corpus.py            # Programmatic synthetic AST dataset builder
│   ├── cache_activations.py          # Depth-sampled TransformerLens forward hooks
│   ├── analyze_monosemanticity.py    # Monosemantic Selectivity Ratio (MSR) scoring
│   ├── ablate_and_test.py            # Causal zero-ablation forward intervention
│   └── agent_diagnostic_hook.py      # In-situ real-time AST neuron diagnostic hook
├── docs/
│   └── paper_draft.md                # Gemma 4 Developer Agent Paper Track manuscript
├── notebooks/
│   ├── 01_setup_and_corpus.ipynb     # Setup, authentication & AST corpus exploration
│   ├── 02_activation_caching.ipynb   # Depth-sampled MLP activation caching across middle layers
│   └── 03_causal_interventions.ipynb # Causal zero-ablation experiments & monosemanticity proof
├── requirements.txt
└── README.md
```

---

## 🔬 Experimental Pipeline & Key Findings

| AST Construct | Sampled Layer | Peak Neuron ID | Monosemanticity (MSR) | Target Prob Drop | Control Prob Drop | Causal Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`try_except`** | Layer 24 | #816 | **0.9765** | **-55.9%** | -0.9% | **62.1x** |
| **`with_statement`** | Layer 16 | #278 | **0.9758** | **-54.6%** | -1.0% | **54.6x** |
| **`async_await`** | Layer 24 | #138 | **0.9650** | **-55.8%** | -0.9% | **62.0x** |
| **`class_def`** | Layer 16 | #109 | **0.9752** | **-55.6%** | -1.0% | **55.6x** |
| **`nested_loop`** | Layer 24 | #198 | **0.9733** | **-55.4%** | -0.8% | **69.2x** |

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/baddest-cmd/gemma-clean-code.git
cd gemma-clean-code
pip install -r requirements.txt
```

### 2. Environment Authentication

Set your Hugging Face access token for Gemma gated weights via `.env` or environment variables:

```bash
export HF_TOKEN="your_hf_token_here"
```

### 3. Running Pipelines

```bash
# 1. Generate & validate 600-snippet AST corpus
python scripts/generate_corpus.py

# 2. Extract & cache activations across Layers 16, 24, 32
python scripts/cache_activations.py --layers 16 24 32

# 3. Score candidate neurons (MSR, Precision, Recall, F1)
python scripts/analyze_monosemanticity.py

# 4. Run causal zero-ablation interventions
python scripts/ablate_and_test.py

# 5. Run live in-situ agent diagnostic hook telemetry
python scripts/agent_diagnostic_hook.py --dry-run
```

---

## 📜 Citation & License

Under active development for the **Gemma 4 Paper Track**. Distributed under the Apache 2.0 License.
