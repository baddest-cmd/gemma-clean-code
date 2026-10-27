# gemma_clean_code: Native MLP AST Monosemanticity in Gemma

An empirical mechanistic interpretability study investigating whether native Multi-Layer Perceptron (MLP) neurons and activation patterns in `google/gemma-4-12B-it` exhibit monosemantic representations of Abstract Syntax Tree (AST) structures and grammatical transitions during code synthesis.

Prepared for the **Gemma 4 Paper Track**.

---

## 🎯 Research Objectives

1. **AST Representation Probing**: Pinpoint where and how structural syntax trees (function scopes, control flow blocks, variable bindings) are encoded within Gemma's native representations without retraining or SAE superposition distortion.
2. **Native MLP Monosemanticity**: Quantify the degree of monosemanticity in individual intermediate MLP features when executing structured code generation tasks.
3. **Causal Intervention & Ablation**: Validate whether ablating or clamping syntax-specific neuron clusters causally disrupts syntax validity while preserving semantic intent.

---

## 📁 Repository Layout

```text
gemma_clean_code/
├── corpus/        # Synthetic AST datasets and structured code corpora
├── scripts/       # Generation, feature extraction, and activation caching utilities
├── notebooks/     # Colab experiment notebooks and interactive probing pipelines
├── docs/          # Technical specifications, research methodology, and paper drafts
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Installation

Ensure Python 3.10+ and PyTorch (with MPS/CUDA support) are installed:

```bash
git clone https://github.com/baddest-cmd/gemma-clean-code.git
cd gemma-clean-code
pip install -r requirements.txt
```

### 2. Experiment Setup

Target Model: `google/gemma-4-12B-it` (via Hugging Face Transformers & TransformerLens)

```bash
# Set your Hugging Face access token for Gemma gated weights
export HF_TOKEN="your_hf_token_here"
```

---

## 🔬 Methodology

- **Corpus Generation (`corpus/`)**: Controlled micro-benchmarks with targeted syntactical variations (nested loops, lambda expressions, try-except trees).
- **Activation Caching (`scripts/`)**: Hooking Gemma-4 MLP activations across transformer blocks using TransformerLens.
- **Analysis & Visualization (`notebooks/`)**: Probing classifiers, cosine similarity matrices, and activation frequency distributions.

---

## 📜 Citation & License

Under active development for the Gemma 4 Paper Track. Distributed under the Apache 2.0 License.
