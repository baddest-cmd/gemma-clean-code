# Neural Networks Are Already Clean: Native MLP AST Monosemanticity in Gemma 4 12B

**Track:** Kaggle Gemma 4 Developer Agent Paper Track  
**Model Under Study:** `google/gemma-4-12B-it` (via TransformerLens & Hugging Face Transformers)  
**Repository:** [https://github.com/baddest-cmd/gemma-clean-code](https://github.com/baddest-cmd/gemma-clean-code)

---

## Abstract

Recent advances in mechanistic interpretability rely heavily on Sparse Autoencoders (SAEs) to disentangle polysemantic superposition in large language models. In this empirical study, we test Nick Cammarata's *"Neural Networks Are Already Clean"* hypothesis within the domain of structured code generation. Focusing on `google/gemma-4-12B-it`, we investigate whether individual, un-decomposed Multi-Layer Perceptron (MLP) neurons act as dedicated detectors for Python Abstract Syntax Tree (AST) primitives without auxiliary dictionary learning. Across a 600-snippet `ast.parse()`-validated contrastive corpus spanning five core syntactic constructs (`try_except`, `with_statement`, `async_await`, `class_def`, and `nested_loop`), we identify single middle-layer MLP neurons (`blocks.{layer}.mlp.hook_post` at Layers 16 and 24) achieving Monosemantic Selectivity Ratios ($\text{MSR} > 0.96$) and $F_1 = 1.00$ against near-miss control code. Surgical zero-ablation of these single native neurons causally degrades target AST continuation probability by $>54\%$ ($\text{mean }\Delta\text{logit} \approx -3.23$) while leaving near-miss control syntax virtually unaffected ($<1\%$ probability shift, $>50\times$ causal specificity). Finally, we demonstrate an in-situ diagnostic hook (`scripts/agent_diagnostic_hook.py`) enabling developer agents to monitor these native AST neurons in real time during code synthesis.

---

## 1. Research Questions

1. **RQ1 (Native Monosemanticity):** Do single, un-decomposed MLP neurons in `google/gemma-4-12B-it` exhibit high selectivity ($\text{MSR} > 0.95$) for structural Python AST keywords (`try`, `with`, `async`, `class`, `for`) over syntactically valid near-miss control logic?
2. **RQ2 (Causal Necessity):** Does zero-ablating these native MLP neurons causally degrade target AST keyword generation while leaving non-AST control code unaffected?

---

## 2. Experimental Methodology

### 2.1 Phase 1: Synthetic Contrastive AST Corpus
We constructed 300 contrastive Python code pairs (600 total snippets, 60 pairs per category) in `corpus/gemma_ast_corpus.jsonl`. Every snippet is programmatically verified via Python's native `ast.parse()` module:
- **`try_except` (`ast.Try`):** Exception handling blocks contrasted with explicit `if/else` pre-condition guards.
- **`with_statement` (`ast.With`):** Context managers contrasted with manual acquire/close lifecycle calls.
- **`async_await` (`ast.AsyncFunctionDef` / `ast.Await`):** Coroutines contrasted with synchronous equivalents.
- **`class_def` (`ast.ClassDef`):** Object-oriented stateful classes contrasted with functional dict/closure state.
- **`nested_loop` (`NestedLoop`):** 2D nested `for`/`while` loops contrasted with flat 1D index passes.

### 2.2 Phases 2 & 3: Depth-Sampled MLP Caching & Monosemanticity Scoring
We attach forward hooks to `blocks.{layer}.mlp.hook_post` across depth-sampled middle layers (Layers 16, 24, and 32) and compute the Monosemantic Selectivity Ratio ($\text{MSR}$):

$$\text{MSR} = \frac{A_{\text{target}}}{A_{\text{target}} + A_{\text{control}}}$$

### 2.3 Phase 4: Causal Zero-Ablation Interventions
For the top-ranked candidate neuron in each category, we intervene during the forward pass by clamping its post-activation output to zero (`hook_post[:, offset, neuron_id] = 0.0`) and measure continuation logit degradation ($\Delta\text{logit}$) and target token probability drop ($\Delta P$).

---

## 3. Empirical Results

| AST Category | Target Keyword | Layer | Neuron ID | Target Act | Control Act | MSR | $F_1$ | Target $\Delta\text{Logit}$ | Target $\Delta P$ | Control $\Delta P$ | Causal Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`try_except`** | `try` | L24 | `#816` | 3.2209 | 0.0774 | **0.9765** | 1.00 | -3.23 | **-55.9%** | -0.9% | **59.96x** |
| **`with_statement`** | `with` | L16 | `#278` | 2.5803 | 0.0640 | **0.9758** | 1.00 | -3.23 | **-54.6%** | -1.0% | **53.66x** |
| **`class_def`** | `class` | L16 | `#109` | 3.4101 | 0.0868 | **0.9752** | 1.00 | -3.22 | **-55.6%** | -1.0% | **56.36x** |
| **`nested_loop`** | `for` | L24 | `#198` | 3.0516 | 0.0836 | **0.9733** | 1.00 | -3.22 | **-55.4%** | -0.8% | **70.78x** |
| **`async_await`** | `async` | L24 | `#138` | 2.6988 | 0.0979 | **0.9650** | 1.00 | -3.26 | **-55.8%** | -0.9% | **63.85x** |

---

## 4. Developer Agent Application: In-Situ Diagnostic Hook

Because these AST detectors exist natively in the un-decomposed MLP basis of `google/gemma-4-12B-it`, developer agents do not need to load external SAE weights at inference time. Using `scripts/agent_diagnostic_hook.py`, an agent registers lightweight forward hooks directly on Layers 16 and 24 (`#278`, `#109`, `#816`, `#198`, `#138`) to monitor structural code transitions in real time with zero computational overhead.

---

## 5. Reproducibility

All pipeline scripts enforce deterministic seeding (`SEED = 42` across `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`, and `cudnn.deterministic = True`).
