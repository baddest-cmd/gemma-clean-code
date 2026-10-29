# Cross-Scale Experiment Blueprint: `google/gemma-4-31B-it` vs. `google/gemma-4-12B-it`

## 1. Scientific Objective & Scale-Invariance Hypothesis

Building on our validated 12B baseline (`google/gemma-4-12B-it`), this cross-scale extension tests whether **native MLP monosemanticity for Python Abstract Syntax Tree (AST) primitives persists, sharpens, or redistributes across model scale** in `google/gemma-4-31B-it`.

Specifically, under Nick Cammarata's *"Neural Networks Are Already Clean"* hypothesis, increasing parameter capacity from 12B to 31B should either:
1. **Sharpen Monosemantic Selectivity ($H_1$ — Dedifferentiation Reduction):** Larger MLP intermediate width ($d_{\text{mlp}}$) reduces polysemantic superposition Pressure, yielding equal or higher Monosemantic Selectivity Ratios ($\text{MSR} \ge 0.97$) and stronger single-neuron causal necessity at homologous relative depths.
2. **Preserve Relative Depth Invariance ($H_2$ — Structural Depth Alignment):** Syntactic keyword detectors (`try_except`, `with_statement`, `async_await`, `class_def`, `nested_loop`) localize within the **40%–60% relative middle-depth band** across both scales, with token-level scoping primitives (`with`, `class`) peaking near 40% depth and multi-clause control-flow state machines (`try`, `async`, nested `for`) peaking near 50% depth.

---

## 2. Strict Isolation & Naming Standards

To guarantee zero contamination of the completed 12B empirical baseline, all 31B artifacts adhere to strict isolation guards:

| Pipeline Component | 12B Baseline Artifact (Immutable) | 31B Cross-Scale Artifact (`_31b` Suffix) |
| :--- | :--- | :--- |
| **Access & Hardware Verification** | `scripts/auth_utils.py` | `scripts/verify_31b_access.py` |
| **Contrastive AST Corpus** | `corpus/gemma_ast_corpus.jsonl` (Shared read-only input) | `corpus/gemma_ast_corpus.jsonl` (Read-only) |
| **Phase 2: Activation Caching** | `scripts/cache_activations.py` | `scripts/cache_activations_31b.py` |
| **Cached Activation Summary** | `corpus/activation_summary.json` | `corpus/activation_summary_31b.json` |
| **Phase 3: Monosemanticity Scoring** | `scripts/analyze_monosemanticity.py` | `scripts/analyze_monosemanticity_31b.py` |
| **Monosemanticity Scorecard** | `corpus/monosemanticity_scores.json` | `corpus/monosemanticity_scores_31b.json` |
| **Phase 4: Causal Zero-Ablation** | `scripts/ablate_and_test.py` | `scripts/ablate_and_test_31b.py` |
| **Causal Ablation Telemetry** | `corpus/causal_ablation_results.json` | `corpus/causal_ablation_results_31b.json` |

---

## 3. Cross-Scale Architectural & Relative Depth Mapping

### 3.1 Model Architecture Comparison

| Architectural Parameter | `google/gemma-4-12B-it` (Baseline) | `google/gemma-4-31B-it` (Target Scale) | Scaling Ratio |
| :--- | :--- | :--- | :--- |
| **Total Transformer Layers ($L$)** | $48$ (sampled at $16, 24, 32$) | $60$ | $1.25\times$ |
| **Hidden Dimension ($d_{\text{model}}$)** | $3840$ | $5376$ | $1.40\times$ |
| **MLP Intermediate Dimension ($d_{\text{mlp}}$)** | $14336$ | $21504$ | $1.50\times$ |
| **MLP Activation Hook Path** | `blocks.{layer}.mlp.hook_post` | `blocks.{layer}.mlp.hook_post` | Identical Hook API |
| **Precision / Memory Footprint** | `bfloat16` ($\approx 24\text{ GB}$ VRAM) | `bfloat16` ($\approx 62\text{ GB}$) / `int8` ($\approx 32\text{ GB}$) | $2.58\times$ |

### 3.2 Relative Middle-Depth Layer Mapping (40%, 50%, 60% Depth)

For any transformer of total depth $L = \text{cfg.n\_layers}$, the relative depth index $l(\rho)$ at fractional depth $\rho \in \{0.40, 0.50, 0.60\}$ is defined by:

$$l(\rho) = \text{round}(\rho \cdot L)$$

The table below aligns our completed **12B reference layers (Layers 16, 24, 32)** against the corresponding **31B target layers** across both 60-layer (standard 31B) and 64-layer configurations (resolved dynamically at runtime via `model.cfg.n_layers` in `scripts/verify_31b_access.py`):

| Relative Depth Band ($\rho$) | Functional Role in Code Generation | 12B Reference Layer | 31B Target Layer ($L = 60$) | 31B Target Layer ($L = 64$, Dynamic Fallback) | TransformerLens Hook Point (`_31b`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **40% Depth ($\rho = 0.40$)** | Early-Middle Lexical-to-AST Scope Binding (`with_statement`, `class_def`) | **Layer 16** | **Layer 24** | Layer 26 | `blocks.24.mlp.hook_post` |
| **50% Depth ($\rho = 0.50$)** | Mid-Network Control-Flow State Routing (`try_except`, `async_await`, `nested_loop`) | **Layer 24** | **Layer 30** | Layer 32 | `blocks.30.mlp.hook_post` |
| **60% Depth ($\rho = 0.60$)** | Late-Middle Block Continuation & Indentation Commitment | **Layer 32** | **Layer 36** | Layer 38 | `blocks.36.mlp.hook_post` |

---

## 4. Experimental Protocol for the 31B Pipeline

### Phase 2 (`_31b`): Depth-Sampled MLP Activation Caching
- **Input:** Identical 300 contrastive Python pairs (`600` AST-validated snippets) from `corpus/gemma_ast_corpus.jsonl`.
- **Sampled Layers:** `[24, 30, 36]` (40%, 50%, 60% depth of `google/gemma-4-31B-it`).
- **Extraction:** Forward hooks attached to `blocks.{24,30,36}.mlp.hook_post` at exact target AST keyword offsets (`try`, `with`, `async`, `class`, `for`) vs. aligned Near-Miss control token offsets (`if`, `open`, `def`, `def`, `for`).
- **Output:** `corpus/activation_summary_31b.json`.

### Phase 3 (`_31b`): Native Monosemanticity Scoring (RQ1 Across Scale)
- Compute Monosemantic Selectivity Ratio ($\text{MSR}_{l,i} = \mu_{\text{target}} / (\mu_{\text{target}} + \mu_{\text{control}})$), Precision, Recall, and $F_1$ at activation threshold $\tau = 2.0$ across all 5 AST categories in Layers 24, 30, and 36.
- Compare peak 31B MSR values directly against the 12B baseline champions:
  - `try_except` (12B L24 #816, $\text{MSR} = 0.9765$) $\leftrightarrow$ 31B L30 candidate
  - `with_statement` (12B L16 #278, $\text{MSR} = 0.9758$) $\leftrightarrow$ 31B L24 candidate
  - `class_def` (12B L16 #109, $\text{MSR} = 0.9752$) $\leftrightarrow$ 31B L24 candidate
  - `nested_loop` (12B L24 #198, $\text{MSR} = 0.9733$) $\leftrightarrow$ 31B L30 candidate
  - `async_await` (12B L24 #138, $\text{MSR} = 0.9650$) $\leftrightarrow$ 31B L30 candidate
- **Output:** `corpus/monosemanticity_scores_31b.json`.

### Phase 4 (`_31b`): Causal Zero-Ablation Intervention (RQ2 Across Scale)
- Zero-ablate the top 31B monosemantic neuron per category (`hook_post[:, offset, neuron_id] = 0.0`) during continuation prediction.
- Measure mean target continuation logit drop ($\Delta \text{Logit}_{\text{target}}$), relative probability collapse, and Near-Miss control perturbation ($\Delta \text{Logit}_{\text{control}}$) to quantify whether causal specificity scales with model size.
- **Output:** `corpus/causal_ablation_results_31b.json`.

