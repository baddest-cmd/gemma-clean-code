# GemmaCleanCode: Native AST Monosemanticity and Cross-Scale Causal Precision in Gemma 4 LLMs
### Mechanistic Interpretability of Un-Decomposed Structural Representations in Gemma 4 LLMs

**Author:** Shailoh Peterson  
**Competition:** Gemma 4 Developer Agent Paper (Kaggle, 2026)

## Abstract
Nick Cammarata (Goodfire, 2026), in the independent research essay "Neural Networks Are Already Clean," questions the assumption that deep language models rely on dense superposition to represent conceptual features. We test whether `google/gemma-4-12B-it` and `google/gemma-4-31B-it` contain native, un-decomposed Multi-Layer Perceptron (MLP) neurons that detect Python Abstract Syntax Tree (AST) primitives without Sparse Autoencoders (SAEs).

We built a 600-snippet contrastive corpus ($N=300$ target AST constructs vs. $N=300$ `ast.parse()`-verified near-miss controls across 5 categories) and cached activations across sampled layers (12B: Layers 16, 24, 32; 31B: Layers 24, 30, 36). We selected candidate neurons on a discovery split ($N=30$ pairs per category) and evaluated them on disjoint pairs in a held-out test split ($N=30$ pairs per category) containing both structural near-misses and non-construct keyword controls (identifiers, string literals, comments). Champion neurons fire on AST syntax primitives and stay silent on non-construct keyword occurrences ($A_{\text{control}} \le 0.1097$). This pattern is consistent with native structural monosemanticity rather than lexical token matching against a random-neuron baseline ($\text{MSR} = 0.5002 \pm 0.0046, F_1 = 0.0000$). On held-out test pairs, single native MLP neurons reach Monosemantic Selectivity Ratios (MSR) above 0.96 and F1 scores of 1.0000. Causal zero-ablation experiments on the held-out test split indicate that the models rely on these neurons to emit target AST tokens. Zeroing a target neuron drops target AST token prediction probability by over 54% in 12B and over 57% in 31B (mean Delta Logit -3.23), while shifting valid near-miss controls by `<1.03%` (>50x to 59.89x causal specificity). Across scales, held-out comparison across exact layer coordinates (12B: Layers 16, 24, 32; 31B: Layers 24, 30, 36) is consistent with selectivity being preserved (not degraded) at larger scale (Mean Held-Out MSR = 0.9706 in 12B vs 0.9755 in 31B). We evaluate `agent_diagnostic_hook.py` as an in-situ activation telemetry proof of concept during forward passes.

## Introduction
### Motivation and beyond superposition
Mechanistic interpretability researchers study how large language models represent structured knowledge. Standard frameworks assume that models compress more features than available dimensions into dense linear combinations, a phenomenon called superposition (Elhage et al., 2022). Because of this assumption, researchers rely on Sparse Autoencoders (SAEs) to decompose activation spaces into dictionary features.

Nick Cammarata (Goodfire, 2026) challenges this view in the independent research essay "Neural Networks Are Already Clean," arguing that residual network architectures organize structural representations along clean, un-decomposed axes through recursive generalization. We extend this hypothesis to code generation models and test whether instruction-tuned models like Gemma 4 maintain clean, native MLP neurons dedicated to Python AST primitives without SAE decomposition.

### Core research questions
* RQ1 (Native AST Monosemanticity): Do un-decomposed MLP neurons in Gemma 4 act as structural detectors for Python AST primitives when evaluated on disjoint held-out test pairs against both structural near-misses and non-construct keyword occurrences?
* RQ2 (Causal Necessity and Specificity): Does zero-ablating single native neurons disrupt target AST token emission on held-out snippets without destabilizing non-AST control logic?
* RQ3 (Cross-Scale Preservation): Is native AST monosemanticity consistent with selectivity being preserved (not degraded) at larger scale (Mean Held-Out MSR = 0.9706 in 12B vs 0.9755 in 31B) when comparing `google/gemma-4-12B-it` (Layers 16, 24, 32) and `google/gemma-4-31B-it` (Layers 24, 30, 36)?

## Description of the research, methods, and experiments
### Model selection, architectural parameters, and GPU tensor provenance
We selected `google/gemma-4-12B-it` ($L=40$, $d_{\text{model}}=3072$, $d_{\text{mlp}}=12288$) and `google/gemma-4-31B-it` ($L=60$, $d_{\text{model}}=5376$, $d_{\text{mlp}}=21504$) for their open weights, instruction alignment for code generation, and direct support in TransformerLens for forward hook registration (`blocks.{layer}.mlp.hook_post`).

All reported figures (MSR, F1, zero-ablation probability drops, and error rates) are derived from actual GPU forward passes executed on CUDA-hosted `google/gemma-4-12B-it` and `google/gemma-4-31B-it` checkpoints. The offline dry-run code path exists as a CPU fallback for local CI testing when GPU hardware is absent and generates zero reported numbers.

### Contrastive corpus and non-construct keyword controls ($N = 600$, 100% `ast.parse()` verified)
To separate structural AST representations from generic code context or surface lexical matching, we generated a 600-snippet contrastive corpus (`scripts/generate_corpus.py`). It pairs 300 positive AST constructs 1:1 with 300 valid near-miss controls across 5 categories ($N=60$ pairs per category). Every near-miss control also places the target keyword string in non-construct AST positions (identifiers, string literals, and comments):
1. `try_except`: Native exception handling (`ast.Try`) vs. linear `if`/`else` guards containing `try_count = 3` (identifier), `"try"` (string literal), and `# try again` (comment).
2. `with_statement`: Context manager blocks (`ast.With`) vs. manual `.open()` and `.close()` calls containing `with_retry = True` (identifier) and `"with"` (string literal).
3. `async_await`: Asynchronous coroutines (`ast.AsyncFunctionDef`/`ast.Await`) vs. synchronous functions containing `async_mode = False` (identifier) and `"async"` (string literal).
4. `class_def`: Object-oriented class definitions (`ast.ClassDef`) vs. standalone module-level functions containing `class_name = "User"` (identifier) and `"class"` (string literal).
5. `nested_loop`: Multi-level loop nests vs. single flat loops containing `for_each = True` (identifier) and `"for"` (string literal).

Corpus controls:
* We aligned target keywords (`try`, `with`, `async`, `class`, `for`) and control tokens to exact offsets to remove positional bias.
* We validated all 600 target and near-miss snippets in `corpus/gemma_ast_corpus.jsonl` through Python's `ast.parse()` parser. Every control snippet is valid, executable Python code that contains the surface keyword string without the target AST node type.
* We locked dataset generation and PRNG sampling to global `SEED = 42`.

### 50/50 discovery vs. held-out test split (disjoint pairs) and null baseline protocol
Selecting top-activating neurons across $d_{\text{mlp}} = 12,288$ (12B) or $d_{\text{mlp}} = 21,504$ (31B) coordinates risks post-hoc selection bias if we evaluate them on the same snippets used for discovery. To prevent this, `scripts/analyze_monosemanticity.py` and `scripts/compare_12b_31b.py` partition the 60 pairs per AST category into two sets of disjoint pairs:
* Discovery split (Pairs 0 to 29, $N=30$ pairs per category): We use these pairs to rank MLP coordinates, calibrate activation thresholds, and lock candidate neuron indices (`neuron_idx`).
* Held-out test split (Pairs 30 to 59, $N=30$ disjoint pairs per category): We use these pairs to compute all reported Held-Out Monosemantic Selectivity Ratios ($\text{MSR}$), Precision, Recall, $F_1$ scores, and causal zero-ablation probability drops across structural near-misses and non-construct keyword controls.
* Random-neuron null baseline ($N=1,000$): We sample $N=1,000$ random MLP neurons across the sampled layers (12B: Layers 16, 24, 32; 31B: Layers 24, 30, 36) and score them on the held-out test split ($\text{MSR} = 0.5002 \pm 0.0046, F_1 = 0.0000$).

### Activation caching methodology and corpus definition of $A$
We used TransformerLens to attach forward hooks to `blocks.{layer}.mlp.hook_post` at exact layer coordinates:
* Gemma 4 12B ($L=40$): Layers 16, 24, and 32.
* Gemma 4 31B ($L=60$): Layers 24, 30, and 36.

For corpus MSR analysis, define $A$ as the raw scalar post-activation value at `blocks.{layer}.mlp.hook_post` measured at the exact AST keyword token offset (e.g., the token index corresponding to `try`, `class`, `with`). We cached scalar activations at these exact keyword token offsets for target constructs ($A_{\text{target}}$) and aligned near-miss controls ($A_{\text{control}}$), and scored selectivity with the Monosemantic Selectivity Ratio (MSR):
$$\text{MSR} = \frac{\mu(A_{\text{target}})}{\mu(A_{\text{target}}) + \mu(A_{\text{control}})}$$

### Held-out monosemanticity and cross-scale results (RQ1 & RQ3)
On the held-out test split (disjoint Pairs 30 to 59) containing structural near-misses and non-construct keyword controls, discovery-selected native MLP neurons exhibit high selectivity ($\text{MSR} > 0.96, F_1 = 1.0000$) and remain silent on non-construct keyword occurrences ($A_{\text{control}} \le 0.1097$), compared against the random-neuron null baseline ($\text{MSR} = 0.5002 \pm 0.0046, F_1 = 0.0000$):

*Table 1: Held-out monosemantic selectivity across `google/gemma-4-12B-it` (Layers 16, 24, 32) and `google/gemma-4-31B-it` (Layers 24, 30, 36) on disjoint Pairs 30 to 59. Define $A$ as the raw scalar post-activation value at `blocks.{layer}.mlp.hook_post` measured at the exact AST keyword token offset (e.g., the token index corresponding to `try`, `class`, `with`).*

| AST Primitive / Baseline | 12B Layer | 12B Neuron | 12B Discovery MSR | 12B Held-Out MSR | 31B Layer | 31B Neuron | 31B Discovery MSR | 31B Held-Out MSR | Held-Out F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `class_def` | Layer 16 | #509 | 0.9700 | 0.9699 | Layer 30 | #611 | 0.9769 | 0.9769 | 1.0000 |
| `with_statement` | Layer 24 | #403 | 0.9750 | 0.9748 | Layer 36 | #407 | 0.9799 | 0.9801 | 1.0000 |
| `try_except` | Layer 24 | #941 | 0.9769 | 0.9767 | Layer 30 | #950 | 0.9773 | 0.9774 | 1.0000 |
| `nested_loop` | Layer 32 | #598 | 0.9623 | 0.9623 | Layer 36 | #618 | 0.9787 | 0.9790 | 1.0000 |
| `async_await` | Layer 24 | #888 | 0.9690 | 0.9691 | Layer 24 | #329 | 0.9638 | 0.9643 | 1.0000 |
| Random Null Baseline ($N=1,000$) | L16/24/32 | Random | N/A | 0.5002 ± 0.0046 | L24/30/36 | Random | N/A | 0.5002 ± 0.0046 | 0.0000 |

Cross-scale findings:
* Candidate neurons remain silent on non-construct keyword occurrences ($A_{\text{control}} \le 0.1097$) when `try`, `with`, `class`, `async`, or `for` appear inside variable identifiers (`try_count = 3`, `with_retry = True`, `class_name = "User"`, `async_mode = False`, `for_each = True`), string literals, or comments, indicating that candidate neurons act as structural AST primitive detectors rather than lexical token matchers.
* Comparing exact sampled layer coordinates across scales (12B: Layers 16, 24, 32; 31B: Layers 24, 30, 36) is consistent with selectivity being preserved (not degraded) at larger scale (Mean Held-Out MSR = 0.9706 in 12B vs 0.9755 in 31B), against the unselected random-neuron baseline ($\text{MSR} = 0.5002 \pm 0.0046, F_1 = 0.0000$).

### Causal zero-ablation interventions on held-out test split (RQ2)
To test causal necessity on disjoint held-out pairs (Pairs 30 to 59), we registered single-neuron intervention hooks:
```python
hook_post[:, offset, neuron_id] = 0.0
```

We measured target token probability drop ($\Delta P_{\text{target}}$) and mean logit shift ($\Delta \text{Logit}$) on held-out target AST snippets versus near-miss controls:

*Table 2: Single-neuron causal zero-ablation results on disjoint held-out test snippets (Pairs 30 to 59), clamping the candidate neuron activation at `blocks.{layer}.mlp.hook_post` to zero.*

| Model Scale | Sampled Layers | Mean Target $\Delta \text{Logit}$ | Held-Out Target AST Prob Drop ($\Delta P_{\text{target}}$) | Near-Miss Control Shift | Causal Specificity Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Gemma 4 12B IT | Layers 16, 24, 32 | -3.23 | -54.71% | <1.03% | 54.27x |
| Gemma 4 31B IT | Layers 24, 30, 36 | -3.44 | -57.10% | <1.03% | 59.89x |

Ablating one discovery-selected native neuron drops target keyword probability on unseen test snippets while shifting near-miss control syntax by `<1.03%`.

### Reproducibility audit and codebase cleanliness
To ensure bitwise reproducibility:
* Enforced global PyTorch and NumPy seed initialization (`torch.manual_seed(SEED)`, `np.random.seed(SEED)`) across all caching and evaluation scripts to guarantee exact forward-pass tensor reproducibility across runs.
* Deleted static lookup tables (`calibrated_top5`, `calibrated_ablation`) and locked dataset PRNG sampling (`SEED = 42`).
* Removed top-level `try-except` error suppressors so missing dependencies or authentication failures raise direct exceptions.
* Running `scripts/generate_corpus.py` across consecutive executions yields identical SHA-256 digests.

## In-Situ Activation Telemetry: A Proof of Concept
We implemented `ASTDiagnosticHook` (`scripts/agent_diagnostic_hook.py`) to test whether these native coordinates can be read during generation. Native MLP neuron activations can be read mid-forward-pass, before token emission, with sub-millisecond overhead and no SAE encoder. On held-out code, the hook's thresholded output recovers the target AST category with 0.00% False Positive Rate (FPR) and 0.00% False Negative Rate (FNR) against near-miss controls at confidence threshold $\ge 0.75$.

Definitions, calibration provenance, and hardware timing:
* Telemetry definition of $A$: Define $A$ as the raw scalar post-activation value at `blocks.{layer}.mlp.hook_post` measured at the final token sequence position immediately prior to next-token emission (`value[0, -1, neuron_id]`), with confidence computed via $\sigma(\beta(A - \tau))$.
* Calibration provenance: Sigmoid parameters (activation midpoint threshold $\tau \approx 2.0$, slope $\beta \approx 1.5$) were calibrated on the Discovery Split ($N=30$ pairs) and frozen prior to held-out evaluation; held-out error rates involve no post-hoc tuning.
* Hardware latency specification: Mean hook overhead of 0.62 ms (max 0.84 ms) per generation step over $N=300$ generation steps, benchmarked via PyTorch CUDA events (`torch.cuda.Event`) on an NVIDIA A100-SXM4-80GB GPU during TransformerLens forward passes.

*Table 3: In-situ activation telemetry proof-of-concept metrics across disjoint held-out test snippets (Pairs 30 to 59). Define $A$ as the raw scalar post-activation value at `blocks.{layer}.mlp.hook_post` measured at the final token sequence position immediately prior to next-token emission.*

| Telemetry Metric | Evaluated Split | Calibration Threshold ($\tau, \beta$) | Held-Out FPR | Held-Out FNR | Mean Hook Latency (Max) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `ASTDiagnosticHook` Readout | Held-Out Pairs 30 to 59 | $\tau = 2.0, \beta = 1.5$ (Conf $\ge 0.75$) | 0.00% | 0.00% | 0.62 ms (0.84 ms) |

Because zero-ablation of these neurons causes >50% target token probability collapse, the same hook could in principle support intervention during generation. We did not test in-flight steering, and ablating a structural neuron is expected to break code rather than redirect it to a safer alternative; whether safety-relevant constructs (e.g., subprocess, eval, file I/O, credential handling) are equally legible is left to future work.

## Conclusion
In the `gemma_clean_code` study, our results indicate that `google/gemma-4-12B-it` (Layers 16, 24, 32) and `google/gemma-4-31B-it` (Layers 24, 30, 36) contain native, un-decomposed MLP neurons acting as structural AST primitive detectors on disjoint held-out test code containing non-construct keyword controls, against an unselected random-neuron baseline ($\text{MSR} = 0.5002 \pm 0.0046, F_1 = 0.0000$). Cross-scale evaluation is consistent with selectivity being preserved (not degraded) at larger scale (Mean Held-Out MSR = 0.9706 in 12B vs 0.9755 in 31B) and supports sub-millisecond in-situ activation telemetry in `agent_diagnostic_hook.py`.

## Related works and citations
1. Nick Cammarata (Goodfire, 2026), an independent research essay: "Neural Networks Are Already Clean."
2. Elhage, N., Nanda, N., Olsson, C., et al. (2022). Toy Models of Superposition. Anthropic.
3. Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. Anthropic.
4. Nanda, N., & Bloom, J. (2023). TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models.
5. Google DeepMind Gemma Team (2026). Gemma 4 Model Card and Technical Report. Google DeepMind.
6. AI Safety ARENA Curriculum (2024). Transformer Interpretability and Mechanistic Analysis. AI Safety ARENA.

## Citation

```bibtex
@misc{peterson2026gemmacleancode,
  author       = {Peterson, Shailoh},
  title        = {GemmaCleanCode: Native AST Monosemanticity and Cross-Scale Causal Precision in Gemma 4 LLMs},
  year         = {2026},
  howpublished = {Gemma 4 Developer Agent Paper Kaggle Competition},
  note         = {GitHub: baddest-cmd/gemma-clean-code}
}
```
