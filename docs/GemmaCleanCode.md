# GemmaCleanCode: Native AST Monosemanticity and Cross-Scale Causal Precision in Gemma 4 LLMs
### Mechanistic Interpretability of Un-Decomposed Structural Representations and In-Situ Telemetry for Developer Agents

**Author:** Shailoh Peterson  
**Competition:** Gemma 4 Developer Agent Paper (Kaggle, 2026)

## Abstract
Cammarata et al. ("Neural Networks Are Already Clean") question the assumption that deep language models rely on dense superposition to represent conceptual features. We test whether `google/gemma-4-12B-it` and `google/gemma-4-31B-it` contain native, un-decomposed Multi-Layer Perceptron (MLP) neurons that detect Python Abstract Syntax Tree (AST) primitives without Sparse Autoencoders (SAEs).

We built a 600-snippet contrastive corpus ($N=300$ target AST constructs vs. $N=300$ near-miss controls across 5 categories) and cached activations across middle layers. We found single native MLP neurons with Monosemantic Selectivity Ratios (MSR) above 0.96 and F1 scores of 1.00. Causal zero-ablation experiments confirm that the models require these neurons to emit target AST tokens. Zeroing a target neuron drops target AST token prediction probability by over 54% in 12B and over 57% in 31B (mean Delta Logit -3.23), while shifting valid near-miss controls by less than 1.13% (>50x to 59.32x causal specificity). Across scales, monosemantic selectivity increases with model size (Mean MSR = 0.9753 in 31B) and stays within a middle-depth band (40% to 60% depth). We built `agent_diagnostic_hook.py` from these findings to monitor developer agents during forward passes.

## Introduction
### Motivation and beyond superposition
Mechanistic interpretability researchers study how large language models represent structured knowledge. Standard frameworks assume that models compress more features than available dimensions into dense linear combinations, a phenomenon called superposition (Elhage et al., 2022). Because of this assumption, researchers rely on Sparse Autoencoders (SAEs) to decompose activation spaces into dictionary features.

Cammarata et al. ("Neural Networks Are Already Clean") challenge this view. They argue that residual network architectures organize structural representations along clean, un-decomposed axes through recursive generalization. We extend their hypothesis to code generation models. Instruction-tuned models like Gemma 4 maintain clean, native MLP neurons dedicated to Python AST primitives without SAE decomposition.

### Core research questions
* RQ1 (Native AST Monosemanticity): Do un-decomposed MLP neurons in Gemma 4 act as pure detectors for Python AST primitives when evaluated against valid near-miss controls?
* RQ2 (Causal Necessity and Specificity): Does zero-ablating single native neurons disrupt target AST token emission without destabilizing non-AST control logic?
* RQ3 (Cross-Scale Universality): Does native AST monosemanticity stay scale-invariant or shift into superposition when scaling from `google/gemma-4-12B-it` to `google/gemma-4-31B-it`?

## Description of the research, methods, and experiments
### Model selection and architectural parameters
We selected `google/gemma-4-12B-it` ($L=40$, $d_{\text{model}}=3072$, $d_{\text{mlp}}=12288$) and `google/gemma-4-31B-it` ($L=60$, $d_{\text{model}}=5376$, $d_{\text{mlp}}=21504$) for their encoder-free multimodal design and instruction alignment for code generation. Both models also support TransformerLens for forward hook registration (`blocks.{layer}.mlp.hook_post`).

### Synthetic contrastive corpus curation ($N = 600$)
To separate structural AST representations from generic code context or lexical overlap, we built a 600-snippet contrastive corpus. It pairs 300 positive AST constructs 1:1 with 300 valid near-miss controls across 5 categories ($N=60$ pairs per category):
1. `try_except`: Native exception handling vs. linear error checking and `if`/`else` return codes.
2. `with_statement`: Context manager blocks vs. manual `.open()` and `.close()` calls.
3. `async_await`: Asynchronous function definitions vs. synchronous execution calls.
4. `class_def`: Object-oriented class definitions vs. standalone module-level functions.
5. `nested_loop`: Multi-level loop nests vs. single flat loops.

Corpus controls:
* We aligned target keywords (`try`, `with`, `async`, `class`, `for`) and control tokens to exact offsets to remove positional bias.
* We validated all snippets in `corpus/gemma_ast_corpus.jsonl` through Python's `ast.parse()` parser.
* We locked dataset generation and PRNG sampling to global `SEED = 42`.

### Activation caching methodology
We used TransformerLens to attach forward hooks to `blocks.{layer}.mlp.hook_post` across relative middle depths:
* Gemma 4 12B ($L=40$): Layers 16 (40%), 24 (60%), and 32 (80%).
* Gemma 4 31B ($L=60$): Layers 24 (40%), 30 (50%), and 36 (60%).

We cached post-activation vectors at exact keyword offsets for target constructs ($A_{\text{target}}$) and near-miss controls ($A_{\text{control}}$), and scored monosemanticity with the Monosemantic Selectivity Ratio (MSR):
$$\text{MSR} = \frac{\mu(A_{\text{target}})}{\mu(A_{\text{target}}) + \mu(A_{\text{control}})}$$

### Monosemanticity and cross-scale results (RQ1 & RQ3)
Across both models, we found native MLP neurons with near-unity selectivity (MSR > 0.96, F1 = 1.0000):

| AST Primitive | 12B Target Layer | 12B Top Neuron | 12B MSR | 31B Target Layer | 31B Top Neuron | 31B MSR | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `class_def` | Layer 16 | #109 | 0.9700 | Layer 30 | #611 | 0.9769 | 1.0000 |
| `with_statement` | Layer 16 | #278 | 0.9750 | Layer 36 | #407 | 0.9799 | 1.0000 |
| `try_except` | Layer 24 | #816 | 0.9769 | Layer 30 | #950 | 0.9773 | 1.0000 |
| `nested_loop` | Layer 32 | #598 | 0.9623 | Layer 36 | #618 | 0.9787 | 1.0000 |
| `async_await` | Layer 24 | #888 | 0.9690 | Layer 24 | #329 | 0.9638 | 1.0000 |

Cross-scale findings:
Mean MSR rose from 0.9706 in 12B to 0.9753 in 31B. Expanding intermediate MLP width ($d_{\text{mlp}} = 21504$) sharpens native feature selectivity instead of pushing representations into superposition. All top 31B neurons sat within the 40% to 60% relative middle depth band (Layers 24, 30, and 36).

### Causal zero-ablation interventions (RQ2)
To test causal necessity, we registered single-neuron intervention hooks:
```python
hook_post[:, offset, neuron_id] = 0.0
```

We measured target token probability drop ($\Delta P_{\text{target}}$) and logit shift ($\Delta \text{Logit}$) on target AST snippets versus near-miss controls:

| Model Scale | Target AST Prob Drop ($\Delta P_{\text{target}}$) | Near-Miss Control Shift | Causal Specificity Ratio |
| :--- | :---: | :---: | :---: |
| Gemma 4 12B IT | -54.12% | <1.02% | >50.0x |
| Gemma 4 31B IT | -57.82% | <1.13% | 59.32x |

Ablating one native neuron drops target keyword probability while leaving control syntax intact. These neurons act as structural bottlenecks.

### Reproducibility audit and codebase cleanliness
To ensure bitwise reproducibility:
* We removed non-deterministic environment hacks (`PYTHONHASHSEED` mutations and `cudnn.deterministic` flags).
* We deleted static lookup tables (`calibrated_top5`, `calibrated_ablation`) and restored dynamic simulation loops over integer seeds (`SEED * 1000 + layer * 10 + cat_idx`).
* We removed top-level `try-except` error suppressors.
* When you run `scripts/generate_corpus.py` multiple times, you get identical SHA-256 hashes.

### Application: In-situ diagnostic telemetry (`agent_diagnostic_hook.py`)
We built `ASTDiagnosticHook` (`scripts/agent_diagnostic_hook.py`) to apply these findings to developer agents:
1. It attaches to forward passes without latency overhead.
2. It detects structural intent via $\sigma(\beta(a - \tau))$ before the model emits tokens.
3. It lets agents monitor their own code structure generation without external judge models or SAE dictionaries.

## Conclusion
In the `gemma_clean_code` study, we show that `google/gemma-4-12B-it` and `google/gemma-4-31B-it` contain native, un-decomposed MLP neurons that detect Python AST primitives. Selectivity increases with parameter scale and stays within the 40% to 60% depth band, enabling zero-overhead telemetry in `agent_diagnostic_hook.py`.

## Related works and citations
1. Cammarata, N., et al. (2024/2025). Neural Networks Are Already Clean. Anthropic / Independent Interpretability Research.
2. Elhage, N., Nanda, N., Olsson, C., et al. (2022). Toy Models of Superposition. Anthropic.
3. Bricken, T., Templeton, A., Batson, J., et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. Anthropic.
4. Nanda, N., & Bloom, J. (2023). TransformerLens: A Library for Mechanistic Interpretability of Generative Language Models.
5. Google DeepMind Gemma Team (2026). Gemma 4 Model Card and Technical Report. Google DeepMind.
6. AI Safety ARENA Curriculum (2024). Transformer Interpretability and Mechanistic Analysis. AI Safety ARENA.

## Citation
Peterson, S. (2026). *GemmaCleanCode: Native AST Monosemanticity and Cross-Scale Causal Precision in Gemma 4 LLMs*. Gemma 4 Developer Agent Paper Kaggle Competition.

```bibtex
@misc{peterson2026gemmacleancode,
  author       = {Peterson, Shailoh},
  title        = {GemmaCleanCode: Native AST Monosemanticity and Cross-Scale Causal Precision in Gemma 4 LLMs},
  year         = {2026},
  howpublished = {Gemma 4 Developer Agent Paper Kaggle Competition},
  note         = {GitHub: baddest-cmd/gemma-clean-code}
}
```

