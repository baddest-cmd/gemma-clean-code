"""Monosemanticity Scoring for google/gemma-4-12B-it AST Neurons.

Phase 3 of gemma_clean_code Mechanistic Interpretability Study.

Analyzes activation distributions from Phase 2 (corpus/activation_summary.json)
and individual snippet activations to quantify:
1. Monosemantic Selectivity Ratio (MSR):
   MSR = Act_target / (Act_target + Act_control)
2. Precision, Recall, and F1 score against out-of-category AST constructs.
3. Ranks top 5 native monosemantic neurons per AST category across Layers 16, 24, and 32.

Exports formatted scoring metrics to `corpus/monosemanticity_scores.json`.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Any

SEED = 42
random.seed(SEED)

logger = logging.getLogger("gemma_clean_code.scoring")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def compute_monosemanticity_metrics(
    target_act: float,
    control_act: float,
    other_cat_acts: List[float],
    threshold_factor: float = 0.5,
) -> Dict[str, float]:
    """Compute MSR, Precision, Recall, and F1 for a candidate neuron.

    Args:
        target_act: Mean activation on positive target AST snippets.
        control_act: Mean activation on near-miss control snippets.
        other_cat_acts: Mean activations on other AST categories.
        threshold_factor: Firing decision boundary as fraction of target_act.

    Returns:
        Dict containing msr, precision, recall, f1, and selectivity_ratio.
    """
    eps = 1e-6
    # 1. Monosemantic Selectivity Ratio (MSR)
    # Range: [0.0, 1.0]. Perfect monosemanticity approaches 1.0, polysemanticity ~ 0.5
    denom = max(target_act + control_act, eps)
    msr = target_act / denom if target_act > 0 else 0.0

    # Decision threshold for binary activation state
    firing_threshold = target_act * threshold_factor

    # Positive category hit (True Positive)
    tp = 1.0 if target_act >= firing_threshold else 0.0
    # Negative control misfire (False Positive)
    fp_control = 1.0 if control_act >= firing_threshold else 0.0
    # Other categories misfires (False Positives)
    fp_others = sum(1.0 for act in other_cat_acts if act >= firing_threshold)
    fp = fp_control + fp_others

    # False Negative (missed target)
    fn = 1.0 - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Cross-construct selectivity: target_act / (mean other categories + eps)
    mean_others = sum(other_cat_acts) / max(len(other_cat_acts), 1)
    selectivity = target_act / max(mean_others, eps)

    return {
        "msr": round(msr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "selectivity_ratio": round(selectivity, 4),
    }


def analyze_monosemanticity(
    summary_path: Path,
    output_path: Path,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Load activation summary and score candidate neurons for monosemanticity."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Activation summary not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        act_summary = json.load(f)

    logger.info("Scoring monosemanticity for model: %s", act_summary["model_name"])

    results_by_layer: Dict[str, Any] = {}
    global_top_candidates: Dict[str, List[Dict[str, Any]]] = {}

    for layer_key, layer_data in act_summary["layers"].items():
        layer_idx = layer_data["layer_index"]
        logger.info("Evaluating layer %d...", layer_idx)

        categories_data = layer_data["categories"]
        cat_names = list(categories_data.keys())
        scored_categories: Dict[str, Any] = {}

        for cat in cat_names:
            c_info = categories_data[cat]
            candidates = c_info.get("top_candidate_neurons", [])

            # Compute background mean activations across other categories for comparison
            other_cats = [c for c in cat_names if c != cat]
            other_acts = [
                categories_data[other]["top_candidate_neurons"][0]["mean_control_act"]
                for other in other_cats
            ]

            scored_candidates = []
            for cand in candidates:
                t_act = cand["mean_target_act"]
                c_act = cand["mean_control_act"]
                metrics = compute_monosemanticity_metrics(
                    target_act=t_act,
                    control_act=c_act,
                    other_cat_acts=other_acts,
                )

                score_entry = {
                    "neuron_idx": cand["neuron_idx"],
                    "layer": layer_idx,
                    "category": cat,
                    "target_act": t_act,
                    "control_act": c_act,
                    "delta": cand["delta"],
                    "msr": metrics["msr"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "selectivity_ratio": metrics["selectivity_ratio"],
                }
                scored_candidates.append(score_entry)

            # Sort by MSR descending, then delta descending
            scored_candidates.sort(key=lambda x: (x["msr"], x["delta"]), reverse=True)
            top_ranked = scored_candidates[:top_k]

            scored_categories[cat] = {
                "sample_pairs": c_info["sample_pairs"],
                "target_keyword": c_info["target_keyword"],
                "top_ranked_neurons": top_ranked,
            }

            # Collect for global category rankings
            if cat not in global_top_candidates:
                global_top_candidates[cat] = []
            global_top_candidates[cat].extend(top_ranked)

        results_by_layer[layer_key] = {
            "layer_index": layer_idx,
            "hook_path": layer_data["hook_path"],
            "categories": scored_categories,
        }

    # Deduplicate and sort global top candidates per category across all sampled layers
    final_ranked_by_category = {}
    for cat, cands in global_top_candidates.items():
        cands.sort(key=lambda x: (x["msr"], x["delta"]), reverse=True)
        final_ranked_by_category[cat] = cands[:top_k]

    output_payload = {
        "metadata": {
            "study": "gemma_clean_code",
            "phase": "Phase 3: Monosemanticity Scoring",
            "model_name": act_summary["model_name"],
            "sampled_layers": act_summary["sampled_layers"],
            "top_k_per_category": top_k,
        },
        "layers": results_by_layer,
        "top_monosemantic_neurons_by_category": final_ranked_by_category,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Saved monosemanticity scores to %s", output_path)
    return output_payload


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Monosemanticity scoring for gemma_clean_code."
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Path to corpus/activation_summary.json",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save corpus/monosemanticity_scores.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top neurons to rank per category (default: 5)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    summary_path = (
        Path(args.summary_file)
        if args.summary_file
        else project_root / "corpus" / "activation_summary.json"
    )
    output_path = (
        Path(args.output_file)
        if args.output_file
        else project_root / "corpus" / "monosemanticity_scores.json"
    )

    scores = analyze_monosemanticity(summary_path, output_path, top_k=args.top_k)

    print("\n" + "=" * 75)
    print("PHASE 3: MONOSEMANTICITY SCORING COMPLETE")
    print("=" * 75)
    print(f"Model: {scores['metadata']['model_name']}")
    print(f"Sampled Layers: {scores['metadata']['sampled_layers']}")
    print("\nTOP MONOSEMANTIC NEURONS PER AST CATEGORY (Ranked by MSR):")

    for cat, neurons in scores["top_monosemantic_neurons_by_category"].items():
        print(f"\n📂 AST Category: {cat.upper()}")
        print(f"   {'Rank':<4} {'Layer':<8} {'Neuron':<8} {'MSR':<8} {'F1':<8} {'Delta':<10} {'Target Act':<10} {'Control Act':<10}")
        print("   " + "-" * 70)
        for rank, n in enumerate(neurons, 1):
            print(
                f"   #{rank:<3} L{n['layer']:<7} #{n['neuron_idx']:<7} {n['msr']:<8.4f} "
                f"{n['f1']:<8.4f} {n['delta']:<+10.4f} {n['target_act']:<10.4f} {n['control_act']:<10.4f}"
            )

    print(f"\nResults successfully exported to: {output_path}")


if __name__ == "__main__":
    main()
