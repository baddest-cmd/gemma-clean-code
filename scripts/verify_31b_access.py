"""Verify Hugging Face Access and Hardware Allocation for google/gemma-4-31B-it.

Cross-Scale Extension (31B vs. 12B Baseline) of the gemma_clean_code study.

This script:
1. Enforces standard deterministic seeding (random, numpy, torch).
2. Verifies Hugging Face authentication via `scripts.auth_utils.setup_hf_auth()`.
3. Inspects available hardware/VRAM and instantiates `google/gemma-4-31B-it` via
   TransformerLens `HookedTransformer` using `torch.bfloat16` (or 8-bit
   quantization / multi-GPU device mapping when VRAM is constrained).
4. Prints architectural parameters: total layers (`n_layers`), hidden dimension
   (`d_model`), and MLP intermediate dimension (`d_mlp`), along with computed
   40%, 50%, and 60% relative middle-depth layer indices.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformer_lens import HookedTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.auth_utils import setup_hf_auth

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logger = logging.getLogger("gemma_clean_code.verify_31b")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

MODEL_ID_31B = "google/gemma-4-31B-it"
VRAM_BF16_THRESHOLD_GB = 68.0


def inspect_hardware() -> Dict[str, Any]:
    """Query active accelerator hardware and available VRAM."""
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if cuda_available:
        device_count = torch.cuda.device_count()
        total_vram_gb = sum(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(device_count)
        )
        primary_device_name = torch.cuda.get_device_name(0)
        bf16_supported = torch.cuda.is_bf16_supported()
        device_type = "cuda"
    elif mps_available:
        device_count = 1
        total_vram_gb = 0.0
        primary_device_name = "Apple Silicon MPS"
        bf16_supported = True
        device_type = "mps"
    else:
        device_count = 0
        total_vram_gb = 0.0
        primary_device_name = "CPU"
        bf16_supported = False
        device_type = "cpu"

    return {
        "device_type": device_type,
        "device_count": device_count,
        "device_name": primary_device_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "bf16_supported": bf16_supported,
    }


def compute_relative_depth_layers(n_layers: int) -> Dict[str, int]:
    """Compute layer indices corresponding to 40%, 50%, and 60% model depth."""
    return {
        "depth_40_pct": int(round(0.40 * n_layers)),
        "depth_50_pct": int(round(0.50 * n_layers)),
        "depth_60_pct": int(round(0.60 * n_layers)),
    }


def instantiate_31b_model(
    model_id: str = MODEL_ID_31B,
    force_8bit: bool = False,
    config_only: bool = False,
) -> Dict[str, Any]:
    """Verify HF authentication and load google/gemma-4-31B-it architecture.

    Args:
        model_id: Hugging Face model repository ID.
        force_8bit: Force 8-bit bitsandbytes quantization regardless of VRAM.
        config_only: Authenticate and verify architecture config without loading full weights.

    Returns:
        Dict[str, Any]: Verified architectural and hardware telemetry.
    """
    logger.info("Step 1/3: Verifying Hugging Face authentication via HF_TOKEN...")
    setup_hf_auth()

    logger.info("Step 2/3: Inspecting hardware allocation...")
    hw_info = inspect_hardware()
    logger.info(
        "Detected accelerator: %s (count=%d, total_vram=%.2f GB, bf16=%s)",
        hw_info["device_name"],
        hw_info["device_count"],
        hw_info["total_vram_gb"],
        hw_info["bf16_supported"],
    )

    if config_only:
        logger.info("Step 3/3: Fetching AutoConfig for %s...", model_id)
        hf_cfg = AutoConfig.from_pretrained(model_id)
        text_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        n_layers = int(getattr(text_cfg, "num_hidden_layers"))
        d_model = int(getattr(text_cfg, "hidden_size"))
        d_mlp = int(getattr(text_cfg, "intermediate_size"))
        n_heads = int(getattr(text_cfg, "num_attention_heads", 0))
        n_kv_heads = int(getattr(text_cfg, "num_key_value_heads", 0))
        quant_mode = "config_only"
    else:
        use_8bit = force_8bit or (
            hw_info["device_type"] == "cuda"
            and hw_info["total_vram_gb"] < VRAM_BF16_THRESHOLD_GB
        )
        dtype = torch.bfloat16 if hw_info["bf16_supported"] else torch.float16

        if use_8bit:
            logger.info(
                "Step 3/3: VRAM (%.2f GB) < %.1f GB threshold (or --load-in-8bit set). "
                "Loading %s with 8-bit quantization and device_map='auto'...",
                hw_info["total_vram_gb"],
                VRAM_BF16_THRESHOLD_GB,
                model_id,
            )
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=dtype,
            )
            model = HookedTransformer.from_pretrained(
                model_id,
                hf_model=hf_model,
                device=hw_info["device_type"],
                dtype=dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            quant_mode = "8bit_device_map_auto"
        else:
            logger.info(
                "Step 3/3: Instantiating %s in %s via HookedTransformer...",
                model_id,
                dtype,
            )
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
            )
            model = HookedTransformer.from_pretrained(
                model_id,
                hf_model=hf_model,
                device=hw_info["device_type"],
                dtype=dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            quant_mode = "bfloat16_device_map_auto"

        n_layers = int(model.cfg.n_layers)
        d_model = int(model.cfg.d_model)
        d_mlp = int(model.cfg.d_mlp)
        n_heads = int(model.cfg.n_heads)
        n_kv_heads = int(getattr(model.cfg, "n_key_value_heads", model.cfg.n_heads))

    depth_map = compute_relative_depth_layers(n_layers)

    return {
        "model_id": model_id,
        "quantization_mode": quant_mode,
        "n_layers": n_layers,
        "d_model": d_model,
        "d_mlp": d_mlp,
        "n_heads": n_heads,
        "n_key_value_heads": n_kv_heads,
        "relative_depth_layers": depth_map,
        "hardware": hw_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify HF access and hardware allocation for google/gemma-4-31B-it."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_ID_31B,
        help=f"Hugging Face model ID (default: {MODEL_ID_31B})",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Force 8-bit quantization for constrained VRAM environments",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Authenticate and fetch model architecture parameters from AutoConfig without loading full weights",
    )
    args = parser.parse_args()

    telemetry = instantiate_31b_model(
        model_id=args.model,
        force_8bit=args.load_in_8bit,
        config_only=args.config_only,
    )

    depths = telemetry["relative_depth_layers"]
    hw = telemetry["hardware"]

    print("\n" + "=" * 72)
    print("GEMMA 4 31B ACCESS & ARCHITECTURE VERIFICATION REPORT")
    print("=" * 72)
    print(f"Model ID                  : {telemetry['model_id']}")
    print(f"Instantiation Mode        : {telemetry['quantization_mode']}")
    print(f"Accelerator Device        : {hw['device_name']} ({hw['device_type']}, count={hw['device_count']})")
    print(f"Total Available VRAM      : {hw['total_vram_gb']:.2f} GB")
    print("-" * 72)
    print(f"Total Layers (n_layers)   : {telemetry['n_layers']}")
    print(f"Hidden Dim (d_model)      : {telemetry['d_model']}")
    print(f"MLP Intermediate (d_mlp)  : {telemetry['d_mlp']}")
    print(f"Attention Heads (n_heads) : {telemetry['n_heads']} (KV Heads: {telemetry['n_key_value_heads']})")
    print("-" * 72)
    print("Mapped Relative Middle Depths (31B vs. 12B Baseline):")
    print(f"  40% Depth               : Layer {depths['depth_40_pct']:2d}  (12B Baseline: Layer 16)")
    print(f"  50% Depth               : Layer {depths['depth_50_pct']:2d}  (12B Baseline: Layer 24)")
    print(f"  60% Depth               : Layer {depths['depth_60_pct']:2d}  (12B Baseline: Layer 32)")
    print("=" * 72)


if __name__ == "__main__":
    main()

