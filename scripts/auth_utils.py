"""Hugging Face Authentication Utility for Local & Google Colab Environments.

This module provides a unified authentication helper to seamlessly authenticate
with Hugging Face across local development (via .env files) and cloud-hosted
notebooks such as Google Colab (via Colab Secrets).
"""

import os
import sys
import logging
from typing import Optional

logger = logging.getLogger("gemma_clean_code.auth")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

__all__ = ["setup_hf_auth", "get_hf_token"]


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face access token from Colab Secrets, local .env, or OS env.

    Search precedence:
    1. Google Colab Secrets (``google.colab.userdata.get('HF_TOKEN')``)
    2. Local environment variables / .env (via ``dotenv.load_dotenv``)

    Returns:
        Optional[str]: The token string if found and non-placeholder, else None.
    """
    token: Optional[str] = None

    # 1. Attempt Colab Secrets
    try:
        from google.colab import userdata  # type: ignore

        try:
            colab_secret = userdata.get("HF_TOKEN")
            if colab_secret and str(colab_secret).strip():
                logger.info("Retrieved HF_TOKEN from Google Colab Secrets.")
                token = str(colab_secret).strip()
        except Exception as e:
            logger.debug("Colab userdata retrieval failed or secret not present: %s", e)
    except (ImportError, ModuleNotFoundError):
        logger.debug("Google Colab environment not detected.")

    # 2. Fall back to local environment variables via python-dotenv
    if not token:
        try:
            from dotenv import load_dotenv, find_dotenv

            # Searches current working directory and parent directories for .env
            env_file = find_dotenv(usecwd=True)
            if env_file:
                load_dotenv(dotenv_path=env_file, override=False)
                logger.debug("Loaded local environment from %s", env_file)
            else:
                load_dotenv(override=False)
        except ImportError:
            logger.debug("python-dotenv not installed; attempting native .env parsing fallback.")
            candidates = [
                os.path.join(os.getcwd(), ".env"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#") and "=" in line:
                                    k, v = line.split("=", 1)
                                    k, v = k.strip(), v.strip().strip("'\"")
                                    if k not in os.environ:
                                        os.environ[k] = v
                    except Exception as parse_err:
                        logger.debug("Error reading %s: %s", cand, parse_err)

        env_val = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env_val and str(env_val).strip():
            token = str(env_val).strip()
            logger.info("Retrieved HF_TOKEN from environment variables.")

    # Check for unreplaced placeholder tokens
    if token and token.strip() in {
        "your_actual_token_here",
        "your_hf_token_here",
        "your_token_here",
        "hf_xxx",
    }:
        logger.warning(
            "Found placeholder HF_TOKEN ('%s'). Please replace it with your actual Hugging Face token.",
            token,
        )
        return None

    return token


def setup_hf_auth(token: Optional[str] = None) -> str:
    """Configure Hugging Face authentication across environment variables and huggingface_hub.

    This function:
    1. Resolves HF_TOKEN from Colab Secrets or local .env if not explicitly passed.
    2. Exports HF_TOKEN and HUGGING_FACE_HUB_TOKEN to os.environ.
    3. Authenticates via ``huggingface_hub.login``.

    Args:
        token: Optional explicit Hugging Face access token. If omitted,
            retrieval from Colab Secrets or .env will be attempted.

    Returns:
        str: The authenticated token string.

    Raises:
        ValueError: If no valid token is found or only placeholder values exist.
        RuntimeError: If authentication with huggingface_hub fails.
    """
    resolved_token = token or get_hf_token()

    if not resolved_token:
        error_msg = (
            "HF_TOKEN could not be found.\n"
            "- In Google Colab: Add 'HF_TOKEN' under Secrets (key icon in the left sidebar).\n"
            "- Locally: Add 'HF_TOKEN=your_token' to a local '.env' file in the project root "
            "or run 'export HF_TOKEN=\"...\"'."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Set both standard environment variables
    os.environ["HF_TOKEN"] = resolved_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = resolved_token

    # Perform Hugging Face Hub login
    try:
        import huggingface_hub

        huggingface_hub.login(token=resolved_token, add_to_git_credential=False)
        logger.info("Successfully authenticated with Hugging Face Hub.")
    except ImportError:
        logger.warning(
            "huggingface_hub is not installed; HF_TOKEN set in os.environ only."
        )
    except Exception as e:
        logger.error("huggingface_hub.login failed: %s", e)
        raise RuntimeError(f"Hugging Face Hub authentication failed: {e}") from e

    return resolved_token


if __name__ == "__main__":
    try:
        setup_hf_auth()
        print("HF authentication test successful.")
    except ValueError as val_err:
        print(f"Validation: {val_err}")
    except Exception as err:
        print(f"Auth execution error: {err}")
