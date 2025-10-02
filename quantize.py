"""
Download model from HF -> Covert to GGUF (f16) -> Quantize to Q4_K_M > Save.
This script is implemented and tested on "Ubuntu 24.04.2 LTS".
"""
import logging
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger(__name__)

# === EDIT THESE ===
ROOT_DIR = Path(__file__).resolve().parent
print(f"ROOT_DIR: {ROOT_DIR}")
LLAMA_CPP_DIR = ROOT_DIR / "llama.cpp"
ARTIFICATS_DIR = LLAMA_CPP_DIR / "artifacts"
quant_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

# MODEL_NAME = "Llama-3.2-1B-Instruct"
# MODEL_ID = "unsloth/" + MODEL_NAME     # Hugging Face model repo id

MODEL_NAME = "Qwen2.5-1.5B-Instruct"
MODEL_ID = "Qwen/" + MODEL_NAME
GGUF_F16_MODEL_PATH = ARTIFICATS_DIR / f"{MODEL_NAME}_Q4_K_M.gguf"

ANDROID_ASSETS_DIR = ROOT_DIR / "llama.cpp/examples/llama.android/app/src/main/assets/models"
ANDROID_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
GGUF_Q4_MODEL_PATH = ANDROID_ASSETS_DIR / f"{MODEL_NAME}_Q4_K_M.gguf"
# ==================

if not LLAMA_CPP_DIR.is_dir():
    raise FileNotFoundError(f"llama.cpp not found at {LLAMA_CPP_DIR}")

# lets make sure llama-quantize exists
if not quant_bin.exists():
    raise FileNotFoundError(f"llama-quantize binary not found. Build llama.cpp (`cd {LLAMA_CPP_DIR} && cmake -B build && cmake --build build --config Release`).")

ARTIFICATS_DIR.mkdir(parents=True, exist_ok=True)

def run(cmd, cwd=None):
    """Helper for running commands"""
    logger.info("[RUN] %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=cwd)

def main():
    """Main function"""
    # 1) Download HF snapshot
    logger.info("Downloading model from Hugging Face: %s", MODEL_ID)
    hf_dir = ARTIFICATS_DIR / "hf_model"
    hf_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(hf_dir), local_dir_use_symlinks=False)
    logger.info("Downloaded to: %s", hf_dir)

    # 2) Convert HF -> GGUF (f16) using convert-hf-to-gguf.py from llama.cpp
    hf_to_gguf_convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not hf_to_gguf_convert_script.exists():
        raise FileNotFoundError(f"convert script not found at {hf_to_gguf_convert_script}. Clone llama.cpp and ensure script is present.")

    logger.info("Converting to GGUF (f16) -> %s", GGUF_F16_MODEL_PATH)
    cmd_convert = [
        sys.executable,
        str(hf_to_gguf_convert_script),
        str(hf_dir),
        "--outfile", str(GGUF_F16_MODEL_PATH),
        "--outtype", "f16"
    ]
    run(cmd_convert)

    logger.info("Quantizing to Q4_K_M -> %s", GGUF_Q4_MODEL_PATH)
    cmd_quant = [
        str(quant_bin),
        str(GGUF_F16_MODEL_PATH),
        str(GGUF_Q4_MODEL_PATH),
        "Q4_K_M"
    ]
    run(cmd_quant)

    logger.info("Done.\nOutputs:")
    logger.info(" - f16 GGUF: %s", GGUF_F16_MODEL_PATH)
    logger.info(" - Q4_K_M GGUF: %s", GGUF_Q4_MODEL_PATH)


if __name__ == "__main__":
    main()
    print('GGUF_Q4_MODEL_PATH', GGUF_Q4_MODEL_PATH)