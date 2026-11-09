from pathlib import Path
import os

BASE_DIR = Path(__file__).parent

API_KEYS = {
    "OPENAI_API_KEY": "",
    "LOCAL_API_KEY": "",
}
_env_file = BASE_DIR / "api_keys.env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            API_KEYS[_k.strip()] = _v.strip().strip("\"'")
for _k, _v in API_KEYS.items():
    if _v and not os.environ.get(_k):
        os.environ[_k] = _v

DATA_DIR = BASE_DIR / "data/DeepPrep-Capped"
RESULTS_DIR = BASE_DIR / "results"

STRUCTURAL_REVEAL_ROWS = 3
SMALL_INPUT_ROWS = 5
SMALL_INPUT_REVEAL_ROWS = 2

MODELS = {
    "gpt-5.6-luna-low": {
        "model_name": "gpt-5.6-luna",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_tokens_param": "max_completion_tokens",
        "temperature": None,
        "reasoning_effort": "low",
    },
    "gpt-oss-120b-local": {
        "model_name": "gpt-oss-120b",
        "base_url": os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
        "api_key_env": "LOCAL_API_KEY",
    },
}

MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", 24000))
TEMPERATURE = 0
