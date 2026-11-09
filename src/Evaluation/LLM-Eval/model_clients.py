import os
import time

from openai import OpenAI

from config import MODELS, MAX_OUTPUT_TOKENS, TEMPERATURE


def get_client(model_key: str) -> OpenAI:
    cfg = MODELS[model_key]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Missing API key for '{model_key}': set env var {cfg['api_key_env']}")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"])


def call_model(model_key: str, prompt: str, retries: int = 3) -> dict:
    cfg = MODELS[model_key]
    client = get_client(model_key)

    extra = {cfg.get("max_tokens_param", "max_tokens"): MAX_OUTPUT_TOKENS}
    temperature = cfg.get("temperature", TEMPERATURE)
    if temperature is not None:
        extra["temperature"] = temperature
    if cfg.get("reasoning_effort") is not None:
        extra["reasoning_effort"] = cfg["reasoning_effort"]

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=cfg["model_name"],
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            usage = resp.usage
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = (getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0
            choice = resp.choices[0]
            try:
                raw = resp.model_dump()
            except Exception:
                raw = {"repr": repr(resp)}
            return {
                "text": choice.message.content or "",
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "finish_reason": choice.finish_reason or "",
                "raw": raw,
            }
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{model_key} failed after {retries} retries: {last_err}")
