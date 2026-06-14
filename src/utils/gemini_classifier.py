import time
from typing import Optional

import pandas as pd
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.utils.message_utils import create_classification_messages
from src.utils.prompts import CLASSIFICATION_SYSTEM_PROMPT

_GEMINI_MODEL_ALIASES = {
    "gemini-3-flash-preview": "gemini-3.5-flash",
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
}

DEFAULT_MAX_RETRIES = 8


def resolve_gemini_model(model: str) -> str:
    if model.startswith("gemini:"):
        _, _, rest = model.partition(":")
        return _GEMINI_MODEL_ALIASES.get(rest, rest)
    return _GEMINI_MODEL_ALIASES.get(model, model)


def _extract_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    err_str = str(exc)
    for code in (429, 500, 502, 503, 504):
        if str(code) in err_str:
            return code
    return None


def retry_wait_seconds(exc: Exception, attempt: int) -> Optional[float]:
    """
    Return seconds to wait before retrying, or None if the error is not retryable.
    attempt is 0-based.
    """
    err_str = str(exc).lower()
    status_code = _extract_status_code(exc)

    if status_code == 429 or "429" in err_str or "rate" in err_str or "quota" in err_str:
        # Rate limits: exponential backoff from 10s
        return min(120.0, 10.0 * (2 ** attempt))

    if status_code == 503 or "503" in err_str or "unavailable" in err_str:
        # Service unavailable: longer linear backoff
        return min(180.0, 30.0 * (attempt + 1))

    if status_code in (500, 502, 504) or any(
        token in err_str for token in ("500", "502", "504", "internal server", "bad gateway")
    ):
        # Server errors: medium exponential backoff
        return min(120.0, 15.0 * (2 ** attempt))

    if any(
        token in err_str
        for token in ("timeout", "timed out", "connection", "disconnect", "reset", "temporarily")
    ):
        # Network / timeout issues
        return min(90.0, 10.0 * (2 ** attempt))

    if isinstance(exc, (genai_errors.ServerError, genai_errors.APIError)):
        # Unknown API/server error: conservative backoff
        return min(90.0, 20.0 * (attempt + 1))

    return None


class GeminiClassifier:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.model = resolve_gemini_model(model)
        self.max_retries = max_retries
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=CLASSIFICATION_SYSTEM_PROMPT,
        )

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Call Gemini with a raw prompt, reusing retry logic."""
        config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
            system_instruction=system_instruction,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                if not response.text:
                    finish_reason = (
                        response.candidates[0].finish_reason
                        if response.candidates
                        else "unknown"
                    )
                    print(
                        f"Warning: Gemini empty response "
                        f"(finish_reason={finish_reason}, attempt {attempt + 1})"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(5.0 * (attempt + 1))
                        continue
                    return ""

                return response.text.strip()

            except Exception as e:
                wait_time = retry_wait_seconds(e, attempt)
                if wait_time is not None and attempt < self.max_retries - 1:
                    print(
                        f"Gemini error (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    print(f"Retrying in {wait_time:.0f}s...")
                    time.sleep(wait_time)
                    continue

                print(f"Gemini failed after {attempt + 1} attempts: {e}")
                return ""

        return ""

    def classify_sentence(
        self,
        sentence: str,
        categories_text: str,
        examples_df: Optional[pd.DataFrame] = None,
    ) -> str:
        messages = create_classification_messages(sentence, categories_text, examples_df)
        prompt = messages[-1]["content"]
        result = self.generate(prompt, system_instruction=self.config.system_instruction)
        return result if result else "Classification Error"
