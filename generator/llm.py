import os
import time
from typing import List, Dict

import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


def call_llm(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", max_retries: int = 3) -> str:
    """Call the OpenAI chat completion API with basic retry logic.

    `messages` should be a list of dicts with `role` and `content`.
    """
    if not openai.api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required to call the LLM")

    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512
            )
            return resp.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            if attempt == max_retries:
                raise
            time.sleep(1 * attempt)
