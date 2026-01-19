import os
from typing import Optional

from dotenv import load_dotenv
import openai
import requests


# Load .env into environment (safe local development; do NOT commit .env)
load_dotenv()


class LLM:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        # Support both OpenAI and Google API keys via env vars or explicit arg
        self.openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.google_key = os.environ.get("GOOGLE_API_KEY")
        self.model = model

        if self.openai_key:
            openai.api_key = self.openai_key

    def chat_completion(self, system: str, user_prompt: str, temperature: float = 0.0) -> Optional[str]:
        # Prefer OpenAI if key present, otherwise try Google Generative API if GOOGLE_API_KEY set
        if self.openai_key:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]

            resp = openai.ChatCompletion.create(model=self.model, messages=messages, temperature=temperature)
            return resp["choices"][0]["message"]["content"].strip()

        if self.google_key:
            # Use Google Generative Language REST endpoint for text generation (v1beta2)
            # Default model name for text: text-bison
            model_name = "text-bison"
            url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model_name}:generate?key={self.google_key}"
            prompt = f"System: {system}\nUser: {user_prompt}"
            body = {
                "prompt": {"text": prompt},
                "temperature": temperature,
            }
            try:
                r = requests.post(url, json=body, timeout=30)
                r.raise_for_status()
                j = r.json()
                # response candidates under 'candidates'
                if "candidates" in j and len(j["candidates"]) > 0:
                    return j["candidates"][0].get("content", "").strip()
                # fallback to text in 'output' or other fields
                if "output" in j and isinstance(j["output"], list) and len(j["output"]) > 0:
                    return j["output"][0].get("content", "").strip()
            except Exception:
                return None

        return None
