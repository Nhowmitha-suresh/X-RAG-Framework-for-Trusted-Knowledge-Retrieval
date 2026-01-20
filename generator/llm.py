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
        # Allow overriding Google model via env var (e.g., GEMINI_MODEL or GOOGLE_MODEL)
        self.google_model = os.environ.get("GEMINI_MODEL") or os.environ.get("GOOGLE_MODEL") or "text-bison"
        # Optional bearer token (e.g., from service account) to use Authorization header
        self.google_bearer = os.environ.get("GOOGLE_BEARER_TOKEN")

        if self.openai_key:
            openai.api_key = self.openai_key

    def chat_completion(self, system: str, user_prompt: str, temperature: float = 0.0) -> Optional[str]:
        # Prefer OpenAI if key present, otherwise try Google Generative API if GOOGLE_API_KEY set
        if self.openai_key:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]

            # Support both openai v1 (OpenAI client) and older interfaces.
            # Newer package exposes an `OpenAI` client with `chat.completions.create`.
            try:
                Client = getattr(openai, "OpenAI", None)
                if Client is not None:
                    client = Client()
                    resp = client.chat.completions.create(model=self.model, messages=messages, temperature=temperature)
                    # resp.choices[0].message.content or dict access
                    try:
                        return resp.choices[0].message.content.strip()
                    except Exception:
                        try:
                            return resp["choices"][0]["message"]["content"].strip()
                        except Exception:
                            pass

            except Exception:
                # fall through to older API usage
                pass

            # If the modern OpenAI client isn't available, avoid calling removed
            # legacy endpoints (openai.ChatCompletion) which raise on newer
            # openai package versions. Return None so callers can fallback.
            return None

        if self.google_key:
            # Use Google Generative API endpoint for configured model (supports Gemini/text-bison)
            model_name = self.google_model
            url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model_name}:generate"
            # If a bearer token is provided, prefer Authorization header; otherwise attach API key as query param
            headers = {}
            params = None
            if self.google_bearer:
                headers["Authorization"] = f"Bearer {self.google_bearer}"
            else:
                params = {"key": self.google_key}

            prompt = f"System: {system}\nUser: {user_prompt}"
            body = {
                "prompt": {"text": prompt},
                "temperature": temperature,
            }
            try:
                r = requests.post(url, json=body, headers=headers or None, params=params, timeout=30)
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
