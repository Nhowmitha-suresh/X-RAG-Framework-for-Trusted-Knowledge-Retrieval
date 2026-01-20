import os
import requests
import json
from dotenv import load_dotenv

# Load variables from .env in project root
load_dotenv()

key = os.environ.get("GOOGLE_API_KEY")
model = os.environ.get("GEMINI_MODEL", "text-bison")
if not key:
    print("ERROR: GOOGLE_API_KEY not set in environment")
    raise SystemExit(1)

url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate?key={key}"
body = {
    "prompt": {"text": "Say: Hello from validation test."},
    "temperature": 0.0,
    "maxOutputTokens": 64,
}
try:
    r = requests.post(url, json=body, timeout=30)
    print("HTTP", r.status_code)
    try:
        j = r.json()
        print(json.dumps(j, indent=2))
    except Exception:
        print("Non-JSON response:\n", r.text)
except Exception as e:
    print("Request failed:", e)
    raise
