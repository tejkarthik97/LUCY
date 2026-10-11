"""Raw Gemini connectivity check on a menu photo (no cleaning, no cache). Run from the project folder:
    python scripts/quick_menu.py path/to/menu.jpg"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
path = sys.argv[1]
keys = [k.strip() for k in ((os.getenv("GEMINI_API_KEYS") or "") + "," + (os.getenv("GEMINI_API_KEY") or "")).split(",") if k.strip()]
if not keys:
    sys.exit("No GEMINI_API_KEY or GEMINI_API_KEYS in .env")
model = (os.getenv("GEMINI_MODEL") or "gemini-3.8-flash").split(",")[0].strip()
mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
print("Sending image to Gemini (%s)..." % model, flush=True)
t = time.time()
client = genai.Client(api_key=keys[0], http_options=types.HttpOptions(timeout=120000))
prompt = """List every dish on this restaurant menu. Ignore banners and offers.
Return JSON: {"items":[{"name":"","price":0,"section":"","diet":"veg|non_veg|unknown"}]}.
Only the JSON."""
try:
    with open(path, "rb") as f:
        data = f.read()
    resp = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
except Exception as e:
    print("ERROR after", round(time.time() - t), "seconds:", type(e).__name__, str(e)[:300])
    sys.exit(1)
print("Got reply in", round(time.time() - t), "seconds", flush=True)
try:
    items = json.loads(resp.text)["items"]
except Exception:
    print("Could not parse JSON. First 500 chars:\n", (resp.text or "")[:500])
    sys.exit(1)
print(len(items), "items found")
for it in items:
    print(it.get("name"), "| Rs", it.get("price"), "|", it.get("section"), "|", it.get("diet"))
