import hmac
import json
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend import ai, bhashini        # importing ai also loads the .env file

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = Path(os.getenv("LUCY_DATA_DIR") or Path(__file__).resolve().parent)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK = DATA_DIR / "feedback.jsonl"
ORDERS = DATA_DIR / "orders.jsonl"

MAX_IMAGE = 25 * 1024 * 1024
MAX_AUDIO = 10 * 1024 * 1024
LANGS = {"en", "hi", "te"}
RATE_LIMIT = int(os.getenv("LUCY_RATE_LIMIT") or 240)   # requests per minute for each client, 0 turns it off
STAFF_KEY = os.getenv("LUCY_STAFF_KEY") or ""            # set it to let staff read orders and feedback

app = FastAPI(title="LUCY")
_lock = threading.Lock()
_hits = defaultdict(deque)


# ---------------------------------------------------------------- errors and limits

@app.exception_handler(bhashini.BhashiniError)
async def bhashini_error(request, exc):
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)


@app.exception_handler(ai.AIError)
async def ai_error(request, exc):
    print("AI error on", request.url.path, ":", exc)
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)


@app.exception_handler(HTTPException)
async def http_error(request, exc):
    return JSONResponse({"ok": False, "error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(Exception)
async def any_error(request, exc):
    print("Unexpected error on", request.url.path, ":", type(exc).__name__, exc)
    return JSONResponse({"ok": False, "error": "Something went wrong"}, status_code=500)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if RATE_LIMIT and request.url.path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return JSONResponse({"ok": False, "error": "Too many requests. Please wait a moment."}, status_code=429)
        q.append(now)
        if len(_hits) > 5000:                 # forget clients that have gone quiet
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 60]:
                _hits.pop(k, None)
    return await call_next(request)


# ---------------------------------------------------------------- helpers

async def read_upload(f: UploadFile, limit: int) -> bytes:
    data = await f.read(limit + 1)
    if not data:
        raise HTTPException(400, "The file is empty")
    if len(data) > limit:
        raise HTTPException(413, "The file is too large")
    return data


def image_mime(f: UploadFile) -> str:
    t = (f.content_type or "").lower()
    return t if t.startswith("image/") else "image/jpeg"


def parse_json(raw: str, want):
    try:
        v = json.loads(raw)
    except Exception:
        raise HTTPException(400, "Bad request")
    if not isinstance(v, want):
        raise HTTPException(400, "Bad request")
    return v


def need_lang(lang: str) -> str:
    if lang not in LANGS:
        raise HTTPException(400, "Unsupported language")
    return lang


def append_line(path: Path, obj: dict):
    with _lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def tail_lines(path: Path, n: int):
    if not path.exists():
        return []
    with _lock:
        lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def check_staff(request: Request):
    key = request.headers.get("x-staff-key", "")
    if not STAFF_KEY or not hmac.compare_digest(key.encode(), STAFF_KEY.encode()):
        raise HTTPException(404, "Not found")


async def translate_any(text: str, src: str, tgt: str) -> str:
    """Bhashini models here only go to or from English, so other pairs pivot through English."""
    if src == tgt:
        return text
    if src != "en" and tgt != "en":
        text = await bhashini.translate(text, src, "en")
        src = "en"
    return await bhashini.translate(text, src, tgt)


# ---------------------------------------------------------------- routes

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/asr")
async def api_asr(audio: UploadFile, lang: str = Form(...)):
    return {"ok": True, "text": await bhashini.asr(await read_upload(audio, MAX_AUDIO), need_lang(lang))}


@app.post("/api/translate")
async def api_translate(text: str = Form(...), src: str = Form(...), tgt: str = Form(...)):
    return {"ok": True, "text": await translate_any(text[:3000], need_lang(src), need_lang(tgt))}


@app.post("/api/tts")
async def api_tts(text: str = Form(...), lang: str = Form(...), gender: str = Form("female")):
    gender = gender if gender in ("female", "male") else "female"
    return Response(content=await bhashini.tts(text[:3000], need_lang(lang), gender), media_type="audio/wav")


@app.post("/api/menu")
async def api_menu(image: UploadFile):
    data = await read_upload(image, MAX_IMAGE)
    items = await run_in_threadpool(ai.extract_menu, data, image_mime(image))
    return {"ok": True, "items": items}


@app.post("/api/bill")
async def api_bill(image: UploadFile):
    data = await read_upload(image, MAX_IMAGE)
    bill = await run_in_threadpool(ai.extract_bill, data, image_mime(image))
    return {"ok": True, **bill}


@app.post("/api/interpret")
async def api_interpret(question: str = Form(...), options: str = Form(...),
                        multi: str = Form(""), text: str = Form(...)):
    opts = [o for o in parse_json(options, list)[:30] if isinstance(o, dict) and "id" in o]
    r = await run_in_threadpool(ai.interpret, question[:300], opts, bool(multi), text[:400])
    return {"ok": True, **r}


@app.post("/api/order")
async def api_order(text: str = Form(...), names: str = Form(...)):
    ns = [n for n in parse_json(names, list)[:500] if isinstance(n, str)]
    r = await run_in_threadpool(ai.match_order, text[:400], ns)
    return {"ok": True, **r}


@app.post("/api/ask")
async def api_ask(question: str = Form(...), context: str = Form("")):
    return {"ok": True, "answer": await run_in_threadpool(ai.answer_query, question, context[:20000])}


@app.post("/api/feedback")
async def api_feedback(text: str = Form(...)):
    text = text.strip()[:1000]
    if not text:
        raise HTTPException(400, "No feedback")
    r = await run_in_threadpool(ai.summarise_feedback, text)
    await run_in_threadpool(append_line, FEEDBACK,
                            {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "said": text, **r})
    return {"ok": True, **r}


@app.post("/api/orderslip")
async def api_orderslip(order: str = Form(...), spice: str = Form(""), notes: str = Form(""),
                        diet: str = Form(""), allergies: str = Form("")):
    lines = []
    for o in parse_json(order, list)[:60]:
        if not isinstance(o, dict) or not str(o.get("name", "")).strip():
            continue
        try:
            qty = max(1, min(50, int(o.get("qty", 1))))
        except Exception:
            qty = 1
        price = o.get("price")
        lines.append({"name": str(o["name"]).strip()[:100], "qty": qty,
                      "price": price if isinstance(price, (int, float)) and not isinstance(price, bool) else None})
    if not lines:
        raise HTTPException(400, "Empty order")
    slip = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "order": lines, "spice": spice[:20],
            "notes": notes[:500], "diet": diet[:20], "allergies": allergies[:200]}
    await run_in_threadpool(append_line, ORDERS, slip)
    return {"ok": True}


@app.get("/api/staff/orders")
def staff_orders(request: Request, limit: int = 100):
    check_staff(request)
    return {"ok": True, "orders": tail_lines(ORDERS, max(1, min(limit, 500)))}


@app.get("/api/staff/feedback")
def staff_feedback(request: Request, limit: int = 100):
    check_staff(request)
    return {"ok": True, "feedback": tail_lines(FEEDBACK, max(1, min(limit, 500)))}


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
