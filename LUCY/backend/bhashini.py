import asyncio
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import wave
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MT_MAX_WORDS = 45    # Bhashini limit is 50
TTS_MAX_WORDS = 25   # Bhashini limit is 30


class BhashiniError(Exception):
    """Raised when a Bhashini call fails, so the app can ask the user to retry."""


def _cfg(task: str, key: str):
    url = os.getenv(f"{task}_{key}_URL")
    token = os.getenv(f"{task}_{key}_TOKEN")
    if not url or not token:
        raise BhashiniError(f"Missing {task}_{key}_URL or _TOKEN in .env")
    return url.strip(), token.strip()


async def _call(url: str, token: str, json=None, files=None) -> dict:
    headers = {"access-token": token}
    body = None
    for attempt in (1, 2):                    # one quick retry for a flaky network or a busy server
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=json, files=files)
                r.raise_for_status()
                body = r.json()
            break
        except Exception as e:
            if attempt == 2:
                raise BhashiniError(f"Request failed: {type(e).__name__}")
            await asyncio.sleep(0.6)
    if body.get("status") != "success":
        raise BhashiniError(str(body.get("message") or body.get("error")))
    return body["data"]


def _hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _chunks(text: str, limit: int):
    """Split text into pieces of at most `limit` words, preferring sentence ends."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out, cur = [], []
    for s in sentences:
        words = s.split()
        if len(cur) + len(words) <= limit:
            cur += words
            continue
        if cur:
            out.append(" ".join(cur))
            cur = []
        while len(words) > limit:
            out.append(" ".join(words[:limit]))
            words = words[limit:]
        cur = words
    if cur:
        out.append(" ".join(cur))
    return out


# ---------------- ASR (speech to text) ----------------

def _to_wav(audio: bytes) -> bytes:
    """Bhashini ASR needs WAV. If we got something else, convert with ffmpeg."""
    if audio[:4] == b"RIFF":
        return audio
    if not shutil.which("ffmpeg"):
        raise BhashiniError("Audio must be WAV (or install ffmpeg to convert)")
    try:
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "in", Path(d) / "out.wav"
            src.write_bytes(audio)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
                check=True, capture_output=True,
            )
            return dst.read_bytes()
    except Exception:
        raise BhashiniError("Could not convert audio to WAV")


async def asr(audio: bytes, lang: str) -> str:
    """Speech to text. lang is 'en', 'hi' or 'te'."""
    wav = _to_wav(audio)
    if len(wav) > 5 * 1024 * 1024:
        raise BhashiniError("Recording too long (limit is about 20 seconds)")
    url, token = _cfg("ASR", lang)
    data = await _call(url, token, files={"audio_file": ("audio.wav", wav, "audio/wav")})
    return (data.get("recognized_text") or "").strip()


# ---------------- Translation ----------------

async def translate(text: str, src: str, tgt: str) -> str:
    text = text.strip()
    if src == tgt or not text:
        return text
    cp = CACHE_DIR / f"mt_{_hash(src, tgt, text)}.txt"
    if cp.exists():
        return cp.read_text(encoding="utf-8")
    url, token = _cfg("TRANSLATE", f"{src}_{tgt}")
    sem = asyncio.Semaphore(4)                # long text: translate the pieces at the same time

    async def one(chunk):
        async with sem:
            data = await _call(url, token, json={"input_text": chunk})
            return data["output_text"].strip()

    outs = await asyncio.gather(*[one(c) for c in _chunks(text, MT_MAX_WORDS)])
    result = " ".join(outs)
    cp.write_text(result, encoding="utf-8")
    return result


# ---------------- TTS (text to speech) ----------------

def _clean_for_tts(text: str) -> str:
    """Bhashini TTS rejects special characters, so keep only letters, digits and . , ?"""
    text = text.replace("₹", " rupees ").replace("&", " and ").replace("।", ".").replace("'", "").replace("’", "")
    keep = []
    for ch in text:
        if unicodedata.category(ch)[0] in "LMN" or ch in " .,?":
            keep.append(ch)
        else:
            keep.append(" ")
    return re.sub(r"\s+", " ", "".join(keep)).strip()


def _merge_wavs(parts):
    if len(parts) == 1:
        return parts[0]
    with wave.open(io.BytesIO(parts[0]), "rb") as w0:
        params = w0.getparams()
    out = io.BytesIO()
    with wave.open(out, "wb") as wo:
        wo.setparams(params)
        for p in parts:
            with wave.open(io.BytesIO(p), "rb") as w:
                wo.writeframes(w.readframes(w.getnframes()))
    return out.getvalue()


async def _tts_piece(url: str, token: str, chunk: str, gender: str) -> bytes:
    data = await _call(url, token, json={"text": chunk, "gender": gender})
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(data["s3_url"])
            r.raise_for_status()
            return r.content
    except Exception:
        raise BhashiniError("Could not download the generated audio")


async def tts(text: str, lang: str, gender: str = "female") -> bytes:
    """Text to speech. Returns WAV bytes. Long text is split and joined."""
    cleaned = _clean_for_tts(text)
    if not cleaned:
        raise BhashiniError("Nothing to say")
    cp = CACHE_DIR / f"tts_{_hash(lang, gender, cleaned)}.wav"
    if cp.exists():
        return cp.read_bytes()
    url, token = _cfg("TTS", lang)
    sem = asyncio.Semaphore(4)                # make all the short pieces at the same time, then join in order

    async def one(chunk):
        async with sem:
            return await _tts_piece(url, token, chunk, gender)

    parts = await asyncio.gather(*[one(c) for c in _chunks(cleaned, TTS_MAX_WORDS)])
    audio = _merge_wavs(list(parts))
    cp.write_bytes(audio)
    return audio