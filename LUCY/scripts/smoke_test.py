"""Check Bhashini translation, speech out and speech in. Run from the project folder:
    python -m scripts.smoke_test     (or: python scripts/smoke_test.py)"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import backend.bhashini as bhashini  # noqa: E402


async def main():
    q = "Do you have any allergies?"

    hi = await bhashini.translate(q, "en", "hi")
    print("English to Hindi      :", hi)

    te = await bhashini.translate(q, "en", "te")
    print("English to Telugu     :", te)

    back = await bhashini.translate(hi, "hi", "en")
    print("Hindi back to English :", back)

    audio = await bhashini.tts(hi, "hi")
    with open("out_hi.wav", "wb") as f:
        f.write(audio)
    print("Saved out_hi.wav,", len(audio), "bytes")

    heard = await bhashini.asr(audio, "hi")
    print("ASR heard it as       :", heard)


asyncio.run(main())
