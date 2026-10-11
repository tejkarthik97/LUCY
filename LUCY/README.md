# LUCY

A voice-first food ordering assistant for restaurant diners, in English, Hindi and Telugu. It is built for blind, deaf and mute diners as well as everyone else.

The web page (`web/index.html`) runs the conversation. Gemini only reads photos (menu, bill) and understands answers. Bhashini does speech recognition, speech output and translation.

## Run

```
pip install fastapi uvicorn python-multipart python-dotenv google-genai
uvicorn backend.main:app --reload
```

Open http://localhost:8000. The microphone and camera need `localhost` or HTTPS.

## Settings (`.env` in the project folder)

| Name | What it does |
|---|---|
| `GEMINI_API_KEY` or `GEMINI_API_KEYS` | One key, or several separated by commas. Quota errors move on to the next key. |
| `GEMINI_MODEL` | One model, or several separated by commas, tried in order. |
| `LUCY_DATA_DIR` | Where `orders.jsonl` and `feedback.jsonl` are written (default `backend/`). |
| `LUCY_RATE_LIMIT` | Requests per minute for each client (default 240, `0` turns it off). |
| `LUCY_STAFF_KEY` | If set, staff can read `GET /api/staff/orders` and `GET /api/staff/feedback` with the header `X-Staff-Key`. |

Bhashini credentials: see `backend/bhashini.py`.

## How the flow works

Language, access needs, menu photo, taste questions, ordering, meal, bill.

- Blind: everything is spoken, and "Lucy, open message bar" asks a question by voice.
- Deaf: captions only. At the end Lucy speaks the order to the waiter in English.
- Mute: answers by tap or typing. Lucy speaks the order to the waiter in English.
- Allergies: dishes the menu reader is unsure about are left out for diners with allergies. Always confirm with the waiter.

## Checks

- `python scripts/smoke_test.py` checks Bhashini.
- `python scripts/quick_menu.py menu.jpg` checks Gemini.
- `python scripts/test_menu.py menu.jpg` runs the real menu reader.
