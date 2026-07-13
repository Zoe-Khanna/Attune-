"""
Attune — backend
================
Holds your OpenRouter API key (never exposed to the browser), reads the
uploaded photo or text, has the AI analyze the CONCEPTS, and returns
brand-new questions verified by a second AI pass.

Run locally:
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-...        (Windows: set OPENROUTER_API_KEY=sk-or-...)
    uvicorn server:app --reload
    open http://127.0.0.1:8000
"""

import os, json, re, base64, random, time
from typing import List, Optional

import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OR_URL = "https://openrouter.ai/api/v1"

app = FastAPI(title="Attune")


# ---------------------------------------------------------------- prompts
GENERATE_PROMPT = """You are an expert teacher and assessment designer building a quiz for a student.

You will receive study material (text, or an image of a textbook page). It may contain
explanations, facts, worked examples, formulas, or existing questions.

WORK IN THIS EXACT ORDER:

STEP 1 - ANALYZE:
- What subject/topic is this? Be specific (e.g. "algebraic identities: expanding (a+b)^2",
  not just "math").
- List every distinct CONCEPT, rule, formula, or idea the material teaches.

STEP 2 - DESIGN {n} NEW QUESTIONS:
- Cover the concepts you listed; spread coverage, don't cluster on one.
- BRAND NEW: different numbers, different examples, fresh wording. NEVER copy or lightly
  reword the material's own sentences or questions.
- Difficulty ladder: start easy (direct application), build to hard (apply the concept to
  an unfamiliar situation, or combine two concepts).
- For math/formula material: invent NEW problems using the same rules, and SOLVE each one
  step by step in your reasoning so the answer is guaranteed correct.
- If reading an IMAGE: first transcribe the key formulas/facts, and state your reading of
  any ambiguous symbol, before writing questions.
- Keep each question SHORT (it appears inside a game, on one line if possible).
- 4 options each, each option short. Wrong options must be PLAUSIBLE student errors
  (sign slips, off-by-one, common misconceptions) - never silly.
- Exactly ONE correct option.

STEP 3 - OUTPUT:
After your reasoning, output the quiz inside <QUIZ> tags, exactly:
<QUIZ>
{{"topic":"specific topic label","questions":[
 {{"prompt":"...","options":["...","...","...","..."],"answer":0,
  "explain":"one short sentence teaching why the correct answer is right",
  "difficulty":"easy"}}
]}}
</QUIZ>
"answer" is the 0-3 index of the correct option. No markdown fences inside the tags."""

VERIFY_PROMPT = """You are a meticulous exam checker. Below is a quiz in JSON. For EACH question,
independently solve it yourself from scratch WITHOUT trusting the marked answer.

- If the marked answer is correct: keep the question unchanged.
- If the marked answer is WRONG: fix the "answer" index to the truly correct option and
  update "explain".
- If a question is ambiguous, has two defensible answers, or is broken: REWRITE it into a
  clean question on the same concept with exactly one correct answer.
- Keep the same number of questions and the same JSON shape.

Show brief checking work, then output the corrected quiz inside <QUIZ> tags in the same
JSON shape. No markdown fences inside the tags.

QUIZ TO CHECK:
"""


# ---------------------------------------------------------------- model discovery
_free_vision: List[str] = []
_free_text: List[str] = []
_picked = {"vision": None, "text": None}


def discover_models():
    """Free models on OpenRouter rotate, so ask which are live right now."""
    global _free_vision, _free_text
    try:
        r = requests.get(f"{OR_URL}/models", timeout=30)
        r.raise_for_status()
        vision, text = [], []
        for m in r.json().get("data", []):
            mid = m.get("id", "")
            if not mid.endswith(":free"):
                continue
            mods = (m.get("architecture") or {}).get("input_modalities") or []
            (vision if "image" in mods else text).append(mid)
        _free_vision, _free_text = vision, text
    except Exception as e:
        print("model discovery failed:", e)
        _free_vision, _free_text = [], []
    return _free_vision, _free_text


def _rank(m: str) -> int:
    s = 0
    if "gemma" in m: s -= 5
    if any(k in m for k in ("31b", "70b", "120b")): s -= 3
    if any(k in m for k in ("nano", "mini", "small")): s += 3
    return s


def pick_model(kind: str) -> str:
    if _picked[kind]:
        return _picked[kind]
    if not _free_vision and not _free_text:
        discover_models()
    pool = _free_vision if kind == "vision" else (_free_text + _free_vision)
    if not pool:
        raise HTTPException(503, "No free models available from OpenRouter right now.")
    for m in sorted(pool, key=_rank)[:6]:
        try:
            _chat([{"role": "user", "content": "ping"}], m, 0.0, max_tokens=5, tries=1)
            _picked[kind] = m
            print(f"using {kind} model: {m}")
            return m
        except Exception:
            continue
    raise HTTPException(503, f"No usable free {kind} model right now.")


# ---------------------------------------------------------------- AI call
def _chat(messages, model, temperature, max_tokens=3000, tries=3):
    if not API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY is not set on the server.")
    last = None
    for attempt in range(tries):
        try:
            r = requests.post(
                f"{OR_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=120,
            )
            if r.status_code != 200:
                raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last))


# ---------------------------------------------------------------- parsing
def _extract_quiz(text: str) -> dict:
    m = re.search(r"<QUIZ>(.*?)</QUIZ>", text, re.S)
    payload = m.group(1) if m else text
    payload = payload.replace("```json", "").replace("```", "").strip()
    a, z = payload.find("{"), payload.rfind("}")
    if a != -1 and z != -1:
        payload = payload[a:z + 1]
    return json.loads(payload)


def _sanitize(quiz: dict) -> dict:
    """Dedupe options, guarantee exactly one valid answer, shuffle positions."""
    clean = []
    for q in quiz.get("questions", []):
        if not (q.get("prompt") and isinstance(q.get("options"), list)):
            continue
        raw = [str(o).strip() for o in q["options"] if str(o).strip()]
        idx = q.get("answer", 0)
        if not isinstance(idx, int) or not (0 <= idx < len(raw)):
            continue
        correct = raw[idx]
        opts = list(dict.fromkeys(raw))          # dedupe, keep order
        if len(opts) < 3 or correct not in opts:
            continue
        opts = opts[:4]
        if correct not in opts:
            opts[-1] = correct
        random.shuffle(opts)
        clean.append({
            "prompt": str(q["prompt"]).strip(),
            "options": opts,
            "answer": opts.index(correct),
            "explain": str(q.get("explain", "")).strip()[:220],
            "difficulty": q.get("difficulty", "medium"),
        })
    return {"topic": quiz.get("topic", "Practice"), "questions": clean}


def generate(user_content, kind: str, n: int) -> dict:
    model = pick_model(kind)
    raw = _chat([{"role": "user", "content": user_content}], model, 0.7)
    draft = _sanitize(_extract_quiz(raw))
    if not draft["questions"]:
        raise HTTPException(422, "Couldn't build questions from that. Try clearer material.")

    # second pass: re-solve everything and fix wrong answer keys
    try:
        tmodel = pick_model("text")
        raw2 = _chat([{"role": "user",
                       "content": VERIFY_PROMPT + json.dumps(draft, ensure_ascii=False)}],
                     tmodel, 0.0)
        verified = _sanitize(_extract_quiz(raw2))
        if len(verified["questions"]) >= max(3, len(draft["questions"]) // 2):
            return verified
    except Exception as e:
        print("verification skipped:", e)
    return draft


# ---------------------------------------------------------------- API
class TextIn(BaseModel):
    text: str
    n: int = 8


@app.get("/api/health")
def health():
    v, t = (_free_vision, _free_text) if (_free_vision or _free_text) else discover_models()
    return {"key_set": bool(API_KEY), "free_vision": len(v), "free_text": len(t)}


@app.post("/api/quiz/text")
def quiz_from_text(body: TextIn):
    if len(body.text.strip()) < 30:
        raise HTTPException(400, "Please provide a bit more text.")
    n = max(3, min(12, body.n))
    content = GENERATE_PROMPT.format(n=n) + "\n\nSTUDY MATERIAL:\n\n" + body.text
    return generate(content, "text", n)


@app.post("/api/quiz/image")
async def quiz_from_image(file: UploadFile = File(...), n: int = Form(8)):
    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 8 MB).")
    mime = file.content_type or "image/jpeg"
    if not mime.startswith("image/"):
        raise HTTPException(400, "That file isn't an image.")
    b64 = base64.b64encode(raw).decode()
    n = max(3, min(12, n))
    content = [
        {"type": "text",
         "text": GENERATE_PROMPT.format(n=n) +
                 "\n\nSTUDY MATERIAL is the image below. Read it, then make the questions."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
    return generate(content, "vision", n)


# ---------------------------------------------------------------- static site
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
