# Attune

**Learning that fits how a child thinks.**

ADHD is treated as one condition, so ADHD tools are built one way. But children with ADHD
differ enormously in *how* they focus, switch tasks, and stay motivated. Attune tests whether
those differences form measurable cognitive profiles — and whether matching teaching to a
child's profile beats treating the condition as uniform.

---

## What it does

**Photograph any textbook page.** An AI reads it, works out the concepts it teaches, and writes
**brand-new** questions on those same concepts — never copying the page.

**Then you play.** The questions become gates in a runner game. Answer correctly and the gate
opens. Get it wrong and the run restarts.

**Every answer is checked first.** An AI that writes a question can also mark it wrong — so a
second, independent AI pass re-solves every question *blind* and corrects any faulty answer key
before the child ever sees it.

---

## Run it

**1. Get a free OpenRouter key** (no credit card): sign up at [openrouter.ai](https://openrouter.ai)
→ avatar → **Keys** → **Create Key**.

**2. Install and start the server**

```bash
pip install -r requirements.txt

# Mac / Linux
export OPENROUTER_API_KEY=sk-or-your-key-here
# Windows
set OPENROUTER_API_KEY=sk-or-your-key-here

uvicorn server:app --reload
```

**3. Open** http://127.0.0.1:8000

> Your API key stays on the server. It is never sent to the browser, and it is never committed
> to this repository.

---

## The research

A profiling engine measures four dimensions from behaviour alone — **attention window**,
**task-switching tolerance**, **motivation type**, and **sensory preference**. No labels, no
self-report, no clinical claims. The child simply plays.

Run the full evaluation:

```bash
cd research
python run_evaluation.py
```

### Results (n = 300 simulated children)

| Dimension | Recovered from behaviour alone |
|---|---|
| Motivation type | **100%** |
| Sensory preference | **97%** |
| Attention window | **86%** |
| Switching tolerance | **76%** |

K-means clustering on the recovered profiles separates most cleanly at **k = 5**
(peak silhouette score 0.289) — evidence that profiles form structured groups rather than a
random spread.

> **Honest limitation:** these results come from **synthetic data**. They show the engine is
> internally consistent — **not** that it works on real children. A pilot study with real
> participants is the next step. **Attune makes no diagnostic claim** and shows no child-facing
> ADHD label.

---

## Repository

```
server.py              FastAPI backend — holds the API key, calls the AI, verifies answers
static/index.html      The runner game (HTML5 canvas) + photo upload
research/              The profiling engine and its evaluation
  simulator.py           generates a synthetic cohort with hidden profiles
  profile_engine.py      infers a profile from behaviour
  clustering.py          tests whether distinct profile types emerge
  intervention.py        maps a profile to a matched learning session
  run_evaluation.py      runs the whole study and prints the numbers
poster/                Research poster (A3)
```

## Built with

Python · FastAPI · HTML5 Canvas · OpenRouter (vision model) · scikit-learn · NumPy

---

*Built for The Innovation Story by Zoe Khanna.*
