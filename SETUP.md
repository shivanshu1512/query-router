# SETUP — Day 1 (Windows / PowerShell)

Ye file sirf setup ke liye hai. Ek-ek step karo, jaldi mat karo.

---

## Step 1 — `uv` install karo

PowerShell kholo (admin ki zaroorat nahi):

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Ab **PowerShell band karke naya kholo.** PATH tabhi refresh hoga.

Confirm:

```powershell
uv --version
```

---

## Step 2 — Project folder mein jao

```powershell
cd path\to\query-router
```

---

## Step 3 — Environment banao

```powershell
uv sync
```

Ye ek saath teen kaam karta hai:
- `.venv\` folder banata hai (project ka apna Python)
- `pyproject.toml` padh kar saari libraries install karta hai
- `uv.lock` banata hai — exact versions ka record, taaki kal koi aur bhi
  bilkul yehi environment bana sake

`.venv` corrupt ho jaaye toh darne ki zaroorat nahi:
```powershell
Remove-Item -Recurse -Force .venv
uv sync
```
Sab wapas aa jaayega.

---

## Step 4 — Apni API key daalo

`.env.example` ko copy karke `.env` banao:

```powershell
Copy-Item .env.example .env
```

Ab `.env` kholo aur apni asli Groq key paste karo:

```
GROQ_API_KEY=gsk_...
```

`.env` already `.gitignore` mein hai — GitHub par kabhi nahi jaayegi.

---

## Step 5 — Config check karo

```powershell
uv run python -m src.config
```

Provider aur teeno model names print hone chahiye. Koi API call nahi hoti
is step mein — ye sirf ye check karta hai ki YAML sahi padh raha hai.

---

## Step 6 — Smoke test (asli test)

```powershell
uv run python scripts\smoke_test.py
```

**Kya dekhna hai:** dono models ka reply, aur non-zero `in tokens` /
`out tokens`.

Wo token numbers hi is poore project ki neenv hain. Agar wo aa rahe hain,
toh cost accounting possible hai aur hum aage badh sakte hain.

---

## Step 7 — Git shuru karo

```powershell
git init
git add .
git commit -m "Day 1: project skeleton, config loader, model client with cost tracking"
```

`git status` chala kar confirm karo ki `.env` list mein **nahi** hai.

---

## Abhi kya nahi banaya, aur kyun

Ye folders jaan-boojh kar khaali hain:

- `src/routers/` — Week 2 ka kaam
- `data/` — Day 3-4 mein 200 queries yahan aayengi
- `tests/` — jaise-jaise code likhenge, saath mein tests
- `notebooks/` — Pareto curve Week 2 ke end mein

Pehle se sab kuch bana dene se sirf confusion hota hai. Har cheez tab
banegi jab uski zaroorat hogi.
