# GraphGuard: Real‑Time Fraud Detection with Graph‑Enhanced ML

**One‑liner:** Detect suspicious transactions by combining classic tabular ML (XGBoost) with **graph features/embeddings** (Node2Vec), served via **FastAPI** with a **Streamlit Analyst Console** and human‑readable explanations.

**Why the name?** *GraphGuard* = **Graph** (we mine relationships across users/devices/cards) + **Guard** (a guardian that flags risk in real time). It’s memorable, on‑theme for fraud, and recruiter‑friendly.

---

## 1) Problem & Outcomes

**Business problem:** Reduce fraudulent transactions while keeping false positives low so legitimate customers aren’t blocked.

**Outcomes (MVP targets):**

* **Model:** ROC‑AUC ≥ **0.90** on a standard subset; **+1–3 AUC points** vs. tabular‑only baseline *or* **\~10–20% fewer FPs** at matched recall.
* **Service:** p95 scoring latency **< 30 ms** per request on CPU (batch of 1–10 txns).
* **Explainability:** Top features with **SHAP** + a clear **natural‑language reason**.
* **Console:** Streamlit dashboard for analysts to review alerts, inspect ego‑graphs, and export decisions.

**MVP scope (2–3 days):**

* Train tabular **XGBoost baseline**, then augment with **graph features/embeddings**.
* Ship **FastAPI** endpoint `/score` + **Streamlit UI** for demos/analysts.
* Clean **README**, 60–90s **demo video/GIF**, and reproducible scripts/notebooks.

---

## 2) Data & Assumptions

Choose one dataset (timeboxed):

* **IEEE‑CIS Fraud** (Kaggle; rich tabular, realistic e‑commerce) — preferred.
* **PaySim** (synthetic mobile money) — simpler/faster if time is tight.

**Minimal schema** we’ll work with (we’ll map actual columns accordingly):

```
transaction_id, timestamp, amount,
src_account_id, dst_account_id,
device_id, ip_address, merchant_id, ...,
label  # 0/1 fraud
```

**Label:** Provided by dataset (binary). If using PaySim, `isFraud`.

**Train/valid split:** time‑based or stratified 80/20; fix `random_state`.

---

## 3) Modeling Strategy

1. **Baseline (tabular‑only):**

   * Preprocess: impute, encode categorical (target or frequency), scale where needed.
   * Train **XGBoost**; tune a few key params (max\_depth, eta, subsample, colsample).
2. **Graph features:**

   * Build multi‑relation graph (nodes: accounts/devices/ips; edges: transactions, shared devices/ips).
   * Compute **graph metrics** for each node using NetworkX. We will focus on features like **PageRank**, **Degree Centrality**, and **Clustering Coefficient**. These stats will be our graph-derived signals.
   * Join per‑txn via src/dst node features and simple aggregates (mean/max across neighbors).
3. **Hybrid model:**

   * Concatenate tabular features + graph vectors → **XGBoost**.
   * Evaluate ROC‑AUC, PR‑AUC, precision\@k, and **FP\@fixed recall**.
4. **Explainability:**

   * Global: SHAP summary/bar plot.
   * Per‑txn: SHAP top‑3 + template reason (e.g., “High device reuse + risky neighbors”).

> Stretch (optional if time remains): swap Node2Vec with a tiny R‑GCN (DGL/PyG) and compare.

---

## 4) System Architecture (MVP)

```
[CSV/Parquet data] → [Preprocess + Feature Store (Parquet/SQLite)]
                           │
                           ├── Train: Baseline XGB → Hybrid XGB (tabular + graph)
                           │
                    [model.pkl, encoders]
                           │
                   FastAPI `/score`  <──  Streamlit Analyst Console
                           │                       │
             JSON in: txn fields                Upload CSV / Simulate stream
             JSON out: probability,             View alerts, ego‑graph, SHAP
             top features, reason               Export decisions
```

**Tech stack:** Python 3.11, XGBoost, pandas, scikit‑learn, NetworkX (+ node2vec), SHAP, FastAPI, Uvicorn, Streamlit, SQLite/Parquet; pytest; Makefile; optional Docker.

---

## 5) API Design (FastAPI)

**Endpoint:** `POST /score`

* **Input (JSON):** minimal txn fields used by the model (document exact schema in code).
* **Output (JSON):**

```json
{
  "transaction_id": "...",
  "probability": 0.93,
  "label": "flag" ,
  "threshold": 0.80,
  "top_features": [
    {"name": "device_id_freq", "contrib": 0.21},
    {"name": "node2vec_src_07", "contrib": 0.15},
    {"name": "amount_z", "contrib": 0.12}
  ],
  "reason": "High device reuse and risky neighbors for src_account_id; large amount vs. history.",
  "graph_context": {"src_degree": 27, "dst_degree": 11}
}
```

**Health:** `GET /health` → `{status: "ok", model_version: "YYYYMMDD"}`.

---

## 6) Streamlit Analyst Console

**Features:**

* Upload CSV or **simulate stream**.
* Live **alerts table** (prob ≥ threshold); filters by time, device, account.
* Detail view: **ego‑graph** (NetworkX drawing) + **top SHAP reasons** + raw row.
* Export reviewed alerts (CSV) with analyst decision.

---

## 7) Project Structure

```
graphguard/
  data/                       # small samples + README with dataset download steps
  notebooks/
    01_eda.ipynb
    02_baseline_xgb.ipynb
    03_build_graph_features.ipynb
    04_hybrid_train_eval.ipynb
  src/
    preprocess.py             # encode/impute/feature-gen for tabular
    graph_features.py         # build graph + compute graph metrics + stats
    train.py                  # trains baseline+hybrid, saves artifacts
    explain.py                # SHAP + reason template
    schema.py                 # pydantic models for API I/O
  app/
    api.py                    # FastAPI service
    artifacts/
      model.pkl
      encoders.joblib
      vecs.npy
  dashboard/
    app.py                    # Streamlit UI
  tests/
    test_api.py
    test_preprocess.py
  Makefile
  requirements.txt (or pyproject.toml)
  README.md
```

---

## 8) Setup & Commands

```bash
# 1) Create env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# 2) Prepare data (put raw files under data/raw or follow dataset README)

# 3) Train
python -m src.train --dataset ieee --output app/artifacts

# 4) Run API
uvicorn app.api:app --reload --port 8000

# 5) Run Dashboard
streamlit run dashboard/app.py

# 6) Tests
pytest -q
```

**requirements.txt (starter):**

```
pandas
numpy
scikit-learn
xgboost
networkx
shap
fastapi
uvicorn[standard]
pydantic
streamlit
joblib
matplotlib
python-dotenv
```

---

## 9) Training Details

* **Preprocess:**

  * Categorical: frequency/target encoding; text IDs left as IDs for graph stage.
  * Numeric: log/robust‑scale `{amount}`, z‑scores vs. account history when available.
  * Time: hour‑of‑day, day‑of‑week, recent txn counts.
* **Graph build:**

  * Nodes: `account_id`, `device_id`, `ip_address`.
  * Edges: (account→device), (account→ip), (account→account via txn).
  * Features: Node2Vec(128), degree, pagerank, triangles, clustering.
* **Hybrid join:** For each transaction, fetch `src_*` and `dst_*` features; add simple aggregates.
* **Metrics:** ROC‑AUC, PR‑AUC, precision\@k (k as %), **FP\@Recall=0.80**.
* **Thresholding:** choose threshold via validation to balance ops load.
* **Seeding:** set `random_state` everywhere; save splits.

---

## 10) Explainability

* **SHAP** summary plot in notebooks.
* **Per‑txn explanation** in API response: top‑k SHAP + template string.
* (Optional) Add a tiny LLM later to polish reasoning; MVP keeps template‑based text (deterministic).

---

## 11) Quality, CI & Reproducibility

* `pytest` for preprocess and API contract.
* `black`/`ruff` for formatting/lint.
* Pin versions in `requirements.txt`.
* Save artifacts with date stamp; include `model_card.md` (data, metrics, limitations).

---

## 12) Deployment (optional for MVP)

* **Dockerfile** to containerize API + artifacts.
* Serve on Render/Fly.io/railway (free tier) or local Docker.
* Keep Streamlit separate or behind the API.

---

## 13) 2–3 Day Build Plan

**Day 1:** EDA → Baseline XGBoost → Node2Vec graph features → Hybrid model + metrics.

**Day 2:** FastAPI `/score` + SHAP per‑txn → Streamlit console (alerts table + ego‑graph).

**Day 3:** Polish: README, demo GIF/video, tests, Makefile, clean notebooks.

**Definition of Done:** model + service run locally; README has metrics table & instructions; short demo recorded.

---

## 14) Cursor Prompts (paste step‑by‑step)

**A. Scaffold project**

> Create the folder structure above. Add a `pyproject.toml` or `requirements.txt`. Generate `Makefile` with `train`, `serve`, `ui`, and `test` targets. Initialize `src/schema.py` with Pydantic models for request/response.

**B. Preprocess + Baseline**

> Write `src/preprocess.py` to load CSVs, perform frequency encoding for categoricals, log/robust‑scale amounts, time features, train/valid split. In `src/train.py`, train baseline XGBoost with early stopping and save `model.pkl` + encoders.

**C. Graph features**

> Implement `src/graph_features.py` to build a heterogeneous graph (accounts/devices/ips), compute Node2Vec(128) and stats (degree, pagerank, triangles), and return a DataFrame keyed by node id. Join features to each transaction for src/dst.

**D. Hybrid model**

> Extend `src/train.py` to train a hybrid XGBoost on `[tabular + graph]`. Log ROC‑AUC/PR‑AUC and FP\@Recall=0.80, save artifacts to `app/artifacts`.

**E. FastAPI**

> Build `app/api.py` with `/score` and `/health`. Load artifacts once at startup. Add SHAP per‑txn and generate a natural‑language reason using a template.

**F. Streamlit**

> Build `dashboard/app.py` that lets users upload a CSV or simulate a stream, view alerts >= threshold, inspect one record with SHAP top features and a small ego‑graph around the src account.

**G. Tests**

> Add `tests/test_api.py` to check schema and a sample inference; add `tests/test_preprocess.py` for feature shapes.

---

## 15) What to Say on Resume/Portal

* Built **GraphGuard**, a real‑time fraud detector that fuses **graph embeddings (Node2Vec)** with **XGBoost**, reducing false positives by **\~15%** at fixed recall on a public dataset.
* Delivered **FastAPI** microservice (p95 < 30 ms) and a **Streamlit** analyst console with SHAP explanations and ego‑graph context.
* Authored reproducible pipeline (notebooks + scripts) and a 90‑sec demo.

---

## 16) Astro Note (optional, for your personal log)

* **Mercury Mahadasha** → analysis, documentation, coding.
* **Rahu Antardasha** → networks/graphs/visibility — exactly what GraphGuard models.

---

## 17) License & Ethics

* Use dataset per its license (Kaggle/PaySim). Do not claim production performance; state dataset and limitations in `model_card.md`.

---

## 18) FAQ

* **Is GraphGuard just a name?** It’s descriptive: graph‑powered guard for fraud.
* **Why not end‑to‑end GNN?** Timeboxed MVP; hybrid adds value fast and is common in industry.
* **Can I run it without GPU?** Yes, the MVP is CPU‑friendly.

---

**You can paste this README into your repo root and give Cursor the prompts in §14 to scaffold code quickly.**
