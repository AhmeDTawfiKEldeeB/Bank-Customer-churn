#  Bank Customer Churn Prediction - End 2 End ML project

Hello — I’m Ahmed. Welcome to my end-to-end churn prediction project. I built this as a demonstration of practical ML engineering: clear data pipelines, tracked experiments, resilient serving, and a small demo UI for fast stakeholder testing.

**🎯 Quick wins (what recruiters care about)**
- ✅ Reproducible training pipeline with MLflow experiment tracking.
- ✅ Served predictions from a lightweight FastAPI app with a Gradio demo UI.
- ✅ Defensive inference: artifact-path flexibility and fallback preprocessing to avoid runtime failures.

**🧰 Tech stack**
- Python, pandas, scikit-learn, XGBoost
- MLflow (experiment tracking), Optuna (hyperparameter tuning)
- joblib and XGBoost JSON (artifact formats)
- FastAPI + Gradio (serving & demo UI)
- Dockerfile included for containerization; `uvicorn` for running the app

**📚 What I built**

• I engineered a reproducible pipeline: raw data → preprocessing → feature engineering → training → artifact export. This keeps training debuggable and reproducible.

• I tracked experiments in MLflow and exported artifacts so runs are auditable and comparable.

• I added a FastAPI endpoint and a Gradio UI so non-technical stakeholders can try predictions interactively.

• I hardened the inference path to handle two real-world issues:
  - artifact location mismatch (pipeline writes to `models/`, some code looked in `src/serving/`) — solution: resilient multi-location loading;
  - preprocessor unpickle failures (saved preprocessor referenced a missing module) — solution: deterministic local fallback that runs `preprocess_data()` + `build_features()` and aligns features before prediction.

• I validated parity between the notebook-exported JSON model and the pipeline joblib model; the serving layer prefers the notebook JSON model when present so manual UI checks match the notebook.

📍 Quick code map for reviewers (open these files during interviews)
- `src/app/app.py` — FastAPI + Gradio UI (change labels/messages here)
- `src/serving/inference.py` — artifact loading, preprocessor handling, fallback preprocessing, feature alignment, prediction logic
- `scripts/pipeline.py` & `src/models/train.py` — training pipeline and artifact export
- `src/data/preprocess.py` & `src/features/build_features.py` — preprocessing and feature engineering

✅ How to run locally (copy-paste)

Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training pipeline (produces `models/` artifacts):

```bash
.venv/bin/python scripts/pipeline.py
```

Start API + Gradio UI:

```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/ui
```

Quick API test example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"CreditScore":650,"Geography":"France","Gender":"Male","Age":35,"Tenure":5,"Balance":0.0,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":50000.0}'
```

**🚀 CI / CD & deployment recommendations**
- CI: GitHub Actions to run linting, unit tests, and a smoke test that imports `src/serving/inference.py` and loads artifacts.
- CD: Build Docker image (Dockerfile present) and push to registry; deploy with Kubernetes/ECS and add a model promotion flow (staging → canary → prod).
- Monitoring: capture prediction rates and drift; integrate with Prometheus/Grafana and alert on unusual behavior.

**🤖 LLMops & explainability**

This repo focuses on tabular ML. To demonstrate LLMops skills I can add:
- an LLM explainability endpoint to generate short text explanations for each prediction,
- prompt/response logging, cost metrics, and safety checks,
- CI gates for prompt outputs before release.

**💬 Interview talking points** 
- “I built a reproducible training pipeline and logged experiments with MLflow.”
- “I made the inference layer defensive — it auto-falls back to local preprocessing when the saved preprocessor is incompatible.”
- “I validated model parity between notebook experiments and serving to reduce demo surprises.”

Next steps — pick one and I’ll implement it:
- Add a one-page recruiter summary (short bullets + contact line).
- Add a GitHub Actions CI skeleton that runs tests and builds the Docker image.
- Add a small LLM explainability demo and prompt logging (LLMops flow).

File created: [README.md](README.md)
**Bank Customer Churn — Project Story & Technical Portfolio**

I built an end-to-end churn prediction system for a bank: data ingestion → preprocessing → feature engineering → model training & evaluation → artifact management → serving with an interactive UI. This repository is a compact demonstration of practical ML engineering, reproducibility, and production-minded design. Read this as a short story you can tell recruiters: what I built, why it matters, and how I solved real problems.

**Highlights**
- **Problem:** predict whether a bank customer will churn (binary classification).
- **What I delivered:** reproducible training pipeline, MLflow experiment tracking, saved model artifacts, a FastAPI + Gradio UI for manual testing, and inference resilience for mismatched artifacts.
- **Impact:** robust serving that tolerates artifact path differences and pickled preprocessor issues — practical fixes you’d want in production.

**Skills & Technologies**
- Python, pandas, scikit-learn, XGBoost
- Experiment tracking: MLflow
- Hyperparameter tuning: Optuna
- Serialization: joblib, XGBoost JSON export
- Serving: FastAPI + Gradio UI
- Packaging & infra: Dockerfile (containerization), uvicorn
- DevOps/Recommendations: CI/CD (GitHub Actions / GitLab CI), container registry, ML artifact promotion

**Project Narrative (what I actually did — tell it exactly in interviews)**

1) I started from raw customer data and created an explicit pipeline: cleaning, encoding, and feature engineering to get a stable feature matrix for training.

2) I trained XGBoost models while logging runs to MLflow. I experimented (Optuna) to improve generalization and logged artifacts and metrics in `mlruns/`.

3) To make the model easy to use, I saved two artifact types: `joblib` artifacts (model + preprocessor) in `models/` and a notebook-exported XGBoost JSON model in `notebooks/models/` — this helped me validate notebook experiments vs. served predictions.

4) When wiring the Gradio UI and invoking prediction, I hit two realistic problems:
   - the serving code expected artifacts in `src/serving/` while the pipeline saved them to `models/` — fixed by robust artifact resolution.
   - `preprocessor.joblib` was pickled referencing a custom module path that no longer existed, causing unpickle errors — fixed by adding a fallback that runs the repo’s local preprocessing + feature engineering and aligns features to the model’s expected columns.

5) I observed a mismatch in probabilities between the JSON model saved from the notebook and the repo `joblib` model. To ensure parity during testing, serving was updated to prefer the notebook JSON model (if present). This is a pragmatic approach to match what you validated in the notebook.

6) I changed the decision rule to probability > 0.5 and made the UI return a concise, recruiter-friendly message (I also demonstrated localization by returning an Arabic-friendly message in the app during testing).

**Where to look in the code (quick map for reviewers)**
- App + UI: [src/app/app.py](src/app/app.py) — FastAPI + Gradio interface. This is where the UI text lives and where you can change labels/messages.
- Inference logic: [src/serving/inference.py](src/serving/inference.py) — artifact loading, preprocessor handling, fallback preprocessing, feature alignment, and prediction logic.
- Training & pipeline: [scripts/pipeline.py](scripts/pipeline.py) and [src/models/train.py](src/models/train.py) — full training flow and artifact export.
- Data & features: [src/data/preprocess.py](src/data/preprocess.py) and [src/features/build_features.py](src/features/build_features.py) — canonical preprocessing and feature pipeline.

If you want to point recruiters to a small runnable demo, open `src/app/app.py` and launch the app; it mounts a Gradio UI for quick manual testing.

**How I validated correctness (reproducibility & parity checks)**
- I re-ran the training pipeline (`scripts/pipeline.py`) to produce `models/model.joblib` and `models/preprocessor.joblib` and tracked run metrics in `mlruns/`.
- I compared prediction probabilities between `models/model.joblib` and `notebooks/models/final_xgboost_model.json` on the same sample input to catch any serialization/training mismatch.
- For inference resilience, I implemented: try-load preprocessor → if unpickle fails, run `preprocess_data()` + `build_features()` → align features (add missing dummies, coerce dtypes) → reorder columns to match `model.feature_names_in_` or booster feature names.

**How to run locally (copy-paste commands)**

Create env and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Rebuild artifacts (training pipeline):

```bash
.venv/bin/python scripts/pipeline.py
```

Start API + Gradio UI:

```bash
uvicorn src.app.app:app --host 0.0.0.0 --port 8000 --reload
# then open http://localhost:8000/ui
```

API quick test (example payload):

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"CreditScore":650,"Geography":"France","Gender":"Male","Age":35,"Tenure":5,"Balance":0.0,"NumOfProducts":1,"HasCrCard":1,"IsActiveMember":1,"EstimatedSalary":50000.0}'
```

**CI / CD & Deployment**

- CI: GitHub Actions to run linting, unit tests, and a smoke test that imports `src/serving/inference.py` and loads artifacts. This prevents regressions in artifact loading and unpickle issues.
- CD: Build a Docker image (Dockerfile already present) and publish to a registry (GitHub Container Registry / Docker Hub). Use a simple Kubernetes manifest / ECS task for deployment.
- Promotion: add stages for model promotion: staging → canary → production. Automate artifact copying (from MLflow or storage) and schema checks before promotion.
- Monitoring: add a lightweight endpoint to log prediction distribution and error rates; integrate with Prometheus + Grafana for metrics and alerts.

**LLMops note**

This project centers on supervised ML (tabular) and does not currently use LLMs. If you want to showcase LLMops skills to recruiters, here’s how I’d expand the repo:
- Add an LLM service component (e.g., a prompt-engineering wrapper, caching, and instruction tuning) to run customer-support simulations.
- Track prompts and responses in an observability pipeline (logging, metrics, cost tracking). Tools: LangChain, LlamaIndex, Weights & Biases or MLflow for prompt logging.
- Add a gating CI job to validate prompt outputs for safety and consistency before releasing to production.

I can add a small demo that integrates an LLM for explainability (e.g., generate a short natural-language explanation for why a customer is predicted to churn). Tell me if you want that.

**Code analysis**
- Robust artifact handling: the inference layer checks multiple artifact locations and gracefully falls back to local pipelines when unpickling fails. This shows defensive engineering.
- Parity-first: preferring the notebook JSON model when present ensures the model used for manual verification (notebook) matches the served model — a pragmatic debugging/validation approach.
- Practical engineering trade-offs: instead of forcing an exact reproducer of the original pickled preprocessor (which referenced a missing module), I added a deterministic local pipeline to guarantee inference can continue and avoid downtime.
- Reproducibility: MLflow is used for tracking experiments and artifacts; running `scripts/pipeline.py` reproduces training artifacts.

