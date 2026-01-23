# SMS Spam Detection Pipeline

This repository implements an incremental SMS spam detection system using embeddings, vector storage, and online learning and trained in a teacher-student manner.
# SMS Spam Detection Pipeline

This repository implements an active-learning SMS spam detection pipeline built with embeddings, a vector store, and incremental training. It combines automated pseudo-labeling with human review and uses Prefect for orchestration and MLflow for experiment tracking.

Goals
- Provide a reproducible pipeline for training and evaluating SMS spam models
- Support incremental / online updates with teacher-student labeling
- Track experiments and artifacts with MLflow

Quick Project Layout

- `main.py` — primary Prefect flow & pipeline implementation
- `configs/config.yaml` — main configuration (queries, thresholds, feature names)
- `src/data_loader/` — DB connection, embedding, preprocessing
- `src/ml/` — model training and utilities
- `src/vector_database/` — FAISS-backed vector store wrapper
- `src/utils/` — environment and helper utilities
- `mlartifacts/` and `mlruns/` — local MLflow artifacts (optional)

Quickstart (local)

1. Create and activate a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure `configs/config.yaml` with DB and MLflow settings.

3. Run the pipeline locally (example):

```bash
python main.py --mlflow_uri=http://localhost:5000 --experiment="SMS_SPAM_DETECTION_V3"
```

Loading and evaluating a saved model

If you saved models to MLflow, load them by run or registry URI. Examples:

```python
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file://" + os.path.abspath("./mlartifacts"))
client = MlflowClient()

# list runs in experiment
exp = client.get_experiment_by_name("SMS_SPAM_DETECTION_V3")
runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=10)
print([r.info.run_id for r in runs])

# discover artifact paths for a run
def list_artifacts_recursive(client, run_id, path=""):
    out = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            out += list_artifacts_recursive(client, run_id, item.path)
        else:
            out.append(item.path)
    return out

run_id = runs[0].info.run_id
print(list_artifacts_recursive(client, run_id))  # typically contains 'model'

# load the MLflow sklearn model (common artifact path = 'model')
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
```

Why accuracy may be unstable

- Transformers fit per-call: Ensure PCA/Scaler are fitted once on training data and reused for inference to avoid data leakage.
- Label handling: Keep labels as a 1-D integer array (`y = np.asarray(y).ravel().astype(int)`).
- SMOTE misuse: Apply SMOTE after splitting on the full training set (not per mini-batch).
- Probability handling: For binary, use `predict_proba(X)[:, 1]` for positive-class confidence.
- Model calibration: `SGDClassifier` may need `CalibratedClassifierCV` for reliable probabilities.

Best practices and production recommendations

- Separate responsibilities:
  - Real-time inference: a small FastAPI service that loads the MLflow model and returns predictions.
  - Offline / training: Prefect flows run training, evaluation and MLflow logging.
  - Artifacts and registry: MLflow Tracking Server + artifact store (S3 or local `mlartifacts/`).

- Use a reproducible preprocessing pipeline (sklearn `Pipeline`) that chains scaler → PCA → classifier and serialize it with MLflow or `joblib`.

- Avoid nested `@flow` decorators in Prefect. Use `@task` for steps that run inside the same flow run; nested flows cause Prefect to serialize parameters which may fail on non-serializable objects.

- Persist transformers and use stratified splits with fixed random_state for reproducible metrics.

Troubleshooting common issues

- `RecursionError` during flow run: caused by Prefect serializing non-serializable objects (DB/clients, models). Return only serializable values from flows or instantiate heavy clients inside tasks.
- `jsonable_encoder` errors: avoid returning complex objects from flows; pass simple config or artifact URIs.
- Model logging errors: log the raw sklearn estimator (not wrapper/task objects) to MLflow: `mlflow.sklearn.log_model(sk_model=sk_model, ...)`.

Development and testing

- Run unit tests:

```bash
pytest -q
```

- Format code:

```bash
pip install black isort ruff
black src/ tests/ main.py
isort src/
ruff check --fix src/
```
