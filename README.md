# SMS Spam Detection Pipeline

This repository implements an incremental SMS spam detection system using embeddings, vector storage, and online learning. The pipeline is orchestrated with Prefect and supports streaming data via Kafka.

## Features

- **Data Loading**: Connects to a MySQL database, retrieves metadata and message data.
- **Preprocessing**: Normalizes and cleans SMS text.
- **Embeddings**: Generates text embeddings using Hugging Face models (via LangChain).
- **Vector Storage**: Stores embeddings and metadata in a vector database (FAISS).
- **Streaming**: Uses Kafka for message and embedding streaming between pipeline stages.
- **Model Training**: Online learning with SGDClassifier, incremental updates using high-confidence pseudo-labels.
- **Experiment Tracking**: MLflow integration for model and experiment management.
- **Orchestration**: Prefect flows for robust, observable pipeline execution.

## Project Structure

```
├── app.py                  # Main orchestrator (entrypoint)
├── run_detection.py        # Streaming detection flow (Kafka consumer)
├── run_embedding.py        # Embedding and streaming flow (Kafka producer)
├── configs/
│   └── config.yaml         # Main configuration file
├── requirements.txt        # Python dependencies
├── src/
│   ├── config_folder/      # Config loader
│   ├── data_loader/        # DB connection, embedding, preprocessing
│   ├── ml/                 # Model training, LLM utilities
│   ├── utils/              # Utility functions (env setup, metadata, etc.)
│   └── vector_database/    # VectorStore (FAISS)
└── tests/
    ├── config_folder/      # Config loader tests
    ├── data_loader/        # Data loader tests
    ├── ml/                 # Model training tests
    └── utils/              # Utility tests
```
# SMS Spam Detection (Active Learning)

This repository implements an active-learning SMS spam detection pipeline. It uses sentence embeddings, a vector store, and iterative model updates with pseudo-labeling and human review. The pipeline is orchestrated with Prefect and logs experiments to MLflow.

Key capabilities
- Database extraction (MySQL)
- Text normalization and embeddings
- Dimensionality reduction (PCA)
- Classifier training and incremental updates
- Pseudo-labeling (high-confidence) + human review (uncertain samples)
- Experiment tracking with MLflow

Project layout (important files)

- [main.py](main.py) — primary Prefect flow & pipeline implementation
- [configs/config.yaml](configs/config.yaml) — main configuration
- [src/data_loader/preprocessing.py](src/data_loader/preprocessing.py) — normalization & preprocessing
- [src/data_loader/embedding.py](src/data_loader/embedding.py) — embedding helpers
- [src/ml/model_training.py](src/ml/model_training.py) — model classes and training utilities
- [src/utils/util.py](src/utils/util.py) — setup, metadata helpers
- [src/vector_database/vectorstore.py](src/vector_database/vectorstore.py) — vector store wrapper
- [requirements.txt](requirements.txt) — Python dependencies
- [tests/](tests/) — unit tests

Pipeline procedure (high level)

1. Setup required components
    - `setup_environment()` loads config, DB connection, embedding model, vectorstore and model.
2. Stratified sampling by time of day of a specific day
    - Query database with `config.data.query` and perform stratified selection by hour.
3. Save the initial batch locally as an Excel file
    - `download_initial_data()` writes to `config.models.initial_data_filepath`.
4. Developer manually labels the first batch of data
    - Pipeline will halt until `finish_labelling()` returns true.
5. Initialize or train model
    - Compute embeddings, reduce dimension, oversample (SMOTE) and fit the model.
    - Ensure the training data is unbiased (stratified) and balanced (SMOTE or other).
6. Select another batch of data with stratified sampling
7. Classify those messages with the model
8. Label high-confidence SMS as `pseudo` (confidence >= 0.975)
9. Select ~1000 most-uncertain SMS (closest to 0.5) and mark for human labeling (`human`)
10. Mark the remaining as unlabeled (`''`)
11. Save results and update metadata
     - `database.save_to_mysql(...)` persists id, datetime, spam_label, confidence_score, label_status, model
12. Repeat steps 6–11 until evaluation threshold reached (e.g., accuracy or other metric)

Quickstart (local development)

1. Create and activate virtualenv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure `configs/config.yaml` (database, queries, embedding model, mlflow)

3. Run the pipeline

```bash
python main.py --mlflow_uri=http://localhost:5000 --experiment="SMS SPAM DETECTION" --skip_initialization=False
```

Notes on configuration & runtime
- MLflow: set `--mlflow_uri` to your MLflow server or `file:./mlruns` for local runs.
- Prefect: the code uses Prefect `@task` and `@flow`; configure Prefect backend if using remote orchestration.
- Embedding caching: to avoid re-calculating embeddings for identical inputs, add a Prefect `cache_key_fn` and enable `persist_result` with a result backend (filesystem or S3).

Testing

Run unit tests with `pytest`:

```bash
pytest -q
```

Development notes
- The main orchestrator is `main.py`; adjust sample sizes, thresholds and model hyperparameters in `configs/config.yaml` or in `src/ml/model_training.py`.
- If you hit sklearn errors like "eta0 must be > 0", ensure model parameters are valid before training (e.g., set `eta0 > 0` for SGD-based classifiers).

Contact / next steps
- If you'd like, I can also:
  - Add embedding caching (Prefect cache key + persisted results).
  - Add a `make` or `tox` task for running tests and linting.

---

This README was updated to reflect the active-learning pipeline implemented in `main.py` and the code under `src/`.
