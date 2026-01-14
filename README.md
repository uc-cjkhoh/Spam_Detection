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

1. Setup required components (database, model, etc)
    - Check data folder if initial data batch exists 
    - Check if the initial data batch has fully labeled
2. Check if the model was fitted or not
    - if no, fit initial data to model
    - if yes, update model
3. Perform stratified sampling
4. Preprocess data
    - Normalize data
    - Convert data to vectors
    - Perform dimension reduction
5. Perform classification
6. Label data by confidence score
    - if confidence score >= threshold, label as `1`
    - if confidence score < threshold, label N uncertain sms as `-1`, the reset as `0`
        - uncertain sms determine by confidence score closer to |p - 0.5|
7. Save model, update mysql  
8. Repeat step 2 - 7
 
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

---

This README was updated to reflect the active-learning pipeline implemented in `main.py` and the code under `src/`.
