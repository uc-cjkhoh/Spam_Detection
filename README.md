# SMS Spam Detection

This repository contains an incremental SMS spam-detection pipeline that uses embeddings, a vectorstore, and an online learning model. The main entrypoint is `app.py` which orchestrates environment setup, data retrieval, embedding, vector storage, prediction, and incremental training.

## What this project does

- Connects to a data source (configured in `configs/config.yaml`) and retrieves population metadata and per-population data.
- Normalizes and preprocesses text messages.
- Generates embeddings using a Hugging Face embeddings model (via `langchain_huggingface`).
- Writes message vectors and metadata into a vectorstore.
- Loads a supervised ML model (configured in `src/model`) and predicts labels on new embeddings.
- Uses prediction confidence to select high-confidence pseudo-labels for incremental training of the model.
- Tracks experiments and models with MLflow.

## Key files and structure

- `app.py` — the orchestrator flow.
- `configs/config.yaml` — main configuration (DB credentials, vectorstore config, models, queries, thresholds).
- `src/` — project source code:
  - `src/data_loader/` — data access, embedding helpers, preprocessing.
  - `src/vector_database/vectorstore.py` — vectorstore writer/reader.
  - `src/model/` — model training, loading and prediction helpers.
  - `src/config_folder/config_loader.py` — configuration loader.
  - `src/utils/util.py` — utility helpers for folders and metadata.

## Minimal contract (what `app.py` expects and produces)

- Inputs
  - `configs/config.yaml` — DB connection info, SQL queries, target column names, model params, vectorstore config, and thresholds.
  - Database access reachable from the environment (credentials in config file).
  - Network access for downloading Hugging Face models if not cached locally.
- Outputs
  - Vectorstore content (written by `VectorStore.write_to_vectorstore`).
  - MLflow runs and artifacts (by default stored in `./mlruns` unless overridden).
  - Updated metadata stored/updated by `update_metadata` helper(s).

## Dependencies

This project uses the packages listed in `requirements.txt`. To install them in your environment:

```powershell
python -m pip install -r requirements.txt
```

Important dependencies (high level):
- Python 3.10+ (project was developed and tested on modern Python 3.11/3.13 environments)
- pandas, numpy
- mlflow
- prefect
- langchain_huggingface (or the Hugging Face embedding layer used by the code)
- scikit-learn (or whichever classifier is used by `src/model`)

If you rely on GPU or large Hugging Face models, ensure you have the transformers and accelerate packages available and configured.

## How `app.py` works (step-by-step)

1. `get_config()` loads settings from `configs/config.yaml`.
2. `create_required_folder_file(config)` ensures the project folders and files exist.
3. `setup_environment(config)` (Prefect task) creates a `Database` object, instantiates a `HuggingFaceEmbeddings` model using the `config.models.text_embedding.model_name`, and a `VectorStore` object.
4. The pipeline queries population-level metadata (`config.metadata.query`), then for each population:
   - Runs `data_query` to fetch the subset of messages and metadata.
   - Preprocesses text via `text_normalize`.
   - Generates embeddings using the Hugging Face embeddings model.
   - Writes (message, embedding, metadata) pairs into the vectorstore.
   - Loads the supervised model via `load_model`.
   - Predicts labels and per-sample confidence scores.
   - Selects high-confidence samples (where confidence > `config.models.confidence_score_threshold`) and calls `train_model` using those pseudo-labels.
   - Updates metadata to mark progress.
5. The DB connection is closed at the end of the run.

## Running the pipeline

From the project root directory, run:

```powershell
python .\app.py --mlflow_uri file:./mlruns --experiment "SMS SPAM DETECTION"
```

Optional flags:
- `--mlflow_uri` — override MLflow tracking URI. Default is `file:./mlruns` (local folder).
- `--experiment` — MLflow experiment name. Default is `SMS SPAM DETECTION`.
- `--model_id` — if provided, `load_model` may use this identifier to load a specific trained model instead of starting from a new model (behavior depends on model implementation).

Notes:
- The script uses Prefect `@task` and `@flow` decorators. If you prefer not to run within a Prefect orchestration server, running the script directly will execute the Prefect flow locally.
- Ensure your `configs/config.yaml` contains valid SQL queries and DB credentials so the `Database` connector can authenticate.

## Configuration hints

- `configs/config.yaml` contains sections such as `metadata`, `data`, `vectorstore`, and `models`.
- `metadata.query` should return rows used to parameterize `config.data.query`.
- `data.query` is expected to be a format string in the config: `config.data.query.format(*metadata.iloc[i])` is used to fill parameters.
- `models.text_embedding.model_name` must be a valid Hugging Face model ID supported by the `langchain_huggingface` embedding wrapper.

## Verification / smoke test

- After a run, confirm MLflow recorded runs under `./mlruns` (or the `--mlflow_uri` you set).
- Confirm the vectorstore has entries (how to inspect depends on the vectorstore backend you configured in `configs/config.yaml`).
- Check logs in `logs/` (if the project writes logs there) and the `mlruns/` UI via `mlflow ui`.

Example quick check (local MLflow UI):

```powershell
# from project root
mlflow ui --backend-store-uri file:./mlruns --port 5000
# then open http://127.0.0.1:5000 in browser
```

## Notes, edge cases, and next steps

- If embedding downloads are large, ensure sufficient disk space and possibly configure a cache directory for Hugging Face models and tokens.
- If the DB returns no rows, the loop will be skipped; check `metadata` query correctness.
- The model's `predict_proba` must support `max(axis=1)` semantics; ensure the loaded model exposes `predict_proba` (or adapt `app.py` if the model uses different API).
- Consider adding unit tests for `text_normalize`, `VectorStore.write_to_vectorstore`, and `model_training` logic.