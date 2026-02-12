# SMS Spam Detection Pipeline

This repository implements an active-learning SMS spam detection pipeline built with embeddings, a vector store, and incremental training. It combines automated pseudo-labeling with human review and uses Prefect for orchestration and MLflow for experiment tracking.

Quick Project Layout

- `main.py` - primary Prefect flow & pipeline implementation
- `simulate_request.py` - for load testing, simulate payloads to model through API
- `configs/config.yaml` - main configuration (queries, thresholds, feature names)
- `deploy/` - for deployment include FastAPI and DockerFile
- `data_validation/` - data type validation
- `src/data_loader/` - DB connection, embedding, preprocessing
- `src/ml/` - model training and utilities
- `src/vector_database/` - FAISS-backed vector store wrapper
- `src/utils/` - environment and helper utilities 
- `mlartifacts/` and `mlruns/` - local MLflow artifacts (optional)

Quickstart

1. Create and activate a virtual environment and install dependencies, make sure you `cd` into the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure `configs/config.yaml` with DB and MLflow settings.

3. Start Prefect and MLFlow

```bash
prefect server start --host host_number, see more in .env file (PREFECT_API_URL) 
mlflow server --host host_number
```

4. Run the pipeline:

```bash
python main.py
```

Loading and evaluating a saved model

If you saved models to MLflow, load them by run or registry URI. Examples:

```python
import os
import mlflow
mlflow.set_tracking_uri("<MLFlOW URI>") 

model = mlflow.sklearn.load_model(f"models:/<model_name>/<model_version>") # eg: "models:/linear_model/1"
```

API Testing

If you want to test the API, start FastAPI. Example:

```bash
cd ./deploy

uvicorn main:app --host host_number --port port_number --reload
```

Then, you could the API with FastAPI web interface, http://<host_number>/docs, usually it will show at the terminal when activated
Or, call the url with any code you prefer


Possible Improvement

- Automate labelling process after each epochs
Currently, the human labelling are required after each training. Find a way to automate this to save some time

- Initial Dataset Optimization
The question of "could I reduce the initial data lesser without lossing performance" always comes to my mind. There is one more method that I did not has a chance to try. What if we not brute forcely select a time range in every day ? What if we could really find out the hours that cover most sms spam or ham message patterns ? 
Create a python scripts that take 5% of random sample data from certain time range, perform HDBSCAN to retrieve cluster id, and select only 5 items from each cluster. Insert all selected items to MySQL and use it as initial dataset. This could help further reduce the data size we need to label but preserve all possible sms pattern.