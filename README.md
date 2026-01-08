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

## Getting Started

### 1. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Configure the Pipeline

Edit `configs/config.yaml` to set:
- MySQL connection info
- SQL queries for metadata and data
- Embedding model name
- Vectorstore settings
- MLflow experiment/model names
- Kafka server URI and topics

### 3. Run Embedding Producer

```powershell
python run_embedding.py --kafka_uri localhost:9092 --topic text_embedding --batch_size 500
```

### 4. Run Detection Consumer

```powershell
python run_detection.py --mlflow_uri file:./mlruns --experiment "SMS SPAM DETECTION"
```

### 5. Main Orchestrator

```powershell
python app.py
```

## Testing

Unit tests are provided in the `tests/` folder. Run all tests with:

```powershell
pytest
```

## Key Dependencies

- Python 3.10+
- pandas, numpy
- prefect
- mlflow
- kafka-python
- sentence-transformers, langchain, langchain-huggingface
- faiss-cpu

## Notes

- Ensure Kafka and MySQL services are running and accessible.
- MLflow UI can be started with `mlflow ui` for experiment tracking.
- Vectorstore uses FAISS for fast similarity search.

---

**For more details, see comments in each main script and configuration file.**