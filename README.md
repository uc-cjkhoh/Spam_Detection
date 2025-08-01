# Spam Detection from MySQL SMS Data

This project provides a modular pipeline for detecting spam in SMS messages sourced directly from a MySQL database. It covers data loading, preprocessing, feature engineering, exploratory data analysis, and is designed for extensibility with model training and evaluation.

## Project Structure

```
Spam_Detection/
├── .gitignore
├── README.md
├── requirements.txt
├── copilot-instructions.md
├── main.py
├── eda_result.txt
├── configs/
<<<<<<< HEAD
│   └── config.yaml
├── result/
│   ├── 20250801-1509_decoded_message.csv
│   └── 20250801-1514_filtered_message.csv
└── src/
    ├── data_loader.py
    ├── preprocess.py
    ├── eda.py
    ├── train.py
    ├── model.py
    └── decorators.py
=======
│   └── config.yaml           # Configuration for database and model parameters
├── models/                   # (Reserved for trained models)
├── src/
│   ├── data_loader.py        # MySQL database connection and data retrieval
│   ├── preprocess.py         # Data cleaning and feature engineering
│   ├── eda.py                # Exploratory data analysis utilities
│   ├── train.py              # (Reserved for training logic)
│   ├── model.py              # Model definition, embedding, and training
│   └── __pycache__/          # Python cache files
├── test/                     # (Reserved for test scripts)
├── main.py                   # Main pipeline script
├── requirements.txt          # Python dependencies
├── copilot-instructions.md   # Instructions for GitHub Copilot
└── README.md                 # Project documentation
>>>>>>> 7b25d2355867d8f65eb206f5761bede1b10bb05e
```

## File/Folder Descriptions

- `.gitignore`  
  Python virtual environment and cache ignore rules.

- `README.md`  
  Project documentation (this file).

- `requirements.txt`  
  Python dependencies for the project.

- `copilot-instructions.md`  
  Coding standards and helper rules for GitHub Copilot.

- `main.py`  
  Main pipeline script: loads config, fetches data, runs preprocessing, EDA, feature engineering, normalization, and model training.

- `eda_result.txt`  
  Output of exploratory data analysis (EDA) statistics.

- `configs/config.yaml`  
  Configuration for database connection, queries, and model parameters.

- `result/`  
  Stores output CSV files with model results.

- `src/data_loader.py`  
  Handles MySQL database connection and data retrieval.

- `src/preprocess.py`  
  Data cleaning, normalization, and feature engineering.

- `src/eda.py`  
  Exploratory data analysis utilities.

- `src/train.py`  
  (Reserved for training logic.)

- `src/model.py`  
  Model definition, embedding, and training logic.

- `src/decorators.py`  
  (Reserved for decorators/utilities.)

## Workflow Overview

1. **Configuration**  
<<<<<<< HEAD
   - Set up database and model parameters in `configs/config.yaml`.
=======
   - Database, query, and model settings are managed in [`configs/config.yaml`](configs/config.yaml).
>>>>>>> 7b25d2355867d8f65eb206f5761bede1b10bb05e

2. **Data Loading**  
   - `src/data_loader.py`: Connects to MySQL and fetches SMS data.

3. **Preprocessing & Feature Engineering**  
<<<<<<< HEAD
   - `src/preprocess.py`: Cleans and augments data with features.

4. **Exploratory Data Analysis**  
   - `src/eda.py`: Provides data statistics and visualization.

5. **Modeling**  
   - `src/model.py`: Embeds text, trains, and evaluates models.

6. **Main Pipeline**  
   - `main.py`: Orchestrates the entire workflow.

7. **Results**  
   - Output CSVs are saved in the `result/` folder.
=======
   - [`src/preprocess.py`](src/preprocess.py):  
     - Cleans text (fixes mojibake, strips whitespace, removes emojis, converts to lowercase).
     - Adds features: message length, numeric/special character counts, URL and phone number detection, language detection, and custom filters.

4. **Exploratory Data Analysis**  
   - [`src/eda.py`](src/eda.py):  
     - Provides basic data description and visualization utilities.

5. **Modeling**  
   - [`src/model.py`](src/model.py):  
     - Embeds text using transformer models (e.g., SentenceTransformer).
     - Supports model training and inference using Hugging Face pipelines or custom classifiers.
     - Handles both text and vector-based classification.

6. **Main Pipeline**  
   - [`main.py`](main.py):  
     - Loads configuration and connects to the database.
     - Fetches data, applies preprocessing, EDA, and feature engineering.
     - Normalizes data and runs model training/inference.
     - Saves results with timestamped filenames.

7. **Extensibility**  
   - Reserved files for advanced training logic (`src/train.py`) and testing (`test/`).

## Key Features

- **Database Integration:** Securely connects and queries MySQL for SMS data.
- **Robust Preprocessing:** Handles text encoding issues, emoji removal, and extracts relevant features for spam detection.
- **Feature Engineering:** Identifies URLs, phone numbers, language, and custom patterns in messages.
- **Flexible Modeling:** Supports both transformer-based text classification and vector-based classification.
- **Modular Design:** Easily extendable for new models, features, or data sources.
- **Copilot Instructions:** Coding standards and helper rules in [`copilot-instructions.md`](copilot-instructions.md).
>>>>>>> 7b25d2355867d8f65eb206f5761bede1b10bb05e

## Requirements

See `requirements.txt` for dependencies, including:
- `mysql-connector-python`
- `numpy`
- `pandas`
- `ftfy`
- `emoji`
- `pyyaml`
- `lingua-language-detector`
- `scikit-learn`
- `sentence-transformers`
- `einops`
- `jupyter`
<<<<<<< HEAD
=======
- (and others as the project grows)
>>>>>>> 7b25d2355867d8f65eb206f5761bede1b10bb05e

## Usage

1. **Set up your environment:**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure your database and model parameters in `configs/config.yaml`.**

3. **Run the main pipeline:**
   ```sh
   python main.py
   ```

## Customization

- Add or modify model logic in `src/model.py` and training routines in `src/train.py`.
- Extend EDA in `src/eda.py` for deeper insights.
- Adjust feature engineering in `src/preprocess.py` as needed.

## Troubleshooting

- **Model Download Errors:** Ensure you have a stable internet connection for downloading Hugging Face models.
- **File Paths:** Check that all file paths in `config.yaml` and scripts are correct relative to your working directory.
- **Vector-based Models:** If using models like `jinaai/jina-embeddings-v4`, ensure you specify the `task` parameter when encoding.

## Copilot Instructions

See `copilot-instructions.md` for coding standards and helper rules.

---

**Author:**  
<<<<<<< HEAD
Khoh Chia
=======
Khoh Chia
>>>>>>> 7b25d2355867d8f65eb206f5761bede1b10bb05e
