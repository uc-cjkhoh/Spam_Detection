# Spam Detection from MySQL SMS Data

This project provides a pipeline for detecting spam in SMS messages sourced directly from a MySQL database. It includes modules for data loading, preprocessing, feature engineering, exploratory data analysis, and is designed for extensibility with model training and evaluation.

## Project Structure

```
├── configs/
│   └── config.yaml           # Configuration for database and queries
├── models/                   # (Reserved for trained models)
├── src/
│   ├── data_loader.py        # MySQL database connection and data retrieval
│   ├── preprocess.py         # Data cleaning and feature engineering
│   ├── eda.py                # Exploratory data analysis utilities
│   ├── train.py              # (Reserved for training logic)
│   ├── model.py              # (Reserved for model definition)
│   └── __pycache__/          # Python cache files
├── test/                     # (Reserved for test scripts) 
├── main.py                   # Main pipeline script
├── requirements.txt          # Python dependencies
├── copilot-instructions.md   # Instructions for GitHub Copilot
└── README.md                 # Project documentation
```

## Workflow Overview

1. **Configuration**  
   - Database and query settings are managed in [`configs/config.yaml`](configs/config.yaml).

2. **Data Loading**  
   - [`src/data_loader.py`](src/data_loader.py):  
     - Connects to MySQL using credentials from the config file.
     - Executes queries to fetch SMS data.

3. **Preprocessing & Feature Engineering**  
   - [`src/preprocess.py`](src/preprocess.py):  
     - Cleans text (fixes mojibake, strips whitespace, removes emojis, converts to lowercase).
     - Adds features:
       - Message length, numeric/special character counts
       - URL and phone number detection
       - Custom filters (e.g., messages starting with 'imsi')

4. **Exploratory Data Analysis**  
   - [`src/eda.py`](src/eda.py):  
     - Placeholder for basic data description and visualization.

5. **Main Pipeline**  
   - [`main.py`](main.py):  
     - Loads configuration and connects to the database.
     - Fetches data and applies preprocessing and feature engineering.
     - Displays filtered results for inspection.

6. **Extensibility**  
   - Reserved files for model definition (`src/model.py`), training (`src/train.py`), and testing (`test/`).

## Key Features

- **Database Integration:** Securely connects and queries MySQL for SMS data.
- **Robust Preprocessing:** Handles text encoding issues and extracts relevant features for spam detection.
- **Feature Engineering:** Identifies URLs, phone numbers, and custom patterns in messages.
- **Modular Design:** Easily extendable for model training and evaluation.

## Requirements

See [`requirements.txt`](requirements.txt):

- `mysql-connector-python`
- `numpy`
- `pandas`
- `ftfy`
- (Other dependencies may be added as the project grows)

## Usage

1. **Set up your environment:**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure your database and query in `configs/config.yaml`.**

3. **Run the main pipeline:**
   ```sh
   python main.py
   ```

## Customization

- Add your model logic in `src/model.py` and training routines in `src/train.py`.
- Extend EDA in `src/eda.py` for deeper insights.

## Copilot Instructions

See [`copilot-instructions.md`](copilot-instructions.md) for coding standards and helper rules.

---

**Author:**  
Khoh Chia Jun
