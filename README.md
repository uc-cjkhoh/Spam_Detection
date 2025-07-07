# Spam Detection with Neural Networks and Sentence Embeddings

This project implements a spam detection system using a multi-layer perceptron (MLP) neural network and sentence embeddings from transformer models. The workflow includes data preprocessing, text embedding, model training, evaluation, and batch testing. Additional utilities for text cleaning and 3D visualization of clusters are provided.

## Project Structure

- [`main.py`](main.py): Main script for training and evaluating the spam detection model.
- [`utils.py`](utils.py): Utility functions and classes, including the neural network model, training, prediction, and text cleaning.
- [`test_batch.py`](test_batch.py): Script for batch inference on new sentences using a trained model.
- [`labelling_with_3D_display.py`](labelling_with_3D_display.py): Script for clustering and visualizing sentence embeddings in 3D.
- [`spam_ham_dataset.csv`](spam_ham_dataset.csv): Dataset for training and evaluation.
- [`requirements.txt`](requirements.txt): Python dependencies.
- [`test.py`](test.py): Additional test script (purpose may vary).
- `README.md`: Project documentation (this file).

## Features

- **Text Preprocessing:** Cleaning, lemmatization, stopword removal, and more via [`utils.clean_text`](utils.py).
- **Sentence Embedding:** Uses [Sentence Transformers](https://www.sbert.net/) (`all-mpnet-base-v2`) for high-quality text embeddings.
- **Neural Network Model:** Deep MLP defined in [`utils.NeuralNetwork`](utils.py) for binary classification (spam/not spam).
- **Training & Evaluation:** Handles class imbalance with oversampling, splits data, trains the model, and evaluates with accuracy, precision, recall, and F1 score.
- **Batch Inference:** Predicts spam/not spam for a batch of sentences from a text file ([`test_batch.py`](test_batch.py)).
- **3D Visualization:** Clusters and visualizes embedded sentences in 3D using UMAP and KMeans ([`labelling_with_3D_display.py`](labelling_with_3D_display.py)).

## Getting Started

### 1. Install Dependencies

Install required packages:

```sh
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure your dataset (e.g., `spam_ham_dataset.csv`) is available in the project directory or update the path in [`main.py`](main.py).

### 3. Train the Model

Run the main script to train and evaluate the model:

```sh
python main.py
```

- The script reads the dataset, preprocesses text, generates embeddings, oversamples to balance classes, splits data, trains the neural network, and prints evaluation metrics.

### 4. Batch Prediction

To predict spam/not spam for a batch of sentences in a text file:

```sh
python test_batch.py input_sentences.txt
```

- Results are saved to an output file with a timestamp.

### 5. 3D Visualization

To visualize clusters of embedded sentences:

```sh
python labelling_with_3D_display.py
```

- This script clusters the embedded sentences and displays them in a 3D plot.

## Key Components

### Neural Network Model

Defined in [`utils.NeuralNetwork`](utils.py):

- Input: 768-dim sentence embeddings
- Deep MLP with multiple hidden layers and ReLU activations
- Output: 2 classes (spam, not spam)

### Text Cleaning

Implemented in [`utils.clean_text`](utils.py):

- Lowercasing, number and punctuation removal, stopword removal, lemmatization, and whitespace normalization.

### Training

See [`utils.train_model`](utils.py):

- Uses Adamax optimizer and cross-entropy loss.
- Model is saved after training.

### Prediction

See [`utils.predict`](utils.py):

- Computes accuracy, precision, recall, F1 score, and confusion matrix.

## Requirements

- Python 3.7+
- torch
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- nltk
- sentence-transformers
- tqdm
- plotly
- umap-learn

See [`requirements.txt`](requirements.txt) for details.

## Notes

- The model expects input embeddings of size 768 (from `all-mpnet-base-v2`).
- Update file paths as needed for your environment.
- NLTK resources are downloaded at runtime if not already present.

## License

This project is for educational and research purposes.

---

**Author:**  
chia jun