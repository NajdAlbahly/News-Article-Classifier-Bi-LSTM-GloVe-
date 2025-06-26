# News Category Classifier

This project implements a deep learning-based text classification system designed to automatically categorize English news headlines into one of four distinct topics. It leverages a BiLSTM architecture enhanced with pre-trained GloVe word embeddings and is trained on the widely-used AG News dataset."

## Task
Automatically categorize news content into one of the following:
- World
- Sports
- Business
- Sci/Tech

## Model Details
- Type: BiLSTM (Bidirectional LSTM)
- Embeddings: GloVe (100-dimensional)
- Dataset: AG News
- Framework: TensorFlow / Keras

## Repository Contents
- `news_classifier_app.py`: Inference script with a simple Gradio web interface.
- `requirements.txt`: Python dependencies used in this project.
- `ag_news_train.csv`, `ag_news_test.csv`: Processed training and test datasets from AG News.

## Notes
This repository is for personal learning and experimentation with text classification, NLP preprocessing, and model deployment using Gradio.
