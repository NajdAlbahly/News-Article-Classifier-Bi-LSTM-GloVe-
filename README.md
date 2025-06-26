# 📰 News Article Classifier with BiLSTM & GloVe

This project showcases an NLP-powered text classification system trained to categorize English news headlines into one of four predefined categories: **World, Sports, Business, and Sci/Tech**. The model uses a **Bidirectional LSTM** architecture with **pre-trained GloVe embeddings** for rich semantic representation.

## Project Overview

The goal is to build a deep learning model capable of understanding short news texts and classifying them accurately. We utilized the **AG News dataset**, a well-known benchmark in NLP tasks involving news classification.

Automatically categorize news content into one of the following:
- World
- Sports
- Business
- Sci/Tech

---

Project Steps :

1. Load the Data
→ Load the AG News dataset with training and test splits.

2. Preprocessing 
→ Remove punctuation, lowercase the text, remove stopwords.

3. Tokenize and Pad
→ Convert words to numbers and pad sequences to the same length.

4. Load Word Embeddings
→ Use GloVe vectors to give meaning to each word.

5. Build the Model
→ BiLSTM model with embedding layer, dropout, and output layer for 4 classes.

6. Train the Model
→ Train on the cleaned data and validate performance.

7. Evaluate Performance
→ Check test accuracy and classification metrics.

8. Build Gradio App
→ Create a user-friendly web app for predicting news categories.

---

## Dataset

- **Source:** [AG News on Hugging Face Datasets](https://huggingface.co/datasets/ag_news)
- **Training Samples:** 120,000
- **Testing Samples:** 7,600
- **Classes:**
  - `0`: World
  - `1`: Sports
  - `2`: Business
  - `3`: Sci/Tech

The dataset is balanced across categories, making it suitable for multi-class classification tasks.

---

## Why BiLSTM and GloVe?

- **BiLSTM (Bidirectional LSTM):** Captures context from both left and right of a word, which is critical in understanding news headlines where word order affects meaning.
- **GloVe Embeddings:** Pre-trained word vectors that provide meaningful dense representations of words, improving model generalization.

---

## Preprocessing and Tokenization

Each news article is:

- Lowercased
- Stripped of punctuation
- Tokenized and cleaned
- Mapped to sequences and padded to fixed length



> **Processed Vocabulary Size:** 83,334
> **GloVe Coverage:** 18,554 / 20,000 words found in the embedding

---

## Model Architecture

```text
Embedding (GloVe pretrained, non-trainable)
→ Bidirectional LSTM (128 units)
→ Dropout (0.3)
→ Dense (Softmax for 4 classes)
```

**Embedding Matrix Shape:** (20000, 100)
**Total Parameters:** 2M (non-trainable)

---

## Project Configuration

| Parameter                 | Value               |
| ------------------------- | ------------------- |
| Max Vocabulary Size       | 20,000              |
| Max Sequence Length       | 200                 |
| GloVe Embedding Dimension | 100                 |
| GloVe File                | `glove.6B.100d.txt` |
| LSTM Units                | 128                 |
| Dropout Rate              | 0.3                 |
| Batch Size                | 64                  |
| Epochs                    | 15                  |
| Number of Classes         | 4                   |

---

## Training Performance

The model converged steadily with strong generalization after around 13–14 epochs.

---

## Evaluation Metrics

**Best Epoch:** 13
**Test Accuracy:** `92.51%`

**Classification Report:**

| Category | Precision | Recall | F1-Score |
| -------- | --------- | ------ | -------- |
| World    | 0.96      | 0.91   | 0.93     |
| Sports   | 0.96      | 0.98   | 0.97     |
| Business | 0.89      | 0.90   | 0.89     |
| Sci/Tech | 0.89      | 0.91   | 0.90     |


---

## Final Gradio App

The project is deployed as an interactive Gradio application for easy testing. Users can enter a news headline and get live classification results with confidence scores and visual probability bars.


---

## Lessons Learned

- Importance of pre-trained embeddings in improving classification accuracy with limited data.
- BiLSTM can capture meaningful patterns even in short text segments like headlines.
- Fine-tuning vocabulary size and padding length is key for performance and efficiency.

---

## Future Improvements

- Make GloVe embeddings trainable for domain adaptation (instead of freezing them).
- Experiment with transformer-based encoders.
- Extend to multilingual or cross-lingual news classification.
- Implement active learning for data-efficient training.

---


## Notes
This repository is for personal learning and experimentation with text classification, NLP preprocessing, and model deployment using Gradio.
