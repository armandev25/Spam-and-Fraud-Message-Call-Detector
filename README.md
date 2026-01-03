# Spam and Fraud Message / Call Detector

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-yellow)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20SVM-green)
![Text Classification](https://img.shields.io/badge/Text%20Classification-Spam%20%26%20Fraud-red)
![Status](https://img.shields.io/badge/Status-Completed-success)


This project implements a **machine learning–based spam and fraud detection system** that classifies incoming messages as either **Normal** or **Fraud/Spam**.  
The goal is to help users identify potentially harmful communications such as phishing messages, scam SMS, and fraudulent call transcripts.

While the current implementation focuses on text-based messages, the project is designed with the broader objective of detecting fraudulent communication patterns across different modalities.

---

## Problem Statement

In today’s digital ecosystem, spam and fraud messages have evolved beyond simple promotional texts into sophisticated phishing and social engineering attacks. These messages often create urgency, promise financial rewards, or demand immediate action, leading to financial loss and identity theft.

The objective of this project is to build a **reliable early-warning system** that can analyze message content and flag suspicious communication before a user interacts with it.

---

## Approach Used

This project frames spam and fraud detection as a **text classification problem** using supervised machine learning.

Key ideas:
- Fraud and spam messages exhibit **distinct linguistic patterns**
- These patterns can be learned from historical data
- Well-engineered text features are often more important than complex models

---

## Dataset

This project uses a combination of two publicly available datasets:

### 1. SMS Spam Collection Dataset
- Labeled SMS messages (`spam` / `ham`)
- Captures common spam structures and legitimate conversational language  
- Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

### 2. Fraud Call India Dataset
- Contains call transcripts labeled as `fraud` or `normal`
- Helps capture fraud-specific language used in scam calls  
- Source: https://www.kaggle.com/datasets/narayanyadav/fraud-call-india-dataset

The two datasets are combined and standardized to create a unified **Normal vs Fraud/Spam** classification task.

---

## Text Preprocessing

Before training the model, the text data is cleaned and normalized:

- Conversion to lowercase
- Removal of punctuation and numbers
- Stop-word removal
- Tokenization
- Lemmatization to reduce words to their base form
- Label standardization (`ham → normal`, `spam → fraud`)

This step significantly improves feature quality and model performance.

---

## Feature Engineering

Text data is converted into numerical form using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

Why TF-IDF:
- Captures word importance rather than raw frequency
- Reduces the influence of common but uninformative words
- Works well for sparse, high-dimensional text data

---

## Model Used

- **Linear Support Vector Classifier (LinearSVC)**
- Chosen for:
  - Strong performance on text classification
  - Efficiency on large sparse feature spaces
  - Robust decision boundaries

The dataset is split into training and testing sets to evaluate generalization.

---

## Model Performance

The trained model achieves strong results on unseen data:

- Accuracy: ~98%
- High precision for fraud detection
- Near-perfect recall for normal messages

This indicates the model is effective at identifying fraud while minimizing false alarms.

---

## Web Application

The trained model and TF-IDF vectorizer are saved and integrated into a **Streamlit web application**.

Users can:
- Enter a message
- Analyze it in real time
- Receive a clear classification result with a warning or confirmation

### Application Interface

#### Screenshot 1: Fraud/Spam Detection Example
<img width="1280" height="555" alt="image" src="https://github.com/user-attachments/assets/fa965920-dfa0-40c9-add8-ec67823194d4" />


#### Screenshot 2: Normal Message Detection Example
<img width="1280" height="564" alt="image" src="https://github.com/user-attachments/assets/1375177d-2ad3-4e17-a899-0927b783feef" />


---

## Key Insights

- Fraud and spam messages often contain urgency-driven language and financial诱 incentives
- Proper text preprocessing has a major impact on model accuracy
- TF-IDF is highly effective for spam detection tasks
- Linear models can outperform complex models when features are well engineered

---

## Limitations and Learnings

- The system relies on text input and cannot intercept live calls
- Real-time call interception is restricted by OS security and privacy laws
- Fraud patterns evolve, requiring continuous retraining
- Balancing false positives and false negatives is critical in real deployments

This project highlights the difference between **local ML applications** and **network-level solutions** used by telecom providers.

---

## Business Relevance

Telecom companies such as Airtel use AI-based systems to flag spam and fraud at scale.  
This project mirrors the **core logic** of such systems at a local application level, demonstrating how machine learning can be applied to protect users and improve trust in communication platforms.

---

## Future Work

- Integration with speech-to-text for call transcription analysis
- Experimentation with deep learning models (LSTM, BERT)
- Explainable AI techniques to highlight fraud indicators
- More granular fraud category classification
- Cloud deployment for wider accessibility

---

## Summary

This project demonstrates a practical, end-to-end implementation of **spam and fraud detection using NLP and machine learning**.  
It emphasizes clean data processing, effective feature engineering, and realistic system constraints, making it relevant for real-world applications and further extensions.



