# Spam and Fraud Message/Call Detector

## 1. Problem Statement & Project Objective

In today's interconnected world, unsolicited and malicious communications like spam SMS and fraudulent calls pose significant threats. These can range from annoying promotional messages to sophisticated phishing attempts, leading to financial loss, identity theft, and a general erosion of trust in digital communication channels.

The primary objective of this project is to develop a robust machine learning system capable of classifying incoming messages and (conceptually) call transcripts as either **"Normal"** or **"Fraud/Spam."** This system aims to empower users with an early warning mechanism, helping them identify and avoid potentially harmful communications. While the immediate implementation focuses on text-based messages, the underlying goal extends to the broader challenge of identifying fraudulent communication across different modalities.

## 2. Data

This project utilizes a combination of two publicly available datasets to train a comprehensive classification model:

*   **SMS Spam Collection Dataset**
    *   **Description:** A collection of SMS messages, each tagged as either 'spam' or 'ham' (legitimate).
    *   **Source:** [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
    *   **Usage:** Provides a rich source of typical SMS spam patterns and legitimate conversational language.

*   **Fraud Call India Dataset**
    *   **Description:** This dataset aims to classify call transcripts as either 'fraud' or 'normal'. It helps in identifying fraudulent conversations.
    *   **Source:** [https://www.kaggle.com/datasets/narayanyadav/fraud-call-india-dataset](https://www.kaggle.com/datasets/narayanyadav/fraud-call-india-dataset)
    *   **Usage:** Crucial for understanding the linguistic patterns indicative of fraud in a call context, which is then applied to text.

## 3. Tools & Techniques

This project leverages a powerful stack of Python libraries and machine learning techniques:

*   **Python 3.x:** The core programming language.
*   **Pandas:** Essential for data loading, manipulation, and cleaning of the datasets.
*   **NLTK (Natural Language Toolkit):** Used for advanced text preprocessing, including:
    *   Stop word removal
    *   Lemmatization (reducing words to their base form)
    *   Tokenization
*   **Scikit-learn:** The primary machine learning library for:
    *   `TfidfVectorizer`: Transforms raw text data into numerical feature vectors by weighting word importance.
    *   `LinearSVC (Support Vector Classifier)`: A robust and efficient classification algorithm, particularly effective for text classification tasks.
    *   `train_test_split`: For splitting data into training and testing sets.
    *   `metrics`: For evaluating model performance (accuracy, precision, recall, F1-score).
*   **Joblib:** Used for efficient serialization (saving) and deserialization (loading) of trained Python objects (the TF-IDF vectorizer and the classifier model).
*   **Streamlit:** An open-source app framework for rapidly building and deploying interactive web applications, providing a user-friendly GUI for real-time predictions.

## 4. Discussion of the Process

The project follows a standard machine learning pipeline:

1.  **Data Ingestion:** Both `spam.csv` and `fraud_call.file` datasets are loaded using Pandas. Careful handling of different file formats (CSV with internal commas, TSV with potential malformed lines) and the absence/presence of headers was crucial.
2.  **Data Preprocessing:**
    *   Text is converted to lowercase.
    *   Punctuation and numbers are removed to reduce noise.
    *   Common English stop words (e.g., "the", "a", "is") are eliminated.
    *   Words are lemmatized to their base forms (e.g., "running" -> "run") to consolidate features.
    *   Labels are standardized (`ham`/`spam` mapped to `normal`/`fraud`).
3.  **Feature Engineering:** The preprocessed text messages are transformed into numerical feature vectors using `TfidfVectorizer`. This technique assigns scores to words based on their frequency in a document and rarity across the entire dataset, effectively capturing the importance of terms.
4.  **Model Training:** The vectorized data is split into training and testing sets. A `LinearSVC` classifier is then trained on the training data.
5.  **Model Evaluation:** The trained model's performance is assessed on the unseen test set using metrics like accuracy, precision, recall, and F1-score, providing insights into its effectiveness.
    ```
    Training LinearSVC Classifier...
    Classifier training complete!

    --- Model Evaluation ---
    Accuracy: 0.9821
    Classification Report:
                   precision    recall  f1-score   support

           fraud       1.00      0.87      0.93       149
          normal       0.98      1.00      0.99       966

        accuracy                           0.98      1115
       macro avg       0.99      0.93      0.96      1115
    weighted avg       0.98      0.98      0.98      1115
    ```
6.  **Model Persistence:** The trained `TfidfVectorizer` and `LinearSVC` model are saved as `.pkl` files using `joblib`. This allows the Streamlit application to load the pre-trained model for predictions without needing to retrain it every time.
7.  **Web Application Development:** A Streamlit application (`app.py`) is built to provide an interactive GUI. Users can input a message, which is then preprocessed, vectorized, and fed into the loaded model to generate a "Normal" or "Fraud/Spam" prediction.

![WhatsApp Image 2025-10-08 at 17 38 39_127c3331](https://github.com/user-attachments/assets/428ee670-9ee8-40c2-8659-3a17d500e998)
![WhatsApp Image 2025-10-08 at 17 39 40_0317f36e](https://github.com/user-attachments/assets/df105aad-d6d4-4eee-936e-a2313a7f25ff)


## 5. Key Insights

*   **Linguistic Patterns of Fraud:** Fraudulent and spam messages often exhibit distinct linguistic patterns:
    *   **Urgency:** Phrases like "Act now!", "Limited time offer!", "Your account will be suspended!"
    *   **Financial Incentives:** "Won a prize!", "Cash reward!", "Free money!"
    *   **Demands for Action:** "Call this number," "Click this link," "Verify your details."
    *   **Grammatical Deviations/Typos:** Sometimes present, though more sophisticated scams are grammatically sound.
*   **Preprocessing is Paramount:** Effective text preprocessing (cleaning, tokenization, lemmatization) is critical for feature engineering, significantly impacting model accuracy.
*   **TF-IDF Effectiveness:** TF-IDF proves to be a highly effective technique for converting text into meaningful numerical features that capture the discriminative power of words in spam/fraud detection.
*   **LinearSVC Robustness:** `LinearSVC` demonstrates strong performance in distinguishing between normal and fraudulent/spam messages, making it a suitable choice for this classification task.

## 6. Business Impact (Leveraging Airtel's Approach)

The challenge of combating spam and fraud is immense, as highlighted by industry leaders like Airtel. Airtel's multi-layered approach, combining AI-powered detection with user-controlled blocking, serves as a real-world example of the business impact of such systems.

*   **Enhanced Customer Protection:** By automatically flagging "Suspected Spam" (as Airtel does) or explicitly classifying messages, users are protected from financial scams, privacy breaches, and unsolicited content.
*   **Improved User Experience:** Reduced spam means less annoyance, fewer distractions, and a more trustworthy communication environment.
*   **Operational Efficiency:** Automated detection reduces the burden on customer service teams dealing with spam complaints.
*   **Real-world Scale:** Airtel's system analyzes over 250 parameters per call in ~2 milliseconds with 97% accuracy for spam calls and 99.5% for spam SMS during testing (Source: Airtel's public statements on their anti-spam initiatives). This underscores the necessity and feasibility of highly accurate, real-time detection.
*   **Adaptability:** The ability to provide alerts in vernacular languages (Airtel's initiative) and screen international spam showcases the need for adaptable and comprehensive solutions.

## 7. Challenges & Learnings

### General Challenges & Learnings:
*   **Data Inconsistency:** Dealing with real-world datasets often involves inconsistent formatting (e.g., CSVs without proper quoting, varying delimiters), requiring careful data loading and preprocessing.
*   **Evolving Threat Landscape:** Spam and fraud tactics constantly evolve, necessitating continuous model retraining with new data to maintain effectiveness.
*   **False Positives/Negatives:** Balancing the trade-off between mistakenly flagging a legitimate message (false positive) and missing a fraudulent one (false negative) is crucial. A system like Airtel's avoids blocking calls directly to prevent legitimate business calls from being flagged.

### Specific Challenge: Call vs. SMS - The Interception Dilemma

This project highlights a significant limitation when extending text-based fraud detection to real-time voice calls: **intercepting and analyzing a live, unclassified call for fraud without prior warning or user action (like in Truecaller) is technically and legally complex for local host applications.**

Airtel, operating at the network level, can analyze call metadata and patterns before the call even reaches the user, displaying "Suspected Spam" on the caller ID. This network-level access is fundamentally different from what a local application can achieve.

*   **Our Current Limitation:** Our model, as implemented, works on *text data* (either from SMS or a transcribed call). It cannot directly "listen" to a live phone call in real-time on a local machine (like your PC or even a typical phone app) and intercept it to perform analysis before it connects or alerts the user. This is due to operating system security restrictions, privacy laws, and the technical challenge of integrating with real-time telephony systems at a low level.
*   **Contrast with Airtel/Truecaller:** Solutions like Airtel's or Truecaller's (which uses community-sourced data and network-level insights) operate with different levels of access and data streams. Truecaller, for instance, identifies many spam calls *after* they are reported by users or by matching numbers against a large database.

## 8. Future Work

*   **Deployment as a Call Transcriber and Analyzer (Local Host):**
    *   Explore APIs or libraries that can transcribe live audio from a phone call (if technically feasible and legally permissible on a local machine, e.g., via microphone input during a speakerphone call).
    *   Integrate the trained text classification model to analyze these real-time transcriptions.
    *   Provide a *local host only* alert system for the user based on the transcribed text, acknowledging the limitations of direct call interception.
*   **Deep Learning Models:** Experiment with more advanced NLP models like Recurrent Neural Networks (RNNs), LSTMs, or Transformer-based models (e.g., BERT) for potentially higher accuracy and better contextual understanding.
*   **Real-time Monitoring & Feedback Loop:** Develop a mechanism for continuous model improvement by collecting user feedback on predictions and retraining the model periodically.
*   **Explainable AI (XAI):** Implement techniques (e.g., LIME, SHAP) to explain *why* the model classified a message as spam/fraud, highlighting key phrases or words.
*   **Specific Fraud Category Detection:** Expand the model to classify into more granular fraud categories (e.g., phishing, lottery scam, tech support scam).
*   **Cloud Deployment:** Deploy the Streamlit application to a cloud platform (e.g., Streamlit Cloud, AWS, GCP, Azure) for broader accessibility.
*   **Integration with Messaging APIs:** For SMS, investigate integration with messaging APIs (e.g., Twilio) for a more automated SMS filtering system (acknowledging costs and platform restrictions).

---
