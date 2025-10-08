import streamlit as st
import joblib
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already downloaded
# This is crucial for Streamlit deployment as well
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# --- Text Preprocessing Function (must match the one used in train_model.py) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower() # Lowercasing
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = ' '.join([word for word in text.split() if word not in stop_words]) # Remove stopwords
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) # Lemmatization
    return text

# --- Load Model and Vectorizer ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    model_path = 'models/text_classifier_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or Vectorizer not found! Please run `Train_Spam_Fraud_Model.ipynb` or `train_model.py` first.")
        return None, None
    
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None

vectorizer, model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Spam & Fraud Detector", layout="centered")

st.title("✉️ Spam & Fraud Message Detector")
st.markdown("""
    Enter a message below to check if it's likely a normal message or a spam/fraud attempt.
    This detector uses a trained machine learning model based on various spam and fraud datasets.
""")

if vectorizer is None or model is None:
    st.warning("Model components could not be loaded. Please ensure the training notebook has been run successfully.")
else:
    user_input = st.text_area("Enter your message here:", height=150, placeholder="e.g., Congratulations! You've won a cash prize! Claim now by calling...")

    if st.button("Analyze Message"):
        if user_input:
            with st.spinner("Analyzing..."):
                processed_input = preprocess_text(user_input)
                
                # Transform the input using the loaded vectorizer
                input_vectorized = vectorizer.transform([processed_input])
                
                # Make prediction
                prediction = model.predict(input_vectorized)
                
                st.subheader("Analysis Result:")
                if prediction[0] == 'fraud':
                    st.error(f"**🔴 Fraud/Spam Alert!** This message is highly likely to be **{prediction[0].upper()}**.")
                    st.write("Exercise caution! Do not click on suspicious links or provide personal information.")
                else:
                    st.success(f"**🟢 Normal Message.** This message appears to be **{prediction[0].upper()}**.")
                    st.write("However, always remain vigilant! If something feels off, trust your instincts.")
        else:
            st.warning("Please enter a message to analyze.")

st.markdown("---")
st.markdown("Developed for a resume project.")