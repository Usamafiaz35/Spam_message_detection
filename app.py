from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the saved model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
rfc = pickle.load(open('model.pkl', 'rb'))

# Initialize NLTK tools
nltk.download('stopwords')
nltk.download('punkt')

# Text Preprocessing Function
def transform_text(text):
    # Lowercasing
    text = text.lower()
    
    # Remove special characters (keeping only alphabets and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Convert tokens back to text
    return ' '.join(tokens)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('message')
    
    if not text:
        return jsonify({'error': 'No input text provided'})
    
    # Apply text preprocessing
    cleaned_text = transform_text(text)
    
    # Transform text using TF-IDF vectorizer
    transformed_text = tfidf.transform([cleaned_text]).toarray()
    
    # Predict using model
    prediction = rfc.predict(transformed_text)[0]
    
    # Convert prediction to label
    result = 'Spam' if prediction == 1 else 'Ham'
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
