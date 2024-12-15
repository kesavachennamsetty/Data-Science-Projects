import pandas as pd
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec


# # Function to load and use the model
# def model(wrd_vctrs):
#     # Load the PyTorch ANN model
#     print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------three")
#     ann_model = torch.load('email_ann_model.pth', weights_only=True)
#     print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------four")
#     wrd_vctrs = np.array(wrd_vctrs, dtype=np.float32)  # Ensure input is a NumPy array
#     print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------five")
#     wrd_vctrs_tensor = torch.tensor(wrd_vctrs, dtype=torch.float32)  # Convert to PyTorch tensor
#     print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------six")
#     print(ann_model(wrd_vctrs_tensor))
#     print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------seven")
#     return torch.argmax(result).item()  # Get the predicted class (e.g., 1 or 0)


# # Function to get word embeddings
def word2_vec(sent):
    # Load the pre-trained Word2Vec model
    print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------one")
    word2vec_model = Word2Vec.load('word2vec.model')
    print("---------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>--------two")
    vectors = [word2vec_model.wv[word] for word in sent if word in word2vec_model.wv]
    if len(vectors) == 0:
        return np.zeros(word2vec_model.vector_size)  # Return a zero vector if no words found
    return np.mean(vectors, axis=0)  # Return the average word vector


# Function to preprocess and filter words
def fltr_wrds(text):
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into words
    words = text.split()
    # Get the set of English stopwords
    stop_words = set(stopwords.words('english'))
    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


# # Flask app setup
# app = Flask(__name__)


# # Route for the homepage
# @app.route('/')
# def index_page():
#     return render_template('index.html')  # Ensure index.html exists in the templates folder


# # Route to handle email spam prediction
# @app.route('/submit', methods=['POST'])
# def submit():
#     ''' Process user input and get model output '''
#     try:
#         data = request.get_json()  # Get JSON data from the request
#         text = data.get('text') 
#         print(text) # Retrieve the email text
#         if not text:
#             return jsonify({'error': 'No text provided'}), 400

#         # Preprocess and filter words
#         filtered_words = fltr_wrds(text)
#         print(filtered_words)
#         # Get word vectors and model output
#         wrd_vctrs = word2_vec(filtered_words)
#         print(wrd_vctrs)
#         model_output = model(wrd_vctrs)
#         print(model_output)
#         # Determine classification
#         clas = 'Spam' if model_output == 1 else 'Not Spam'

#         # Return the classification as JSON
#         return jsonify({'classification': clas})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)
# Define the ANN model architecture
import joblib
import pickle

model = joblib.load('email_logistic_reg.sav')