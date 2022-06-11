import numpy as np
from flask import Flask, request, render_template
from joblib import load
import joblib
from tensorflow.keras.models import load_model 
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import backend
from gevent.pywsgi import WSGIServer
import os
import pickle 
import re 
import nltk
from nltk.corpus import stopwords # removing all the stop words
from nltk.stem.porter import PorterStemmer # stemming of words
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
#create an object for wordnet lemmatizer
wordnet=WordNetLemmatizer()
tf.keras.backend.clear_session()
app = Flask(__name__)
loaded_model = pickle.load(open("amazon.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/y_predict',methods=['POST'])
def y_predict():
    

    '''
    For rendering results on HTML GUI
    '''
    d = request.form['Sentence']
    print(d)
    new_review = re.sub('[^a-zA-Z]', ' ',d)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in   set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = loaded_model.predict(new_X_test)
    new_review = new_y_pred
    print(new_review)
    if new_review[0]==True:
        print("Positive review")
        output="positive review"
    else :
        print("Negative review")
        output="negative review"
    
    return render_template('index.html', prediction_text='{}'.format(output)) 





if __name__ == "__main__":
    
    app.run(debug=True)
    