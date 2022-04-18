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

tf.keras.backend.clear_session()
app = Flask(__name__)
model=load_model(r"C:\Users\User\OneDrive\Desktop\ibm\Amazon-Kindle-Store-Reviews-Analysis-using-IBM-Watson-Services\Amazon kindle\amazonibm.h5")
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
    loaded=CountVectorizer(decode_error='replace',vocabulary=joblib.load(r"C:\Users\User\OneDrive\Desktop\ibm\Amazon-Kindle-Store-Reviews-Analysis-using-IBM-Watson-Services\Amazon kindle\ibmamazon.save"))
    d=d.split("delimiter")
    result=model.predict(loaded.transform(d))
    print(result)
    prediction=result>0.5
  
    if prediction[0] == False:
    	output="Positive review"
    elif prediction[0] == True:
    	output="Negative review"
    return render_template('index.html', prediction_text='{}'.format(output)) 


if __name__ == "__main__":
    app.run(debug=False)
    