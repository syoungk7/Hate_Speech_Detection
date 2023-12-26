import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask, render_template, request

# apply flask
app=Flask(__name__)

# home
@app.route('/')

# index
@app.route('/index')
def index():
    return flask.render_template('index.html')

# Function to load and use the RF model
def rf_predict(tweet):
    rf_model = pickle.load(open("model_RF.pkl", "rb"))
    rf_prediction = rf_model.predict(tweet)[-1]
    return rf_prediction

# Result route
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        data = request.form['tweet']
        tweet = [data]

        # Use RF model for prediction
        rf_prediction = rf_predict(tweet)

        # Determine final prediction based on both models
        if rf_prediction == 1:
            final_prediction = 'RF Model \nHate speech Detected .'
        else:
            final_prediction = 'Not a Hate speech.'

        return render_template("result.html", prediction=final_prediction)

if __name__ == "__main__":
    app.run(debug=True)