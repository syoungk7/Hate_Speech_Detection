import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask, render_template, request
from tensorflow import keras

# apply flask
app=Flask(__name__)

# home
@app.route('/')

# index
@app.route('/index')
def index():
    return flask.render_template('index.html')

# Function to load and use the LSTM model
def lstm_predict(tweet):
    # Load the saved LSTM model
    lstm_model = keras.models.load_model('model_LSTM.h5')

    # Tokenize new tweet and pad sequences
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(tweet)
    sequences = tokenizer.texts_to_sequences(tweet)
    max_len = 30
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

    # Make predictions
    lstm_predictions = lstm_model.predict(padded_sequences)
    lstm_predicted_class = np.argmax(lstm_predictions)

    return lstm_predicted_class

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
        print(data)
        tweet = [data]
        print(tweet)

        # Use LSTM model for prediction
        lstm_prediction = lstm_predict(tweet)

        # Use RF model for prediction
        rf_prediction = rf_predict(tweet)

        # Determine final prediction based on both models
        if rf_prediction == 1:
            final_prediction = 'Hate speech Detected (RF Model).'
        elif lstm_prediction == 1:
            final_prediction = 'Hate speech Detected (LSTM Model).'
        else:
            final_prediction = 'Not a Hate speech.'

        return render_template("result.html", prediction=final_prediction)

if __name__ == "__main__":
    app.run(debug=True)