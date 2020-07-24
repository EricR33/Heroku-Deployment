
from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('best_model-07-23-2020')

cols = ['milestones', 'relationships', 'Web?', 'Software?', 'Consulting?',
       'Gamesvideo?', 'Ecommerce?', 'Mobile?', 'Enterprise?', 'Cleantech?',
       'Hardware?', 'Biotech?', 'Search?', 'Advertising?', 'Security?',
       'Analytics?', 'Realestate?', 'Tech?', 'funding_rounds_bin1',
       'funding_rounds_bin2', 'funding_rounds_bin3', 'investment_rounds_bin1',
       'investment_rounds_bin2', 'funding_total_usd_bin1',
       'funding_total_usd_bin2', 'funding_total_usd_bin3',
       'funding_total_usd_bin4', 'valuation_amount_bin1',
       'valuation_amount_bin2', 'valuation_amount_bin3',
       'valuation_amount_bin4', 'raised_amount_bin1', 'raised_amount_bin2',
       'raised_amount_bin3', 'raised_amount_bin4', 'CA', 'NY', 'TX', 'MA',
       'WA', 'Series_A_Amount_FE', 'Series_B_Amount_FE',
       'Series_C_Amount_FE']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Your selected company will be a {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)