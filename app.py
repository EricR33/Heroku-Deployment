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

INDUSTRY_VERITCAL_FORM_MAP = {
    "$1": cols.index("Web?"),
    "$2": cols.index("Software?"),
    "$3": cols.index("Consulting?"), 
    "$4": cols.index("Gamesvideo?"),
    "$5": cols.index("Ecommerce?"),
    "$6": cols.index("Mobile?"),
    "$7": cols.index("Enterprise?"),
    "$8": cols.index("Cleantech?"),
    "$9": cols.index("Hardware?"),
    "$10": cols.index("Biotech?"),
    "$11": cols.index("Search?"),
    "$12": cols.index("Advertising?"),
    "$13": cols.index("Security?"),
    "$14": cols.index("Analytics?"),
    "$15": cols.index("Realestate?"),
    "$16": cols.index("Tech?")
}

FUNDING_ROUNDS_FORM_MAP = {
    "$1": cols.index("funding_rounds_bin1"),
    "$2": cols.index("funding_rounds_bin2"),
    "$3": cols.index("funding_rounds_bin3")
}


INVESTMENT_ROUNDS_FORM_MAP = {
    "$1": cols.index("investment_rounds_bin1"),
    "$2": cols.index("investment_rounds_bin2"),
}

FUNDING_TOTAL_FORM_MAP = {
    "$1": cols.index("funding_total_usd_bin1"),
    "$2": cols.index("funding_total_usd_bin2"),
    "$3": cols.index("funding_total_usd_bin3"),
    "$4": cols.index("funding_total_usd_bin4"),
}

VALUATION_AMOUNT_FORM_MAP = {
    "$1": cols.index("valuation_amount_bin1"),
    "$2": cols.index("valuation_amount_bin2"),
    "$3": cols.index("valuation_amount_bin3"),
    "$4": cols.index("valuation_amount_bin4"),
}

RAISED_AMOUNT_FORM_MAP = {
    "$1": cols.index("raised_amount_bin1"),
    "$2": cols.index("raised_amount_bin2"),
    "$3": cols.index("raised_amount_bin3"),
    "$4": cols.index("raised_amount_bin4"),
}

OFFICE_LOCATION_FORM_MAP = {
    "$1": cols.index("CA"),
    "$2": cols.index("NY"),
    "$3": cols.index("TX"),
    "$4": cols.index("MA"),
    "$5": cols.index("WA")
}



FORM_MAP = {
    "raised_amount": RAISED_AMOUNT_FORM_MAP,
    "office_location": OFFICE_LOCATION_FORM_MAP,
    "industry_vertical": INDUSTRY_VERITCAL_FORM_MAP,
    "funding_rounds": FUNDING_ROUNDS_FORM_MAP,
    "investment_rounds": INVESTMENT_ROUNDS_FORM_MAP,
    "funding_total": FUNDING_TOTAL_FORM_MAP,
    "valuation_amount":  VALUATION_AMOUNT_FORM_MAP,
}


def set_raised_amount_features(value, int_features):
    int_features[RAISED_AMOUNT_FORM_MAP[value]] = 1
    return int_features

def set_office_location(value, int_features):
    int_features[OFFICE_LOCATION_FORM_MAP[value]] = 1
    return int_features

def set_industry_vertical(value, int_features):
    int_features[INDUSTRY_VERITCAL_FORM_MAP[value]] = 1
    return int_features

def set_funding_rounds(value, int_features):
    int_features[FUNDING_ROUNDS_FORM_MAP[value]] = 1
    return int_features

def set_investment_rounds(value, int_features):
    int_features[INVESTMENT_ROUNDS_FORM_MAP[value]] = 1
    return int_features

def set_funding_total(value, int_features):
    int_features[FUNDING_TOTAL_FORM_MAP[value]] = 1
    return int_features

def set_valuation_amount(value, int_features):
    int_features[VALUATION_AMOUNT_FORM_MAP[value]] = 1
    return int_features


def set_milestones(value, int_features):
    int_features[cols.index("milestones")] = value
    return int_features

def set_relationships(value, int_features):
    int_features[cols.index("relationships")] = value
    return int_features

def set_series_a_amount(value, int_features):
    int_features[cols.index("Series_A_Amount_FE")] = value
    return int_features

def set_series_b_amount(value, int_features):
    int_features[cols.index("Series_B_Amount_FE")] = value
    return int_features

def set_series_c_amount(value, int_features):
    int_features[cols.index("Series_C_Amount_FE")] = value
    return int_features


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [0 for x in range(len(cols))]
    

    int_features = set_relationships(request.form["relationships"], int_features)
    int_features = set_milestones(request.form["milestones"], int_features)
    int_features = set_office_location(request.form["office_location"], int_features)
    int_features = set_raised_amount_features(request.form["raised_amount"], int_features)
    int_features = set_industry_vertical(request.form["industry_vertical"], int_features) 
    int_features = set_funding_rounds(request.form["funding_rounds"], int_features)  
    int_features = set_investment_rounds(request.form["investment_rounds"], int_features)  
    int_features = set_funding_total(request.form["funding_total"], int_features)
    int_features = set_valuation_amount(request.form["valuation_amount"], int_features)  
    int_features = set_series_a_amount(request.form["Series_A_Amount_FE"], int_features)
    int_features = set_series_b_amount(request.form["Series_B_Amount_FE"], int_features)
    int_features = set_series_c_amount(request.form["Series_C_Amount_FE"], int_features)

    final = np.array(int_features).astype(int)
    print(final)
    #final = [0 for x in range(len(cols))]
    #print(final)
    data_unseen = pd.DataFrame([final], columns = cols)
    print(data_unseen.head())
    prediction = predict_model(model, data=data_unseen)
    prediction = int(prediction.Label[0])
    return render_template('home.html',pred='Your selected company will reach an IPO/Acquisition (1 = Yes, 0 = No) {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
