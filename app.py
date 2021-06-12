import json
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from utils import all_purpose_transformer, custom_imputer, skewness_remover


app = Flask(__name__)

model = pickle.load(open('stacking_model.pkl', 'rb'))
prep_pipe = pickle.load(open('prep_pipe.pkl', 'rb'))
one_hot_enc = pickle.load(open('one_hot_enc.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))


@app.route('/line/<int:row>')
# Get data from json and return the requested row defined by the variable Line
def line(row):
    with open('test.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())
    # We can then find the data for the requested row and send it back as json
    return json.dumps(file_data[row])



@app.route('/prediction/<int:row>', methods=['POST', 'GET'])
# Return prediction with the requested row as input
def prediction(row):
    data = pd.read_json('test.json')
    df = pd.DataFrame([data.iloc[row]], columns=data.columns)
    df_prep = scaler.transform(one_hot_enc.transform(prep_pipe.transform(df)).toarray())
    prediction = np.expm1(model.predict(df_prep))[0]

    return {'Predicted Item Outlet Sales': prediction}


if __name__ == "__main__":
    app.run(debug=True)



