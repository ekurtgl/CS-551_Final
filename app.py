import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")


@app.route("/")
def loadPage():
    return render_template('home.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']

    model = pickle.load(open("knn.sav", "rb"))

    data = [[float(inputQuery1), float(inputQuery2), float(inputQuery3), float(inputQuery4), float(inputQuery5)]]
    new_df = pd.DataFrame(data, columns=['concavity_worst', 'smoothness_se', 'texture_mean', 'concave points_worst',
                                         'compactness_mean'])
    new_df['concavity_worst'] /= 1.252
    new_df['smoothness_se'] /= 0.03113
    new_df['texture_mean'] /= 39.28
    new_df['concave points_worst'] /= 0.291
    new_df['compactness_mean'] /= 0.3454

    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:, 1][0]

    if single == 1:
        o1 = "The patient is diagnosed with Breast Cancer"
        o2 = "Confidence: {}".format(probability * 100)
    else:
        o1 = "The patient is not diagnosed with Breast Cancer"
        o2 = "Confidence: {}".format(100 - (probability * 100))

    return render_template('home.html', output1=o1, output2=o2, query1=request.form['query1'],
                           query2=request.form['query2'], query3=request.form['query3'], query4=request.form['query4'],
                           query5=request.form['query5'])


if __name__ == "__main__":
    app.run()

