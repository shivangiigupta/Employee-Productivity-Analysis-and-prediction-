from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

filename = 'KWSVC.pkl'
classifier = pickle.load(open(filename,'rb'))
model = pickle.load(open('KWSVC.pkl','rb'))

app = Flask(__name__, template_folder= "templates") #template folder

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict_value():
    try:
        Sex = int(request.form['Sex'])
        Age = int(request.form['Age'])
        EducationLevel = int(request.form['EducationLevel'])
        HireMonth = int(request.form['HireMonth'])
        JobTitle = int(request.form['JobTitle'])
        LengthService = int(request.form['LengthService'])
        Status = int(request.form['Status'])
        TravelRequired = int(request.form['TravelRequired'])
        JobSatisfaction = int(request.form['JobSatisfaction'])
        PerformanceRating = int(request.form['PerformanceRating'])
        EmployeeAbsentismRate = int(request.form['EmployeeAbsentismRate'])
        ProductivityRate = int(request.form['ProductivityRate'])
        NumberofteamChanged = int(request.form['NumberofteamChanged'])
        AnnualRate = int(request.form['AnnualRate'])
        HourlyRate = int(request.form['HourlyRate'])
        

        input_features = [Sex, Age, EducationLevel, HireMonth, JobTitle,
       LengthService, Status, TravelRequired, PerformanceRating,
       JobSatisfaction, EmployeeAbsentismRate, ProductivityRate,
       NumberofteamChanged, AnnualRate, HourlyRate]
        features_value = [np.array(input_features)]
        feature_name = ['Sex', 'Age', 'EducationLevel', 'HireMonth', 'JobTitle',
       'LengthService', 'Status', 'TravelRequired', 'PerformanceRating',
       'JobSatisfaction', 'EmployeeAbsentismRate', 'ProductivityRate',
       'NumberofteamChanged', 'AnnualRate', 'HourlyRate']

        df = pd.DataFrame(features_value, columns=feature_name)
        output = model.predict(df)

        return render_template('index.html', prediction_text='Productivity Prediction: {:.2f}'.format(output[0]))
    except Exception as e:
        return render_template('index.html', prediction_text='Error: {}'.format(e))

    
if __name__ == "__main__":
    app.run(debug=True)
