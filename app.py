from flask import Flask,request,render_template # Importing necessary modules from Flask
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler # Importing StandardScaler from sklearn
from SRC.pipeline.predict_pipeline import CustomData, PredictPipeline 
#here we are gonna import from SRC folder the CustomData and PredictPipeline classes

application=Flask(__name__) # Initializing a Flask application

app=application # Assigning the Flask application to a variable 'app'

## Route for a home page

@app.route('/') 
def index():
    return render_template('index.html')  # Rendering the index.html template for the home page

@app.route('/predictdata',methods=['GET','POST']) # Defining a route for handling data prediction

def predict_datapoint():
    if request.method=='GET': # If the request method is GET, render the home.html template
        return render_template('home.html')
    else:
        data=CustomData( # Creating an instance of CustomData with form data
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )

        pred_df=data.get_data_as_data_frame() # Converting the input data to a DataFrame
        print(pred_df) #here we are gonna print the dataframe
        print("Before Prediction")

        predict_pipeline=PredictPipeline() 
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0]) # Rendering the home.html template with prediction results
    

if __name__=="__main__":
    app.run(host="0.0.0.0")  # Running the Flask application on all available IP addresses