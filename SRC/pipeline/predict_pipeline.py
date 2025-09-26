import sys
import pandas as pd
import os
from SRC.exception import CustomException
from SRC.utils import load_object


class PredictPipeline: #Here we are gonna create a class called PredictPipeline
    def __init__(self): #this is an empty constructor to call this class from outside
        pass

    def predict(self,features): #this function is gonna take the features as input
        try:
            #let´s define the path where our model and preprocessor are saved 
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            print("Before Loading")

            #let´s call the function load_object that we created in utils.py

            model=load_object(file_path=model_path) #loading the model
            preprocessor=load_object(file_path=preprocessor_path) #loading the preprocessor

            print("After Loading")

            data_scaled=preprocessor.transform(features) #applying the preprocessor to the features
            preds=model.predict(data_scaled) #making predictions based on the preprocessed data
            return preds #returning the predictionsD
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData: #in this class we are gonna define the input data that we are gonna get from the user
    def __init__(  self,
                 
                  #define the type of each variable
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

            #initializing the variables
        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self): #this function is gonna convert the input data to a dataframe
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict) #returning the dataframe

        except Exception as e:
            raise CustomException(e, sys)

