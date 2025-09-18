import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer #Imputer is used to fill in missing values in the dataset
from sklearn.pipeline import Pipeline #Pipeline is used to chain multiple data transformation steps together
from sklearn.preprocessing import OneHotEncoder,StandardScaler

#let´s manage the exception and logging
from SRC.exception import CustomException
from SRC.logger import logging
import os

from SRC.utils import save_object #we use this function to save the preprocessor object

@dataclass 
#By applying @dataclass above a class definition, we reduce boilerplate code and make this code more readable and maintainable.
#now, instead of manually writing an initializer to assign values to each attribute, the dataclass will handle this.
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") #path to save the preprocessor object

class DataTransformation:
    def __init__(self): 
        self.data_transformation_config=DataTransformationConfig() #creating an instance of the config class

    def get_data_transformer_object(self):
        
       # This function is responsible for data transformation
        
       
        try:
            #here we define writing and reading score as numerical columns and as independent variables, and math score as dependent variable
            numerical_columns = ["writing_score", "reading_score"]  #independent numerical variables
            categorical_columns = [ #independent categorical variables
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            #pipeline for numerical columns and their transformation
            num_pipeline= Pipeline( #pipeline for numerical columns
                steps=[ #let's define the steps
                ("imputer",SimpleImputer(strategy="median")), #filling missing values with median
                ("scaler",StandardScaler()) #standardizing the data

                ]
            )
            #pipeline for categorical columns and their transformation
            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #filling missing values with most frequent value
                ("one_hot_encoder",OneHotEncoder()), #one hot encoding to convert categorical variables into a format that can be provided to ML algorithms
                ("scaler",StandardScaler(with_mean=False)) #standardizing the data
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}") #log the categorical columns
            logging.info(f"Numerical columns: {numerical_columns}") #log the numerical columns

            preprocessor=ColumnTransformer( #applying the transformations to the respective columns
                [
                ("num_pipeline",num_pipeline,numerical_columns), #applying numerical pipeline to numerical columns
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor #returning the preprocessor object
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path): #this function will initiate the data transformation

        try:
            train_df=pd.read_csv(train_path) #reading the train data from data ingestion
            logging.info("Read train and test data completed") #log the completion of reading data
            test_df=pd.read_csv(test_path) #reading the test data from data ingestion
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object() #getting the preprocessor object

            target_column_name="math_score" #dependent variable
            numerical_columns = ["writing_score", "reading_score"] #independent numerical variables

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) #independent variables for train data except math score
            target_feature_train_df=train_df[target_column_name] #dependent variable for train data

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) #independent variables for test data except math score
            target_feature_test_df=test_df[target_column_name]  #dependent variable for test data

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            #transforming the data using preprocessor object in an array format and fit the data
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) #fit and transform the train data
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df) #only transform the test data

            train_arr = np.c_[ #np.c_ is used to concatenate two arrays, in this case, the input features and target feature
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object( #call the save object function to save the preprocessor object in utils.py

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return ( #send train and test array and the path where the preprocessor object is saved
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
