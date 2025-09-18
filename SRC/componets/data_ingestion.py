#This code defines a data ingestion pipeline for a machine learning project using Python. 

#The script is structured to read a raw dataset, split it into training and testing sets,
#  and save these sets to specific locations for later use.

import os #  for interacting with the operating system.
#OS - This module provides a portable way of using operating system-dependent functionality.
import sys #sys module is used to access system-specific parameters and functions FROM src package

#now we need to manage the exception and logging
from SRC.exception import CustomException
from SRC.logger import logging
import pandas as pd

#
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 
#dataclass is a decorator that automatically generates special methods for classes, such as __init__() and __repr__()

from SRC.componets.data_transformation import DataTransformation #we import the data transformation class
from SRC.componets.data_transformation import DataTransformationConfig #we import the data transformation config class

from SRC.componets.model_trainer import ModelTrainerConfig #we import the model trainer config class
from SRC.componets.model_trainer import ModelTrainer #we import the model trainer class

#The DataIngestionConfig class, decorated with @dataclass, defines default file paths for the raw, training, and testing datasets. 
# Using a dataclass here simplifies the creation and management of configuration objects by automatically generating methods like __init__.
@dataclass
class DataIngestionConfig:
    #The new files will be stored in an "artifacts" directory.
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        #Let´s relate the store, with the clss that split the data
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component") #save this log
        try:
            df=pd.read_csv('notebook\data\stud.csv') #reading the dataset
            logging.info('Read the dataset as dataframe') #save this log
            
            #now create the directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            
            
            #let´s save the raw file, in the config path
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            #let´s split the data
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            #save the data
             #let´s save the news file, in the config path
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path

            )
        #If any error occurs during this process, it is caught and re-raised as a CustomException,
        #  which likely provides more detailed error information for debugging.
        except Exception as e:
            raise CustomException(e,sys)

#entry point check to allow the script to be run directly. 
# This modular design makes the code reusable and easy to 
# integrate into larger machine learning workflows. 
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion() 

    data_transformation=DataTransformation()#here we call the transformation method
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer() 
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


    

