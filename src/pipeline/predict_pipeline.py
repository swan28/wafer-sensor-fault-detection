import shutil
import os,sys
import pickle
import sys
import pandas as pd

from flask import request
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.constant import *
from src.utils.main_utils import MainUtils


@dataclass
class PredictionPipelineConfig:
    """
        `PredictionPipelineConfig` - Its the dataclass defines the configuration for the prediction pipeline, including paths for models, preprocessors, and prediction outputs.

        Attributes:
            - `prediction_output_dirname`: Directory where the prediction results will be saved.
            - `prediction_file_name`: Name of the file where the preictions will be stored.
            - `model_file_path`: Path to the serialized ML model (model.pkl).
            - `preprocessor_path`: Path to the preprocessor used for the data transformation.
            - `prediction_file_path`: Path where the prediction file will be saved.
    """
    prediction_output_dirname:str = "predictions"
    prediction_file_name:str =  "prediction_file.csv"
    model_file_path:str = os.path.join(artifact_folder, "model.pkl" )
    preprocessor_path:str = os.path.join(artifact_folder, "preprocessor.pkl")
    prediction_file_path:str = os.path.join(prediction_output_dirname, prediction_file_name)


class PredictionPipeline:
    """
        `PredictionPipeline` - Its the main class that handles the prediction process.

        Attributes:
            - `request`: The incomming request containing the input data (likely an uploaded CSV file).
            - `utils`: Instance of `MainUtils` for loading models, preprocessors, and other utility functions.
            - `prediction_pipeline_config`: Configuration object containing paths for files and directories.
    """
    def __init__(self, request: request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_pipeline_config = PredictionPipelineConfig()

    def save_input_files(self)-> str:
        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 

            Output      :   Input dataframe
            On Failure  :   Write an exception log and then raise an exception

            Steps:
                - Create a directory called `prediction_artifacts` if it doesn't exists.
                - Extract the uploaded file from the request and save it to the directory.
        """
        try:
            #creating the file
            pred_file_input_dir = "prediction_artifacts"
            os.makedirs(pred_file_input_dir, exist_ok=True)
            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)
            input_csv_file.save(pred_file_path)

            # Returns the path to the saved input CSV file
            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, features):
            """
                Method Name :  predict() 
                Description :  This method uses the pre-trained model and preprocessor to make predictions on th einput features.

                Steps       
                    - Load the trained model and preprocessor using `MainUtils.load_object()`
                    - Apply the preprocessor to transform the input features.
                    - Use the model to predict the output based on the transformed features.
            """
            try:
                model = self.utils.load_object(self.prediction_pipeline_config.model_file_path)
                preprocessor = self.utils.load_object(
                    file_path=self.prediction_pipeline_config.preprocessor_path
                    ) 
                transformed_x = preprocessor.transform(features)
                preds = model.predict(transformed_x)
                
                # Returns the predicted values
                return preds
            except Exception as e:
                raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:pd.DataFrame):
        """
            Method Name :   get_predicted_dataframe
            Description :   Reads the input CSV file, makes predictions and adds a new column for the predictions

            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception

            Steps:
                - Read the input data from the CSV file.
                - Drop any unwanted columns if any.
                - Call the `predict()` method to gets the predictions for the input data.
                - Map the predictions values(0/1) to human readable labels(Bad/Good).
                - Save the resulting DataFrame with the predictions to a CSV file
        """
        try:

            prediction_column_name : str = TARGET_COLUMN
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            input_dataframe =  input_dataframe.drop(columns="Unnamed: 0") if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'bad', 1:'good'}
            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping) 
            os.makedirs(self.prediction_pipeline_config.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_pipeline_config.prediction_file_path, index= False)
            logging.info("predictions completed. ")
        except Exception as e:
            raise CustomException(e, sys) from e
        

        
    def run_pipeline(self):
        """
            Method       :  run_pipeline()
            Description  :  Orchestrates the prediction process by running the entire pipeline.

            Steps: 
                - Calls save_input_files() to save the input file.
                - Calls get_predicted_dataframe() to generate predictions and save them to a file.
                - Returns the configuration object, which contains the file_paths used in the prediction process.
        """
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            # Returns the configuration object, which contains the file_paths used in the prediction process.
            return self.prediction_pipeline_config
        except Exception as e:
            raise CustomException(e,sys)
            
        

 
        

        