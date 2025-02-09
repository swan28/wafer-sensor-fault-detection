import sys
import os
import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    """
        `DataTransformationConfig` is a `dataclass` which defines the configuration for data transformation, including paths where transformed data and the preprocessor object will be saved
    """
    # Directory where all artifacts will be stored
    artifact_dir=os.path.join(artifact_folder)

    # Filepath for the transformed training data (in NumPy format thus .npy)
    transformed_train_file_path=os.path.join(artifact_dir, 'train.npy')

    # Filepath for the transformed test data (in NumPy format)
    transformed_test_file_path=os.path.join(artifact_dir, 'test.npy') 

    # Filepath for saving the preprocessor object(which contains scaling and imputation steps)
    transformed_object_file_path=os.path.join( artifact_dir, 'preprocessor.pkl' )


class DataTransformation:
    """
        `DataTransformation` is the main class responsible for performing data transformation
    """
    def __init__(self, feature_store_file_path):
        # Path to feature store file(the CSV file to be processed)
        self.feature_store_file_path = feature_store_file_path

        # Instance for DataTransformationConfig that contains path for saving transformed data
        self.data_transformation_config = DataTransformationConfig()

        # Instance of MainUtils, which provides utility functions like saving objects(eg. preprocessor)
        self.utils =  MainUtils()


    # Reads the raw data from a CSV file and returns it as a Pandas DataFrame
    @staticmethod
    def get_data(feature_store_file_path:str)->pd.DataFrame:
        """
            Method Name :   `get_data`
            Description :   This method reads all the validated raw data from the `feature_store_file_path` and returns a pandas DataFrame containing the merged data. 
        
            Output      :   a pandas DataFrame containing the merged data 
            On Failure  :   Write an exception log and then raise an exception
        
        """
        try:
            # Reads the CSV file from the given feature_store_file_path
            data = pd.read_csv(feature_store_file_path)

            # Rename the column Good/Bad to TARGET_COLUMN(ie.quality)
            data.rename(columns={"Good/Bad": TARGET_COLUMN}, inplace=True)

            # Returns a pandas DF containing the data
            return data
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_data_transformer_object(self):
        """
            `get_data_transformer_object` method defines the data transformation steps to be applied including:

            1. Imputation: Fills the missing values with 0 using SimpleImputer
            2. Scaling: Scales features using RobustScaler to reduce the impact of outliers
        """
        try:
            # Define the steps for the preprocessor pipeline
            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            # Pipeline
            preprocessor = Pipeline(
                steps=[
                    imputer_step, 
                    scaler_step
                    ]
                )
            
            # Returns a preprocessor object, which is a Pipeline containing the imputation and scaling steps
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self) :
        """
            Method Name :   `initiate_data_transformation`
            Description :   This method initiates the data transformation component for the pipeline 
            
            Output      :   data transformation artifact is created and returned 
            On Failure  :   Write an exception log and then raise an exception
        """

        logging.info("Entered initiate_data_transformation method of Data_Transformation class")

        try:
            # Read the Data: Calls get_data() to load the raw data from the CSV file
            dataframe = self.get_data(feature_store_file_path=self.feature_store_file_path)

            # Feature and Target Split
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN]==-1, 0, 1)  #replacing the -1 with 0 for model training
 
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Preprocessing 
            preprocessor = self.get_data_transformer_object()

            # Training data: Fits and Transforms the training data using the preprocessor pipeline(Imputation and Scaling)
            X_train_scaled =  preprocessor.fit_transform(X_train)

            # Test data: Applies the same transformation to the data w/o fitting again
            X_test_scaled  =  preprocessor.transform(X_test)

            # Save Preprocessor: Saves the preprocessor pipeline object to a file for future use(eg. during model inference)
            preprocessor_path = self.data_transformation_config.transformed_object_file_path
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

            # Save Transformed Data
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

            # Combines the scaled features and target values into Numpy arrays using np.c_[] concatenates arrays along second axis(ie., column)
            # np.r_[] concatenates arrays along first axis(ie., row).
            train_arr = np.c_[X_train_scaled, np.array(y_train)] 
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Returns the training and test arrays along with the preprocessor file path
            return (train_arr, test_arr, preprocessor_path)
        except Exception as e:  
            raise CustomException(e, sys) from e