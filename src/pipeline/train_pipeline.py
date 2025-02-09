import sys,os

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

class TraininingPipeline:
    """
        TraininingPipeline - This class orchestrates the entire ML pipeline by running the components sequentially:
            - Data Ingestion
            - Data Transformation
            - Model Training
    """
    
    def start_data_ingestion(self):
        """
            Initiates the data ingestion process, which is repsonsible for fetching data from a source.(eg. a DB, CSV file)

            Steps:
                - An instance of DataIngestion is created .
                - The initiate_data_ingestion() method of DataIngestion is called, which ingests the data.
                - The path to the feature store file(where the data is stored) is returned. 
        """
        try:
            data_ingestion = DataIngestion()
            feature_store_file_path = data_ingestion.initiate_data_ingestion()
            return feature_store_file_path    
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_transformation(self, feature_store_file_path):
        """
            Initiates the data tranformation process, which is responsible for preprocessing the data(eg., scaling, encoding) and splitting it into training and testing sets

            Steps:
                - An instance of DataTranformation is created, with the feature store file path passed to it.
                - The initiate_data_transformation() method of DataTransformation is called which transforms the data and splits it into training and test sets.
                - The method returns:
                    train_arr: The transformed training data
                    test_arr: The transformed test data
                    preprocessor_path: The path where the preprocessor(like scaling, imputing etc) is saved
        """
        try:
            data_transformation = DataTransformation(feature_store_file_path= feature_store_file_path)
            train_arr, test_arr,preprocessor_path = data_transformation.initiate_data_transformation()
            return train_arr, test_arr,preprocessor_path 
        except Exception as e:
            raise CustomException(e,sys)

    def start_model_training(self, train_arr, test_arr):
        """
            Initiates the model training process, which is responsible for training ML models and evaluating their performance

            Steps:
                - An instance of ModelTrainer() is created.
                - The initiate_model_trainer() method of ModelTrainer is called, passing the training and test data arrays(train_arr, test_arr).
                - The model is trained, and the final model score(such as r2_score or accuracy_score) s returned
        """
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
            return model_score     
        except Exception as e:
            raise CustomException(e,sys)

    def run_pipeline(self):
        """
            This is the main method that runs the entire ML pipeline, executing data ingestion, transformation and model training in sequence

            Steps:
                - Data Ingestion: Calls start_data_ingestion() to ingest the data and get the feature store file path.
                - Data Transformation: Calls start_data_transformation() to preprocess the data and split it into training and test sets.
                - Model Training: Calls start_model_training() to train the model and get its score.
                - The final model score is printed to the console after training is completed 
        """ 
        try:
            feature_store_file_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(feature_store_file_path)
            r2_square = self.start_model_training(train_arr, test_arr)
            
            print("Training completed. Trained model score : ", r2_square)
        except Exception as e:
            raise CustomException(e, sys)