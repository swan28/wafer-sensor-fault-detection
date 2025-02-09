import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils

@dataclass
class ModelTrainerConfig:
    """
        `ModelTrainerConfig` class defines the configuration for the model training process.
    """
    # Directory where artifacts(like trained models) are stored
    artifact_folder = os.path.join(artifact_folder)

    # Path where final model will be saved
    trained_model_path = os.path.join(artifact_folder, "model.pkl" )

    # The minimum expected accuracy for the model
    expected_accuracy = 0.45

    # Path to the configuration file(model.yaml) that contains the model Hyperparameters
    model_config_file_path = os.path.join('config', 'model.yaml')

class ModelTrainer:
    """
        `ModelTrainer` is the main class responsible for training, evaluating, and fine-tuning models to find the best model
    """
    def __init__(self):
        # Instance of ModelTrainerConfig, which contains the configurations for the model training process
        self.model_trainer_config = ModelTrainerConfig()

        # Instance of MainUtils, which povides utility functions for saving objects
        self.utils = MainUtils()

        # Dictionary of ML models that will be trained and evaluated
        self.models = {
            'XGBClassifier': XGBClassifier(),
            'GradientBoostingClassifier' : GradientBoostingClassifier(),
            'SVC' : SVC(),
            'RandomForestClassifier': RandomForestClassifier()
            }
    
    def evaluate_models(self, X, y, models):
        """
            `evaluate_models()` - Trains and evaluates each model in the models dictionary on the training data, then calculates accuracy for both the training and test sets

            Parameters:
                `X`: The feature matrix
                `y`: The target vector
                `models`: Dictionary of models to evaluate

            Steps:
                For each model in the models dictionary
                    1. Train the model on the training data.
                    2. Predict the labels for both the training and test sets.
                    3. Calcularte accuracy for both training and test sets using `accuracy_score`().
                    4. Store the test accuracy in the report dictionary.
        """
        try:
            # Train-Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            report = {}

            # Iterate through each model
            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)  # Train model
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[list(models.keys())[i]] = test_model_score

            # Returns a report dictionary with models as keys and their accuracies as values
            return report

        except Exception as e:
            raise CustomException(e, sys)
        
    def get_best_model(
            self, 
            x_train:np.array, 
            y_train: np.array, 
            x_test:np.array, 
            y_test: np.array
            ):
        """
            `get_best_model()` - Finds the best model based on the accuracy score from `evaluate_models()`

            Parameters:
                `x_train, y_train, x_test, y_test`: Feature and Target arrays for training and testing

            Steps:
                1. Calls `evaluate_models()` to evaluate the models on the training data.
                2. Finds the model with the highest test accuracy form the `model_report`.
                3. Returns the best model's name, object and score.
        """
        try:
            model_report: dict = self.evaluate_models(
                x_train =  x_train, 
                y_train = y_train, 
                x_test =  x_test, 
                y_test = y_test, 
                models = self.models
                )
            
            print(model_report)
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(
                model_report.keys()
                )[list(model_report.values()).index(best_model_score)]

            best_model_object = self.models[best_model_name]
            return best_model_name, best_model_object, best_model_score
        except Exception as e:
            raise CustomException(e,sys)
            
    def finetune_best_model(
            self,
            best_model_object:object,
            best_model_name,
            X_train,
            y_train,
            )->object:
        """
            `finetune_best_model()` - Fine-tunes the best model using GridSearchCV to search for the best hyperparameters

            Parameters:
                `best_model_object`: The best model selected from get_best_model()
                `best_model_name`: The name of best model
                `X_train, y_train`: Training data used for fine-tuning
            
            Steps:
                1. Reads the models hyperparameter grid form the YAML file using the MainUtils.read_yaml_file().
                2. Performs a grid search on the model to find the best hyperparameter.
                3. Updates the best model with the fine-tuned parameters and returns the fine-tuned model.
        """
        try:
            # Go to the model.yaml via utils.read_yaml_file() then go to model_selection --> model --> best_model_name(whichever it is) --> search_param_grid
            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
                )["model_selection"]["model"][best_model_name]["search_param_grid"]


            grid_search = GridSearchCV(
                best_model_object, 
                param_grid=model_param_grid, 
                cv=5, 
                n_jobs=-1, 
                verbose=1 
                )
            
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            print("best params are:", best_params)
            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_trainer(self, train_array, test_array):
        """
            `initiate_model_trainer()` - The main method that orchestrates the entire model training, evaluation, and fine-tuning process

            Parameters:
                `train_array, test_array`: Arrays containing trianing and test data
            
            Steps:
                1. Splits the input arrays into features(X) and target(y) for both training and testing sets .
                2. Calls `evaluate_models()` to get the performance of each model.
                3. Identifies the best model using `get_best_model()`.
                4. Fine-tunes the best model using `finetune_best_model()`.
                5. Trains the fine-tuned model on the training set and evaluate its performance on the test set.
                6. If the model meets the accuracy threshold, it saves the model to disk as .pkl file.
        """
        try:
            logging.info(f"Splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info(f"Extracting model config file path")

            model_report: dict = self.evaluate_models(X=x_train, y=y_train, models=self.models)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(
                model_report.keys()
                )[list(model_report.values()).index(best_model_score)]

            best_model = self.models[best_model_name]

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=x_train,
                y_train=y_train
            )

            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)
            
            print(f"best model name {best_model_name} and score: {best_model_score}")

            if best_model_score<0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")
            
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"Saving model at path: {self.model_trainer_config.trained_model_path}")

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path), 
                exist_ok=True
                )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            # Returns the path to the saved model
            return self.model_trainer_config.trained_model_path
        except Exception as e:
            raise CustomException(e, sys)