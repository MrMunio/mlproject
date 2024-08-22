import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

@dataclass
class DataTransformation:
    DataTransformationConfig=DataTransformationConfig()
    def get_data_transformer_object(self):
        
        '''
        this function is responsible for data transformation for various types of data
        '''
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"]
            num_pipeline=Pipeline(
                steps=[
                    ('inputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info(f"pipeline created for numerical columns {numerical_columns}")
            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler())
            ])
            logging.info(f"pipeline created for numerical columns {categorical_columns}")
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("train test data loaded as df")
            logging.info("obtaining preprocessing obj")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            save_obj(
                file_path=self.DataTransformationConfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("preprocessing object saved")

        except Exception as e:
            raise CustomException(e,sys)