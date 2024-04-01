import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from source.exception import CustomException
from source.logger import logging
import os
from source.utils import save_object 


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,train_df,test_df):     ### for data transformation
        try:
            numerical_columns=["Ratings","No_of_reviews","Discount"]
            categorical_columns= ["Name","Brand","No_of_ratings","Product_features","MSP","MRP"]

            # Convert specified columns to integer data types
            columns_to_convert = ["No_of_ratings", "MSP", "MRP"]
            train_df[columns_to_convert] = train_df[columns_to_convert].replace(',', '', regex=True)
            test_df[columns_to_convert] = test_df[columns_to_convert].replace(',', '', regex=True)

            train_df[columns_to_convert] = train_df[columns_to_convert].astype(int)
            test_df[columns_to_convert] = test_df[columns_to_convert].astype(int)
            
            num_pipeline=Pipeline(
                steps=[
                    ("scaler",StandardScaler())
                ]
            )
        
      
            logging.info("Numerical Columns Standard Scaling Completed")

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)    
                ]
            )    

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)    

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Print column names of training and testing DataFrames
            print("Training DataFrame columns:", train_df.columns)
            print("Testing DataFrame columns:", test_df.columns)

            logging.info("Read train and test data is completed")
            logging.info("Obtaining preprocessing object")


            preprocessing_obj = self.get_data_transformer_object(train_df,test_df)

            target_column_name = "MSP"
            num_columns = ["Ratings", "No_of_ratings", "No_of_reviews", "MSP", "MRP", "Discount"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                # self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)