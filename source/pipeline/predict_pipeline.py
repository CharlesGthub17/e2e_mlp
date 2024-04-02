import os
import sys
import pandas as pd
from source.exception import CustomException
from source.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Ratings: float,
        No_of_ratings: int,
        No_of_reviews: int,
        MRP: int,
        Discount: int):


        self.Ratings = Ratings

        self.No_of_ratings = No_of_ratings

        self.No_of_reviews = No_of_reviews

        self.MRP = MRP

        self.Discount = Discount


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Ratings": [self.Ratings],
                "No_of_ratings": [self.No_of_ratings],
                "No_of_reviews": [self.No_of_reviews],
                "MRP": [self.MRP],
                "Discount": [self.Discount]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)