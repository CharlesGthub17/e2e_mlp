import os
import sys
import pandas as pd
import numpy as np
import dill
from source.exception import CustomException
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)        
    
def evaluate_models(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            
            # model.fit(x_train, y_train)   # Training the model 

            # Making predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            model_train_mae = mean_absolute_error(y_train, y_train_pred)
            model_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            model_train_r2 = r2_score(y_train, y_train_pred)

            model_test_mae = mean_absolute_error(y_test, y_test_pred)
            model_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            model_test_r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = model_test_r2

        return report 
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
