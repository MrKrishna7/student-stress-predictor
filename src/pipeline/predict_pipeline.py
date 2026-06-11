import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            proba= model.predict_proba(data_scaled)
            preds = proba.argmax(axis=1)
            confidence = round(max(proba[0]) * 100, 2)

            if(preds==0):
                return "High Stress" , confidence
            elif(preds==1):
                return "Low Stress" , confidence
            else:
                return "Moderate Stress", confidence

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 study_hours: float,
                 sleep_hours: float,
                 social_hours: float,
                 physical_activity: float,
                 extracurricular: float,
                 gpa: float):

        self.study_hours  = study_hours
        self.sleep_hours  = sleep_hours
        self.social_hours= social_hours
        self.physical_activity = physical_activity
        self.extracurricular   = extracurricular
        self.gpa = gpa

    def get_data_as_data_frame(self):
        try:
            data = {
                "Study_Hours_Per_Day": [self.study_hours],
                "Sleep_Hours_Per_Day": [self.sleep_hours],
                "Social_Hours_Per_Day":  [self.social_hours],
                "Physical_Activity_Hours_Per_Day": [self.physical_activity],
                "Extracurricular_Hours_Per_Day":   [self.extracurricular],
                "GPA": [self.gpa]
            }
            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)