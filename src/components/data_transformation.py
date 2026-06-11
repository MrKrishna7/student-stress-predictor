import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join(
        'artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "Study_Hours_Per_Day",
                "Sleep_Hours_Per_Day",
                "Social_Hours_Per_Day",
                "Physical_Activity_Hours_Per_Day",
                "Extracurricular_Hours_Per_Day",
                "GPA"
            ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns)
            ])

            logging.info(f"Numerical columns: {numerical_columns}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data read successfully")

            target_column = "Stress_Level"

            encoder = LabelEncoder()
            train_df[target_column] = encoder.fit_transform(
                train_df[target_column])
            test_df[target_column]  = encoder.transform(
                test_df[target_column])

            logging.info(f"Label encoding: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

            X_train = train_df.drop([target_column,"Student_ID"], axis=1)
            y_train = train_df[target_column]
            X_test  = test_df.drop([target_column,"Student_ID"], axis=1)
            y_test  = test_df[target_column]

            preprocessing_obj = self.get_data_transformer_object()

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr  = preprocessing_obj.transform(X_test)

            train_arr = np.column_stack((X_train_arr, np.array(y_train)))
            test_arr  = np.column_stack((X_test_arr, np.array(y_test)))

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
