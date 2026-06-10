import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test  = test_array[:, :-1]
            y_test  = test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree":       DecisionTreeClassifier(),
                "Random Forest":       RandomForestClassifier(),
                "Gradient Boosting":   GradientBoostingClassifier(),
                "XGBoost":             XGBClassifier(),
                "KNN":                 KNeighborsClassifier(),
                "AdaBoost":            AdaBoostClassifier(),
            }
            model_report = evaluate_models(
                X_train, y_train,
                X_test,  y_test,
                models
            )
            best_model_name  = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} "
                         f"with accuracy: {best_model_score:.4f}")
            print(f"\nBest Model: {best_model_name}")
            print(f"Accuracy:   {best_model_score:.4f}")
            if best_model_score < 0.52:
                raise CustomException("No good model found", sys)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved successfully")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)