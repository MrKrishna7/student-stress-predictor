import os
import sys
from dataclasses import dataclass
from time import time
import mlflow
import mlflow.sklearn
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
            mlflow.set_experiment(
                "Student Lifestyle Stress Predictor"
            )
            best_model = None
            best_model_name = ""
            best_model_score = 0

            for model_name, model in models.items():
                with mlflow.start_run(run_name=model_name):
                    logging.info(
                        f"Training started for {model_name}"
                    )
                    mlflow.log_params(
                        model.get_params()
                    )
                    start_time = time()

                    model.fit(X_train, y_train)

                    training_time = (
                        time() - start_time
                    )
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    mlflow.log_metrics(
                       {"training_time":training_time,
                        "accuracy": accuracy,
                        "precision": precision_score(y_test, y_pred, average='weighted'),
                        "recall": recall_score(y_test, y_pred, average='weighted'),
                        "f1_score": f1_score(y_test, y_pred, average='weighted')
                        
                       }
                    )
                    mlflow.log_param("model_name", model_name)
                    mlflow.sklearn.log_model(model, "model")
                    print(f"{model_name}: {accuracy:.4f}")
                    logging.info(f"{model_name} accuracy: {accuracy:.4f}")

                    if accuracy > best_model_score:
                        best_model_score= accuracy
                        best_model_name = model_name
                        best_model = model

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