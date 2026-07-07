# Student Stress Level Predictor

A Flask web app that predicts a student's stress level (**High / Moderate / Low**) from daily lifestyle habits, using a scikit-learn/XGBoost classifier trained on a student lifestyle dataset.

**Live demo:** https://student-stress-predictor-ucno.onrender.com/

## How it works

Enter six daily lifestyle metrics and get back a predicted stress level with a confidence score:

Study Hours Per Day
Sleep Hours Per Day
Social Hours Per Day (Time spent socializing)
Physical Activity Hours Per Day ( Gym, sports, walking, exercise )
Extracurricular Hours Per Day ( Clubs, events, volunteering )
GPA - Current GPA (out of 4.0)

## Project structure

This follows a modular ML pipeline layout (ingestion → transformation → training → inference), rather than a single training script:

```
├── app.py                          # Flask app / inference endpoint
├── notebook/
│   ├── EDA.ipynb                   # Exploratory data analysis
│   └── data/student_lifestyle_dataset.csv
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Reads raw CSV, does stratified train/test split
│   │   ├── data_transformation.py  # Imputation + scaling, saves preprocessor.pkl
│   │   └── model_trainer.py        # Trains & compares multiple models, saves best one
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Orchestrates the training pipeline end-to-end
│   │   └── predict_pipeline.py     # Loads artifacts and serves predictions
│   ├── exception.py                # Custom exception with traceback context
│   ├── logger.py                   # Logging config
│   └── utils.py                    # save_object / load_object / evaluate_models helpers
├── artifacts/
│   ├── model.pkl                   # Trained model (best performer)
│   └── preprocessor.pkl            # Fitted ColumnTransformer (imputer + scaler)
├── templates/home.html             # Frontend form
├── static/home.css
└── requirements.txt
```

## Model

`model_trainer.py` trains and compares several classifiers under a single MLflow experiment (`Student Lifestyle Stress Predictor`), logging params, accuracy, precision, recall, and F1 for each run:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- KNN
- AdaBoost

The model with the highest test accuracy is persisted to `artifacts/model.pkl`. Preprocessing (median imputation + standard scaling on the six numerical features) is fit on the training split and saved separately to `artifacts/preprocessor.pkl`, then reused at inference time so training and serving stay consistent.

The label is `Stress_Level` (`High` / `Moderate` / `Low`), label-encoded during training and decoded back to text in `predict_pipeline.py`. Predictions are made with `predict_proba`, so the app can also surface a confidence percentage alongside the predicted class.

## Running locally

```bash
git clone https://github.com/MrKrishna7/student-stress-predictor.git
cd student-stress-predictor
pip install -r requirements.txt
```

**Use the app with the pre-trained model** (artifacts already included in `artifacts/`):

```bash
python app.py
```

Visit `http://localhost:5000`.

**Retrain the model from scratch:**

```bash
python -m src.pipeline.train_pipeline
```

This re-runs ingestion → transformation → training and overwrites `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

**View MLflow experiment results:**

```bash
mlflow ui
```

## Tech stack

Python · Flask · scikit-learn · XGBoost · MLflow · pandas / numpy · Gunicorn (deployment) · Render (hosting)
