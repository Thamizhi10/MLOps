# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction")

api = HfApi()

# Data paths (update with your HF dataset paths)
Xtrain_path = "hf://datasets/tam3222/Tourism/Xtrain.csv"
Xtest_path = "hf://datasets/tam3222/Tourism/Xtest.csv"
ytrain_path = "hf://datasets/tam3222/Tourism/ytrain.csv"
ytest_path = "hf://datasets/tam3222/Tourism/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Define features
numeric_features = [
    "Age", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
     "NumberOfTrips", 
    "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation", "PreferredPropertyStar", "PitchSatisfactionScore", "Passport", "OwnCar", "CityTier"
]



# Set class weight for imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Cleaning the data

# Fix data quality issues before preprocessing
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize 'Gender'
    df['Gender'] = df['Gender'].replace({
        'Fe Male': 'Female'
    })
    
    # Standardize 'MaritalStatus'
    df['MaritalStatus'] = df['MaritalStatus'].replace({
        'Unmarried': 'Single'
    })
    
    return df

# Apply cleaning
Xtrain = clean_data(Xtrain)
Xtest = clean_data(Xtest)

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),  
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)


# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning (faster with RandomizedSearchCV)
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid,   # same param grid as before
        n_iter=20,                        # try 20 random combinations
        cv=3,                             # fewer folds to speed up
        n_jobs=-1,
        random_state=42
    )
    random_search.fit(Xtrain, ytrain)

    # Log all parameter combinations (nested runs)
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters in main run
    mlflow.log_params(random_search.best_params_)

    # Best model evaluation
    best_model = random_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save locally
    model_path = "best_tourism_package_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "tam3222/Tourism"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )
