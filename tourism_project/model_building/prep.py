import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from huggingface_hub import HfApi

# Authenticate with Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define dataset path
DATASET_PATH = "hf://datasets/tam3222/tourism/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define target variable
target = "ProdTaken"

# Drop identifiers
df = df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore")

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


# Train-test split
X = df[numeric_features + categorical_features]
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

# Upload files to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

from huggingface_hub import HfApi, login
login(os.getenv("HF_TOKEN"))

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="tam3222/Tourism",
        repo_type="dataset",
    )
