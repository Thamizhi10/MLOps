import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

from huggingface_hub import HfApi, login
login(os.getenv("HF_TOKEN"))

# Download the model from the Model Hub
model_path = hf_hub_download(
    repo_id="tam3222/tourism",
    filename="best_tourism_package_prediction_model_v1.joblib"
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Conversion Prediction
st.title("Tourism Package Prediction App")
st.write("This app predicts whether a customer is likely to purchase the travel package based on their details.")
st.write("Please enter the customer details below:")

# Collect user input
Age = st.number_input("Age of the customer", min_value=18, max_value=100, value=30)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, value=5)
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Followups Done", min_value=0, value=1)
PreferredPropertyStar = st.number_input("Preferred Property Star Rating", min_value=1, max_value=5, value=3)
NumberOfTrips = st.number_input("Number of Trips Taken by Customer", min_value=0, value=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Large Business", "Small Business", "Salaried", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
CityTier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])

# Prepare input DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'CityTier': 1 if CityTier=="Tier 1" else 2 if CityTier=="Tier 2" else 3,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0
}])

# Classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "likely to purchase the package" if prediction == 1 else "not likely to purchase the package"
    st.write(f"Based on the information provided, the customer is {result}.")
