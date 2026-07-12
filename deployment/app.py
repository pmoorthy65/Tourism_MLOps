import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="pdhakshinamoor/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for Visit With Us staff that predicts whether a customer may purchase a tourism package.")
st.write("Kindly enter the customer details to check whether they are likely to purchase.")

# Collect user input
Age = st.number_input("Age (customer's age)", min_value = 8, value = 35)
TypeofContact = st.selectbox("Whether the customer is company invited or self enquired", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("City Tier the customer is from ", min_value = 1, max_value = 3, value = 1)
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer in minutes.", min_value = 15)
Occupation = st.selectbox("Occupation of the customer", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
Gender = st.selectbox("Gender of the customer", ['Male', 'Female'])
NumberOfPersonVisiting = st.number_input("Number of persons accompanying the customer on the trip", min_value = 0, value = 3)
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch.", min_value = 3)
ProductPitched = st.selectbox("The type of product pitched to the customer..", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer", min_value = 1, max_value = 7, value = 3)
MaritalStatus = st.selectbox("Marital status of the customer", ["Married", "Single", "Divorced"])
NumberOfTrips = st.number_input("Average number of trips the customer takes annually.", min_value = 0, value = 3)
Passport = st.selectbox("Whether the customer holds a valid passport", ['Yes', 'No'])
PitchSatisfactionScore = st.selectbox("Score indicating the customer's satisfaction with the sales pitch.", [1,2,3,4,5])
OwnCar = st.selectbox("Whether the customer owns a car", ['Yes', 'No'])
Designation = st.selectbox("Customer's designation in their current organization.", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
MonthlyIncome = st.number_input("Gross monthly income of the customer.", min_value = 25000)


# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "purchase" if prediction == 1 else "not purchase"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
