import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)

model_path = os.path.join(os.path.dirname(__file__), "best_tourism_purchase_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism package Purchase Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package based on their personal and travel information.
Enter the customer datails below to get a prediction.
""")

Age                       = st.number_input("Age", 18, 100, 35)
TypeofContact             = st.selectbox("Type of Contact",['Self Enquiry' 'Company Invited'])
CityTier                  = st.selectbox("City Tier", [1, 2, 3])
Occupation                = st.selectbox("Occupation",['Salaried' 'Free Lancer' 'Small Business' 'Large Business'])
Gender                    = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting    = st.number_input("Number of Person Visiting", 1, 10, 2)
PreferredPropertyStar     = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus             = st.selectbox("Marital Status",['Single' 'Divorced' 'Married' 'Unmarried'])
NumberOfTrips             = st.number_input("Number of Trips", 0, 30, 3)
Passport                  = st.selectbox("Passport", [0,1])
OwnCar                    = st.selectbox("Own Car", [0,1])
NumberOfChildrenVisiting  = st.number_input("Number of Children Visiting", 0, 5, 0)
Designation               = st.selectbox("Designation",['Manager' 'Executive' 'Senior Manager' 'AVP' 'VP'])
MonthlyIncome             = st.number_input("Monthly Income", 0.0, 1000000.0, 30000.0)
PitchSatisfactionScore    = st.number_input("Pitch Satisfaction Score", 1, 5 ,3)
ProductPitched            = st.selectbox("Product Pitched",['Deluxe' 'Basic' 'Standard' 'Super Deluxe' 'King'])
NumberOfFollowups         = st.number_input("Number Of Followups", 0, 10, 3)
DurationOfPitch           = st.number_input("Duration Of Pitch(minutes)", 0, 60, 15)

input_data = pd.DataFrame([{
"Age": Age,
"TypeofContact": TypeofContact,
"Occupation": Occupation,
"Gender": Gender,
"NumberOfPersonVisiting":NumberOfPersonVisiting,
"PreferredPropertyStar": PreferredPropertyStar,
"MaritalStatus": MaritalStatus,
"NumberOfTrips": NumberOfTrips,
"Passport": Passport,
"OwnCar": OwnCar,
"NumberOfChildrenVisiting": NumberOfChildrenVisiting,
"Designation": Designation,
"MonthlyIncome": MonthlyIncome,
"PitchSatisfactionScore": PitchSatisfactionScore,
"ProductPitched": ProductPitched,
"NumberOfFollowups": NumberOfFollowups,
"DurationOfPitch": DurationOfPitch,

}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchase" if prediction == 1 else "No Package Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
