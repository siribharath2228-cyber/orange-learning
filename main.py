import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Exam Score Predictor")
st.write("Predict exam score based on hours studied")

# Sample dataset
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8],
    "Exam_Score": [30,35,50,55,65,70,80,90]
}

df = pd.DataFrame(data)

# Train model
X = df[["Hours_Studied"]]
y = df["Exam_Score"]

model = LinearRegression()
model.fit(X,y)

# User input
hours = st.number_input("Enter hours studied")

# Prediction
if st.button("Predict Score"):
    prediction = model.predict([[hours]])
    st.write("Predicted Exam Score:", round(prediction[0],2))