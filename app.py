import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# Page title
st.title("Student Dropout Prediction")

st.write(
    "Enter student academic information to predict the student outcome."
)


# Load dataset only once
@st.cache_data
def load_data():
    return pd.read_csv("data.csv", sep=";")


df = load_data()


# Features used by the application
features = [
    "Age at enrollment",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)"
]


# Train model only once
@st.cache_resource
def train_model(data):

    X = data[features]
    y = data["Target"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = LogisticRegression(max_iter=10000)
    model.fit(X, y_encoded)

    return model, label_encoder


model, label_encoder = train_model(df)


# Student information
st.header("Student Information")


age = st.number_input(
    "Age at enrollment",
    min_value=int(df["Age at enrollment"].min()),
    max_value=int(df["Age at enrollment"].max()),
    value=20
)


first_approved = st.number_input(
    "1st semester approved units",
    min_value=0,
    max_value=int(
        df["Curricular units 1st sem (approved)"].max()
    ),
    value=0
)


first_grade = st.number_input(
    "1st semester grade",
    min_value=float(
        df["Curricular units 1st sem (grade)"].min()
    ),
    max_value=float(
        df["Curricular units 1st sem (grade)"].max()
    ),
    value=0.0
)


second_approved = st.number_input(
    "2nd semester approved units",
    min_value=0,
    max_value=int(
        df["Curricular units 2nd sem (approved)"].max()
    ),
    value=0
)


second_grade = st.number_input(
    "2nd semester grade",
    min_value=float(
        df["Curricular units 2nd sem (grade)"].min()
    ),
    max_value=float(
        df["Curricular units 2nd sem (grade)"].max()
    ),
    value=0.0
)


# Prediction
if st.button("Predict"):

    student = pd.DataFrame({
        "Age at enrollment": [age],
        "Curricular units 1st sem (approved)": [first_approved],
        "Curricular units 1st sem (grade)": [first_grade],
        "Curricular units 2nd sem (approved)": [second_approved],
        "Curricular units 2nd sem (grade)": [second_grade]
    })


    prediction = model.predict(student)

    probabilities = model.predict_proba(student)


    predicted_class = label_encoder.inverse_transform(
        prediction
    )[0]


    dropout_index = list(
        label_encoder.classes_
    ).index("Dropout")


    dropout_probability = probabilities[0][dropout_index]


    if dropout_probability >= 0.70:
        risk_level = "High Risk"

    elif dropout_probability >= 0.40:
        risk_level = "Medium Risk"

    else:
        risk_level = "Low Risk"


    st.subheader("Prediction Result")

    st.write(
        "Predicted Outcome:",
        predicted_class
    )

    st.write(
        "Dropout Probability:",
        round(dropout_probability * 100, 2),
        "%"
    )

    st.write(
        "Risk Level:",
        risk_level
    )


    st.subheader("Class Probabilities")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):
        st.write(
            class_name + ":",
            round(probability * 100, 2),
            "%"
        )
