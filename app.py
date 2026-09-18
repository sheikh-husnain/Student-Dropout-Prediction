import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# Page title
st.title("Student Dropout Prediction")

st.write("Enter student academic information to predict the student outcome.")


# Categorical columns
categorical_columns = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance\t",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]


# Train model only once and reuse it
@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv("data.csv", sep=";")

    # Separate features and target
    X = df.drop("Target", axis=1)
    y = df["Target"]

    # Encode target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # One-hot encoding
    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_encoded, y_encoded)

    return df, X_encoded, model, label_encoder


# Load the trained model
df, X_encoded, model, label_encoder = train_model()


# Student inputs
st.header("Student Information")

age = st.number_input(
    "Age at enrollment",
    min_value=int(df["Age at enrollment"].min()),
    max_value=int(df["Age at enrollment"].max()),
    value=int(df["Age at enrollment"].iloc[0])
)

first_approved = st.number_input(
    "1st semester approved units",
    min_value=0,
    max_value=int(df["Curricular units 1st sem (approved)"].max()),
    value=int(df["Curricular units 1st sem (approved)"].iloc[0])
)

first_grade = st.number_input(
    "1st semester grade",
    min_value=float(df["Curricular units 1st sem (grade)"].min()),
    max_value=float(df["Curricular units 1st sem (grade)"].max()),
    value=float(df["Curricular units 1st sem (grade)"].iloc[0])
)

second_approved = st.number_input(
    "2nd semester approved units",
    min_value=0,
    max_value=int(df["Curricular units 2nd sem (approved)"].max()),
    value=int(df["Curricular units 2nd sem (approved)"].iloc[0])
)

second_grade = st.number_input(
    "2nd semester grade",
    min_value=float(df["Curricular units 2nd sem (grade)"].min()),
    max_value=float(df["Curricular units 2nd sem (grade)"].max()),
    value=float(df["Curricular units 2nd sem (grade)"].iloc[0])
)


# Prediction
if st.button("Predict"):

    # Use first dataset row as the base student
    student = df.iloc[[0]].drop("Target", axis=1).copy()

    # Replace selected values with user inputs
    student["Age at enrollment"] = age
    student["Curricular units 1st sem (approved)"] = first_approved
    student["Curricular units 1st sem (grade)"] = first_grade
    student["Curricular units 2nd sem (approved)"] = second_approved
    student["Curricular units 2nd sem (grade)"] = second_grade

    # Encode student data
    student_encoded = pd.get_dummies(
        student,
        columns=categorical_columns,
        drop_first=True
    )

    # Make sure student has exactly the same features as training data
    student_encoded = student_encoded.reindex(
        columns=X_encoded.columns,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(student_encoded)
    probabilities = model.predict_proba(student_encoded)

    # Convert prediction back to original class name
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    # Dropout probability
    dropout_probability = probabilities[0][0]

    # Risk level
    if dropout_probability >= 0.70:
        risk_level = "High Risk"
    elif dropout_probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    # Display result
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

    # Display all probabilities
    st.subheader("Class Probabilities")

    st.write(
        "Dropout:",
        round(probabilities[0][0] * 100, 2),
        "%"
    )

    st.write(
        "Enrolled:",
        round(probabilities[0][1] * 100, 2),
        "%"
    )

    st.write(
        "Graduate:",
        round(probabilities[0][2] * 100, 2),
        "%"
    )