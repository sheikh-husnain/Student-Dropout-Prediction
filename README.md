# Student Dropout Prediction

This is my Machine Learning project for predicting student academic outcomes.

The project uses student information and academic performance data to predict whether a student is Dropout, Enrolled, or Graduate.

## Dataset

The dataset contains 4424 student records and 36 columns.

The target column is Target, which has three classes:

Dropout  
Enrolled  
Graduate

Before training the model, I checked the dataset for missing values and duplicate records. There were no missing values and no duplicate records.

## Data Preprocessing

First, I separated the student information from the Target column.

The Target values were converted into numbers using LabelEncoder.

The categorical columns were converted using one-hot encoding.

After preprocessing, the dataset had 238 feature columns.

## Exploratory Data Analysis

I explored the dataset using Python and Matplotlib.

I checked the distribution of student outcomes and looked at the relationship between student outcomes and academic performance.

Some observations from the analysis were:

Students who graduated had higher average approved curricular units.

Dropout students had lower average first semester grades.

The average age of dropout students was higher than the average age of enrolled and graduate students.

## Machine Learning Model

I used Logistic Regression for this project.

The dataset was divided into 80% training data and 20% testing data.

Training records: 3539

Testing records: 885

The model was then trained using the training data and used to make predictions on the testing data.

## Model Results

The model achieved the following results:

Accuracy: 76.50%

Precision: 74.79%

Recall: 76.50%

F1 Score: 74.85%

ROC-AUC: 89.55%

## Prediction

I also used the trained model to predict the outcome of a student.

The prediction was:

Predicted Outcome: Dropout

Dropout Probability: 92.09%

Risk Level: High Risk

## Tools Used

Python

Pandas

Scikit-learn

Matplotlib

Jupyter Notebook

## Project Files

Student_Dropout_Prediction.ipynb  
data.csv  
README.md

## Conclusion

This project helped me understand the basic Machine Learning workflow from data cleaning and analysis to model training, evaluation, and prediction.

The model can be used as a starting point for identifying students who may need additional academic support.