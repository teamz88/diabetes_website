from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def home(request):
    return render(request, 'index.html')

def result(request):
    df = pd.read_csv(r"C:\Users\ASUS\Documents\diabetes.csv")

    # Columns with 0 values, but can not be 0
    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']

    # Replacing 0 values to NaN values
    df[zero_columns] = df[zero_columns].replace(0, np.nan)

    # Imputing 0 values with the average
    imputer = SimpleImputer(strategy='mean') 
    df[zero_columns] = imputer.fit_transform(df[zero_columns])

    # Split the Data into features (x) and labels (y)
    x = df.drop('Outcome', axis=1) 
    y = df['Outcome']

    # Instantiate the SMOTE object 
    smote = SMOTE(random_state=39)

    # Apply SMOTE to generate samples
    x_resampled, y_resampled = smote.fit_resample(x, y)

    # Convert the resampled data back into a DataFrame
    df_resampled = pd.DataFrame(x_resampled, columns=x.columns)
    df_resampled['Outcome'] = y_resampled

    # Split the resampled data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=39)

    # Instantiate a Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=39)

    # Train the Random Forest classifier
    rf_classifier.fit(x_train, y_train)

    query_dict = request.GET  
    # param_value = query_dict.get('param_name')
    val1 = query_dict.get('n1')
    val2 = float(query_dict.get('n2', 0))
    val3 = float(query_dict.get('n3', 0))
    val4 = float(query_dict.get('n4', 0))
    val5 = float(query_dict.get('n5', 0))
    val6 = float(query_dict.get('n6', 0))
    val7 = float(query_dict.get('n7', 0))
    val8 = float(query_dict.get('n8', 0))
    val9 = float(query_dict.get('n9', 0))

    pred = rf_classifier.predict([[val2, val3, val4, val5, val6, val7, val8, val9]])

    result3 = ''
    
    if pred == [1]:
        result3 = 'Positive'
    else:
        result3 = 'Negative'

    return render(request, 'index.html', {"result2": result3})
