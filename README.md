# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step 1:
Import the required libraries: pandas for data manipulation and linear_model from sklearn for regression.

### Step 2:
Load the dataset using pd.read_csv() and store it in a DataFrame.

### Step 3:
Separate the features (independent variables) and the target (dependent variable):
Assign the relevant columns to X (features).
Assign the target column to y (output).
### Step 4:
Create a linear regression model using linear_model.LinearRegression() and fit it to the data using regr.fit(X, y).

### Step 5:
Print the regression coefficients (regr.coef_) and intercept (regr.intercept_).

### Step 6:
Use the trained model to make predictions using regr.predict() with the given input values.

### Step 7:
Display the predicted output.

## Program:
```python
# Import required libraries
import pandas as pd
from sklearn import linear_model

# Load the dataset
df = pd.read_csv("carsemission.csv")

# Define the independent variables (features) and dependent variable (target)
X = df[['Weight', 'Volume']]  # Features: Weight and Volume
y = df['CO2']                # Target: CO2 emissions

# Create a linear regression model
regr = linear_model.LinearRegression()

# Train the model using the data
regr.fit(X, y)

# Print the coefficients and intercept of the regression model
print("Coefficients (Weight, Volume):", regr.coef_)
print("Intercept:", regr.intercept_)

# Predict CO2 emissions for a given Weight and Volume
input_data = pd.DataFrame({'Weight': [3300], 'Volume': [1300]})
predictedCO2 = regr.predict(input_data)

# Output the predicted CO2 value
print("Predicted CO2 for the corresponding weight and volume:", predictedCO2)







```
## Output:
```python
Coefficients (Weight, Volume): [0.00755095 0.00780526]
Intercept: 79.69471929115939
Predicted CO2 for the corresponding weight and volume: [114.75968007]
```

### Insert your output

<br>

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
