# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing dataset.
2. Select appropriate features and targets.
3. Split the dataset into training and testing sets.
4. Normalize both features and targets using StandardScaler.
5. Fit a SGDRegressor wrapped with MultiOutputRegressor on the scaled training data.
6. Predict on the test set and apply inverse transformation.
7. Calculate and print the Mean Squared Error (MSE).

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: T Ajay
RegisterNumber: 212223230007
*/
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice'] = dataset.target
print("Dataset Preview:\n", df.head())
X = df.drop(columns=['AveOccup', 'HousingPrice'])
Y = df[['AveOccup', 'HousingPrice']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("\nMean Squared Error:", mse)
print("\nSample Predictions:\n", Y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/b52559c0-a020-4747-bcbc-722c35941532)
![image](https://github.com/user-attachments/assets/a954674f-401a-489d-9e15-57f61788f31f)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
