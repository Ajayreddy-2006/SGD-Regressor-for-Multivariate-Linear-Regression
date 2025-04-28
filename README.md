# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:Start the program.

Step 2: Load the California Housing dataset and select the first three features as inputs (X), and the target and an additional feature (Y) for prediction.

Step 3: Scale both the input features (X) and target variables (Y) using StandardScaler.

Step 4: Initialise SGDRegressor and use MultiOutputRegressor to handle multiple output variables.

Step 5: Initialise SGDRegressor and use MultiOutputRegressor to handle multiple output variables.

Step 6: Train the model using the scaled training data, and predict outputs on the test data.

Step 7: Inverse transform predictions and evaluate the model using the mean squared error (MSE). Print the MSE and sample predictions.

Step 6: Stop the program.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: T Ajay
RegisterNumber:  212223230007
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
```
X=df.drop(columns=['AveOccup','HousingPrice'])
Y=df[['AveOccup','HousingPrice']]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
```

```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",Y_pred[:5])


print("T Ajay")
print(212223230007)
```



## Output:

![image](https://github.com/user-attachments/assets/8105cb56-0813-4f84-8660-a4f7fe7928cc)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
