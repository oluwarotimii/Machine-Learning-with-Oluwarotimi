from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import numpy as np
from sklearn.datasets import fetch_california_housing
import pandas as pd


data = fetch_california_housing(as_frame = True)

df = data.frame

print(df.head())

X = data.data
Y = data.target

X_train, X_test, Y_train,  Y_test, = train_test_split( X, Y, test_size=0.3, random_state= 40 )

model = LinearRegression()
model.fit(X_train, Y_train)


predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(Y_test, predictions)

print(f"Mean Squared Error :", mse)

print(data.feature_names)