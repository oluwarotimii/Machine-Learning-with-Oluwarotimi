import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# LOAD THE DATASET

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [ "Pregnancies", "Glucose", "Blood Pressure", "Skin-Thickness", 
            "Insulin", "BMI", "Diabetes-Pedigree-Function", "Age", "Outcome" ]

df = pd.read_csv(url, names = columns)

#display
print(df.head())

print("Dataset information")
print(df.info())

print(f" Checking for missing values...")
print(df.isnull().sum())

print(f"Dataset Summary and Statistics")
print(df.describe())


