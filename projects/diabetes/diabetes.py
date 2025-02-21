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



# Data preprocessing
# replacing the rows and cells of data with 0 with the median of that column

# identifyingg the columns with 0 values
ZeroCols = [col for col in df.columns if (df[col] == 0).sum() > 0 and col != "Outcome"]

#  replace the columns
df[ZeroCols] = df[ZeroCols].replace(0, np.nan)

# fil the missing values with the median of eacch column
df.fillna(df.median(), inplace=True)

# check to make sure it has all being implemented 

print("Columns with Zero have being Replaced...")
print(ZeroCols)

print("Missing Values after handling: ")
print(df.isnull().sum())



# Building the model
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Split to  features and target.
X = df.drop(columns=["Outcome"]) # all columns except the Outome column

Y = df["Outcome"]

# dividing the data into traina and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify=Y)


# Normalizing the columns with different scales using the StandardScaler()

scaler = StandardScaler()



# Fit the training data and transform teh entire data.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("seeing the result of the Fit and Transfomr of X_train, X test")
print(X_train, X_test)

# Training the model
X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)

model = LogisticRegression()

model.fit(X_train, y_train)


#Running Predicitions?

y_pred = model.predict(X_test)

print(f" The Prediction from y_pred.. \n{y_pred}")

# Testing for accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuraccy: {accuracy:.2f}")

#Prfint Confusion Matrix
print("This is the Confusion Matrix")
print(f"{confusion_matrix(y_test, y_pred)}")

# Classification Report
print("Classification Report: ")
print(classification_report(y_test, y_pred))