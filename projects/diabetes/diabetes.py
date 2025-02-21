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


# ----------- VISUALIZATIONS -----------

# 1️⃣ Feature Importance (Coefficients)
feature_importance = np.abs(model.coef_[0])
feature_names = X.columns

plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance, color="skyblue")
plt.xlabel("Feature Importance (Coefficient Magnitude)")
plt.ylabel("Feature Names")
plt.title("Feature Importance in Logistic Regression")
plt.show()

# 2️⃣ Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# # 3️⃣ ROC Curve
# fpr, tpr, _ = roc_curve(y_test, y_probs)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random guess line
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC) Curve")
# plt.legend()
# plt.show()

# 4️⃣ Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="green", label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()