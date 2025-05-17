from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

irisData = load_iris()

df = pd.DataFrame(irisData.data, columns=irisData.feature_names)
df['species'] = irisData.target

df['species'] = df['species'].map({0: 'sestosa', 1:'versicolor', 2:'virginica'})
df.head()

# ANALYSIS
sns.pairplot(df, hue="species", markers=["o","s","D"])
plt.show()

#Preprocessing
X = df.drop(columns=['species'])
Y = df['species']

# Spliting to for testing and  train
X_train, X_tests, Y_train, Y_tests = train_test_split(X,Y, test_size=0.2, random_state=42)

print(f"Samples: {len(X_train)}, Testing Samples: {len(X_tests)}")


#MODEL SELECTION
from sklearn.neighbors import KNeighborsClassifier\

knnModel = KNeighborsClassifier(n_neighbors=3)

# training
knnModel.fit(X_train, Y_train)

Y_prediciton  = knnModel.predict(X_tests)

# Evaluation
modelAccuracy = accuracy_score(Y_tests, Y_prediciton)
print(f"Classification Report: ")
print(classification_report(Y_tests, Y_prediciton))


# Confusion Matrix
plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(Y_tests, Y_prediciton), annot=True, fmt="d", cmap="Blues", xticklabels=irisData.target_names, yticklabels=irisData.target_names)

plt.xlabel("Predicted Label")
plt.ylabel('True Label')
plt.title("Confusion Matrix")
plt.show()


#TESTING 
sample = [[5.1,5.6,3.5,0.4]]

newPrediction = knnModel.predict(sample)
print(f"New Prediction:", newPrediction[0])
