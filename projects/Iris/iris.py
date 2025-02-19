from sklearn.datasets import load_iris
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

df['species'] = df['species'].map({0: 'setosa', 1:'versicolor', 2:'virginica'})

df.head()

#Data analysis
sns.pairplot(df, hue="species", markers=["o",'s',"D"])
plt.show()


#preprocessing
#Spliting the dat into Train and test

# define the features
X = df.drop(columns=['species'])
y = df['species']

#Split into 80 - 20 for train and test

X_train, X_tests, y_train, y_tests = train_test_split(X,y, test_size=0.2, random_state=42)

print(f'Training Samples: {len(X_train)}, Testing samples:{len(X_tests)}')


#Model Selection
from sklearn.neighbors import KNeighborsClassifier
#Create  a KNN model

knn = KNeighborsClassifier(n_neighbors=3)

#train the model
knn.fit(X_train, y_train)

#model evaluation
y_pred = knn.predict(X_tests)

#Evaluate accuracy
accuracy = accuracy_score(y_tests, y_pred)
print("/nClassification Report:")
print(classification_report(y_tests, y_pred))

#Confucin matrix visualization
plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_tests, y_pred), annot=True, fmt="d", cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


#New predicitions
new_sample = [[5.1, 3.5, 1.4, 0.2]]

#predicted_species = kmn.predict(new_sample)
print('Predicted Species: {predcited_species[0]a}')