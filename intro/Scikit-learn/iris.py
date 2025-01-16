from sklearn.datasets import load_iris

# load the Iris datasets  (it is an inbuilt dataset)
iris = load_iris()
print(iris)
x = iris.data  #features
y = iris.target  #labels

print("Features:\n", x[:5])
print("labels:\n", y[:5])