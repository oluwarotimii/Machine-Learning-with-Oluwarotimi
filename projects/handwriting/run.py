import numpy as np
import matplotlib.pyplot as plt
import tensorflow as keras


# Loading the Data set

(X_train, y_train), (X_test, y_test) = keras.datasets,minst.load_data()

#print the  datasets

print(f"Training data Shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# Visualize some of the sample
plt.figure(figsize=(10,5))


for i in range(5):
    plt.subplot(1,5, i+1)
    plt.imshow(X_train[i], cmap="gray") # Shows the image in grayscale
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")