import numpy as np

#Indexing and Slicing?
# arr1 = np.arange(12,22,2)
# arr2 = np.arange(20,32,2)

arr = np.array([[1,2,4,5], [6,7,8,9]])
print(arr)

# Getting a single Element from the array?
print("Element at (1,2):", arr[1,2])

# Slice the row?
print("Second row: ", arr[1,:])

# Slice the Column?
print("Second Column: ", arr[:,1])

# slicing a sub matrix
print("Submatrix:\n", arr[0:2, 1:3])