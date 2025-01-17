import numpy as np


random_arr = np.random.randint(1, 100, size=(4,5))

print(random_arr)

print("=========================")
# My bonus let me reshape
reshape_arr = random_arr.reshape(4,5)

print(reshape_arr)  # I know but let me do it

# random_arr = np.mean(random_arr)
# print(random_arr)   this is for the entire array

row_means = np.mean(random_arr, axis=1)
print("Row means: ",row_means)

print("=========================")

arr_mean = np.mean(random_arr)
arr_std = np.std(random_arr)

print('Array Mean: ', arr_mean, '\n', 'Array Standard Deviation:', arr_std )


normalized = (random_arr - arr_mean) / arr_std

print('Normalised version of the array is: ', normalized)

print("=========================")

# Multiply two matrices, 2 x 3 and 3 x 2

arr1 = np.random.randint(1,10, size=(2,3))
arr2 = np.random.randint(10,20, size=(3,2))

print(arr1, arr2)

full_arr = np.dot(arr1, arr2)
print("Matrix Multiplication:\n", full_arr)
