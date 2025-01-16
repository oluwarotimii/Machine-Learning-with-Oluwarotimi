import numpy as np

# Reshaping and stacking with Numpy

arr = np.arange(1,10)

print(arr)

#Reshape  to a 3 X 3 matrix

reshaped_arr = arr.reshape(3,3)
print("Reshaped Array (3 x 3): \n", reshaped_arr)

# flatten the aarray
flattened_arr = reshaped_arr.flatten()
print(flattened_arr)   # this brins it back to normal form

#Stacking Arrays vertically
a = np.array([1,3,4])
b = np.array([2,4,5])

stacked_arr = np.vstack((a,b))
print("Stacking Arrays vertically\n", stacked_arr)