import numpy as np


#Create a 1D array from a list?
arr1 = np.array([1,2,3,4,5])
print("This is a One dimensional array: ", arr1)

# 2 Dimensional array
arr2 = np.array([[1,2,3,4,5], [2,3,4,5,5]])
print("This is a 2 dimensional array: ", arr2)

#  an array of zeros?
zerosArr = np.array((3,3))
print("Zeros Array:\n", zerosArr)

# an array of ones
OnesArr = np.ones((2,4))
print("Array of Ones: ", OnesArr)

# creating an array with a range of values
RangeArr = np.arange(0,10,2) # Start, stop, step
print("Making an array from a range of values: ", RangeArr)

#Array Operations
#Element Wise operations
a = np.array([1,3,5,6])
b = np.array([2,4,6,8,])

print(a,b)
print("Addition: ", a + b)
print("Multiplication: ", a * b) 

#Broadcasting?
c = np.array([1,2,4,5])
print(c)
print("Broadcasting Addition ", c + 5)


#indexing and slicing in numpy

