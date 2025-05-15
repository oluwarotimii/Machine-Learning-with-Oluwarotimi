#Numerical Python
# Numpy provides arrays like python list or faster

import numpy as np
arr = np.array([[1,4,5,6,6], [6,5,4,7,8]]) # this is a 2d array?
print(arr)

arr1 = arr[0]
arr2 = arr[1]
print( arr1, arr2)


final_result = arr1 * arr2
print(final_result)
newarr1 = arr1 - 1
newarr2 = arr2 - 1

print(f"New Array1 :{newarr1}, New Array 2 : {newarr2}")

# getting the array properties
print(f"The array shape for Arr is : ", {arr.shape})
print(f"The array shape for Arr1 is : ", {arr1.shape})
print(f"The array shape for Arr2 is : ", {arr2.shape})

#Type
print(f"The array type for Arr is : ", {arr.dtype})



# Class Work (Practice)
# Create a NumPy array containing integers from 10 to 30 (inclusive).

# Reshape it into a 5x3 matrix.

# Multiply every element in the matrix by 2.

# Find the mean and sum of all the elements in the matrix.

# Extract the second column of the matrix and print it.

lex = np.arange(10,25)
print(lex)

newLex = lex.reshape(5,3)

print(newLex)

newLex = newLex * 2

print(newLex)

meanLex = newLex.mean()
print(meanLex)

sumLex = newLex.sum()
print(sumLex)

Lex2cln = newLex[:,1]
print(f"This is the second Column of the Array.", Lex2cln)