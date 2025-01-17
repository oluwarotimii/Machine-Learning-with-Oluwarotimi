import pandas as pd

# Creating a Series
# This is similar to a column in na spreadsheet

# s = pd.Series([1,2,3,4,5,67,8])
# print("S series is:\n", s)


#CreatinG A datafram from a dictionary

data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25,30,35],
    'City':['New York', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
print(df)


