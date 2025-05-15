import pandas as pd

numbers = pd.Series([10,3,43,23,45])

print(numbers)

data = {
    'Name': ['Isreal', 'Jaden', 'Mike', 'Eunice', "Precious", "Rotimi", 'Toyosi', 'Faith', 'Joy', 'Ifeoluwa'],
    'Age': [12,34,56,78,34,7,34,23,7,6],
    'Location': ['Abuja', 'Lagos', 'Ilorin','Abuja', 'Lagos', 'Ilorin','Abuja', 'Lagos', 'Ilorin', 'Delta']
}

df = pd.DataFrame(data)

print(df)



# 🧪 Class Work
# Task:
# Create a DataFrame with the following data:
# Names: "Tolu", "Sarah", "Mike"

# Ages: 22, 28, 24

# Courses: "Math", "Physics", "Biology"

# Print:

# The entire DataFrame.

# Just the names column.

# The average age.

# Try it out and show me what you get!


newData = {
    "Names":["Tolu", "Sarah", "Mike"],
    'Ages': [22, 28, 24],
    'Courses': ["Math", "Physics", "Biology"]
}

newDf = pd.DataFrame(newData)

print(newDf)

print(f"Names are: ", newDf['Names'])

Age = newDf['Ages']

avAge = Age.mean()

print(f"Average age is :", avAge)