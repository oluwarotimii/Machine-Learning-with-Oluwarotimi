import pandas as pd
import numpy as np
from faker import Faker

# Initialize Faker for generating random data
fake = Faker()

# Set the number of students
num_students = 10000

# Generate synthetic data
data = {
    "Student ID": [f"STU{str(i).zfill(5)}" for i in range(1, num_students + 1)],
    "Age": np.random.randint(10, 31, size=num_students),  # Age between 10 and 30
    "Learning Style": np.random.choice(["Visual", "Auditory", "Kinesthetic"], size=num_students),
    "IQ Score": np.random.randint(80, 161, size=num_students),  # IQ between 80 and 160
    "Quiz Score": np.random.randint(0, 101, size=num_students),  # Quiz score between 0 and 100
    "Time Taken (mins)": np.random.randint(5, 61, size=num_students)  # Time between 5 and 60 minutes
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("synthetic_student_data.csv", index=False)

# Display the first 5 rows of the dataset
print(df.head())