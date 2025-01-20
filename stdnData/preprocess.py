import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("synthetic_student_data.csv")

# Encode categorical variables (Learning Style)
encoder = OneHotEncoder(drop="first")
encoded_learning_style = encoder.fit_transform(df[["Learning Style"]])
encoded_learning_style_df = pd.DataFrame(encoded_learning_style.toarray(), columns=encoder.get_feature_names_out(["Learning Style"]))

# Drop the original Learning Style column and concatenate the encoded columns
df = pd.concat([df.drop("Learning Style", axis=1), encoded_learning_style_df], axis=1)

# Normalize numerical data
scaler = MinMaxScaler()
numerical_columns = ["Age", "IQ Score", "Quiz Score", "Time Taken (mins)"]
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Define features (X) and target (y)
X = df.drop(["Student ID", "Quiz Score"], axis=1)
y = df["Quiz Score"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Neural Network
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(64, input_dim=X_train.shape[1], activation="relu"))  # 64 neurons, ReLU activation

# Second hidden layer
model.add(Dense(32, activation="relu"))  # 32 neurons, ReLU activation

# Output layer
model.add(Dense(1, activation="linear"))  # 1 neuron (for regression), linear activation

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Save the trained model
model.save("models/neural_network_model.h5")
print("Model saved to 'models/neural_network_model.h5'.")