import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# Define the CNN model
def create_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(400, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Replace this with the actual shape of your input data
input_shape = (20, 20, 1)

# Load the dataset from the .npz file
dataset_dir_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "gameplay", "processed")
dataset_path = os.path.join(dataset_dir_path, "0_0_10_1.npz")

loaded_data = np.load(dataset_path, allow_pickle=True)
dataset = loaded_data['dataset']

# Extract input and output from the dataset
inputs = np.array([data['input'] for data in dataset])
outputs = np.array([data['output'].flatten() for data in dataset])

# Create and compile the CNN model
model = create_cnn_model(input_shape)

# Train the model
model.fit(inputs, outputs, epochs=10, batch_size=20)

# Save the model
model.save("test.keras")

test_input = np.zeros((20, 20))
# test_input[3][3] = 1
# test_input[3][4] = 1
# test_input[3][5] = 1
# test_input[4][4] = -1
# test_input[5][4] = -1
# test_input[6][4] = -1
# test_input[7][4] = -1
test_input[2][2] = -1

# Make predictions with the trained model
predictions = model.predict(np.array([test_input]))

# Convert predictions to row and column indices
predicted_indices = np.argmax(predictions, axis=1)
predicted_rows = predicted_indices // 20
predicted_cols = predicted_indices % 20

# Print the predicted rows and columns
print("Predicted Rows:", predicted_rows)
print("Predicted Columns:", predicted_cols)
