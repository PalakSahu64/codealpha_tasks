import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 1. Load MNIST dataset (for handwritten digits 0–9)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Normalize the data (0-255 to 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. One-hot encode the labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 4. Build the model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))        # Input layer
model.add(Dense(128, activation='relu'))        # Hidden layer
model.add(Dense(10, activation='softmax'))      # Output layer (10 classes)

# 5. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Train the model
model.fit(X_train, y_train_cat, epochs=5, validation_split=0.1)

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print("Test Accuracy:", test_acc)

# 8. Predict and visualize
predictions = model.predict(X_test)

# Show one example
index = 5
plt.imshow(X_test[index], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[index])}, Actual: {y_test[index]}")
plt.axis('off')
plt.show()
