# Determinant of 2x2 matrix

import tensorflow as tf 
import math
import numpy as np
import matplotlib.pyplot as plt

train_matrices = []
train_outputs = []
for _ in range(500):
    #temp = np.random.rand(2,2)
    temp = np.random.randint(1,10,size=(2,2))
    train_matrices.append(temp)
    #train_outputs.append(float(np.linalg.det(temp)))
    train_outputs.append(int(np.linalg.det(temp)))

test_matrices = []
test_outputs = []
for _ in range(50):
    #temp = np.random.rand(2,2)
    temp = np.random.randint(1,10,size=(2,2))
    test_matrices.append(temp)
    #test_outputs.append(float(np.linalg.det(temp)))
    test_outputs.append(int(np.linalg.det(temp)))

train_matrices = np.array(train_matrices)
train_outputs = np.array(train_outputs)

test_matrices = np.array(test_matrices)
test_outputs = np.array(test_outputs)

input_layer = tf.keras.layers.Flatten(input_shape=(2,2,1))
act1 = tf.keras.layers.Activation(tf.math.log)
hidden_layer = tf.keras.layers.Dense(units=2)
act2 = tf.keras.layers.Activation(tf.math.exp)
hidden_layer2 = tf.keras.layers.Dense(units=1)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([input_layer, act1, hidden_layer, act2, hidden_layer2, output])

model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mae','mse'])
print(model.summary())

history = model.fit(train_matrices, train_outputs, epochs=3000, verbose=True)
print("Finished training the model.")

print(f"These are the layer 1 variables: {hidden_layer.get_weights()}")
print(f'These are the layer 2 variables: {hidden_layer2.get_weights()}')

# Plot the loss magnitude vs (training) epoch number
# If loss magnitude approaches 0, the model fits the training data very well
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

print('Evaluating Test Data:')
results = model.evaluate(test_matrices,test_outputs)
print(f"test results: {results}")

predicter = np.random.randint(1,10,size=(2,2))
print(predicter)
pred = model.predict(np.array([predicter]))
print(f'Predicted Determinant: {pred}')
print(f'Acual Determinant: {np.linalg.det(predicter)}')