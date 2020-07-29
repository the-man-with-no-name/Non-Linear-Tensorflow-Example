# Tensorflow to Model Non-linear input interactions
#   - Determinant of 2x2 matrix

import tensorflow as tf 
import math
import numpy as np
import matplotlib.pyplot as plt

# Initialize the training matrices
train_matrices = []
train_outputs = []
for _ in range(500):
    temp = np.random.randint(1,10,size=(2,2))
    train_matrices.append(temp)
    train_outputs.append(int(np.linalg.det(temp)))

# Initialize the test matrices
test_matrices = []
test_outputs = []
for _ in range(50):
    temp = np.random.randint(1,10,size=(2,2))
    test_matrices.append(temp)
    test_outputs.append(int(np.linalg.det(temp)))

# Tensorflow takes numpy arrays as input
train_matrices = np.array(train_matrices)
train_outputs = np.array(train_outputs)

test_matrices = np.array(test_matrices)
test_outputs = np.array(test_outputs)

#
#  [ a b ]
#  [ c d ]  --> [a b c d]
#
#  Comments above lines give the reasoning of why we choose the layer sizes and normalizations.
#    - Note that this can only be done since we have the prior knowledge of a nonlinear formula
#      for the determinant.
#    - The combination of the log activation, then a dense layer, then an exp activation 
#      creates the desired nonlinearity.
#
# Flatten the 2x2 matrix to a 4 element array
input_layer = tf.keras.layers.Flatten(input_shape=(2,2,1))
# Apply log to data [log(a) log(b) log(c) log(d)]
act1 = tf.keras.layers.Activation(tf.math.log)
# Here we hope to have something like log(ad) = log(a) + log(d), log(bc) = log(b) + log(c)
hidden_layer = tf.keras.layers.Dense(units=2)
# Now renormalize since e^(log(a) + log(d)) = ad and e^(log(b) + log(c)) = bc
act2 = tf.keras.layers.Activation(tf.math.exp)
# Combine the two using one Neuron ad - bc
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([input_layer, act1, hidden_layer, act2, output])

# Compile the model and display the architecture
model.compile(optimizer='adam', loss="mean_squared_error", metrics=['mae','mse'])
print(model.summary())

# Train the model
history = model.fit(train_matrices, train_outputs, epochs=1000, verbose=True)
print("Finished training the model.")

# Hopefully we should have 2 parameters on the diagonal or antidiagonal close to 1 
#   and the other should have 2 parameters close to 0
print(f"These are the layer 1 parameters: {hidden_layer.get_weights()}")
# Here, one parameter should be close to 1, the other close to -1
print(f'These are the layer 2 parameters: {output.get_weights()}')

# Plot the loss magnitude vs (training) epoch number
# If loss magnitude approaches 0, the model fits the training data very well
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

# Evaluate the model on the test data
print('Evaluating Test Data:')
results = model.evaluate(test_matrices,test_outputs)
print(f"test results: {results}")

# Give a prediction on an unseen matrix
predicter = np.random.randint(1,10,size=(2,2))
print(predicter)
pred = model.predict(np.array([predicter]))
print(f'Predicted Determinant: {pred}')
print(f'Acual Determinant: {int(np.linalg.det(predicter))}')
