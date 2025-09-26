import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_dim=1, activation='linear'),
    tf.keras.layers.Dense(units=1, activation='linear')
])
model.compile(optimizer='sgd', loss='mean_squared_error')

#Formula: Y = 2X – 1
train_input_X   = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0])
train_output_Y  = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

#Train the model
#model.fit(train_input_X, train_output_Y, epochs=5)
model.fit(train_input_X, train_output_Y, epochs=50)

#Test the model
test_input_X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
test_prediction_Y = model.predict(test_input_X)

#Plot the results
plt.scatter(test_input_X, test_prediction_Y, color='red', label='Prediction')
plt.plot(train_input_X, train_output_Y, color='green', label='Truth')
plt.legend()
plt.show()
