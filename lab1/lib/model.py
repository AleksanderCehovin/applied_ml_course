import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def getDefaultParams():
    params = {"model":'linear',
              "train":{ "X":np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0]),
                         "Y":np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
                      },
              "test":{
                         "X":np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                     }
             }
    return params

class LabModel:

    def __init__(self, params=None):
        self.params=params
        if self.params is None:
            self.params = getDefaultParams()
        assert self.params["model"] == 'linear'
        self.model = tf.keras.models.Sequential([
                                tf.keras.layers.Dense(units=2, input_dim=1, activation='linear'),
                                tf.keras.layers.Dense(units=1, activation='linear')
                     ])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')
        plt.ion()

    def train(self):
        #Formula: Y = 2X – 1
        self.train_input_X   = self.params["train"]["X"]
        self.train_output_Y  = self.params["train"]["Y"]

        #Train the model
        #model.fit(train_input_X, train_output_Y, epochs=5)
        self.model.fit(self.train_input_X, self.train_output_Y, epochs=50)

        #Test the model
        self.test_input_X = self.params["test"]["X"]
        self.test_prediction_Y = self.model.predict(self.test_input_X)


    def plot(self): 
        #Plot the results
        plt.figure()
        plt.scatter(self.test_input_X, self.test_prediction_Y, color='red', label='Prediction')
        plt.plot(self.train_input_X, self.train_output_Y, color='green', label='Truth')
        plt.legend()
        plt.show()

