import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def getDefaultParams():
    """ Default parameters for LabModel class """
    params = {"train":{ "X":np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0]),
                         "Y":np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])
                      },
              "test":{
                         "X":np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
                     },
              "loss": 'mean_squared_error',
              "epocs": 50
             }
    return params

class LabModel:
    """
    Class for re-using model setup, training, and plotting steps common to many different training cases.

    Use the default params dictionary to change simple parameters and training data.

    Neural network model and optimizer is setup by caller to make thinks simples. If none is provided use
    a basic linear model instead.
    """
    def __init__(self, model: tf.keras.models.Sequential=None, optimizer: tf.keras.optimizers=None, params: dict=None) -> None:
        """ Setup model, optimizer and parameters """
        self.params=params
        if self.params is None:
            self.params = getDefaultParams()        
        self.model = model
        if model is None:
            self.model = tf.keras.models.Sequential([
                                tf.keras.layers.Dense(units=2, input_dim=1, activation='linear'),
                                tf.keras.layers.Dense(units=1, activation='linear')
                     ])
        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = 'sgd'                    
        self.model.compile(self.optimizer, loss='mean_squared_error')
        self.model.summary()
        plt.ion()

    def train(self) -> None:
        """ Train model setup by class constructor """
        # Training data
        self.train_input_X   = self.params["train"]["X"]
        self.train_output_Y  = self.params["train"]["Y"]

        #Train the model, save history
        self.history = self.model.fit(self.train_input_X, self.train_output_Y, epochs=self.params["epocs"])

        #Test the model
        self.test_input_X = self.params["test"]["X"]
        self.test_prediction_Y = self.model.predict(self.test_input_X)

    def plot(self, training_summary: str="Training Summary", loss_summary: str="Training Loss") -> None: 
        """ Plot result of training """
        #Plot the results
        plt.figure()
        plt.title(training_summary)
        plt.scatter(self.test_input_X, self.test_prediction_Y, color='red', label='Prediction')
        plt.plot(self.train_input_X, self.train_output_Y, color='green', label='Truth')
        plt.legend()
        plt.show()

        # Plot loss data during training, useful information how well the optimizer found a solution
        plt.figure()
        plt.title(loss_summary)
        Y = self.history.history['loss']
        X = range(1, len(Y) + 1)
        #plt.scatter(X, np.transpose(Y), linestyle='-', marker='s', color='b', label="Training loss")
        plt.plot(X, np.transpose(Y), 'bs-', label="Training loss")
        plt.legend()
        plt.show()

    def run(self) -> None:
        """ Run both training and plotting in one step. """
        self.train()
        self.plot()
