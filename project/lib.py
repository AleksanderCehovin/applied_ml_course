import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import dataset

class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim=64, shape=(28,28,1)):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape),
      #tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    print(f"Autoencode:init: {np.prod(shape)}")
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(np.prod(shape), activation='sigmoid'),
      tf.keras.layers.Reshape(shape),
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


#shape = x_test.shape[1:]
#latent_dim = 64
#autoencoder = Autoencoder(latent_dim, shape)
