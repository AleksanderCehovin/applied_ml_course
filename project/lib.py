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

  def train_step(self, data):
    # Unpack data
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(y, y_pred)

        # Apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update standard metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Custom metric using input and output
        custom_metric_value = tf.reduce_mean(tf.abs(x - y_pred))

        # Return dict of results (appears in History)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        results["custom_reconstruction_error"] = custom_metric_value
        return results

  def test_step(self, data):
    x, y = data
    y_pred = self(x, training=False)
    loss = self.compiled_loss(y, y_pred)

    self.compiled_metrics.update_state(y, y_pred)

    custom_metric_value = tf.reduce_mean(tf.abs(x - y_pred))

    results = {m.name: m.result() for m in self.metrics}
    results["loss"] = loss
    results["custom_reconstruction_error"] = custom_metric_value
    return results


#shape = x_test.shape[1:]
#latent_dim = 64
#autoencoder = Autoencoder(latent_dim, shape)
