import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import dataset
import zlib

class Autoencoder(tf.keras.Model):
  """
  Autoencoder model class that defines an encoder and decoder. Supports both dense
  and convolutional architectures based on the isDense flag.
  """
  
  def __init__(self, latent_dim=64, shape=(28,28,1), isDense=True):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    # Define encoder and decoder
    self.encoder, self.decoder = self.define_model(isDense)
    # Parameters for compression metrics
    self.entropy_noise_stddev = 0.001
    self.image_error_threshold = 0.05
    #DEBUG
    self.sample = {"x": None, "y": None, "y_feature": None}

  def define_model(self, isDense=True):
    if isDense:
      print("Autoencoder with dense layers")
      encoder = tf.keras.Sequential([
        tf.keras.layers.Input(self.shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5*self.latent_dim, activation='relu'),      
        tf.keras.layers.Dense(3*self.latent_dim, activation='relu'),      
        tf.keras.layers.Dense(self.latent_dim, activation='relu'),
      ])
      decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(3*self.latent_dim, activation='relu'),
        tf.keras.layers.Dense(5*self.latent_dim, activation='relu'),
        tf.keras.layers.Dense(np.prod(self.shape), activation='sigmoid'),
        tf.keras.layers.Reshape(self.shape),
      ])
    else:
      print("Autoencoder with convolutional layers")
      encoder = tf.keras.Sequential([
        tf.keras.layers.Input(self.shape),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(padding='same'),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(self.latent_dim, activation='relu')
      ])
      decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*32, activation='relu'),  # Reshape to spatial dims
        tf.keras.layers.Reshape((7, 7, 32)),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.UpSampling2D(),
        tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid'),
      ])
    return encoder, decoder

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def train_step(self, data) -> dict:
    """
    This is where the training step is defined for custom training loops. From what I understand,
    all of the custom metric in this project couuld be defined here instead of subclassing the
    tf.keras.callbacks.Callback class. At the time I did not know how to fully defince custom functions
    that support both the eagerly and graph modes, so I used the Callback class instead. It seems that
    with the correct decorators one can define custom metrics that can be debugged in eagerly mode and
    once verified, run in graph mode for performance.
    
    :param self: Description
    :param data: Description
    :return: Description
    :rtype: dict
    """
    # Unpack data
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        y_feature = self.encoder(x)
        loss = self.compiled_loss(y, y_pred)

        #Just save some samples for debugging in eagerly mode
        #self.sample["x"]=x
        #self.sample["y"]=y
        #self.sample["y_pred"]=y_pred        
        #self.sample["y_feature"]=y_feature

        # Apply gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update standard metrics
        self.compiled_metrics.update_state(y, y_pred)
        
        # Custom metric using input and output
        #custom_metric_value = tf.reduce_mean(tf.abs(x - y_pred))
        custom_metric_value = tf.reduce_sum(tf.abs(y - y_pred))/tf.reduce_sum(tf.abs(x))

        # Return dict of results (appears in History)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        results["custom_reconstruction_error"] = custom_metric_value
        return results

  def compute_compression_metrics(self, dataset) -> dict:
    """
    Most of the custom metric are defined here. In this implementation this part is executed by the
    CompressionMetricCallback at the end of each epoch. This way, the main model can the executed in
    graph mode for performance, while this part can be executed in eagerly mode for easier debugging.
    """
    total_original_size = 0
    total_original_noise_size = 0
    total_feature_size = 0
    total_error_correction_size = 0
    total_compressed_original_size = 0
    total_compressed_original_noise_size = 0
    total_compressed_feature_size = 0
    total_compressed_error_correction_size = 0

    for batch in dataset:
        x, y = batch
        y_pred = self(x, training=False)
        y_feature = self.encoder(x)
        # Exact error
        y_error = x - y_pred
        y_error = tf.where(tf.abs(y_error) < self.image_error_threshold, 0, y_error)

        # Add minimal symbolic noise to original image to add entropy to dataset
        delta_noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=self.entropy_noise_stddev, dtype=tf.float32)
        x_noise = x + delta_noise
        x_noise = tf.clip_by_value(x_noise,0.0,1.0)

        # Convert tensors to numpy arrays and then to bytes
        original_bytes = tf.io.serialize_tensor(x).numpy()
        original_noise_bytes = tf.io.serialize_tensor(x_noise).numpy()       
        feature_bytes = tf.io.serialize_tensor(y_feature).numpy()
        error_bytes = tf.io.serialize_tensor(y_error).numpy()
        # Compress using zlib
        compressed_original = zlib.compress(original_bytes,wbits=9)
        compressed_original_noise = zlib.compress(original_noise_bytes,wbits=9)
        compressed_feature = zlib.compress(feature_bytes,wbits=9)
        compressed_error = zlib.compress(error_bytes,wbits=9)
        total_compressed_original_size += len(compressed_original) 
        total_compressed_original_noise_size += len(compressed_original_noise)
        total_compressed_feature_size += len(compressed_feature)
        total_compressed_error_correction_size += len(compressed_error)
        total_original_size += len(original_bytes) 
        total_original_noise_size += len(original_noise_bytes)
        total_feature_size += len(feature_bytes)
        total_error_correction_size += len(error_bytes)
    if total_original_size == 0:
        return 0.0
    # dataset ratio of zlib(Autoencoder feature+error correction)/Original images
    compression_ratio = (total_compressed_feature_size + total_compressed_error_correction_size) / total_original_size
    # Dataset ratio of zlib(Autoencoder feature)/Original images
    compression_ratio_lossy = total_compressed_feature_size  / total_original_size
    # Dataset ratio of zlib(Original image)/Original images
    x_ratio = total_compressed_original_size / total_original_size
    # Dataset ratio of zlib(Original image + minimal noise)/Raw images + minimal noise
    x_ratio_noise = total_compressed_original_noise_size / total_original_noise_size
    # Dataset ratio of zlib(Error correction)/Raw error correction
    e_ratio = total_compressed_error_correction_size / total_error_correction_size
    return {"compression_ratio": compression_ratio, 
            "compression_ratio_lossy": compression_ratio_lossy, 
            "total_compressed_original_size": total_compressed_original_size, 
            "total_original_size": total_original_size, 
            "total_original_noise_size": total_original_noise_size, 
            "total_feature_size": total_feature_size, 
            "total_error_correction_size": total_error_correction_size,
            "total_compressed_feature_size": total_compressed_feature_size, 
            "total_compressed_error_correction_size": total_compressed_error_correction_size, 
            "x_ratio": x_ratio, 
            "x_ratio_noise": x_ratio_noise, 
            "e_ratio": e_ratio}

  def test_step(self, data) -> dict:
    x, y = data
    y_pred = self(x, training=False)
    loss = self.compiled_loss(y, y_pred)

    self.compiled_metrics.update_state(y, y_pred)

    custom_metric_value = tf.reduce_mean(tf.abs(x - y_pred))

    results = {m.name: m.result() for m in self.metrics}
    results["loss"] = loss
    results["custom_reconstruction_error"] = custom_metric_value
    return results

class CompressionMetricCallback(tf.keras.callbacks.Callback):
    """
    This callback computes custom compression metrics at the end of each epoch. It
    is defined as a callback so that the main model can be executed in graph mode
    for performance, while this part can be executed in eagerly mode for easier debugging.
    It is specified in the model.fit() call as a callback.
    """
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def on_epoch_end(self, epoch, logs=None):
        metrics_data = self.model.compute_compression_metrics(self.dataset)
        logs['compression_ratio'] = metrics_data['compression_ratio']
        logs['compression_ratio_lossy'] = metrics_data['compression_ratio_lossy']
        logs['total_original_size'] = metrics_data['total_original_size']
        logs['total_feature_size'] = metrics_data['total_feature_size']
        logs['total_error_correction_size'] = metrics_data['total_error_correction_size']
        logs['x_ratio'] = metrics_data['x_ratio']
        logs['x_ratio_noise'] = metrics_data['x_ratio_noise']
        logs['e_ratio'] = metrics_data['e_ratio']
        logs['entropy_noise_stddev'] = self.model.entropy_noise_stddev
        logs['image_error_threshold'] = self.model.image_error_threshold