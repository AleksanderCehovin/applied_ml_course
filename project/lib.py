import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import dataset
import zlib

class Autoencoder(tf.keras.Model):
  def __init__(self, latent_dim=64, shape=(28,28,1)):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      tf.keras.layers.Input(shape),
      #tf.keras.layers.Rescaling(1./255),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(5*latent_dim, activation='relu'),      
      tf.keras.layers.Dense(3*latent_dim, activation='relu'),      
      tf.keras.layers.Dense(latent_dim, activation='relu'),
    ])
    print(f"Autoencode:init: {np.prod(shape)}")
    self.decoder = tf.keras.Sequential([
      tf.keras.layers.Dense(3*latent_dim, activation='relu'),
      tf.keras.layers.Dense(5*latent_dim, activation='relu'),
      tf.keras.layers.Dense(np.prod(shape), activation='sigmoid'),
      tf.keras.layers.Reshape(shape),
    ])
    self.entropy_noise_stddev = 0.001
    self.image_error_threshold = 0.05
    #DEBUG
    self.sample = {"x": None, "y": None, "y_feature": None}

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def train_step(self, data):
    # Unpack data
    x, y = data

    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        y_feature = self.encoder(x)
        loss = self.compiled_loss(y, y_pred)

        #Just save some samples for debugging
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

  def compute_compression_metrics(self, dataset) -> float:
    total_original_size = 0
    total_original_noise_size = 0
    total_feature_size = 0
    total_error_correction_size = 0
    total_compressed_original_size = 0
    total_compressed_original_noise_size = 0
    total_compressed_feature_size = 0
    total_compressed_error_correction_size = 0
    # TODO: Maybe actually run compression on each image separately
    for batch in dataset:
        x, y = batch
        y_pred = self(x, training=False)
        y_feature = self.encoder(x)
        # Exact error
        y_error = x - y_pred
        y_error = tf.where(tf.abs(y_error) < self.image_error_threshold, 0, y_error)

        # To be tested: add minimal symbolic noise to original image to add entropy to dataset
        delta_noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=self.entropy_noise_stddev, dtype=tf.float32)
        x_noise = x + delta_noise
        x_noise = tf.clip_by_value(x_noise,0.0,1.0)

        #y_error = x+0.05
        #Convert to integers in [0,255] or [-127,127] range
        #x = tf.clip_by_value(x,0.0,1.0)
        #y_feature = tf.clip_by_value(y_feature,0.0,1.0)
        #y_error = tf.clip_by_value(y_error,-1.0,1.0)
        #x = tf.cast(x*255.0, tf.uint8)
        #y_feature = tf.cast(y_feature*255.0, tf.uint8)
        #y_error = tf.cast((y_error+1.0)*127.5, tf.uint8)
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
    compression_ratio = (total_compressed_feature_size + total_compressed_error_correction_size) / total_compressed_original_size
    compression_ratio_lossy = total_compressed_feature_size  / total_compressed_original_size
    x_ratio = total_compressed_original_size / total_original_size
    x_ratio_noise = total_compressed_original_noise_size / total_original_noise_size
    e_ratio = total_compressed_error_correction_size / total_error_correction_size
    return compression_ratio, compression_ratio_lossy, total_compressed_original_size, total_compressed_feature_size, total_compressed_error_correction_size, x_ratio, x_ratio_noise, e_ratio

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

class CompressionMetricCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def on_epoch_end(self, epoch, logs=None):
        compression_ratio, compression_ratio_lossy, total_original_size, total_feature_size, total_error_correction_size, x_ratio, x_ratio_noise, e_ratio = self.model.compute_compression_metrics(self.dataset)
        logs['compression_ratio'] = compression_ratio
        logs['compression_ratio_lossy'] = compression_ratio_lossy
        logs['total_original_size'] = total_original_size
        logs['total_feature_size'] = total_feature_size
        logs['total_error_correction_size'] = total_error_correction_size
        logs['x_ratio'] = x_ratio
        logs['x_ratio_noise'] = x_ratio_noise
        logs['e_ratio'] = e_ratio
        logs['entropy_noise_stddev'] = self.model.entropy_noise_stddev
        logs['image_error_threshold'] = self.model.image_error_threshold