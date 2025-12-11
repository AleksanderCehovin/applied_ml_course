import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: Create a CLI program with proper arguments with argparse
def autoencoder_preprocess(image, label):
    image = tf.cast(image, tf.float32)/255.0
    return image, image

def get_data(batch_size=32,is_autoencoder=False) -> dict:
    ds = tfds.load(
                    'mnist', 
                    split=["train[20%:]","train[:20%]","test"], 
                    shuffle_files=True, 
                    as_supervised=True, 
                    batch_size=batch_size
    )
    
    if(is_autoencoder):
        # Apply preprocessing that normalizes the image, remove labels for autoencoder training.
        # Autotune is performande relatred for optimal batch processing
        ds[0] = ds[0].map(autoencoder_preprocess).shuffle(10000).prefetch(tf.data.AUTOTUNE)
        ds[1] = ds[1].map(autoencoder_preprocess).prefetch(tf.data.AUTOTUNE)
        ds[2] = ds[2].map(autoencoder_preprocess).prefetch(tf.data.AUTOTUNE)
    
    return {"validate":ds[0], "train":ds[1], "test":ds[2]}


if __name__=="__main__":
    get_data()
