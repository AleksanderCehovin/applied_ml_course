import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: Create a CLI program with proper arguments with argparse
def autoencoder_preprocess(image, label):
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
        # Apply preprocessing, remove labels for autoencoder training
        ds_train = ds[0].map(autoencoder_preprocess).shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        ds_val   = ds[1].map(autoencoder_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        ds_test  = ds[2].map(autoencoder_preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return {"validate":ds[0], "train":ds[1], "test":ds[2]}


#def get_data() -> dict:
#    ds = tfds.load('mnist', split=["train[20%:]","train[:20%]","test"], shuffle_files=True, as_supervised=True, batch_size=32)
#    return {"validate":ds[0], "train":ds[1], "test":ds[2]}


if __name__=="__main__":
    get_data()
