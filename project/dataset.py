import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# TODO: Create a CLI program with proper arguments with argparse


def get_data() -> dict:
    ds = tfds.load('mnist', split=["train[20%:]","train[:20%]","test"], shuffle_files=True, as_supervised=True, batch_size=32)
    return {"validate":ds[0], "train":ds[1], "test":ds[2]}

if __name__=="__main__":
    get_data()
