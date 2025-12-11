####################################################################################
# First draft for an autoencoder model. 
####################################################################################
from xml.parsers.expat import model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import dataset
import lib

# Variables
epochs=50
batch_size=32
latent_dim=128

def load_data() -> dict:
    return dataset.get_data(batch_size=batch_size,is_autoencoder=True)

def run(dataset, isDense=False) -> None:
    dataset = load_data()    
    train_ds = dataset["train"]
    validate_ds = dataset["validate"]
    test_ds = dataset["test"]

    # Extract image dimensions and channels
    IMG_X = train_ds.element_spec[0].shape[1]
    IMG_Y = train_ds.element_spec[0].shape[2]
    IMG_CHANNELS = train_ds.element_spec[0].shape[3]
    print(f"Shape {train_ds.element_spec[0].shape}")

    # Define the model. We can choose between dense or convolutional layers with the isDense flasg    
    model = lib.Autoencoder(latent_dim=latent_dim,shape=(IMG_X,IMG_Y,IMG_CHANNELS),isDense=isDense)

    # run_eagerly=True for debugging purposes, but makes training much slower.
    # See eagerly vs graph mode in TF documentation for details.
    #model.compile(optimizer='adam', loss="mse", run_eagerly=True)
    model.compile(optimizer='adam', loss="mse", run_eagerly=False)
    
    # Print a summary of the whole model
    model.summary()

    # Train the model and validate
    history = model.fit(
        train_ds,
        validation_data=validate_ds,
        epochs=epochs,
        # We mainly monitor compression metrics on training data for autoencoder
        callbacks=[lib.CompressionMetricCallback(dataset=train_ds)]
    
    )
    # Save the trained model for later
    model.save('saved_model.keras')
    
    return model, history

def plot(history) -> None:
    # Visualize loss plots. No accuracy for autoencoder case. In this case we
    # now have many custom metrics avaialble in history.history. These are defines
    # in lib.CompressionMetricCallback class.
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    custom = history.history['custom_reconstruction_error']
    val_custom = history.history['val_custom_reconstruction_error']
    compression_ratio = history.history['compression_ratio']
    compression_ratio_lossy = history.history['compression_ratio_lossy']
    total_original_size = history.history['total_original_size']
    total_feature_size = history.history['total_feature_size']
    total_error_correction_size = history.history['total_error_correction_size']
    x_ratio = history.history['x_ratio']
    x_ratio_noise = history.history['x_ratio_noise']
    e_ratio = history.history['e_ratio']
    image_error_threshold = history.history['image_error_threshold'][0]
    entropy_noise_stddev = history.history['entropy_noise_stddev'][0]
    print(f"Available keys in history: {history.history.keys()}")

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 10))
    plt.subplot(3,1,1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.subplot(3,1,2)
    plt.ylim(0,1.1*max(compression_ratio + compression_ratio_lossy + x_ratio_noise + e_ratio))
    plt.plot(epochs_range, compression_ratio,'bs-',label=f'Compression Ratio zlib(Feature+ Error>|{image_error_threshold}|)/original images')
    plt.plot(epochs_range, compression_ratio_lossy,'rs-', label='Compression Ratio zlib(Feature only)/original images')
    plt.plot(epochs_range, x_ratio_noise,'m--', label=f'Compression Ratio zlib(original images+/-|{entropy_noise_stddev}| noise)/original images')
    plt.plot(epochs_range, x_ratio,'m:', label=f'Compression Ratio zlib(original images)/original images')
    plt.legend(loc='upper right')
    plt.title(f"Custom Metrics: Autoencoder Latent Dimension={latent_dim}")

    plt.subplot(3,1,3)
    plt.ylim(0,1.1*max(custom + e_ratio))
    plt.plot(epochs_range, custom,'bs-',label=r'Autoencoder Reconstruction Error Ratio: $\Sigma$|prediction error|/$\Sigma$|image|')
    plt.legend(loc='upper right')
    plt.title("Autoencoder Custom Error Correction Ratio")


    plt.show()


def plot_reconstruction_examples(model, dataset, N=5):
    """
    Plots N examples of autoencoder reconstructions from the dataset using the model. This
    gives an idea of how well the autoencoder is performing. The bottom row with absolue errors
    are the type of error that the sparse error matrices would correct.
    
    :param model: Description
    :param dataset: Description
    :param N: Description
    """
    assert N < batch_size, "N must be smaller than a batch_size"
    
    # Randomize images so we don't check the same ones every time
    examples=dataset.shuffle(10000).take(1)

    for image, image in examples:
        preds = model.predict(image)
        #print(f"preds.shape {preds.shape}")
        #print(f"image.shape {image.shape}")

    plt.figure(figsize=(2.5*N,9))
    plt.title("Autoencoder reconstruction examples [No extra error correction]")
    for i in range(0,N):
        # Original image
        ax = plt.subplot(3,N,i+1)
        plt.imshow(image[i].numpy().squeeze(), cmap="gray")
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Predicted image
        ax = plt.subplot(3, N, i+1+N)
        plt.imshow(preds[i], cmap="gray")
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # Difference
        ax = plt.subplot(3, N, i+1+2*N)
        plt.imshow(np.abs(image[i]-preds[i]), cmap="gray")
        min_val = tf.reduce_min(image[i]-preds[i])
        max_val = tf.reduce_max(image[i]-preds[i])
        avg_val = tf.reduce_mean(image[i]-preds[i])
        plt.title(f"|diff|[min,avg,max]\n[{min_val:.2f},{avg_val:.2f},{max_val:.2f}]")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()

def run_model(isDense=True):
    dataset = load_data()
    model, history=run(dataset,isDense)
    return dataset, model, history

def run_all(isDense=True) -> None:
    dataset, model, history=run_model(isDense)
    plot(history)
    plot_reconstruction_examples(model, dataset['test'], 5)
    #DEBUG. Only run on small subset of test data for speed in Eagerly mode
    return model.sample

if __name__ == "__main__":
    run_all()
