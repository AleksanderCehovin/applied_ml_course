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
epochs = 10
batch_size=32
latent_dim=392

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
    # Visualize loss plots. No accuracy for autoencoder case
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

    plt.figure(figsize=(12, 8))
    plt.subplot(3,1,1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.subplot(3,1,2)
    plt.ylim(0,1.1*max(compression_ratio + compression_ratio_lossy + x_ratio_noise + e_ratio))
    plt.plot(epochs_range, compression_ratio,'bs-',label=f'Compression Ratio + Error>|{image_error_threshold}| ')
    plt.plot(epochs_range, compression_ratio_lossy,'rs-', label='Compression Ratio')
    plt.plot(epochs_range, x_ratio_noise,'m--', label=f'Image+/-|{entropy_noise_stddev}| Compression Ratio')
    plt.plot(epochs_range, e_ratio,'gs-', label='Error Compression Ratio')
    plt.legend(loc='upper right')
    plt.title("Custom Metrics")

    plt.subplot(3,1,3)
    plt.ylim(0,1.1*max(custom + e_ratio))
    plt.plot(epochs_range, custom,'bs-',label='Custom Reconstruction Error $\Sigma$|prediction error|/$\Sigma$|image|')
    plt.legend(loc='upper right')
    plt.title("Custom Error Correction Metrics")


    plt.show()


def plot_reconstruction_examples(model, dataset, N=5):
    assert N < batch_size, "N must be smaller than a batch_size"
    
    # Randomize images so we don't check the same ones every time
    examples=dataset.shuffle(10000).take(1)

    for image, image in examples:
        preds = model.predict(image)
        #print(f"preds.shape {preds.shape}")
        #print(f"image.shape {image.shape}")

    plt.figure(figsize=(2*N,7))
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

def run_all(isDense=False) -> None:
    dataset = load_data()
    model, history=run(dataset,isDense)
    plot(history)
    plot_reconstruction_examples(model, dataset['test'], 5)
    #DEBUG
    return model.sample


if __name__ == "__main__":
    run_all()
