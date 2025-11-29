####################################################################################
# First draft for an autoencoder model. 
####################################################################################
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import dataset
import lib

# Variables
epochs = 5
batch_size=32
latent_dim=32

def load_data() -> dict:
    return dataset.get_data(batch_size=batch_size,is_autoencoder=True)

def run(dataset) -> None:
    dataset = load_data()    
    train_ds = dataset["train"]
    validate_ds = dataset["validate"]
    test_ds = dataset["test"]

    # Extract image dimensions and channels
    IMG_X = train_ds.element_spec[0].shape[1]
    IMG_Y = train_ds.element_spec[0].shape[2]
    IMG_CHANNELS = train_ds.element_spec[0].shape[3]
    print(f"Shape {train_ds.element_spec[0].shape}")

    model = lib.Autoencoder(latent_dim=latent_dim,shape=(IMG_X,IMG_Y,IMG_CHANNELS))

    model.compile(optimizer='adam', loss="mse")

    # Print a summary of the whole model
    model.summary()

    # Train the model and validate
    history = model.fit(
        train_ds,
        validation_data=validate_ds,
        epochs=epochs
    
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
    print(f"Available keys in history: {history.history.keys()}")

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.subplot(1,2,2)
    plt.plot(epochs_range, custom, label='Custom')
    plt.plot(epochs_range, val_custom, label='Validation Custom')
    plt.legend(loc='upper right')
    plt.title("Custom Metrics")
    plt.show()


def plot_reconstruction_examples(model, dataset, N=5):
    assert N < batch_size, "N must be smaller than a batch_size"
    
    # Randomize images so we don't check the same ones every time
    examples=dataset.shuffle(10000).take(1)

    for image, image in examples:
        preds = model.predict(image)
        #print(f"preds.shape {preds.shape}")
        #print(f"image.shape {image.shape}")

    plt.figure(figsize=(2*N,4))
    for i in range(0,N):
        # Original image
        ax = plt.subplot(2,N,i+1)
        plt.imshow(image[i].numpy().squeeze(), cmap="gray")
        plt.title("original")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Predicted image
        ax = plt.subplot(2, N, i+1+N)
        plt.imshow(preds[i], cmap="gray")
        plt.title("reconstructed")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def run_all() -> None:
    dataset = load_data()
    model, history=run(dataset)
    plot(history)
    plot_reconstruction_examples(model, dataset['test'], 10)


if __name__ == "__main__":
    run_all()
