# DT075A Laboration 2
# This code will need tensorflow and matplotlib to be installed. Use pip to install them.
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import dataset
import lib

# Variables
epochs = 10
batch_size=32

def load_data() -> dict:
    return dataset.get_data(batch_size=batch_size,is_autoencoder=True)

def run() -> None:
    dataset = load_data()    
    train_ds = dataset["train"]
    validate_ds = dataset["validate"]
    test_ds = dataset["test"]

    # Extract image dimensions and channels
    IMG_X = train_ds.element_spec[0].shape[1]
    IMG_Y = train_ds.element_spec[0].shape[2]
    IMG_CHANNELS = train_ds.element_spec[0].shape[3]
    print(f"Shape {train_ds.element_spec[0].shape}")

    model = lib.Autoencoder(latent_dim=64,shape=(IMG_X,IMG_Y,IMG_CHANNELS))

    #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

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

    return history

def plot(history) -> None:
    # Visualize the accuracy and loss plots
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def run_all() -> None:
    history=run()
    plot(history)


if __name__ == "__main__":
    run_all()
