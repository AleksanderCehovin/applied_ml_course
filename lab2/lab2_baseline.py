# DT075A Laboration 2
# This code will need tensorflow and matplotlib to be installed. Use pip to install them.
import tensorflow as tf
import matplotlib.pyplot as plt

# Variables
IMG_SIZE = 150
batch_size = 32
epochs = 40

def run():
    # Create train dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'images/train',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size)

    # Create validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'images/validation',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Print a summary of the whole model
    model.summary()

    # Train the model and validate
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    
    )
    # Save the trained model for later
    model.save('saved_model.keras')

    return history

def plot(history):
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


def run_all():
    history=run()
    plot(history)


if __name__ == "__main__":
    run_all()
