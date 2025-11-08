# DT075A Laboration 2
# This code will need tensorflow and matplotlib to be installed. Use pip to install them.
import tensorflow as tf
import matplotlib.pyplot as plt
import urllib

# Variables
IMG_SIZE = 150
batch_size = 32
epochs = 40

def run():
    # Create train dataset
    train_datageneration = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_ds = train_datageneration.flow_from_directory(
        'images/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='binary')

    # Create validation dataset
    val_datageneration = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    val_ds = val_datageneration.flow_from_directory(
        'images/validation',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='binary')


    # Define the model
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)

    pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
                                input_shape=(IMG_SIZE,IMG_SIZE,3),
                                include_top=False,
                                weights=None)
    pre_trained_model.load_weights(weights_file)

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    x = tf.keras.layers.Flatten()(last_output)

    x = tf.keras.layers.Dense(1024, activation='relu')(x)

    x = tf.keras.layers.Dense(2)(x)

    model = tf.keras.Model(pre_trained_model.input, x)



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
