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

    #train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #  'images/train',
    #  image_size=(IMG_SIZE, IMG_SIZE),
    #  batch_size=batch_size)

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

    #val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    #  'images/validation',
    #  image_size=(IMG_SIZE, IMG_SIZE),
    #  batch_size=batch_size)

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', 
                           input_shape=(IMG_SIZE,IMG_SIZE,3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(2)
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    #model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['accuracy'])

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
