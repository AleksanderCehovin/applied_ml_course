# DT075A Laboration 2
# This code will need tensorflow, and numpy to be installed. Use pip to install them.
import tensorflow as tf
import numpy as np

def test_image(image_file, image):
    test_array = tf.keras.preprocessing.image.img_to_array(image)
    test_array = np.array([test_array])

    # Predict using model
    prediction = model.predict(test_array)

    # Interpret the results
    print ("Test image evaluation: ", prediction)
    #if prediction[0][0] > 0.5:
    #    print (f"Test {image_file} is of category: cat")
    #else:
    #    print (f"Test {image_file} is of category: dog")
    if prediction[0][0] > prediction[0][1]:
        print (f"Test {image_file} is of category: cat")
    else:
        print (f"Test {image_file} is of category: dog")

# Load our pre trained model
model = tf.keras.models.load_model('saved_model.keras')

# Define images files
image_files=['test/mypet.jpg',
             'test/cat_1.jpg',
             'test/cat_2.jpg',
             'test/dog_1.jpg',
             'test/dog_2.jpg',
             'test/dog.11821.jpg',
             'test/dog.11539.jpg',
             'test/dog.11801.jpg']

print("image_files",image_files)

# Load images
images=[]
for image in image_files:
    pil_image=tf.keras.preprocessing.image.load_img(image, target_size=(150, 150))
    pil_image.show()
    images.append(pil_image)

print("images", images)

# Evaluate images
for i in range(0,len(image_files)):
    test_image(image_files[i], images[i])
