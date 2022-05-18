import numpy as np
from keras.models import model_from_json
from keras_preprocessing import image
from tkinter import *
from tkinter import filedialog as fd

def load_model():
    """ This function loads the model.

    Parameters
        -----
            It takes no parameters.

        Return
        -----
            None.
    """
    # Load the CNN architecture file
    model_architecture_file = open('./cnn_generator/model_architecture.json', 'r')
    model_architecture = model_architecture_file.read()
    model_architecture_file.close()

    # Load the CNN model
    classifier = model_from_json(model_architecture)

    # Load the model weights
    classifier.load_weights('./cnn_generator/model_weights.h5')

    return classifier

def classify_image(file_address):
    """ This function classify the chosen image as dog or cat.

    Parameters
        -----
            It takes no parameters.

        Return
        -----
            None.
    """
    # Load model
    classifier = load_model()

    # Converting Image to Array
    test_image = image.load_img(path=file_address, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = test_image / 255
    test_image = np.expand_dims(test_image, axis = 0)

    # Run prediction
    prediction = classifier.predict(test_image)
    print(prediction)

    return prediction