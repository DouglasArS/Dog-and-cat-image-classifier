from time import time
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


def get_dataset():
    """ This function loads the set of images and performs the necessary pre-processing.

    Parameters
    -----
        It takes no parameters.

    Return
    -----
        training_data: set of pre-processed training images.
        test_data: set of pre-processed test images.
    """

    print("\n\n===== NUMBER OF IMAGES ===== \n")

    # Builds an image generator for the training set.
    training_set_generator = ImageDataGenerator(rescale=1.0 / 255.0,
                                             rotation_range=7, horizontal_flip=True, shear_range=0.2,
                                             height_shift_range=0.07, zoom_range=0.2)

    # Getting the images from the directory and running data augmentation on the training set.
    training_data = training_set_generator.flow_from_directory('cnn_generator/dataset/training_set',
                                                               target_size=(64, 64), class_mode='binary',
                                                               batch_size=64)

    # Builds an image generator for the test set. 
    test_set_generator = ImageDataGenerator(rescale=1.0 / 255.0)

    # Getting the images from the directory and running data augmentation on the test set.
    # Does not generate new images - no changes to images
    test_data = test_set_generator.flow_from_directory('cnn_generator/dataset/test_set',
                                                   target_size=(64, 64), class_mode='binary', batch_size=64)

    return training_data, test_data

def define_cnn():
    """ This function defines the CNN layers for its construction.

    Parameters
    -----
        It takes no parameters.

    Return
    -----
        Sequential model.
    """

    # Initializing empty model
    classifier = Sequential()

    # First Block
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Dropout(0.2))

    # Second block
    classifier.add(Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Dropout(0.2))

    # Third block
    classifier.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Dropout(0.2))

    # Flatting layer
    classifier.add(Flatten())

    # First hidden layer
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.3)) # Layer 30% dropout
                  
    # Second hidden layer
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.4)) # Layer 40% dropout
    
    # Output layer
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Compile model
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Printing the CNN summary
    print("\n\n===== SUMMARY: =====\n")
    classifier.summary()

    return classifier

def plot_learning_curves(history) -> None:
    """ This function builds the loss plot and the accuracy plot for both training and testing.

    Parameters
    -----
        history: History object (Its History.history attribute is a record of training loss 
        values and metrics values at successive epochs, as well as validation loss values and 
        validation metrics values if applicable).

    Return
    -----
        None.
    """

    # Plot Loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')

    # Plot Accuracy
    plt.subplot(212)
    plt.title('Acurácia  da Classificação')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')

    # Save plot to .png image file
    plt.savefig('./cnn_generator/history_plot.png')
    plt.close()

def build_cnn() -> None:
    """ This function creates and trains a CNN, and then saves the model.

    Parameters
    -----
        It takes no parameters.

    Return
    -----
        None.
    """

    # Initialize time
    start_time = time()

    # Load dataset
    training_data, test_data = get_dataset()

    # Build CNN
    cnn = define_cnn()

    # Fit CNN
    print("\n\n===== Fitting: =====\n")

    history = cnn.fit(training_data, steps_per_epoch=len(training_data),
                          epochs=50, validation_data=test_data, validation_steps=len(test_data))

    # Draw plot learning curves
    plot_learning_curves(history)

    # Get accuracy
    _, accuracy = cnn.evaluate(test_data, steps=len(test_data))

    # Printing the metrics
    metrics_file = open("./cnn_generator/metrics.txt","w")
    metrics_file.write("===== METRICS =====\n")
    metrics_file.write("Accuracy: {:.2f} %\n".format(accuracy*100))
    metrics_file.write("Time: {:.2f} seconds".format(time() - start_time))

    # Save model parameters
    classifier_json = cnn.to_json()

    with open('./cnn_generator/model_architecture.json', 'w') as json_file:
        json_file.write(classifier_json)

    # Save model weights
    cnn.save_weights("./cnn_generator/model_weights.h5")


if __name__ == '__main__':
    # Execute build CNN
    build_cnn()