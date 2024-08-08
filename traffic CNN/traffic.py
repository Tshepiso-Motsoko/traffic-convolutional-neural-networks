import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Constants defining the neural network's parameters
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():
    """
    Main function to execute the program.
    """
    # Check command-line arguments to ensure correct usage
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets and convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network model
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance on the testing set
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file if a filename was provided
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []  # List to store the loaded images
    labels = []  # List to store the corresponding labels of the images

    # Iterate over all categories of traffic signs
    for category in range(NUM_CATEGORIES):
        # Build the path to the category directory
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue

        # Iterate over all images in the category directory
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            # Read the image using OpenCV
            image = cv2.imread(image_path)
            # Resize the image to the desired dimensions
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # Append the processed image and its label to the lists
            images.append(image)
            labels.append(category)

    return images, labels

def get_model():
    """
    Returns a compiled convolutional neural network model.

    The input shape of the first layer is (IMG_WIDTH, IMG_HEIGHT, 3).
    The output layer has NUM_CATEGORIES units, one for each traffic sign category.
    """
    # Create a sequential model
    model = tf.keras.models.Sequential([
        # Convolutional layer to extract features from the input image
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max pooling layer to reduce the spatial dimensions of the output volume
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # Flatten the 3D output to 1D for the dense layers
        tf.keras.layers.Flatten(),
        # Dense (fully connected) layer for learning non-linear combinations
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout layer to reduce overfitting
        tf.keras.layers.Dropout(0.5),
        # The output layer with softmax activation for multi-class classification
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model with optimizer, loss function, and metrics
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Standard boilerplate to call the main function
if __name__ == "__main__":
    main()
