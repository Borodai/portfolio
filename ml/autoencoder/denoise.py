import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.python.keras import Model
from utils import preprocess, noise, display


def create_model():
    """Define autoencoder model"""

    # Build the autoencoder
    # We are going to use the Functional API to build our convolutional autoencoder.
    my_input = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(my_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    # Autoencoder
    autoencoder = Model(my_input, x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return autoencoder


def train_denoiser_model():
    """Train and save denoiser model"""

    # Prepare the data
    # Since we only need images from the dataset to encode and decode,
    # we won't use the labels.
    data = np.load('datasets/mnist.npz')
    x_train, x_test = data['x_train'], data['x_test']
    data.close()

    # Normalize and reshape the data
    x_train = preprocess(x_train)
    x_test = preprocess(x_test)

    # Create a copy of the data with added noise
    x_noisy_train = noise(x_train)
    x_noisy_test = noise(x_test)

    # Create new model
    denoiser = create_model()
    denoiser.summary()

    # Train model using the noisy data as our input and the clean data as our target.
    # We want our autoencoder to learn how to denoise the images.
    denoiser.fit(
        x=x_noisy_train,
        y=x_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_data=(x_noisy_test, x_test), )

    # Save trained model
    denoiser.save('saved_model/denoiser')

    score = denoiser.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

def test_denoiser_model():
    """Evaluate saved model"""

    # Prepare data for test
    data = np.load('datasets/mnist.npz')
    x_test = data['x_test']
    data.close()

    # Normalize and reshape the data, add noise
    x_test = preprocess(x_test)
    x_noisy_test = noise(x_test)

    # Display the train data and a version of it with added noise
    display(x_test, x_noisy_test, 10)

    # Open saved trained model
    model = tf.keras.models.load_model('saved_model/denoiser')

    # Let's now predict on the noisy data and display the results of our autoencoder.
    prediction = model.predict(x_noisy_test)
    display(x_noisy_test, prediction, 10)


def denoise_image(img_arr):
    # Open trained model
    model = tf.keras.models.load_model('saved_model/denoiser')
    prediction = model.predict(img_arr)

    return prediction


def main():
    train_denoiser_model()


if __name__ == '__main__':
    main()
