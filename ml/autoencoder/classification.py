import keras
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from constants import ORIGINAL, NOISED, DENOISED


def train_classification_model():
    """Create and train multiclass classification model"""

    # Define constants
    num_classes = 3
    input_shape = (28, 28, 1)

    # Open prepared dataset. 3000 items. Data already shuffled
    data = np.load('datasets/classification.npz')
    x_train = data['x']
    y_train = data['y']
    data.close()

    # Split dataset. train - 2700. test - 300
    x_test = x_train[:300]
    y_test = y_train[:300]

    x_train = x_train[300:]
    y_train = y_train[300:]

    # Converts a class vector (integers) to binary class matrix.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Create multiclass classification model
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    batch_size = 128
    epochs = 50

    # Train model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    # Save trained classification
    model.save('saved_model/classification')


def classification(img_arr) -> str:

    # Open trained model
    model = tf.keras.models.load_model('saved_model/classification')

    # Make prediction
    prediction = model.predict(img_arr)
    prediction = np.around(prediction)
    if prediction[0][0]:
        return ORIGINAL
    elif prediction[0][1]:
        return NOISED
    elif prediction[0][2]:
        return DENOISED


def main():
    train_classification_model()


if __name__ == '__main__':
    main()
