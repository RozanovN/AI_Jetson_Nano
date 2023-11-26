from tensorflow.keras.src.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential


def preprocess_input(x):
    """Preprocess the input image according to AlexNet requirements."""
    x /= 127.5
    x -= 1.0
    return x


def alexnet():
    model = Sequential()
    model.add(Input((224, 224, 3)))
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid", activation='relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid"))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=3, activation="softmax"))

    model.compile(
        optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model