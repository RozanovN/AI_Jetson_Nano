from pathlib import Path

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def model(compiled=False):
    if not compiled:
        image_size = (32, 32)
        batch_size = 32
        folder = Path(__file__).parent.parent.parent / "datasets/go_imgs/img/classification"
        datagen = ImageDataGenerator(
            rotation_range=5,
            rescale=1. / 255,
            horizontal_flip=True,
            fill_mode='nearest')
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        train_gen = datagen.flow_from_directory(
            str(folder / 'train'),
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True)
        test_gen = test_datagen.flow_from_directory(
            str(folder / 'test'),
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False)

    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    if not compiled:
        model.summary()
    for layer in model.layers:
        layer.trainable = False
    x = model.output
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions)
    if not compiled:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        epochs = 10
        history = model.fit(
            train_gen,
            epochs=epochs,
            verbose=1,
            validation_data=test_gen)
        model.save_weights('model_VGG16.h5')
        print(history.history)
    return model


if __name__ == '__main__':
    model()