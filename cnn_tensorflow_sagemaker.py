import subprocess
subprocess.call('pip install "pillow==8.0.1"', shell=True)

import argparse
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
from PIL import ImageFile
import PIL

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs["val_acc"]
        if val_acc >= self.threshold:
            self.model.stop_training = True


if __name__ == '__main__':

    print(PIL.PILLOW_VERSION)    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])

    args, _ = parser.parse_known_args()

    img_height, img_width = 256, 256
    num_classes = 133
    batch_size = 40
    epochs = 50

    train_path = args.train
    valid_path = args.val
    test_path = args.test

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        horizontal_flip=True,
        vertical_flip=True,
        data_format="channels_last",
        dtype=tf.float32
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1./255,
        data_format="channels_last",
        dtype=tf.float32
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=123,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True,
        subset="training"
    )

    val_generator = val_test_datagen.flow_from_directory(
        valid_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=123,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True,
        subset="training"
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=123,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True,
        subset="training"
    )

    input_shape = (img_height, img_width, 1)

    model = keras.Sequential()
    model.add(layers.Conv2D(16,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same',
                            activation='relu',
                            input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    callbacks = [
        MyThresholdCallback(threshold=0.1)
    ]
    
    start = time()
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        verbose=1,
                        epochs=epochs,
                        callbacks=callbacks,
                        batch_size=batch_size)
    print(f"Total training time: {time() - start} seconds")

    score = model.evaluate(test_generator, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
