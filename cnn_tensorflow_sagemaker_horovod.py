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
import horovod.tensorflow.keras as hvd
from tensorflow.keras import backend as K
import PIL
import math

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

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--val', type=str, default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--model_dir', type=str)

    args, _ = parser.parse_known_args()

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    img_height, img_width = 256, 256
    num_classes = 133
    batch_size = 40
    epochs = 50

    # Horovod: adjust number of epochs based on number of GPUs.
    epochs = int(math.ceil(epochs / hvd.size()))

    train_path = args.train
    valid_path = args.val
    test_path = args.test

    print(PIL.PILLOW_VERSION)

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
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        valid_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=123,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        seed=123,
        color_mode="grayscale",
        class_mode="sparse",
        shuffle=True
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

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.keras.optimizers.Adam()

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        MyThresholdCallback(threshold=0.05)
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    start = time()
    model.fit(train_generator,
              epochs=epochs,
              verbose=1,
              batch_size=batch_size,
              callbacks=callbacks,
              validation_data=val_generator)

    print(f"Total training time: {time() - start} seconds")


    score = model.evaluate(test_generator, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Model Dir:', args.model_dir)
    
    # Horovod: Save model only on worker 0 (i.e. master)
    if hvd.rank() == 0:
        saved_model_path = tf.contrib.saved_model.save_keras_model(model, args.model_dir)
        print("Model successfully saved at: {}".format(saved_model_path))