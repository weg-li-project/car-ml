
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('data_dir', '/../../data/cars_color', 'path to input data')
flags.DEFINE_integer('epochs', 10, 'number of training epochs')
flags.DEFINE_string('checkpoint_dir', '../checkpoints/cnn/training', 'path to save model')
flags.DEFINE_integer('patience', 5, 'patience of training')

class CNN(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):

        self.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(123, 95, 3)))
        self.add(MaxPool2D())
        self.add(Conv2D(40, kernel_size=5, padding='same', activation='relu'))
        self.add(MaxPool2D())
        self.add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
        self.add(MaxPool2D())
        self.add(Flatten())
        self.add(Dense(512, activation='relu'))
        self.add(Dropout(0.4))
        self.add(Dense(15, activation='softmax'))

def train(_argv):
    batch_size = 3
    img_height = 95
    img_width = 123

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FLAGS.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        FLAGS.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds_normalized = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds_normalized = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_normalized = train_ds_normalized.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds_normalized = val_ds_normalized.cache().prefetch(buffer_size=AUTOTUNE)

    model = CNN()
    model.create_model()
    sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    latest = tf.train.latest_checkpoint(os.path.dirname(FLAGS.checkpoint_dir))
    checkpoint_dir = FLAGS.checkpoint_dir
    try:
        model.load_weights(latest)
    except:
        print('There is no existing checkpoint')

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
        save_weights_only=True,
        save_freq='epoch')

    # Create a callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=FLAGS.patience,
        verbose=1)

    model.fit(train_ds_normalized, validation_data=val_ds_normalized, epochs=FLAGS.epochs, callbacks=[cp_callback, es_callback])

if __name__ == '__main__':
    try:
        app.run(train)
    except SystemExit:
        pass

