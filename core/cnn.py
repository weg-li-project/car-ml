
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from core.utils import load_data
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('data_dir', '../data/letters_cleaned', 'path to input data')
flags.DEFINE_integer('epochs', 5, 'number of training epochs')
flags.DEFINE_string('checkpoint_dir', '../checkpoints/cnn/training', 'path to save model')
flags.DEFINE_boolean('advanced', False, 'whether to use the advanced model or not')

class CNN(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self, advanced=False):
        if advanced == False:
            self.add(Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(40, 24, 1)))
            self.add(MaxPool2D())
            self.add(Conv2D(40, kernel_size=5, padding='same', activation='relu'))
            self.add(MaxPool2D())
            self.add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
            self.add(MaxPool2D())
            self.add(Flatten())
            self.add(Dense(512, activation='relu'))
            self.add(Dropout(0.4))
            self.add(Dense(38, activation='softmax'))

        else:
            self.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(40, 24, 1)))
            self.add(BatchNormalization())
            self.add(Conv2D(32, kernel_size=3, activation='relu'))
            self.add(BatchNormalization())
            self.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
            self.add(Dropout(0.4))

            self.add(Conv2D(40, kernel_size=3, activation='relu'))
            self.add(BatchNormalization())
            self.add(Conv2D(40, kernel_size=3, activation='relu'))
            self.add(BatchNormalization())
            self.add(Conv2D(40, kernel_size=5, padding='same', activation='relu'))
            self.add(Dropout(0.4))

            self.add(Conv2D(48, kernel_size=3, activation='relu'))
            self.add(BatchNormalization())
            self.add(Conv2D(48, kernel_size=3, activation='relu'))
            self.add(BatchNormalization())
            self.add(Conv2D(48, kernel_size=5, padding='same', activation='relu'))
            self.add(Dropout(0.4))

            self.add(Flatten())
            self.add(Dense(512, activation='relu'))
            self.add(BatchNormalization())
            self.add(Dropout(0.4))
            self.add(Dense(38, activation='softmax'))

        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def train(_argv):
    data_dir = FLAGS.data_dir
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train, y_train = tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test, y_test = tf.convert_to_tensor(X_test, dtype=tf.float32), tf.convert_to_tensor(y_test, dtype=tf.float32)
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Add a channels dimension
    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]

    batch_size = 32

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0]).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    model = CNN()
    model.create_model(advanced=FLAGS.advanced)

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
        patience=5,
        verbose=1)

    model.summary()

    model.fit(train_ds, validation_data=test_ds, epochs=FLAGS.epochs, callbacks=[cp_callback, es_callback])

if __name__ == '__main__':
    try:
        app.run(train)
    except SystemExit:
        pass

