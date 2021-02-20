import os
import sys
import argparse as arg
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


class CNN(tf.keras.models.Sequential):

    def __init__(self):
        super().__init__()

    def create_model(self):
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

        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def train(data_dir, checkpoint_dir, epochs=5, patience=5):
    batch_size = 32
    img_height = 40
    img_width = 24

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        color_mode='grayscale',
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        color_mode='grayscale',
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds_normalized = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds_normalized = val_ds.map(lambda x, y: (normalization_layer(x), y))

    model = CNN()
    model.create_model()

    latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
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
        patience=patience,
        verbose=1)

    model.summary()

    model.fit(train_ds_normalized, validation_data=val_ds_normalized, epochs=epochs,
              callbacks=[cp_callback, es_callback])


def main(argv):
    parser = arg.ArgumentParser(description='Train the model.')
    parser.add_argument('--data_dir', metavar='<directory>', type=str, help='the data directory', required=True)
    parser.add_argument('--checkpoint_dir', metavar='<directory>', type=str, help='the checkpoint directory',
                        required=True)
    parser.add_argument('--epochs', metavar='<number>', type=int, help='the number of epochs to train', required=True)
    parser.add_argument('--patience', metavar='<number>', type=int,
                        help='the number of epochs to wait after training did not get better', required=True)
    args = parser.parse_args()

    train(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, epochs=args.epochs, patience=args.patience)


if __name__ == "__main__":
    main(sys.argv[1:])
