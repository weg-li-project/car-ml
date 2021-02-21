import os
import sys
import argparse as arg
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, ZeroPadding2D, BatchNormalization, \
    Activation, add, AveragePooling2D, InputSpec
from tensorflow.keras import initializers as initializations
from tensorflow.keras import backend as K
from tensorflow.keras import Model


class Scale(tf.keras.layers.Layer):
    '''
    Layer that learns a set of weights and biases used for scaling the input data.
    @param weights: initialization weights
    @param axis: axis along which to normalize
    @param momentum: momentum in the computation of the exponential average of the mean and standard deviation of the data
    @param beta_init: name of the initialization function for shift parameter
    @param gamma_init: name of the initialization function for scale parameter
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        This method can be used to create weights that depend on the shape(s) of the input(s), using add_weight(). __call__() will automatically build the layer (if it has not been built yet) by calling build().
        @param input_shape: shape of the input
        '''
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        self._trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        '''
        Called in __call__ after making sure build() has been called. call() performs the logic of applying the layer to the input tensors (which should be passed in as argument).
        @param x:
        @param mask:
        @return:
        '''
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]
        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        '''
        Get the configuration.
        @return: dictionary of configuration
        '''
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''
    The convolutional block is the block that has a convolutional layer at shortcut
    @param input_tensor: input tensor
    @param kernel_size: kernel size of middle convolutional layer at main path
    @param filters: the nb_filters of 3 convolutional layers at main path
    @param stage: current stage label
    @param block: current block label
    @return: layer output
    '''
    eps = 1.1e-5
    bn_axis = 3
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''
    The identity block is the block that has no convolutional layer at shortcut
    @param input_tensor: input tensor
    @param kernel_size: kernel size of middle convolutional layer at main path
    @param filters: nb_filters of 3 convolutional layers at main path
    @param stage: current stage label
    @param block: current block label
    @return: layer output
    '''
    eps = 1.1e-5
    bn_axis = 3
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def resnet152_model(img_height, img_width, color_type=1, num_classes=None, new_model=True,
                    resnet_weights_filepath=None):
    '''
    Resnet152 model to recognize car brand and car color recognition.
    @param img_height: image height
    @param img_width: image width
    @param color_type: number of colors of the image
    @param num_classes: number of categories in the dataset
    @param new_model: whether to train a new model or use one trained for car brand or car color recognition
    @param resnet_weights_filepath:
    @return: model output
    '''
    eps = 1.1e-5
    global bn_axis
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
        img_input = Input(shape=(img_height, img_width, color_type), name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_height, img_width), name='data')

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if new_model:
        x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_fc = Flatten()(x_fc)
        x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

        model = Model(img_input, x_fc)

        try:
            model.load_weights(resnet_weights_filepath, by_name=True)
        except:
            raise ValueError('the given resnet_weights_filepath is invalid')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc8')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train(data_dir, checkpoint_dir, epochs=5, patience=5, num_classes=70, resnet_weights_filepath=None):
    '''
    Training loop to train the cnn for car color or car brand recognition.
    @param data_dir: directory where the data is located
    @param checkpoint_dir: directory where the model is saved to or loaded from in case there is an existing model already
    @param epochs: number of epochs to train
    @param patience: number of epochs to wait after training did not get better
    @param num_classes: number of categories in the dataset
    @param resnet_weights_filepath: weights for the pretrained resnet152 model needed for training
    '''
    batch_size = 3
    img_height = 224
    img_width = 224

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    train_ds_normalized = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds_normalized = val_ds.map(lambda x, y: (normalization_layer(x), y))

    model = resnet152_model(img_height, img_width, color_type=3, num_classes=num_classes, new_model=True,
                            resnet_weights_filepath=resnet_weights_filepath)
    sgd = tf.keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    parser.add_argument('--num_classes', metavar='<number>', type=int, help='the number of categories in the dataset',
                        required=True)
    parser.add_argument('--resnet_weights', metavar='<directory>', type=str,
                        help='the weights for the pretrained resnet152 model', required=True)
    args = parser.parse_args()

    train(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir, epochs=args.epochs, patience=args.patience,
          num_classes=args.num_classes, resnet_weights_filepath=args.resnet_weights)


if __name__ == "__main__":
    main(sys.argv[1:])
