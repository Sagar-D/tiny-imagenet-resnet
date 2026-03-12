import tensorflow as tf


class ResNetBuilder:

    def __init__(self):
        pass

    def identity_block(self, X, filters):

        X_shortcut = X

        X = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.ReLU()(X)

        X = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.ReLU()(X)

        return X

    def projection_block(self, X, filters, strides=2):

        X_shortcut = X

        X = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.ReLU()(X)

        X = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)

        X_shortcut = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=(1, 1), strides=strides, padding="valid"
        )(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=-1)(X_shortcut)

        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.ReLU()(X)

        return X

    def data_augmentation_block():

        data_augmentaion = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ]
        )
        return data_augmentaion

    def build_resnet18(self):

        inputs = tf.keras.layers.Input(shape=(64, 64, 3))

        X = self.data_augmentation_block()(inputs)

        X = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(X)
        X = tf.keras.layers.BatchNormalization(axis=-1)(X)
        X = tf.keras.layers.ReLU()(X)
        X = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(X)

        X = self.identity_block(X, filters=64)
        X = self.identity_block(X, filters=64)

        X = self.projection_block(X, filters=128)
        X = self.identity_block(X, filters=128)

        X = self.projection_block(X, filters=256)
        X = self.identity_block(X, filters=256)

        X = self.projection_block(X, filters=512)
        X = self.identity_block(X, filters=512)

        X = tf.keras.layers.GlobalAveragePooling2D()(X)
        outputs = tf.keras.layers.Dense(units=200, activation="softmax")(X)

        model = tf.keras.Model(inputs, outputs)
        return model
